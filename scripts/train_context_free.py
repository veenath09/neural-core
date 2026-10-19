
"""C3-10: Train and validate the context-free RouterBench baseline.

Usage:
    python -m scripts.train_context_free --run-id c3-10-training-smoke --components 16 --max-features 5000

Training:
    Only prompts assigned to the train split.

Validation:
    Model predictions are evaluated on the val split.

Test:
    Test labels are never used to fit or select the model.

Outputs:
    results/<run_id>/context_free_router.joblib
    results/<run_id>/validation_metrics.csv
    results/<run_id>/training_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import log_loss

from data.routerbench import POOL
from measurement.context_free_router import ContextFreeRouter


def load_datasets():
    """Load clean data and join leakage-safe split assignments."""

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    if clean["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt IDs in clean data")

    if splits["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt IDs in splits")

    clean = clean.copy()
    splits = splits.copy()

    clean["prompt_id"] = clean["prompt_id"].astype(str)
    splits["prompt_id"] = splits["prompt_id"].astype(str)

    assignments = splits.set_index("prompt_id")["split"]

    if set(clean["prompt_id"]) != set(splits["prompt_id"]):
        raise ValueError(
            "Clean data and split assignments do not match"
        )

    clean["split"] = clean["prompt_id"].map(assignments)

    allowed = {"train", "val", "test"}

    if not set(clean["split"]).issubset(allowed):
        raise ValueError("Unexpected dataset split label")

    train = clean.loc[
        clean["split"].eq("train")
    ].copy()

    validation = clean.loc[
        clean["split"].eq("val")
    ].copy()

    test = clean.loc[
        clean["split"].eq("test")
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Training, validation and test splits must be nonempty"
        )

    train_ids = set(train["prompt_id"])
    val_ids = set(validation["prompt_id"])
    test_ids = set(test["prompt_id"])

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise ValueError(
            "Dataset leakage: overlapping prompt IDs"
        )

    text_lookup = dict(
        zip(
            clean["prompt_id"],
            clean["prompt_text"],
        )
    )

    return train, validation, test, text_lookup


def evaluate_validation(
    router: ContextFreeRouter,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate validation metrics using trained features only.

    The vectorizer, SVD, normalizer and classifiers
    are not fitted or modified during validation.
    """

    texts = validation["prompt_text"].astype(str).tolist()

    features = router.vectorizer.transform(texts)
    features = router.reducer.transform(features)
    features = router.normalizer.transform(features)

    results = []

    for model_id in router.model_ids:
        predictor = router.predictors[model_id]

        if isinstance(predictor, (float, np.floating)):
            probabilities = np.full(
                len(validation),
                float(predictor),
            )
        else:
            probabilities = predictor.predict_proba(
                features
            )[:, 1]

        probabilities = np.clip(
            probabilities.astype(float),
            0.0,
            1.0,
        )

        labels = validation[
            f"{model_id}_correct"
        ].to_numpy(dtype=int)

        brier = float(
            np.mean((probabilities - labels) ** 2)
        )

        cross_entropy = float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        )

        results.append(
            {
                "model_id": model_id,
                "validation_prompts": len(validation),
                "observed_accuracy": float(
                    labels.mean()
                ),
                "mean_predicted_success": float(
                    probabilities.mean()
                ),
                "brier_score": brier,
                "log_loss": cross_entropy,
                "train_mean_cost_usd": (
                    router.model_costs[model_id]
                ),
            }
        )

    return pd.DataFrame(results)


def train_router(
    *,
    run_id: str,
    components: int = 16,
    max_features: int = 5000,
    seed: int = 1234,
):
    """Train a model, validate it, and save reproducible artifacts."""

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        run_id,
    ):
        raise ValueError("Invalid run ID")

    output_dir = Path("results") / run_id

    if output_dir.exists():
        raise FileExistsError(
            f"Training run already exists: {output_dir}"
        )

    print("Loading RouterBench data...")

    train, validation, test, text_lookup = (
        load_datasets()
    )

    score_config = yaml.safe_load(
        Path("conf/score.yaml").read_text(
            encoding="utf-8"
        )
    )

    cost_ref = float(
        score_config["cost_ref_usd"]
    )

    if not np.isfinite(cost_ref) or cost_ref <= 0:
        raise ValueError("Invalid configured cost reference")

    print("\nC3-10 TRAINING CONFIGURATION")
    print("-" * 55)
    print("Run ID:", run_id)
    print("Training prompts:", len(train))
    print("Validation prompts:", len(validation))
    print("Reserved test prompts:", len(test))
    print("Models:", len(POOL))
    print("Low-rank components:", components)
    print("Maximum TF-IDF features:", max_features)
    print("Random seed:", seed)
    print("Cost reference (USD):", cost_ref)

    print("\nFitting context-free router on TRAIN split...")

    router = ContextFreeRouter.fit(
        train_frame=train,
        prompt_text_by_id=text_lookup,
        model_ids=POOL,
        cost_ref_usd=cost_ref,
        n_components=components,
        max_features=max_features,
        random_state=seed,
    )

    print("PASS: Training completed.")

    print("\nEvaluating on VALIDATION split...")

    metrics = evaluate_validation(
        router,
        validation,
    )

    print(
        metrics[
            [
                "model_id",
                "observed_accuracy",
                "mean_predicted_success",
                "brier_score",
                "log_loss",
            ]
        ].round(6).to_string(index=False)
    )

    # Record training provenance for reproducibility.
    train_id_hash = hashlib.sha256(
        "\n".join(
            sorted(train["prompt_id"])
        ).encode("utf-8")
    ).hexdigest()

    manifest = {
        "task": "C3-10",
        "method": "TF-IDF + TruncatedSVD + LogisticRegression",
        "method_note": (
            "RouteLLM-inspired context-free baseline; "
            "not an exact reproduction of RouteLLM."
        ),
        "run_id": run_id,
        "train_prompts": len(train),
        "validation_prompts": len(validation),
        "reserved_test_prompts": len(test),
        "train_prompt_ids_sha256": train_id_hash,
        "model_ids": list(POOL),
        "components_requested": components,
        "components_fitted": int(
            router.reducer.n_components
        ),
        "max_features": max_features,
        "random_seed": seed,
        "cost_ref_usd": cost_ref,
        "model_costs_train_only": router.model_costs,
        "calibration_status": "not_calibrated",
        "test_metrics_computed": False,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_path = (
        output_dir / "context_free_router.joblib"
    )

    metrics_path = (
        output_dir / "validation_metrics.csv"
    )

    manifest_path = (
        output_dir / "training_manifest.json"
    )

    joblib.dump(
        router,
        model_path,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Load the saved model to verify the artifact.
    restored = joblib.load(model_path)

    sample_prompt = str(
        validation.iloc[0]["prompt_id"]
    )

    original = router.predict_probabilities(
        sample_prompt
    )

    reloaded = restored.predict_probabilities(
        sample_prompt
    )

    for model_id in POOL:
        if not np.isclose(
            original[model_id],
            reloaded[model_id],
        ):
            raise RuntimeError(
                "Saved model failed prediction verification"
            )

    print("\n" + "=" * 60)
    print("C3-10 TRAINING ARTIFACTS SAVED")
    print("=" * 60)

    print("Model:", model_path)
    print("Validation metrics:", metrics_path)
    print("Manifest:", manifest_path)

    print(
        "\nPASS: Trained model saved and reloaded successfully."
    )

    print(
        "NOTE: Predictions are raw probabilities, "
        "not yet calibrated confidence values."
    )

    print(
        "NOTE: Test labels were not used during training "
        "or validation."
    )

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train the C3-10 context-free router"
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--components",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    args = parser.parse_args()

    train_router(
        run_id=args.run_id,
        components=args.components,
        max_features=args.max_features,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
