
"""C3-10: Tests for training and validating the context-free router.

Use synthetic RouterBench-style data to verify:
- Leakage-safe data splits
- Training and validation
- Reproducible saved artifacts
- Correct validation metric calculations
- Independence from test labels
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from data.routerbench import POOL
from measurement.context_free_router import ContextFreeRouter
from scripts.train_context_free import (
    load_datasets,
    evaluate_validation,
    train_router,
)


@pytest.fixture
def synthetic_repo(tmp_path, monkeypatch):
    """Create a small isolated repository dataset."""

    monkeypatch.chdir(tmp_path)

    clean_dir = tmp_path / "data" / "clean"
    config_dir = tmp_path / "conf"

    clean_dir.mkdir(parents=True)
    config_dir.mkdir()

    rows = []
    assignments = []

    for split, count in (
        ("train", 18),
        ("val", 8),
        ("test", 6),
    ):
        for i in range(count):
            prompt_id = f"{split}-{i}"

            if i % 2 == 0:
                text = (
                    "astronomy telescope planet orbit "
                    "galaxy science observation"
                )
            else:
                text = (
                    "cooking recipe kitchen dinner "
                    "ingredients food preparation"
                )

            # Unique held-out markers must never enter
            # the vocabulary fitted on training text.
            if split == "val":
                text += " validationuniquemarker"

            if split == "test":
                text += " testuniquemarker"

            row = {
                "prompt_id": prompt_id,
                "prompt_text": text,
                "source": "synthetic",
                "step_type": i % 2,
                "difficulty": 0,
            }

            for model_index, model_id in enumerate(POOL):
                row[f"{model_id}_correct"] = int(
                    (i + model_index) % (model_index + 2)
                    != 0
                )

                row[f"{model_id}_cost"] = (
                    0.0001 * (model_index + 1)
                    + 0.00001 * (i % 3)
                )

            rows.append(row)

            assignments.append(
                {
                    "prompt_id": prompt_id,
                    "split": split,
                }
            )

    clean = pd.DataFrame(rows)
    splits = pd.DataFrame(assignments)

    clean.to_parquet(
        clean_dir / "routerbench_binary.parquet",
        index=False,
    )

    splits.to_parquet(
        clean_dir / "splits.parquet",
        index=False,
    )

    (config_dir / "score.yaml").write_text(
        "cost_ref_usd: 0.001756663098\n",
        encoding="utf-8",
    )

    return tmp_path, clean, splits


def fit_small_router():
    """Train using only the synthetic training split."""

    train, validation, test, lookup = load_datasets()

    router = ContextFreeRouter.fit(
        train_frame=train,
        prompt_text_by_id=lookup,
        model_ids=POOL,
        cost_ref_usd=0.001756663098,
        n_components=3,
        max_features=100,
        random_state=1234,
    )

    return router, train, validation, test


def test_dataset_split_sizes(synthetic_repo):
    train, validation, test, lookup = load_datasets()

    assert len(train) == 18
    assert len(validation) == 8
    assert len(test) == 6
    assert len(lookup) == 32


def test_dataset_splits_do_not_overlap(synthetic_repo):
    train, validation, test, _ = load_datasets()

    train_ids = set(train["prompt_id"])
    val_ids = set(validation["prompt_id"])
    test_ids = set(test["prompt_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_missing_clean_prompt_assignment_is_rejected(
    synthetic_repo,
):
    root, _, splits = synthetic_repo

    incomplete = splits.iloc[:-1].copy()

    incomplete.to_parquet(
        root / "data/clean/splits.parquet",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        load_datasets()


def test_duplicate_clean_prompt_is_rejected(
    synthetic_repo,
):
    root, clean, _ = synthetic_repo

    duplicated = pd.concat(
        [clean, clean.iloc[[0]]],
        ignore_index=True,
    )

    duplicated.to_parquet(
        root / "data/clean/routerbench_binary.parquet",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate prompt IDs",
    ):
        load_datasets()


def test_duplicate_split_assignment_is_rejected(
    synthetic_repo,
):
    root, _, splits = synthetic_repo

    duplicated = pd.concat(
        [splits, splits.iloc[[0]]],
        ignore_index=True,
    )

    duplicated.to_parquet(
        root / "data/clean/splits.parquet",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate prompt IDs",
    ):
        load_datasets()


def test_unknown_split_label_is_rejected(
    synthetic_repo,
):
    root, _, splits = synthetic_repo

    modified = splits.copy()
    modified.loc[0, "split"] = "unknown"

    modified.to_parquet(
        root / "data/clean/splits.parquet",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected dataset split",
    ):
        load_datasets()


def test_empty_validation_split_is_rejected(
    synthetic_repo,
):
    root, _, splits = synthetic_repo

    modified = splits.copy()
    modified.loc[
        modified["split"] == "val",
        "split",
    ] = "train"

    modified.to_parquet(
        root / "data/clean/splits.parquet",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="nonempty",
    ):
        load_datasets()


def test_training_produces_all_four_model_predictors(
    synthetic_repo,
):
    router, _, _, _ = fit_small_router()

    assert set(router.predictors) == set(POOL)
    assert len(router.model_costs) == 4


def test_validation_returns_expected_metrics(
    synthetic_repo,
):
    router, _, validation, _ = fit_small_router()

    metrics = evaluate_validation(router, validation)

    assert len(metrics) == 4
    assert set(metrics["model_id"]) == set(POOL)

    assert metrics["validation_prompts"].eq(8).all()

    assert metrics["brier_score"].between(
        0.0, 1.0
    ).all()

    assert np.isfinite(
        metrics["log_loss"]
    ).all()


def test_validation_does_not_refit_vocabulary(
    synthetic_repo,
):
    router, _, validation, _ = fit_small_router()

    before = dict(router.vectorizer.vocabulary_)

    evaluate_validation(router, validation)

    assert router.vectorizer.vocabulary_ == before

    assert (
        "validationuniquemarker"
        not in router.vectorizer.vocabulary_
    )


def test_test_text_not_used_for_training_features(
    synthetic_repo,
):
    router, _, _, test = fit_small_router()

    assert (
        "testuniquemarker"
        not in router.vectorizer.vocabulary_
    )

    # Test text is available for later inference,
    # but is not included in the fitted vocabulary.
    prompt_id = str(test.iloc[0]["prompt_id"])

    predictions = router.predict_probabilities(
        prompt_id
    )

    assert len(predictions) == 4


def test_validation_brier_score_matches_manual_calculation(
    synthetic_repo,
):
    router, _, validation, _ = fit_small_router()

    metrics = evaluate_validation(router, validation)

    model_id = POOL[0]

    probabilities = np.array(
        [
            router.predict_probabilities(prompt_id)[model_id]
            for prompt_id in validation["prompt_id"]
        ]
    )

    labels = validation[
        f"{model_id}_correct"
    ].to_numpy(dtype=float)

    expected = np.mean(
        (probabilities - labels) ** 2
    )

    actual = metrics.loc[
        metrics["model_id"] == model_id,
        "brier_score",
    ].iloc[0]

    assert actual == pytest.approx(expected)


def test_training_saves_artifacts(synthetic_repo):
    root, _, _ = synthetic_repo

    output_dir = train_router(
        run_id="synthetic-training",
        components=3,
        max_features=100,
        seed=1234,
    )

    assert output_dir.exists()

    assert (
        output_dir / "context_free_router.joblib"
    ).is_file()

    assert (
        output_dir / "validation_metrics.csv"
    ).is_file()

    assert (
        output_dir / "training_manifest.json"
    ).is_file()


def test_saved_training_manifest_is_correct(
    synthetic_repo,
):
    output_dir = train_router(
        run_id="manifest-check",
        components=3,
        max_features=100,
    )

    manifest = json.loads(
        (
            output_dir / "training_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["task"] == "C3-10"
    assert manifest["train_prompts"] == 18
    assert manifest["validation_prompts"] == 8
    assert manifest["reserved_test_prompts"] == 6
    assert manifest["test_metrics_computed"] is False
    assert manifest["calibration_status"] == "not_calibrated"
    assert manifest["model_ids"] == list(POOL)


def test_saved_model_can_be_reloaded(synthetic_repo):
    output_dir = train_router(
        run_id="reload-check",
        components=3,
        max_features=100,
    )

    restored = joblib.load(
        output_dir / "context_free_router.joblib"
    )

    assert isinstance(
        restored,
        ContextFreeRouter,
    )

    predictions = restored.predict_probabilities(
        "val-0"
    )

    assert len(predictions) == 4

    assert all(
        0.0 <= probability <= 1.0
        for probability in predictions.values()
    )


def test_existing_training_run_is_not_overwritten(
    synthetic_repo,
):
    train_router(
        run_id="duplicate-run",
        components=3,
        max_features=100,
    )

    with pytest.raises(FileExistsError):
        train_router(
            run_id="duplicate-run",
            components=3,
            max_features=100,
        )


def test_invalid_training_run_id_is_rejected(
    synthetic_repo,
):
    with pytest.raises(
        ValueError,
        match="Invalid run ID",
    ):
        train_router(
            run_id="../invalid",
            components=3,
            max_features=100,
        )


def test_changing_test_labels_does_not_change_model(
    synthetic_repo,
):
    root, clean, _ = synthetic_repo

    first_dir = train_router(
        run_id="before-test-label-change",
        components=3,
        max_features=100,
        seed=1234,
    )

    first_model = joblib.load(
        first_dir / "context_free_router.joblib"
    )

    original = first_model.predict_probabilities(
        "val-0"
    )

    # Alter only labels belonging to reserved test prompts.
    # Training outcomes and training texts are unchanged.
    altered = clean.copy()

    test_mask = altered["prompt_id"].str.startswith(
        "test-"
    )

    for model_id in POOL:
        column = f"{model_id}_correct"

        altered.loc[test_mask, column] = (
            1 - altered.loc[test_mask, column]
        )

    altered.to_parquet(
        root / "data/clean/routerbench_binary.parquet",
        index=False,
    )

    second_dir = train_router(
        run_id="after-test-label-change",
        components=3,
        max_features=100,
        seed=1234,
    )

    second_model = joblib.load(
        second_dir / "context_free_router.joblib"
    )

    repeated = second_model.predict_probabilities(
        "val-0"
    )

    for model_id in POOL:
        assert repeated[model_id] == pytest.approx(
            original[model_id],
            abs=1e-10,
        )
