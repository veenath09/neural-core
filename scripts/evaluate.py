
"""Pillar 3: Reproducible RouterBench evaluation runner.

Supports:
    C3-09: Four baseline controllers.
    C3-10: Trained context-free router baseline.

Run the original baseline evaluation:

    python -m scripts.evaluate --eval-id c3-09-test --episodes 8 --seeds 0

Run all five policies, including the trained router:

    python -m scripts.evaluate ^
        --eval-id c3-10-router-smoke ^
        --episodes 8 ^
        --seeds 0 ^
        --policies always-cheapest always-strongest random episode-oracle context-free-router ^
        --router-model results/c3-10-training-smoke/context_free_router.joblib

All evaluation predictions are saved to:

    results/<eval-id>/predictions.parquet

Reports and figures must be calculated from persisted
predictions rather than in-memory evaluation results.

The context-free router uses prompt text only, without
episode traces or privileged correctness information.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd

from controller.contracts import ModelCard
from data.episodes import from_frame
from data.routerbench import POOL

from measurement.baselines import (
    AlwaysCheapest,
    AlwaysStrongest,
    RandomBaseline,
    EpisodeOracleBaseline,
)

from measurement.context_free_router import ContextFreeRouter
from measurement.episodes import _prepare_inputs

from measurement.evaluate import (
    evaluate_policies,
    save_predictions,
    load_predictions,
)

from measurement.scorer import CompositeScorer

from scripts.validate_oracle import sample_ids


# ------------------------------------------------------------
# AVAILABLE EVALUATION POLICIES
# ------------------------------------------------------------

# Original C3-09 baseline policies.
POLICY_NAMES = (
    "always-cheapest",
    "always-strongest",
    "random",
    "episode-oracle",
)

# New C3-10 context-free router.
CONTEXT_FREE_POLICY = "context-free-router"

# All supported evaluation policies.
ALL_POLICY_NAMES = POLICY_NAMES + (
    CONTEXT_FREE_POLICY,
)


# ------------------------------------------------------------
# TRAINING-ONLY BASELINE MODEL CARDS
# ------------------------------------------------------------

def make_model_cards(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
) -> list[ModelCard]:
    """Build baseline cards using training-split costs only.

    These simple baselines use model IDs and mean costs.

    The capability placeholders are not used to predict
    held-out model correctness.
    """

    assignments = (
        splits.set_index("prompt_id")["split"]
    )

    training = clean.loc[
        clean["prompt_id"]
        .map(assignments)
        .eq("train")
    ]

    if training.empty:
        raise ValueError(
            "Training split is empty"
        )

    cards = []

    for index, model_id in enumerate(POOL):

        mean_cost = float(
            training[
                f"{model_id}_cost"
            ].mean()
        )

        cards.append(
            ModelCard(
                model_id=model_id,
                provider="routerbench",
                tier=index,
                mean_cost_usd=mean_cost,
                price_source="historical_routerbench",
                price_date="unknown",
                capability=[0.0] * 6,
            )
        )

    return cards


# ------------------------------------------------------------
# LOAD TRAINED CONTEXT-FREE ROUTER
# ------------------------------------------------------------

def load_context_free_router(
    router_model: str | Path,
) -> ContextFreeRouter:
    """Load and validate a previously trained C3-10 router.

    Load only artifacts created locally or obtained from
    trusted sources, since joblib deserialization can
    execute Python code.
    """

    model_path = Path(router_model)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained router model not found: {model_path}"
        )

    print(
        "Loading trained context-free router..."
    )

    loaded = joblib.load(model_path)

    if not isinstance(
        loaded,
        ContextFreeRouter,
    ):
        raise TypeError(
            "The supplied artifact is not a ContextFreeRouter"
        )

    if tuple(loaded.model_ids) != tuple(POOL):
        raise ValueError(
            "Trained router model pool does not match "
            "the RouterBench evaluation pool"
        )

    print(
        "PASS: Trained context-free router loaded successfully"
    )

    print(
        "Router artifact:",
        model_path,
    )

    return loaded


# ------------------------------------------------------------
# CREATE EVALUATION POLICIES
# ------------------------------------------------------------

def make_policies(
    cards: list[ModelCard],
    names: list[str],
    seed: int,
    router: ContextFreeRouter | None = None,
) -> dict:
    """Create the requested evaluation policies.

    Existing C3-09 policies remain unchanged.

    C3-10 adds an optional, trained context-free router.
    """

    if (
        CONTEXT_FREE_POLICY in names
        and router is None
    ):
        raise ValueError(
            "context-free-router requires a trained router model"
        )

    available = {
        "always-cheapest": (
            lambda: AlwaysCheapest(cards)
        ),

        "always-strongest": (
            lambda: AlwaysStrongest(cards)
        ),

        "random": (
            lambda: RandomBaseline(
                cards,
                seed=seed,
            )
        ),

        "episode-oracle": (
            lambda: EpisodeOracleBaseline(
                POOL
            )
        ),
    }

    # Register the trained context-free policy only
    # when the corresponding artifact is provided.
    if router is not None:
        available[CONTEXT_FREE_POLICY] = (
            lambda: router
        )

    unknown = [
        name
        for name in names
        if name not in available
    ]

    if unknown:
        raise ValueError(
            f"Unknown or unavailable policies: {unknown}"
        )

    return {
        name: available[name]()
        for name in names
    }


# ------------------------------------------------------------
# MAIN EVALUATION RUNNER
# ------------------------------------------------------------

def run_evaluation(
    *,
    eval_id: str,
    episodes: int,
    seeds: list[int],
    policy_names: list[str],
    lam: float,
    router_model: str | None = None,
) -> Path:
    """Run selected policies and save prediction-level evidence.

    Supports:
        - C3-09 original four baselines
        - C3-10 trained context-free router

    Returns:
        Path to the saved predictions.parquet file.
    """

    # --------------------------------------------------------
    # 1. VALIDATE EVALUATION CONFIGURATION
    # --------------------------------------------------------

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        eval_id,
    ):
        raise ValueError(
            "Evaluation ID may contain only letters, "
            "numbers, hyphens and underscores"
        )

    if episodes < 1:
        raise ValueError(
            "Episode count must be positive"
        )

    if not seeds:
        raise ValueError(
            "At least one seed is required"
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Evaluation seeds must be unique"
        )

    if not policy_names:
        raise ValueError(
            "At least one evaluation policy is required"
        )

    if len(set(policy_names)) != len(policy_names):
        raise ValueError(
            "Policy names must be unique"
        )

    unknown_policies = (
        set(policy_names)
        - set(ALL_POLICY_NAMES)
    )

    if unknown_policies:
        raise ValueError(
            "Unknown evaluation policies: "
            f"{sorted(unknown_policies)}"
        )

    if not 0 <= lam <= 100:
        raise ValueError(
            "Lambda must be between 0 and 100"
        )

    destination = (
        Path("results")
        / eval_id
        / "predictions.parquet"
    )

    # Do not accidentally overwrite earlier evaluations.
    if destination.exists():
        raise FileExistsError(
            f"Evaluation already exists: {destination}"
        )

    # --------------------------------------------------------
    # 2. LOAD THE OPTIONAL TRAINED ROUTER
    # --------------------------------------------------------

    router = None

    if CONTEXT_FREE_POLICY in policy_names:

        if router_model is None:
            raise ValueError(
                "The context-free router requires a "
                "trained model supplied using --router-model"
            )

        router = load_context_free_router(
            router_model
        )

    # --------------------------------------------------------
    # 3. LOAD ROUTERBENCH DATA
    # --------------------------------------------------------

    print(
        "Loading cleaned RouterBench data..."
    )

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    episode_frame = pd.read_parquet(
        "data/episodes/skeletons_test.parquet"
    )

    # --------------------------------------------------------
    # 4. PREPARE TEST ENVIRONMENT
    # --------------------------------------------------------

    table, _ = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    # Existing simple policies use training-only
    # mean-cost information.
    cards = make_model_cards(
        clean,
        splits,
    )

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    # --------------------------------------------------------
    # 5. SELECT TEST EPISODES
    # --------------------------------------------------------

    selected_ids = sample_ids(
        episode_frame,
        episodes,
    )

    selected_frame = episode_frame.loc[
        episode_frame["episode_id"].isin(
            selected_ids
        )
    ]

    skeletons = from_frame(
        selected_frame
    )

    if not skeletons:
        raise ValueError(
            "No test episode skeletons were selected"
        )

    # Difficulty is recorded for evaluation analysis.
    # It is not passed to our ContextFreeRouter as
    # an input to decide().
    difficulty_by_prompt = dict(
        zip(
            clean["prompt_id"],
            clean["difficulty"],
        )
    )

    # --------------------------------------------------------
    # 6. DISPLAY EVALUATION CONFIGURATION
    # --------------------------------------------------------

    print(
        "\nEvaluation configuration"
    )

    print(
        "------------------------"
    )

    print(
        "Evaluation ID:",
        eval_id,
    )

    print(
        "Episodes:",
        len(skeletons),
    )

    print(
        "Policies:",
        ", ".join(policy_names),
    )

    print(
        "Seeds:",
        seeds,
    )

    print(
        "Lambda:",
        lam,
    )

    print(
        "Split: test"
    )

    print(
        "Episode types:",
        selected_frame.drop_duplicates(
            "episode_id"
        )["ep_type"].value_counts().to_dict(),
    )

    if router is not None:
        print(
            "Context-free router: enabled"
        )

        print(
            "Router model:",
            router_model,
        )

        print(
            "Router inputs: current prompt text "
            "and remaining budget"
        )

        print(
            "Router calibration: not yet established"
        )

    # --------------------------------------------------------
    # 7. EXECUTE POLICY EVALUATIONS
    # --------------------------------------------------------

    all_predictions = []

    for seed in seeds:

        print(
            f"\nEvaluating seed {seed}..."
        )

        policies = make_policies(
            cards,
            policy_names,
            seed,
            router=router,
        )

        predictions = evaluate_policies(
            skeletons,
            table,
            POOL,
            policies,
            scorer=scorer,
            seed=seed,
            lam=lam,
            split="test",
            difficulty_by_prompt=(
                difficulty_by_prompt
            ),
        )

        all_predictions.append(
            predictions
        )

        print(
            f"Seed {seed}: evaluation completed"
        )

    # --------------------------------------------------------
    # 8. COMBINE PREDICTION RECORDS
    # --------------------------------------------------------

    combined = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 9. SAVE RESULTS
    # --------------------------------------------------------

    saved_path = save_predictions(
        combined,
        destination,
    )

    # --------------------------------------------------------
    # 10. VERIFY PERSISTED RESULTS
    # --------------------------------------------------------

    # Research metrics must be derived from saved data.
    saved = load_predictions(
        saved_path
    )

    expected_terminal_rows = (
        len(skeletons)
        * len(policy_names)
        * len(seeds)
    )

    actual_terminal_rows = int(
        saved["terminal"].sum()
    )

    if (
        actual_terminal_rows
        != expected_terminal_rows
    ):
        raise RuntimeError(
            "Saved predictions have an unexpected "
            "number of completed policy episodes"
        )

    # Verify every policy is represented.
    actual_policies = set(
        saved["policy"].unique()
    )

    if actual_policies != set(policy_names):
        raise RuntimeError(
            "Saved predictions do not contain "
            "all requested policies"
        )

    # Verify every seed is represented.
    actual_seeds = set(
        int(seed)
        for seed in saved["seed"].unique()
    )

    if actual_seeds != set(seeds):
        raise RuntimeError(
            "Saved predictions do not contain "
            "all requested seeds"
        )

    # --------------------------------------------------------
    # 11. DISPLAY FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "C3-09 / C3-10 EVALUATION SAVED"
    )

    print(
        "=" * 60
    )

    print(
        "Output:",
        saved_path,
    )

    print(
        "Prediction rows:",
        len(saved),
    )

    print(
        "Completed policy episodes:",
        actual_terminal_rows,
    )

    print(
        "Policies:",
        sorted(actual_policies),
    )

    print(
        "Seeds:",
        sorted(actual_seeds),
    )

    print(
        "\nPASS: Predictions saved and verified."
    )

    print(
        "Quality, cost and reward reports will "
        "be derived from this saved file."
    )

    if router is not None:
        print(
            "NOTE: Router success probabilities are "
            "not yet calibrated confidence estimates."
        )

    print(
        "NOTE: These are simulated RouterBench "
        "evaluations, not live API measurements."
    )

    return saved_path


# ------------------------------------------------------------
# COMMAND-LINE INTERFACE
# ------------------------------------------------------------

def main() -> None:
    """Parse evaluation options and start the evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "C3-09 / C3-10 RouterBench evaluation runner"
        )
    )

    parser.add_argument(
        "--eval-id",
        required=True,
        help=(
            "Unique identifier for this evaluation"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help=(
            "Number of held-out test episodes"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Evaluation seeds"
        ),
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=ALL_POLICY_NAMES,
        default=list(POLICY_NAMES),
        help=(
            "Policies to evaluate. "
            "Defaults to the original four baselines."
        ),
    )

    parser.add_argument(
        "--lam",
        type=float,
        default=0.25,
        help=(
            "Composite scorer cost weight"
        ),
    )

    parser.add_argument(
        "--router-model",
        type=str,
        default=None,
        help=(
            "Path to a trained ContextFreeRouter "
            "joblib artifact"
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        eval_id=args.eval_id,
        episodes=args.episodes,
        seeds=args.seeds,
        policy_names=args.policies,
        lam=args.lam,
        router_model=args.router_model,
    )


if __name__ == "__main__":
    main()
