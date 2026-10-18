
"""Pillar 3 C3-09: Reproducible RouterBench evaluation runner.

Run from the repository root:

    python -m scripts.evaluate --eval-id c3-09-smoke --episodes 8 --seeds 0

Full baseline evaluation:

    python -m scripts.evaluate --eval-id c3-09-baselines --episodes 5000 --seeds 0 1 2

All outputs are stored in:

    results/<eval-id>/predictions.parquet

Reports and figures must be calculated from saved
predictions, never independently from in-memory results.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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
from measurement.episodes import _prepare_inputs
from measurement.evaluate import (
    evaluate_policies,
    save_predictions,
    load_predictions,
)
from measurement.scorer import CompositeScorer

from scripts.validate_oracle import sample_ids


POLICY_NAMES = (
    "always-cheapest",
    "always-strongest",
    "random",
    "episode-oracle",
)


def make_model_cards(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
) -> list[ModelCard]:
    """Build baseline cards using training-split costs only.

    The simple policies use model IDs and mean costs.
    Capability placeholders are not used for predictions.
    """

    assignments = splits.set_index("prompt_id")["split"]

    training = clean.loc[
        clean["prompt_id"].map(assignments).eq("train")
    ]

    if training.empty:
        raise ValueError("Training split is empty")

    cards = []

    for index, model_id in enumerate(POOL):
        mean_cost = float(
            training[f"{model_id}_cost"].mean()
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


def make_policies(
    cards: list[ModelCard],
    names: list[str],
    seed: int,
) -> dict:
    """Create separate policy instances for an evaluation seed."""

    available = {
        "always-cheapest": lambda: AlwaysCheapest(cards),
        "always-strongest": lambda: AlwaysStrongest(cards),
        "random": lambda: RandomBaseline(
            cards,
            seed=seed,
        ),
        "episode-oracle": lambda: EpisodeOracleBaseline(
            POOL
        ),
    }

    return {
        name: available[name]()
        for name in names
    }


def run_evaluation(
    *,
    eval_id: str,
    episodes: int,
    seeds: list[int],
    policy_names: list[str],
    lam: float,
) -> Path:
    """Run selected policies and save prediction-level evidence."""

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        eval_id,
    ):
        raise ValueError(
            "Evaluation ID may contain only letters, "
            "numbers, hyphens and underscores"
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Evaluation seeds must be unique"
        )

    if len(set(policy_names)) != len(policy_names):
        raise ValueError(
            "Policy names must be unique"
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

    # Reject existing output before performing expensive work.
    if destination.exists():
        raise FileExistsError(
            f"Evaluation already exists: {destination}"
        )

    print("Loading cleaned RouterBench data...")

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    episode_frame = pd.read_parquet(
        "data/episodes/skeletons_test.parquet"
    )

    table, _ = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    cards = make_model_cards(
        clean,
        splits,
    )

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    selected_ids = sample_ids(
        episode_frame,
        episodes,
    )

    selected_frame = episode_frame.loc[
        episode_frame["episode_id"].isin(selected_ids)
    ]

    skeletons = from_frame(selected_frame)

    difficulty_by_prompt = dict(
        zip(
            clean["prompt_id"],
            clean["difficulty"],
        )
    )

    print("\nEvaluation configuration")
    print("------------------------")
    print("Evaluation ID:", eval_id)
    print("Episodes:", len(skeletons))
    print("Policies:", ", ".join(policy_names))
    print("Seeds:", seeds)
    print("Lambda:", lam)
    print("Split: test")

    print(
        "Episode types:",
        selected_frame.drop_duplicates(
            "episode_id"
        )["ep_type"].value_counts().to_dict(),
    )

    all_predictions = []

    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")

        policies = make_policies(
            cards,
            policy_names,
            seed,
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
            difficulty_by_prompt=difficulty_by_prompt,
        )

        all_predictions.append(predictions)

        print(
            f"Seed {seed}: evaluation completed"
        )

    combined = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    saved_path = save_predictions(
        combined,
        destination,
    )

    # Verify the persisted output rather than reporting
    # from in-memory evaluation objects.
    saved = load_predictions(saved_path)

    expected_terminal_rows = (
        len(skeletons)
        * len(policy_names)
        * len(seeds)
    )

    actual_terminal_rows = int(
        saved["terminal"].sum()
    )

    if actual_terminal_rows != expected_terminal_rows:
        raise RuntimeError(
            "Saved predictions have an unexpected "
            "number of completed policy episodes"
        )

    print("\n" + "=" * 60)
    print("C3-09 EVALUATION SAVED")
    print("=" * 60)

    print("Output:", saved_path)
    print("Prediction rows:", len(saved))
    print("Completed policy episodes:", actual_terminal_rows)
    print("Policies:", sorted(saved["policy"].unique()))
    print("Seeds:", sorted(saved["seed"].unique()))

    print(
        "\nPASS: Predictions saved and verified."
    )

    print(
        "Quality, cost, confidence and figure reports "
        "will be derived from this saved file."
    )

    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C3-09 RouterBench evaluation runner"
    )

    parser.add_argument(
        "--eval-id",
        required=True,
        help="Unique identifier for this evaluation",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Number of held-out test episodes",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Evaluation seeds",
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=list(POLICY_NAMES),
        help="Baseline policies to evaluate",
    )

    parser.add_argument(
        "--lam",
        type=float,
        default=0.25,
        help="Composite scorer cost weight",
    )

    args = parser.parse_args()

    run_evaluation(
        eval_id=args.eval_id,
        episodes=args.episodes,
        seeds=args.seeds,
        policy_names=args.policies,
        lam=args.lam,
    )


if __name__ == "__main__":
    main()
