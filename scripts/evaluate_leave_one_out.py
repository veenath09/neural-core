
"""C3-12: Leave-one-model-out onboarding experiment.

Purpose:
    Profile a held-out RouterBench model using training
    observations, add its ModelCard, and evaluate routing
    without retraining a policy.

This experiment tests:
    - Training-only probe profiling
    - Weighted profile updating (w = 50)
    - Three-model versus four-model evaluation
    - Capability-aware routing with a changing model pool
    - Reproducible saved results

IMPORTANT:
    The capability-aware router below is a deterministic
    heuristic, NOT the trained neural Controller.

    This experiment demonstrates the model-onboarding
    pipeline, not learned zero-shot generalization.

No API calls are made.
No trained checkpoints are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from controller.contracts import (
    ModelCard,
    Strategy,
)

from data.episodes import from_frame
from data.routerbench import (
    OutcomeTable,
    POOL,
    SOURCES,
)

from measurement.baselines import AlwaysCheapest
from measurement.evaluate import evaluate_episode
from measurement.probe_profiler import (
    ProbeProfiler,
    PROBE_VERSION,
)
from measurement.profile_updater import (
    ProfileUpdater,
    PRIOR_WEIGHT,
)

from scripts.evaluate import (
    CompositeScorer,
    _prepare_inputs,
    make_model_cards,
    sample_ids,
)


class ProfileAwareRouter:
    """Deterministic, capability-aware evaluation policy.

    Selects an affordable model using its estimated
    task-type accuracy and relative execution cost.

    This is not a trained controller.
    """

    def __init__(
        self,
        cards: list[ModelCard],
        *,
        cost_penalty: float = 0.25,
    ) -> None:

        if not cards:
            raise ValueError(
                "At least one model card is required"
            )

        if (
            not math.isfinite(cost_penalty)
            or cost_penalty < 0
        ):
            raise ValueError(
                "cost_penalty must be finite "
                "and nonnegative"
            )

        self.cards = list(cards)
        self.cost_penalty = cost_penalty

        self.cost_ref = max(
            float(card.mean_cost_usd)
            for card in cards
        )

        if self.cost_ref <= 0:
            raise ValueError(
                "Model costs must be positive"
            )

    def decide(self, task) -> Strategy:
        """Select an affordable model for the step type."""

        step_type = int(task.step_type)
        budget = float(task.budget_left_usd)

        if not 0 <= step_type < len(SOURCES):
            raise ValueError(
                "Unknown task type"
            )

        if not math.isfinite(budget):
            raise ValueError(
                "Invalid episode budget"
            )

        affordable = [
            card
            for card in self.cards
            if card.mean_cost_usd <= budget
        ]

        if affordable:
            candidates = affordable
        else:
            # Follow the inexpensive fallback style
            # of the existing simple baselines.
            candidates = [
                min(
                    self.cards,
                    key=lambda card: (
                        card.mean_cost_usd,
                        card.model_id,
                    ),
                )
            ]

        def ranking(card: ModelCard):
            accuracy = float(
                card.capability[step_type]
            )

            relative_cost = (
                card.mean_cost_usd
                / self.cost_ref
            )

            score = (
                accuracy
                - self.cost_penalty * relative_cost
            )

            # Deterministic tie-breaking.
            return (
                score,
                -card.mean_cost_usd,
                card.model_id,
            )

        selected = max(
            candidates,
            key=ranking,
        )

        return Strategy(
            step_id=task.step_id,
            round=task.round,
            model_id=selected.model_id,
            halt="stop",
        )


def select_training_probes(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    probes_per_type: int,
) -> pd.DataFrame:
    """Choose a deterministic training-only probe set."""

    train_ids = splits.loc[
        splits["split"].eq("train"),
        ["prompt_id", "split"],
    ]

    train = clean.merge(
        train_ids,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )

    # Stable selection independent of the input row order.
    train = train.sort_values(
        "prompt_id",
        kind="stable",
    )

    probes = (
        train.groupby(
            "source",
            sort=False,
        )
        .head(probes_per_type)
        .copy()
    )

    counts = probes[
        "source"
    ].value_counts()

    for source in SOURCES:
        if int(counts.get(source, 0)) != probes_per_type:
            raise ValueError(
                f"Insufficient training probes for {source}"
            )

    if not probes["split"].eq("train").all():
        raise RuntimeError(
            "Nontraining probes detected"
        )

    test_ids = set(
        splits.loc[
            splits["split"].eq("test"),
            "prompt_id",
        ]
    )

    if set(probes["prompt_id"]) & test_ids:
        raise RuntimeError(
            "Training probes overlap test prompts"
        )

    return probes


def build_onboarded_card(
    *,
    held_out: ModelCard,
    existing_cards: list[ModelCard],
    probes: pd.DataFrame,
) -> tuple[ModelCard, list[float]]:
    """Profile and onboard the unseen model.

    Initial accuracy priors come from the three
    already-known models.

    The held-out model's observed accuracy and cost
    come only from its training probes.
    """

    if len(existing_cards) != 3:
        raise ValueError(
            "Expected three existing model cards"
        )

    profiler = ProbeProfiler()

    observed_vector = profiler.run(
        held_out,
        probes,
    )

    # Build cold-start accuracy priors from
    # the existing models, not the held-out
    # model's full C3-03 capability estimates.
    prior_accuracies = [
        sum(
            card.capability[index]
            for card in existing_cards
        ) / len(existing_cards)
        for index in range(len(SOURCES))
    ]

    # The sixth feature represents cost.
    # Estimate it from the held-out model's
    # training probes rather than from test data.
    observed_cost = 10 ** observed_vector[-1]

    cold_card = replace(
        held_out,
        mean_cost_usd=observed_cost,
        capability=[
            *prior_accuracies,
            observed_vector[-1],
        ],
    )

    # Each source gets its own observed accuracy
    # and count, following the required w=50 rule.
    observations = {}

    for index, source in enumerate(SOURCES):
        count = int(
            probes["source"].eq(source).sum()
        )

        observations[index] = (
            observed_vector[index],
            count,
        )

    updated = ProfileUpdater(
        prior_weight=PRIOR_WEIGHT,
    ).update_many(
        cold_card,
        observations,
    )

    # Cost and model identity must remain consistent.
    if updated.model_id != held_out.model_id:
        raise RuntimeError(
            "Held-out model identity changed"
        )

    if updated.capability[-1] != cold_card.capability[-1]:
        raise RuntimeError(
            "Cost capability changed during update"
        )

    return updated, observed_vector


def select_model_columns(
    table: OutcomeTable,
    indices: list[int],
) -> OutcomeTable:
    """Create an OutcomeTable for a selected model pool."""

    correct = np.asarray(
        table.correct
    )

    cost = np.asarray(
        table.cost
    )

    if (
        correct.ndim != 2
        or cost.ndim != 2
        or correct.shape != cost.shape
    ):
        raise ValueError(
            "Unexpected OutcomeTable array shapes"
        )

    if correct.shape[1] != len(POOL):
        raise ValueError(
            "Expected the original four-model outcome table"
        )

    return OutcomeTable(
        prompt_ids=np.asarray(
            table.prompt_ids
        ).copy(),
        step_type=np.asarray(
            table.step_type
        ).copy(),
        correct=correct[:, indices].copy(),
        cost=cost[:, indices].copy(),
    )


def run_experiment(
    *,
    eval_id: str,
    episodes: int = 20,
    held_out_index: int = 3,
    probes_per_type: int = 20,
    seed: int = 0,
    lam: float = 0.25,
) -> Path:
    """Run the model-onboarding evaluation offline."""

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        eval_id,
    ):
        raise ValueError(
            "Invalid evaluation ID"
        )

    if episodes < 1:
        raise ValueError(
            "episodes must be positive"
        )

    if probes_per_type < 1:
        raise ValueError(
            "probes_per_type must be positive"
        )

    if not 0 <= held_out_index < len(POOL):
        raise ValueError(
            "Invalid held-out model index"
        )

    if not math.isfinite(lam) or not 0 <= lam <= 100:
        raise ValueError(
            "Invalid lambda"
        )

    output_dir = (
        Path("results")
        / eval_id
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Evaluation already exists: {output_dir}"
        )

    # --------------------------------------------------
    # Load original dataset and C3-03 cards
    # --------------------------------------------------

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    episode_frame = pd.read_parquet(
        "data/episodes/skeletons_test.parquet"
    )

    cards = make_model_cards(
        clean,
        splits,
    )

    held_out = cards[
        held_out_index
    ]

    existing_indices = [
        index
        for index in range(len(cards))
        if index != held_out_index
    ]

    existing_cards = [
        cards[index]
        for index in existing_indices
    ]

    # --------------------------------------------------
    # Profile held-out model using training data
    # --------------------------------------------------

    probes = select_training_probes(
        clean,
        splits,
        probes_per_type=probes_per_type,
    )

    onboarded, observed_vector = build_onboarded_card(
        held_out=held_out,
        existing_cards=existing_cards,
        probes=probes,
    )

    expanded_cards = [
        onboarded if index == held_out_index
        else cards[index]
        for index in range(len(cards))
    ]

    # --------------------------------------------------
    # Build three-model and four-model outcome tables
    # --------------------------------------------------

    full_table, _ = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    initial_table = select_model_columns(
        full_table,
        existing_indices,
    )

    initial_model_ids = [
        POOL[index]
        for index in existing_indices
    ]

    expanded_model_ids = list(
        POOL
    )

    # --------------------------------------------------
    # Select identical held-out test episodes
    # --------------------------------------------------

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

    if len(skeletons) != episodes:
        raise RuntimeError(
            "Unexpected selected episode count"
        )

    difficulty_by_prompt = dict(
        zip(
            clean["prompt_id"],
            clean["difficulty"],
        )
    )

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    # --------------------------------------------------
    # Run both pools on the same test episodes
    # --------------------------------------------------

    print(
        "\nC3-12 LEAVE-ONE-MODEL-OUT EXPERIMENT"
    )

    print(
        "Held-out model:", held_out.model_id
    )

    print(
        "Training probes:", len(probes)
    )

    print(
        "Test episodes:", len(skeletons)
    )

    print(
        "Prior weight:", PRIOR_WEIGHT
    )

    print(
        "No training or API calls are performed."
    )

    configurations = [
        (
            "before",
            existing_cards,
            initial_model_ids,
            initial_table,
        ),
        (
            "after",
            expanded_cards,
            expanded_model_ids,
            full_table,
        ),
    ]

    frames = []

    for phase, phase_cards, model_ids, table in configurations:

        print(
            f"\nEvaluating {phase} onboarding "
            f"({len(model_ids)} models)..."
        )

        policies = {
            "profile-aware": ProfileAwareRouter(
                phase_cards,
            ),
            "always-cheapest": AlwaysCheapest(
                phase_cards,
            ),
        }

        for skeleton in skeletons:

            for policy_name, policy in policies.items():

                frame = evaluate_episode(
                    skeleton,
                    table,
                    model_ids,
                    policy,
                    policy_name=policy_name,
                    scorer=scorer,
                    seed=seed,
                    lam=lam,
                    split="test",
                    difficulty_by_prompt=(
                        difficulty_by_prompt
                    ),
                )

                frame = frame.copy()

                frame["onboarding_phase"] = (
                    phase
                )

                frame["pool_size"] = len(
                    model_ids
                )

                frame["held_out_model"] = (
                    held_out.model_id
                )

                frames.append(
                    frame
                )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    terminals = combined.loc[
        combined["terminal"]
    ].copy()

    expected = (
        len(skeletons)
        * len(configurations)
        * 2
    )

    if len(terminals) != expected:
        raise RuntimeError(
            "Unexpected completed evaluation count"
        )

    # --------------------------------------------------
    # Save reproducible results
    # --------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_path = (
        output_dir
        / "predictions.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False,
    )

    saved = pd.read_parquet(
        output_path
    )

    if int(saved["terminal"].sum()) != expected:
        raise RuntimeError(
            "Saved terminal row count mismatch"
        )

    summary_rows = []

    for (phase, policy), group in terminals.groupby(
        ["onboarding_phase", "policy"],
        sort=True,
    ):

        item = {
            "onboarding_phase": phase,
            "policy": policy,
            "completed_episodes": len(group),
            "mean_episode_cost_usd": float(
                group["episode_cost_usd"].mean()
            ),
        }

        if "episode_quality" in group.columns:
            item["mean_episode_quality"] = float(
                group["episode_quality"].mean()
            )

        summary_rows.append(
            item
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    metadata = {
        "experiment": "c3-12-leave-one-model-out",
        "experiment_type": (
            "offline capability-aware heuristic onboarding"
        ),
        "held_out_model": held_out.model_id,
        "held_out_index": held_out_index,
        "initial_model_ids": initial_model_ids,
        "expanded_model_ids": expanded_model_ids,
        "probe_version": PROBE_VERSION,
        "probe_split": "train",
        "evaluation_split": "test",
        "probe_count": len(probes),
        "probes_per_type": probes_per_type,
        "prior_weight": PRIOR_WEIGHT,
        "observed_probe_vector": observed_vector,
        "onboarded_model_card": asdict(
            onboarded
        ),
        "test_episode_count": len(skeletons),
        "test_episode_ids": [
            str(skeleton.episode_id)
            for skeleton in skeletons
        ],
        "seed": seed,
        "lambda": lam,
        "trained_controller_evaluated": False,
        "limitation": (
            "No trained neural-controller checkpoint was "
            "used. Results test model onboarding with a "
            "deterministic capability-aware router only."
        ),
    }

    (
        output_dir / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nC3-12 RESULTS"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved predictions:", output_path
    )

    print(
        "Saved summary:",
        output_dir / "summary.csv",
    )

    print(
        "Saved metadata:",
        output_dir / "metadata.json",
    )

    print(
        "\nNOTE: These results do not demonstrate "
        "trained-controller generalization."
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "C3-12 leave-one-model-out onboarding experiment"
        )
    )

    parser.add_argument(
        "--eval-id",
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--held-out-index",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--probes-per-type",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lam",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    run_experiment(
        eval_id=args.eval_id,
        episodes=args.episodes,
        held_out_index=args.held_out_index,
        probes_per_type=args.probes_per_type,
        seed=args.seed,
        lam=args.lam,
    )


if __name__ == "__main__":
    main()
