
"""Validate the C3-06 exact oracle against simple baselines.

Uses held-out RouterBench test episodes and the existing
data.env.Episode simulator.

Run:
    python scripts/validate_oracle.py --episodes 8
    python scripts/validate_oracle.py --episodes 1000

These are evaluation-only baseline rollouts.
C3-08 will implement the official baseline ControllerAPI classes.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from data.env import Episode
from data.episodes import from_frame
from data.routerbench import POOL
from measurement.episodes import _prepare_inputs
from measurement.oracle import (
    exact_episode_oracle,
    per_step_oracle,
)


SHARES = {
    "correlated": 0.40,
    "retry": 0.25,
    "propagation": 0.20,
    "random": 0.15,
}

BASELINES = (
    "always-cheapest",
    "always-strongest",
    "random",
    "per-step-oracle",
)


def sample_ids(frame: pd.DataFrame, count: int) -> list[str]:
    """Choose an episode-type-stratified test sample."""

    metadata = frame.drop_duplicates("episode_id")

    if count < 1 or count > len(metadata):
        raise ValueError(
            f"Episode count must be between 1 and {len(metadata)}"
        )

    quotas = {
        name: int(count * share)
        for name, share in SHARES.items()
    }

    remaining = count - sum(quotas.values())

    by_fraction = sorted(
        SHARES,
        key=lambda name: (
            -(count * SHARES[name] - quotas[name]),
            list(SHARES).index(name),
        ),
    )

    for name in by_fraction[:remaining]:
        quotas[name] += 1

    selected = []

    for name, required in quotas.items():
        available = metadata.loc[
            metadata["ep_type"] == name,
            "episode_id",
        ].tolist()

        if len(available) < required:
            raise ValueError(
                f"Not enough {name} episodes"
            )

        selected.extend(available[:required])

    return selected


def run_baseline(
    episode: Episode,
    policy: str,
    seed: int,
) -> tuple[int, float]:
    """Execute a one-model-per-round baseline policy.

    Retry handling and budget exhaustion are controlled
    by the existing Episode environment.
    """

    state = episode.initial_state()

    rng = random.Random(
        f"{seed}:{episode.skeleton.episode_id}"
    )

    while not state.done:
        if policy == "always-cheapest":
            model = 0

        elif policy == "always-strongest":
            model = len(episode.model_ids) - 1

        elif policy == "random":
            model = rng.randrange(
                len(episode.model_ids)
            )

        elif policy == "per-step-oracle":
            model = per_step_oracle(
                episode, state
            ).model_tier

        else:
            raise ValueError(
                f"Unknown baseline: {policy}"
            )

        state = episode.play(state, model)

        if not state.done:
            state = episode.end_step(state)

    return int(state.n_correct), float(state.spent)


def validate(count: int, seed: int) -> None:
    print("Loading held-out RouterBench data...")

    clean = pd.read_parquet(
        Path("data/clean/routerbench_binary.parquet")
    )

    splits = pd.read_parquet(
        Path("data/clean/splits.parquet")
    )

    table, _ = _prepare_inputs(
        clean, splits, "test"
    )

    frame = pd.read_parquet(
        Path("data/episodes/skeletons_test.parquet")
    )

    ids = sample_ids(frame, count)

    selected_frame = frame[
        frame["episode_id"].isin(ids)
    ]

    skeletons = from_frame(selected_frame)

    print(f"\nEpisodes selected: {len(skeletons)}")

    print(
        "Episode types:",
        selected_frame.drop_duplicates(
            "episode_id"
        )["ep_type"].value_counts().to_dict(),
    )

    policies = ("exact-oracle",) + BASELINES

    statistics = {
        name: {
            "quality": 0.0,
            "cost": 0.0,
        }
        for name in policies
    }

    violations = []

    started = time.perf_counter()

    for index, skeleton in enumerate(skeletons, 1):
        episode = Episode(
            skeleton,
            table,
            POOL,
            seed=seed,
        )

        oracle = exact_episode_oracle(episode)

        statistics["exact-oracle"]["quality"] += (
            oracle.quality
        )
        statistics["exact-oracle"]["cost"] += (
            oracle.cost_usd
        )

        for policy in BASELINES:
            correct, cost = run_baseline(
                episode,
                policy,
                seed,
            )

            quality = correct / episode.n_steps

            statistics[policy]["quality"] += quality
            statistics[policy]["cost"] += cost

            # The oracle's quality must never be lower.
            worse_quality = (
                oracle.correct_steps < correct
            )

            # With equal quality, its cost must be minimal.
            worse_cost = (
                oracle.correct_steps == correct
                and oracle.cost_usd > cost + 1e-9
            )

            if worse_quality or worse_cost:
                violations.append(
                    (
                        skeleton.episode_id,
                        policy,
                        oracle.correct_steps,
                        correct,
                        oracle.cost_usd,
                        cost,
                    )
                )

        if index % 100 == 0:
            print(
                f"Validated {index}/{len(skeletons)} episodes"
            )

    elapsed = time.perf_counter() - started

    print("\n" + "=" * 62)
    print("C3-06 ORACLE BASELINE VALIDATION")
    print("=" * 62)

    print(
        f"{'Policy':<22}"
        f"{'Mean quality':>15}"
        f"{'Mean cost USD':>19}"
    )

    for policy in policies:
        avg_quality = (
            statistics[policy]["quality"]
            / len(skeletons)
        )

        avg_cost = (
            statistics[policy]["cost"]
            / len(skeletons)
        )

        print(
            f"{policy:<22}"
            f"{avg_quality:>15.4f}"
            f"{avg_cost:>19.8f}"
        )

    print("\nTotal seconds:", round(elapsed, 3))
    print(
        "Average seconds per episode:",
        round(elapsed / len(skeletons), 5),
    )
    print("Oracle dominance violations:", len(violations))

    if violations:
        print("\nFirst violations:")

        for violation in violations[:10]:
            print(violation)

        raise SystemExit(
            "FAIL: Oracle was worse than a baseline."
        )

    print("\nPASS: Oracle dominates all four baselines.")
    print(
        "Comparison used quality-first, cost-second ranking."
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    arguments = parser.parse_args()

    validate(
        count=arguments.episodes,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
