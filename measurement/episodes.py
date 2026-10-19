
"""Pillar 3 C3-04: Leakage-safe episode skeleton generation.

Uses the existing team episode structure and configuration.
Never stores simulated outcomes inside skeletons.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data.episodes import (
    BUDGET_FACTOR,
    EPISODE_TYPES,
    MAX_STEPS,
    MIN_STEPS,
    TYPE_SHARES,
    Skeleton,
    generate_skeletons,
    to_frame,
)
from data.routerbench import OutcomeTable, POOL
from measurement.routerbench import validate_clean_routerbench


CLEAN_PATH = Path("data/clean/routerbench_binary.parquet")
SPLITS_PATH = Path("data/clean/splits.parquet")
OUTPUT_DIR = Path("data/episodes")

DEFAULT_SIZES = {
    "train": 20_000,
    "val": 2_000,
    "test": 5_000,
}

VALID_SPLITS = ("train", "val", "test")


def _check_mix(
    mix: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate episode type proportions."""

    proportions = dict(TYPE_SHARES if mix is None else mix)

    if set(proportions) != set(EPISODE_TYPES):
        raise ValueError(
            "Episode mix must contain all four episode types"
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in proportions.values()
    ):
        raise ValueError(
            "Episode proportions must be finite and non-negative"
        )

    if not math.isclose(
        sum(proportions.values()),
        1.0,
        abs_tol=1e-9,
        rel_tol=0,
    ):
        raise ValueError(
            "Episode proportions must sum to 1"
        )

    return proportions


def _type_counts(
    n: int,
    mix: Mapping[str, float],
) -> dict[str, int]:
    """Allocate exact episode counts by largest remainder."""

    exact = {
        kind: n * mix[kind]
        for kind in EPISODE_TYPES
    }

    counts = {
        kind: math.floor(exact[kind])
        for kind in EPISODE_TYPES
    }

    remaining = n - sum(counts.values())

    order = sorted(
        EPISODE_TYPES,
        key=lambda kind: (
            -(exact[kind] - counts[kind]),
            EPISODE_TYPES.index(kind),
        ),
    )

    for kind in order[:remaining]:
        counts[kind] += 1

    return counts


def _prepare_inputs(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
    split: str,
) -> tuple[OutcomeTable, list[str]]:
    """Validate datasets and build the existing OutcomeTable."""

    validate_clean_routerbench(clean)

    required_splits = {"prompt_id", "split"}

    missing = required_splits.difference(splits.columns)

    if missing:
        raise ValueError(
            f"Missing split columns: {sorted(missing)}"
        )

    if splits[["prompt_id", "split"]].isna().any().any():
        raise ValueError("Split data contains missing values")

    if splits["prompt_id"].duplicated().any():
        raise ValueError("Duplicate split prompt IDs")

    if not splits["split"].isin(VALID_SPLITS).all():
        raise ValueError("Unknown split assignment")

    if set(clean["prompt_id"]) != set(splits["prompt_id"]):
        raise ValueError(
            "Clean and split prompt IDs do not match"
        )

    selected_ids = splits.loc[
        splits["split"] == split,
        "prompt_id",
    ].astype(str).tolist()

    if len(selected_ids) < MAX_STEPS:
        raise ValueError(
            "Not enough prompts in the selected split"
        )

    # Adapt the C3 cleaned schema to the team's OutcomeTable.
    # The existing generator expects correct_0, cost_0, etc.
    adapted = clean[
        ["prompt_id", "step_type"]
    ].copy()

    for index, model in enumerate(POOL):
        adapted[f"correct_{index}"] = (
            clean[f"{model}_correct"].astype(bool)
        )

        adapted[f"cost_{index}"] = (
            clean[f"{model}_cost"].astype(float)
        )

    table = OutcomeTable.from_frame(
        adapted.reset_index(drop=True)
    )

    return table, selected_ids


def _generate_custom_mix(
    table: OutcomeTable,
    prompt_ids: list[str],
    n: int,
    mix: Mapping[str, float],
    seed: int,
    split: str,
) -> list[Skeleton]:
    """Generate episodes using a custom, validated type mix."""

    rng = np.random.default_rng(seed)

    rows = np.asarray(
        [table.index[prompt_id] for prompt_id in prompt_ids],
        dtype=int,
    )

    # A correlated episode must share source and difficulty.
    keys = table.step_type[rows] * 10 + table.band[rows]

    groups = [
        rows[keys == key]
        for key in np.unique(keys)
    ]

    groups = [
        group
        for group in groups
        if len(group) >= MAX_STEPS
    ]

    counts = _type_counts(n, mix)

    if counts["correlated"] > 0 and not groups:
        raise ValueError(
            "No source/difficulty group can support "
            "correlated episodes"
        )

    group_sizes = np.asarray(
        [len(group) for group in groups],
        dtype=float,
    )

    if len(group_sizes):
        group_weights = group_sizes / group_sizes.sum()
    else:
        group_weights = np.asarray([], dtype=float)

    episode_types = rng.permutation(
        np.repeat(
            EPISODE_TYPES,
            [counts[kind] for kind in EPISODE_TYPES],
        )
    )

    skeletons = []

    for index, episode_type in enumerate(episode_types):
        length = int(
            rng.integers(MIN_STEPS, MAX_STEPS + 1)
        )

        if episode_type == "correlated":
            group_index = int(
                rng.choice(
                    len(groups),
                    p=group_weights,
                )
            )
            candidates = groups[group_index]
        else:
            candidates = rows

        chosen = rng.choice(
            candidates,
            size=length,
            replace=False,
        )

        strongest_cost = float(
            table.cost[chosen, -1].sum()
        )

        factor = float(rng.uniform(*BUDGET_FACTOR))
        budget = factor * strongest_cost

        skeletons.append(
            Skeleton(
                episode_id=f"{split}-{index:06d}",
                ep_type=str(episode_type),
                prompt_ids=tuple(
                    str(prompt)
                    for prompt in table.prompt_ids[chosen]
                ),
                budget_usd=budget,
            )
        )

    return skeletons


def generate(
    split: str,
    n: int,
    mix: Mapping[str, float] | None = None,
    seed: int = 1234,
    *,
    clean: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
) -> list[Skeleton]:
    """Generate leakage-safe episode skeletons.

    Args:
        split: train, val, or test.
        n: Number of episodes.
        mix: Optional episode type proportions.
        seed: Reproducibility seed.
        clean: Optional in-memory cleaned dataset.
        splits: Optional in-memory split assignments.
    """

    if split not in VALID_SPLITS:
        raise ValueError("Unknown dataset split")

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("Episode count must be positive")

    proportions = _check_mix(mix)

    if clean is None:
        clean = pd.read_parquet(CLEAN_PATH)

    if splits is None:
        splits = pd.read_parquet(SPLITS_PATH)

    table, prompt_ids = _prepare_inputs(
        clean,
        splits,
        split,
    )

    # Reuse the team's generator for the standard mix.
    if proportions == TYPE_SHARES:
        return generate_skeletons(
            table=table,
            prompt_ids=prompt_ids,
            n=n,
            seed=seed,
            split=split,
        )

    # Support a different mix without changing team code.
    return _generate_custom_mix(
        table=table,
        prompt_ids=prompt_ids,
        n=n,
        mix=proportions,
        seed=seed,
        split=split,
    )


def save_skeletons(
    split: str,
    n: int,
    mix: Mapping[str, float] | None = None,
    seed: int = 1234,
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Generate and persist C3-04 episode skeletons."""

    skeletons = generate(
        split=split,
        n=n,
        mix=mix,
        seed=seed,
    )

    frame = to_frame(skeletons)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"skeletons_{split}.parquet"

    frame.to_parquet(path, index=False)

    return frame
