"""Leakage-safe, stratified prompt splitting for Pillar 3 C3-02."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 1234
RATIOS = {"train": 0.72, "val": 0.08, "test": 0.20}

CLEAN_PATH = Path("data/clean/routerbench_binary.parquet")
SPLITS_PATH = Path("data/clean/splits.parquet")


def _allocate_counts(
    sizes: np.ndarray,
    fraction: float,
    target: int,
) -> np.ndarray:
    """Distribute exact global split counts across strata."""

    ideal = sizes * fraction
    counts = np.floor(ideal).astype(int)

    remaining = target - int(counts.sum())

    if remaining < 0 or remaining > len(counts):
        raise ValueError("Unable to allocate stratified split")

    remainders = ideal - counts
    order = np.argsort(-remainders, kind="stable")

    counts[order[:remaining]] += 1
    return counts


def create_splits(
    table: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create reproducible 72/8/20 splits by source and difficulty."""

    required = {"prompt_id", "prompt_text", "source", "difficulty"}
    missing = required.difference(table.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if table.empty:
        raise ValueError("Dataset must not be empty")

    if table[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing values")

    if table["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt IDs could cause leakage")

    if table["prompt_text"].duplicated().any():
        raise ValueError("Duplicate prompt texts could cause leakage")

    if table["prompt_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty prompt IDs")

    if table["prompt_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty prompt texts")

    if table["source"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty source labels")

    difficulty = pd.to_numeric(
        table["difficulty"], errors="raise"
    )

    if not difficulty.isin(range(5)).all():
        raise ValueError("Difficulty must be an integer from 0 to 4")

    groups = list(
        table.groupby(["source", "difficulty"], sort=True)
    )

    sizes = np.array(
        [len(group) for _, group in groups],
        dtype=int,
    )

    total = len(table)

    test_target = round(total * RATIOS["test"])
    val_target = round(total * RATIOS["val"])

    test_counts = _allocate_counts(
        sizes, RATIOS["test"], test_target
    )

    val_counts = _allocate_counts(
        sizes, RATIOS["val"], val_target
    )

    train_counts = sizes - test_counts - val_counts

    if np.any(train_counts < 0):
        raise ValueError("Insufficient prompts for stratification")

    rng = np.random.default_rng(seed)
    assignments = {}

    for index, (_, group) in enumerate(groups):
        # Sort first so input row order cannot affect assignments.
        ids = sorted(group["prompt_id"].astype(str))
        shuffled = rng.permutation(ids)

        n_train = int(train_counts[index])
        n_val = int(val_counts[index])

        for prompt_id in shuffled[:n_train]:
            assignments[str(prompt_id)] = "train"

        for prompt_id in shuffled[n_train:n_train + n_val]:
            assignments[str(prompt_id)] = "val"

        for prompt_id in shuffled[n_train + n_val:]:
            assignments[str(prompt_id)] = "test"

    result = table[
        ["prompt_id", "source", "difficulty"]
    ].copy()

    result["split"] = result["prompt_id"].astype(str).map(
        assignments
    )

    expected = {
        "train": total - test_target - val_target,
        "val": val_target,
        "test": test_target,
    }

    actual = result["split"].value_counts().to_dict()

    if actual != expected:
        raise ValueError(
            f"Unexpected split counts: {actual} != {expected}"
        )

    if result["prompt_id"].duplicated().any():
        raise ValueError("Prompt leakage detected")

    return result.reset_index(drop=True)


def save_splits(
    clean_path: str | Path = CLEAN_PATH,
    output_path: str | Path = SPLITS_PATH,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate and save the official C3-02 split file."""

    clean_path = Path(clean_path)
    output_path = Path(output_path)

    clean = pd.read_parquet(clean_path)
    splits = create_splits(clean, seed=seed)

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    splits.to_parquet(output_path, index=False)

    return splits
