
"""RouterBench cleaning and validation for Pillar 3 (C3-01).

Reuses the existing data loader without modifying its public interfaces.
The final dataset contains binary correctness, per-prompt model costs,
and difficulty labels for the four selected RouterBench models.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data.routerbench import POOL, SOURCES, build_table


def clean_routerbench(
    raw: pd.DataFrame,
    pool: Sequence[str] = POOL,
    sources: Sequence[str] = SOURCES,
) -> pd.DataFrame:
    """Convert RouterBench 0-shot data into the C3-01 clean schema."""

    if len(pool) != 4 or len(set(pool)) != 4:
        raise ValueError("C3-01 requires four distinct pool models")

    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Sources must be nonempty and distinct")

    required_raw = {"sample_id", "eval_name", "prompt"}
    required_raw.update(pool)
    required_raw.update(f"{model}|total_cost" for model in pool)

    missing = required_raw.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")

    # Reuse the team's source filtering, parsing and text deduplication.
    table = build_table(raw, pool=pool, sources=sources)

    if table.empty:
        raise ValueError("No usable RouterBench prompts found")

    rename_map = {
        "text": "prompt_text",
        "band": "difficulty",
    }

    for index, model in enumerate(pool):
        rename_map[f"correct_{index}"] = f"{model}_correct"
        rename_map[f"cost_{index}"] = f"{model}_cost"

    table = table.rename(columns=rename_map)

    # Convert correctness booleans to explicit 0/1 labels.
    for model in pool:
        table[f"{model}_correct"] = (
            table[f"{model}_correct"].astype("int8")
        )

    columns = [
        "prompt_id",
        "source",
        "step_type",
        "prompt_text",
    ]

    for model in pool:
        columns.extend([
            f"{model}_correct",
            f"{model}_cost",
        ])

    columns.append("difficulty")

    clean = table.loc[:, columns].copy()

    validate_clean_routerbench(clean, pool=pool, sources=sources)

    return clean.reset_index(drop=True)


def validate_clean_routerbench(
    table: pd.DataFrame,
    pool: Sequence[str] = POOL,
    sources: Sequence[str] = SOURCES,
) -> None:
    """Reject invalid or inconsistent cleaned RouterBench records."""

    if table.empty:
        raise ValueError("Clean dataset must not be empty")

    required = {
        "prompt_id",
        "source",
        "step_type",
        "prompt_text",
        "difficulty",
    }

    correct_columns = [f"{model}_correct" for model in pool]
    cost_columns = [f"{model}_cost" for model in pool]

    required.update(correct_columns)
    required.update(cost_columns)

    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing clean columns: {sorted(missing)}")

    if table.isna().any().any():
        raise ValueError("Clean dataset contains missing values")

    if table["prompt_id"].duplicated().any():
        raise ValueError("Duplicate prompt IDs detected")

    if table["prompt_text"].duplicated().any():
        raise ValueError("Duplicate prompt texts detected")

    if table["prompt_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty prompt text detected")

    if table["prompt_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty prompt ID detected")

    source_to_type = {
        source: index for index, source in enumerate(sources)
    }

    expected_types = table["source"].map(source_to_type)

    if expected_types.isna().any():
        raise ValueError("Unknown source detected")

    if not np.array_equal(
        table["step_type"].to_numpy(),
        expected_types.to_numpy(),
    ):
        raise ValueError("Source and step type do not match")

    correctness = table[correct_columns].to_numpy()

    if not np.isin(correctness, [0, 1]).all():
        raise ValueError("Correctness values must be binary")

    costs = table[cost_columns].to_numpy(dtype=float)

    if not np.isfinite(costs).all():
        raise ValueError("Model costs must be finite")

    if (costs < 0).any():
        raise ValueError("Model costs cannot be negative")

    expected_difficulty = (correctness == 0).sum(axis=1)

    if not np.array_equal(
        table["difficulty"].to_numpy(),
        expected_difficulty,
    ):
        raise ValueError("Incorrect difficulty calculation")


def save_clean_routerbench(
    raw_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Load verified raw data, validate it, and save cleaned Parquet."""

    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    # Only read a pickle obtained from a trusted, verified source.
    raw = pd.read_pickle(raw_path)

    clean = clean_routerbench(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False)

    return clean
