
"""C3-13: Generate a workload-shifted, test-only RouterBench dataset.

The shifted workload preserves:
- Episode IDs, types, and step counts
- Correlated episode source/difficulty-band constraints
- Each episode's relative strongest-model budget factor
- Original skeleton file schema

Only prompts assigned to the TEST split may be sampled.

The generator never modifies the original test skeletons.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data.episodes import (
    MAX_STEPS,
    Skeleton,
    from_frame,
    to_frame,
)
from data.routerbench import OutcomeTable, SOURCES
from scripts.evaluate import _prepare_inputs


TARGET_SHARES = {
    "mmlu": 0.15,
    "hellaswag": 0.15,
    "arc-challenge": 0.20,
    "winogrande": 0.20,
    "mbpp": 0.30,
}

DEFAULT_INPUT = Path(
    "data/episodes/skeletons_test.parquet"
)

DEFAULT_OUTPUT = Path(
    "data/episodes/skeletons_test_shifted.parquet"
)


def validate_targets(
    target_shares: dict[str, float],
) -> np.ndarray:
    """Validate target shares in the canonical source order."""

    if set(target_shares) != set(SOURCES):
        raise ValueError(
            "Target shares must include every RouterBench source"
        )

    weights = np.asarray(
        [target_shares[source] for source in SOURCES],
        dtype=float,
    )

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(
            "Target shares must be finite, nonnegative, "
            "and sum to one"
        )

    return weights


def build_test_only_table(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
) -> OutcomeTable:
    """Build an OutcomeTable containing ONLY test-split prompts.

    Important:
        _prepare_inputs(..., 'test') returns the full
        outcome table AND a separate list of test IDs.

        It does NOT automatically filter the table.
    """

    full_table, test_ids = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    if not test_ids:
        raise ValueError(
            "The test split contains no prompts"
        )

    if len(set(test_ids)) != len(test_ids):
        raise ValueError(
            "Duplicate test prompt IDs"
        )

    # Select the actual rows belonging to the test split.
    rows = np.asarray(
        [
            full_table.index[prompt_id]
            for prompt_id in test_ids
        ],
        dtype=int,
    )

    test_table = OutcomeTable(
        prompt_ids=np.asarray(
            full_table.prompt_ids
        )[rows].copy(),
        step_type=np.asarray(
            full_table.step_type
        )[rows].copy(),
        correct=np.asarray(
            full_table.correct
        )[rows].copy(),
        cost=np.asarray(
            full_table.cost
        )[rows].copy(),
    )

    expected_test_ids = set(
        splits.loc[
            splits["split"].eq("test"),
            "prompt_id",
        ].astype(str)
    )

    training_ids = set(
        splits.loc[
            splits["split"].eq("train"),
            "prompt_id",
        ].astype(str)
    )

    validation_ids = set(
        splits.loc[
            splits["split"].eq("val"),
            "prompt_id",
        ].astype(str)
    )

    actual_ids = set(
        str(prompt_id)
        for prompt_id in test_table.prompt_ids
    )

    if actual_ids != expected_test_ids:
        raise RuntimeError(
            "Test-only outcome table does not match test split"
        )

    if actual_ids & (training_ids | validation_ids):
        raise RuntimeError(
            "Data leakage: non-test prompts in test table"
        )

    if len(test_table.prompt_ids) != len(expected_test_ids):
        raise RuntimeError(
            "Unexpected test-table prompt count"
        )

    return test_table


def build_source_pools(
    table: OutcomeTable,
) -> tuple[dict, dict]:
    """Build source and eligible correlated-group pools."""

    source_pools = {}
    correlated_pools = {}

    bands = np.asarray(table.band)

    for source_index, source in enumerate(SOURCES):

        rows = np.flatnonzero(
            table.step_type == source_index
        )

        if len(rows) < MAX_STEPS:
            raise ValueError(
                f"Not enough test prompts for {source}"
            )

        source_pools[source] = rows

        groups = []

        for band in np.unique(bands[rows]):

            group = rows[
                bands[rows] == band
            ]

            if len(group) >= MAX_STEPS:
                groups.append(group)

        correlated_pools[source] = groups

    return source_pools, correlated_pools


def sample_prompt_rows(
    *,
    ep_type: str,
    n_steps: int,
    rng: np.random.Generator,
    source_pools: dict,
    correlated_pools: dict,
    weights: np.ndarray,
) -> np.ndarray:
    """Choose distinct test prompts for a single episode."""

    if not 1 <= n_steps <= MAX_STEPS:
        raise ValueError(
            "Invalid episode length"
        )

    if ep_type == "correlated":

        eligible_sources = [
            source
            for source in SOURCES
            if correlated_pools[source]
        ]

        eligible_weights = np.asarray(
            [
                weights[SOURCES.index(source)]
                for source in eligible_sources
            ],
            dtype=float,
        )

        if (
            not eligible_sources
            or eligible_weights.sum() <= 0
        ):
            raise ValueError(
                "No eligible correlated prompt groups"
            )

        eligible_weights /= eligible_weights.sum()

        source = str(
            rng.choice(
                eligible_sources,
                p=eligible_weights,
            )
        )

        groups = correlated_pools[source]

        sizes = np.asarray(
            [len(group) for group in groups],
            dtype=float,
        )

        group_index = int(
            rng.choice(
                len(groups),
                p=sizes / sizes.sum(),
            )
        )

        return np.asarray(
            rng.choice(
                groups[group_index],
                size=n_steps,
                replace=False,
            ),
            dtype=int,
        )

    if ep_type not in {
        "retry",
        "propagation",
        "random",
    }:
        raise ValueError(
            f"Unsupported episode type: {ep_type}"
        )

    chosen = []

    for _ in range(n_steps):

        source = str(
            rng.choice(
                SOURCES,
                p=weights,
            )
        )

        candidates = source_pools[source]

        available = candidates[
            ~np.isin(
                candidates,
                chosen,
            )
        ]

        if len(available) == 0:
            raise ValueError(
                f"No unused test prompts for {source}"
            )

        chosen.append(
            int(rng.choice(available))
        )

    return np.asarray(
        chosen,
        dtype=int,
    )


def generate_shifted_skeletons(
    original: list[Skeleton],
    table: OutcomeTable,
    *,
    seed: int = 0,
    target_shares: dict[str, float] | None = None,
) -> list[Skeleton]:
    """Generate a changed workload from an eligible prompt table.

    The caller must provide a test-only OutcomeTable.
    Use build_test_only_table() to construct that table.
    """

    if target_shares is None:
        target_shares = TARGET_SHARES

    weights = validate_targets(
        target_shares
    )

    source_pools, correlated_pools = (
        build_source_pools(table)
    )

    rng = np.random.default_rng(
        seed
    )

    shifted = []

    for skeleton in original:

        try:
            original_rows = np.asarray(
                [
                    table.index[prompt_id]
                    for prompt_id in skeleton.prompt_ids
                ],
                dtype=int,
            )

        except KeyError as exc:
            raise ValueError(
                "Original skeleton contains a prompt "
                "outside the selected test split"
            ) from exc

        new_rows = sample_prompt_rows(
            ep_type=skeleton.ep_type,
            n_steps=len(skeleton.prompt_ids),
            rng=rng,
            source_pools=source_pools,
            correlated_pools=correlated_pools,
            weights=weights,
        )

        # Preserve the original relative budget factor.
        original_reference_cost = float(
            table.cost[original_rows, -1].sum()
        )

        shifted_reference_cost = float(
            table.cost[new_rows, -1].sum()
        )

        if (
            original_reference_cost <= 0
            or shifted_reference_cost <= 0
        ):
            raise ValueError(
                "Reference execution costs must be positive"
            )

        budget_factor = (
            skeleton.budget_usd
            / original_reference_cost
        )

        new_budget = (
            budget_factor
            * shifted_reference_cost
        )

        new_prompt_ids = tuple(
            str(prompt_id)
            for prompt_id in table.prompt_ids[new_rows]
        )

        if len(set(new_prompt_ids)) != len(new_prompt_ids):
            raise RuntimeError(
                "Duplicate prompt inside shifted episode"
            )

        if skeleton.ep_type == "correlated":

            new_sources = set(
                table.step_type[new_rows].tolist()
            )

            new_bands = set(
                table.band[new_rows].tolist()
            )

            if (
                len(new_sources) != 1
                or len(new_bands) != 1
            ):
                raise RuntimeError(
                    "Correlated episode constraint violated"
                )

        shifted.append(
            Skeleton(
                episode_id=skeleton.episode_id,
                ep_type=skeleton.ep_type,
                prompt_ids=new_prompt_ids,
                budget_usd=float(new_budget),
            )
        )

    return shifted


def source_distribution(
    skeletons: list[Skeleton],
    table: OutcomeTable,
) -> dict[str, float]:
    """Calculate proportions of source types across steps."""

    counts = Counter()
    total = 0

    for skeleton in skeletons:

        for prompt_id in skeleton.prompt_ids:

            row = table.index[prompt_id]

            source_index = int(
                table.step_type[row]
            )

            source = SOURCES[source_index]

            counts[source] += 1
            total += 1

    if total == 0:
        raise ValueError(
            "Cannot calculate distribution of empty workload"
        )

    return {
        source: float(
            counts[source] / total
        )
        for source in SOURCES
    }


def validate_shifted_skeletons(
    original: list[Skeleton],
    shifted: list[Skeleton],
    table: OutcomeTable,
    *,
    target_shares: dict[str, float],
    tolerance: float = 0.02,
) -> dict[str, float]:
    """Validate episode structure, budgets, and source shift."""

    validate_targets(
        target_shares
    )

    if len(original) != len(shifted):
        raise RuntimeError(
            "Episode count changed"
        )

    for before, after in zip(
        original,
        shifted,
    ):

        if (
            before.episode_id != after.episode_id
            or before.ep_type != after.ep_type
            or len(before.prompt_ids) != len(after.prompt_ids)
        ):
            raise RuntimeError(
                "Original episode structure changed"
            )

        if len(set(after.prompt_ids)) != len(
            after.prompt_ids
        ):
            raise RuntimeError(
                "Duplicate prompt within an episode"
            )

        if not (
            np.isfinite(after.budget_usd)
            and after.budget_usd > 0
        ):
            raise RuntimeError(
                "Invalid shifted episode budget"
            )

        try:
            before_rows = np.asarray(
                [
                    table.index[p]
                    for p in before.prompt_ids
                ],
                dtype=int,
            )

            after_rows = np.asarray(
                [
                    table.index[p]
                    for p in after.prompt_ids
                ],
                dtype=int,
            )

        except KeyError as exc:
            raise RuntimeError(
                "Non-test prompt found in an episode"
            ) from exc

        if after.ep_type == "correlated":

            if (
                len(set(
                    table.step_type[after_rows].tolist()
                )) != 1
                or len(set(
                    table.band[after_rows].tolist()
                )) != 1
            ):
                raise RuntimeError(
                    "Correlated source/band constraint violated"
                )

        original_reference = float(
            table.cost[before_rows, -1].sum()
        )

        shifted_reference = float(
            table.cost[after_rows, -1].sum()
        )

        if (
            original_reference <= 0
            or shifted_reference <= 0
        ):
            raise RuntimeError(
                "Invalid reference execution cost"
            )

        factor_before = (
            before.budget_usd
            / original_reference
        )

        factor_after = (
            after.budget_usd
            / shifted_reference
        )

        if not np.isclose(
            factor_before,
            factor_after,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Relative budget factor changed"
            )

    observed = source_distribution(
        shifted,
        table,
    )

    for source in SOURCES:

        if abs(
            observed[source]
            - target_shares[source]
        ) > tolerance:
            raise RuntimeError(
                f"Shift target not reached for {source}: "
                f"observed={observed[source]:.4f}, "
                f"target={target_shares[source]:.4f}"
            )

    return observed


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    input_path: Path = DEFAULT_INPUT,
    seed: int = 0,
) -> Path:
    """Generate, validate, and save a test-only workload."""

    output = Path(output)
    input_path = Path(input_path)

    metadata_path = output.with_name(
        output.stem + "_metadata.json"
    )

    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            "Output already exists. Use a new output path."
        )

    if output.resolve() == input_path.resolve():
        raise ValueError(
            "Cannot overwrite original test skeletons"
        )

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    original_frame = pd.read_parquet(
        input_path
    )

    # C3-13 FIX:
    # Build a genuine test-only OutcomeTable.
    table = build_test_only_table(
        clean,
        splits,
    )

    original = from_frame(
        original_frame
    )

    shifted = generate_shifted_skeletons(
        original,
        table,
        seed=seed,
    )

    observed = validate_shifted_skeletons(
        original,
        shifted,
        table,
        target_shares=TARGET_SHARES,
    )

    original_distribution = source_distribution(
        original,
        table,
    )

    shifted_frame = to_frame(
        shifted
    )

    if len(shifted_frame) != len(original_frame):
        raise RuntimeError(
            "Total step count changed"
        )

    if list(shifted_frame.columns) != list(
        original_frame.columns
    ):
        raise RuntimeError(
            "Skeleton file schema changed"
        )

    allowed_test_ids = set(
        str(prompt_id)
        for prompt_id in table.prompt_ids
    )

    shifted_ids = set(
        shifted_frame["prompt_id"].astype(str)
    )

    if not shifted_ids <= allowed_test_ids:
        raise RuntimeError(
            "Non-test prompt found in generated workload"
        )

    episode_types = Counter(
        skeleton.ep_type
        for skeleton in shifted
    )

    prompt_counts = Counter(
        shifted_frame["prompt_id"].astype(str)
    )

    metadata = {
        "task": "C3-13",
        "name": "workload-shift test set",
        "seed": seed,
        "source_split": "test",
        "original_path": str(input_path),
        "shifted_path": str(output),
        "episode_count": len(shifted),
        "step_count": len(shifted_frame),
        "eligible_test_prompt_count": len(
            allowed_test_ids
        ),
        "episode_type_counts": dict(
            episode_types
        ),
        "target_source_shares": TARGET_SHARES,
        "original_source_shares": original_distribution,
        "achieved_source_shares": observed,
        "unique_prompt_count": len(
            prompt_counts
        ),
        "repeated_prompt_steps": (
            len(shifted_frame)
            - len(prompt_counts)
        ),
        "duplicate_prompts_within_episode": False,
        "correlated_source_band_preserved": True,
        "budget_policy": (
            "Preserve each original episode's budget "
            "factor relative to strongest-model cost"
        ),
        "training_prompts_used": False,
        "validation_prompts_used": False,
        "test_only_table": True,
        "note": (
            "Test prompts may repeat across episodes, "
            "but not within the same episode."
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shifted_frame.to_parquet(
        output,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nC3-13 WORKLOAD-SHIFT TEST SET")
    print("Episodes:", len(shifted))
    print("Steps:", len(shifted_frame))
    print("Eligible test prompts:", len(allowed_test_ids))
    print("Seed:", seed)

    print("\nSOURCE DISTRIBUTION")
    print(
        f"{'Source':<18}"
        f"{'Original':>12}"
        f"{'Target':>12}"
        f"{'Shifted':>12}"
    )

    for source in SOURCES:
        print(
            f"{source:<18}"
            f"{original_distribution[source] * 100:>11.2f}%"
            f"{TARGET_SHARES[source] * 100:>11.2f}%"
            f"{observed[source] * 100:>11.2f}%"
        )

    print(
        "\nRepeated prompt steps across episodes:",
        metadata["repeated_prompt_steps"],
    )

    print("\nPASS: Test-only prompts")
    print("PASS: Original episode structure")
    print("PASS: Correlated source and difficulty band")
    print("PASS: Relative episode budget factors")
    print("PASS: Shifted workload proportions")

    print("\nSaved skeletons:", output)
    print("Saved metadata:", metadata_path)

    return output


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate a test-only shifted RouterBench workload"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run(
        input_path=args.input,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
