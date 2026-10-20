
"""C3-15: Export a reproducible simulator execution manifest.

Selects the profile-aware, after-onboarding baseline and
joins each simulated execution attempt with its original
RouterBench prompt.

The output contains SIMULATOR observations only.

No API requests, code execution, live verification,
or paid model calls occur in this script.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


DEFAULT_PREDICTIONS = Path(
    "results/c3-12-lomo-fold3-100/predictions.parquet"
)

DEFAULT_PROMPTS = Path(
    "data/clean/routerbench_binary.parquet"
)

DEFAULT_OUTPUT = Path(
    "results/c3-15/profile_aware_after_manifest.jsonl"
)

POLICY = "profile-aware"
ONBOARDING_PHASE = "after"

EXPECTED_EPISODES = 100
EXPECTED_ATTEMPTS = 446

MANIFEST_COLUMNS = [
    "policy",
    "onboarding_phase",
    "split",
    "seed",
    "episode_id",
    "decision_index",
    "step_id",
    "round",
    "prompt_id",
    "step_type",
    "source",
    "model_id",
    "prompt_text",
    "selected_model_correct",
    "execution_cost_usd",
    "measurement_source",
]


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    """Check that the input contains every required column."""

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}"
        )


def export_manifest(
    predictions_path: Path,
    prompts_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Export the selected historical simulation attempts."""

    predictions = pd.read_parquet(
        predictions_path
    )

    prompts = pd.read_parquet(
        prompts_path
    )

    prediction_columns = [
        column
        for column in MANIFEST_COLUMNS
        if column not in (
            "source",
            "prompt_text",
            "measurement_source",
        )
    ]

    require_columns(
        predictions,
        prediction_columns + ["executed"],
        "predictions",
    )

    require_columns(
        prompts,
        [
            "prompt_id",
            "source",
            "step_type",
            "prompt_text",
        ],
        "RouterBench prompts",
    )

    selected = predictions.loc[
        (
            predictions["policy"] == POLICY
        )
        & (
            predictions["onboarding_phase"]
            == ONBOARDING_PHASE
        )
        & predictions["executed"].eq(True)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No executions found for the selected baseline"
        )

    if "measurement_source" in selected.columns:
        if not selected[
            "measurement_source"
        ].eq("simulator").all():
            raise ValueError(
                "Selected predictions include non-simulator data"
            )

    attempts = [
        "episode_id",
        "decision_index",
    ]

    if selected[attempts].isna().any().any():
        raise ValueError(
            "Missing execution attempt identifiers"
        )

    if selected.duplicated(
        subset=attempts
    ).any():
        raise ValueError(
            "Duplicate execution attempt identifiers"
        )

    episode_count = selected[
        "episode_id"
    ].nunique()

    attempt_count = len(selected)

    if episode_count != EXPECTED_EPISODES:
        raise ValueError(
            f"Expected {EXPECTED_EPISODES} episodes; "
            f"found {episode_count}"
        )

    if attempt_count != EXPECTED_ATTEMPTS:
        raise ValueError(
            f"Expected {EXPECTED_ATTEMPTS} attempts; "
            f"found {attempt_count}"
        )

    if prompts["prompt_id"].isna().any():
        raise ValueError(
            "RouterBench contains missing prompt IDs"
        )

    if prompts.duplicated(
        subset=["prompt_id"]
    ).any():
        raise ValueError(
            "RouterBench contains duplicate prompt IDs"
        )

    prompt_lookup = prompts[
        [
            "prompt_id",
            "source",
            "step_type",
            "prompt_text",
        ]
    ].rename(
        columns={
            "step_type": "dataset_step_type",
        }
    )

    manifest = selected.merge(
        prompt_lookup,
        on="prompt_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not manifest["_merge"].eq("both").all():
        missing = manifest.loc[
            manifest["_merge"] != "both",
            "prompt_id",
        ].tolist()

        raise ValueError(
            "Missing RouterBench prompts: "
            f"{missing[:10]}"
        )

    if not manifest[
        "step_type"
    ].eq(
        manifest["dataset_step_type"]
    ).all():
        raise ValueError(
            "Simulator and dataset step types disagree"
        )

    for column in (
        "prompt_id",
        "step_id",
        "model_id",
        "source",
        "prompt_text",
    ):
        valid = manifest[column].map(
            lambda value: (
                isinstance(value, str)
                and bool(value.strip())
            )
        )

        if not valid.all():
            raise ValueError(
                f"Invalid or empty {column}"
            )

    if manifest[
        MANIFEST_COLUMNS[:8]
    ].isna().any().any():
        raise ValueError(
            "Missing required evaluation identifiers"
        )

    costs = pd.to_numeric(
        manifest["execution_cost_usd"],
        errors="coerce",
    )

    if (
        costs.isna().any()
        or not (
            (costs >= 0)
            & (costs < float("inf"))
        ).all()
    ):
        raise ValueError(
            "Invalid historical execution costs"
        )

    if not manifest[
        "selected_model_correct"
    ].isin(
        [True, False]
    ).all():
        raise ValueError(
            "Invalid historical correctness values"
        )

    manifest["measurement_source"] = "simulator"

    manifest = manifest[
        MANIFEST_COLUMNS
    ].sort_values(
        [
            "episode_id",
            "decision_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
        double_precision=15,
    )

    digest = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    print("C3-15 SIMULATOR EXECUTION MANIFEST")
    print("-" * 42)

    print(
        f"Policy: {POLICY}"
    )

    print(
        f"Onboarding phase: {ONBOARDING_PHASE}"
    )

    print(
        f"Episodes: {episode_count}"
    )

    print(
        f"Execution attempts: {attempt_count}"
    )

    print(
        "Historical execution cost USD: "
        f"{costs.sum():.6f}"
    )

    print("\nModel distribution:")

    print(
        manifest["model_id"]
        .value_counts()
        .to_string()
    )

    print("\nSource distribution:")

    print(
        manifest["source"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nOutput: {output_path}"
    )

    print(
        f"SHA-256: {digest}"
    )

    print(
        "\nStatus: SIMULATOR MANIFEST ONLY"
    )

    print(
        "No real model executions were performed."
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the C3-15 simulator execution manifest"
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )

    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    export_manifest(
        predictions_path=args.predictions,
        prompts_path=args.prompts,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
