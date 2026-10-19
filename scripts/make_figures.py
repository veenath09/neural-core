
"""C3-09: Generate research reports from saved predictions.

Usage:
    python -m scripts.make_figures --eval-id c3-09-smoke

All reported metrics are derived exclusively from:
    results/<eval_id>/predictions.parquet

Outputs:
    results/<eval_id>/figures/baseline_summary.csv
    results/<eval_id>/figures/episode_type_summary.csv
    results/<eval_id>/figures/quality_vs_cost.png

A single operating point per policy is NOT a full
quality-cost frontier. Missing latency and confidence
measurements are not fabricated.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Two-sided Student-t critical value for 95% CI
# with three independent seed-level observations (df=2).
T_CRITICAL_3_SEEDS = 4.302652729911275

REQUIRED_COLUMNS = {
    "policy",
    "seed",
    "episode_id",
    "episode_type",
    "terminal",
    "episode_quality",
    "episode_cost_usd",
    "episode_reward",
}


def read_terminal_predictions(
    predictions_path: Path,
) -> pd.DataFrame:
    """Read and validate final episode records."""

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Missing predictions: {predictions_path}"
        )

    predictions = pd.read_parquet(predictions_path)

    missing = REQUIRED_COLUMNS - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    terminal = predictions.loc[
        predictions["terminal"].eq(True)
    ].copy()

    if terminal.empty:
        raise ValueError(
            "Predictions contain no completed episodes"
        )

    keys = ["policy", "seed", "episode_id"]

    if terminal.duplicated(keys).any():
        raise ValueError(
            "Duplicate terminal policy-episode records"
        )

    for column in (
        "episode_quality",
        "episode_cost_usd",
        "episode_reward",
    ):
        values = terminal[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid or missing {column}"
            )

    if not terminal["episode_quality"].between(
        0.0, 1.0
    ).all():
        raise ValueError(
            "Episode quality must be between 0 and 1"
        )

    if (terminal["episode_cost_usd"] < 0).any():
        raise ValueError(
            "Episode cost cannot be negative"
        )

    # All policies must be compared on the same
    # (seed, episode_id) combinations.
    comparison_sets = {}

    for policy, group in terminal.groupby("policy"):
        comparison_sets[policy] = set(
            zip(group["seed"], group["episode_id"])
        )

    reference = next(iter(comparison_sets.values()))

    for policy, combinations in comparison_sets.items():
        if combinations != reference:
            raise ValueError(
                f"Policies were evaluated on different "
                f"seed/episode combinations: {policy}"
            )

    return terminal


def confidence_interval_3_seeds(
    values: np.ndarray,
) -> tuple[float, float]:
    """95% t-interval over exactly three seed means.

    Do not substitute episodes for independent seeds.
    Missing intervals are represented as NaN.
    """

    values = np.asarray(values, dtype=float)

    if len(values) != 3:
        return math.nan, math.nan

    mean = float(values.mean())

    standard_error = float(
        values.std(ddof=1) / math.sqrt(3)
    )

    margin = (
        T_CRITICAL_3_SEEDS * standard_error
    )

    return mean - margin, mean + margin


def build_baseline_summary(
    terminal: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate policy means from saved episode results.

    With multiple seeds, average seed-level means
    so every evaluation seed has equal weight.
    """

    seed_metrics = (
        terminal.groupby(
            ["policy", "seed"],
            as_index=False,
        )
        .agg(
            episodes=("episode_id", "nunique"),
            quality=("episode_quality", "mean"),
            cost_usd=("episode_cost_usd", "mean"),
            reward=("episode_reward", "mean"),
        )
    )

    records = []

    for policy, group in seed_metrics.groupby("policy"):
        row = {
            "policy": policy,
            "seeds": int(group["seed"].nunique()),
            "episodes_per_seed": int(
                group["episodes"].min()
            ),
        }

        for source, target in (
            ("quality", "mean_quality"),
            ("cost_usd", "mean_cost_usd"),
            ("reward", "mean_reward"),
        ):
            values = group[source].to_numpy(
                dtype=float
            )

            row[target] = float(values.mean())

            lower, upper = confidence_interval_3_seeds(
                values
            )

            row[f"{target}_ci95_low"] = lower
            row[f"{target}_ci95_high"] = upper

        records.append(row)

    return (
        pd.DataFrame(records)
        .sort_values(
            ["mean_quality", "policy"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_episode_type_summary(
    terminal: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate policy performance by episode type."""

    seed_metrics = (
        terminal.groupby(
            ["policy", "episode_type", "seed"],
            as_index=False,
        )
        .agg(
            episodes=("episode_id", "nunique"),
            quality=("episode_quality", "mean"),
            cost_usd=("episode_cost_usd", "mean"),
            reward=("episode_reward", "mean"),
        )
    )

    summary = (
        seed_metrics.groupby(
            ["policy", "episode_type"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            episodes_per_seed=("episodes", "min"),
            mean_quality=("quality", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
            mean_reward=("reward", "mean"),
        )
    )

    return summary.sort_values(
        ["episode_type", "mean_quality"],
        ascending=[True, False],
    ).reset_index(drop=True)


def make_quality_cost_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed baseline policy operating points.

    Not a swept or interpolated quality-cost frontier.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in summary.iterrows():
        cost = float(row["mean_cost_usd"])
        quality = float(row["mean_quality"]) * 100

        ax.scatter(cost, quality, s=85)

        ax.annotate(
            str(row["policy"]),
            (cost, quality),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title(
        "Baseline Quality vs Cost\n"
        "Observed policy operating points"
    )

    ax.set_xlabel(
        "Mean simulated episode cost (USD)"
    )

    ax.set_ylabel(
        "Mean episode quality (%)"
    )

    ax.set_ylim(0, 105)
    ax.margins(x=0.2)

    ax.ticklabel_format(
        axis="x",
        style="plain",
        useOffset=False,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_figures(
    eval_id: str,
    results_root: str | Path = "results",
) -> Path:
    """Generate all available baseline reports."""

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        eval_id,
    ):
        raise ValueError(
            "Invalid evaluation ID"
        )

    run_dir = Path(results_root) / eval_id

    predictions_path = (
        run_dir / "predictions.parquet"
    )

    terminal = read_terminal_predictions(
        predictions_path
    )

    summary = build_baseline_summary(terminal)

    episode_types = build_episode_type_summary(
        terminal
    )

    output_dir = run_dir / "figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir / "baseline_summary.csv"
    )

    types_path = (
        output_dir / "episode_type_summary.csv"
    )

    chart_path = (
        output_dir / "quality_vs_cost.png"
    )

    summary.to_csv(
        summary_path,
        index=False,
        float_format="%.10f",
    )

    episode_types.to_csv(
        types_path,
        index=False,
        float_format="%.10f",
    )

    make_quality_cost_plot(
        summary,
        chart_path,
    )

    print("\n" + "=" * 65)
    print("C3-09 RESEARCH REPORTS")
    print("=" * 65)

    print("Source:", predictions_path)
    print("Completed policy episodes:", len(terminal))
    print(
        "Evaluation seeds:",
        sorted(terminal["seed"].unique().tolist()),
    )

    print("\nBaseline summary:")
    print(
        summary[
            [
                "policy",
                "mean_quality",
                "mean_cost_usd",
                "mean_reward",
            ]
        ].round(6).to_string(index=False)
    )

    print("\nGenerated files:")
    print(summary_path)
    print(types_path)
    print(chart_path)

    if terminal["seed"].nunique() != 3:
        print(
            "\nNOTE: 95% confidence intervals are unavailable "
            "because this run does not contain exactly "
            "three evaluation seeds."
        )
    else:
        print(
            "\n95% confidence intervals calculated from "
            "three evaluation-seed means."
        )
        print(
            "These are not automatically confidence "
            "intervals over independently trained models."
        )

    print(
        "\nNOTE: This figure shows baseline operating "
        "points, not a full swept frontier."
    )

    print(
        "Missing latency and confidence measurements "
        "are not estimated."
    )

    print(
        "\nPASS: Reports generated from saved predictions."
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate C3-09 reports from saved predictions"
        )
    )

    parser.add_argument(
        "--eval-id",
        required=True,
        help="Existing evaluation ID",
    )

    args = parser.parse_args()

    generate_figures(args.eval_id)


if __name__ == "__main__":
    main()
