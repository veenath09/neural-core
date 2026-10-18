
"""C3-09: Automated tests for saved-prediction reports.

Tests data validation, baseline summaries, confidence
intervals, episode-type tables, and generated figures.

No RouterBench downloads or paid API calls are required.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_figures import (
    read_terminal_predictions,
    confidence_interval_3_seeds,
    build_baseline_summary,
    build_episode_type_summary,
    make_quality_cost_plot,
    generate_figures,
)


def sample_predictions(seeds=(0, 1, 2)):
    """Generate synthetic saved-prediction records."""

    rows = []

    episodes = (
        ("episode-a", "random"),
        ("episode-b", "retry"),
    )

    policies = (
        ("always-cheapest", 0.2, 0.001),
        ("episode-oracle", 0.8, 0.004),
    )

    for seed in seeds:
        for episode_index, (
            episode_id,
            episode_type,
        ) in enumerate(episodes):

            for policy, base_quality, cost in policies:
                quality = (
                    base_quality
                    + seed * 0.02
                    + episode_index * 0.01
                )

                rows.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "episode_id": episode_id,
                        "episode_type": episode_type,
                        "terminal": True,
                        "episode_quality": quality,
                        "episode_cost_usd": cost,
                        "episode_reward": (
                            quality - 0.1 * cost
                        ),
                    }
                )

    return pd.DataFrame(rows)


def write_predictions(tmp_path, frame, eval_id="demo"):
    """Write test predictions into a temporary directory."""

    directory = tmp_path / eval_id
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / "predictions.parquet"
    frame.to_parquet(path, index=False)

    return path


def test_saved_predictions_are_loaded(tmp_path):
    frame = sample_predictions()
    path = write_predictions(tmp_path, frame)

    loaded = read_terminal_predictions(path)

    assert len(loaded) == 12
    assert set(loaded["policy"]) == {
        "always-cheapest",
        "episode-oracle",
    }


def test_missing_predictions_file_is_rejected(tmp_path):
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        read_terminal_predictions(path)


def test_missing_required_column_is_rejected(tmp_path):
    frame = sample_predictions()
    frame = frame.drop(columns=["episode_reward"])

    path = write_predictions(tmp_path, frame)

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        read_terminal_predictions(path)


def test_no_terminal_records_are_rejected(tmp_path):
    frame = sample_predictions()
    frame["terminal"] = False

    path = write_predictions(tmp_path, frame)

    with pytest.raises(
        ValueError,
        match="no completed episodes",
    ):
        read_terminal_predictions(path)


def test_duplicate_terminal_records_are_rejected(tmp_path):
    frame = sample_predictions()
    frame = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    path = write_predictions(tmp_path, frame)

    with pytest.raises(
        ValueError,
        match="Duplicate terminal",
    ):
        read_terminal_predictions(path)


def test_unequal_episode_sets_are_rejected(tmp_path):
    frame = sample_predictions()

    frame = frame.loc[
        ~(
            (frame["policy"] == "episode-oracle")
            & (frame["episode_id"] == "episode-b")
            & (frame["seed"] == 0)
        )
    ]

    path = write_predictions(tmp_path, frame)

    with pytest.raises(
        ValueError,
        match="different",
    ):
        read_terminal_predictions(path)


@pytest.mark.parametrize(
    "bad_quality",
    [-0.1, 1.1, float("nan")],
)
def test_invalid_quality_is_rejected(
    tmp_path,
    bad_quality,
):
    frame = sample_predictions()
    frame.loc[0, "episode_quality"] = bad_quality

    path = write_predictions(tmp_path, frame)

    with pytest.raises(ValueError):
        read_terminal_predictions(path)


def test_negative_cost_is_rejected(tmp_path):
    frame = sample_predictions()
    frame.loc[0, "episode_cost_usd"] = -0.01

    path = write_predictions(tmp_path, frame)

    with pytest.raises(
        ValueError,
        match="cost",
    ):
        read_terminal_predictions(path)


def test_missing_reward_is_rejected(tmp_path):
    frame = sample_predictions()
    frame.loc[0, "episode_reward"] = float("nan")

    path = write_predictions(tmp_path, frame)

    with pytest.raises(ValueError):
        read_terminal_predictions(path)


def test_baseline_summary_has_correct_episode_counts():
    terminal = sample_predictions()

    summary = build_baseline_summary(terminal)

    assert len(summary) == 2
    assert summary["seeds"].eq(3).all()
    assert summary["episodes_per_seed"].eq(2).all()


def test_baseline_quality_is_calculated_correctly():
    terminal = sample_predictions()

    summary = build_baseline_summary(terminal)

    cheap = summary.loc[
        summary["policy"] == "always-cheapest"
    ].iloc[0]

    expected = (
        0.2
        + 0.02
        + 0.005
    )

    assert cheap["mean_quality"] == pytest.approx(
        expected
    )

    assert cheap["mean_cost_usd"] == pytest.approx(
        0.001
    )


def test_three_seed_confidence_interval_is_available():
    terminal = sample_predictions()

    summary = build_baseline_summary(terminal)

    assert summary[
        "mean_quality_ci95_low"
    ].notna().all()

    assert summary[
        "mean_quality_ci95_high"
    ].notna().all()

    assert (
        summary["mean_quality_ci95_low"]
        <= summary["mean_quality"]
    ).all()

    assert (
        summary["mean_quality"]
        <= summary["mean_quality_ci95_high"]
    ).all()


def test_one_seed_confidence_interval_is_unavailable():
    terminal = sample_predictions(seeds=(0,))

    summary = build_baseline_summary(terminal)

    assert summary["seeds"].eq(1).all()

    assert summary[
        "mean_quality_ci95_low"
    ].isna().all()

    assert summary[
        "mean_quality_ci95_high"
    ].isna().all()


def test_confidence_interval_helper():
    import numpy as np

    low, high = confidence_interval_3_seeds(
        np.array([0.2, 0.3, 0.4])
    )

    assert math.isfinite(low)
    assert math.isfinite(high)
    assert low < 0.3 < high


def test_confidence_interval_requires_three_seeds():
    import numpy as np

    low, high = confidence_interval_3_seeds(
        np.array([0.2, 0.3])
    )

    assert math.isnan(low)
    assert math.isnan(high)


def test_episode_type_summary():
    terminal = sample_predictions()

    summary = build_episode_type_summary(terminal)

    assert len(summary) == 4

    assert set(summary["episode_type"]) == {
        "random",
        "retry",
    }

    assert summary["seeds"].eq(3).all()


def test_quality_cost_plot_is_created(tmp_path):
    terminal = sample_predictions()
    summary = build_baseline_summary(terminal)

    output = tmp_path / "quality_vs_cost.png"

    make_quality_cost_plot(summary, output)

    assert output.is_file()
    assert output.stat().st_size > 0

    with output.open("rb") as file:
        assert file.read(8) == b"\x89PNG\r\n\x1a\n"


def test_generate_figures_creates_all_outputs(tmp_path):
    frame = sample_predictions()
    write_predictions(tmp_path, frame, eval_id="test-run")

    output_dir = generate_figures(
        "test-run",
        results_root=tmp_path,
    )

    expected = (
        "baseline_summary.csv",
        "episode_type_summary.csv",
        "quality_vs_cost.png",
    )

    for filename in expected:
        path = output_dir / filename

        assert path.is_file()
        assert path.stat().st_size > 0


def test_generated_csv_matches_saved_predictions(tmp_path):
    frame = sample_predictions()
    write_predictions(tmp_path, frame, eval_id="test-run")

    output_dir = generate_figures(
        "test-run",
        results_root=tmp_path,
    )

    saved_summary = pd.read_csv(
        output_dir / "baseline_summary.csv"
    )

    expected = build_baseline_summary(frame)

    saved_summary = saved_summary.sort_values(
        "policy"
    ).reset_index(drop=True)

    expected = expected.sort_values(
        "policy"
    ).reset_index(drop=True)

    for column in (
        "mean_quality",
        "mean_cost_usd",
        "mean_reward",
    ):
        assert saved_summary[column].tolist() == (
            pytest.approx(expected[column].tolist())
        )


def test_invalid_evaluation_id_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="Invalid evaluation ID",
    ):
        generate_figures(
            "../invalid",
            results_root=tmp_path,
        )


def test_figures_require_saved_predictions(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_figures(
            "not-created",
            results_root=tmp_path,
        )
