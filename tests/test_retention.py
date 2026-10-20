import pytest
from train.retention import (
    forgetting_score,
    recovery_score,
    retention_summary,
    retention_run,
    aggregate_retention_runs,
    mean_confidence_interval_95,
)

def test_forgetting_score_detects_performance_drop():
    before_shift = 0.80
    after_shift = 0.65

    forgetting = forgetting_score(
        before_shift=before_shift,
        after_shift=after_shift,
    )

    assert forgetting == pytest.approx(0.15)

def test_recovery_score_measures_return_to_a_improvement():
    after_shift = 0.65
    after_return = 0.76

    recovery = recovery_score(
        after_shift=after_shift,
        after_return=after_return,
    )

    assert recovery == pytest.approx(0.11)

def test_retention_summary_reports_full_aba_metrics():
    result = retention_summary(
        before_shift=0.80,
        after_shift=0.65,
        after_return=0.76,
    )

    assert result["forgetting"] == pytest.approx(0.15)
    assert result["recovery"] == pytest.approx(0.11)
    assert result["final_gap"] == pytest.approx(0.04)


def test_retention_run_records_seed_and_metrics():
    result = retention_run(
        seed=0,
        before_shift=0.80,
        after_shift=0.65,
        after_return=0.76,
    )

    assert result["seed"] == 0
    assert result["before_shift"] == pytest.approx(0.80)
    assert result["after_shift"] == pytest.approx(0.65)
    assert result["after_return"] == pytest.approx(0.76)

    assert result["forgetting"] == pytest.approx(0.15)
    assert result["recovery"] == pytest.approx(0.11)
    assert result["final_gap"] == pytest.approx(0.04)


def test_aggregate_retention_runs_reports_mean_metrics():
    runs = [
        retention_run(
            seed=0,
            before_shift=0.80,
            after_shift=0.65,
            after_return=0.76,
        ),
        retention_run(
            seed=1,
            before_shift=0.82,
            after_shift=0.70,
            after_return=0.79,
        ),
        retention_run(
            seed=2,
            before_shift=0.78,
            after_shift=0.66,
            after_return=0.75,
        ),
    ]

    result = aggregate_retention_runs(
        runs
    )

    assert result["n_runs"] == 3

    assert result["mean_forgetting"] == pytest.approx(
        (0.15 + 0.12 + 0.12) / 3
    )

    assert result["mean_recovery"] == pytest.approx(
        (0.11 + 0.09 + 0.09) / 3
    )

    assert result["mean_final_gap"] == pytest.approx(
        (0.04 + 0.03 + 0.03) / 3
    )

def test_mean_confidence_interval_contains_mean():
    values = [
        0.15,
        0.12,
        0.12,
    ]

    result = mean_confidence_interval_95(
        values
    )

    assert result["n"] == 3

    assert result["mean"] == pytest.approx(
        0.13
    )

    assert result["lower"] < result["mean"]

    assert result["upper"] > result["mean"]


def test_aggregate_retention_runs_includes_confidence_intervals():
    runs = [
        retention_run(
            seed=0,
            before_shift=0.80,
            after_shift=0.65,
            after_return=0.76,
        ),
        retention_run(
            seed=1,
            before_shift=0.82,
            after_shift=0.70,
            after_return=0.79,
        ),
        retention_run(
            seed=2,
            before_shift=0.78,
            after_shift=0.66,
            after_return=0.75,
        ),
    ]

    result = aggregate_retention_runs(
        runs
    )

    assert result["forgetting_ci"]["n"] == 3
    assert result["recovery_ci"]["n"] == 3
    assert result["final_gap_ci"]["n"] == 3

    assert result["forgetting_ci"]["mean"] == pytest.approx(
        result["mean_forgetting"]
    )

    assert result["recovery_ci"]["mean"] == pytest.approx(
        result["mean_recovery"]
    )

    assert result["final_gap_ci"]["mean"] == pytest.approx(
        result["mean_final_gap"]
    )