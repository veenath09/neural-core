from __future__ import annotations

import math
import statistics


def forgetting_score(
    before_shift: float,
    after_shift: float,
) -> float:
    return before_shift - after_shift

def recovery_score(
    after_shift: float,
    after_return: float,
) -> float:
    return after_return - after_shift


def retention_summary(
    before_shift: float,
    after_shift: float,
    after_return: float,
) -> dict[str, float]:
    forgetting = forgetting_score(
        before_shift=before_shift,
        after_shift=after_shift,
    )

    recovery = recovery_score(
        after_shift=after_shift,
        after_return=after_return,
    )

    final_gap = (
        before_shift - after_return
    )

    return {
        "forgetting": forgetting,
        "recovery": recovery,
        "final_gap": final_gap,
    }

def retention_run(
    seed: int,
    before_shift: float,
    after_shift: float,
    after_return: float,
) -> dict[str, float | int]:
    metrics = retention_summary(
        before_shift=before_shift,
        after_shift=after_shift,
        after_return=after_return,
    )

    return {
        "seed": seed,
        "before_shift": before_shift,
        "after_shift": after_shift,
        "after_return": after_return,
        **metrics,
    }

def aggregate_retention_runs(
    runs: list[dict],
) -> dict:
    if not runs:
        raise ValueError(
            "runs must not be empty"
        )

    forgetting_values = [
        run["forgetting"]
        for run in runs
    ]

    recovery_values = [
        run["recovery"]
        for run in runs
    ]

    final_gap_values = [
        run["final_gap"]
        for run in runs
    ]

    mean_forgetting = (
        sum(forgetting_values)
        / len(forgetting_values)
    )

    mean_recovery = (
        sum(recovery_values)
        / len(recovery_values)
    )

    mean_final_gap = (
        sum(final_gap_values)
        / len(final_gap_values)
    )

    forgetting_ci = (
        mean_confidence_interval_95(
            forgetting_values
        )
    )

    recovery_ci = (
        mean_confidence_interval_95(
            recovery_values
        )
    )

    final_gap_ci = (
        mean_confidence_interval_95(
            final_gap_values
        )
    )

    return {
        "n_runs": len(runs),
        "mean_forgetting": mean_forgetting,
        "mean_recovery": mean_recovery,
        "mean_final_gap": mean_final_gap,
        "forgetting_ci": forgetting_ci,
        "recovery_ci": recovery_ci,
        "final_gap_ci": final_gap_ci,
    }

def mean_confidence_interval_95(
    values: list[float],
) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError(
            "at least two values are required"
        )

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)

    # 95% two-sided Student's t critical values.
    t_critical_95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }

    degrees_of_freedom = n - 1

    if degrees_of_freedom in t_critical_95:
        t_critical = t_critical_95[
            degrees_of_freedom
        ]
    elif degrees_of_freedom < 25:
        # Use the nearest lower available df value.
        available = [
            df
            for df in t_critical_95
            if df <= degrees_of_freedom
        ]

        t_critical = t_critical_95[
            max(available)
        ]
    else:
        # Conservative approximation for larger samples.
        t_critical = 1.96

    standard_error = (
        std / math.sqrt(n)
    )

    margin = (
        t_critical * standard_error
    )

    return {
        "n": n,
        "mean": mean,
        "lower": mean - margin,
        "upper": mean + margin,
    }