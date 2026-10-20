from __future__ import annotations

import math
from collections.abc import Sequence

from measurement.scorer import Score


def rewards_from_scores(
    scores: Sequence[Score],
    expected_count: int,
) -> list[float]:
    """Extract final C3 episode rewards for C2 learning."""

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError(
            "expected_count must be a non-negative integer"
        )

    if len(scores) != expected_count:
        raise ValueError(
            "score count must match expected episode count"
        )

    rewards: list[float] = []

    for score in scores:
        reward = score.reward

        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise ValueError(
                "score reward must be a finite number"
            )

        rewards.append(float(reward))

    return rewards