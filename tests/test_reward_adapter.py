import pytest

from measurement.scorer import Score
from train.reward_adapter import rewards_from_scores


def make_score(reward: float) -> Score:
    return Score(
        quality=0.8,
        cost_usd=0.01,
        latency_ms=None,
        penalty=0.0,
        lam=0.0,
        mu=0.0,
        cost_ref_usd=0.01,
        latency_ref_ms=None,
        normalized_cost=1.0,
        normalized_latency=None,
        reward=reward,
        latency_included=False,
    )


def test_rewards_from_scores_extracts_c3_episode_rewards():
    scores = [
        make_score(0.8),
        make_score(0.6),
    ]

    rewards = rewards_from_scores(
        scores,
        expected_count=2,
    )

    assert rewards == [0.8, 0.6]


def test_rewards_from_scores_preserves_negative_reward():
    scores = [
        make_score(-0.25),
    ]

    rewards = rewards_from_scores(
        scores,
        expected_count=1,
    )

    assert rewards == [-0.25]


def test_rewards_from_scores_rejects_count_mismatch():
    scores = [
        make_score(0.8),
    ]

    with pytest.raises(
        ValueError,
        match="score count",
    ):
        rewards_from_scores(
            scores,
            expected_count=2,
        )