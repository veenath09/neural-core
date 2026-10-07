"""Pillar 3 C3: Tests for the RL reward and lambda-weighted oracle.

Verifies that the per-step reward sums exactly to the episode reward,
and that the lambda-oracle's objective equals that same reward once
its lambda is rescaled by the per-step cost reference.
"""

import json

import pytest

from measurement.oracle import weighted_episode_oracle
from measurement.rl_reward import (
    LAMS,
    episode_reward,
    lam_oracle,
    lam_oracle_points,
    load_cost_ref,
    step_reward,
)
from measurement.scorer import CompositeScorer

from tests.test_env import episode


def test_load_cost_ref_reads_conf():
    assert load_cost_ref() == pytest.approx(0.001756663098)


def test_episode_reward_uses_per_step_cost():
    result = episode_reward(
        scorer=_scorer_with_cost_ref(1.0),
        quality=0.5,
        spent=6,
        n_steps=3,
        lam=0.5,
    )
    assert result == pytest.approx(-0.5)


def _scorer_with_cost_ref(cost_ref: float) -> CompositeScorer:
    return CompositeScorer(
        {
            "version": 1,
            "cost_ref_usd": cost_ref,
            "cost_ref_model": "test-model",
            "cost_ref_split": "train",
            "cost_ref_statistic": "mean_per_step_cost",
            "train_prompt_count": 1,
            "latency_ref_ms": None,
            "latency_source": "unavailable_in_routerbench",
        }
    )


def test_step_rewards_sum_to_episode_reward():
    cost_ref = 1.0
    n_steps = 3
    lam = 0.5
    scorer = _scorer_with_cost_ref(cost_ref)

    # Hand-built sequence of (d_correct, d_spent) pairs across 3 steps.
    deltas = [(1, 1.0), (0, 2.0), (1, 4.0)]

    rewards = [
        step_reward(
            d_quality=d_correct,
            d_spent=d_spent,
            n_steps=n_steps,
            lam=lam,
            cost_ref=cost_ref,
        )
        for d_correct, d_spent in deltas
    ]

    total_correct = sum(d_correct for d_correct, _ in deltas)
    total_spent = sum(d_spent for _, d_spent in deltas)
    quality = total_correct / n_steps

    expected = episode_reward(
        scorer=scorer,
        quality=quality,
        spent=total_spent,
        n_steps=n_steps,
        lam=lam,
    )

    assert sum(rewards) == pytest.approx(expected)


@pytest.mark.parametrize("lam", LAMS)
def test_lam_oracle_objective_equals_reward(lam):
    cost_ref = 1.0
    scorer = _scorer_with_cost_ref(cost_ref)
    ep = episode("random", [[0, 0, 0, 1], [1, 1, 1, 1]])

    result = lam_oracle(ep, lam, cost_ref)

    expected = episode_reward(
        scorer=scorer,
        quality=result.quality,
        spent=result.cost_usd,
        n_steps=ep.n_steps,
        lam=lam,
    )

    assert result.objective == pytest.approx(expected)


def test_lam_oracle_trades_quality_for_cost():
    cost_ref = 1.0
    ep = episode("random", [[0, 0, 0, 1]])

    cheap_quality = lam_oracle(ep, 0.01, cost_ref)
    assert cheap_quality.quality == 1
    assert cheap_quality.cost_usd == 8

    expensive_quality = lam_oracle(ep, 1.0, cost_ref)
    assert expensive_quality.quality == 0
    assert expensive_quality.cost_usd == 1


def test_lam_oracle_points_shape_and_cache(table, tmp_path):
    from data.episodes import Skeleton

    model_ids = ["m0", "m1", "m2", "m3"]
    skeletons = [
        Skeleton(f"e{i}", "random", tuple(table.prompt_ids[:3]), 100.0)
        for i in range(3)
    ]

    df = lam_oracle_points(
        skeletons,
        table,
        model_ids,
        cost_ref=0.001756663098,
        cache_dir=tmp_path,
    )

    assert list(df.columns) == ["lam", "quality", "cost_usd", "reward"]
    assert len(df) == 5
    assert list(df["lam"]) == list(LAMS)

    cache_files = list(tmp_path.glob("lam_oracle_*.json"))
    assert len(cache_files) == 1

    cached_before = json.loads(cache_files[0].read_text())

    df_again = lam_oracle_points(
        skeletons,
        table,
        model_ids,
        cost_ref=0.001756663098,
        cache_dir=tmp_path,
    )

    cached_after = json.loads(cache_files[0].read_text())
    assert cached_before == cached_after
    assert list(df_again["lam"]) == list(LAMS)
