"""Every RL reward is scored with the pool's own cost_ref (final review, finding 1).

R = quality - lam * spent / (n_steps * cost_ref), with the cost_ref the caller passes (cost_ref_for(data_dir)),
never the RouterBench value in conf/score.yaml."""

import numpy as np
import pytest
import torch

from measurement.rl_reward import LAMS, load_cost_ref
from tests.test_rollout import tiny_model, world_and_episodes
from train.rollout import rollout

COST_REF = 7.0 * load_cost_ref()  # a depth-pool-like reference, far from the RouterBench one


def _expected(model, eps, world, lam_idx, policy, rng=None):
    with torch.no_grad():
        out = rollout(model, eps, world, policy=policy, rng=rng, lam_idx=np.full(len(eps), lam_idx), cost_ref=COST_REF)
    lam = LAMS[lam_idx]
    return np.array([r.quality - lam * r.cost_usd / (e.n_steps * COST_REF) for r, e in zip(out.results, eps)])


def test_evaluate_rl_scores_with_the_given_cost_ref():
    from train.rl import evaluate_rl

    world, eps = world_and_episodes(16)
    model = tiny_model("full").eval()
    sks = [e.skeleton for e in eps]
    for lam_idx in (1, 4):
        got = evaluate_rl(model, sks, world, lam_idx, COST_REF)["reward"]
        assert got == pytest.approx(_expected(model, eps, world, lam_idx, "serve").mean(), abs=1e-9)


def test_sample_reward_scores_with_the_given_cost_ref():
    from train.policy_init import sample_reward

    world, eps = world_and_episodes(16)
    model = tiny_model("full").eval()
    sks = [e.skeleton for e in eps]
    got = sample_reward(model, sks, world, COST_REF, 4, seed=3)
    want = _expected(model, eps, world, 4, "sample", rng=np.random.default_rng(3)).mean()
    assert got == pytest.approx(want, abs=1e-9)


def test_online_sim_episode_rewards_score_with_the_given_cost_ref():
    from eval.online_sim import episode_rewards

    world, eps = world_and_episodes(16)
    model = tiny_model("full").eval()
    got = episode_rewards(model, [e.skeleton for e in eps], world, 4, COST_REF)
    assert np.allclose(got, _expected(model, eps, world, 4, "serve"), atol=1e-12)
