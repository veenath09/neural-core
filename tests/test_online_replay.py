"""Tests for Part 2 Task 4: run_episode and replay recompute (train/online.py)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from controller.embedding_cache import EmbeddingCache
from controller.features import HALT
from controller.policy import log_price
from controller.session import ControllerSession
from data.env import Episode
from data.episodes import Skeleton
from measurement.online_env import SimStepEnv
from measurement.rl_reward import LAMS, episode_reward, load_cost_ref
from measurement.scorer import CompositeScorer
from tests.test_env import IDS
from tests.test_rollout import tiny_model, world_and_episodes
from train.online import EpisodeLog, replay, run_episode


def _session(model, world, cost_ref):
    emb = EmbeddingCache(model, dict(world.emb_index), embed_fn=None)
    avail = np.ones(len(world.model_ids), dtype=bool)
    return ControllerSession(model, lambda: world.card_costs, cost_ref, emb,
                             lambda: list(world.model_ids), lambda: avail, lams=LAMS)


def _run(model, world, eps, mode, bonus=None, quality_source="verifier", version=0):
    cost_ref = load_cost_ref()
    session = _session(model, world, cost_ref)
    bonus = np.zeros(len(world.model_ids)) if bonus is None else bonus
    logs, envs = [], []
    for k, ep in enumerate(eps):
        env = SimStepEnv(ep, quality_source=quality_source)
        logs.append(run_episode(session, env, k % len(LAMS), mode, lambda: bonus, None, version))
        envs.append(env)
    return logs, envs


def _price_fn(world):
    cost_ref = load_cost_ref()
    return lambda pool_size: log_price(world.card_costs[:pool_size], cost_ref)


@pytest.mark.parametrize("name", ["full", "full_attn", "context_free"])
def test_replay_reproduces_behaviour_logp(name):
    world, eps = world_and_episodes(12)
    model = tiny_model(name)
    # A non-zero bonus checks that replay reapplies the logged bonus.
    bonus = np.array([0.0, 0.3, -0.2, 0.5])
    logs, _ = _run(model, world, eps, "sample", bonus=bonus)
    assert len({len(log.decisions) for log in logs}) > 1  # lockstep must handle ragged episodes

    trace, behaviour = replay(model, logs, _price_fn(world))

    n, T = trace.valid.shape
    assert behaviour.shape == (n, T) and n == len(logs)
    assert T == max(len(log.decisions) for log in logs)
    assert trace.choice.any() and not (trace.choice & ~trace.valid).any()
    assert trace.lam_idx.tolist() == [log.lam_idx for log in logs]
    for i, log in enumerate(logs):
        t = len(log.decisions)
        assert int(trace.valid[i].sum()) == t and bool(trace.valid[i, :t].all())
        assert trace.choice[i, :t].tolist() == [d.has_choice for d in log.decisions]
        assert trace.halt_slot[i, :t].tolist() == [d.phase == HALT for d in log.decisions]
        assert torch.allclose(trace.reward[i, :t], torch.tensor([d.reward for d in log.decisions]))
        assert torch.allclose(trace.complexity[i, :t].detach(),
                              torch.tensor([d.complexity for d in log.decisions]), atol=1e-5)
    assert ((trace.band >= 0) & (trace.band <= 4)).all()

    c = trace.choice
    assert torch.allclose(trace.logp[c].detach(), behaviour[c], atol=1e-5), (
        (trace.logp[c].detach() - behaviour[c]).abs().max())
    # Forced stops (no escalation option) carry no log-prob in either.
    assert (trace.logp[trace.valid & ~c].detach() == 0).all() and (behaviour[trace.valid & ~c] == 0).all()


def test_replay_has_grad_through_memory():
    world, eps = world_and_episodes(6)
    model = tiny_model("full")
    logs, _ = _run(model, world, eps, "sample")
    trace, _ = replay(model, logs, _price_fn(world))
    model.zero_grad()
    trace.logp[trace.choice].sum().backward()
    assert model.memory.gate.readout.weight.grad.abs().sum() > 0


def test_run_episode_rewards_sum_to_episode_reward():
    world, eps = world_and_episodes(8)
    table = world.table
    cheapest = float(table.cost.min())
    # Out of budget at once, and part-way through: unreached steps still count with quality 0.
    eps = eps + [Episode(Skeleton("broke", "random", tuple(table.prompt_ids[:3]), 0.25 * cheapest), table, IDS),
                 Episode(Skeleton("short", "random", tuple(table.prompt_ids[:4]),
                                  float(table.cost[0, 3]) * 1.5), table, IDS)]
    model = tiny_model("full")
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    for mode in ("sample", "serve"):
        logs, envs = _run(model, world, eps, mode, quality_source="truth")
        for log, env, ep in zip(logs, envs, eps):
            assert isinstance(log, EpisodeLog)
            assert log.n_steps == ep.n_steps
            assert log.quality == pytest.approx(env.quality_truth)  # truth source: same as the simulator
            assert log.cost_usd == pytest.approx(env.state.spent)
            expected = episode_reward(scorer, log.quality, log.cost_usd, log.n_steps, LAMS[log.lam_idx])
            assert sum(d.reward for d in log.decisions) == pytest.approx(expected, abs=1e-9)
            assert len(log.used) == len(env.state.trace)
            assert [u[0] for u in log.used] == [it.model_id for it in env.state.trace]
    assert logs[-2].quality == 0.0 and len(logs[-2].decisions) == 1


def _tight(eps, factor):
    """The same episodes with budget * factor and fresh ids (distinct verifier and audit flips)."""
    return [Episode(Skeleton(f"{ep.skeleton.episode_id}-b{factor}", ep.skeleton.ep_type, ep.skeleton.prompt_ids,
                             ep.skeleton.budget_usd * factor), ep.table, IDS) for ep in eps]


def test_verifier_rewards_are_unbiased_for_true_quality():
    """Rewards from the 15%-flip verifier (audited, de-biased) average to the rewards of what actually happened
    (true quality of the final answers), including episodes that run out of budget. Every episode has its own
    id, so its flips are independent draws (copies of one episode would repeat the same flips)."""
    world, eps = world_and_episodes(192)
    eps = _tight(eps, 0.15) + _tight(eps, 0.2) + eps[:64]
    assert len({ep.skeleton.episode_id for ep in eps}) == len(eps)
    model = tiny_model("full")
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    logs, envs = _run(model, world, eps, "sample", quality_source="verifier")
    assert sum(env.state.step_index < ep.n_steps for env, ep in zip(envs, eps)) > 200  # many budget-cut episodes
    r_noisy = np.mean([sum(d.reward for d in log.decisions) for log in logs])
    r_truth = np.mean([episode_reward(scorer, env.quality_truth, log.cost_usd, log.n_steps, LAMS[log.lam_idx])
                       for log, env in zip(logs, envs)])
    assert abs(r_noisy - r_truth) < 0.02, (r_noisy, r_truth)


class _AuditAlwaysPasses(SimStepEnv):
    """Verifier-sourced env whose audit always reads 1, so a step's reward quality is exactly known."""

    def play(self, model_id, round):
        out = super().play(model_id, round)
        return out if out.trace_item is None else replace(out, audit_quality=1.0)


def test_noise_correction_leaves_unplayed_steps_at_zero():
    """Only steps whose final round was played and audited are de-biased; steps never reached, and a step
    whose round was cut by the budget, are exact zeros (not (0 - f) / (1 - 2f))."""
    world, eps = world_and_episodes(24)
    table = world.table
    cheapest = float(table.cost.min())
    eps = eps + _tight(eps, 0.2) + [
        Episode(Skeleton("broke", "random", tuple(table.prompt_ids[:3]), 0.25 * cheapest), table, IDS)]
    model = tiny_model("full")
    cost_ref, f = load_cost_ref(), 0.15
    session = _session(model, world, cost_ref)
    bonus = np.zeros(len(world.model_ids))
    cut = 0
    for k, ep in enumerate(eps):
        env = _AuditAlwaysPasses(ep, quality_source="verifier")
        log = run_episode(session, env, k % len(LAMS), "sample", lambda: bonus, None, 0)
        scored = env.state.step_index  # steps whose final round was played (end_step ran)
        cut += scored < ep.n_steps
        lam = LAMS[log.lam_idx]
        want = scored * (1 - f) / (1 - 2 * f) / ep.n_steps - lam * log.cost_usd / (ep.n_steps * cost_ref)
        assert sum(d.reward for d in log.decisions) == pytest.approx(want, abs=1e-9), ep.skeleton.episode_id
    assert cut > 5
    assert sum(d.reward for d in log.decisions) == 0.0  # "broke": nothing played, nothing spent


def test_serve_mode_logs_are_marked():
    world, eps = world_and_episodes(6)
    model = tiny_model("full")
    a, _ = _run(model, world, eps, "serve", version=3)
    b, _ = _run(model, world, eps, "serve", version=3)
    s, _ = _run(model, world, eps, "sample", version=3)
    for la, lb, ls, ep in zip(a, b, s, eps):
        assert la.mode == lb.mode == "serve" and ls.mode == "sample"
        assert la.episode_id == ep.skeleton.episode_id and la.ep_type == ep.skeleton.ep_type
        assert la.policy_version == 3 and all(d.policy_version == 3 for d in la.decisions)
        assert [d.action for d in la.decisions] == [d.action for d in lb.decisions]
        assert (la.quality, la.cost_usd) == (lb.quality, lb.cost_usd)


def test_bonus_is_in_pre_sharpness_units():
    """At sharpness k the bonus shifts logits by k * bonus, as if added before sharpening."""
    import math

    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    with torch.no_grad():
        model.heads.log_sharpness.fill_(math.log(10.0))
    bonus = np.array([0.0, 0.0, 0.0, 0.5])
    logs, _ = _run(model, world, eps, "sample", bonus=bonus)
    trace, behaviour = replay(model, logs, _price_fn(world))
    c = trace.choice
    assert torch.allclose(trace.logp[c].detach(), behaviour[c], atol=1e-5)  # replay still reproduces behaviour
    zero, _ = _run(model, world, eps[:1], "serve", bonus=np.zeros(4))
    big, _ = _run(model, world, eps[:1], "serve", bonus=np.array([0.0, 0.0, 0.0, 5.0]))
    assert big[0].decisions[0].action == 3  # 5.0 pre-sharpness = 50 post: dominates
