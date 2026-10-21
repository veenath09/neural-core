"""Escalation goes to a strictly pricier model by card cost, not by pool index (final review, finding 4).

PoolState appends joiners at the end, so a cheap joiner has the highest index: under the index rule it
became an "escalation" target from every model and could never escalate itself."""

import numpy as np
import pytest
import torch

from controller.embedding_cache import EmbeddingCache
from controller.features import HALT
from controller.session import ControllerSession
from data.env import Episode
from eval.online_sim import subset_world
from measurement.online_env import SimStepEnv
from measurement.rl_reward import LAMS, load_cost_ref
from tests.test_env import IDS
from tests.test_rollout import tiny_model, world_and_episodes
from train.online import run_episode
from train.rollout import rollout
from train.train import make_episodes


def _cheap_joiner_world():
    """The test pool with its cheapest model (m0) moved to the end, as if it had joined last."""
    world, eps = world_and_episodes(24)
    order = IDS[1:] + IDS[:1]
    return subset_world(world, order), [ep.skeleton for ep in eps]


def test_episode_escalates_by_model_cost_when_given():
    world, sks = _cheap_joiner_world()
    costs = world.card_costs  # [2, 4, 8, 1] * 1e-4
    ep = Episode(sks[0], world.table, world.model_ids, model_costs=costs)
    s = ep.initial_state()
    s = s.__class__(**{**s.__dict__, "budget_left": 1e9})
    after0 = ep.play(s, 0)
    assert 3 not in ep.escalation_options(after0)
    assert ("play", 3) not in ep.legal_actions(after0)
    with pytest.raises(ValueError):
        ep.play(after0, 3)
    after3 = ep.play(s, 3)  # the cheap joiner can escalate to every pricier model
    assert ep.escalation_options(after3) == [0, 1, 2]
    assert ep.play(after3, 0).rounds == 2
    # default (no model_costs): the index rule, unchanged
    plain = Episode(sks[0], world.table, world.model_ids)
    assert plain.escalation_options(plain.play(s, 3)) == []


def test_make_episodes_carries_card_costs():
    world, sks = _cheap_joiner_world()
    ep = make_episodes(sks[:1], world)[0]
    assert np.array_equal(ep.model_costs, world.card_costs)


def test_session_halt_options_are_pricier_by_card_cost():
    world, sks = _cheap_joiner_world()
    costs = world.card_costs
    model = tiny_model("full")
    with torch.no_grad():
        model.heads.halt.bias.fill_(50.0)  # always wants to escalate
    emb = EmbeddingCache(model, dict(world.emb_index), embed_fn=None)
    avail = np.ones(4, dtype=bool)
    session = ControllerSession(model, lambda: costs, load_cost_ref(), emb, lambda: list(world.model_ids),
                                lambda: avail, lams=LAMS)
    bonus = np.array([0.0, 0.0, 0.0, 3.0])  # push CHOOSE toward the cheap joiner, so it is escalated from
    from_joiner = 0
    for k, sk in enumerate(sks):
        env = SimStepEnv(Episode(sk, world.table, world.model_ids, model_costs=costs))
        log = run_episode(session, env, k % len(LAMS), "sample", lambda: bonus, None, 0)
        for prev, d in zip(log.decisions, log.decisions[1:]):
            if d.phase != HALT:
                continue
            cur = prev.action
            assert all(costs[m] > costs[cur] for m in range(4) if d.mask[m]), (cur, d.mask)
            if cur == 3 and d.action >= 0:
                from_joiner += 1
                assert d.action in (0, 1, 2)
    assert from_joiner > 0


def test_rollout_escalates_by_card_cost():
    world, sks = _cheap_joiner_world()
    model = tiny_model("full").eval()
    with torch.no_grad():
        model.heads.halt.bias.fill_(50.0)
    eps = make_episodes(sks, world)
    for policy, kw in (("serve", {"lam_idx": np.zeros(len(eps), dtype=int), "cost_ref": load_cost_ref()}),
                       ("sample", {"lam_idx": np.zeros(len(eps), dtype=int), "cost_ref": load_cost_ref()}),
                       ("greedy", {"halt_tau": 0.0})):
        with torch.no_grad():
            out = rollout(model, eps, world, policy=policy, rng=np.random.default_rng(0), **kw)
        n_esc = 0
        for acts in out.decision_actions:
            for a, b in zip(acts, acts[1:]):
                if b["phase"] == "halt" and b["halt"] == "escalate":
                    n_esc += 1
                    assert world.card_costs[b["model"]] > world.card_costs[a["model"]], (policy, a, b)
        assert n_esc > 0, policy
