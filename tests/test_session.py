import numpy as np
import pytest
import torch

from controller.contracts import ModelCard, validate_strategy
from controller.embedding_cache import EmbeddingCache
from controller.features import CHOOSE, HALT
from controller.policy import choose_step, halt_step, score_choose, score_halt
from controller.session import ControllerSession, DecisionLog
from measurement.rl_reward import LAMS, load_cost_ref
from tests.test_env import IDS
from tests.test_rollout import tiny_model, world_and_episodes
from train.rollout import rollout


def test_score_matches_step():
    torch.manual_seed(3)
    B, M = 64, 5
    logits = torch.randn(B, M, requires_grad=True)
    allowed = torch.rand(B, M) < 0.6
    allowed[:, 2] |= ~allowed.any(1)
    gen = torch.Generator().manual_seed(0)
    m, logp, ent = choose_step(logits, allowed, "sample", gen)
    s_logp, s_ent = score_choose(logits, allowed, m)
    assert torch.equal(s_logp, logp) and torch.equal(s_ent, ent)
    s_logp.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    x = torch.randn(B, requires_grad=True)
    options = torch.rand(B, M) < 0.4
    options[:8] = False  # forced stops
    m2, logp, ent, has = halt_step(x, logits, options, "sample", torch.Generator().manual_seed(1))
    assert (m2 == -1).any() and (m2 >= 0).any()
    s_logp, s_ent, s_has = score_halt(x, logits, options, m2)
    assert torch.equal(s_logp, logp) and torch.equal(s_ent, ent) and torch.equal(s_has, has)
    s_logp.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def _pool_cards(world):
    return [ModelCard(model_id=m, provider="sim", tier=i, mean_cost_usd=float(world.card_costs[i]),
                      price_source="test", price_date="2026-10-21", capability=[0.5, 0.5, -4.0])
            for i, m in enumerate(world.model_ids)]


def _session(model, world, cost_ref, available=None):
    emb = EmbeddingCache(model, dict(world.emb_index), embed_fn=None)
    avail = np.ones(len(world.model_ids), dtype=bool) if available is None else available
    return ControllerSession(model, lambda: world.card_costs, cost_ref, emb,
                             lambda: list(world.model_ids), lambda: avail, lams=LAMS)


def drive(session, ep, lam_idx, mode="serve", seed=0, cards=None):
    """Mirror of rollout's state machine over a data.env.Episode, fed to a session."""
    sk = ep.skeleton
    session.reset(sk.episode_id, sk.prompt_ids[0], None, sk.budget_usd, ep.n_steps, lam_idx, mode, seed=seed)
    zero = np.zeros(ep.n_models)
    s = ep.initial_state()
    actions, strategies, passes = [], [], 0
    while not s.done:
        strat = session.decide(f"{sk.episode_id}:{s.step_index}", sk.prompt_ids[s.slot], None,
                               int(ep.table.step_type[ep.row(s)]), zero)
        passes += 1
        strategies.append(strat)
        m = IDS.index(strat.model_id)
        actions.append({"phase": "choose", "model": m, "halt": None})
        s = ep.play(s, m)
        if s.done:  # unaffordable round: the episode ends
            break
        nxt = session.after_round(s.trace[-1], zero)
        passes += 1
        if nxt is None:
            actions.append({"phase": "halt", "model": None, "halt": "stop"})
            s = ep.end_step(s)
            session.end_step()
            continue
        strategies.append(nxt)
        m2 = IDS.index(nxt.model_id)
        actions.append({"phase": "halt", "model": m2, "halt": "escalate"})
        s = ep.play(s, m2)
        if s.done:
            break
        assert session.after_round(s.trace[-1], zero) is None
        s = ep.end_step(s)  # automatic end of step after round 2
        session.end_step()
    return actions, ep.quality(s), s.spent, strategies, passes, session.finish()


@pytest.mark.parametrize("lam_i", [0, 2, 4])
def test_session_matches_serve_rollout(lam_i):
    world, eps = world_and_episodes()
    cost_ref = load_cost_ref()
    model = tiny_model().eval()
    with torch.no_grad():
        out = rollout(model, eps, world, policy="serve", lam_idx=np.full(len(eps), lam_i), cost_ref=cost_ref)
    session = _session(model, world, cost_ref)
    for i, ep in enumerate(eps):
        actions, quality, spent, *_ = drive(session, ep, lam_i)
        assert actions == out.decision_actions[i]
        assert quality == out.results[i].quality
        assert spent == out.results[i].cost_usd


def test_decision_log_is_replayable():
    world, eps = world_and_episodes(6)
    cost_ref = load_cost_ref()
    model = tiny_model()
    session = _session(model, world, cost_ref)
    for k, ep in enumerate(eps):
        actions, _, _, _, passes, logs = drive(session, ep, k % len(LAMS), mode="sample", seed=k)
        assert len(logs) == passes == len(actions)
        for log, a in zip(logs, actions):
            assert isinstance(log, DecisionLog)
            assert log.phase == (CHOOSE if a["phase"] == "choose" else HALT)
            assert log.view is not None and log.view.phase == log.phase
            assert len(log.mask) == len(log.bonus) == log.pool_size == ep.n_models
            assert log.action == (a["model"] if a["model"] is not None else -1)
            assert np.isfinite(log.behaviour_logp) and log.behaviour_logp <= 0.0
            assert 0.0 <= log.complexity <= 1.0
            assert log.lam_idx == k % len(LAMS) and log.reward == 0.0
            if log.phase == CHOOSE:
                assert log.mask[log.action] and log.has_choice
            elif log.action >= 0:
                assert log.mask[log.action]
    assert not model.training  # reset puts the model in eval mode


def test_bonus_and_availability_steer_choose():
    world, eps = world_and_episodes(1)
    cost_ref = load_cost_ref()
    model = tiny_model().eval()
    ep = eps[0]
    sk = ep.skeleton
    avail = np.array([True, False, True, True])
    session = _session(model, world, cost_ref, available=avail)
    session.reset(sk.episode_id, sk.prompt_ids[0], None, sk.budget_usd, ep.n_steps, 0, "serve")
    bonus = np.array([0.0, 0.0, 0.0, 1e3])
    strat = session.decide("s0", sk.prompt_ids[0], None, int(ep.table.step_type[ep.row(ep.initial_state())]), bonus)
    assert strat.model_id == IDS[3]
    log = session.finish()[0]
    assert log.mask == [True, False, True, True] and log.bonus == bonus.tolist() and log.action == 3


def test_strategy_is_contract_valid():
    world, eps = world_and_episodes()
    cost_ref = load_cost_ref()
    cards = _pool_cards(world)
    session = _session(tiny_model(), world, cost_ref)
    rounds = set()
    for k, ep in enumerate(eps):
        for mode in ("serve", "sample"):
            _, _, _, strategies, _, _ = drive(session, ep, k % len(LAMS), mode=mode, seed=k)
            for st in strategies:
                validate_strategy(st, cards)
                rounds.add(st.round)
    assert rounds == {1, 2}


def test_after_round_rejects_a_round_from_another_model():
    import dataclasses

    import numpy as np
    import pytest

    from controller.embedding_cache import EmbeddingCache
    from controller.session import ControllerSession
    from measurement.online_env import SimStepEnv
    from measurement.rl_reward import LAMS, load_cost_ref
    from tests.test_rollout import tiny_model, world_and_episodes

    world, eps = world_and_episodes(2)
    model = tiny_model("full")
    avail = np.ones(4, dtype=bool)
    s = ControllerSession(model, lambda: world.card_costs, load_cost_ref(),
                          EmbeddingCache(model, dict(world.emb_index), embed_fn=None),
                          lambda: list(world.model_ids), lambda: avail, lams=LAMS)
    env = SimStepEnv(eps[0])
    task_key, task_text, budget, n = env.start()
    s.reset("e", task_key, task_text, budget, n, 0, "serve")
    spec = env.current_step()
    strategy = s.decide(spec.step_id, spec.step_key, spec.step_text, spec.step_type, np.zeros(4))
    out = env.play(strategy.model_id, 1)
    other = next(m for m in world.model_ids if m != strategy.model_id)
    with pytest.raises(ValueError):
        s.after_round(dataclasses.replace(out.trace_item, model_id=other), np.zeros(4))
