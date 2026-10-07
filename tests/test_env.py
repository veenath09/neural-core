import numpy as np
import pytest

from data.env import Episode
from data.episodes import Skeleton
from data.routerbench import OutcomeTable

IDS = ["m0", "m1", "m2", "m3"]


def tiny_table(correct_rows):
    correct = np.array(correct_rows, dtype=bool)
    n = len(correct)
    cost = np.tile([1.0, 2.0, 4.0, 8.0], (n, 1))
    return OutcomeTable(np.array([f"p{i}" for i in range(n)]), np.zeros(n, int), correct, cost)


def episode(ep_type, correct_rows, budget=100.0, flip_p=0.0):
    t = tiny_table(correct_rows)
    sk = Skeleton("e0", ep_type, tuple(t.prompt_ids), budget)
    return Episode(sk, t, IDS, flip_p=flip_p)


def run(ep, models):
    """Play one round per step with the given models; return final state."""
    s = ep.initial_state()
    for m in models:
        s = ep.end_step(ep.play(s, m))
    return s


def test_quality_and_cost_accumulate():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    s = run(ep, [0, 0, 3])
    assert s.done and s.n_correct == 2 and ep.quality(s) == pytest.approx(2 / 3)
    assert s.spent == 1 + 1 + 8 and s.budget_left == 100 - 10


def test_propagation_requires_stronger_model_after_failure():
    ep = episode("propagation", [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))  # tier 0 fails step 0
    assert list(ep.succeeds(s)) == [False, True, True, True]
    s = ep.end_step(ep.play(s, 1))  # tier 1 succeeds -> no propagation
    assert list(ep.succeeds(s)) == [True, True, True, True]


def test_propagation_does_not_apply_to_other_types():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))
    assert ep.succeeds(s).all()


def test_retry_repeats_failed_item_once():
    ep = episode("retry", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))  # fail item 0
    assert (s.slot, s.retried) == (0, True)
    s = ep.end_step(ep.play(s, 3))  # fail again: not retried a second time
    assert (s.slot, s.retried) == (1, False)
    s = ep.end_step(ep.play(s, 0))
    assert s.done and s.step_index == 3 and s.n_correct == 1


def test_successful_step_is_not_retried():
    ep = episode("retry", [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))
    assert (s.slot, s.retried) == (1, False)


def test_escalation_must_be_strictly_more_expensive_and_replaces_answer():
    ep = episode("random", [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.play(ep.initial_state(), 1)
    assert ep.escalation_options(s) == [2, 3]
    with pytest.raises(ValueError, match="strictly more expensive"):
        ep.play(s, 1)
    s = ep.play(s, 2)
    assert s.cur_correct and s.spent == 2 + 4 and ep.escalation_options(s) == []
    with pytest.raises(ValueError, match="no rounds left"):
        ep.play(s, 3)
    assert ep.end_step(s).n_correct == 1


def test_unaffordable_round_ends_episode_with_step_scored_zero():
    ep = episode("random", [[1, 1, 1, 1]] * 3, budget=10.0)
    s = ep.end_step(ep.play(ep.initial_state(), 3))  # spend 8, budget left 2
    s = ep.play(s, 3)  # costs 8 > 2: not run
    assert s.done and s.n_correct == 1 and s.spent == 8 and ep.quality(s) == pytest.approx(1 / 3)


def test_legal_actions():
    ep = episode("random", [[0, 0, 0, 0]] * 3, budget=6.0)
    s0 = ep.initial_state()
    assert ep.legal_actions(s0) == [("play", m) for m in range(4)]
    s1 = ep.play(s0, 0)
    assert ep.legal_actions(s1) == [("stop", None), ("play", 1), ("play", 2), ("play", 3)]
    assert ep.escalation_options(s1) == [1, 2]  # 8 > 5 left: not affordable


def test_verifier_flips_at_configured_rate_and_is_deterministic():
    rows = [[1, 1, 1, 1]] * 6
    flips = []
    for i in range(400):
        t = tiny_table(rows)
        sk = Skeleton(f"e{i}", "random", tuple(t.prompt_ids), 1e9)
        ep = Episode(sk, t, IDS, flip_p=0.15, seed=0)
        s = run(ep, [0] * 6)
        flips += [not item.verifier_ok for item in s.trace]
        assert run(ep, [0] * 6).trace == s.trace
    assert np.mean(flips) == pytest.approx(0.15, abs=0.02)


def test_trace_items_record_rounds():
    ep = episode("random", [[0, 1, 1, 1]] * 3)
    s = ep.play(ep.play(ep.initial_state(), 0), 2)
    assert [(i.model_id, i.round, i.escalated, i.cost_usd) for i in s.trace] == [
        ("m0", 1, False, 1.0), ("m2", 2, True, 4.0)]
    assert [i.verifier_ok for i in s.trace] == [False, True]  # flip_p = 0
