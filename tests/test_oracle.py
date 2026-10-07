import time

import numpy as np
import pytest

from data.env import Episode
from data.episodes import Skeleton, generate_skeletons
from data.oracle import halt_label, plan, step_labels
from tests.conftest import make_table
from tests.test_env import IDS, episode


def brute_force(ep, s):
    """Best (n_correct, -spent) over every legal action sequence, no pruning."""
    if s.done:
        return s.n_correct, -s.spent
    return max(brute_force(ep, ep.apply(s, a)) for a in ep.legal_actions(s))


def test_step_labels_pick_cheapest_success_else_cheapest():
    ep = episode("random", [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    lab = step_labels(ep, ep.initial_state())
    assert list(lab.succeeds) == [False, False, True, True] and lab.oracle_model == 2 and lab.band == 2
    s = ep.end_step(ep.play(ep.initial_state(), 0))
    assert step_labels(ep, s).oracle_model == 0 and step_labels(ep, s).band == 4


def test_halt_label():
    ep = episode("random", [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], budget=5.0)
    s = ep.play(ep.initial_state(), 0)  # wrong; tier 2 succeeds, costs 4 <= 4 left
    assert halt_label(ep, s) == 1
    assert halt_label(ep, ep.play(ep.initial_state(), 2)) == 0  # already right
    tight = episode("random", [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], budget=4.0)
    assert halt_label(tight, tight.play(tight.initial_state(), 0)) == 0  # 4 > 3 left
    nobody = episode("random", [[0, 0, 0, 0]] * 3)
    assert halt_label(nobody, nobody.play(nobody.initial_state(), 0)) == 0


def test_plan_prefers_cheapest_correct_route():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    p = plan(ep, ep.initial_state())
    assert p.n_correct == 3 and p.spent == 2 + 1 + 8
    assert p.actions == (("play", 1), ("stop", None), ("play", 0), ("stop", None), ("play", 3), ("stop", None))


def test_plan_skips_hopeless_step_to_save_budget():
    # Budget 9: solving step 0 (tier 3, cost 8) leaves 1, so neither later step (tier 2, cost 4) is affordable.
    ep = episode("random", [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]], budget=9.0)
    p = plan(ep, ep.initial_state())
    assert p.n_correct == 2 and p.spent == 1 + 4 + 4
    assert p.actions[0] == ("play", 0)


@pytest.mark.parametrize("ep_type", ["correlated", "retry", "propagation", "random"])
def test_plan_matches_brute_force(ep_type):
    table = make_table(n=60, seed=1)
    sks = generate_skeletons(table, list(table.prompt_ids), n=200, seed=2, split="t")
    sks = [s for s in sks if s.ep_type == ep_type and len(s.prompt_ids) <= 4][:15]
    for sk in sks:
        ep = Episode(sk, table, IDS)
        p = plan(ep, ep.initial_state())
        n, neg_spent = brute_force(ep, ep.initial_state())
        assert p.n_correct == n and p.spent == pytest.approx(-neg_spent)
        s = ep.initial_state()
        for a in p.actions:
            s = ep.apply(s, a)
        assert s.done and s.n_correct == p.n_correct and s.spent == pytest.approx(p.spent)


def test_plan_from_mid_step_state():
    ep = episode("random", [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.play(ep.initial_state(), 0)
    p = plan(ep, s)
    assert p.actions[0] == ("play", 2) and p.n_correct == 3


def test_plan_is_fast_on_six_step_propagation():
    table = make_table(n=300, seed=3)
    sk = Skeleton("e", "propagation", tuple(table.prompt_ids[:6]), 1.0)
    ep = Episode(sk, table, IDS)
    start = time.perf_counter()
    plan(ep, ep.initial_state())
    assert time.perf_counter() - start < 0.5
