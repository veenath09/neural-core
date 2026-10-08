from dataclasses import replace

import torch

from controller.features import BUDGET_FEATS, CHOOSE, HALT, TRACE_FEATS, collate, collate_targets, target, view
from tests.test_env import IDS, episode

EMB = {f"p{i}": i for i in range(10)}
MIDX = {m: i for i, m in enumerate(IDS)}
ROWS = [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]


def test_first_view_has_empty_trace_and_nothing_to_write():
    ep = episode("random", ROWS)
    v = view(ep, ep.initial_state(), CHOOSE, 0, EMB, MIDX)
    assert v.trace_present == [False, False] and not v.write and v.write_dt == 0.0
    assert (v.task_row, v.step_row, v.round, v.phase) == (0, 0, 1, CHOOSE)
    assert v.budget[:4] == [1.0, 0.0, 0.0, 1 / 3]


def test_post_round_view_writes_newest_round_of_current_step():
    ep = episode("random", ROWS)
    s = ep.play(ep.initial_state(), 0)
    v = view(ep, s, HALT, 0, EMB, MIDX)
    assert v.write and v.write_dt == 1 / 3
    assert v.trace_present == [True, False] and v.trace_earlier == [False, False]
    assert (v.trace_model[0], v.round, v.phase) == (0, 1, HALT)


def test_next_step_lists_earlier_rounds_newest_first():
    ep = episode("random", ROWS)
    s = ep.end_step(ep.play(ep.play(ep.initial_state(), 0), 2))
    v = view(ep, s, CHOOSE, 1, EMB, MIDX)
    assert v.trace_model == [2, 0] and v.trace_earlier == [True, True]
    assert v.write and v.step_row == 1 and v.trace_row == [0, 0]


def test_view_ignores_ground_truth():
    ep = episode("random", ROWS)
    s = ep.play(ep.initial_state(), 0)
    flipped = replace(s, cur_correct=not s.cur_correct, n_correct=7)
    assert view(ep, s, HALT, 0, EMB, MIDX) == view(ep, flipped, HALT, 0, EMB, MIDX)


def test_targets():
    ep = episode("random", [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], budget=5.0)
    s = ep.play(ep.initial_state(), 0)
    t = target(ep, s, HALT)
    assert (t.succeeds, t.halt, t.band) == ([0.0, 0.0, 1.0, 1.0], 1.0, 0.5)
    assert target(ep, ep.initial_state(), CHOOSE).halt == 0.0


def test_collate_shapes_and_types():
    ep = episode("random", ROWS)
    s = ep.play(ep.initial_state(), 0)
    b = collate([view(ep, ep.initial_state(), CHOOSE, 0, EMB, MIDX), view(ep, s, HALT, 0, EMB, MIDX)])
    assert b.trace_feats.shape == (2, 2, TRACE_FEATS) and b.budget.shape == (2, BUDGET_FEATS)
    assert b.write.dtype == torch.bool and b.task_row.dtype == torch.long and len(b) == 2
    succ, halt, band = collate_targets([target(ep, s, HALT)] * 3)
    assert succ.shape == (3, 4) and halt.shape == (3,) and band.shape == (3,)
