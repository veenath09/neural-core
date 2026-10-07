import json
import random
from dataclasses import asdict

import pytest

from controller.contracts import (
    K_TRACE,
    ModelCard,
    Strategy,
    TaskRep,
    TraceItem,
    strategy_from_dict,
    validate_strategy,
    validate_task_rep,
)
from controller.stub import stub_strategy


def make_cards():
    return [
        ModelCard(model_id=f"m{i}", provider="test", tier=i, mean_cost_usd=10.0 ** (i - 4),
                  price_source="test", price_date="2024-03", capability=[0.5, 0.5, i - 4.0])
        for i in range(4)
    ]


def make_rep(round_=1, trace=(), budget_left=0.5):
    return TaskRep(episode_id="e0", step_id="e0-s0", round=round_, task_emb_key="p0", step_type=0,
                   step_emb_key="p0", trace=list(trace), budget_total_usd=1.0, budget_left_usd=budget_left,
                   spent_usd=1.0 - budget_left, step_index=0, expected_steps=3)


def make_item(round_=1):
    return TraceItem(model_id="m0", round=round_, verifier_ok=False, cost_usd=0.1, latency_ms=0.0,
                     tokens_out=0, escalated=False, output_emb_key=None)


@pytest.mark.parametrize("round_", [1, 2])
def test_stub_returns_valid_strategy(round_):
    pool, rng = make_cards(), random.Random(0)
    for _ in range(200):
        s = stub_strategy(make_rep(round_), pool, rng)
        validate_strategy(s, pool)
        assert (s.step_id, s.round) == ("e0-s0", round_)


def test_stub_never_escalates_in_last_round():
    pool, rng = make_cards(), random.Random(0)
    assert {stub_strategy(make_rep(2), pool, rng).halt for _ in range(100)} == {"stop"}


@pytest.mark.parametrize("field,value", [
    ("model_id", "nope"), ("halt", "maybe"), ("memory", "disk"), ("depth", "max"),
    ("verification", "vote"), ("topology", "mesh"), ("complexity", 1.5), ("confidence", -0.1),
    ("round", 3), ("version", "v0"),
])
def test_validate_strategy_rejects_bad_field(field, value):
    s = Strategy(step_id="x", round=1, model_id="m0", halt="stop")
    setattr(s, field, value)
    with pytest.raises(ValueError, match=field):
        validate_strategy(s, make_cards())


def test_validate_strategy_rejects_escalate_in_last_round():
    s = Strategy(step_id="x", round=2, model_id="m0", halt="escalate")
    with pytest.raises(ValueError, match="halt"):
        validate_strategy(s, make_cards())


def test_strategy_round_trips_through_json():
    s = Strategy(step_id="x", round=1, model_id="m1", halt="escalate", tools=["search"],
                 complexity=0.25, confidence=0.9)
    assert strategy_from_dict(json.loads(json.dumps(asdict(s)))) == s


def test_validate_task_rep_limits_trace():
    validate_task_rep(make_rep(trace=[make_item(), make_item()]))
    with pytest.raises(ValueError, match="trace"):
        validate_task_rep(make_rep(trace=[make_item()] * (K_TRACE + 1)))


def test_validate_task_rep_rejects_budget_above_total():
    with pytest.raises(ValueError, match="budget"):
        validate_task_rep(make_rep(budget_left=2.0))
