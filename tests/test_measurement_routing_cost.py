
"""C3-11: Tests for simulator routing-cost accounting.

Verify routing expenses are:
- Deducted from the available budget.
- Included in cumulative episode spending.
- Kept separate from execution trace items.
- Applied before model execution.
- Rejected when invalid or unaffordable.

These tests do not make live API calls.
"""

from __future__ import annotations

import math

import pytest

from controller.contracts import Strategy
from measurement.simulator import Simulator

from tests.test_measurement_baselines import (
    make_environment,
)


@pytest.fixture
def simulator():
    episode, _ = make_environment()

    return Simulator(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        flip_p=0.0,
    )


def test_routing_cost_decreases_budget(simulator):
    before = simulator.current_task()

    simulator.charge_routing_cost(0.75)

    after = simulator.current_task()

    assert after.budget_left_usd == pytest.approx(
        before.budget_left_usd - 0.75
    )


def test_routing_cost_increases_total_spending(simulator):
    simulator.charge_routing_cost(0.75)

    assert simulator._state.spent == pytest.approx(
        0.75
    )

    assert simulator._state.budget_left == pytest.approx(
        99.25
    )


def test_multiple_routing_costs_accumulate(simulator):
    simulator.charge_routing_cost(0.10)
    simulator.charge_routing_cost(0.20)
    simulator.charge_routing_cost(0.30)

    assert simulator._state.spent == pytest.approx(
        0.60
    )

    assert simulator._state.budget_left == pytest.approx(
        99.40
    )


def test_zero_routing_cost_is_allowed(simulator):
    original = simulator._state

    simulator.charge_routing_cost(0.0)

    assert simulator._state.spent == original.spent
    assert (
        simulator._state.budget_left
        == original.budget_left
    )


def test_original_frozen_state_is_preserved(simulator):
    original = simulator._state

    simulator.charge_routing_cost(0.25)

    assert original.spent == 0.0
    assert original.budget_left == 100.0

    assert simulator._state is not original


def test_routing_cost_does_not_change_execution_position(
    simulator,
):
    before = simulator._state

    simulator.charge_routing_cost(0.25)

    after = simulator._state

    assert after.step_index == before.step_index
    assert after.slot == before.slot
    assert after.rounds == before.rounds
    assert after.done == before.done
    assert after.n_correct == before.n_correct


def test_routing_charge_does_not_add_execution_trace(
    simulator,
):
    original_trace = simulator._state.trace

    simulator.charge_routing_cost(0.25)

    assert simulator._state.trace == original_trace
    assert len(simulator._state.trace) == 0


def test_routing_cost_is_charged_before_execution(
    simulator,
):
    simulator.charge_routing_cost(0.50)

    task = simulator.current_task()

    strategy = Strategy(
        step_id=task.step_id,
        round=task.round,
        model_id="m0",
        halt="stop",
    )

    transition = simulator.apply(strategy)

    assert transition.trace_item is not None

    # The mock environment charges $1 for m0.
    execution_cost = transition.trace_item.cost_usd

    assert execution_cost == pytest.approx(1.0)

    # Episode spending includes both charges.
    assert simulator._state.spent == pytest.approx(
        0.50 + execution_cost
    )

    assert simulator._state.budget_left == pytest.approx(
        100.0 - 0.50 - execution_cost
    )


@pytest.mark.parametrize(
    "invalid_cost",
    [
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        None,
        "invalid",
    ],
)
def test_invalid_routing_cost_is_rejected(
    simulator,
    invalid_cost,
):
    original = simulator._state

    with pytest.raises(
        ValueError,
        match="Routing cost",
    ):
        simulator.charge_routing_cost(
            invalid_cost
        )

    assert simulator._state is original


def test_unaffordable_routing_cost_is_rejected(simulator):
    original = simulator._state

    with pytest.raises(
        ValueError,
        match="exceeds remaining",
    ):
        simulator.charge_routing_cost(100.01)

    # The method must never silently clip the charge.
    assert simulator._state is original


def test_exact_remaining_budget_can_be_charged(
    simulator,
):
    simulator.charge_routing_cost(100.0)

    assert simulator._state.spent == pytest.approx(
        100.0
    )

    assert simulator._state.budget_left == pytest.approx(
        0.0
    )


def test_no_execution_when_routing_uses_entire_budget(
    simulator,
):
    simulator.charge_routing_cost(100.0)

    task = simulator.current_task()

    strategy = Strategy(
        step_id=task.step_id,
        round=task.round,
        model_id="m0",
        halt="stop",
    )

    transition = simulator.apply(strategy)

    assert transition.done is True
    assert transition.status == "budget_exhausted"
    assert transition.trace_item is None

    metrics = simulator.final_metrics()

    assert metrics["cost_usd"] == pytest.approx(
        100.0
    )

    assert metrics["budget_left_usd"] == pytest.approx(
        0.0
    )


def test_cannot_charge_completed_episode(simulator):
    simulator.charge_routing_cost(100.0)

    task = simulator.current_task()

    strategy = Strategy(
        step_id=task.step_id,
        round=task.round,
        model_id="m0",
        halt="stop",
    )

    simulator.apply(strategy)

    assert simulator._state.done is True

    with pytest.raises(
        ValueError,
        match="after episode completion",
    ):
        simulator.charge_routing_cost(0.01)


def test_routing_cost_preserves_budget_invariant(
    simulator,
):
    charges = [0.001, 0.002, 0.003, 0.004]

    for charge in charges:
        simulator.charge_routing_cost(charge)

    total = (
        simulator._state.spent
        + simulator._state.budget_left
    )

    assert math.isclose(
        total,
        100.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    )
