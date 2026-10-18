
"""Tests for Pillar 3 C3-05: RouterBench-backed Simulator."""

import math

import numpy as np
import pytest

from controller.contracts import Strategy, TaskRep, TraceItem
from data.episodes import Skeleton
from data.routerbench import OutcomeTable
from measurement.simulator import Simulator


MODELS = ["m0", "m1", "m2", "m3"]

DEFAULT_ROWS = (
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 1, 1),
)


def make_sim(
    rows=DEFAULT_ROWS,
    ep_type="random",
    budget=100.0,
    flip_p=0.0,
    seed=1234,
    lam=None,
    trace_window=None,
    model_ids=None,
):
    """Build a small simulator with known model outcomes."""

    correct = np.asarray(rows, dtype=bool)

    costs = np.tile(
        [1.0, 2.0, 4.0, 8.0],
        (len(rows), 1),
    )

    prompt_ids = np.asarray(
        [f"p{i}" for i in range(len(rows))]
    )

    step_types = np.zeros(len(rows), dtype=int)

    table = OutcomeTable(
        prompt_ids=prompt_ids,
        step_type=step_types,
        correct=correct,
        cost=costs,
    )

    skeleton = Skeleton(
        episode_id="test-episode",
        ep_type=ep_type,
        prompt_ids=tuple(prompt_ids),
        budget_usd=budget,
    )

    return Simulator(
        skeleton=skeleton,
        table=table,
        model_ids=MODELS if model_ids is None else model_ids,
        flip_p=flip_p,
        seed=seed,
        lam=lam,
        trace_window=trace_window,
    )


def play(sim, model="m0", halt="stop"):
    """Submit a Controller Strategy for the current task."""

    task = sim.current_task()

    assert task is not None

    strategy = Strategy(
        step_id=task.step_id,
        round=task.round,
        model_id=model,
        halt=halt,
    )

    return sim.apply(strategy)


def test_initial_task_uses_controller_contract():
    sim = make_sim(lam=0.5)

    task = sim.current_task()

    assert isinstance(task, TaskRep)
    assert task.episode_id == "test-episode"
    assert task.step_id == "test-episode:step:0"
    assert task.step_index == 0
    assert task.round == 0
    assert task.step_emb_key == "p0"
    assert task.expected_steps == 3
    assert task.budget_total_usd == 100.0
    assert task.budget_left_usd == 100.0
    assert task.spent_usd == 0.0
    assert task.lam == 0.5
    assert task.trace == []


def test_one_round_execution_returns_trace():
    sim = make_sim()

    result = play(sim, "m1", "stop")

    assert result.status == "running"
    assert result.done is False
    assert isinstance(result.trace_item, TraceItem)
    assert result.trace_item.model_id == "m1"
    assert result.trace_item.round == 1
    assert result.trace_item.verifier_ok is True
    assert result.trace_item.cost_usd == 2.0

    assert result.task is not None
    assert result.task.step_index == 1
    assert result.task.step_emb_key == "p1"


def test_episode_completion_and_final_metrics():
    sim = make_sim()

    play(sim, "m1")
    play(sim, "m0")
    result = play(sim, "m2")

    assert result.status == "completed"
    assert result.done is True
    assert result.task is None

    metrics = sim.final_metrics()

    assert metrics["n_correct"] == 3
    assert metrics["expected_steps"] == 3
    assert metrics["quality"] == pytest.approx(1.0)
    assert metrics["cost_usd"] == pytest.approx(7.0)
    assert metrics["budget_left_usd"] == pytest.approx(93.0)


def test_final_metrics_require_completion():
    sim = make_sim()

    with pytest.raises(ValueError, match="completed episode"):
        sim.final_metrics()


def test_counterfactual_returns_all_model_outcomes():
    sim = make_sim()

    outcomes = sim.counterfactual()

    assert outcomes == {
        "m0": False,
        "m1": True,
        "m2": True,
        "m3": True,
    }


def test_counterfactual_is_not_a_controller_observation():
    sim = make_sim()

    task = sim.current_task()

    assert task is not None
    assert not hasattr(task, "correct")
    assert not hasattr(task, "counterfactual")
    assert not hasattr(task, "ground_truth")


def test_counterfactual_reflects_propagation_history():
    sim = make_sim(
        ep_type="propagation",
        rows=(
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        ),
    )

    play(sim, "m0")

    assert sim.counterfactual() == {
        "m0": False,
        "m1": True,
        "m2": True,
        "m3": True,
    }

    play(sim, "m1")

    assert sim.counterfactual() == {
        "m0": True,
        "m1": True,
        "m2": True,
        "m3": True,
    }


def test_counterfactual_reports_correctness_not_affordability():
    sim = make_sim(budget=1.5)

    outcomes = sim.counterfactual()

    assert outcomes["m3"] is True
    assert "m3" not in sim.affordable_models()


def test_counterfactual_unavailable_after_completion():
    sim = make_sim()

    play(sim, "m1")
    play(sim, "m0")
    play(sim, "m2")

    with pytest.raises(ValueError, match="completion"):
        sim.counterfactual()


def test_escalation_runs_a_second_round():
    sim = make_sim()

    first = play(sim, "m0", "escalate")

    assert first.status == "awaiting_escalation"
    assert first.done is False
    assert first.trace_item.verifier_ok is False

    assert first.task is not None
    assert first.task.step_index == 0
    assert first.task.round == 1
    assert first.task.step_id == "test-episode:step:0"

    second = play(sim, "m2", "stop")

    assert second.status == "running"
    assert second.trace_item.model_id == "m2"
    assert second.trace_item.round == 2
    assert second.trace_item.escalated is True
    assert second.trace_item.verifier_ok is True

    assert second.task.step_index == 1
    assert second.task.spent_usd == pytest.approx(5.0)


def test_escalation_requires_stronger_model():
    sim = make_sim()

    play(sim, "m1", "escalate")

    with pytest.raises(
        ValueError,
        match="strictly more expensive",
    ):
        play(sim, "m0", "stop")


def test_second_round_cannot_escalate_again():
    sim = make_sim()

    play(sim, "m0", "escalate")

    with pytest.raises(
        ValueError,
        match="Second round must stop",
    ):
        play(sim, "m2", "escalate")

    assert sim.current_task().round == 1


def test_escalation_requires_affordable_stronger_model():
    sim = make_sim(budget=1.5)

    with pytest.raises(
        ValueError,
        match="No affordable stronger model",
    ):
        play(sim, "m0", "escalate")

    # A rejected Strategy must not change the saved state.
    assert sim.current_task().round == 0
    assert sim.current_task().spent_usd == 0.0
    assert sim.current_task().trace == []


def test_affordable_models_follow_remaining_budget():
    sim = make_sim(budget=6.0)

    assert sim.affordable_models() == [
        "m0", "m1", "m2"
    ]

    play(sim, "m0", "escalate")

    assert sim.affordable_models() == ["m1", "m2"]


def test_unaffordable_first_round_exhausts_budget():
    sim = make_sim(budget=0.5)

    result = play(sim, "m3", "stop")

    assert result.status == "budget_exhausted"
    assert result.done is True
    assert result.task is None
    assert result.trace_item is None

    metrics = sim.final_metrics()

    assert metrics["n_correct"] == 0
    assert metrics["quality"] == 0.0
    assert metrics["cost_usd"] == 0.0
    assert metrics["budget_left_usd"] == 0.5


def test_budget_exhaustion_after_one_completed_step():
    sim = make_sim(
        budget=1.5,
        rows=(
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        ),
    )

    play(sim, "m0")

    result = play(sim, "m0")

    assert result.status == "budget_exhausted"
    assert result.trace_item is None

    metrics = sim.final_metrics()

    assert metrics["n_correct"] == 1
    assert metrics["quality"] == pytest.approx(1 / 3)
    assert metrics["cost_usd"] == 1.0


def test_unaffordable_escalation_is_not_charged():
    sim = make_sim(budget=3.0)

    first = play(sim, "m0", "escalate")

    assert first.status == "awaiting_escalation"

    second = play(sim, "m3", "stop")

    assert second.status == "budget_exhausted"
    assert second.trace_item is None
    assert second.done is True

    metrics = sim.final_metrics()

    assert metrics["cost_usd"] == 1.0
    assert metrics["n_correct"] == 0
    assert metrics["budget_left_usd"] == 2.0


def test_retry_is_decided_by_noisy_verifier():
    sim = make_sim(
        ep_type="retry",
        rows=((1, 1, 1, 1),) * 3,
        flip_p=1.0,
    )

    first = play(sim, "m0")

    assert first.trace_item.verifier_ok is False
    assert first.task.step_emb_key == "p0"
    assert first.task.step_index == 1

    second = play(sim, "m0")

    # A prompt is retried at most once.
    assert second.task.step_emb_key == "p1"
    assert second.task.step_index == 2


def test_verifier_can_accept_an_incorrect_result():
    sim = make_sim(
        ep_type="retry",
        rows=(
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        ),
        flip_p=1.0,
    )

    result = play(sim, "m0")

    # Actual answer is wrong; the noisy verifier says OK.
    assert result.trace_item.verifier_ok is True

    # There must be no retry when verifier_ok is True.
    assert result.task.step_emb_key == "p1"


def test_trace_window_limits_controller_history():
    sim = make_sim(trace_window=1)

    play(sim, "m1")
    second = play(sim, "m0")

    assert len(second.task.trace) == 1
    assert second.task.trace[0].step_index == 1


def test_reset_clears_episode_progress():
    sim = make_sim()

    play(sim, "m1")

    task = sim.reset()

    assert task.step_index == 0
    assert task.round == 0
    assert task.spent_usd == 0.0
    assert task.budget_left_usd == 100.0
    assert task.trace == []


@pytest.mark.parametrize(
    "step_id,round_number,model_id,halt,error",
    [
        ("wrong-step", 0, "m0", "stop", "step ID"),
        (
            "test-episode:step:0",
            1,
            "m0",
            "stop",
            "round mismatch",
        ),
        (
            "test-episode:step:0",
            0,
            "unknown",
            "stop",
            "Unknown model",
        ),
        (
            "test-episode:step:0",
            0,
            "m0",
            "invalid",
            "halt decision",
        ),
    ],
)
def test_invalid_strategy_is_rejected_without_changes(
    step_id,
    round_number,
    model_id,
    halt,
    error,
):
    sim = make_sim()

    strategy = Strategy(
        step_id=step_id,
        round=round_number,
        model_id=model_id,
        halt=halt,
    )

    with pytest.raises(ValueError, match=error):
        sim.apply(strategy)

    task = sim.current_task()

    assert task.step_index == 0
    assert task.round == 0
    assert task.spent_usd == 0.0


def test_cannot_execute_after_completion():
    sim = make_sim()

    play(sim, "m1")
    play(sim, "m0")
    play(sim, "m2")

    strategy = Strategy(
        step_id="test-episode:step:2",
        round=0,
        model_id="m0",
        halt="stop",
    )

    with pytest.raises(ValueError, match="already complete"):
        sim.apply(strategy)

    assert sim.current_task() is None
    assert sim.affordable_models() == []


def test_invalid_model_count_is_rejected():
    with pytest.raises(ValueError, match="count"):
        make_sim(model_ids=["m0", "m1"])


def test_duplicate_model_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        make_sim(
            model_ids=["m0", "m0", "m2", "m3"]
        )


def test_empty_model_id_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        make_sim(
            model_ids=["m0", "", "m2", "m3"]
        )


@pytest.mark.parametrize(
    "invalid_budget",
    [-1.0, float("inf"), float("nan")],
)
def test_invalid_budget_is_rejected(invalid_budget):
    with pytest.raises(ValueError):
        make_sim(budget=invalid_budget)


@pytest.mark.parametrize(
    "invalid_probability",
    [-0.1, 1.1, float("nan")],
)
def test_invalid_verifier_probability_is_rejected(
    invalid_probability,
):
    with pytest.raises(ValueError):
        make_sim(flip_p=invalid_probability)


def test_invalid_lambda_is_rejected():
    with pytest.raises(ValueError, match="Lambda"):
        make_sim(lam=math.inf)


@pytest.mark.parametrize("invalid_window", [0, -1, True])
def test_invalid_trace_window_is_rejected(invalid_window):
    with pytest.raises(ValueError, match="Trace window"):
        make_sim(trace_window=invalid_window)


def test_verifier_is_deterministic_for_same_seed():
    first = make_sim(flip_p=0.15, seed=77)
    second = make_sim(flip_p=0.15, seed=77)

    first_result = play(first, "m1")
    second_result = play(second, "m1")

    assert (
        first_result.trace_item.verifier_ok
        == second_result.trace_item.verifier_ok
    )
