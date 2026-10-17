
"""Tests for multi-parent execution dependencies."""

import pytest

from execution.trace import (
    ExecutionStep,
    validate_execution_step,
)


def make_step(
    index: int = 2,
    parent_step: int | None = None,
    parent_steps: tuple[int, ...] | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        step_index=index,
        parent_step=parent_step,
        model_id="model-a",
        tool=None,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        wall_ms=10.0,
        status="ok",
        node_name="test-node",
        parent_steps=parent_steps,
    )


def test_node_can_have_two_parents():
    step = make_step(
        parent_steps=(0, 1)
    )

    validate_execution_step(step)

    assert step.dependencies == (0, 1)


def test_legacy_single_parent_still_works():
    step = make_step(
        parent_step=0
    )

    validate_execution_step(step)

    assert step.dependencies == (0,)


def test_root_node_has_no_parents():
    step = make_step(
        index=0,
        parent_steps=()
    )

    validate_execution_step(step)

    assert step.dependencies == ()


def test_duplicate_parents_rejected():
    step = make_step(
        parent_steps=(0, 0)
    )

    with pytest.raises(
        ValueError,
        match="duplicate parents",
    ):
        validate_execution_step(step)


def test_future_parent_rejected():
    step = make_step(
        parent_steps=(0, 2)
    )

    with pytest.raises(
        ValueError,
        match="earlier execution step",
    ):
        validate_execution_step(step)


def test_conflicting_parent_formats_rejected():
    step = make_step(
        parent_step=0,
        parent_steps=(0, 1),
    )

    with pytest.raises(
        ValueError,
        match="conflicts",
    ):
        validate_execution_step(step)
