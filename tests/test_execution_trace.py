import pytest

from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
    validate_execution_step,
    validate_execution_trace,
)


def make_valid_trace() -> ExecutionTrace:
    return ExecutionTrace(
        execution_id="exec-001",
        strategy_step_id="step-001",
        strategy_round=1,
        framework="test-framework",
        catalogue_version="v1",
        adapter_version="v1",
        steps=(
            ExecutionStep(
                step_index=0,
                parent_step=None,
                model_id="model-a",
                tool=None,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
                wall_ms=200.0,
                status="ok",
            ),
            ExecutionStep(
                step_index=1,
                parent_step=0,
                model_id="model-b",
                tool="search",
                input_tokens=50,
                output_tokens=25,
                cost_usd=0.002,
                wall_ms=300.0,
                status="ok",
            ),
        ),
        final_status="ok",
    )


def test_valid_execution_trace_passes():
    trace = make_valid_trace()

    validate_execution_trace(trace)


def test_parent_step_must_reference_earlier_step():
    step = ExecutionStep(
        step_index=1,
        parent_step=1,
        model_id="model-a",
        tool=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        wall_ms=100.0,
        status="ok",
    )

    with pytest.raises(
        ValueError,
        match="parent_step must refer to an earlier execution step",
    ):
        validate_execution_step(step)


def test_negative_cost_is_rejected():
    step = ExecutionStep(
        step_index=0,
        parent_step=None,
        model_id="model-a",
        tool=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=-0.001,
        wall_ms=100.0,
        status="ok",
    )

    with pytest.raises(
        ValueError,
        match="cost_usd cannot be negative",
    ):
        validate_execution_step(step)


def test_ok_step_cannot_have_error_type():
    step = ExecutionStep(
        step_index=0,
        parent_step=None,
        model_id="model-a",
        tool=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        wall_ms=100.0,
        status="ok",
        error_type="SomeError",
    )

    with pytest.raises(
        ValueError,
        match="error_type must be None",
    ):
        validate_execution_step(step)


def test_trace_requires_consecutive_step_indexes():
    trace = ExecutionTrace(
        execution_id="exec-002",
        strategy_step_id="step-002",
        strategy_round=1,
        framework="test-framework",
        catalogue_version="v1",
        adapter_version="v1",
        steps=(
            ExecutionStep(
                step_index=1,
                parent_step=None,
                model_id="model-a",
                tool=None,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
                wall_ms=100.0,
                status="ok",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="consecutive",
    ):
        validate_execution_trace(trace)