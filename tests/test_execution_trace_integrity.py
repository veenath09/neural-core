
"""Reliability tests for Pillar 4 execution trace validation."""

from dataclasses import replace

import pytest

from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
    validate_execution_trace,
)


def make_valid_trace():
    """Create a valid successful execution trace."""

    step = ExecutionStep(
        step_index=0,
        parent_step=None,
        parent_steps=(),
        model_id="local-model",
        tool=None,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        wall_ms=10.0,
        status="ok",
        node_name="test-node",
    )

    return ExecutionTrace(
        execution_id="exec-integrity-001",
        strategy_step_id="step-integrity-001",
        strategy_round=1,
        framework="crewai",
        catalogue_version="v1",
        adapter_version="v1",
        steps=(step,),
        final_status="ok",
    )


def test_valid_trace_accepted():
    trace = make_valid_trace()

    validate_execution_trace(trace)


def test_successful_trace_cannot_be_empty():
    trace = replace(
        make_valid_trace(),
        steps=(),
    )

    with pytest.raises(ValueError):
        validate_execution_trace(trace)


def test_successful_trace_cannot_contain_failed_step():
    trace = make_valid_trace()

    failed_step = replace(
        trace.steps[0],
        status="error",
        error_type="RuntimeError",
    )

    invalid_trace = replace(
        trace,
        steps=(failed_step,),
        final_status="ok",
    )

    with pytest.raises(ValueError):
        validate_execution_trace(invalid_trace)


def test_failed_trace_must_record_failure():
    trace = replace(
        make_valid_trace(),
        final_status="error",
    )

    with pytest.raises(ValueError):
        validate_execution_trace(trace)


def test_invalid_final_status_rejected():
    trace = replace(
        make_valid_trace(),
        final_status="unknown",
    )

    with pytest.raises(ValueError):
        validate_execution_trace(trace)


def test_nan_latency_rejected():
    trace = make_valid_trace()

    invalid_step = replace(
        trace.steps[0],
        wall_ms=float("nan"),
    )

    invalid_trace = replace(
        trace,
        steps=(invalid_step,),
    )

    with pytest.raises(ValueError):
        validate_execution_trace(invalid_trace)


def test_infinite_cost_rejected():
    trace = make_valid_trace()

    invalid_step = replace(
        trace.steps[0],
        cost_usd=float("inf"),
    )

    invalid_trace = replace(
        trace,
        steps=(invalid_step,),
    )

    with pytest.raises(ValueError):
        validate_execution_trace(invalid_trace)
