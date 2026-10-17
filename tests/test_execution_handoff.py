
"""Tests for Pillar 4 measurement handoff."""

from dataclasses import replace

import pytest

from execution.adapters.base import AdapterResult
from execution.handoff import build_measurement_handoff
from execution.trace import ExecutionStep, ExecutionTrace


def make_result(
    input_tokens=0,
    output_tokens=0,
    cost_usd=0.0,
):
    step = ExecutionStep(
        step_index=0,
        parent_step=None,
        parent_steps=(),
        model_id="local-model",
        tool=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        wall_ms=10.0,
        status="ok",
        node_name="test-node",
    )

    trace = ExecutionTrace(
        execution_id="handoff-exec-001",
        strategy_step_id="handoff-step-001",
        strategy_round=1,
        framework="crewai",
        catalogue_version="v1",
        adapter_version="v1",
        steps=(step,),
        final_status="ok",
    )

    return AdapterResult(
        output_text="HELLO RESEARCH!",
        trace=trace,
    )


def build(result, token_status, cost_status):
    return build_measurement_handoff(
        result,
        token_accounting=token_status,
        cost_accounting=cost_status,
        latency_basis="method_event_observed",
    )


def test_local_flow_reports_zero_usage():
    handoff = build(
        make_result(),
        "not_applicable",
        "not_applicable",
    )

    assert handoff.input_tokens == 0
    assert handoff.output_tokens == 0
    assert handoff.total_cost_usd == 0.0


def test_unavailable_usage_is_not_zero():
    handoff = build(
        make_result(),
        "unavailable",
        "unavailable",
    )

    assert handoff.input_tokens is None
    assert handoff.output_tokens is None
    assert handoff.total_cost_usd is None


def test_measured_usage_is_preserved():
    handoff = build(
        make_result(
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.002,
        ),
        "measured",
        "measured",
    )

    assert handoff.input_tokens == 100
    assert handoff.output_tokens == 50
    assert handoff.total_cost_usd == 0.002


def test_nonzero_tokens_cannot_be_not_applicable():
    with pytest.raises(
        ValueError,
        match="Nonzero tokens",
    ):
        build(
            make_result(input_tokens=10),
            "not_applicable",
            "unavailable",
        )


def test_nonzero_cost_cannot_be_not_applicable():
    with pytest.raises(
        ValueError,
        match="Nonzero cost",
    ):
        build(
            make_result(cost_usd=0.01),
            "unavailable",
            "not_applicable",
        )


def test_invalid_accounting_status_rejected():
    with pytest.raises(
        ValueError,
        match="Invalid token accounting status",
    ):
        build(
            make_result(),
            "unknown",
            "unavailable",
        )


def test_invalid_latency_basis_rejected():
    with pytest.raises(
        ValueError,
        match="Invalid latency basis",
    ):
        build_measurement_handoff(
            make_result(),
            token_accounting="not_applicable",
            cost_accounting="not_applicable",
            latency_basis="unknown",
        )


def test_invalid_trace_rejected():
    result = make_result()

    empty_trace = replace(
        result.trace,
        steps=(),
    )

    invalid_result = replace(
        result,
        trace=empty_trace,
    )

    with pytest.raises(ValueError):
        build(
            invalid_result,
            "not_applicable",
            "not_applicable",
        )


def test_handoff_preserves_execution_identity():
    handoff = build(
        make_result(),
        "not_applicable",
        "not_applicable",
    )

    assert handoff.execution_id == "handoff-exec-001"
    assert handoff.framework == "crewai"
    assert handoff.final_status == "ok"
    assert len(handoff.trace.steps) == 1
