
"""Tests for Pillar 4 measurement handoff JSON persistence."""

import json
from dataclasses import replace

import pytest

from execution.adapters.base import AdapterResult

from execution.handoff import build_measurement_handoff

from execution.handoff_serialization import (
    handoff_to_dict,
    handoff_from_dict,
    save_handoff_json,
    load_handoff_json,
    validate_measurement_handoff,
)

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
        wall_ms=12.5,
        status="ok",
        node_name="uppercase",
    )

    trace = ExecutionTrace(
        execution_id="handoff-save-001",
        strategy_step_id="step-001",
        strategy_round=1,
        framework="crewai",
        catalogue_version="v1",
        adapter_version="v1",
        steps=(step,),
        final_status="ok",
    )

    return AdapterResult(
        output_text="HELLO!",
        trace=trace,
    )


def make_handoff(
    token_accounting="not_applicable",
    cost_accounting="not_applicable",
    **usage,
):
    return build_measurement_handoff(
        make_result(**usage),
        token_accounting=token_accounting,
        cost_accounting=cost_accounting,
        latency_basis="method_event_observed",
    )


def test_local_handoff_round_trip():
    original = make_handoff()

    restored = handoff_from_dict(
        handoff_to_dict(original)
    )

    assert restored == original
    assert restored.total_cost_usd == 0.0
    assert restored.token_accounting == "not_applicable"


def test_unavailable_usage_remains_null():
    original = make_handoff(
        token_accounting="unavailable",
        cost_accounting="unavailable",
    )

    data = handoff_to_dict(original)

    assert data["input_tokens"] is None
    assert data["output_tokens"] is None
    assert data["total_cost_usd"] is None

    restored = handoff_from_dict(data)

    assert restored == original


def test_measured_usage_is_preserved():
    original = make_handoff(
        token_accounting="measured",
        cost_accounting="measured",
        input_tokens=100,
        output_tokens=25,
        cost_usd=0.004,
    )

    restored = handoff_from_dict(
        handoff_to_dict(original)
    )

    assert restored.input_tokens == 100
    assert restored.output_tokens == 25
    assert restored.total_cost_usd == 0.004


def test_save_and_load_file(tmp_path):
    original = make_handoff()

    path = (
        tmp_path
        / "measurement"
        / "handoff.json"
    )

    save_handoff_json(original, path)

    assert path.exists()

    restored = load_handoff_json(path)

    assert restored == original
    assert restored.execution_id == "handoff-save-001"


def test_saved_json_is_readable(tmp_path):
    original = make_handoff()

    path = tmp_path / "handoff.json"

    save_handoff_json(original, path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert data["handoff_version"] == "v1"
    assert data["trace"]["framework"] == "crewai"
    assert len(data["trace"]["steps"]) == 1


def test_unsupported_handoff_version_rejected():
    data = handoff_to_dict(make_handoff())

    data["handoff_version"] = "v99"

    with pytest.raises(
        ValueError,
        match="Unsupported measurement handoff version",
    ):
        handoff_from_dict(data)


def test_tampered_cost_is_rejected():
    data = handoff_to_dict(make_handoff())

    data["total_cost_usd"] = 500.0

    with pytest.raises(
        ValueError,
        match="Saved total_cost_usd does not match",
    ):
        handoff_from_dict(data)


def test_invalid_trace_is_rejected():
    data = handoff_to_dict(make_handoff())

    # A step cannot depend on itself.
    data["trace"]["steps"][0]["parent_steps"] = [0]

    with pytest.raises(ValueError):
        handoff_from_dict(data)


def test_inconsistent_handoff_is_rejected():
    original = make_handoff()

    invalid = replace(
        original,
        total_cost_usd=100.0,
    )

    with pytest.raises(
        ValueError,
        match="does not match execution trace",
    ):
        validate_measurement_handoff(invalid)
