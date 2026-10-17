
"""Tests for execution trace JSON export and import."""

import json
from dataclasses import replace

import pytest

from execution.serialization import (
    load_trace_json,
    save_trace_json,
    trace_from_dict,
    trace_to_dict,
)

from execution.trace import ExecutionStep, ExecutionTrace


def make_step(index, parent_steps=()):
    return ExecutionStep(
        step_index=index,
        parent_step=None,
        parent_steps=parent_steps,
        model_id="model-a",
        tool=None,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        wall_ms=10.0,
        status="ok",
        node_name=f"node-{index}",
    )


def make_parallel_trace():
    return ExecutionTrace(
        execution_id="exec-001",
        strategy_step_id="step-001",
        strategy_round=1,
        framework="langgraph",
        catalogue_version="v1",
        adapter_version="v1",
        steps=(
            make_step(0),
            make_step(1),
            make_step(2, (0, 1)),
        ),
        final_status="ok",
    )


def test_trace_converts_to_dictionary():
    trace = make_parallel_trace()

    data = trace_to_dict(trace)

    assert data["execution_id"] == "exec-001"
    assert len(data["steps"]) == 3
    assert data["steps"][2]["parent_steps"] == [0, 1]


def test_trace_round_trip_preserves_data():
    original = make_parallel_trace()

    data = trace_to_dict(original)
    restored = trace_from_dict(data)

    assert restored == original


def test_parallel_dependencies_survive_json():
    original = make_parallel_trace()

    encoded = json.dumps(trace_to_dict(original))
    restored = trace_from_dict(json.loads(encoded))

    assert restored.steps[2].dependencies == (0, 1)


def test_save_and_load_json_file(tmp_path):
    original = make_parallel_trace()

    path = tmp_path / "traces" / "execution.json"

    save_trace_json(original, path)

    assert path.exists()

    restored = load_trace_json(path)

    assert restored == original


def test_invalid_trace_version_rejected():
    data = trace_to_dict(make_parallel_trace())

    data["version"] = "unsupported"

    with pytest.raises(
        ValueError,
        match="Unsupported execution trace version",
    ):
        trace_from_dict(data)


def test_invalid_parent_dependency_rejected():
    data = trace_to_dict(make_parallel_trace())

    # The last step cannot depend on itself.
    data["steps"][2]["parent_steps"] = [2]

    with pytest.raises(
        ValueError,
        match="earlier execution step",
    ):
        trace_from_dict(data)


def test_legacy_trace_without_parent_steps_loads():
    original = make_parallel_trace()

    # Simulate an older single-parent trace.
    old_step = replace(
        original.steps[2],
        parent_step=0,
        parent_steps=None,
    )

    legacy = replace(
        original,
        steps=original.steps[:2] + (old_step,),
    )

    data = trace_to_dict(legacy)

    # Older files may not have this optional field.
    for item in data["steps"]:
        item.pop("parent_steps", None)

    restored = trace_from_dict(data)

    assert restored.steps[2].dependencies == (0,)
