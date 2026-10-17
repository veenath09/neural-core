
"""JSON serialization for Pillar 4 execution traces.

Supports:
- Exporting execution traces to JSON
- Importing execution traces from JSON
- Preserving parallel workflow dependencies
- Trace version validation
- Saving and loading JSON files
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from execution.trace import (
    EXECUTION_TRACE_VERSION,
    ExecutionStep,
    ExecutionTrace,
    validate_execution_trace,
)


def trace_to_dict(trace: ExecutionTrace) -> dict[str, Any]:
    """Convert an execution trace to a JSON-compatible dictionary."""

    validate_execution_trace(trace)

    payload = asdict(trace)
    payload["steps"] = []

    for step in trace.steps:
        item = asdict(step)

        # Tuples must be represented as JSON arrays.
        if step.parent_steps is not None:
            item["parent_steps"] = list(step.parent_steps)

        payload["steps"].append(item)

    return payload


def trace_from_dict(data: dict[str, Any]) -> ExecutionTrace:
    """Reconstruct an execution trace from its JSON representation."""

    if not isinstance(data, dict):
        raise ValueError("Execution trace must be a JSON object")

    version = data.get("version")

    if version != EXECUTION_TRACE_VERSION:
        raise ValueError(
            f"Unsupported execution trace version: {version!r}"
        )

    raw_steps = data.get("steps")

    if not isinstance(raw_steps, list):
        raise ValueError(
            "Execution trace steps must be a list"
        )

    steps = []

    for index, raw_step in enumerate(raw_steps):

        if not isinstance(raw_step, dict):
            raise ValueError(
                f"Execution step {index} must be an object"
            )

        step_data = dict(raw_step)

        parents = step_data.get("parent_steps")

        if parents is not None:
            if (
                not isinstance(parents, list)
                or not all(type(p) is int for p in parents)
            ):
                raise ValueError(
                    f"Invalid parent_steps in step {index}"
                )

            step_data["parent_steps"] = tuple(parents)

        try:
            step = ExecutionStep(**step_data)
        except TypeError as exc:
            raise ValueError(
                f"Invalid execution step {index}: {exc}"
            ) from exc

        steps.append(step)

    trace_data = {
        key: value
        for key, value in data.items()
        if key != "steps"
    }

    try:
        trace = ExecutionTrace(
            **trace_data,
            steps=tuple(steps),
        )
    except TypeError as exc:
        raise ValueError(
            f"Invalid execution trace: {exc}"
        ) from exc

    validate_execution_trace(trace)

    return trace


def save_trace_json(
    trace: ExecutionTrace,
    path: str | Path,
) -> Path:
    """Save an execution trace to a formatted JSON file."""

    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = trace_to_dict(trace)

    destination.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return destination


def load_trace_json(path: str | Path) -> ExecutionTrace:
    """Load and validate an execution trace from a JSON file."""

    source = Path(path)

    data = json.loads(
        source.read_text(encoding="utf-8")
    )

    return trace_from_dict(data)
