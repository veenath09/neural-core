
"""JSON persistence for Pillar 4 measurement handoffs.

Preserves:
- Complete ExecutionTrace and dependencies
- Input/output token accounting
- Total execution cost
- Accounting provenance
- Latency measurement basis

Checks saved totals against the underlying execution trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from execution.adapters.base import AdapterResult
from execution.handoff import (
    MeasurementHandoff,
    build_measurement_handoff,
)
from execution.serialization import (
    trace_from_dict,
    trace_to_dict,
)


HANDOFF_VERSION = "v1"

ACCOUNTING_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_cost_usd",
)


def validate_measurement_handoff(
    handoff: MeasurementHandoff,
) -> None:
    """Verify that the handoff agrees with its execution trace."""

    expected = build_measurement_handoff(
        AdapterResult(
            output_text="",
            trace=handoff.trace,
        ),
        token_accounting=handoff.token_accounting,
        cost_accounting=handoff.cost_accounting,
        latency_basis=handoff.latency_basis,
    )

    for field_name in ACCOUNTING_FIELDS:
        actual_value = getattr(handoff, field_name)
        expected_value = getattr(expected, field_name)

        if (
            isinstance(actual_value, bool)
            or actual_value != expected_value
        ):
            raise ValueError(
                f"Measurement handoff {field_name} "
                "does not match execution trace"
            )


def handoff_to_dict(
    handoff: MeasurementHandoff,
) -> dict[str, Any]:
    """Convert a validated handoff to a JSON-compatible object."""

    validate_measurement_handoff(handoff)

    return {
        "handoff_version": HANDOFF_VERSION,
        "trace": trace_to_dict(handoff.trace),
        "input_tokens": handoff.input_tokens,
        "output_tokens": handoff.output_tokens,
        "total_cost_usd": handoff.total_cost_usd,
        "token_accounting": handoff.token_accounting,
        "cost_accounting": handoff.cost_accounting,
        "latency_basis": handoff.latency_basis,
    }


def handoff_from_dict(
    data: dict[str, Any],
) -> MeasurementHandoff:
    """Reconstruct and validate a saved measurement handoff."""

    if not isinstance(data, dict):
        raise ValueError(
            "Measurement handoff must be a JSON object"
        )

    if data.get("handoff_version") != HANDOFF_VERSION:
        raise ValueError(
            "Unsupported measurement handoff version"
        )

    required_fields = {
        "trace",
        "input_tokens",
        "output_tokens",
        "total_cost_usd",
        "token_accounting",
        "cost_accounting",
        "latency_basis",
    }

    missing = required_fields - data.keys()

    if missing:
        raise ValueError(
            f"Missing measurement handoff fields: {sorted(missing)}"
        )

    trace = trace_from_dict(data["trace"])

    # Recompute totals using the validated trace.
    # Do not blindly trust totals from the JSON file.
    handoff = build_measurement_handoff(
        AdapterResult(
            output_text="",
            trace=trace,
        ),
        token_accounting=data["token_accounting"],
        cost_accounting=data["cost_accounting"],
        latency_basis=data["latency_basis"],
    )

    for field_name in ACCOUNTING_FIELDS:
        stored_value = data[field_name]
        expected_value = getattr(handoff, field_name)

        if (
            isinstance(stored_value, bool)
            or stored_value != expected_value
        ):
            raise ValueError(
                f"Saved {field_name} does not match "
                "the execution trace"
            )

    return handoff


def save_handoff_json(
    handoff: MeasurementHandoff,
    path: str | Path,
) -> Path:
    """Save a complete measurement handoff as JSON."""

    destination = Path(path)

    data = handoff_to_dict(handoff)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    return destination


def load_handoff_json(
    path: str | Path,
) -> MeasurementHandoff:
    """Load and validate a saved measurement handoff."""

    source = Path(path)

    data = json.loads(
        source.read_text(encoding="utf-8")
    )

    return handoff_from_dict(data)
