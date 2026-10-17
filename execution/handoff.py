
"""Measurement handoff contract for Pillar 4.

Transfers validated execution traces to Pillar 3.

Important:
- Zero cost is different from unavailable cost.
- Zero tokens are different from unavailable tokens.
- Per-node wall_ms is not necessarily end-to-end latency.
- Measurement provenance must be stated explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from execution.adapters.base import AdapterResult
from execution.trace import (
    ExecutionTrace,
    validate_execution_trace,
)


AccountingStatus = Literal[
    "measured",
    "not_applicable",
    "unavailable",
]

LatencyBasis = Literal[
    "node_event_observed",
    "method_event_observed",
    "aggregate_wall_clock",
]


@dataclass(frozen=True)
class MeasurementHandoff:
    """Validated execution information for Pillar 3."""

    trace: ExecutionTrace

    input_tokens: int | None
    output_tokens: int | None
    total_cost_usd: float | None

    token_accounting: AccountingStatus
    cost_accounting: AccountingStatus

    latency_basis: LatencyBasis

    @property
    def execution_id(self) -> str:
        return self.trace.execution_id

    @property
    def framework(self) -> str:
        return self.trace.framework

    @property
    def final_status(self) -> str:
        return self.trace.final_status


def build_measurement_handoff(
    result: AdapterResult,
    *,
    token_accounting: AccountingStatus,
    cost_accounting: AccountingStatus,
    latency_basis: LatencyBasis,
) -> MeasurementHandoff:
    """Build a validated measurement handoff.

    The caller must provide truthful accounting provenance.

    'measured': Actual usage was recorded by the adapter.
    'not_applicable': No LLM usage applies, e.g. local Flow.
    'unavailable': The adapter did not obtain reliable usage.

    This function cannot independently verify an adapter's
    claim that its usage metrics were measured.
    """

    validate_execution_trace(result.trace)

    allowed_accounting = {
        "measured",
        "not_applicable",
        "unavailable",
    }

    allowed_latency = {
        "node_event_observed",
        "method_event_observed",
        "aggregate_wall_clock",
    }

    if token_accounting not in allowed_accounting:
        raise ValueError(
            "Invalid token accounting status"
        )

    if cost_accounting not in allowed_accounting:
        raise ValueError(
            "Invalid cost accounting status"
        )

    if latency_basis not in allowed_latency:
        raise ValueError(
            "Invalid latency basis"
        )

    steps = result.trace.steps

    recorded_input_tokens = sum(
        step.input_tokens for step in steps
    )

    recorded_output_tokens = sum(
        step.output_tokens for step in steps
    )

    recorded_cost = sum(
        step.cost_usd for step in steps
    )

    # "Not applicable" must genuinely contain
    # no recorded token or cost usage.
    if token_accounting == "not_applicable":
        if recorded_input_tokens or recorded_output_tokens:
            raise ValueError(
                "Nonzero tokens cannot be not_applicable"
            )

    if cost_accounting == "not_applicable":
        if recorded_cost:
            raise ValueError(
                "Nonzero cost cannot be not_applicable"
            )

    # Never interpret missing usage information as zero.
    if token_accounting == "unavailable":
        input_tokens = None
        output_tokens = None
    else:
        input_tokens = recorded_input_tokens
        output_tokens = recorded_output_tokens

    if cost_accounting == "unavailable":
        total_cost_usd = None
    else:
        total_cost_usd = recorded_cost

    return MeasurementHandoff(
        trace=result.trace,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=total_cost_usd,
        token_accounting=token_accounting,
        cost_accounting=cost_accounting,
        latency_basis=latency_basis,
    )
