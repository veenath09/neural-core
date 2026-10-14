"""Execution trace contracts for Pillar 4.

These types describe what actually happened during execution.
They are separate from controller.contracts.TraceItem, which is
a short controller-history representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EXECUTION_TRACE_VERSION = "v1"

ExecutionStatus = Literal["ok", "error", "timeout"]


@dataclass(frozen=True)
class ExecutionStep:
    """One real execution step performed by Pillar 4."""

    step_index: int
    parent_step: int | None

    model_id: str
    tool: str | None

    input_tokens: int
    output_tokens: int

    cost_usd: float
    wall_ms: float

    status: ExecutionStatus
    error_type: str | None = None
    node_name: str | None = None


@dataclass(frozen=True)
class ExecutionTrace:
    """Complete trace produced for one executed strategy segment."""

    execution_id: str

    strategy_step_id: str
    strategy_round: int

    framework: str
    catalogue_version: str
    adapter_version: str

    steps: tuple[ExecutionStep, ...] = field(default_factory=tuple)

    final_status: ExecutionStatus = "ok"
    result_ref: str | None = None

    version: str = EXECUTION_TRACE_VERSION


def validate_execution_step(step: ExecutionStep) -> None:
    """Validate one execution step."""

    errors: list[str] = []

    if step.step_index < 0:
        errors.append("step_index cannot be negative")

    if step.parent_step is not None:
        if step.parent_step < 0:
            errors.append("parent_step cannot be negative")

        if step.parent_step >= step.step_index:
            errors.append(
                "parent_step must refer to an earlier execution step"
            )

    if step.input_tokens < 0:
        errors.append("input_tokens cannot be negative")

    if step.output_tokens < 0:
        errors.append("output_tokens cannot be negative")

    if step.cost_usd < 0:
        errors.append("cost_usd cannot be negative")

    if step.wall_ms < 0:
        errors.append("wall_ms cannot be negative")

    if step.status not in ("ok", "error", "timeout"):
        errors.append(f"invalid status {step.status!r}")

    if step.status == "ok" and step.error_type is not None:
        errors.append(
            "error_type must be None when status is 'ok'"
        )

    if errors:
        raise ValueError("; ".join(errors))


def validate_execution_trace(trace: ExecutionTrace) -> None:
    """Validate a complete execution trace."""

    errors: list[str] = []

    if not trace.execution_id:
        errors.append("execution_id cannot be empty")

    if not trace.strategy_step_id:
        errors.append("strategy_step_id cannot be empty")

    if trace.strategy_round < 1:
        errors.append("strategy_round must be at least 1")

    if not trace.framework:
        errors.append("framework cannot be empty")

    if trace.version != EXECUTION_TRACE_VERSION:
        errors.append(
            f"version {trace.version!r} "
            f"!= {EXECUTION_TRACE_VERSION!r}"
        )

    for expected_index, step in enumerate(trace.steps):
        if step.step_index != expected_index:
            errors.append(
                "execution steps must use consecutive "
                "step_index values starting from 0"
            )
            break

        try:
            validate_execution_step(step)
        except ValueError as exc:
            errors.append(
                f"invalid execution step {step.step_index}: {exc}"
            )

    if errors:
        raise ValueError("; ".join(errors))