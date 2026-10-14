
"""Execution trace contracts for Pillar 4.

These types describe what actually happened during execution.

They are separate from controller.contracts.TraceItem, which is
a short controller-history representation.

Supports:
- Individual execution steps
- Execution cost and latency
- Input and output token counts
- Execution status and error information
- Node-level tracing
- Single-parent dependencies (legacy)
- Multiple-parent dependencies (parallel workflows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EXECUTION_TRACE_VERSION = "v1"

ExecutionStatus = Literal["ok", "error", "timeout"]


@dataclass(frozen=True)
class ExecutionStep:
    """One execution step performed by Pillar 4."""

    step_index: int
    parent_step: int | None

    model_id: str
    tool: str | None

    input_tokens: int
    output_tokens: int

    cost_usd: float
    wall_ms: float

    status: ExecutionStatus

    # Error information
    error_type: str | None = None

    # LangGraph node-level tracing (Task 7)
    node_name: str | None = None

    # Multiple dependencies (Task 8)
    # None means use the legacy parent_step field.
    # An empty tuple means the node has no parents.
    parent_steps: tuple[int, ...] | None = None

    @property
    def dependencies(self) -> tuple[int, ...]:
        """Return all parent execution-step indexes."""

        if self.parent_steps is not None:
            return self.parent_steps

        if self.parent_step is None:
            return ()

        return (self.parent_step,)


@dataclass(frozen=True)
class ExecutionTrace:
    """Complete trace produced for one executed strategy segment."""

    execution_id: str

    strategy_step_id: str
    strategy_round: int

    framework: str
    catalogue_version: str
    adapter_version: str

    steps: tuple[ExecutionStep, ...] = field(
        default_factory=tuple
    )

    final_status: ExecutionStatus = "ok"
    result_ref: str | None = None

    version: str = EXECUTION_TRACE_VERSION


def validate_execution_step(step: ExecutionStep) -> None:
    """Validate one execution step and its dependencies."""

    errors: list[str] = []

    # Step index validation
    if step.step_index < 0:
        errors.append("step_index cannot be negative")

    # Legacy single-parent validation
    if step.parent_step is not None:

        if step.parent_step < 0:
            errors.append(
                "parent_step cannot be negative"
            )

        if step.parent_step >= step.step_index:
            errors.append(
                "parent_step must refer to an earlier execution step"
            )

    # Task 8: Multiple-parent validation
    if step.parent_steps is not None:

        parents = step.parent_steps

        # Reject duplicate dependencies
        if len(parents) != len(set(parents)):
            errors.append(
                "parent_steps contains duplicate parents"
            )

        # Every parent must be an earlier step
        for parent in parents:

            if parent < 0:
                errors.append(
                    "parent_steps cannot contain negative indexes"
                )

            if parent >= step.step_index:
                errors.append(
                    "every parent must refer to an earlier execution step"
                )

        # Prevent contradictory dependency formats
        if (
            step.parent_step is not None
            and parents != (step.parent_step,)
        ):
            errors.append(
                "parent_step conflicts with parent_steps"
            )

    # Token validation
    if step.input_tokens < 0:
        errors.append(
            "input_tokens cannot be negative"
        )

    if step.output_tokens < 0:
        errors.append(
            "output_tokens cannot be negative"
        )

    # Cost validation
    if step.cost_usd < 0:
        errors.append(
            "cost_usd cannot be negative"
        )

    # Latency validation
    if step.wall_ms < 0:
        errors.append(
            "wall_ms cannot be negative"
        )

    # Execution status validation
    if step.status not in ("ok", "error", "timeout"):
        errors.append(
            f"invalid status {step.status!r}"
        )

    # Successful steps must not contain an error
    if step.status == "ok" and step.error_type is not None:
        errors.append(
            "error_type must be None when status is 'ok'"
        )

    # Raise all validation errors together
    if errors:
        raise ValueError("; ".join(errors))


def validate_execution_trace(trace: ExecutionTrace) -> None:
    """Validate a complete execution trace."""

    errors: list[str] = []

    # Execution identity
    if not trace.execution_id:
        errors.append(
            "execution_id cannot be empty"
        )

    if not trace.strategy_step_id:
        errors.append(
            "strategy_step_id cannot be empty"
        )

    # Strategy round
    if trace.strategy_round < 1:
        errors.append(
            "strategy_round must be at least 1"
        )

    # Framework
    if not trace.framework:
        errors.append(
            "framework cannot be empty"
        )

    # Trace version
    if trace.version != EXECUTION_TRACE_VERSION:
        errors.append(
            f"version {trace.version!r} "
            f"!= {EXECUTION_TRACE_VERSION!r}"
        )

    # Validate every execution step
    for expected_index, step in enumerate(trace.steps):

        # Step indexes must be consecutive
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
                f"invalid execution step "
                f"{step.step_index}: {exc}"
            )

    # Raise all trace errors together
    if errors:
        raise ValueError("; ".join(errors))
