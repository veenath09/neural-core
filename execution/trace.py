
"""Execution trace contracts for Pillar 4.

Defines the standardized execution data shared across adapters.

Supports:
- Execution identity and status
- Model, tool, token, cost and latency records
- LangGraph and CrewAI node-level tracing
- Single-parent and multiple-parent dependencies
- Trace integrity and reliability validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal


EXECUTION_TRACE_VERSION = "v1"

ExecutionStatus = Literal["ok", "error", "timeout"]


@dataclass(frozen=True)
class ExecutionStep:
    """One recorded step in an execution workflow."""

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

    # None: use legacy parent_step.
    # (): root node with no dependencies.
    # (0, 1): node depends on steps 0 and 1.
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
    """Complete trace produced by one execution."""

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
    """Validate an individual execution step."""

    errors: list[str] = []

    # 1. Step index
    if step.step_index < 0:
        errors.append(
            "step_index cannot be negative"
        )

    # 2. Legacy single-parent dependency
    if step.parent_step is not None:

        if step.parent_step < 0:
            errors.append(
                "parent_step cannot be negative"
            )

        if step.parent_step >= step.step_index:
            errors.append(
                "parent_step must refer to an earlier execution step"
            )

    # 3. Multiple-parent dependencies
    if step.parent_steps is not None:

        parents = step.parent_steps

        if len(parents) != len(set(parents)):
            errors.append(
                "parent_steps contains duplicate parents"
            )

        for parent in parents:

            if parent < 0:
                errors.append(
                    "parent_steps cannot contain negative indexes"
                )

            if parent >= step.step_index:
                errors.append(
                    "every parent must refer to an earlier execution step"
                )

        if (
            step.parent_step is not None
            and parents != (step.parent_step,)
        ):
            errors.append(
                "parent_step conflicts with parent_steps"
            )

    # 4. Token counts
    if step.input_tokens < 0:
        errors.append(
            "input_tokens cannot be negative"
        )

    if step.output_tokens < 0:
        errors.append(
            "output_tokens cannot be negative"
        )

    # 5. Cost integrity
    # NaN and infinity cannot be valid execution costs.
    if (
        isinstance(step.cost_usd, bool)
        or not isinstance(step.cost_usd, (int, float))
        or not isfinite(step.cost_usd)
    ):
        errors.append(
            "cost_usd must be a finite number"
        )

    elif step.cost_usd < 0:
        errors.append(
            "cost_usd cannot be negative"
        )

    # 6. Latency integrity
    # Reject NaN and both positive and negative infinity.
    if (
        isinstance(step.wall_ms, bool)
        or not isinstance(step.wall_ms, (int, float))
        or not isfinite(step.wall_ms)
    ):
        errors.append(
            "wall_ms must be a finite number"
        )

    elif step.wall_ms < 0:
        errors.append(
            "wall_ms cannot be negative"
        )

    # 7. Individual execution status
    valid_statuses = ("ok", "error", "timeout")

    if step.status not in valid_statuses:
        errors.append(
            f"invalid status {step.status!r}"
        )

    # A successful step must not report an error.
    if (
        step.status == "ok"
        and step.error_type is not None
    ):
        errors.append(
            "error_type must be None when status is 'ok'"
        )

    if errors:
        raise ValueError("; ".join(errors))


def validate_execution_trace(trace: ExecutionTrace) -> None:
    """Validate complete execution trace integrity."""

    errors: list[str] = []

    # 1. Execution identity
    if not trace.execution_id:
        errors.append(
            "execution_id cannot be empty"
        )

    if not trace.strategy_step_id:
        errors.append(
            "strategy_step_id cannot be empty"
        )

    # 2. Strategy round
    if trace.strategy_round < 1:
        errors.append(
            "strategy_round must be at least 1"
        )

    # 3. Framework identity
    if not trace.framework:
        errors.append(
            "framework cannot be empty"
        )

    # 4. Version compatibility
    if trace.version != EXECUTION_TRACE_VERSION:
        errors.append(
            f"version {trace.version!r} "
            f"!= {EXECUTION_TRACE_VERSION!r}"
        )

    # 5. Final execution status
    valid_statuses = ("ok", "error", "timeout")

    if trace.final_status not in valid_statuses:
        errors.append(
            f"invalid final_status {trace.final_status!r}"
        )

    # 6. Validate all individual steps
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
                f"invalid execution step "
                f"{step.step_index}: {exc}"
            )

    # 7. Successful executions must contain steps
    if (
        trace.final_status == "ok"
        and not trace.steps
    ):
        errors.append(
            "successful execution trace cannot be empty"
        )

    # 8. Successful executions cannot contain
    # failed or timed-out steps.
    if trace.final_status == "ok":

        if any(
            step.status != "ok"
            for step in trace.steps
        ):
            errors.append(
                "successful execution trace "
                "cannot contain failed or timed-out steps"
            )

    # 9. Failed executions must record at least
    # one actual failed step.
    if trace.final_status == "error":

        if not any(
            step.status == "error"
            for step in trace.steps
        ):
            errors.append(
                "failed execution trace "
                "must contain a failed step"
            )

    # 10. Timed-out executions must record
    # at least one timed-out step.
    if trace.final_status == "timeout":

        if not any(
            step.status == "timeout"
            for step in trace.steps
        ):
            errors.append(
                "timed-out execution trace "
                "must contain a timed-out step"
            )

    if errors:
        raise ValueError("; ".join(errors))
