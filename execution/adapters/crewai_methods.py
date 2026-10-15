
"""CrewAI method-level execution tracing for Pillar 4.

Supported:
- Local CrewAI Flow execution
- Method start, finish, and failure events
- Ordering by CrewAI emission_sequence
- Event-timestamp durations
- Explicit single- and multiple-parent dependencies
- Standard ExecutionTrace output

Limitations:
- Each Flow method may execute only once per run.
- Dependencies must be supplied explicitly.
- LLM token usage and cost accounting are not supported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from threading import Lock
from time import perf_counter

from crewai.flow.flow import Flow
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    MethodExecutionStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionFailedEvent,
)

from execution.adapters.base import AdapterResult, ExecutionRequest
from execution.adapters.crewai import CrewAIAdapter
from execution.trace import ExecutionStep, ExecutionTrace


ParentSpec = str | tuple[str, ...] | None
MethodEvent = tuple[int, datetime, str, str, str | None]


class CrewAIMethodAdapter(CrewAIAdapter):
    """Trace individual methods of a local CrewAI Flow."""

    framework = "crewai"

    def __init__(
        self,
        flow_factory: Callable[[], Flow],
        *,
        method_parents: Mapping[str, ParentSpec],
        adapter_version: str = "v1",
    ) -> None:
        super().__init__(
            flow_factory=flow_factory,
            adapter_version=adapter_version,
        )
        self.method_parents = dict(method_parents)

    def _resolve_parents(
        self,
        method_name: str,
        completed: dict[str, int],
    ) -> tuple[int | None, tuple[int, ...] | None]:

        if method_name not in self.method_parents:
            raise ValueError(
                f"Missing dependency mapping for {method_name!r}"
            )

        specification = self.method_parents[method_name]

        if specification is None:
            parent_names = ()
        elif isinstance(specification, str):
            parent_names = (specification,)
        else:
            parent_names = tuple(specification)

        if len(parent_names) != len(set(parent_names)):
            raise ValueError("Duplicate parent method names")

        indexes = []

        for name in parent_names:
            if name not in completed:
                raise ValueError(
                    f"Parent method {name!r} has not completed"
                )

            indexes.append(completed[name])

        indexes = tuple(sorted(indexes))

        if not indexes:
            return None, ()

        if len(indexes) == 1:
            return indexes[0], None

        return None, indexes

    def _build_steps(
        self,
        records: list[MethodEvent],
        request: ExecutionRequest,
    ) -> list[ExecutionStep]:
        """Convert ordered method events into trace steps."""

        starts: dict[str, datetime] = {}
        completed: dict[str, int] = {}
        steps: list[ExecutionStep] = []

        seen_sequences: set[int] = set()

        for sequence, timestamp, phase, name, error_type in sorted(
            records,
            key=lambda record: record[0],
        ):
            if sequence in seen_sequences:
                raise ValueError(
                    "Duplicate CrewAI event sequence"
                )

            seen_sequences.add(sequence)

            if phase == "start":
                if name in starts or name in completed:
                    raise ValueError(
                        f"Repeated method {name!r} "
                        "is not supported"
                    )

                starts[name] = timestamp
                continue

            if name not in starts:
                raise ValueError(
                    f"Method {name!r} finished "
                    "without a start event"
                )

            started = starts.pop(name)

            wall_ms = (
                timestamp - started
            ).total_seconds() * 1000.0

            if wall_ms < 0:
                raise ValueError(
                    "Method finish timestamp precedes start"
                )

            parent_step, parent_steps = (
                self._resolve_parents(name, completed)
            )

            step = ExecutionStep(
                step_index=len(steps),
                parent_step=parent_step,
                parent_steps=parent_steps,
                model_id=request.resource.card.model_id,
                tool=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                wall_ms=wall_ms,
                status=(
                    "error" if phase == "failed" else "ok"
                ),
                error_type=error_type,
                node_name=name,
            )

            steps.append(step)
            completed[name] = step.step_index

        if starts:
            raise ValueError(
                "Unmatched CrewAI method start events"
            )

        return steps

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        """Run a Flow and capture its method-level trace."""

        execution_started = perf_counter()

        records: list[MethodEvent] = []
        lock = Lock()

        steps: list[ExecutionStep] = []
        output_text = ""
        run_error: Exception | None = None

        try:
            flow = self.flow_factory()

            def record(phase, source, event):
                if source is not flow:
                    return

                error_type = None

                if phase == "failed":
                    error_type = type(event.error).__name__

                with lock:
                    records.append(
                        (
                            event.emission_sequence,
                            event.timestamp,
                            phase,
                            event.method_name,
                            error_type,
                        )
                    )

            def on_started(source, event):
                record("start", source, event)

            def on_finished(source, event):
                record("finish", source, event)

            def on_failed(source, event):
                record("failed", source, event)

            handlers = (
                (MethodExecutionStartedEvent, on_started),
                (MethodExecutionFinishedEvent, on_finished),
                (MethodExecutionFailedEvent, on_failed),
            )

            for event_type, handler in handlers:
                crewai_event_bus.on(event_type)(handler)

            try:
                result = flow.kickoff(
                    inputs={"input_text": request.input_text}
                )

                if not isinstance(result, str):
                    raise TypeError(
                        "CrewAI Flow must return a string"
                    )

                # This adapter is intended for local-only
                # Flows without AI model calls.
                usage = getattr(
                    flow,
                    "usage_metrics",
                    None,
                )

                if (
                    usage is not None
                    and getattr(
                        usage,
                        "successful_requests",
                        0,
                    ) > 0
                ):
                    raise RuntimeError(
                        "LLM usage accounting is not "
                        "supported by this adapter"
                    )

                output_text = result

            except Exception as exc:
                run_error = exc

            finally:
                try:
                    flushed = crewai_event_bus.flush(
                        timeout=20
                    )
                finally:
                    for event_type, handler in handlers:
                        crewai_event_bus.off(
                            event_type,
                            handler,
                        )

            if not flushed:
                raise TimeoutError(
                    "CrewAI event handlers did not finish"
                )

            steps = self._build_steps(
                records,
                request,
            )

            if not steps:
                raise ValueError(
                    "No CrewAI method events were recorded"
                )

            if any(
                step.status == "error"
                for step in steps
            ):
                if run_error is None:
                    run_error = RuntimeError(
                        "CrewAI Flow method failed"
                    )

        except Exception as exc:
            if run_error is None:
                run_error = exc

        if run_error is not None:
            output_text = ""

            # Preserve any completed method steps.
            # Add an aggregate error step if no failed
            # method event explains the execution failure.
            if not steps or steps[-1].status != "error":
                steps.append(
                    ExecutionStep(
                        step_index=len(steps),
                        parent_step=None,
                        parent_steps=(),
                        model_id=request.resource.card.model_id,
                        tool=None,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        wall_ms=(
                            perf_counter() - execution_started
                        ) * 1000.0,
                        status="error",
                        error_type=type(run_error).__name__,
                        node_name=None,
                    )
                )

        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=tuple(steps),
            final_status=(
                "error" if run_error is not None else "ok"
            ),
        )

        return AdapterResult(
            output_text=output_text,
            trace=trace,
        )
