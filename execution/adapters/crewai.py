
"""CrewAI Flow execution adapter for Pillar 4.

Initial integration supports local CrewAI Flows without LLM calls.
Each Flow run is recorded as one execution step.

Per-method tracing and actual LLM usage/cost accounting are
not implemented in this initial adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from crewai.flow.flow import Flow

from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
)
from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
)


class CrewAIAdapter(ExecutionAdapter):
    """Execute a local CrewAI Flow using the standard interface."""

    framework = "crewai"

    def __init__(
        self,
        flow_factory: Callable[[], Flow],
        *,
        adapter_version: str = "v1",
    ) -> None:
        self.flow_factory = flow_factory
        self.adapter_version = adapter_version

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        """Execute one Flow and return a standardized trace."""

        started = perf_counter()

        output_text = ""
        status = "ok"
        error_type = None

        try:
            # Create a fresh Flow for each execution.
            flow = self.flow_factory()

            result = flow.kickoff(
                inputs={
                    "input_text": request.input_text,
                }
            )

            # This first adapter supports string results.
            if not isinstance(result, str):
                raise TypeError(
                    "CrewAI Flow must return a string"
                )

            # Do not silently report zero LLM usage
            # if the Flow actually called an AI model.
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
                    "LLM usage and cost accounting "
                    "are not supported by this "
                    "local-only adapter yet"
                )

            output_text = result

        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__

        wall_ms = (
            perf_counter() - started
        ) * 1000.0

        # One aggregate execution step for the entire Flow.
        # Zero LLM tokens and cost are valid for local-only
        # Python-function workflows.
        step = ExecutionStep(
            step_index=0,
            parent_step=None,
            parent_steps=(),
            model_id=request.resource.card.model_id,
            tool=None,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            wall_ms=wall_ms,
            status=status,
            error_type=error_type,
        )

        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=(step,),
            final_status=status,
        )

        return AdapterResult(
            output_text=output_text,
            trace=trace,
        )
