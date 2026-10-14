"""LangGraph execution adapter for Pillar 4.

This adapter allows the framework-independent ExecutionEngine to execute
a compiled LangGraph workflow while returning the standard ExecutionTrace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
)
from execution.adapters.callable import whitespace_token_counter
from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
)


GraphInputBuilder = Callable[[ExecutionRequest], dict[str, Any]]
GraphOutputExtractor = Callable[[Any], str]
TokenCounter = Callable[[str], int]


def default_input_builder(
    request: ExecutionRequest,
) -> dict[str, Any]:
    """Create the default LangGraph state from an execution request."""

    return {
        "input_text": request.input_text,
    }


def default_output_extractor(result: Any) -> str:
    """Extract output_text from the final LangGraph state."""

    if not isinstance(result, Mapping):
        raise ValueError(
            "LangGraph result must be a mapping"
        )

    if "output_text" not in result:
        raise ValueError(
            "LangGraph result does not contain 'output_text'"
        )

    return str(result["output_text"])


class LangGraphAdapter(ExecutionAdapter):
    """Execute a compiled LangGraph through the common adapter interface."""

    framework = "langgraph"

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        adapter_version: str = "v1",
        input_builder: GraphInputBuilder = default_input_builder,
        output_extractor: GraphOutputExtractor = default_output_extractor,
        token_counter: TokenCounter = whitespace_token_counter,
    ) -> None:
        self.graph = graph
        self.adapter_version = adapter_version
        self.input_builder = input_builder
        self.output_extractor = output_extractor
        self.token_counter = token_counter

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        """Execute the LangGraph workflow and build a standard trace."""

        input_tokens = self.token_counter(
            request.input_text
        )

        started = perf_counter()

        try:
            graph_input = self.input_builder(request)

            graph_result = self.graph.invoke(
                graph_input
            )

            output_text = self.output_extractor(
                graph_result
            )

        except Exception as exc:
            wall_ms = (
                perf_counter() - started
            ) * 1000.0

            trace = ExecutionTrace(
                execution_id=request.execution_id,
                strategy_step_id=request.strategy.step_id,
                strategy_round=request.strategy.round,
                framework=self.framework,
                catalogue_version=request.catalogue_version,
                adapter_version=self.adapter_version,
                steps=(
                    ExecutionStep(
                        step_index=0,
                        parent_step=None,
                        model_id=request.resource.card.model_id,
                        tool=None,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        cost_usd=0.0,
                        wall_ms=wall_ms,
                        status="error",
                        error_type=type(exc).__name__,
                    ),
                ),
                final_status="error",
            )

            return AdapterResult(
                output_text="",
                trace=trace,
            )

        wall_ms = (
            perf_counter() - started
        ) * 1000.0

        output_tokens = self.token_counter(
            output_text
        )

        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=(
                ExecutionStep(
                    step_index=0,
                    parent_step=None,
                    model_id=request.resource.card.model_id,
                    tool=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=0.0,
                    wall_ms=wall_ms,
                    status="ok",
                ),
            ),
            final_status="ok",
        )

        return AdapterResult(
            output_text=output_text,
            trace=trace,
        )