"""Concrete local callable adapter for Pillar 4.

This adapter executes a Python callable while using the same interface
that future framework adapters such as LangGraph and CrewAI will use.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
)
from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
)


Runner = Callable[[str], str]
TokenCounter = Callable[[str], int]


def whitespace_token_counter(text: str) -> int:
    """Small deterministic token-count approximation for local testing."""

    if not text:
        return 0

    return len(text.split())


class CallableAdapter(ExecutionAdapter):
    """Execute a local Python callable through the common adapter contract."""

    framework = "callable"

    def __init__(
        self,
        runner: Runner,
        *,
        adapter_version: str = "v1",
        token_counter: TokenCounter = whitespace_token_counter,
    ) -> None:
        self.runner = runner
        self.adapter_version = adapter_version
        self.token_counter = token_counter

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        """Execute the configured callable and record a real runtime trace."""

        input_tokens = self.token_counter(request.input_text)

        started = perf_counter()

        try:
            output_text = self.runner(request.input_text)

        except Exception as exc:
            wall_ms = (perf_counter() - started) * 1000.0

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

        wall_ms = (perf_counter() - started) * 1000.0

        output_tokens = self.token_counter(output_text)

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