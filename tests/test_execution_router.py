
"""Tests for framework selection in Pillar 4."""

from dataclasses import asdict

import pytest

from controller.contracts import ModelCard, Strategy
from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
)
from execution.catalogue import ResourceCatalogue
from execution.router import ExecutionRouter
from execution.serialization import trace_from_dict, trace_to_dict
from execution.trace import ExecutionStep, ExecutionTrace


class LocalTestAdapter(ExecutionAdapter):
    """Framework-labelled local adapter with no LLM calls."""

    adapter_version = "v1"

    def __init__(self, framework: str):
        self.framework = framework

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        step = ExecutionStep(
            step_index=0,
            parent_step=None,
            parent_steps=(),
            model_id=request.resource.card.model_id,
            tool=None,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            wall_ms=1.0,
            status="ok",
        )

        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=(step,),
            final_status="ok",
        )

        return AdapterResult(
            output_text=f"{self.framework}:{request.input_text}",
            trace=trace,
        )


def make_strategy(model_id="model-a"):
    return Strategy(
        step_id="router-test-step",
        round=1,
        model_id=model_id,
        halt="stop",
    )


def make_router():
    card = ModelCard(
        model_id="model-a",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-16",
        capability=[0.5, -10.0],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [card]
    )

    return ExecutionRouter(
        catalogue=catalogue,
        adapters={
            "langgraph": LocalTestAdapter("langgraph"),
            "crewai": LocalTestAdapter("crewai"),
        },
    )


def execute(router, framework, strategy=None):
    return router.execute(
        strategy=strategy or make_strategy(),
        input_text="hello research",
        execution_id="router-exec-001",
        framework=framework,
    )


def test_both_frameworks_are_registered():
    router = make_router()

    assert router.available_frameworks == (
        "crewai",
        "langgraph",
    )


def test_langgraph_is_selected():
    result = execute(make_router(), "langgraph")

    assert result.output_text == "langgraph:hello research"
    assert result.trace.framework == "langgraph"
    assert result.trace.final_status == "ok"


def test_crewai_is_selected():
    result = execute(make_router(), "crewai")

    assert result.output_text == "crewai:hello research"
    assert result.trace.framework == "crewai"
    assert result.trace.final_status == "ok"


def test_same_model_can_use_both_frameworks():
    router = make_router()

    first = execute(router, "langgraph")
    second = execute(router, "crewai")

    assert first.trace.steps[0].model_id == "model-a"
    assert second.trace.steps[0].model_id == "model-a"
    assert first.trace.framework != second.trace.framework


def test_unknown_framework_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported execution framework",
    ):
        execute(make_router(), "unknown")


def test_invalid_registry_name_rejected():
    router = make_router()

    with pytest.raises(
        ValueError,
        match="registry key must match",
    ):
        ExecutionRouter(
            catalogue=router.catalogue,
            adapters={
                "incorrect": LocalTestAdapter("langgraph")
            },
        )


def test_router_does_not_modify_controller_strategy():
    router = make_router()
    strategy = make_strategy()

    original = asdict(strategy)

    execute(router, "crewai", strategy)

    assert asdict(strategy) == original


def test_routed_trace_is_json_serializable():
    result = execute(make_router(), "langgraph")

    restored = trace_from_dict(
        trace_to_dict(result.trace)
    )

    assert restored == result.trace
