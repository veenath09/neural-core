
"""Task 14 Part B: Real measurement handoff integration.

Tests the complete path:

Controller Strategy
    -> ExecutionRouter
    -> LangGraph / CrewAI
    -> ExecutionTrace
    -> MeasurementHandoff

Uses local Python workflows with no LLM API calls.
"""

from dataclasses import asdict
from typing import TypedDict

import pytest

from crewai.flow.flow import Flow, start, listen

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph

from controller.contracts import ModelCard, Strategy

from execution.adapters.crewai_methods import (
    CrewAIMethodAdapter,
)

from execution.adapters.langgraph_nodes import (
    LangGraphNodeAdapter,
)

from execution.catalogue import ResourceCatalogue

from execution.handoff import (
    build_measurement_handoff,
)

from execution.router import ExecutionRouter

from execution.serialization import (
    trace_from_dict,
    trace_to_dict,
)

from execution.trace import validate_execution_trace


# ---------------------------------------------------------
# 1. LangGraph workflow
# ---------------------------------------------------------

class GraphState(TypedDict, total=False):
    input_text: str
    output_text: str


def build_langgraph():

    def uppercase(state: GraphState):
        return {
            "output_text": state["input_text"].upper()
        }

    def add_exclamation(state: GraphState):
        return {
            "output_text": state["output_text"] + "!"
        }

    builder = StateGraph(GraphState)

    builder.add_node("uppercase", uppercase)

    builder.add_node(
        "add_exclamation",
        add_exclamation,
    )

    builder.add_edge(START, "uppercase")

    builder.add_edge(
        "uppercase",
        "add_exclamation",
    )

    builder.add_edge("add_exclamation", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )


# ---------------------------------------------------------
# 2. CrewAI workflow
# ---------------------------------------------------------

class LocalResearchFlow(Flow):

    @start()
    def uppercase(self):
        return self.state["input_text"].upper()

    @listen(uppercase)
    def add_exclamation(self, previous_result):
        return previous_result + "!"


# ---------------------------------------------------------
# 3. Build shared ExecutionRouter
# ---------------------------------------------------------

def make_router():

    card = ModelCard(
        model_id="handoff-local-model",
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

    langgraph_adapter = LangGraphNodeAdapter(
        graph=build_langgraph(),
        node_parents={
            "uppercase": None,
            "add_exclamation": "uppercase",
        },
    )

    crewai_adapter = CrewAIMethodAdapter(
        flow_factory=LocalResearchFlow,
        method_parents={
            "uppercase": None,
            "add_exclamation": "uppercase",
        },
    )

    return ExecutionRouter(
        catalogue=catalogue,
        adapters={
            "langgraph": langgraph_adapter,
            "crewai": crewai_adapter,
        },
    )


# ---------------------------------------------------------
# 4. Controller Strategy
# ---------------------------------------------------------

def make_strategy():

    return Strategy(
        step_id="handoff-integration-step",
        round=1,
        model_id="handoff-local-model",
        halt="stop",
    )


# ---------------------------------------------------------
# 5. Execute both real frameworks and build handoffs
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def integration_handoffs():

    router = make_router()

    strategy = make_strategy()

    original_strategy = asdict(strategy)

    results = {}

    framework_settings = {
        "langgraph": "node_event_observed",
        "crewai": "method_event_observed",
    }

    for framework, latency_basis in (
        framework_settings.items()
    ):

        # Execute the real local framework.
        result = router.execute(
            strategy=strategy,
            input_text="hello research",
            execution_id=f"handoff-{framework}-001",
            framework=framework,
        )

        # No LLM is called in either workflow.
        handoff = build_measurement_handoff(
            result,
            token_accounting="not_applicable",
            cost_accounting="not_applicable",
            latency_basis=latency_basis,
        )

        results[framework] = {
            "result": result,
            "handoff": handoff,
        }

    results["strategy_before"] = original_strategy
    results["strategy_after"] = asdict(strategy)

    return results


# ---------------------------------------------------------
# Test 1: Both workflows execute correctly
# ---------------------------------------------------------

def test_both_workflows_execute(
    integration_handoffs,
):

    for framework in ("langgraph", "crewai"):

        result = integration_handoffs[framework]["result"]

        assert result.output_text == (
            "HELLO RESEARCH!"
        )

        assert result.trace.final_status == "ok"


# ---------------------------------------------------------
# Test 2: Handoff preserves identity
# ---------------------------------------------------------

def test_handoff_preserves_execution_identity(
    integration_handoffs,
):

    for framework in ("langgraph", "crewai"):

        handoff = integration_handoffs[framework]["handoff"]

        assert handoff.execution_id == (
            f"handoff-{framework}-001"
        )

        assert handoff.framework == framework

        assert handoff.final_status == "ok"

        assert handoff.trace.strategy_step_id == (
            "handoff-integration-step"
        )


# ---------------------------------------------------------
# Test 3: No LLM usage is reported accurately
# ---------------------------------------------------------

def test_local_usage_accounting(
    integration_handoffs,
):

    for framework in ("langgraph", "crewai"):

        handoff = integration_handoffs[framework]["handoff"]

        assert handoff.token_accounting == (
            "not_applicable"
        )

        assert handoff.cost_accounting == (
            "not_applicable"
        )

        assert handoff.input_tokens == 0
        assert handoff.output_tokens == 0

        assert handoff.total_cost_usd == 0.0


# ---------------------------------------------------------
# Test 4: Correct latency provenance
# ---------------------------------------------------------

def test_framework_latency_basis(
    integration_handoffs,
):

    langgraph = integration_handoffs[
        "langgraph"
    ]["handoff"]

    crewai = integration_handoffs[
        "crewai"
    ]["handoff"]

    assert langgraph.latency_basis == (
        "node_event_observed"
    )

    assert crewai.latency_basis == (
        "method_event_observed"
    )

    for handoff in (langgraph, crewai):

        assert len(handoff.trace.steps) == 2

        assert all(
            step.wall_ms >= 0.0
            for step in handoff.trace.steps
        )


# ---------------------------------------------------------
# Test 5: Dependencies are preserved
# ---------------------------------------------------------

def test_handoff_preserves_dependencies(
    integration_handoffs,
):

    for framework in ("langgraph", "crewai"):

        handoff = integration_handoffs[framework]["handoff"]

        validate_execution_trace(handoff.trace)

        first, second = handoff.trace.steps

        assert first.node_name == "uppercase"

        assert second.node_name == (
            "add_exclamation"
        )

        assert first.dependencies == ()

        assert second.dependencies == (0,)


# ---------------------------------------------------------
# Test 6: Execution traces remain JSON-compatible
# ---------------------------------------------------------

def test_handoff_trace_serialization(
    integration_handoffs,
):

    for framework in ("langgraph", "crewai"):

        handoff = integration_handoffs[framework]["handoff"]

        original_trace = handoff.trace

        restored_trace = trace_from_dict(
            trace_to_dict(original_trace)
        )

        assert restored_trace == original_trace

        assert restored_trace.framework == framework


# ---------------------------------------------------------
# Test 7: Controller Strategy remains unchanged
# ---------------------------------------------------------

def test_controller_strategy_unchanged(
    integration_handoffs,
):

    assert (
        integration_handoffs["strategy_before"]
        == integration_handoffs["strategy_after"]
    )
