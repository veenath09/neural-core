
"""Task 13 Part C: Real framework integration tests.

Verifies:
- Controller Strategy -> ExecutionRouter
- Actual LangGraph node adapter
- Actual CrewAI method adapter
- Standardized execution traces
- JSON serialization
- Controller Strategy remains unchanged

All workflows use local Python functions, not LLM APIs.
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
from execution.router import ExecutionRouter
from execution.serialization import (
    trace_from_dict,
    trace_to_dict,
)
from execution.trace import validate_execution_trace


# ---------------------------------------------------------
# 1. Real LangGraph workflow
# ---------------------------------------------------------

class GraphState(TypedDict, total=False):
    input_text: str
    output_text: str


def build_langgraph():
    """Create a two-node sequential LangGraph workflow."""

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
# 2. Real CrewAI Flow
# ---------------------------------------------------------

class ResearchCrewAIFlow(Flow):
    """Create a two-method local CrewAI Flow."""

    @start()
    def uppercase(self):
        return self.state["input_text"].upper()

    @listen(uppercase)
    def add_exclamation(self, previous_result):
        return previous_result + "!"


# ---------------------------------------------------------
# 3. Shared controller Strategy
# ---------------------------------------------------------

def make_strategy():
    return Strategy(
        step_id="integration-step-001",
        round=1,
        model_id="local-integration-model",
        halt="stop",
    )


# ---------------------------------------------------------
# 4. Register real adapters
# ---------------------------------------------------------

def make_router():
    """Build a router connected to both real frameworks."""

    card = ModelCard(
        model_id="local-integration-model",
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
        flow_factory=ResearchCrewAIFlow,
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
# 5. Execute both actual frameworks
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def integration_results():
    """Execute both frameworks once and share test results."""

    router = make_router()

    strategy = make_strategy()

    before = asdict(strategy)

    langgraph_result = router.execute(
        strategy=strategy,
        input_text="hello research",
        execution_id="integration-langgraph-001",
        framework="langgraph",
    )

    crewai_result = router.execute(
        strategy=strategy,
        input_text="hello research",
        execution_id="integration-crewai-001",
        framework="crewai",
    )

    return {
        "langgraph": langgraph_result,
        "crewai": crewai_result,
        "strategy_before": before,
        "strategy_after": asdict(strategy),
    }


# ---------------------------------------------------------
# Test 1: Both real frameworks execute correctly
# ---------------------------------------------------------

def test_both_frameworks_execute(integration_results):

    langgraph_result = integration_results["langgraph"]
    crewai_result = integration_results["crewai"]

    assert langgraph_result.output_text == (
        "HELLO RESEARCH!"
    )

    assert crewai_result.output_text == (
        "HELLO RESEARCH!"
    )

    assert langgraph_result.trace.final_status == "ok"
    assert crewai_result.trace.final_status == "ok"


# ---------------------------------------------------------
# Test 2: Real LangGraph node-level trace
# ---------------------------------------------------------

def test_langgraph_node_trace(integration_results):

    trace = integration_results["langgraph"].trace

    validate_execution_trace(trace)

    assert trace.framework == "langgraph"
    assert len(trace.steps) == 2

    assert [
        step.node_name
        for step in trace.steps
    ] == [
        "uppercase",
        "add_exclamation",
    ]

    assert trace.steps[0].dependencies == ()
    assert trace.steps[1].dependencies == (0,)


# ---------------------------------------------------------
# Test 3: Real CrewAI method-level trace
# ---------------------------------------------------------

def test_crewai_method_trace(integration_results):

    trace = integration_results["crewai"].trace

    validate_execution_trace(trace)

    assert trace.framework == "crewai"
    assert len(trace.steps) == 2

    assert [
        step.node_name
        for step in trace.steps
    ] == [
        "uppercase",
        "add_exclamation",
    ]

    assert trace.steps[0].dependencies == ()
    assert trace.steps[1].dependencies == (0,)


# ---------------------------------------------------------
# Test 4: Framework identities stay independent
# ---------------------------------------------------------

def test_framework_execution_ids(integration_results):

    langgraph_trace = integration_results["langgraph"].trace
    crewai_trace = integration_results["crewai"].trace

    assert langgraph_trace.execution_id == (
        "integration-langgraph-001"
    )

    assert crewai_trace.execution_id == (
        "integration-crewai-001"
    )

    assert langgraph_trace.execution_id != (
        crewai_trace.execution_id
    )

    assert langgraph_trace.strategy_step_id == (
        "integration-step-001"
    )

    assert crewai_trace.strategy_step_id == (
        "integration-step-001"
    )


# ---------------------------------------------------------
# Test 5: Both traces support JSON serialization
# ---------------------------------------------------------

def test_both_framework_traces_serialize(
    integration_results,
):

    for framework in ("langgraph", "crewai"):

        original = integration_results[framework].trace

        restored = trace_from_dict(
            trace_to_dict(original)
        )

        assert restored == original
        assert restored.framework == framework


# ---------------------------------------------------------
# Test 6: Controller Strategy is not modified
# ---------------------------------------------------------

def test_controller_strategy_unchanged(
    integration_results,
):

    assert (
        integration_results["strategy_before"]
        == integration_results["strategy_after"]
    )
