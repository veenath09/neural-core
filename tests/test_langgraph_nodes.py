
"""Integration tests for LangGraph node-level traces."""

import time
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from controller.contracts import ModelCard, Strategy
from execution.adapters.langgraph_nodes import (
    LangGraphNodeAdapter,
)
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine


class GraphState(TypedDict):
    input_text: str
    output_text: str


def build_graph():
    def uppercase(state: GraphState):
        time.sleep(0.01)
        return {
            "output_text": state["input_text"].upper()
        }

    def add_exclamation(state: GraphState):
        time.sleep(0.01)
        return {
            "output_text": state["output_text"] + "!"
        }

    builder = StateGraph(GraphState)
    builder.add_node("uppercase", uppercase)
    builder.add_node("add_exclamation", add_exclamation)

    builder.add_edge(START, "uppercase")
    builder.add_edge("uppercase", "add_exclamation")
    builder.add_edge("add_exclamation", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )


def make_engine():
    card = ModelCard(
        model_id="local-graph-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-14",
        capability=[0.5, -10.0],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [card]
    )

    adapter = LangGraphNodeAdapter(
        graph=build_graph(),
        node_parents={
            "uppercase": None,
            "add_exclamation": "uppercase",
        },
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def run_graph():
    strategy = Strategy(
        step_id="step-001",
        round=1,
        model_id="local-graph-model",
        halt="stop",
    )

    return make_engine().execute(
        strategy=strategy,
        input_text="hello research",
        execution_id="exec-nodes-001",
    )


def test_graph_produces_correct_output():
    result = run_graph()

    assert result.output_text == "HELLO RESEARCH!"
    assert result.trace.final_status == "ok"


def test_trace_contains_two_nodes():
    result = run_graph()

    assert len(result.trace.steps) == 2

    assert [
        step.node_name
        for step in result.trace.steps
    ] == [
        "uppercase",
        "add_exclamation",
    ]


def test_node_dependencies_are_recorded():
    result = run_graph()

    assert [
        step.parent_step
        for step in result.trace.steps
    ] == [None, 0]


def test_node_execution_times_are_measured():
    result = run_graph()

    for step in result.trace.steps:
        assert step.wall_ms >= 0.0
        assert step.status == "ok"


def test_execution_identity_is_preserved():
    result = run_graph()

    assert result.trace.execution_id == "exec-nodes-001"
    assert result.trace.strategy_step_id == "step-001"
    assert result.trace.framework == "langgraph"


def test_local_nodes_do_not_claim_api_usage():
    result = run_graph()

    for step in result.trace.steps:
        assert step.cost_usd == 0.0
        assert step.input_tokens == 0
        assert step.output_tokens == 0
