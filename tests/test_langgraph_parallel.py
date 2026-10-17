
"""Integration tests for parallel LangGraph execution traces."""

import time
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph

from controller.contracts import ModelCard, Strategy
from execution.adapters.langgraph_nodes import LangGraphNodeAdapter
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine


class ParallelState(TypedDict, total=False):
    input_text: str
    left_text: str
    right_text: str
    output_text: str


def build_parallel_graph():

    def left(state: ParallelState):
        time.sleep(0.02)
        return {
            "left_text": state["input_text"].upper()
        }

    def right(state: ParallelState):
        time.sleep(0.03)
        return {
            "right_text": state["input_text"].lower()
        }

    def join(state: ParallelState):
        return {
            "output_text": (
                state["left_text"]
                + " | "
                + state["right_text"]
            )
        }

    builder = StateGraph(ParallelState)

    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_node("join", join)

    builder.add_edge(START, "left")
    builder.add_edge(START, "right")

    # Join must wait for BOTH parallel nodes.
    builder.add_edge(
        ["left", "right"],
        "join",
    )

    builder.add_edge("join", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )


def make_engine():

    card = ModelCard(
        model_id="parallel-model",
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
        graph=build_parallel_graph(),
        node_parents={
            "left": None,
            "right": None,
            "join": ("left", "right"),
        },
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def run_parallel_graph():

    strategy = Strategy(
        step_id="parallel-step",
        round=1,
        model_id="parallel-model",
        halt="stop",
    )

    return make_engine().execute(
        strategy=strategy,
        input_text="Hello Research",
        execution_id="exec-parallel-001",
    )


def test_parallel_graph_returns_correct_output():

    result = run_parallel_graph()

    assert result.output_text == (
        "HELLO RESEARCH | hello research"
    )

    assert result.trace.final_status == "ok"


def test_trace_contains_three_nodes():

    result = run_parallel_graph()

    assert len(result.trace.steps) == 3

    names = [
        step.node_name
        for step in result.trace.steps
    ]

    assert set(names) == {
        "left",
        "right",
        "join",
    }


def test_parallel_nodes_have_no_parents():

    result = run_parallel_graph()

    steps = {
        step.node_name: step
        for step in result.trace.steps
    }

    assert steps["left"].dependencies == ()
    assert steps["right"].dependencies == ()


def test_join_has_two_parent_dependencies():

    result = run_parallel_graph()

    steps = {
        step.node_name: step
        for step in result.trace.steps
    }

    join_step = steps["join"]

    expected = {
        steps["left"].step_index,
        steps["right"].step_index,
    }

    assert set(join_step.dependencies) == expected
    assert len(join_step.dependencies) == 2

    # Both dependencies must occur before join.
    assert all(
        index < join_step.step_index
        for index in join_step.dependencies
    )


def test_parallel_node_timings_are_recorded():

    result = run_parallel_graph()

    for step in result.trace.steps:
        assert step.wall_ms >= 0.0
        assert step.status == "ok"


def test_parallel_trace_preserves_identity():

    result = run_parallel_graph()

    assert result.trace.execution_id == (
        "exec-parallel-001"
    )

    assert result.trace.strategy_step_id == (
        "parallel-step"
    )

    assert result.trace.framework == "langgraph"
