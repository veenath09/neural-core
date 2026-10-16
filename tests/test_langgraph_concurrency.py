
"""Task 12 Part C: Concurrent LangGraph execution tests.

Tests several LangGraph executions sharing one ExecutionEngine.

Each workflow contains two parallel nodes and a join node.

No LLM calls or API keys are required.
"""

from concurrent.futures import ThreadPoolExecutor
import time
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph

from controller.contracts import ModelCard, Strategy

from execution.adapters.langgraph_nodes import (
    LangGraphNodeAdapter,
)
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine
from execution.serialization import (
    trace_from_dict,
    trace_to_dict,
)
from execution.trace import validate_execution_trace


class GraphState(TypedDict, total=False):
    input_text: str
    left_text: str
    right_text: str
    output_text: str


def build_graph():
    """Build a LangGraph with parallel execution branches."""

    def left(state: GraphState):
        time.sleep(0.05)

        return {
            "left_text": state["input_text"].upper()
        }

    def right(state: GraphState):
        time.sleep(0.03)

        return {
            "right_text": state["input_text"].lower()
        }

    def join(state: GraphState):
        return {
            "output_text": (
                state["left_text"]
                + " | "
                + state["right_text"]
            )
        }

    builder = StateGraph(GraphState)

    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_node("join", join)

    builder.add_edge(START, "left")
    builder.add_edge(START, "right")

    # The join waits for both parallel branches.
    builder.add_edge(
        ["left", "right"],
        "join",
    )

    builder.add_edge("join", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )


def make_engine():
    """Create one shared LangGraph execution engine."""

    card = ModelCard(
        model_id="langgraph-concurrent-model",
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

    adapter = LangGraphNodeAdapter(
        graph=build_graph(),
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


def execute_one(engine, word):
    """Execute one workflow with its own ID."""

    strategy = Strategy(
        step_id=f"strategy-{word}",
        round=1,
        model_id="langgraph-concurrent-model",
        halt="stop",
    )

    return engine.execute(
        strategy=strategy,
        input_text=word,
        execution_id=f"execution-{word}",
    )


def run_concurrently(words):
    """Execute several workflows using one shared engine."""

    engine = make_engine()

    with ThreadPoolExecutor(
        max_workers=len(words)
    ) as pool:

        futures = {
            word: pool.submit(
                execute_one,
                engine,
                word,
            )
            for word in words
        }

        return {
            word: future.result(timeout=30)
            for word, future in futures.items()
        }


def test_concurrent_langgraph_outputs():
    """Each execution must return its own output."""

    words = ("apple", "banana", "cherry")

    results = run_concurrently(words)

    assert len(results) == 3

    for word in words:

        result = results[word]

        assert result.output_text == (
            word.upper() + " | " + word.lower()
        )

        assert result.trace.final_status == "ok"


def test_concurrent_langgraph_traces():
    """Node traces must remain independent."""

    words = ("alpha", "beta", "gamma")

    results = run_concurrently(words)

    execution_ids = set()

    for word in words:

        trace = results[word].trace

        validate_execution_trace(trace)

        assert trace.execution_id == (
            f"execution-{word}"
        )

        assert trace.strategy_step_id == (
            f"strategy-{word}"
        )

        assert trace.framework == "langgraph"

        assert len(trace.steps) == 3

        steps = {
            step.node_name: step
            for step in trace.steps
        }

        assert set(steps) == {
            "left",
            "right",
            "join",
        }

        # Parallel branches have no parents.
        assert steps["left"].dependencies == ()
        assert steps["right"].dependencies == ()

        # The join depends on both branches.
        expected_parents = {
            steps["left"].step_index,
            steps["right"].step_index,
        }

        assert set(
            steps["join"].dependencies
        ) == expected_parents

        assert all(
            step.status == "ok"
            for step in trace.steps
        )

        execution_ids.add(trace.execution_id)

    assert len(execution_ids) == 3


def test_concurrent_langgraph_json():
    """Concurrent traces must survive JSON conversion."""

    words = ("first", "second")

    results = run_concurrently(words)

    for word in words:

        original = results[word].trace

        restored = trace_from_dict(
            trace_to_dict(original)
        )

        assert restored == original

        assert restored.execution_id == (
            f"execution-{word}"
        )
