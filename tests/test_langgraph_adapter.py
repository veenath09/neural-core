from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from controller.contracts import ModelCard, Strategy
from execution.adapters.langgraph import LangGraphAdapter
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine


class GraphState(TypedDict):
    input_text: str
    output_text: str


def build_uppercase_graph():
    """Build a small real LangGraph workflow."""

    def transform(state: GraphState):
        return {
            "output_text": state["input_text"].upper()
        }

    builder = StateGraph(GraphState)

    builder.add_node(
        "transform",
        transform,
    )

    builder.add_edge(
        START,
        "transform",
    )

    builder.add_edge(
        "transform",
        END,
    )

    return builder.compile()


def build_failing_graph():
    """Build a LangGraph whose node raises an exception."""

    def fail(state: GraphState):
        raise RuntimeError(
            "LangGraph execution failed"
        )

    builder = StateGraph(GraphState)

    builder.add_node(
        "fail",
        fail,
    )

    builder.add_edge(
        START,
        "fail",
    )

    builder.add_edge(
        "fail",
        END,
    )

    return builder.compile()


def make_card() -> ModelCard:
    return ModelCard(
        model_id="langgraph-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local integration test",
        price_date="2026-10-14",
        capability=[0.5, -10.0],
    )


def make_strategy() -> Strategy:
    return Strategy(
        step_id="step-001",
        round=1,
        model_id="langgraph-model",
        halt="stop",
    )


def make_engine(graph) -> ExecutionEngine:
    catalogue = ResourceCatalogue.from_model_cards(
        [make_card()]
    )

    adapter = LangGraphAdapter(
        graph=graph,
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def test_langgraph_adapter_executes_real_graph():
    engine = make_engine(
        build_uppercase_graph()
    )

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello adaptive world",
        execution_id="exec-lg-001",
    )

    assert (
        result.output_text
        == "HELLO ADAPTIVE WORLD"
    )

    assert result.trace.framework == "langgraph"
    assert result.trace.final_status == "ok"


def test_langgraph_trace_preserves_identity():
    engine = make_engine(
        build_uppercase_graph()
    )

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello",
        execution_id="exec-lg-002",
    )

    trace = result.trace

    assert trace.execution_id == "exec-lg-002"
    assert trace.strategy_step_id == "step-001"
    assert trace.strategy_round == 1


def test_langgraph_trace_records_resource():
    engine = make_engine(
        build_uppercase_graph()
    )

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello",
        execution_id="exec-lg-003",
    )

    step = result.trace.steps[0]

    assert step.model_id == "langgraph-model"
    assert step.status == "ok"
    assert step.wall_ms >= 0.0


def test_langgraph_adapter_counts_test_tokens():
    engine = make_engine(
        build_uppercase_graph()
    )

    result = engine.execute(
        strategy=make_strategy(),
        input_text="one two three",
        execution_id="exec-lg-004",
    )

    step = result.trace.steps[0]

    assert step.input_tokens == 3
    assert step.output_tokens == 3


def test_langgraph_adapter_records_framework_error():
    engine = make_engine(
        build_failing_graph()
    )

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello",
        execution_id="exec-lg-005",
    )

    step = result.trace.steps[0]

    assert result.output_text == ""
    assert result.trace.final_status == "error"

    assert step.status == "error"
    assert step.error_type == "RuntimeError"