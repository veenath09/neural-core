
"""Tests for CrewAI method-level execution tracing."""

from crewai.flow.flow import Flow, start, listen

from controller.contracts import ModelCard, Strategy
from execution.adapters.crewai_methods import CrewAIMethodAdapter
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine
from execution.serialization import trace_from_dict, trace_to_dict


class ResearchFlow(Flow):

    @start()
    def uppercase(self):
        return self.state["input_text"].upper()

    @listen(uppercase)
    def add_exclamation(self, previous_result):
        return previous_result + "!"


class FailingFlow(Flow):

    @start()
    def fail(self):
        raise RuntimeError("Deliberate test failure")


def make_engine(
    flow_factory=ResearchFlow,
    method_parents=None,
):
    card = ModelCard(
        model_id="crewai-method-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-15",
        capability=[0.5, -10.0],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [card]
    )

    if method_parents is None:
        method_parents = {
            "uppercase": None,
            "add_exclamation": "uppercase",
        }

    adapter = CrewAIMethodAdapter(
        flow_factory=flow_factory,
        method_parents=method_parents,
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def execute(
    flow_factory=ResearchFlow,
    method_parents=None,
    execution_id="crewai-method-001",
):
    strategy = Strategy(
        step_id="crewai-method-step",
        round=1,
        model_id="crewai-method-model",
        halt="stop",
    )

    return make_engine(
        flow_factory,
        method_parents,
    ).execute(
        strategy=strategy,
        input_text="hello research",
        execution_id=execution_id,
    )


def test_crewai_method_flow_executes():
    result = execute()

    assert result.output_text == "HELLO RESEARCH!"
    assert result.trace.final_status == "ok"


def test_trace_records_both_methods():
    result = execute()

    assert len(result.trace.steps) == 2

    assert [
        step.node_name
        for step in result.trace.steps
    ] == [
        "uppercase",
        "add_exclamation",
    ]


def test_method_dependencies():
    result = execute()

    first, second = result.trace.steps

    assert first.dependencies == ()
    assert second.dependencies == (0,)


def test_method_durations_recorded():
    result = execute()

    for step in result.trace.steps:
        assert step.wall_ms >= 0.0
        assert step.status == "ok"


def test_method_cost_is_zero_for_local_flow():
    result = execute()

    for step in result.trace.steps:
        assert step.cost_usd == 0.0
        assert step.input_tokens == 0
        assert step.output_tokens == 0


def test_method_trace_serialization():
    result = execute()

    restored = trace_from_dict(
        trace_to_dict(result.trace)
    )

    assert restored == result.trace


def test_method_execution_identity():
    result = execute()

    assert result.trace.execution_id == (
        "crewai-method-001"
    )

    assert result.trace.framework == "crewai"


def test_failing_method_records_error():
    result = execute(
        flow_factory=FailingFlow,
        method_parents={"fail": None},
    )

    assert result.output_text == ""
    assert result.trace.final_status == "error"

    assert any(
        step.status == "error"
        for step in result.trace.steps
    )


def test_missing_dependency_mapping_rejected():
    result = execute(
        method_parents={
            "uppercase": None,
            # add_exclamation deliberately omitted
        },
    )

    assert result.trace.final_status == "error"
    assert result.output_text == ""
