
"""Integration tests for the CrewAI execution adapter."""

from crewai.flow.flow import Flow, start, listen

from controller.contracts import ModelCard, Strategy
from execution.adapters.crewai import CrewAIAdapter
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine
from execution.serialization import (
    trace_from_dict,
    trace_to_dict,
)


class ResearchFlow(Flow):
    """A real two-method CrewAI Flow without LLM calls."""

    @start()
    def uppercase(self):
        return self.state["input_text"].upper()

    @listen(uppercase)
    def add_exclamation(self, previous_result):
        return previous_result + "!"


class FailingFlow(Flow):
    """CrewAI Flow that deliberately fails."""

    @start()
    def fail(self):
        raise RuntimeError("CrewAI test failure")


def make_engine(
    flow_factory=ResearchFlow,
):
    card = ModelCard(
        model_id="crewai-local-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local integration test",
        price_date="2026-10-14",
        capability=[0.5, -10.0],
    )

    catalogue = ResourceCatalogue.from_model_cards(
        [card]
    )

    adapter = CrewAIAdapter(
        flow_factory=flow_factory,
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def make_strategy():
    return Strategy(
        step_id="crewai-step-001",
        round=1,
        model_id="crewai-local-model",
        halt="stop",
    )


def execute_flow(
    flow_factory=ResearchFlow,
    execution_id="exec-crewai-001",
):
    engine = make_engine(flow_factory)

    return engine.execute(
        strategy=make_strategy(),
        input_text="hello research",
        execution_id=execution_id,
    )


def test_crewai_flow_executes():
    result = execute_flow()

    assert result.output_text == "HELLO RESEARCH!"
    assert result.trace.final_status == "ok"


def test_crewai_trace_identity():
    result = execute_flow()
    trace = result.trace

    assert trace.execution_id == "exec-crewai-001"
    assert trace.strategy_step_id == "crewai-step-001"
    assert trace.strategy_round == 1
    assert trace.framework == "crewai"


def test_crewai_records_execution_step():
    result = execute_flow()

    assert len(result.trace.steps) == 1

    step = result.trace.steps[0]

    assert step.model_id == "crewai-local-model"
    assert step.status == "ok"
    assert step.wall_ms >= 0.0
    assert step.dependencies == ()


def test_local_flow_has_no_llm_cost():
    result = execute_flow()
    step = result.trace.steps[0]

    assert step.cost_usd == 0.0
    assert step.input_tokens == 0
    assert step.output_tokens == 0


def test_crewai_error_is_recorded():
    result = execute_flow(
        flow_factory=FailingFlow,
    )

    assert result.output_text == ""
    assert result.trace.final_status == "error"

    step = result.trace.steps[0]

    assert step.status == "error"
    assert step.error_type is not None


def test_crewai_trace_is_serializable():
    result = execute_flow()

    data = trace_to_dict(result.trace)
    restored = trace_from_dict(data)

    assert restored == result.trace
    assert restored.framework == "crewai"


def test_separate_executions_have_separate_ids():
    engine = make_engine()

    first = engine.execute(
        strategy=make_strategy(),
        input_text="first",
        execution_id="crewai-run-1",
    )

    second = engine.execute(
        strategy=make_strategy(),
        input_text="second",
        execution_id="crewai-run-2",
    )

    assert first.output_text == "FIRST!"
    assert second.output_text == "SECOND!"

    assert first.trace.execution_id != (
        second.trace.execution_id
    )
