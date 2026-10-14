import time

from controller.contracts import ModelCard, Strategy
from execution.adapters.callable import (
    CallableAdapter,
    whitespace_token_counter,
)
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine


def make_card() -> ModelCard:
    return ModelCard(
        model_id="local-model",
        provider="local",
        tier=0,
        mean_cost_usd=0.0,
        price_source="local test",
        price_date="2026-10-14",
        capability=[0.5, -10.0],
    )


def make_strategy() -> Strategy:
    return Strategy(
        step_id="step-001",
        round=1,
        model_id="local-model",
        halt="stop",
    )


def make_engine(runner) -> ExecutionEngine:
    catalogue = ResourceCatalogue.from_model_cards(
        [make_card()]
    )

    adapter = CallableAdapter(
        runner=runner,
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=adapter,
    )


def test_whitespace_token_counter():
    assert whitespace_token_counter("") == 0
    assert whitespace_token_counter("hello") == 1
    assert whitespace_token_counter("hello adaptive world") == 3


def test_callable_adapter_executes_real_function():
    def runner(text: str) -> str:
        return text.upper()

    engine = make_engine(runner)

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello world",
        execution_id="exec-local-001",
    )

    assert result.output_text == "HELLO WORLD"

    step = result.trace.steps[0]

    assert result.trace.framework == "callable"
    assert result.trace.final_status == "ok"

    assert step.model_id == "local-model"
    assert step.input_tokens == 2
    assert step.output_tokens == 2
    assert step.status == "ok"
    assert step.wall_ms >= 0.0


def test_callable_adapter_measures_wall_time():
    def runner(text: str) -> str:
        time.sleep(0.01)
        return text

    engine = make_engine(runner)

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello",
        execution_id="exec-local-002",
    )

    wall_ms = result.trace.steps[0].wall_ms

    assert wall_ms >= 5.0


def test_callable_adapter_records_exception():
    def failing_runner(text: str) -> str:
        raise RuntimeError("execution failed")

    engine = make_engine(failing_runner)

    result = engine.execute(
        strategy=make_strategy(),
        input_text="hello",
        execution_id="exec-local-003",
    )

    step = result.trace.steps[0]

    assert result.output_text == ""
    assert result.trace.final_status == "error"

    assert step.status == "error"
    assert step.error_type == "RuntimeError"
    assert step.output_tokens == 0


def test_callable_adapter_preserves_execution_identity():
    def runner(text: str) -> str:
        return "done"

    engine = make_engine(runner)

    result = engine.execute(
        strategy=make_strategy(),
        input_text="run",
        execution_id="exec-local-004",
    )

    assert result.trace.execution_id == "exec-local-004"
    assert result.trace.strategy_step_id == "step-001"
    assert result.trace.strategy_round == 1