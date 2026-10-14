import pytest

from controller.contracts import ModelCard, Strategy
from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
)
from execution.catalogue import (
    ResourceCatalogue,
    ResourceEntry,
)
from execution.engine import ExecutionEngine
from execution.trace import (
    ExecutionStep,
    ExecutionTrace,
)


def make_card(
    model_id: str = "model-a",
) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        provider="test",
        tier=0,
        mean_cost_usd=0.001,
        price_source="test",
        price_date="2026-01-01",
        capability=[0.5, -3.0],
    )


def make_strategy(
    model_id: str = "model-a",
) -> Strategy:
    return Strategy(
        step_id="step-001",
        round=1,
        model_id=model_id,
        halt="stop",
    )


class FakeAdapter(ExecutionAdapter):
    framework = "fake"
    adapter_version = "v1"

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=(
                ExecutionStep(
                    step_index=0,
                    parent_step=None,
                    model_id=request.resource.card.model_id,
                    tool=None,
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                    wall_ms=20.0,
                    status="ok",
                ),
            ),
            final_status="ok",
        )

        return AdapterResult(
            output_text="executed output",
            trace=trace,
        )


def make_engine() -> ExecutionEngine:
    catalogue = ResourceCatalogue.from_model_cards(
        [make_card()]
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=FakeAdapter(),
    )


def test_engine_executes_strategy():
    engine = make_engine()

    result = engine.execute(
        strategy=make_strategy(),
        input_text="Hello",
        execution_id="exec-001",
    )

    assert result.output_text == "executed output"
    assert result.trace.execution_id == "exec-001"
    assert result.trace.framework == "fake"
    assert result.trace.steps[0].model_id == "model-a"


def test_engine_resolves_correct_model():
    cards = [
        make_card("model-a"),
        ModelCard(
            model_id="model-b",
            provider="test",
            tier=1,
            mean_cost_usd=0.002,
            price_source="test",
            price_date="2026-01-01",
            capability=[0.6, -2.7],
        ),
    ]

    engine = ExecutionEngine(
        catalogue=ResourceCatalogue.from_model_cards(cards),
        adapter=FakeAdapter(),
    )

    result = engine.execute(
        strategy=make_strategy("model-b"),
        input_text="Hello",
        execution_id="exec-002",
    )

    assert result.trace.steps[0].model_id == "model-b"


def test_engine_rejects_unknown_model():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="unknown model",
    ):
        engine.execute(
            strategy=make_strategy("missing-model"),
            input_text="Hello",
            execution_id="exec-003",
        )


def test_engine_rejects_unavailable_model():
    card = make_card()

    catalogue = ResourceCatalogue(
        resources=(
            ResourceEntry(
                card=card,
                available=False,
            ),
        ),
    )

    engine = ExecutionEngine(
        catalogue=catalogue,
        adapter=FakeAdapter(),
    )

    with pytest.raises(
        ValueError,
        match="unavailable model",
    ):
        engine.execute(
            strategy=make_strategy(),
            input_text="Hello",
            execution_id="exec-004",
        )


def test_engine_rejects_empty_execution_id():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="execution_id cannot be empty",
    ):
        engine.execute(
            strategy=make_strategy(),
            input_text="Hello",
            execution_id="",
        )


class BadTraceAdapter(ExecutionAdapter):
    framework = "fake"
    adapter_version = "v1"

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:
        trace = ExecutionTrace(
            execution_id="wrong-id",
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
        )

        return AdapterResult(
            output_text="bad output",
            trace=trace,
        )


def test_engine_rejects_invalid_adapter_trace():
    catalogue = ResourceCatalogue.from_model_cards(
        [make_card()]
    )

    engine = ExecutionEngine(
        catalogue=catalogue,
        adapter=BadTraceAdapter(),
    )

    with pytest.raises(
        ValueError,
        match="execution_id does not match",
    ):
        engine.execute(
            strategy=make_strategy(),
            input_text="Hello",
            execution_id="exec-005",
        )