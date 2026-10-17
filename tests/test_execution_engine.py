
"""Tests for the Pillar 4 framework-independent execution engine.

Covers:
- Successful strategy execution
- Correct model selection
- Unknown model rejection
- Unavailable model rejection
- Empty execution ID rejection
- Invalid adapter execution ID rejection

Updated for Task 12 trace integrity validation.
"""

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


# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------

def make_card(
    model_id: str = "model-a",
) -> ModelCard:
    """Create a model resource for testing."""

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
    """Create a controller strategy for testing."""

    return Strategy(
        step_id="step-001",
        round=1,
        model_id=model_id,
        halt="stop",
    )


# ---------------------------------------------------------
# Fake adapter for successful execution
# ---------------------------------------------------------

class FakeAdapter(ExecutionAdapter):
    """Adapter that returns a valid execution result."""

    framework = "fake"
    adapter_version = "v1"

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:

        execution_step = ExecutionStep(
            step_index=0,
            parent_step=None,
            model_id=request.resource.card.model_id,
            tool=None,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            wall_ms=20.0,
            status="ok",
        )

        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=(execution_step,),
            final_status="ok",
        )

        return AdapterResult(
            output_text="executed output",
            trace=trace,
        )


def make_engine() -> ExecutionEngine:
    """Create an execution engine with a valid test adapter."""

    catalogue = ResourceCatalogue.from_model_cards(
        [make_card()]
    )

    return ExecutionEngine(
        catalogue=catalogue,
        adapter=FakeAdapter(),
    )


# ---------------------------------------------------------
# Test 1: Successful strategy execution
# ---------------------------------------------------------

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
    assert result.trace.final_status == "ok"

    assert len(result.trace.steps) == 1
    assert result.trace.steps[0].model_id == "model-a"


# ---------------------------------------------------------
# Test 2: Correct model selection
# ---------------------------------------------------------

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

    catalogue = ResourceCatalogue.from_model_cards(cards)

    engine = ExecutionEngine(
        catalogue=catalogue,
        adapter=FakeAdapter(),
    )

    result = engine.execute(
        strategy=make_strategy("model-b"),
        input_text="Hello",
        execution_id="exec-002",
    )

    assert result.trace.steps[0].model_id == "model-b"


# ---------------------------------------------------------
# Test 3: Unknown model rejection
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Test 4: Unavailable model rejection
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Test 5: Empty execution ID rejection
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Invalid adapter for execution ID validation
# ---------------------------------------------------------

class BadTraceAdapter(ExecutionAdapter):
    """Adapter that deliberately returns an incorrect execution ID.

    Task 12 update:
    The execution trace must contain a valid execution step.

    The only intentional error is the execution_id mismatch.
    """

    framework = "fake"
    adapter_version = "v1"

    def execute(
        self,
        request: ExecutionRequest,
    ) -> AdapterResult:

        # FIX: Include a valid, nonempty execution step.
        # This allows trace integrity validation to pass.
        execution_step = ExecutionStep(
            step_index=0,
            parent_step=None,
            model_id=request.resource.card.model_id,
            tool=None,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            wall_ms=20.0,
            status="ok",
        )

        trace = ExecutionTrace(
            # Intentionally wrong execution ID for testing.
            execution_id="wrong-id",

            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,

            # FIX: A successful trace cannot be empty.
            steps=(execution_step,),

            final_status="ok",
        )

        return AdapterResult(
            output_text="bad output",
            trace=trace,
        )


# ---------------------------------------------------------
# Test 6: Invalid adapter execution ID rejection
# ---------------------------------------------------------

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
