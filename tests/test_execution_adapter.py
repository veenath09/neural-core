import pytest

from controller.contracts import ModelCard, Strategy
from execution.adapters.base import (
    AdapterResult,
    ExecutionAdapter,
    ExecutionRequest,
    validate_adapter_result,
    validate_execution_request,
)
from execution.catalogue import ResourceEntry
from execution.trace import ExecutionStep, ExecutionTrace


def make_card(model_id: str = "model-a") -> ModelCard:
    return ModelCard(
        model_id=model_id,
        provider="test",
        tier=0,
        mean_cost_usd=0.001,
        price_source="test",
        price_date="2026-01-01",
        capability=[0.5, -3.0],
    )


def make_strategy(model_id: str = "model-a") -> Strategy:
    return Strategy(
        step_id="step-001",
        round=1,
        model_id=model_id,
        halt="stop",
    )


def make_request(
    model_id: str = "model-a",
) -> ExecutionRequest:
    strategy = make_strategy(model_id)
    resource = ResourceEntry(
        card=make_card(model_id)
    )

    return ExecutionRequest(
        execution_id="exec-001",
        strategy=strategy,
        resource=resource,
        catalogue_version="v1",
        input_text="Test input",
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
                    input_tokens=2,
                    output_tokens=2,
                    cost_usd=0.001,
                    wall_ms=10.0,
                    status="ok",
                ),
            ),
            final_status="ok",
        )

        return AdapterResult(
            output_text="fake output",
            trace=trace,
        )


def test_valid_execution_request_passes():
    request = make_request()

    validate_execution_request(request)


def test_request_rejects_model_mismatch():
    request = ExecutionRequest(
        execution_id="exec-001",
        strategy=make_strategy("model-a"),
        resource=ResourceEntry(
            card=make_card("model-b")
        ),
        catalogue_version="v1",
        input_text="Test input",
    )

    with pytest.raises(
        ValueError,
        match="strategy model_id does not match",
    ):
        validate_execution_request(request)


def test_request_rejects_unavailable_resource():
    request = ExecutionRequest(
        execution_id="exec-001",
        strategy=make_strategy("model-a"),
        resource=ResourceEntry(
            card=make_card("model-a"),
            available=False,
        ),
        catalogue_version="v1",
        input_text="Test input",
    )

    with pytest.raises(
        ValueError,
        match="resource is unavailable",
    ):
        validate_execution_request(request)


def test_fake_adapter_returns_result():
    adapter = FakeAdapter()
    request = make_request()

    result = adapter.execute(request)

    assert result.output_text == "fake output"
    assert result.trace.framework == "fake"
    assert result.trace.steps[0].model_id == "model-a"


def test_valid_adapter_result_passes():
    adapter = FakeAdapter()
    request = make_request()

    result = adapter.execute(request)

    validate_adapter_result(
        adapter,
        request,
        result,
    )


def test_adapter_result_rejects_wrong_execution_id():
    adapter = FakeAdapter()
    request = make_request()

    bad_trace = ExecutionTrace(
        execution_id="wrong-execution",
        strategy_step_id=request.strategy.step_id,
        strategy_round=request.strategy.round,
        framework=adapter.framework,
        catalogue_version=request.catalogue_version,
        adapter_version=adapter.adapter_version,
    )

    result = AdapterResult(
        output_text="fake output",
        trace=bad_trace,
    )

    with pytest.raises(
        ValueError,
        match="execution_id does not match",
    ):
        validate_adapter_result(
            adapter,
            request,
            result,
        )


def test_adapter_result_rejects_wrong_framework():
    adapter = FakeAdapter()
    request = make_request()

    bad_trace = ExecutionTrace(
        execution_id=request.execution_id,
        strategy_step_id=request.strategy.step_id,
        strategy_round=request.strategy.round,
        framework="wrong-framework",
        catalogue_version=request.catalogue_version,
        adapter_version=adapter.adapter_version,
    )

    result = AdapterResult(
        output_text="fake output",
        trace=bad_trace,
    )

    with pytest.raises(
        ValueError,
        match="framework does not match",
    ):
        validate_adapter_result(
            adapter,
            request,
            result,
        )