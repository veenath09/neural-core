
"""C3-11 offline tests for routing spending safeguards.

All provider calls are simulated.
No API credentials or network access are required.
"""

from types import SimpleNamespace

import pytest

from controller.contracts import ModelCard
from measurement.llm_router import RouterPricing
from measurement.openai_router_backend import OpenAIResponsesBackend
from measurement.routing_spend_guard import (
    BudgetGuardedLLMRouter,
    RoutingBudgetError,
    RoutingSpendGuard,
)


VALID_RESPONSE = '{"model_id":"model-cheap","halt":"stop"}'


class FakeResponses:
    def __init__(
        self,
        *,
        content=VALID_RESPONSE,
        input_tokens=120,
        output_tokens=20,
        error=None,
        status="completed",
    ):
        self.calls = []
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error = error
        self.status = status

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            output_text=self.content,
            model="test-router-model",
            status=self.status,
            usage=SimpleNamespace(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
        )


def make_router(
    *,
    enabled=True,
    max_total_usd=1.0,
    max_request_usd=0.1,
    max_calls=10,
    budget=1.0,
    content=VALID_RESPONSE,
    error=None,
    input_tokens=120,
    output_tokens=20,
    synthetic=False,
):
    responses = FakeResponses(
        content=content,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    client = SimpleNamespace(responses=responses)

    backend = OpenAIResponsesBackend(
        client=client,
        model="test-router-model",
        max_output_tokens=128,
    )

    pricing = RouterPricing(
        input_usd_per_million=10.0,
        output_usd_per_million=20.0,
        price_source=(
            "synthetic-test-pricing"
            if synthetic
            else "provider-pricing-fixture"
        ),
    )

    cards = [
        ModelCard(
            model_id="model-cheap",
            provider="test",
            tier=0,
            mean_cost_usd=0.001,
            price_source="fixture",
            price_date="fixture",
            capability=[0.3, 0.4],
        ),
        ModelCard(
            model_id="model-strong",
            provider="test",
            tier=1,
            mean_cost_usd=0.008,
            price_source="fixture",
            price_date="fixture",
            capability=[0.8, 0.9],
        ),
    ]

    guard = RoutingSpendGuard(
        max_total_usd=max_total_usd,
        max_request_usd=max_request_usd,
        max_calls=max_calls,
        enabled=enabled,
    )

    router = BudgetGuardedLLMRouter(
        spending_guard=guard,
        cards=cards,
        prompt_text_by_id={
            "p0": "Explain computer networks.",
        },
        backend=backend,
        pricing=pricing,
    )

    task = SimpleNamespace(
        episode_id="test-episode",
        step_id="test-episode:step:0",
        round=0,
        step_emb_key="p0",
        step_type=0,
        budget_left_usd=budget,
        trace=[],
    )

    return router, guard, responses, task


# ------------------------------------------------------------
# Configuration validation
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_total_usd": 0},
        {"max_total_usd": -1},
        {"max_request_usd": 0},
        {"max_request_usd": float("inf")},
        {"max_calls": 0},
        {"max_calls": True},
        {"max_input_bytes": 0},
        {"input_token_multiplier": 0},
        {"message_overhead_tokens": -1},
        {"enabled": "yes"},
    ],
)
def test_invalid_guard_configuration(kwargs):
    options = dict(
        max_total_usd=1.0,
        max_request_usd=0.1,
        max_calls=10,
    )
    options.update(kwargs)

    with pytest.raises(ValueError):
        RoutingSpendGuard(**options)


# ------------------------------------------------------------
# Pre-call blocking
# ------------------------------------------------------------

def test_disabled_guard_blocks_before_api_call():
    router, guard, responses, task = make_router(
        enabled=False
    )

    with pytest.raises(RoutingBudgetError):
        router.decide(task)

    assert responses.calls == []
    assert guard.calls_started == 0


def test_episode_budget_blocks_before_api_call():
    router, guard, responses, task = make_router(
        budget=0.000001
    )

    with pytest.raises(
        RoutingBudgetError,
        match="episode budget",
    ):
        router.decide(task)

    assert responses.calls == []


def test_request_limit_blocks_before_api_call():
    router, guard, responses, task = make_router(
        max_request_usd=0.000001
    )

    with pytest.raises(
        RoutingBudgetError,
        match="per-request",
    ):
        router.decide(task)

    assert responses.calls == []


def test_total_limit_blocks_before_api_call():
    router, guard, responses, task = make_router(
        max_total_usd=0.000001
    )

    with pytest.raises(
        RoutingBudgetError,
        match="evaluation budget",
    ):
        router.decide(task)

    assert responses.calls == []


def test_synthetic_pricing_blocked():
    router, guard, responses, task = make_router(
        synthetic=True
    )

    with pytest.raises(
        RoutingBudgetError,
        match="Synthetic pricing",
    ):
        router.decide(task)

    assert responses.calls == []


def test_maximum_call_count_blocks_second_call():
    router, guard, responses, task = make_router(
        max_calls=1
    )

    router.decide(task)

    with pytest.raises(
        RoutingBudgetError,
        match="Maximum routing call count",
    ):
        router.decide(task)

    assert len(responses.calls) == 1
    assert guard.calls_started == 1


# ------------------------------------------------------------
# Successful accounting
# ------------------------------------------------------------

def test_successful_call_records_provider_cost():
    router, guard, responses, task = make_router()

    strategy = router.decide(task)

    assert strategy.model_id == "model-cheap"
    assert strategy.halt == "stop"

    assert len(responses.calls) == 1
    assert len(router.call_records) == 1

    expected_cost = (
        120 * 10.0 + 20 * 20.0
    ) / 1_000_000

    assert guard.actual_spent_usd == pytest.approx(
        expected_cost
    )
    assert router.total_routing_cost_usd == pytest.approx(
        expected_cost
    )
    assert guard.remaining_run_budget_usd == pytest.approx(
        1.0 - expected_cost
    )


def test_multiple_calls_accumulate_cost():
    router, guard, responses, task = make_router()

    router.decide(task)
    router.decide(task)

    assert len(responses.calls) == 2
    assert len(router.call_records) == 2
    assert guard.calls_started == 2
    assert guard.actual_spent_usd == pytest.approx(
        0.0032
    )


# ------------------------------------------------------------
# Invalid provider responses
# ------------------------------------------------------------

def test_invalid_json_still_charges_routing_cost():
    router, guard, responses, task = make_router(
        content="invalid JSON"
    )

    with pytest.raises(ValueError):
        router.decide(task)

    assert len(responses.calls) == 1
    assert len(router.call_records) == 1
    assert router.call_records[0].valid_choice is False
    assert guard.actual_spent_usd == pytest.approx(
        0.0016
    )
    assert guard.usage_uncertain is False


def test_provider_exception_freezes_future_calls():
    router, guard, responses, task = make_router(
        error=RuntimeError("Simulated timeout")
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated timeout",
    ):
        router.decide(task)

    assert len(responses.calls) == 1
    assert guard.usage_uncertain is True
    assert guard.blocked is True

    with pytest.raises(RoutingBudgetError):
        router.decide(task)

    assert len(responses.calls) == 1


def test_missing_provider_usage_freezes_calls():
    router, guard, responses, task = make_router(
        input_tokens=None
    )

    with pytest.raises(
        RuntimeError,
        match="input_tokens",
    ):
        router.decide(task)

    assert len(responses.calls) == 1
    assert guard.usage_uncertain is True
    assert guard.blocked is True


# ------------------------------------------------------------
# Overspending protection
# ------------------------------------------------------------

def test_actual_cost_above_estimate_blocks_future_calls():
    router, guard, responses, task = make_router(
        input_tokens=10000,
        output_tokens=128,
    )

    router.decide(task)

    assert guard.estimate_exceeded is True
    assert guard.blocked is True
    assert guard.actual_spent_usd == pytest.approx(
        (10000 * 10 + 128 * 20) / 1_000_000
    )

    with pytest.raises(RoutingBudgetError):
        router.decide(task)

    assert len(responses.calls) == 1


def test_oversized_prompt_blocks_before_api_call():
    router, guard, responses, task = make_router()

    guard.max_input_bytes = 10

    with pytest.raises(
        RoutingBudgetError,
        match="maximum byte size",
    ):
        router.decide(task)

    assert responses.calls == []


def test_guard_does_not_call_api_during_initialization():
    router, guard, responses, task = make_router()

    assert responses.calls == []
    assert guard.calls_started == 0
    assert guard.actual_spent_usd == 0.0


def test_guard_starts_disabled_by_default():
    guard = RoutingSpendGuard(
        max_total_usd=1.0,
        max_request_usd=0.1,
        max_calls=10,
    )

    assert guard.enabled is False
    assert guard.calls_started == 0
    assert guard.actual_spent_usd == 0.0
