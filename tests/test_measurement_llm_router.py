
"""C3-11: Tests for the LLM-as-Router baseline.

Tests cover:
- Model card and prompt construction
- Execution trace inclusion
- Deterministic offline mock routing
- JSON response validation
- Routing token and cost accounting
- Invalid responses and audit records

No live LLM API calls are made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from controller.contracts import ModelCard

from measurement.llm_router import (
    LLMRouter,
    MockLLMBackend,
    RouterPricing,
    RouterResponse,
)


@pytest.fixture
def cards():
    return [
        ModelCard(
            model_id="model-cheap",
            provider="test-provider",
            tier=0,
            mean_cost_usd=0.001,
            price_source="test-fixture",
            price_date="2026-10-19",
            capability=[0.3, 0.4],
        ),
        ModelCard(
            model_id="model-strong",
            provider="test-provider",
            tier=1,
            mean_cost_usd=0.008,
            price_source="test-fixture",
            price_date="2026-10-19",
            capability=[0.8, 0.9],
        ),
    ]


@pytest.fixture
def pricing():
    return RouterPricing(
        input_usd_per_million=10.0,
        output_usd_per_million=20.0,
        price_source="synthetic-test-prices",
    )


@pytest.fixture
def backend():
    return MockLLMBackend(
        input_tokens=120,
        output_tokens=20,
    )


@pytest.fixture
def router(cards, pricing, backend):
    return LLMRouter(
        cards=cards,
        prompt_text_by_id={
            "p0": "Explain how a computer network works.",
            "p1": "Write a Python sorting function.",
        },
        backend=backend,
        pricing=pricing,
    )


def make_task(
    *,
    prompt_id="p0",
    budget=1.0,
    round_number=0,
    trace=None,
):
    return SimpleNamespace(
        episode_id="test-episode",
        step_id="test-episode:step:0",
        round=round_number,
        step_emb_key=prompt_id,
        step_type=0,
        budget_left_usd=budget,
        trace=[] if trace is None else trace,
    )


class StaticBackend:
    """Offline fake backend returning a specified response."""

    def __init__(
        self,
        content,
        *,
        input_tokens=100,
        output_tokens=10,
    ):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def complete(self, messages):
        self.calls += 1

        return RouterResponse(
            content=self.content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            router_model="offline-test-router",
            usage_source="mock",
        )


def make_static_router(cards, pricing, content):
    return LLMRouter(
        cards=cards,
        prompt_text_by_id={
            "p0": "Explain how a computer network works.",
        },
        backend=StaticBackend(content),
        pricing=pricing,
    )


def test_router_initialization(router):
    assert router.model_ids == (
        "model-cheap",
        "model-strong",
    )

    assert router.call_records == []
    assert router.total_routing_cost_usd == 0.0


def test_build_messages_includes_prompt(router):
    messages = router.build_messages(
        make_task()
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    request = json.loads(
        messages[1]["content"]
    )

    assert request["prompt_id"] == "p0"

    assert request["prompt_text"] == (
        "Explain how a computer network works."
    )


def test_build_messages_includes_model_cards(router):
    messages = router.build_messages(
        make_task()
    )

    request = json.loads(
        messages[1]["content"]
    )

    model_cards = request["model_cards"]

    assert len(model_cards) == 2

    assert model_cards[0]["model_id"] == "model-cheap"
    assert model_cards[1]["model_id"] == "model-strong"

    assert model_cards[0]["capability"] == [0.3, 0.4]


def test_build_messages_includes_budget(router):
    messages = router.build_messages(
        make_task(budget=0.005)
    )

    request = json.loads(
        messages[1]["content"]
    )

    assert request["budget_left_usd"] == pytest.approx(
        0.005
    )


def test_build_messages_includes_execution_trace(router):
    previous = SimpleNamespace(
        model_id="model-cheap",
        round=0,
        verifier_ok=False,
        cost_usd=0.001,
        escalated=True,
        step_index=0,
        step_emb_key="p0",
    )

    messages = router.build_messages(
        make_task(trace=[previous])
    )

    request = json.loads(
        messages[1]["content"]
    )

    assert len(request["recent_trace"]) == 1

    assert (
        request["recent_trace"][0]["model_id"]
        == "model-cheap"
    )

    assert request["recent_trace"][0]["verifier_ok"] is False


def test_trace_window_is_respected(cards, pricing):
    router = LLMRouter(
        cards=cards,
        prompt_text_by_id={"p0": "Test prompt"},
        backend=MockLLMBackend(),
        pricing=pricing,
        max_trace_items=2,
    )

    trace = [
        SimpleNamespace(
            model_id="model-cheap",
            round=0,
            verifier_ok=False,
            cost_usd=0.001,
            escalated=False,
            step_index=i,
            step_emb_key=f"previous-{i}",
        )
        for i in range(5)
    ]

    messages = router.build_messages(
        make_task(trace=trace)
    )

    request = json.loads(
        messages[1]["content"]
    )

    assert len(request["recent_trace"]) == 2

    assert [
        item["step_index"]
        for item in request["recent_trace"]
    ] == [3, 4]


def test_mock_backend_selects_cheapest_model(router, backend):
    strategy = router.decide(
        make_task(budget=1.0)
    )

    assert strategy.model_id == "model-cheap"
    assert strategy.halt == "stop"

    assert backend.calls == 1


def test_mock_backend_respects_budget(router):
    strategy = router.decide(
        make_task(budget=0.002)
    )

    assert strategy.model_id == "model-cheap"


def test_mock_backend_handles_insufficient_budget(router):
    strategy = router.decide(
        make_task(budget=0.0)
    )

    # An unaffordable attempt is returned.
    # The simulator must enforce budget exhaustion.
    assert strategy.model_id == "model-cheap"


def test_valid_strong_model_selection(cards, pricing):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "model-strong", "halt": "stop"}',
    )

    strategy = router.decide(
        make_task()
    )

    assert strategy.model_id == "model-strong"
    assert strategy.halt == "stop"


def test_valid_escalation_selection(cards, pricing):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "model-strong", "halt": "escalate"}',
    )

    strategy = router.decide(
        make_task(round_number=0)
    )

    assert strategy.halt == "escalate"


def test_escalation_after_first_round_is_rejected(
    cards,
    pricing,
):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "model-strong", "halt": "escalate"}',
    )

    with pytest.raises(
        ValueError,
        match="Invalid LLM routing response",
    ):
        router.decide(
            make_task(round_number=1)
        )

    assert len(router.call_records) == 1
    assert router.call_records[0].valid_choice is False


def test_invalid_json_is_rejected_and_recorded(
    cards,
    pricing,
):
    router = make_static_router(
        cards,
        pricing,
        "this is not JSON",
    )

    with pytest.raises(
        ValueError,
        match="Invalid LLM routing response",
    ):
        router.decide(make_task())

    assert len(router.call_records) == 1

    record = router.call_records[0]

    assert record.valid_choice is False
    assert record.selected_model_id is None
    assert record.routing_cost_usd > 0


def test_unknown_model_id_is_rejected(cards, pricing):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "unknown-model", "halt": "stop"}',
    )

    with pytest.raises(
        ValueError,
        match="unknown model",
    ):
        router.decide(make_task())

    assert router.call_records[0].valid_choice is False


def test_missing_response_field_is_rejected(
    cards,
    pricing,
):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "model-cheap"}',
    )

    with pytest.raises(
        ValueError,
        match="exactly",
    ):
        router.decide(make_task())


def test_extra_response_field_is_rejected(
    cards,
    pricing,
):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "model-cheap", "halt": "stop", "extra": 1}',
    )

    with pytest.raises(
        ValueError,
        match="exactly",
    ):
        router.decide(make_task())


def test_invalid_halt_is_rejected(cards, pricing):
    router = make_static_router(
        cards,
        pricing,
        '{"model_id": "model-cheap", "halt": "retry"}',
    )

    with pytest.raises(
        ValueError,
        match="Invalid router halt",
    ):
        router.decide(make_task())


def test_routing_cost_calculation(pricing):
    cost = pricing.calculate(
        input_tokens=120,
        output_tokens=20,
    )

    # (120 * 10 + 20 * 20) / 1_000_000
    assert cost == pytest.approx(0.0016)


def test_routing_call_record_is_created(router):
    router.decide(make_task())

    assert len(router.call_records) == 1

    record = router.call_records[0]

    assert record.episode_id == "test-episode"
    assert record.step_id == "test-episode:step:0"
    assert record.input_tokens == 120
    assert record.output_tokens == 20
    assert record.router_model == "mock-llm-router"
    assert record.usage_source == "mock"

    assert record.selected_model_id == "model-cheap"
    assert record.valid_choice is True

    assert record.routing_cost_usd == pytest.approx(
        0.0016
    )


def test_multiple_calls_accumulate_routing_cost(router):
    router.decide(make_task(prompt_id="p0"))
    router.decide(make_task(prompt_id="p1"))

    assert len(router.call_records) == 2

    assert router.total_routing_cost_usd == pytest.approx(
        0.0032
    )


def test_invalid_token_counts_are_rejected(pricing):
    with pytest.raises(ValueError):
        pricing.calculate(-1, 20)

    with pytest.raises(ValueError):
        pricing.calculate(100, -1)


def test_negative_token_price_is_rejected():
    with pytest.raises(ValueError):
        RouterPricing(
            input_usd_per_million=-1.0,
            output_usd_per_million=1.0,
            price_source="test",
        )


def test_missing_prompt_is_rejected(router):
    with pytest.raises(
        KeyError,
        match="Unknown prompt ID",
    ):
        router.build_messages(
            make_task(prompt_id="missing")
        )


def test_mock_calls_are_not_provider_calls(router):
    router.decide(make_task())

    assert all(
        record.usage_source == "mock"
        for record in router.call_records
    )


def test_no_confidence_is_claimed(router):
    assert router.provides_confidence is False
