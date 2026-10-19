
"""C3-11: Offline tests for the OpenAI Responses API router backend.

All tests use an injected fake OpenAI client.

No API key, internet connection, or paid API call
is required.

Tests cover:
- Constructor validation
- OpenAI Responses API request arguments
- Structured JSON configuration
- Provider-reported token usage
- Missing and invalid usage
- Incomplete responses
- Provider exceptions
- Compatibility with LLMRouter
- Routing charges for invalid responses
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from controller.contracts import ModelCard

from measurement.llm_router import (
    LLMRouter,
    RouterPricing,
    RouterResponse,
)

from measurement.openai_router_backend import (
    OpenAIResponsesBackend,
    ROUTING_RESPONSE_FORMAT,
)


# ============================================================
# 1. FAKE OPENAI CLIENT
# ============================================================

VALID_CONTENT = (
    '{"model_id":"model-cheap","halt":"stop"}'
)


def make_response(
    *,
    content=VALID_CONTENT,
    input_tokens=120,
    output_tokens=20,
    model="test-router-model",
    status="completed",
    usage_present=True,
):
    """Create a simulated OpenAI Responses API response."""

    usage = (
        SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if usage_present
        else None
    )

    return SimpleNamespace(
        output_text=content,
        usage=usage,
        model=model,
        status=status,
    )


class FakeResponses:
    """Fake client.responses implementation."""

    def __init__(
        self,
        response=None,
        error=None,
    ):
        self.response = (
            response
            if response is not None
            else make_response()
        )

        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return self.response


class FakeOpenAIClient:
    """An injected fake OpenAI client."""

    def __init__(
        self,
        response=None,
        error=None,
    ):
        self.responses = FakeResponses(
            response=response,
            error=error,
        )


def make_backend(
    response=None,
    error=None,
):
    client = FakeOpenAIClient(
        response=response,
        error=error,
    )

    backend = OpenAIResponsesBackend(
        client=client,
        model="test-router-model",
        max_output_tokens=128,
    )

    return backend, client


def make_messages():
    return [
        {
            "role": "system",
            "content": (
                "Choose a model and return JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Route this programming task."
            ),
        },
    ]


# ============================================================
# 2. CONSTRUCTOR TESTS
# ============================================================

def test_client_is_required():
    with pytest.raises(
        ValueError,
        match="explicit OpenAI client",
    ):
        OpenAIResponsesBackend(
            client=None,
            model="test-router-model",
        )


@pytest.mark.parametrize(
    "model",
    [
        "",
        "   ",
        None,
    ],
)
def test_invalid_model_is_rejected(model):
    with pytest.raises(
        ValueError,
        match="router model ID",
    ):
        OpenAIResponsesBackend(
            client=FakeOpenAIClient(),
            model=model,
        )


@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_max_output_tokens_rejected(
    max_tokens,
):
    with pytest.raises(
        ValueError,
        match="max_output_tokens",
    ):
        OpenAIResponsesBackend(
            client=FakeOpenAIClient(),
            model="test-router-model",
            max_output_tokens=max_tokens,
        )


def test_initialization_makes_no_api_call():
    _, client = make_backend()

    assert client.responses.calls == []


# ============================================================
# 3. REQUEST STRUCTURE
# ============================================================

def test_backend_uses_responses_api():
    backend, client = make_backend()

    result = backend.complete(
        make_messages()
    )

    assert len(client.responses.calls) == 1

    request = client.responses.calls[0]

    assert request["model"] == (
        "test-router-model"
    )

    assert request["input"] == (
        make_messages()
    )

    assert request["max_output_tokens"] == 128

    assert request["store"] is False

    assert isinstance(
        result,
        RouterResponse,
    )


def test_strict_structured_output_requested():
    backend, client = make_backend()

    backend.complete(
        make_messages()
    )

    request = client.responses.calls[0]

    response_format = request[
        "text"
    ]["format"]

    assert response_format == (
        ROUTING_RESPONSE_FORMAT
    )

    assert response_format[
        "type"
    ] == "json_schema"

    assert response_format[
        "strict"
    ] is True

    schema = response_format["schema"]

    assert set(
        schema["required"]
    ) == {
        "model_id",
        "halt",
    }

    assert schema[
        "additionalProperties"
    ] is False


@pytest.mark.parametrize(
    "messages",
    [
        [],
        ["invalid"],
        [
            {
                "role": "assistant",
                "content": "Hello",
            }
        ],
        [
            {
                "role": "user",
                "content": 123,
            }
        ],
    ],
)
def test_invalid_messages_rejected(messages):
    backend, client = make_backend()

    with pytest.raises(ValueError):
        backend.complete(
            messages
        )

    assert client.responses.calls == []


# ============================================================
# 4. PROVIDER USAGE AND RESPONSE HANDLING
# ============================================================

def test_provider_token_usage_is_preserved():
    backend, _ = make_backend(
        response=make_response(
            input_tokens=245,
            output_tokens=37,
        )
    )

    result = backend.complete(
        make_messages()
    )

    assert result.input_tokens == 245

    assert result.output_tokens == 37

    assert result.router_model == (
        "test-router-model"
    )

    assert result.usage_source == "provider"

    assert result.content == VALID_CONTENT


def test_missing_usage_is_rejected():
    backend, _ = make_backend(
        response=make_response(
            usage_present=False,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="no token usage",
    ):
        backend.complete(
            make_messages()
        )


@pytest.mark.parametrize(
    "invalid_tokens",
    [
        None,
        -1,
        1.5,
        True,
    ],
)
def test_invalid_input_usage_rejected(
    invalid_tokens,
):
    backend, _ = make_backend(
        response=make_response(
            input_tokens=invalid_tokens,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="input_tokens",
    ):
        backend.complete(
            make_messages()
        )


@pytest.mark.parametrize(
    "invalid_tokens",
    [
        None,
        -1,
        2.5,
        True,
    ],
)
def test_invalid_output_usage_rejected(
    invalid_tokens,
):
    backend, _ = make_backend(
        response=make_response(
            output_tokens=invalid_tokens,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="output_tokens",
    ):
        backend.complete(
            make_messages()
        )


def test_missing_provider_model_is_rejected():
    backend, _ = make_backend(
        response=make_response(
            model=None,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="missing its model ID",
    ):
        backend.complete(
            make_messages()
        )


def test_incomplete_response_preserves_usage():
    backend, _ = make_backend(
        response=make_response(
            content="partial JSON",
            status="incomplete",
            input_tokens=300,
            output_tokens=80,
        )
    )

    result = backend.complete(
        make_messages()
    )

    assert result.content == ""

    assert result.input_tokens == 300

    assert result.output_tokens == 80

    assert result.usage_source == "provider"


def test_missing_output_text_is_unusable():
    backend, _ = make_backend(
        response=make_response(
            content=None,
        )
    )

    result = backend.complete(
        make_messages()
    )

    assert result.content == ""

    # Token usage is still preserved.
    assert result.input_tokens == 120
    assert result.output_tokens == 20


def test_provider_error_propagates():
    backend, client = make_backend(
        error=RuntimeError(
            "Simulated provider connection failure"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="connection failure",
    ):
        backend.complete(
            make_messages()
        )

    assert len(client.responses.calls) == 1


# ============================================================
# 5. INTEGRATION WITH THE EXISTING LLMROUTER
# ============================================================

def make_router(
    *,
    response=None,
):
    backend, client = make_backend(
        response=response,
    )

    cards = [
        ModelCard(
            model_id="model-cheap",
            provider="test-provider",
            tier=0,
            mean_cost_usd=0.001,
            price_source="synthetic-test",
            price_date="test",
            capability=[0.3, 0.4],
        ),
        ModelCard(
            model_id="model-strong",
            provider="test-provider",
            tier=1,
            mean_cost_usd=0.008,
            price_source="synthetic-test",
            price_date="test",
            capability=[0.8, 0.9],
        ),
    ]

    pricing = RouterPricing(
        input_usd_per_million=10.0,
        output_usd_per_million=20.0,
        price_source="synthetic-test-prices",
    )

    router = LLMRouter(
        cards=cards,
        prompt_text_by_id={
            "p0": "Explain a computer network.",
        },
        backend=backend,
        pricing=pricing,
    )

    return router, client


def make_task():
    return SimpleNamespace(
        episode_id="test-episode",
        step_id="test-episode:step:0",
        round=0,
        step_emb_key="p0",
        step_type=0,
        budget_left_usd=1.0,
        trace=[],
    )


def test_openai_backend_integrates_with_llm_router():
    router, client = make_router()

    strategy = router.decide(
        make_task()
    )

    assert strategy.model_id == "model-cheap"
    assert strategy.halt == "stop"

    assert len(client.responses.calls) == 1

    assert len(router.call_records) == 1

    record = router.call_records[0]

    assert record.router_model == (
        "test-router-model"
    )

    assert record.usage_source == "provider"

    assert record.input_tokens == 120
    assert record.output_tokens == 20

    # 120 input tokens at $10/M
    # plus 20 output tokens at $20/M.
    expected_cost = 0.0016

    assert record.routing_cost_usd == (
        pytest.approx(expected_cost)
    )

    assert record.valid_choice is True

    assert router.total_routing_cost_usd == (
        pytest.approx(expected_cost)
    )


def test_invalid_provider_json_still_records_cost():
    router, _ = make_router(
        response=make_response(
            content="not valid JSON",
        )
    )

    with pytest.raises(ValueError):
        router.decide(
            make_task()
        )

    assert len(router.call_records) == 1

    record = router.call_records[0]

    assert record.valid_choice is False

    assert record.usage_source == "provider"

    assert record.routing_cost_usd == (
        pytest.approx(0.0016)
    )


def test_incomplete_provider_response_still_records_cost():
    router, _ = make_router(
        response=make_response(
            status="incomplete",
            input_tokens=300,
            output_tokens=80,
        )
    )

    with pytest.raises(ValueError):
        router.decide(
            make_task()
        )

    assert len(router.call_records) == 1

    record = router.call_records[0]

    assert record.valid_choice is False

    expected_cost = (
        300 * 10.0
        + 80 * 20.0
    ) / 1_000_000

    assert record.routing_cost_usd == (
        pytest.approx(expected_cost)
    )

    assert record.usage_source == "provider"
