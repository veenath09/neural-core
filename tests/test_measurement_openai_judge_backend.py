
"""C3-14: Offline tests for the OpenAI judge backend.

Tests cover:
- Structured judge-model responses
- Provider-reported token usage
- Input and output token pricing
- Incorrect model IDs
- Invalid or missing usage information
- Malformed JSON and invalid quality values
- Explicit client injection
- Compatibility with C3-07 Score

All provider calls use an in-memory fake.
No real API requests or generated-code execution.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from measurement.live_verifiers import (
    JudgeModelVerifier,
    JudgeResult,
)
from measurement.openai_judge_backend import (
    JUDGE_RESPONSE_FORMAT,
    OpenAIJudgeBackend,
)
from measurement.scorer import (
    CompositeScorer,
    Score,
)


MODEL = "fake-judge"

INPUT_PRICE = 0.5
OUTPUT_PRICE = 2.0


class FakeResponses:
    """Capture requests without making network calls."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    """Fake of client.responses.create(...)."""

    def __init__(self, response):
        self.responses = FakeResponses(response)


def make_response(
    *,
    quality=0.8,
    input_tokens=100,
    output_tokens=20,
    model=MODEL,
    status="completed",
    output_text=None,
):
    """Create a simulated provider response."""

    if output_text is None:
        output_text = json.dumps(
            {"quality": quality}
        )

    return SimpleNamespace(
        model=model,
        status=status,
        output_text=output_text,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def make_backend(
    response=None,
    **overrides,
):
    """Return a backend and its fake client."""

    if response is None:
        response = make_response()

    client = FakeClient(response)

    options = {
        "client": client,
        "model": MODEL,
        "input_price_per_million": INPUT_PRICE,
        "output_price_per_million": OUTPUT_PRICE,
    }

    options.update(overrides)

    return OpenAIJudgeBackend(**options), client


def evaluate(backend):
    """Perform one fake open-text evaluation."""

    return backend.judge(
        question="What is a VPC?",
        answer="A logically isolated virtual network.",
        rubric="Evaluate technical correctness.",
        reference="An isolated virtual network.",
    )


def test_successful_judgement():

    backend, client = make_backend()

    result = evaluate(backend)

    assert isinstance(result, JudgeResult)
    assert result.quality == pytest.approx(0.8)
    assert result.cost_usd == pytest.approx(0.00009)
    assert result.latency_ms is None

    assert len(client.responses.calls) == 1


def test_token_cost_calculation():

    response = make_response(
        input_tokens=500,
        output_tokens=200,
    )

    backend, _ = make_backend(response)

    result = evaluate(backend)

    expected = (
        500 * INPUT_PRICE
        + 200 * OUTPUT_PRICE
    ) / 1_000_000

    assert result.cost_usd == pytest.approx(
        expected
    )


def test_zero_token_usage_is_valid():

    response = make_response(
        input_tokens=0,
        output_tokens=0,
    )

    backend, _ = make_backend(response)

    result = evaluate(backend)

    assert result.cost_usd == 0.0


def test_structured_response_format():

    backend, client = make_backend()

    evaluate(backend)

    request = client.responses.calls[0]

    assert request["model"] == MODEL
    assert request["store"] is False

    assert request["text"]["format"] == (
        JUDGE_RESPONSE_FORMAT
    )

    assert request["max_output_tokens"] == 128


def test_question_and_rubric_are_forwarded():

    backend, client = make_backend()

    evaluate(backend)

    messages = client.responses.calls[0]["input"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    payload = json.loads(
        messages[1]["content"]
    )

    assert payload["question"] == "What is a VPC?"

    assert payload["candidate_answer"] == (
        "A logically isolated virtual network."
    )

    assert payload["rubric"] == (
        "Evaluate technical correctness."
    )

    assert payload["reference_answer"] == (
        "An isolated virtual network."
    )


def test_reference_is_optional():

    backend, client = make_backend()

    result = backend.judge(
        question="Explain cloud computing.",
        answer="Computing services delivered over a network.",
        rubric="Evaluate correctness.",
        reference=None,
    )

    payload = json.loads(
        client.responses.calls[0]["input"][1]["content"]
    )

    assert payload["reference_answer"] is None
    assert result.quality == pytest.approx(0.8)


@pytest.mark.parametrize(
    "quality",
    [0.0, 0.25, 0.5, 1.0],
)
def test_valid_quality_values(quality):

    backend, _ = make_backend(
        make_response(quality=quality)
    )

    assert evaluate(backend).quality == pytest.approx(
        quality
    )


@pytest.mark.parametrize(
    "quality",
    [
        -0.1,
        1.1,
        True,
        None,
        "high",
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_quality_rejected(quality):

    backend, _ = make_backend(
        make_response(quality=quality)
    )

    with pytest.raises(RuntimeError):
        evaluate(backend)


@pytest.mark.parametrize(
    "output_text",
    [
        "",
        "not json",
        "[]",
        "{}",
        '{"score": 0.8}',
        '{"quality": 0.8, "extra": true}',
    ],
)
def test_malformed_response_rejected(output_text):

    backend, _ = make_backend(
        make_response(output_text=output_text)
    )

    with pytest.raises(RuntimeError):
        evaluate(backend)


def test_missing_usage_rejected():

    response = make_response()
    response.usage = None

    backend, _ = make_backend(response)

    with pytest.raises(
        RuntimeError,
        match="token usage",
    ):
        evaluate(backend)


@pytest.mark.parametrize(
    "tokens",
    [
        None,
        -1,
        2.5,
        True,
        "100",
    ],
)
def test_invalid_input_tokens_rejected(tokens):

    backend, _ = make_backend(
        make_response(input_tokens=tokens)
    )

    with pytest.raises(
        RuntimeError,
        match="input_tokens",
    ):
        evaluate(backend)


@pytest.mark.parametrize(
    "tokens",
    [
        None,
        -1,
        2.5,
        True,
        "20",
    ],
)
def test_invalid_output_tokens_rejected(tokens):

    backend, _ = make_backend(
        make_response(output_tokens=tokens)
    )

    with pytest.raises(
        RuntimeError,
        match="output_tokens",
    ):
        evaluate(backend)


def test_incorrect_provider_model_rejected():

    backend, _ = make_backend(
        make_response(model="different-model")
    )

    with pytest.raises(
        RuntimeError,
        match="configured judge model",
    ):
        evaluate(backend)


def test_missing_provider_model_rejected():

    backend, _ = make_backend(
        make_response(model=None)
    )

    with pytest.raises(
        RuntimeError,
        match="model ID",
    ):
        evaluate(backend)


@pytest.mark.parametrize(
    "status",
    ["failed", "incomplete", None],
)
def test_incomplete_response_rejected(status):

    backend, _ = make_backend(
        make_response(status=status)
    )

    with pytest.raises(
        RuntimeError,
        match="did not complete",
    ):
        evaluate(backend)


@pytest.mark.parametrize(
    "price",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
        None,
        "0.5",
    ],
)
def test_invalid_input_price_rejected(price):

    with pytest.raises(ValueError):
        make_backend(
            input_price_per_million=price,
        )


@pytest.mark.parametrize(
    "price",
    [
        -0.1,
        float("nan"),
        float("inf"),
        True,
        None,
        "2.0",
    ],
)
def test_invalid_output_price_rejected(price):

    with pytest.raises(ValueError):
        make_backend(
            output_price_per_million=price,
        )


def test_missing_client_rejected():

    with pytest.raises(ValueError):
        make_backend(client=None)


def test_invalid_client_interface_rejected():

    with pytest.raises(TypeError):
        make_backend(client=object())


@pytest.mark.parametrize(
    "model",
    ["", "   ", None, 123],
)
def test_invalid_model_rejected(model):

    with pytest.raises(ValueError):
        make_backend(model=model)


@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
        1.5,
        True,
        "128",
    ],
)
def test_invalid_max_output_tokens_rejected(
    max_tokens,
):

    with pytest.raises(ValueError):
        make_backend(
            max_output_tokens=max_tokens,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", ""),
        ("question", None),
        ("answer", "   "),
        ("answer", 42),
        ("rubric", ""),
        ("rubric", None),
        ("reference", ""),
        ("reference", 123),
    ],
)
def test_invalid_request_fields_rejected(
    field,
    value,
):

    backend, client = make_backend()

    request = {
        "question": "Question",
        "answer": "Answer",
        "rubric": "Rubric",
        "reference": None,
    }

    request[field] = value

    with pytest.raises(ValueError):
        backend.judge(**request)

    # Invalid requests must never reach the provider.
    assert client.responses.calls == []


def test_oversized_question_rejected():

    backend, client = make_backend()

    with pytest.raises(ValueError):
        backend.judge(
            question="Q" * 20001,
            answer="Answer",
            rubric="Rubric",
            reference=None,
        )

    assert client.responses.calls == []


def test_integration_with_existing_score():

    backend, _ = make_backend()

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    verifier = JudgeModelVerifier(
        scorer,
        backend,
    )

    score = verifier.verify(
        question="What is a VPC?",
        answer="A logically isolated virtual network.",
        rubric="Evaluate technical correctness.",
        reference="An isolated virtual network.",
        generation_cost_usd=0.001,
        lam=0.25,
    )

    assert isinstance(score, Score)

    assert score.quality == pytest.approx(0.8)

    assert score.cost_usd == pytest.approx(
        0.00109
    )

    expected_reward = (
        0.8
        - 0.25 * (
            0.00109 / scorer.cfg.cost_ref_usd
        )
    )

    assert score.reward == pytest.approx(
        expected_reward
    )

    assert score.latency_ms is None


def test_no_request_during_construction():

    backend, client = make_backend()

    assert backend.model == MODEL

    assert client.responses.calls == []
