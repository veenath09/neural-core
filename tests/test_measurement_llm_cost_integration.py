
"""C3-11: LLM routing-cost integration tests.

All tests use offline, synthetic router responses.
No provider APIs or live charges are involved.
"""

from __future__ import annotations

import pandas as pd
import pytest

from measurement.baselines import AlwaysCheapest
from measurement.evaluate import (
    evaluate_episode,
    evaluate_policies,
)
from measurement.llm_router import (
    LLMRouter,
    MockLLMBackend,
    RouterPricing,
    RouterResponse,
)
from measurement.scorer import CompositeScorer

from tests.test_measurement_baselines import (
    make_environment,
    make_pool,
)


@pytest.fixture
def scorer():
    return CompositeScorer.from_yaml(
        "conf/score.yaml"
    )


def make_router(backend=None):
    if backend is None:
        backend = MockLLMBackend(
            input_tokens=120,
            output_tokens=20,
        )

    return LLMRouter(
        cards=make_pool(),
        prompt_text_by_id={
            "p0": "Astronomy science question",
            "p1": "Programming question",
            "p2": "General knowledge question",
        },
        backend=backend,
        pricing=RouterPricing(
            input_usd_per_million=10.0,
            output_usd_per_million=20.0,
            price_source="synthetic-test-prices",
        ),
    )


def run_llm(scorer, *, budget=100.0, backend=None):
    episode, _ = make_environment(
        budget=budget,
    )

    router = make_router(backend)

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        router,
        policy_name="llm-router-mock",
        scorer=scorer,
        flip_p=0.0,
    )

    return frame, router


def test_routing_cost_is_recorded_per_decision(scorer):
    frame, router = run_llm(scorer)

    assert len(frame) == 3
    assert len(router.call_records) == 3

    assert frame["routing_cost_usd"].tolist() == (
        pytest.approx([0.0016] * 3)
    )


def test_routing_and_execution_costs_are_separate(scorer):
    frame, _ = run_llm(scorer)

    final = frame.iloc[-1]

    assert final["episode_routing_cost_usd"] == (
        pytest.approx(0.0048)
    )

    assert final["episode_execution_cost_usd"] == (
        pytest.approx(3.0)
    )

    assert final["episode_cost_usd"] == (
        pytest.approx(3.0048)
    )


def test_final_reward_includes_routing_cost(scorer):
    frame, _ = run_llm(scorer)

    final = frame.iloc[-1]

    expected = scorer.score(
        quality=final["episode_quality"],
        cost_usd=3.0048,
        lam=0.25,
        mu=0.0,
    )

    assert final["episode_reward"] == (
        pytest.approx(expected.reward)
    )


def test_mock_usage_source_is_preserved(scorer):
    frame, _ = run_llm(scorer)

    assert set(frame["routing_usage_source"]) == {
        "mock"
    }

    assert frame["routing_input_tokens"].tolist() == [
        120, 120, 120
    ]

    assert frame["routing_output_tokens"].tolist() == [
        20, 20, 20
    ]


def test_non_llm_baseline_keeps_original_cost(scorer):
    episode, _ = make_environment()

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        AlwaysCheapest(make_pool()),
        policy_name="always-cheapest",
        scorer=scorer,
        flip_p=0.0,
    )

    final = frame.iloc[-1]

    assert frame["routing_cost_usd"].eq(0.0).all()

    assert final["episode_routing_cost_usd"] == 0.0

    assert final["episode_execution_cost_usd"] == (
        pytest.approx(3.0)
    )

    assert final["episode_cost_usd"] == (
        pytest.approx(3.0)
    )


def test_routing_charge_can_exhaust_execution_budget(
    scorer,
):
    # The first model costs $1 in this fixture.
    # Its execution is affordable before routing,
    # but unaffordable after the routing charge.
    frame, _ = run_llm(
        scorer,
        budget=1.001,
    )

    assert len(frame) == 1

    row = frame.iloc[0]

    assert row["status"] == "budget_exhausted"
    assert bool(row["executed"]) is False

    assert row["routing_cost_usd"] == (
        pytest.approx(0.0016)
    )

    assert row["execution_cost_usd"] == 0.0

    assert row["episode_cost_usd"] == (
        pytest.approx(0.0016)
    )


class InvalidJSONBackend:
    def complete(self, messages):
        return RouterResponse(
            content="not valid JSON",
            input_tokens=120,
            output_tokens=20,
            router_model="invalid-json-mock",
            usage_source="mock",
        )


def test_invalid_router_response_still_costs_money(
    scorer,
):
    frame, router = run_llm(
        scorer,
        backend=InvalidJSONBackend(),
    )

    assert len(frame) == 1

    row = frame.iloc[0]

    assert row["status"] == "router_invalid_response"
    assert bool(row["terminal"]) is True
    assert bool(row["executed"]) is False

    assert row["model_id"] is None

    assert row["routing_valid_choice"] == False

    assert row["execution_cost_usd"] == 0.0

    assert row["episode_routing_cost_usd"] == (
        pytest.approx(0.0016)
    )

    assert row["episode_cost_usd"] == (
        pytest.approx(0.0016)
    )

    assert len(router.call_records) == 1


def test_routing_budget_overrun_is_explicit(
    scorer,
):
    frame, _ = run_llm(
        scorer,
        budget=0.0005,
    )

    assert len(frame) == 1

    row = frame.iloc[0]

    assert row["status"] == "routing_budget_overrun"
    assert bool(row["terminal"]) is True
    assert bool(row["executed"]) is False

    # Full synthetic invoice is recorded.
    assert row["episode_cost_usd"] == (
        pytest.approx(0.0016)
    )

    # This is a disclosed overrun, not hidden spending.
    assert row["budget_left_after_usd"] == (
        pytest.approx(-0.0011)
    )

    assert row["routing_budget_overrun_usd"] == (
        pytest.approx(0.0016)
    )


def test_reused_router_keeps_episode_costs_separate(
    scorer,
):
    first, _ = make_environment(
        episode_id="episode-one"
    )

    second, _ = make_environment(
        episode_id="episode-two"
    )

    router = make_router()

    frame = evaluate_policies(
        [first.skeleton, second.skeleton],
        first.table,
        first.model_ids,
        {"llm-router-mock": router},
        scorer=scorer,
        flip_p=0.0,
    )

    terminal = frame.loc[frame["terminal"]]

    assert len(terminal) == 2

    assert terminal[
        "episode_routing_cost_usd"
    ].tolist() == pytest.approx(
        [0.0048, 0.0048]
    )

    assert len(router.call_records) == 6


def test_router_cost_records_survive_parquet(
    tmp_path,
    scorer,
):
    frame, _ = run_llm(scorer)

    path = tmp_path / "predictions.parquet"

    frame.to_parquet(
        path,
        index=False,
    )

    restored = pd.read_parquet(path)

    assert restored["routing_cost_usd"].sum() == (
        pytest.approx(0.0048)
    )

    assert restored.iloc[-1][
        "episode_cost_usd"
    ] == pytest.approx(3.0048)
