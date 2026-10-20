
"""C3-12: Leave-one-model-out experiment tests.

Verify:
- Deterministic, training-only probe selection
- No overlap between training probes and test prompts
- Correct held-out model capability estimation
- Prior-weighted profile updating
- Original ModelCard preservation
- Correct three-model outcome table
- Capability-aware model selection
- Compatibility with each of the four held-out models

No API requests or neural-controller training.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd
import pytest

from data.routerbench import POOL, SOURCES

from scripts.evaluate import (
    _prepare_inputs,
    make_model_cards,
)

from scripts.evaluate_leave_one_out import (
    ProfileAwareRouter,
    build_onboarded_card,
    select_model_columns,
    select_training_probes,
)


@pytest.fixture(scope="module")
def research_data():
    """Load original research inputs once."""

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    cards = make_model_cards(clean, splits)

    return clean, splits, cards


def test_training_probes_have_five_sources(research_data):
    clean, splits, _ = research_data

    probes = select_training_probes(
        clean,
        splits,
        probes_per_type=10,
    )

    assert len(probes) == 50

    for source in SOURCES:
        assert (
            probes["source"].eq(source).sum()
            == 10
        )


def test_probe_selection_is_training_only(research_data):
    clean, splits, _ = research_data

    probes = select_training_probes(
        clean,
        splits,
        probes_per_type=10,
    )

    assert probes["split"].eq("train").all()


def test_probes_do_not_overlap_test_data(research_data):
    clean, splits, _ = research_data

    probes = select_training_probes(
        clean,
        splits,
        probes_per_type=10,
    )

    test_ids = set(
        splits.loc[
            splits["split"].eq("test"),
            "prompt_id",
        ]
    )

    assert not (
        set(probes["prompt_id"]) & test_ids
    )


def test_probe_selection_is_deterministic(research_data):
    clean, splits, _ = research_data

    original = select_training_probes(
        clean,
        splits,
        probes_per_type=10,
    )

    shuffled = select_training_probes(
        clean.sample(frac=1, random_state=42),
        splits,
        probes_per_type=10,
    )

    assert (
        original["prompt_id"].tolist()
        == shuffled["prompt_id"].tolist()
    )


def test_insufficient_probes_rejected(research_data):
    clean, splits, _ = research_data

    with pytest.raises(
        ValueError,
        match="Insufficient training probes",
    ):
        select_training_probes(
            clean,
            splits,
            probes_per_type=100000,
        )


@pytest.mark.parametrize(
    "held_out_index",
    [0, 1, 2, 3],
)
def test_each_model_can_be_onboarded(
    research_data,
    held_out_index,
):
    """Check all four leave-one-model-out folds."""

    clean, splits, cards = research_data

    held_out = cards[held_out_index]

    existing = [
        card
        for index, card in enumerate(cards)
        if index != held_out_index
    ]

    original = deepcopy(held_out)

    probes = select_training_probes(
        clean,
        splits,
        probes_per_type=10,
    )

    onboarded, observed = build_onboarded_card(
        held_out=held_out,
        existing_cards=existing,
        probes=probes,
    )

    assert onboarded.model_id == held_out.model_id
    assert onboarded.provider == held_out.provider

    assert len(onboarded.capability) == 6
    assert len(observed) == 6

    # Onboarding must not modify the original card.
    assert held_out == original

    # Verify that w=50 is applied independently
    # for each of the five task types.
    for index, source in enumerate(SOURCES):

        prior = sum(
            card.capability[index]
            for card in existing
        ) / 3

        count = int(
            probes["source"].eq(source).sum()
        )

        expected = (
            50 * prior
            + count * observed[index]
        ) / (
            50 + count
        )

        assert onboarded.capability[index] == (
            pytest.approx(expected)
        )

    # The cost feature must represent the
    # observed training-probe mean cost.
    assert onboarded.capability[5] == (
        pytest.approx(observed[5])
    )

    expected_cost = float(
        probes[f"{held_out.model_id}_cost"].mean()
    )

    assert onboarded.mean_cost_usd == (
        pytest.approx(expected_cost)
    )


def test_onboarding_requires_three_existing_models(
    research_data,
):
    clean, splits, cards = research_data

    probes = select_training_probes(
        clean,
        splits,
        probes_per_type=10,
    )

    with pytest.raises(
        ValueError,
        match="three existing",
    ):
        build_onboarded_card(
            held_out=cards[3],
            existing_cards=cards[:2],
            probes=probes,
        )


def test_three_model_outcome_table(research_data):
    clean, splits, _ = research_data

    full_table, _ = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    indices = [0, 1, 2]

    subset = select_model_columns(
        full_table,
        indices,
    )

    assert subset.correct.shape[1] == 3
    assert subset.cost.shape[1] == 3

    np.testing.assert_array_equal(
        subset.correct,
        full_table.correct[:, indices],
    )

    np.testing.assert_array_equal(
        subset.cost,
        full_table.cost[:, indices],
    )

    np.testing.assert_array_equal(
        subset.prompt_ids,
        full_table.prompt_ids,
    )


@pytest.fixture
def policy_cards(research_data):
    """Create two cards with controlled capabilities."""

    _, _, cards = research_data

    cheap = replace(
        cards[0],
        mean_cost_usd=0.001,
        capability=[
            0.20, 0.20, 0.20, 0.20, 0.20,
            math.log10(0.001),
        ],
    )

    strong = replace(
        cards[1],
        mean_cost_usd=0.002,
        capability=[
            0.95, 0.95, 0.95, 0.95, 0.95,
            math.log10(0.002),
        ],
    )

    return cheap, strong


def make_task(budget):
    return SimpleNamespace(
        step_id="test-step",
        round=0,
        step_type=0,
        budget_left_usd=budget,
    )


def test_profile_router_prefers_capability(policy_cards):
    cheap, strong = policy_cards

    router = ProfileAwareRouter(
        [cheap, strong],
        cost_penalty=0.1,
    )

    decision = router.decide(
        make_task(1.0)
    )

    assert decision.model_id == strong.model_id
    assert decision.halt == "stop"


def test_profile_router_respects_budget(policy_cards):
    cheap, strong = policy_cards

    router = ProfileAwareRouter(
        [cheap, strong],
        cost_penalty=0.1,
    )

    decision = router.decide(
        make_task(0.0015)
    )

    assert decision.model_id == cheap.model_id


def test_profile_router_uses_cheapest_fallback(
    policy_cards,
):
    cheap, strong = policy_cards

    router = ProfileAwareRouter(
        [cheap, strong],
        cost_penalty=0.1,
    )

    decision = router.decide(
        make_task(0.0)
    )

    assert decision.model_id == cheap.model_id


def test_profile_router_rejects_invalid_step_type(
    policy_cards,
):
    router = ProfileAwareRouter(
        list(policy_cards)
    )

    task = make_task(1.0)
    task.step_type = 99

    with pytest.raises(
        ValueError,
        match="Unknown task type",
    ):
        router.decide(task)


def test_four_model_pool_matches_original_order(
    research_data,
):
    _, _, cards = research_data

    assert len(cards) == 4

    assert [
        card.model_id
        for card in cards
    ] == list(POOL)
