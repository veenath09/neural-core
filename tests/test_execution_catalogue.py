import pytest

from controller.contracts import ModelCard, Strategy
from execution.catalogue import (
    ResourceCatalogue,
    ResourceEntry,
    validate_resource_catalogue,
    validate_resource_entry,
)


def make_card(
    model_id: str,
    tier: int,
    cost: float,
) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        provider="test",
        tier=tier,
        mean_cost_usd=cost,
        price_source="test",
        price_date="2026-01-01",
        capability=[0.5, -3.0],
    )


def make_strategy(model_id: str) -> Strategy:
    return Strategy(
        step_id="step-001",
        round=1,
        model_id=model_id,
        halt="stop",
    )


def test_catalogue_builds_from_model_cards():
    cards = [
        make_card("model-a", 0, 0.001),
        make_card("model-b", 1, 0.002),
    ]

    catalogue = ResourceCatalogue.from_model_cards(cards)

    assert catalogue.available_model_ids() == [
        "model-a",
        "model-b",
    ]


def test_strategy_resolves_to_requested_resource():
    cards = [
        make_card("model-a", 0, 0.001),
        make_card("model-b", 1, 0.002),
    ]

    catalogue = ResourceCatalogue.from_model_cards(cards)

    resource = catalogue.resolve(
        make_strategy("model-b")
    )

    assert resource.card.model_id == "model-b"


def test_unavailable_resource_is_rejected():
    card = make_card("model-a", 0, 0.001)

    catalogue = ResourceCatalogue(
        resources=(
            ResourceEntry(
                card=card,
                available=False,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="unavailable model",
    ):
        catalogue.resolve(
            make_strategy("model-a")
        )


def test_unknown_model_is_rejected():
    catalogue = ResourceCatalogue.from_model_cards(
        [make_card("model-a", 0, 0.001)]
    )

    with pytest.raises(
        ValueError,
        match="unknown model",
    ):
        catalogue.resolve(
            make_strategy("missing-model")
        )


def test_duplicate_model_ids_are_rejected():
    catalogue = ResourceCatalogue(
        resources=(
            ResourceEntry(
                card=make_card("model-a", 0, 0.001)
            ),
            ResourceEntry(
                card=make_card("model-a", 1, 0.002)
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate model_id",
    ):
        validate_resource_catalogue(catalogue)


def test_negative_expected_latency_is_rejected():
    resource = ResourceEntry(
        card=make_card("model-a", 0, 0.001),
        expected_latency_ms=-10.0,
    )

    with pytest.raises(
        ValueError,
        match="expected_latency_ms cannot be negative",
    ):
        validate_resource_entry(resource)


def test_available_model_cards_excludes_disabled_resources():
    card_a = make_card("model-a", 0, 0.001)
    card_b = make_card("model-b", 1, 0.002)

    catalogue = ResourceCatalogue(
        resources=(
            ResourceEntry(
                card=card_a,
                available=True,
            ),
            ResourceEntry(
                card=card_b,
                available=False,
            ),
        ),
    )

    cards = catalogue.available_model_cards()

    assert [card.model_id for card in cards] == [
        "model-a",
    ]