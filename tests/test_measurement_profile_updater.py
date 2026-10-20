
"""C3-12: Tests for the weighted ProfileUpdater.

Tests cover:
- Required prior weight of 50
- Correct weighted update calculation
- Independent updates for each task type
- Preservation of model cost and metadata
- Original ModelCard immutability
- Multiple aggregated observations
- Invalid inputs and configuration

No external API calls are made.
"""

from copy import deepcopy
import math

import pytest

from controller.contracts import ModelCard
from data.routerbench import SOURCES

from measurement.profile_updater import (
    ProfileUpdater,
    PRIOR_WEIGHT,
    PROFILE_UPDATE_VERSION,
)


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture
def model_card():
    """Return a six-feature C3-03-compatible model card."""

    return ModelCard(
        model_id="test/model-a",
        provider="test",
        tier=1,
        mean_cost_usd=0.002,
        price_source="test-fixture",
        price_date="test-date",
        capability=[
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            math.log10(0.002),
        ],
        version="v1",
    )


# ============================================================
# CONFIGURATION TESTS
# ============================================================

def test_default_prior_weight():
    """Research specification requires w = 50."""

    updater = ProfileUpdater()

    assert PRIOR_WEIGHT == 50.0

    assert updater.prior_weight == 50.0


def test_profile_update_version():
    """The updater must expose a reproducible version."""

    assert (
        PROFILE_UPDATE_VERSION
        == "rb_profile_update_v1"
    )


def test_custom_prior_weight():
    """The updater supports explicit prior strength."""

    updater = ProfileUpdater(
        prior_weight=20,
    )

    assert updater.prior_weight == 20.0


@pytest.mark.parametrize(
    "weight",
    [
        0,
        -1,
        True,
        float("nan"),
        float("inf"),
        "invalid",
    ],
)
def test_rejects_invalid_prior_weight(weight):
    """Invalid prior weights must be rejected."""

    with pytest.raises(ValueError):
        ProfileUpdater(
            prior_weight=weight,
        )


# ============================================================
# WEIGHTED UPDATE FORMULA
# ============================================================

def test_required_weighted_formula(model_card):
    """Validate new = (50*prior + n*observed)/(50+n)."""

    updater = ProfileUpdater()

    updated = updater.update(
        model_card,
        step_type=0,
        observed=0.90,
        n=10,
    )

    expected = (
        50 * 0.70
        + 10 * 0.90
    ) / 60

    assert updated.capability[0] == pytest.approx(
        expected
    )

    assert updated.capability[0] == pytest.approx(
        0.7333333333333333
    )


def test_update_with_zero_observed_accuracy(model_card):
    """A completely unsuccessful probe lowers accuracy."""

    updater = ProfileUpdater()

    updated = updater.update(
        model_card,
        step_type=0,
        observed=0.0,
        n=10,
    )

    expected = (
        50 * 0.70
        + 10 * 0.0
    ) / 60

    assert updated.capability[0] == pytest.approx(
        expected
    )


def test_update_with_perfect_accuracy(model_card):
    """A fully successful probe raises accuracy."""

    updater = ProfileUpdater()

    updated = updater.update(
        model_card,
        step_type=0,
        observed=1.0,
        n=10,
    )

    expected = (
        50 * 0.70
        + 10 * 1.0
    ) / 60

    assert updated.capability[0] == pytest.approx(
        expected
    )


def test_larger_sample_has_greater_influence(model_card):
    """More observations should influence the prior more."""

    updater = ProfileUpdater()

    small = updater.update(
        model_card,
        step_type=0,
        observed=1.0,
        n=5,
    )

    large = updater.update(
        model_card,
        step_type=0,
        observed=1.0,
        n=100,
    )

    assert (
        large.capability[0]
        > small.capability[0]
        > model_card.capability[0]
    )


def test_custom_weight_changes_result(model_card):
    """A smaller prior weight gives observations more influence."""

    default = ProfileUpdater(
        prior_weight=50,
    ).update(
        model_card,
        step_type=0,
        observed=1.0,
        n=10,
    )

    lighter = ProfileUpdater(
        prior_weight=10,
    ).update(
        model_card,
        step_type=0,
        observed=1.0,
        n=10,
    )

    assert (
        lighter.capability[0]
        > default.capability[0]
    )


# ============================================================
# INDEPENDENT TASK-TYPE UPDATES
# ============================================================

def test_only_selected_task_type_changes(model_card):
    """Updating MMLU must not change other task types."""

    updater = ProfileUpdater()

    updated = updater.update(
        model_card,
        step_type=0,
        observed=0.95,
        n=20,
    )

    assert (
        updated.capability[0]
        != model_card.capability[0]
    )

    assert (
        updated.capability[1:]
        == model_card.capability[1:]
    )


def test_each_step_type_can_be_updated(model_card):
    """All five task types must be independently updateable."""

    updater = ProfileUpdater()

    for index in range(len(SOURCES)):

        updated = updater.update(
            model_card,
            step_type=index,
            observed=1.0,
            n=10,
        )

        expected = (
            50 * model_card.capability[index]
            + 10
        ) / 60

        assert updated.capability[index] == pytest.approx(
            expected
        )

        for other in range(len(SOURCES)):

            if other != index:
                assert (
                    updated.capability[other]
                    == model_card.capability[other]
                )


def test_multiple_step_types(model_card):
    """Multiple task types use independent weighted updates."""

    updater = ProfileUpdater()

    updated = updater.update_many(
        model_card,
        {
            0: (0.90, 10),
            2: (0.80, 20),
            4: (0.60, 5),
        },
    )

    assert updated.capability[0] == pytest.approx(
        (50 * 0.70 + 10 * 0.90) / 60
    )

    assert updated.capability[2] == pytest.approx(
        (50 * 0.50 + 20 * 0.80) / 70
    )

    assert updated.capability[4] == pytest.approx(
        (50 * 0.30 + 5 * 0.60) / 55
    )

    # Unobserved types retain their prior values.
    assert updated.capability[1] == 0.60
    assert updated.capability[3] == 0.40


def test_update_and_update_many_match(model_card):
    """Single and multiple update interfaces must agree."""

    updater = ProfileUpdater()

    single = updater.update(
        model_card,
        step_type=1,
        observed=0.75,
        n=30,
    )

    multiple = updater.update_many(
        model_card,
        {
            1: (0.75, 30),
        },
    )

    assert single.capability == pytest.approx(
        multiple.capability
    )


def test_aggregated_observations(model_card):
    """An aggregate must use the combined sample count once.

    Example:
        6 successes in 10 observations
        16 successes in 20 observations

    Combined accuracy = 22 / 30.

    The prior weight of 50 is applied only once.
    """

    updater = ProfileUpdater()

    successes = 6 + 16
    count = 10 + 20

    observed = successes / count

    updated = updater.update(
        model_card,
        step_type=0,
        observed=observed,
        n=count,
    )

    expected = (
        50 * 0.70 + successes
    ) / (
        50 + count
    )

    assert updated.capability[0] == pytest.approx(
        expected
    )


# ============================================================
# MODEL CARD PRESERVATION
# ============================================================

def test_original_card_is_unchanged(model_card):
    """The updater must not modify its input card."""

    original = deepcopy(model_card)

    updater = ProfileUpdater()

    updater.update(
        model_card,
        step_type=0,
        observed=0.90,
        n=10,
    )

    assert model_card == original


def test_updated_card_is_new_object(model_card):
    """An update must return a separate ModelCard."""

    updater = ProfileUpdater()

    updated = updater.update(
        model_card,
        step_type=0,
        observed=0.90,
        n=10,
    )

    assert updated is not model_card

    assert (
        updated.capability
        is not model_card.capability
    )


def test_model_metadata_is_preserved(model_card):
    """Model identity, tier and pricing metadata stay unchanged."""

    updater = ProfileUpdater()

    updated = updater.update(
        model_card,
        step_type=0,
        observed=0.90,
        n=10,
    )

    assert updated.model_id == model_card.model_id
    assert updated.provider == model_card.provider
    assert updated.tier == model_card.tier
    assert updated.price_source == model_card.price_source
    assert updated.price_date == model_card.price_date
    assert updated.version == model_card.version


def test_cost_feature_is_preserved(model_card):
    """Accuracy updates must not modify model pricing."""

    updater = ProfileUpdater()

    updated = updater.update_many(
        model_card,
        {
            0: (0.95, 10),
            1: (0.85, 20),
            2: (0.75, 30),
            3: (0.65, 40),
            4: (0.55, 50),
        },
    )

    assert updated.mean_cost_usd == (
        model_card.mean_cost_usd
    )

    assert updated.capability[5] == (
        model_card.capability[5]
    )


# ============================================================
# INVALID INPUT TESTS
# ============================================================

@pytest.mark.parametrize(
    "step_type",
    [
        -1,
        5,
        100,
        True,
        1.5,
        "mmlu",
    ],
)
def test_rejects_invalid_step_type(
    model_card,
    step_type,
):
    """Only step types 0 through 4 are accepted."""

    updater = ProfileUpdater()

    with pytest.raises(ValueError):
        updater.update(
            model_card,
            step_type=step_type,
            observed=0.8,
            n=10,
        )


@pytest.mark.parametrize(
    "observed",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        True,
        "invalid",
    ],
)
def test_rejects_invalid_observed_accuracy(
    model_card,
    observed,
):
    """Observed accuracies must be finite values in [0, 1]."""

    updater = ProfileUpdater()

    with pytest.raises(ValueError):
        updater.update(
            model_card,
            step_type=0,
            observed=observed,
            n=10,
        )


@pytest.mark.parametrize(
    "count",
    [
        0,
        -1,
        1.5,
        True,
        "10",
    ],
)
def test_rejects_invalid_sample_count(
    model_card,
    count,
):
    """Sample count must be a positive integer."""

    updater = ProfileUpdater()

    with pytest.raises(ValueError):
        updater.update(
            model_card,
            step_type=0,
            observed=0.8,
            n=count,
        )


def test_rejects_empty_observations(model_card):
    """At least one observation is required."""

    updater = ProfileUpdater()

    with pytest.raises(
        ValueError,
        match="At least one observation",
    ):
        updater.update_many(
            model_card,
            {},
        )


def test_rejects_invalid_observations_type(model_card):
    """Observations must be a mapping."""

    updater = ProfileUpdater()

    with pytest.raises(TypeError):
        updater.update_many(
            model_card,
            [(0, (0.8, 10))],
        )


@pytest.mark.parametrize(
    "values",
    [
        (0.8,),
        (0.8, 10, 20),
        "invalid",
    ],
)
def test_rejects_invalid_observation_structure(
    model_card,
    values,
):
    """Each mapping entry must contain (accuracy, count)."""

    updater = ProfileUpdater()

    with pytest.raises(ValueError):
        updater.update_many(
            model_card,
            {
                0: values,
            },
        )


def test_rejects_invalid_capability_dimension(model_card):
    """The updater requires the six-feature C3-03 layout."""

    invalid = deepcopy(model_card)

    invalid.capability = [
        0.5,
        0.5,
    ]

    updater = ProfileUpdater()

    with pytest.raises(
        ValueError,
        match="six capability features",
    ):
        updater.update(
            invalid,
            step_type=0,
            observed=0.8,
            n=10,
        )


def test_rejects_invalid_prior_accuracy(model_card):
    """Prior accuracy must be in [0, 1]."""

    invalid = deepcopy(model_card)

    invalid.capability[0] = 1.5

    updater = ProfileUpdater()

    with pytest.raises(
        ValueError,
        match="Accuracy capabilities",
    ):
        updater.update(
            invalid,
            step_type=0,
            observed=0.8,
            n=10,
        )


def test_rejects_invalid_card_type():
    """The input must be a ModelCard instance."""

    updater = ProfileUpdater()

    with pytest.raises(TypeError):
        updater.update(
            "not-a-card",
            step_type=0,
            observed=0.8,
            n=10,
        )
