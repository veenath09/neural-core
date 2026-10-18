
"""Pillar 3 C3-07: Tests for the Composite Scorer.

Tests the reward formula, cost normalization,
latency handling, frozen configuration and
invalid scoring inputs.
"""

import math

import pytest

from measurement.scorer import (
    CompositeScorer,
    Score,
    ScoreConfig,
)


COST_REF = 0.001756663098


def valid_config(**overrides):
    """Create a valid scoring configuration."""

    values = {
        "version": 1,
        "cost_ref_usd": COST_REF,
        "cost_ref_model": "gpt-4-1106-preview",
        "cost_ref_split": "train",
        "cost_ref_statistic": "mean_per_step_cost",
        "train_prompt_count": 19614,
        "latency_ref_ms": None,
        "latency_source": "unavailable_in_routerbench",
    }

    values.update(overrides)
    return values


@pytest.fixture
def scorer():
    return CompositeScorer.from_yaml(
        "conf/score.yaml"
    )


def test_score_config_loads_from_yaml(scorer):
    config = scorer.cfg

    assert config.cost_ref_usd == pytest.approx(
        COST_REF
    )
    assert config.cost_ref_model == (
        "gpt-4-1106-preview"
    )
    assert config.cost_ref_split == "train"
    assert config.cost_ref_statistic == (
        "mean_per_step_cost"
    )
    assert config.train_prompt_count == 19614
    assert config.latency_ref_ms is None
    assert config.version == 1


def test_configuration_can_be_loaded_from_mapping():
    config = ScoreConfig.from_mapping(
        valid_config()
    )

    assert config.cost_ref_usd == pytest.approx(
        COST_REF
    )
    assert config.cost_ref_split == "train"


def test_composite_scorer_accepts_mapping():
    scorer = CompositeScorer(valid_config())

    result = scorer.score(
        quality=1.0,
        cost_usd=COST_REF,
        lam=0.2,
    )

    assert result.reward == pytest.approx(0.8)


def test_composite_scorer_accepts_config_object():
    config = ScoreConfig.from_mapping(
        valid_config()
    )

    scorer = CompositeScorer(config)

    assert scorer.cfg is config


def test_score_returns_score_contract(scorer):
    result = scorer.score(
        quality=0.8,
        cost_usd=COST_REF,
        lam=0.25,
        penalty=0.1,
    )

    assert isinstance(result, Score)
    assert result.reward == pytest.approx(0.45)


def test_reward_with_only_quality(scorer):
    result = scorer.score(
        quality=0.75,
        cost_usd=0.0,
    )

    assert result.reward == pytest.approx(0.75)
    assert result.normalized_cost == 0.0
    assert result.penalty == 0.0


def test_cost_reference_normalizes_to_one(scorer):
    result = scorer.score(
        quality=1.0,
        cost_usd=COST_REF,
        lam=0.3,
    )

    assert result.normalized_cost == pytest.approx(
        1.0
    )
    assert result.reward == pytest.approx(0.7)


def test_double_reference_cost_normalizes_to_two(scorer):
    result = scorer.score(
        quality=1.0,
        cost_usd=2 * COST_REF,
        lam=0.2,
    )

    assert result.normalized_cost == pytest.approx(
        2.0
    )
    assert result.reward == pytest.approx(0.6)


def test_reward_formula_includes_penalty(scorer):
    result = scorer.score(
        quality=0.8,
        cost_usd=COST_REF,
        lam=0.25,
        penalty=0.1,
    )

    expected = 0.8 - 0.25 - 0.1

    assert result.reward == pytest.approx(expected)


def test_penalty_is_subtracted_once(scorer):
    without_penalty = scorer.score(
        quality=0.9,
        cost_usd=COST_REF,
        lam=0.1,
    )

    with_penalty = scorer.score(
        quality=0.9,
        cost_usd=COST_REF,
        lam=0.1,
        penalty=0.2,
    )

    assert (
        without_penalty.reward
        - with_penalty.reward
    ) == pytest.approx(0.2)


def test_zero_cost_weight_disables_cost_penalty(scorer):
    result = scorer.score(
        quality=0.9,
        cost_usd=10 * COST_REF,
        lam=0.0,
    )

    assert result.normalized_cost == pytest.approx(
        10.0
    )
    assert result.reward == pytest.approx(0.9)


def test_missing_latency_is_not_fabricated(scorer):
    result = scorer.score(
        quality=0.8,
        cost_usd=COST_REF,
        lam=0.1,
        mu=0.0,
    )

    assert result.latency_ms is None
    assert result.normalized_latency is None
    assert result.latency_included is False
    assert result.reward == pytest.approx(0.7)


def test_supplied_latency_can_be_recorded_without_penalty(
    scorer,
):
    result = scorer.score(
        quality=0.8,
        cost_usd=COST_REF,
        latency_ms=250.0,
        mu=0.0,
    )

    assert result.latency_ms == 250.0
    assert result.latency_included is False
    assert result.normalized_latency is None
    assert result.reward == pytest.approx(0.8)


def test_positive_mu_requires_measured_latency(scorer):
    with pytest.raises(
        ValueError,
        match="Measured latency is required",
    ):
        scorer.score(
            quality=0.8,
            cost_usd=COST_REF,
            mu=0.2,
        )


def test_positive_mu_requires_latency_reference(scorer):
    with pytest.raises(
        ValueError,
        match="latency reference",
    ):
        scorer.score(
            quality=0.8,
            cost_usd=COST_REF,
            latency_ms=250.0,
            mu=0.2,
        )


def test_latency_aware_reward_formula():
    config = valid_config(
        cost_ref_usd=0.002,
        latency_ref_ms=100.0,
        latency_source="measured",
    )

    scorer = CompositeScorer(config)

    result = scorer.score(
        quality=0.8,
        cost_usd=0.004,
        latency_ms=200.0,
        lam=0.1,
        mu=0.2,
        penalty=0.03,
    )

    # 0.8 - 0.1*(0.004/0.002)
    #     - 0.2*(200/100) - 0.03
    expected = 0.8 - 0.2 - 0.4 - 0.03

    assert result.normalized_cost == pytest.approx(
        2.0
    )
    assert result.normalized_latency == pytest.approx(
        2.0
    )
    assert result.latency_included is True
    assert result.reward == pytest.approx(expected)


def test_zero_latency_is_valid_with_measured_reference():
    config = valid_config(
        latency_ref_ms=100.0,
        latency_source="measured",
    )

    scorer = CompositeScorer(config)

    result = scorer.score(
        quality=0.8,
        cost_usd=0.0,
        latency_ms=0.0,
        mu=0.5,
    )

    assert result.normalized_latency == 0.0
    assert result.reward == pytest.approx(0.8)


def test_reward_can_be_negative(scorer):
    result = scorer.score(
        quality=0.1,
        cost_usd=3 * COST_REF,
        lam=1.0,
    )

    assert result.reward == pytest.approx(-2.9)


def test_scoring_is_deterministic(scorer):
    kwargs = {
        "quality": 0.75,
        "cost_usd": 0.001,
        "lam": 0.2,
        "mu": 0.0,
        "penalty": 0.05,
    }

    first = scorer.score(**kwargs)
    second = scorer.score(**kwargs)

    assert first == second


@pytest.mark.parametrize(
    "quality",
    [-0.1, 1.1, math.nan, math.inf, True],
)
def test_invalid_quality_is_rejected(scorer, quality):
    with pytest.raises(ValueError):
        scorer.score(
            quality=quality,
            cost_usd=0.0,
        )


@pytest.mark.parametrize(
    "cost",
    [-0.1, math.nan, math.inf, True],
)
def test_invalid_cost_is_rejected(scorer, cost):
    with pytest.raises(ValueError):
        scorer.score(
            quality=0.8,
            cost_usd=cost,
        )


@pytest.mark.parametrize(
    "weight",
    [-0.1, math.nan, math.inf, True],
)
def test_invalid_lambda_is_rejected(scorer, weight):
    with pytest.raises(ValueError):
        scorer.score(
            quality=0.8,
            cost_usd=0.0,
            lam=weight,
        )


@pytest.mark.parametrize(
    "weight",
    [-0.1, math.nan, math.inf, True],
)
def test_invalid_mu_is_rejected(scorer, weight):
    with pytest.raises(ValueError):
        scorer.score(
            quality=0.8,
            cost_usd=0.0,
            mu=weight,
        )


@pytest.mark.parametrize(
    "penalty",
    [-0.1, math.nan, math.inf, True],
)
def test_invalid_penalty_is_rejected(scorer, penalty):
    with pytest.raises(ValueError):
        scorer.score(
            quality=0.8,
            cost_usd=0.0,
            penalty=penalty,
        )


@pytest.mark.parametrize(
    "latency",
    [-1.0, math.nan, math.inf, True],
)
def test_invalid_latency_is_rejected(scorer, latency):
    with pytest.raises(ValueError):
        scorer.score(
            quality=0.8,
            cost_usd=0.0,
            latency_ms=latency,
        )


@pytest.mark.parametrize(
    "cost_ref",
    [0.0, -1.0, math.nan, math.inf],
)
def test_invalid_cost_reference_is_rejected(cost_ref):
    values = valid_config(
        cost_ref_usd=cost_ref
    )

    with pytest.raises(ValueError):
        ScoreConfig.from_mapping(values)


def test_validation_split_cannot_define_reference():
    values = valid_config(
        cost_ref_split="val"
    )

    with pytest.raises(
        ValueError,
        match="train split",
    ):
        ScoreConfig.from_mapping(values)


def test_test_split_cannot_define_reference():
    values = valid_config(
        cost_ref_split="test"
    )

    with pytest.raises(
        ValueError,
        match="train split",
    ):
        ScoreConfig.from_mapping(values)


def test_invalid_reference_statistic_is_rejected():
    values = valid_config(
        cost_ref_statistic="median_cost"
    )

    with pytest.raises(ValueError):
        ScoreConfig.from_mapping(values)


def test_missing_reference_model_is_rejected():
    values = valid_config(
        cost_ref_model=""
    )

    with pytest.raises(ValueError):
        ScoreConfig.from_mapping(values)


def test_invalid_training_count_is_rejected():
    values = valid_config(
        train_prompt_count=0
    )

    with pytest.raises(ValueError):
        ScoreConfig.from_mapping(values)


def test_invalid_latency_reference_is_rejected():
    values = valid_config(
        latency_ref_ms=0.0
    )

    with pytest.raises(ValueError):
        ScoreConfig.from_mapping(values)


def test_unsupported_configuration_version_is_rejected():
    values = valid_config(version=2)

    with pytest.raises(
        ValueError,
        match="version",
    ):
        ScoreConfig.from_mapping(values)


def test_invalid_scorer_configuration_type_is_rejected():
    with pytest.raises(TypeError):
        CompositeScorer("invalid configuration")


def test_configuration_is_not_changed_by_scoring(scorer):
    before = scorer.cfg

    scorer.score(
        quality=0.8,
        cost_usd=COST_REF,
        lam=0.3,
    )

    assert scorer.cfg == before


def test_missing_yaml_file_is_reported(tmp_path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        ScoreConfig.from_yaml(missing)


def test_invalid_yaml_mapping_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"

    path.write_text(
        "- first\n- second\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mapping",
    ):
        ScoreConfig.from_yaml(path)
