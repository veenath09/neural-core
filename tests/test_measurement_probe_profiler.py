
"""C3-12: Tests for the training-only ProbeProfiler.

Tests:
- Correct six-dimensional capability calculations
- Accuracy for each RouterBench task type
- Mean execution cost in log10 representation
- Training-only data restrictions
- Missing and invalid data handling
- Preservation of the original ModelCard
- Compatibility with real C3-03 RouterBench data

No API calls are made.
"""

from copy import deepcopy
import math

import pandas as pd
import pytest

from controller.contracts import ModelCard
from data.routerbench import SOURCES

from measurement.probe_profiler import (
    ProbeProfiler,
    PROBE_VERSION,
)


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture
def model_card():
    """Create a valid model card for testing."""

    return ModelCard(
        model_id="test/model-a",
        provider="test",
        tier=0,
        mean_cost_usd=0.002,
        price_source="test-fixture",
        price_date="test-date",
        capability=[
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            math.log10(0.002),
        ],
    )


@pytest.fixture
def probe_set():
    """Create two training probes per task type.

    The first probe fails.
    The second probe succeeds.

    Therefore, every task-type accuracy is 0.5.

    Probe costs alternate between 0.001 and 0.003,
    so the overall mean cost is 0.002.
    """

    rows = []

    for step_type, source in enumerate(SOURCES):

        for index in range(2):

            rows.append(
                {
                    "prompt_id": (
                        f"prompt-{step_type}-{index}"
                    ),
                    "source": source,
                    "step_type": step_type,
                    "split": "train",
                    "test/model-a_correct": index,
                    "test/model-a_cost": (
                        0.001
                        if index == 0
                        else 0.003
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# BASIC PROFILE TESTS
# ============================================================

def test_probe_version():
    """Probe profiler must expose its version."""

    assert PROBE_VERSION == "rb_probe_v1"


def test_profiler_returns_six_features(
    model_card,
    probe_set,
):
    """Capability vector must match the C3-03 contract."""

    profiler = ProbeProfiler()

    result = profiler.run(
        model_card,
        probe_set,
    )

    assert isinstance(result, list)
    assert len(result) == 6


def test_accuracy_calculation(
    model_card,
    probe_set,
):
    """Each task type must have accuracy 0.5."""

    profiler = ProbeProfiler()

    result = profiler.run(
        model_card,
        probe_set,
    )

    expected_accuracies = [
        0.5
        for _ in SOURCES
    ]

    assert result[:5] == pytest.approx(
        expected_accuracies
    )


def test_log10_mean_cost(
    model_card,
    probe_set,
):
    """Sixth feature must be log10(mean execution cost)."""

    profiler = ProbeProfiler()

    result = profiler.run(
        model_card,
        probe_set,
    )

    assert result[5] == pytest.approx(
        math.log10(0.002)
    )


def test_original_model_card_unchanged(
    model_card,
    probe_set,
):
    """Profiling must not mutate the existing card."""

    original = deepcopy(model_card)

    profiler = ProbeProfiler()

    profiler.run(
        model_card,
        probe_set,
    )

    assert model_card == original


# ============================================================
# DATA LEAKAGE PREVENTION
# ============================================================

@pytest.mark.parametrize(
    "split",
    ["test", "val"],
)
def test_rejects_non_training_split(
    model_card,
    probe_set,
    split,
):
    """Validation and test observations must be rejected."""

    data = probe_set.copy()

    data.loc[0, "split"] = split

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="training probes only",
    ):
        profiler.run(
            model_card,
            data,
        )


def test_rejects_missing_split_column(
    model_card,
    probe_set,
):
    """The input must explicitly identify the training split."""

    data = probe_set.drop(
        columns=["split"]
    )

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="Missing probe columns",
    ):
        profiler.run(
            model_card,
            data,
        )


def test_rejects_duplicate_prompt_ids(
    model_card,
    probe_set,
):
    """Duplicate prompts must not inflate capability scores."""

    data = probe_set.copy()

    data.loc[
        data.index[1],
        "prompt_id",
    ] = data.loc[
        data.index[0],
        "prompt_id",
    ]

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="Duplicate probe prompt IDs",
    ):
        profiler.run(
            model_card,
            data,
        )


# ============================================================
# INVALID DATA TESTS
# ============================================================

def test_rejects_empty_probe_set(
    model_card,
    probe_set,
):
    """An empty probe set must not produce a profile."""

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="Probe set cannot be empty",
    ):
        profiler.run(
            model_card,
            probe_set.iloc[0:0],
        )


def test_rejects_non_binary_correctness(
    model_card,
    probe_set,
):
    """Correctness must be either zero or one.

    Pandas requires the correctness column to accept
    floating-point values before inserting 0.7.

    This allows the test to reach ProbeProfiler.run()
    and verify its binary-correctness validation.
    """

    data = probe_set.copy()

    # FIX: Convert the int64 column to float64.
    data["test/model-a_correct"] = (
        data["test/model-a_correct"].astype(float)
    )

    # Introduce an invalid non-binary observation.
    data.loc[
        data.index[0],
        "test/model-a_correct",
    ] = 0.7

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="must be binary",
    ):
        profiler.run(
            model_card,
            data,
        )


def test_rejects_negative_cost(
    model_card,
    probe_set,
):
    """Negative execution costs must not be accepted."""

    data = probe_set.copy()

    data.loc[
        data.index[0],
        "test/model-a_cost",
    ] = -0.01

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="Negative probe costs",
    ):
        profiler.run(
            model_card,
            data,
        )


def test_rejects_nonfinite_cost(
    model_card,
    probe_set,
):
    """Infinite execution costs must be rejected."""

    data = probe_set.copy()

    data.loc[
        data.index[0],
        "test/model-a_cost",
    ] = float("inf")

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="Non-finite probe costs",
    ):
        profiler.run(
            model_card,
            data,
        )


def test_rejects_source_step_type_mismatch(
    model_card,
    probe_set,
):
    """Source and step-type identifiers must agree."""

    data = probe_set.copy()

    data.loc[
        data.index[0],
        "step_type",
    ] = 99

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="source and step type mismatch",
    ):
        profiler.run(
            model_card,
            data,
        )


# ============================================================
# MISSING TASK-TYPE OBSERVATIONS
# ============================================================

def test_missing_step_type_rejected_by_default(
    model_card,
    probe_set,
):
    """Default mode requires probes for every task type."""

    missing_source = SOURCES[-1]

    data = probe_set.loc[
        probe_set["source"] != missing_source
    ].copy()

    profiler = ProbeProfiler()

    with pytest.raises(
        ValueError,
        match="no observations",
    ):
        profiler.run(
            model_card,
            data,
        )


def test_missing_step_type_can_preserve_prior(
    model_card,
    probe_set,
):
    """Optional mode preserves the prior for an unseen type."""

    missing_source = SOURCES[2]

    data = probe_set.loc[
        probe_set["source"] != missing_source
    ].copy()

    profiler = ProbeProfiler(
        require_all_step_types=False,
    )

    result = profiler.run(
        model_card,
        data,
    )

    assert result[2] == pytest.approx(
        model_card.capability[2]
    )

    for index in [0, 1, 3, 4]:
        assert result[index] == pytest.approx(
            0.5
        )

    assert result[5] == pytest.approx(
        math.log10(0.002)
    )


# ============================================================
# REAL ROUTERBENCH INTEGRATION
# ============================================================

def test_profiler_with_real_training_data():
    """Profile a real pool model using train-only probes.

    The selected probes come from C3-03's training split.
    No validation or test correctness is used.
    """

    import json

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    with open(
        "data/clean/model_cards.json",
        encoding="utf-8",
    ) as file:
        raw_cards = json.load(file)

    card = ModelCard(
        **raw_cards[0]
    )

    # Select training prompt IDs only.
    train_ids = splits.loc[
        splits["split"] == "train",
        ["prompt_id", "split"],
    ]

    train = clean.merge(
        train_ids,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )

    # Deterministically select ten probes per source.
    probes = (
        train.groupby(
            "source",
            sort=False,
        )
        .head(10)
        .copy()
    )

    assert len(probes) == len(SOURCES) * 10

    assert probes["split"].eq(
        "train"
    ).all()

    profiler = ProbeProfiler()

    result = profiler.run(
        card,
        probes,
    )

    assert len(result) == 6

    assert all(
        0 <= value <= 1
        for value in result[:5]
    )

    assert math.isfinite(
        result[5]
    )

    # Verify each accuracy against the observed
    # training subset.
    correctness_column = (
        f"{card.model_id}_correct"
    )

    for index, source in enumerate(SOURCES):

        expected = float(
            probes.loc[
                probes["source"] == source,
                correctness_column,
            ].mean()
        )

        assert result[index] == pytest.approx(
            expected
        )

    # Verify the mean execution cost feature.
    cost_column = (
        f"{card.model_id}_cost"
    )

    expected_log_cost = math.log10(
        float(
            probes[cost_column].mean()
        )
    )

    assert result[5] == pytest.approx(
        expected_log_cost
    )
