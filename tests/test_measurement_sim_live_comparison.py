
"""C3-15: Offline simulator-versus-live comparison tests.

Verify:
- Matching the same episode, policy, and execution attempt
- Preserving retries and repeated prompts
- Matching prompt, step, model, and step type
- Success agreement calculations
- Per-source execution cost differences
- Required live verification provenance
- Rejection of missing or mismatched records
- Rejection of invalid outcomes and costs
- Preservation of original input DataFrames

All records are synthetic.
No live API calls are made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.routerbench import SOURCES

from measurement.sim_live_comparison import (
    compare_sim_live,
    prepare_simulator_records,
)


MODEL_A = "test-model-a"
MODEL_B = "test-model-b"


def simulator_row(**changes):
    """Construct one synthetic simulator execution."""

    row = {
        "episode_id": "test-000001",
        "policy": "always-cheapest",
        "decision_index": 0,
        "step_id": "test-000001:step:0",
        "prompt_id": "mmlu.test.example-1",
        "step_type": 0,
        "model_id": MODEL_A,
        "round": 0,
        "executed": True,
        "selected_model_correct": True,
        "execution_cost_usd": 0.001,
        "measurement_source": "simulator",
    }

    row.update(changes)

    return row


def live_row(**changes):
    """Construct one SYNTHETIC live-shaped execution.

    The values are invented for testing only.
    They are not actual provider measurements.
    """

    row = {
        "episode_id": "test-000001",
        "policy": "always-cheapest",
        "decision_index": 0,
        "step_id": "test-000001:step:0",
        "prompt_id": "mmlu.test.example-1",
        "step_type": 0,
        "model_id": MODEL_A,
        "round": 0,
        "executed": True,
        "selected_model_correct": True,
        "execution_cost_usd": 0.0012,
        "measurement_source": "live",
        "verification_method": "synthetic-test-verifier",
        "cost_source": "synthetic-test-cost",
    }

    row.update(changes)

    return row


def frames(
    simulator_rows=None,
    live_rows=None,
):
    """Return independent simulator and live-shaped tables."""

    if simulator_rows is None:
        simulator_rows = [
            simulator_row()
        ]

    if live_rows is None:
        live_rows = [
            live_row()
        ]

    return (
        pd.DataFrame(simulator_rows),
        pd.DataFrame(live_rows),
    )


# -------------------------------------------------------
# Basic comparisons
# -------------------------------------------------------


def test_matching_execution_returns_two_dataframes():

    simulator, live = frames()

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert isinstance(
        paired,
        pd.DataFrame,
    )

    assert isinstance(
        summary,
        pd.DataFrame,
    )

    assert len(paired) == 1
    assert len(summary) == 1


def test_successful_execution_agreement():

    simulator, live = frames()

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert bool(
        paired["success_agreement"].iloc[0]
    )

    assert summary[
        "success_agreement"
    ].iloc[0] == 1.0

    assert summary[
        "sim_success_rate"
    ].iloc[0] == 1.0

    assert summary[
        "live_success_rate"
    ].iloc[0] == 1.0


def test_both_failures_still_agree():

    simulator, live = frames(
        [
            simulator_row(
                selected_model_correct=False
            )
        ],
        [
            live_row(
                selected_model_correct=False
            )
        ],
    )

    _, summary = compare_sim_live(
        simulator,
        live,
    )

    assert summary[
        "success_agreement"
    ].iloc[0] == 1.0

    assert summary[
        "sim_success_rate"
    ].iloc[0] == 0.0

    assert summary[
        "live_success_rate"
    ].iloc[0] == 0.0


def test_success_disagreement():

    simulator, live = frames(
        [
            simulator_row(
                selected_model_correct=True
            )
        ],
        [
            live_row(
                selected_model_correct=False
            )
        ],
    )

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert not bool(
        paired["success_agreement"].iloc[0]
    )

    assert summary[
        "success_agreement"
    ].iloc[0] == 0.0

    assert summary[
        "sim_success_rate"
    ].iloc[0] == 1.0

    assert summary[
        "live_success_rate"
    ].iloc[0] == 0.0


def test_execution_cost_difference():

    simulator, live = frames()

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert paired[
        "cost_difference_usd"
    ].iloc[0] == pytest.approx(
        0.0002
    )

    assert summary[
        "cost_mae_usd"
    ].iloc[0] == pytest.approx(
        0.0002
    )

    assert summary[
        "mean_cost_difference_usd"
    ].iloc[0] == pytest.approx(
        0.0002
    )


def test_negative_cost_difference_is_allowed():

    simulator, live = frames(
        [
            simulator_row(
                execution_cost_usd=0.003
            )
        ],
        [
            live_row(
                execution_cost_usd=0.002
            )
        ],
    )

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert paired[
        "cost_difference_usd"
    ].iloc[0] == pytest.approx(
        -0.001
    )

    assert summary[
        "cost_mae_usd"
    ].iloc[0] == pytest.approx(
        0.001
    )


def test_zero_execution_cost_is_valid():

    simulator, live = frames(
        [
            simulator_row(
                execution_cost_usd=0.0
            )
        ],
        [
            live_row(
                execution_cost_usd=0.0
            )
        ],
    )

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert paired[
        "absolute_cost_error_usd"
    ].iloc[0] == 0.0

    assert summary[
        "cost_mae_usd"
    ].iloc[0] == 0.0


# -------------------------------------------------------
# Execution identity and retries
# -------------------------------------------------------


def test_retries_are_compared_separately():

    simulator, live = frames(
        [
            simulator_row(
                decision_index=0,
                selected_model_correct=False,
            ),
            simulator_row(
                decision_index=1,
                round=1,
                selected_model_correct=True,
            ),
        ],
        [
            live_row(
                decision_index=0,
                selected_model_correct=True,
            ),
            live_row(
                decision_index=1,
                round=1,
                selected_model_correct=True,
            ),
        ],
    )

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert len(paired) == 2

    assert summary[
        "executions"
    ].iloc[0] == 2

    assert summary[
        "success_agreement"
    ].iloc[0] == pytest.approx(
        0.5
    )


def test_repeated_prompt_ids_are_allowed():

    simulator, live = frames(
        [
            simulator_row(
                decision_index=0
            ),
            simulator_row(
                decision_index=1,
                round=1,
            ),
        ],
        [
            live_row(
                decision_index=0
            ),
            live_row(
                decision_index=1,
                round=1,
            ),
        ],
    )

    paired, _ = compare_sim_live(
        simulator,
        live,
    )

    assert len(paired) == 2

    assert paired[
        "prompt_id_sim"
    ].nunique() == 1


def test_different_policies_are_not_combined():

    simulator, live = frames(
        [
            simulator_row(
                policy="always-cheapest"
            ),
            simulator_row(
                policy="profile-aware"
            ),
        ],
        [
            live_row(
                policy="always-cheapest"
            ),
            live_row(
                policy="profile-aware"
            ),
        ],
    )

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert len(paired) == 2

    assert summary[
        "executions"
    ].iloc[0] == 2


def test_model_mismatch_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                model_id=MODEL_B
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="model_id",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_prompt_mismatch_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                prompt_id="different-prompt"
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="prompt_id",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_step_id_mismatch_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                step_id="different-step"
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="step_id",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_step_type_mismatch_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                step_type=1
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="step_type",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_round_mismatch_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                round=1
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="round",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_missing_live_attempt_rejected():

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                decision_index=0
            ),
            simulator_row(
                decision_index=1
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="missing live attempts",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_extra_live_attempt_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                decision_index=0
            ),
            live_row(
                decision_index=1
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="missing simulator attempts",
    ):
        compare_sim_live(
            simulator,
            live,
        )


@pytest.mark.parametrize(
    "side",
    ["simulator", "live"],
)
def test_duplicate_attempt_keys_rejected(side):

    simulator, live = frames()

    if side == "simulator":

        simulator = pd.concat(
            [
                simulator,
                simulator,
            ],
            ignore_index=True,
        )

    else:

        live = pd.concat(
            [
                live,
                live,
            ],
            ignore_index=True,
        )

    with pytest.raises(
        ValueError,
        match="duplicate execution identifiers",
    ):
        compare_sim_live(
            simulator,
            live,
        )


# -------------------------------------------------------
# Per-source reporting
# -------------------------------------------------------


def test_per_source_reporting():

    simulator, live = frames(
        [
            simulator_row(
                step_type=0,
                prompt_id="mmlu-example",
            ),
            simulator_row(
                episode_id="test-000002",
                step_id="test-000002:step:0",
                step_type=4,
                prompt_id="mbpp-example",
                execution_cost_usd=0.002,
            ),
        ],
        [
            live_row(
                step_type=0,
                prompt_id="mmlu-example",
            ),
            live_row(
                episode_id="test-000002",
                step_id="test-000002:step:0",
                step_type=4,
                prompt_id="mbpp-example",
                execution_cost_usd=0.003,
            ),
        ],
    )

    _, summary = compare_sim_live(
        simulator,
        live,
    )

    assert len(summary) == 2

    assert summary[
        "source"
    ].tolist() == [
        "mmlu",
        "mbpp",
    ]

    assert summary[
        "step_type"
    ].tolist() == [
        0,
        4,
    ]


def test_all_five_sources_supported():

    sim_rows = []
    live_rows = []

    for index, source in enumerate(SOURCES):

        common = {
            "episode_id": f"test-{index:06d}",
            "step_id": f"test-{index:06d}:step:0",
            "prompt_id": f"{source}-example",
            "step_type": index,
        }

        sim_rows.append(
            simulator_row(**common)
        )

        live_rows.append(
            live_row(**common)
        )

    simulator, live = frames(
        sim_rows,
        live_rows,
    )

    _, summary = compare_sim_live(
        simulator,
        live,
    )

    assert summary[
        "source"
    ].tolist() == list(SOURCES)

    assert summary[
        "executions"
    ].tolist() == [1] * len(SOURCES)


def test_cost_mae_uses_absolute_errors():

    simulator, live = frames(
        [
            simulator_row(
                decision_index=0,
                execution_cost_usd=0.002,
            ),
            simulator_row(
                decision_index=1,
                execution_cost_usd=0.002,
            ),
        ],
        [
            live_row(
                decision_index=0,
                execution_cost_usd=0.003,
            ),
            live_row(
                decision_index=1,
                execution_cost_usd=0.001,
            ),
        ],
    )

    _, summary = compare_sim_live(
        simulator,
        live,
    )

    assert summary[
        "mean_cost_difference_usd"
    ].iloc[0] == pytest.approx(
        0.0
    )

    assert summary[
        "cost_mae_usd"
    ].iloc[0] == pytest.approx(
        0.001
    )


# -------------------------------------------------------
# Measurement provenance
# -------------------------------------------------------


def test_simulator_preparation_adds_label():

    original = pd.DataFrame(
        [
            simulator_row()
        ]
    ).drop(
        columns=["measurement_source"]
    )

    prepared = prepare_simulator_records(
        original
    )

    assert prepared[
        "measurement_source"
    ].eq("simulator").all()

    assert (
        "measurement_source"
        not in original.columns
    )


def test_simulator_preparation_preserves_existing_label():

    simulator, _ = frames()

    prepared = prepare_simulator_records(
        simulator
    )

    assert prepared[
        "measurement_source"
    ].eq("simulator").all()


def test_cannot_relabel_live_data_as_simulator():

    _, live = frames()

    with pytest.raises(
        ValueError,
        match="Cannot relabel",
    ):
        prepare_simulator_records(
            live
        )


def test_missing_live_provenance_rejected():

    simulator, live = frames()

    live = live.drop(
        columns=["verification_method"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_empty_live_verification_method_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                verification_method=""
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="verification_method",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_empty_live_cost_source_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                cost_source="   "
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="cost_source",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_wrong_live_measurement_label_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                measurement_source="simulator"
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Expected live measurement_source",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_wrong_simulator_label_rejected():

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                measurement_source="live"
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Expected simulator measurement_source",
    ):
        compare_sim_live(
            simulator,
            live,
        )


# -------------------------------------------------------
# Input validation
# -------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_cost",
    [
        -0.01,
        float("nan"),
        float("inf"),
        True,
        "unknown",
    ],
)
def test_invalid_simulator_cost_rejected(
    invalid_cost,
):

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                execution_cost_usd=invalid_cost
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="execution_cost_usd",
    ):
        compare_sim_live(
            simulator,
            live,
        )


@pytest.mark.parametrize(
    "invalid_cost",
    [
        -0.01,
        float("nan"),
        float("inf"),
        True,
        "unknown",
    ],
)
def test_invalid_live_cost_rejected(
    invalid_cost,
):

    simulator, live = frames(
        live_rows=[
            live_row(
                execution_cost_usd=invalid_cost
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="execution_cost_usd",
    ):
        compare_sim_live(
            simulator,
            live,
        )


@pytest.mark.parametrize(
    "invalid_type",
    [-1, 5, 99, True],
)
def test_invalid_step_type_rejected(
    invalid_type,
):

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                step_type=invalid_type
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="step_type",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_invalid_decision_index_rejected():

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                decision_index=-1
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="decision_index",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_non_boolean_correctness_rejected():

    simulator, live = frames(
        live_rows=[
            live_row(
                selected_model_correct="yes"
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="selected_model_correct",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_non_boolean_executed_rejected():

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                executed="yes"
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="executed",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_nonexecuted_decisions_are_excluded():

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                decision_index=0
            ),
            simulator_row(
                decision_index=1,
                executed=False,
            ),
        ],
    )

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert len(paired) == 1

    assert summary[
        "executions"
    ].iloc[0] == 1


def test_no_executed_records_rejected():

    simulator, live = frames(
        simulator_rows=[
            simulator_row(
                executed=False
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="No executed simulator records",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_missing_required_column_rejected():

    simulator, live = frames()

    simulator = simulator.drop(
        columns=["prompt_id"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_optional_key_must_exist_on_both_sides():

    simulator, live = frames()

    simulator["onboarding_phase"] = "after"

    with pytest.raises(
        ValueError,
        match="onboarding_phase",
    ):
        compare_sim_live(
            simulator,
            live,
        )


def test_matching_optional_keys_supported():

    simulator, live = frames()

    for frame in (
        simulator,
        live,
    ):
        frame["onboarding_phase"] = "after"
        frame["split"] = "test"
        frame["seed"] = 0
        frame["run_id"] = "synthetic-test-run"

    paired, summary = compare_sim_live(
        simulator,
        live,
    )

    assert len(paired) == 1

    assert summary[
        "executions"
    ].iloc[0] == 1


def test_original_dataframes_are_not_modified():

    simulator, live = frames()

    before_simulator = simulator.copy(
        deep=True
    )

    before_live = live.copy(
        deep=True
    )

    compare_sim_live(
        simulator,
        live,
    )

    pd.testing.assert_frame_equal(
        simulator,
        before_simulator,
    )

    pd.testing.assert_frame_equal(
        live,
        before_live,
    )
