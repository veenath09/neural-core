
"""C3-15: Tests for the simulator manifest exporter.

All test records are synthetic.
No provider APIs or live model executions are used.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts.export_c3_15_manifest import (
    export_manifest,
)


def make_test_data():
    """Create exactly 100 synthetic episodes and 446 attempts."""

    prediction_rows = []
    prompt_rows = []

    for episode_number in range(100):

        episode_id = f"test-{episode_number:06d}"

        # 46 episodes have 5 attempts.
        # The other 54 episodes have 4 attempts.
        attempts = 5 if episode_number < 46 else 4

        for decision_index in range(attempts):

            prompt_id = (
                f"synthetic.prompt."
                f"{episode_number}.{decision_index}"
            )

            prediction_rows.append(
                {
                    "policy": "profile-aware",
                    "onboarding_phase": "after",
                    "split": "test",
                    "seed": 1234,
                    "episode_id": episode_id,
                    "decision_index": decision_index,
                    "step_id": (
                        f"{episode_id}:step:"
                        f"{decision_index}"
                    ),
                    "round": 0,
                    "prompt_id": prompt_id,
                    "step_type": 0,
                    "model_id": "synthetic-model",
                    "executed": True,
                    "selected_model_correct": True,
                    "execution_cost_usd": 0.001,
                }
            )

            prompt_rows.append(
                {
                    "prompt_id": prompt_id,
                    "source": "mmlu",
                    "step_type": 0,
                    "prompt_text": (
                        f"Synthetic question "
                        f"{episode_number}-"
                        f"{decision_index}"
                    ),
                }
            )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(prompt_rows),
    )


@pytest.fixture
def dataset_paths(tmp_path):
    """Write synthetic Parquet inputs into temporary files."""

    predictions, prompts = make_test_data()

    predictions_path = tmp_path / "predictions.parquet"
    prompts_path = tmp_path / "prompts.parquet"
    output_path = tmp_path / "manifest.jsonl"

    predictions.to_parquet(predictions_path)
    prompts.to_parquet(prompts_path)

    return predictions_path, prompts_path, output_path


def test_export_creates_manifest(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    manifest = export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    assert output_path.exists()
    assert len(manifest) == 446
    assert manifest["episode_id"].nunique() == 100


def test_export_contains_prompt_text(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    manifest = export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    assert manifest["prompt_text"].notna().all()
    assert manifest["prompt_text"].str.len().gt(0).all()


def test_export_preserves_simulator_identity(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    manifest = export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    assert manifest["policy"].eq("profile-aware").all()
    assert manifest["onboarding_phase"].eq("after").all()
    assert manifest["measurement_source"].eq("simulator").all()


def test_export_preserves_all_attempts(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    manifest = export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    assert len(
        manifest.drop_duplicates(
            ["episode_id", "decision_index"]
        )
    ) == 446


def test_export_is_reproducible(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    first_hash = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    second_hash = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    assert first_hash == second_hash


def test_output_is_valid_jsonl(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    loaded = pd.read_json(
        output_path,
        lines=True,
    )

    assert len(loaded) == 446
    assert "prompt_text" in loaded.columns
    assert "model_id" in loaded.columns


def test_missing_prompt_rejected(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    prompts = pd.read_parquet(prompts_path)
    prompts = prompts.iloc[:-1].copy()
    prompts.to_parquet(prompts_path)

    with pytest.raises(
        ValueError,
        match="Missing RouterBench prompts",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_duplicate_prompt_ids_rejected(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    prompts = pd.read_parquet(prompts_path)

    prompts = pd.concat(
        [prompts, prompts.iloc[[0]]],
        ignore_index=True,
    )

    prompts.to_parquet(prompts_path)

    with pytest.raises(
        ValueError,
        match="duplicate prompt IDs",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_duplicate_attempts_rejected(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    predictions = pd.read_parquet(predictions_path)

    predictions.iloc[1, predictions.columns.get_loc(
        "decision_index"
    )] = 0

    predictions.to_parquet(predictions_path)

    with pytest.raises(
        ValueError,
        match="Duplicate execution attempt identifiers",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_step_type_mismatch_rejected(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    prompts = pd.read_parquet(prompts_path)

    prompts.loc[0, "step_type"] = 1

    prompts.to_parquet(prompts_path)

    with pytest.raises(
        ValueError,
        match="step types disagree",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_invalid_cost_rejected(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    predictions = pd.read_parquet(predictions_path)

    predictions.loc[0, "execution_cost_usd"] = -0.1

    predictions.to_parquet(predictions_path)

    with pytest.raises(
        ValueError,
        match="Invalid historical execution costs",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_wrong_episode_count_rejected(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    predictions = pd.read_parquet(predictions_path)

    predictions = predictions[
        predictions["episode_id"] != "test-000099"
    ]

    predictions.to_parquet(predictions_path)

    with pytest.raises(
        ValueError,
        match="Expected 100 episodes",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_non_simulator_data_cannot_be_relabelled(
    dataset_paths,
):

    predictions_path, prompts_path, output_path = dataset_paths

    predictions = pd.read_parquet(predictions_path)

    predictions["measurement_source"] = "live"

    predictions.to_parquet(predictions_path)

    with pytest.raises(
        ValueError,
        match="non-simulator data",
    ):
        export_manifest(
            predictions_path,
            prompts_path,
            output_path,
        )


def test_original_parquet_inputs_unchanged(dataset_paths):

    predictions_path, prompts_path, output_path = dataset_paths

    before_predictions = predictions_path.read_bytes()
    before_prompts = prompts_path.read_bytes()

    export_manifest(
        predictions_path,
        prompts_path,
        output_path,
    )

    assert predictions_path.read_bytes() == before_predictions
    assert prompts_path.read_bytes() == before_prompts
