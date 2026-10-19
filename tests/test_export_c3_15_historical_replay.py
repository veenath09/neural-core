
"""C3-15 tests for historical RouterBench response replay.

All inputs are synthetic and generated inside temporary
directories. No external APIs or live models are called.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts.export_c3_15_historical_replay import (
    export_historical_replay,
    parse_response,
)

MODEL_A = "gpt-3.5-turbo-1106"
MODEL_B = "mistralai/mixtral-8x7b-chat"


@pytest.fixture
def replay_files(tmp_path):
    """Prepare two synthetic historical execution attempts."""

    manifest = pd.DataFrame(
        [
            {
                "policy": "profile-aware",
                "onboarding_phase": "after",
                "split": "test",
                "seed": 1234,
                "episode_id": "test-000000",
                "decision_index": 0,
                "step_id": "test-000000:step:0",
                "round": 0,
                "prompt_id": "example-question-1",
                "step_type": 0,
                "source": "mmlu",
                "model_id": MODEL_A,
                "prompt_text": "Synthetic question one",
                "selected_model_correct": True,
                "execution_cost_usd": 0.001,
                "measurement_source": "simulator",
            },
            {
                "policy": "profile-aware",
                "onboarding_phase": "after",
                "split": "test",
                "seed": 1234,
                "episode_id": "test-000001",
                "decision_index": 0,
                "step_id": "test-000001:step:0",
                "round": 0,
                "prompt_id": "example-question-2",
                "step_type": 1,
                "source": "hellaswag",
                "model_id": MODEL_B,
                "prompt_text": "Synthetic question two",
                "selected_model_correct": False,
                "execution_cost_usd": 0.002,
                "measurement_source": "simulator",
            },
        ]
    )

    raw = pd.DataFrame(
        [
            {
                "sample_id": "example-question-1",
                MODEL_A: 1.0,
                MODEL_B: 0.0,
                f"{MODEL_A}|model_response": "['A']",
                f"{MODEL_B}|model_response": "['B']",
                f"{MODEL_A}|total_cost": 0.001,
                f"{MODEL_B}|total_cost": 0.002,
            },
            {
                "sample_id": "example-question-2",
                MODEL_A: 0.0,
                MODEL_B: 1.0,
                f"{MODEL_A}|model_response": "['C']",
                f"{MODEL_B}|model_response": "['D)']",
                f"{MODEL_A}|total_cost": 0.001,
                f"{MODEL_B}|total_cost": 0.002,
            },
        ]
    )

    manifest_path = tmp_path / "manifest.jsonl"
    raw_path = tmp_path / "synthetic_raw.pkl"
    output_path = tmp_path / "replay.jsonl"

    manifest.to_json(
        manifest_path,
        orient="records",
        lines=True,
    )

    raw.to_pickle(raw_path)

    return manifest_path, raw_path, output_path


def run_export(paths):
    return export_historical_replay(
        manifest_path=paths[0],
        raw_path=paths[1],
        output_path=paths[2],
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("['A']", "A"),
        ("['C)']", "C)"),
        ("['first', 'second']", "first\nsecond"),
        ("('one', 'two')", "one\ntwo"),
        ("ordinary answer", "ordinary answer"),
    ],
)
def test_parse_response(raw, expected):
    assert parse_response(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        123,
        "[]",
        "[123]",
        "{'answer': 'A'}",
    ],
)
def test_invalid_response_rejected(raw):
    with pytest.raises(ValueError):
        parse_response(raw)


def test_export_creates_historical_file(replay_files):
    replay = run_export(replay_files)

    assert replay_files[2].exists()
    assert len(replay) == 2

    assert replay[
        "measurement_source"
    ].eq("historical_replay").all()


def test_original_responses_preserved(replay_files):
    replay = run_export(replay_files)

    assert replay[
        "historical_response_raw"
    ].tolist() == [
        "['A']",
        "['D)']",
    ]

    assert replay[
        "historical_response_text"
    ].tolist() == [
        "A",
        "D)",
    ]


def test_correctness_mismatch_is_preserved(replay_files):
    replay = run_export(replay_files)

    assert replay[
        "simulator_correct"
    ].tolist() == [
        True,
        False,
    ]

    assert replay[
        "historical_correct"
    ].tolist() == [
        True,
        True,
    ]

    assert replay[
        "correctness_agreement"
    ].tolist() == [
        True,
        False,
    ]


def test_costs_preserved(replay_files):
    replay = run_export(replay_files)

    assert replay[
        "simulator_cost_usd"
    ].sum() == pytest.approx(0.003)

    assert replay[
        "historical_cost_usd"
    ].sum() == pytest.approx(0.003)


def test_execution_identity_preserved(replay_files):
    replay = run_export(replay_files)

    assert replay[
        "episode_id"
    ].tolist() == [
        "test-000000",
        "test-000001",
    ]

    assert replay[
        "decision_index"
    ].tolist() == [0, 0]

    assert replay[
        "model_id"
    ].tolist() == [
        MODEL_A,
        MODEL_B,
    ]


def test_export_is_reproducible(replay_files):
    run_export(replay_files)

    first = hashlib.sha256(
        replay_files[2].read_bytes()
    ).hexdigest()

    run_export(replay_files)

    second = hashlib.sha256(
        replay_files[2].read_bytes()
    ).hexdigest()

    assert first == second


def test_export_is_valid_jsonl(replay_files):
    run_export(replay_files)

    loaded = pd.read_json(
        replay_files[2],
        lines=True,
    )

    assert len(loaded) == 2
    assert "historical_response_text" in loaded.columns
    assert "historical_correct" in loaded.columns


def test_missing_manifest_column_rejected(replay_files):
    manifest = pd.read_json(
        replay_files[0],
        lines=True,
    )

    manifest = manifest.drop(
        columns=["prompt_id"]
    )

    manifest.to_json(
        replay_files[0],
        orient="records",
        lines=True,
    )

    with pytest.raises(
        ValueError,
        match="Manifest missing columns",
    ):
        run_export(replay_files)


def test_wrong_measurement_source_rejected(replay_files):
    manifest = pd.read_json(
        replay_files[0],
        lines=True,
    )

    manifest["measurement_source"] = "live"

    manifest.to_json(
        replay_files[0],
        orient="records",
        lines=True,
    )

    with pytest.raises(
        ValueError,
        match="non-simulator records",
    ):
        run_export(replay_files)


def test_duplicate_attempt_rejected(replay_files):
    manifest = pd.read_json(
        replay_files[0],
        lines=True,
    )

    manifest = pd.concat(
        [
            manifest,
            manifest.iloc[[0]],
        ],
        ignore_index=True,
    )

    manifest.to_json(
        replay_files[0],
        orient="records",
        lines=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate simulator execution attempts",
    ):
        run_export(replay_files)


def test_duplicate_raw_sample_rejected(replay_files):
    raw = pd.read_pickle(replay_files[1])

    raw = pd.concat(
        [
            raw,
            raw.iloc[[0]],
        ],
        ignore_index=True,
    )

    raw.to_pickle(replay_files[1])

    with pytest.raises(
        ValueError,
        match="Duplicate RouterBench sample IDs",
    ):
        run_export(replay_files)


def test_missing_historical_prompt_rejected(replay_files):
    raw = pd.read_pickle(replay_files[1])

    raw = raw.iloc[[0]].copy()

    raw.to_pickle(replay_files[1])

    with pytest.raises(
        ValueError,
        match="missing from the original RouterBench",
    ):
        run_export(replay_files)


def test_missing_response_column_rejected(replay_files):
    raw = pd.read_pickle(replay_files[1])

    raw = raw.drop(
        columns=[f"{MODEL_B}|model_response"]
    )

    raw.to_pickle(replay_files[1])

    with pytest.raises(
        ValueError,
        match="Missing RouterBench column",
    ):
        run_export(replay_files)


def test_invalid_historical_score_rejected(replay_files):
    raw = pd.read_pickle(replay_files[1])

    raw.loc[0, MODEL_A] = 0.5

    raw.to_pickle(replay_files[1])

    with pytest.raises(
        ValueError,
        match="Invalid historical correctness",
    ):
        run_export(replay_files)


def test_cost_mismatch_rejected(replay_files):
    raw = pd.read_pickle(replay_files[1])

    raw.loc[
        0,
        f"{MODEL_A}|total_cost",
    ] = 0.01

    raw.to_pickle(replay_files[1])

    with pytest.raises(
        ValueError,
        match="Historical cost differs",
    ):
        run_export(replay_files)


def test_empty_historical_response_rejected(replay_files):
    raw = pd.read_pickle(replay_files[1])

    raw.loc[
        0,
        f"{MODEL_A}|model_response",
    ] = ""

    raw.to_pickle(replay_files[1])

    with pytest.raises(
        ValueError,
        match="Historical response must be nonempty",
    ):
        run_export(replay_files)


def test_input_files_remain_unchanged(replay_files):
    original_manifest = replay_files[0].read_bytes()
    original_raw = replay_files[1].read_bytes()

    run_export(replay_files)

    assert replay_files[0].read_bytes() == original_manifest
    assert replay_files[1].read_bytes() == original_raw
