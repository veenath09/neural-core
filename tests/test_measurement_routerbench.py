
"""Tests for Pillar 3 C3-01 RouterBench cleaning."""

import numpy as np
import pandas as pd
import pytest

from data.routerbench import POOL
from measurement.routerbench import (
    clean_routerbench,
    save_clean_routerbench,
    validate_clean_routerbench,
)


COSTS = (0.00001, 0.00002, 0.00003, 0.00004)


def make_row(
    prompt_id,
    source,
    prompt,
    scores=(0, 1, 1, 1),
    costs=COSTS,
):
    row = {
        "sample_id": prompt_id,
        "eval_name": source,
        "prompt": repr([prompt]),
    }

    for model, score, cost in zip(POOL, scores, costs):
        row[model] = score
        row[f"{model}|total_cost"] = cost

    return row


def sample_raw():
    return pd.DataFrame([
        make_row(
            "p1", "mmlu-anatomy", "Question one",
            (0, 1, 1, 0),
        ),
        make_row(
            "p2", "mbpp", "Question two",
            (1, 1, 1, 1),
        ),
        make_row(
            "p3", "grade-school-math", "Excluded",
            (0.25, 1, 1, 1),
        ),
    ])


def test_cleaning_keeps_only_selected_sources():
    cleaned = clean_routerbench(sample_raw())

    assert len(cleaned) == 2
    assert set(cleaned["source"]) == {"mmlu", "mbpp"}
    assert "grade-school-math" not in cleaned["source"].values


def test_correctness_values_are_binary():
    cleaned = clean_routerbench(sample_raw())

    for model in POOL:
        assert set(cleaned[f"{model}_correct"]).issubset({0, 1})


def test_difficulty_is_calculated_correctly():
    cleaned = clean_routerbench(sample_raw())

    difficulties = dict(
        zip(cleaned["prompt_id"], cleaned["difficulty"])
    )

    assert difficulties["p1"] == 2
    assert difficulties["p2"] == 0


def test_all_four_models_have_results():
    cleaned = clean_routerbench(sample_raw())

    for model in POOL:
        assert f"{model}_correct" in cleaned.columns
        assert f"{model}_cost" in cleaned.columns
        assert cleaned[f"{model}_correct"].notna().all()
        assert cleaned[f"{model}_cost"].notna().all()


def test_missing_raw_column_is_rejected():
    raw = sample_raw().drop(columns=[POOL[0]])

    with pytest.raises(ValueError, match="Missing raw columns"):
        clean_routerbench(raw)


def test_nonbinary_score_is_rejected():
    raw = sample_raw()
    raw.loc[0, POOL[0]] = 0.5

    with pytest.raises(ValueError, match="non-binary"):
        clean_routerbench(raw)


def test_missing_model_results_are_removed():
    raw = sample_raw()
    raw.loc[0, POOL[1]] = np.nan

    cleaned = clean_routerbench(raw)

    assert list(cleaned["prompt_id"]) == ["p2"]


def test_duplicate_prompt_text_is_removed():
    raw = sample_raw()
    duplicate = make_row(
        "p4", "mbpp", "Question one",
    )

    raw = pd.concat(
        [raw, pd.DataFrame([duplicate])],
        ignore_index=True,
    )

    cleaned = clean_routerbench(raw)

    assert len(cleaned) == 2
    assert cleaned["prompt_text"].is_unique


def test_duplicate_prompt_id_is_rejected():
    raw = sample_raw()
    raw.loc[1, "sample_id"] = "p1"

    with pytest.raises(ValueError, match="Duplicate prompt IDs"):
        clean_routerbench(raw)


def test_negative_cost_is_rejected():
    raw = sample_raw()
    raw.loc[0, f"{POOL[0]}|total_cost"] = -0.01

    with pytest.raises(ValueError, match="negative"):
        clean_routerbench(raw)


def test_infinite_cost_is_rejected():
    raw = sample_raw()
    raw.loc[0, f"{POOL[0]}|total_cost"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        clean_routerbench(raw)


def test_wrong_difficulty_is_rejected():
    cleaned = clean_routerbench(sample_raw())
    cleaned.loc[0, "difficulty"] = 4

    with pytest.raises(
        ValueError, match="Incorrect difficulty calculation"
    ):
        validate_clean_routerbench(cleaned)


def test_empty_prompt_text_is_rejected():
    cleaned = clean_routerbench(sample_raw())
    cleaned.loc[0, "prompt_text"] = "   "

    with pytest.raises(ValueError, match="Empty prompt text"):
        validate_clean_routerbench(cleaned)


def test_cleaned_dataset_can_be_saved_and_loaded(tmp_path):
    raw_path = tmp_path / "test_routerbench.pkl"
    output_path = tmp_path / "clean" / "routerbench.parquet"

    # This synthetic pickle is created locally by this test.
    sample_raw().to_pickle(raw_path)

    cleaned = save_clean_routerbench(
        raw_path=raw_path,
        output_path=output_path,
    )

    assert output_path.is_file()

    loaded = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(cleaned, loaded)
