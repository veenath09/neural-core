
"""Automated tests for Pillar 3 C3-02: leakage-safe splitting."""

import pandas as pd
import pytest

from measurement.splits import (
    RATIOS,
    SEED,
    create_splits,
    save_splits,
)


def sample_data():
    """Create 1,000 prompts across 25 source/difficulty groups."""

    sources = [
        "mmlu",
        "hellaswag",
        "arc-challenge",
        "winogrande",
        "mbpp",
    ]

    rows = []

    for source in sources:
        for difficulty in range(5):
            for i in range(40):
                prompt_id = f"{source}-{difficulty}-{i}"

                rows.append({
                    "prompt_id": prompt_id,
                    "prompt_text": f"Question {prompt_id}",
                    "source": source,
                    "difficulty": difficulty,
                })

    return pd.DataFrame(rows)


def test_correct_split_ratios():
    result = create_splits(sample_data())

    counts = result["split"].value_counts()

    assert counts["train"] == 720
    assert counts["val"] == 80
    assert counts["test"] == 200


def test_all_prompts_are_assigned():
    original = sample_data()
    result = create_splits(original)

    assert len(result) == len(original)

    assert set(result["prompt_id"]) == set(
        original["prompt_id"]
    )

    assert not result["split"].isna().any()


def test_no_prompt_leakage():
    result = create_splits(sample_data())

    train = set(
        result.loc[result["split"] == "train", "prompt_id"]
    )

    val = set(
        result.loc[result["split"] == "val", "prompt_id"]
    )

    test = set(
        result.loc[result["split"] == "test", "prompt_id"]
    )

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    assert len(train | val | test) == 1000


def test_stratification_by_source_and_difficulty():
    result = create_splits(sample_data())

    for _, group in result.groupby(
        ["source", "difficulty"]
    ):
        counts = group["split"].value_counts()

        for split_name, target_ratio in RATIOS.items():
            expected = len(group) * target_ratio
            actual = counts.get(split_name, 0)

            assert abs(actual - expected) <= 1


def test_same_seed_produces_same_split():
    data = sample_data()

    first = create_splits(data, seed=SEED)
    second = create_splits(data, seed=SEED)

    pd.testing.assert_frame_equal(first, second)


def test_row_order_does_not_change_assignments():
    original = sample_data()

    shuffled = original.sample(
        frac=1,
        random_state=77,
    ).reset_index(drop=True)

    first = create_splits(original).sort_values(
        "prompt_id"
    ).reset_index(drop=True)

    second = create_splits(shuffled).sort_values(
        "prompt_id"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_assignments():
    data = sample_data()

    first = create_splits(data, seed=1234)
    second = create_splits(data, seed=5678)

    assert not first["split"].equals(second["split"])


def test_duplicate_prompt_ids_are_rejected():
    data = sample_data()

    data.loc[1, "prompt_id"] = data.loc[0, "prompt_id"]

    with pytest.raises(
        ValueError, match="Duplicate prompt IDs"
    ):
        create_splits(data)


def test_duplicate_prompt_text_is_rejected():
    data = sample_data()

    data.loc[1, "prompt_text"] = data.loc[
        0, "prompt_text"
    ]

    with pytest.raises(
        ValueError, match="Duplicate prompt texts"
    ):
        create_splits(data)


def test_missing_required_column_is_rejected():
    data = sample_data().drop(columns=["source"])

    with pytest.raises(
        ValueError, match="Missing columns"
    ):
        create_splits(data)


@pytest.mark.parametrize(
    "invalid_difficulty",
    [-1, 5, 1.5, None],
)
def test_invalid_difficulty_is_rejected(
    invalid_difficulty,
):
    data = sample_data()

    # Convert integer column to float before inserting 1.5.
    # This allows the validation function to test
    # and reject invalid decimal difficulty values.
    data["difficulty"] = data["difficulty"].astype(float)

    data.loc[0, "difficulty"] = invalid_difficulty

    with pytest.raises(ValueError):
        create_splits(data)


@pytest.mark.parametrize(
    "column",
    ["prompt_id", "prompt_text", "source"],
)
def test_empty_identifiers_are_rejected(column):
    data = sample_data()

    data.loc[0, column] = "   "

    with pytest.raises(ValueError):
        create_splits(data)


def test_empty_dataset_is_rejected():
    data = sample_data().iloc[0:0]

    with pytest.raises(
        ValueError, match="must not be empty"
    ):
        create_splits(data)


def test_splits_can_be_saved_and_loaded(tmp_path):
    data = sample_data()

    input_file = tmp_path / "clean.parquet"
    output_file = tmp_path / "splits.parquet"

    data.to_parquet(input_file, index=False)

    generated = save_splits(
        clean_path=input_file,
        output_path=output_file,
        seed=SEED,
    )

    assert output_file.exists()

    loaded = pd.read_parquet(output_file)

    pd.testing.assert_frame_equal(generated, loaded)
