
"""Tests for Pillar 3 C3-03: train-only model capabilities."""

import json
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from data.routerbench import POOL, SOURCES
from measurement.capabilities import (
    PROBE_VERSION,
    build_capability_profiles,
    save_capability_profiles,
)


def sample_inputs():
    """Create synthetic train, validation and test prompts."""

    rows = []
    assignments = []

    for source_index, source in enumerate(SOURCES):
        for index, split in enumerate(
            ("train", "train", "val", "test")
        ):
            prompt_id = f"{source}-{index}"

            row = {
                "prompt_id": prompt_id,
                "source": source,
                "step_type": source_index,
            }

            for tier, model in enumerate(POOL):
                row[f"{model}_correct"] = int(
                    (source_index + index + tier) % 3 != 0
                )

                row[f"{model}_cost"] = (
                    (tier + 1) * 0.00001 * (1 + index / 10)
                )

            rows.append(row)

            assignments.append({
                "prompt_id": prompt_id,
                "split": split,
            })

    return pd.DataFrame(rows), pd.DataFrame(assignments)


def test_four_model_cards_are_created():
    clean, splits = sample_inputs()

    cards, metadata = build_capability_profiles(
        clean, splits
    )

    assert len(cards) == 4
    assert [c.model_id for c in cards] == list(POOL)
    assert [c.tier for c in cards] == [0, 1, 2, 3]

    assert metadata["training_prompt_count"] == 10


def test_capabilities_use_training_accuracy():
    clean, splits = sample_inputs()

    cards, _ = build_capability_profiles(clean, splits)

    train_ids = splits.loc[
        splits["split"] == "train", "prompt_id"
    ]

    train = clean[
        clean["prompt_id"].isin(train_ids)
    ]

    for card in cards:
        assert len(card.capability) == 6

        for index, source in enumerate(SOURCES):
            expected = train.loc[
                train["source"] == source,
                f"{card.model_id}_correct",
            ].mean()

            assert card.capability[index] == pytest.approx(
                expected
            )


def test_mean_cost_and_log_cost_are_correct():
    clean, splits = sample_inputs()

    cards, _ = build_capability_profiles(clean, splits)

    train_ids = set(
        splits.loc[
            splits["split"] == "train", "prompt_id"
        ]
    )

    train = clean[
        clean["prompt_id"].isin(train_ids)
    ]

    for card in cards:
        expected_cost = train[
            f"{card.model_id}_cost"
        ].mean()

        assert card.mean_cost_usd == pytest.approx(
            expected_cost
        )

        assert card.capability[-1] == pytest.approx(
            math.log10(expected_cost)
        )


def test_validation_and_test_outcomes_cannot_leak():
    clean, splits = sample_inputs()

    first_cards, first_metadata = (
        build_capability_profiles(clean, splits)
    )

    changed = clean.copy()

    held_out_ids = splits.loc[
        splits["split"].isin(["val", "test"]),
        "prompt_id",
    ]

    mask = changed["prompt_id"].isin(held_out_ids)

    for model in POOL:
        correct_col = f"{model}_correct"
        cost_col = f"{model}_cost"

        changed.loc[mask, correct_col] = (
            1 - changed.loc[mask, correct_col]
        )

        changed.loc[mask, cost_col] = 999.0

    second_cards, second_metadata = (
        build_capability_profiles(changed, splits)
    )

    assert [
        asdict(c) for c in first_cards
    ] == [
        asdict(c) for c in second_cards
    ]

    assert first_metadata == second_metadata


def test_changing_training_data_changes_capability():
    clean, splits = sample_inputs()

    before, _ = build_capability_profiles(
        clean, splits
    )

    changed = clean.copy()

    train_id = splits.loc[
        splits["split"] == "train",
        "prompt_id",
    ].iloc[0]

    row_mask = changed["prompt_id"] == train_id

    column = f"{POOL[0]}_correct"

    changed.loc[row_mask, column] = (
        1 - changed.loc[row_mask, column]
    )

    after, _ = build_capability_profiles(
        changed, splits
    )

    assert before[0].capability != after[0].capability


def test_probe_version_and_latency_provenance():
    clean, splits = sample_inputs()

    cards, metadata = build_capability_profiles(
        clean, splits
    )

    assert metadata["probe_version"] == PROBE_VERSION
    assert metadata["training_only"] is True
    assert metadata["latency_status"] == "unavailable"

    for card in cards:
        assert len(card.capability) == 6

    for profile in metadata["profiles"]:
        assert profile["probe_version"] == "rb_train_v1"
        assert profile["mean_latency_ms"] is None
        assert profile["latency_available"] is False
        assert profile["latency_feature_is_placeholder"] is True


def test_profiles_can_be_saved_as_json(tmp_path):
    clean, splits = sample_inputs()

    clean_path = tmp_path / "clean.parquet"
    splits_path = tmp_path / "splits.parquet"
    cards_path = tmp_path / "model_cards.json"
    metadata_path = tmp_path / "metadata.json"

    clean.to_parquet(clean_path, index=False)
    splits.to_parquet(splits_path, index=False)

    cards, metadata = save_capability_profiles(
        clean_path=clean_path,
        splits_path=splits_path,
        cards_path=cards_path,
        metadata_path=metadata_path,
    )

    assert cards_path.is_file()
    assert metadata_path.is_file()

    saved_cards = json.loads(
        cards_path.read_text(encoding="utf-8")
    )

    saved_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert saved_cards == [asdict(c) for c in cards]
    assert saved_metadata == metadata


def test_missing_clean_column_is_rejected():
    clean, splits = sample_inputs()

    clean = clean.drop(
        columns=[f"{POOL[0]}_correct"]
    )

    with pytest.raises(ValueError, match="Missing clean"):
        build_capability_profiles(clean, splits)


def test_duplicate_clean_prompt_id_is_rejected():
    clean, splits = sample_inputs()

    clean.loc[1, "prompt_id"] = clean.loc[0, "prompt_id"]

    with pytest.raises(ValueError, match="Duplicate clean"):
        build_capability_profiles(clean, splits)


def test_duplicate_split_prompt_id_is_rejected():
    clean, splits = sample_inputs()

    splits.loc[1, "prompt_id"] = splits.loc[0, "prompt_id"]

    with pytest.raises(ValueError, match="Duplicate split"):
        build_capability_profiles(clean, splits)


def test_mismatched_prompt_ids_are_rejected():
    clean, splits = sample_inputs()

    splits.loc[0, "prompt_id"] = "unknown-prompt"

    with pytest.raises(ValueError, match="do not match"):
        build_capability_profiles(clean, splits)


def test_invalid_split_label_is_rejected():
    clean, splits = sample_inputs()

    splits.loc[0, "split"] = "unknown"

    with pytest.raises(ValueError, match="Unknown split"):
        build_capability_profiles(clean, splits)


def test_unknown_training_source_is_rejected():
    clean, splits = sample_inputs()

    clean.loc[0, "source"] = "unknown-source"

    with pytest.raises(ValueError, match="Unknown training source"):
        build_capability_profiles(clean, splits)


def test_source_and_step_type_must_match():
    clean, splits = sample_inputs()

    clean.loc[0, "step_type"] = 99

    with pytest.raises(ValueError, match="mismatch"):
        build_capability_profiles(clean, splits)


def test_every_source_requires_training_examples():
    clean, splits = sample_inputs()

    missing_source_ids = set(
        clean.loc[
            clean["source"] == SOURCES[0],
            "prompt_id",
        ]
    )

    splits.loc[
        splits["prompt_id"].isin(missing_source_ids),
        "split",
    ] = "val"

    with pytest.raises(
        ValueError, match="training examples"
    ):
        build_capability_profiles(clean, splits)


def test_nonbinary_training_score_is_rejected():
    clean, splits = sample_inputs()

    column = f"{POOL[0]}_correct"

    # FIX: Convert to float before inserting an invalid decimal.
    # Pandas otherwise rejects 0.5 in an integer column.
    clean[column] = clean[column].astype(float)

    # Insert a non-binary score to test our validation.
    clean.loc[0, column] = 0.5

    with pytest.raises(ValueError, match="Non-binary"):
        build_capability_profiles(clean, splits)


@pytest.mark.parametrize(
    "invalid_cost",
    [-1.0, np.nan, np.inf],
)
def test_invalid_training_cost_is_rejected(invalid_cost):
    clean, splits = sample_inputs()

    clean.loc[0, f"{POOL[0]}_cost"] = invalid_cost

    with pytest.raises(ValueError):
        build_capability_profiles(clean, splits)


def test_zero_mean_cost_is_rejected():
    clean, splits = sample_inputs()

    train_ids = splits.loc[
        splits["split"] == "train",
        "prompt_id",
    ]

    mask = clean["prompt_id"].isin(train_ids)

    clean.loc[mask, f"{POOL[0]}_cost"] = 0.0

    with pytest.raises(ValueError, match="positive"):
        build_capability_profiles(clean, splits)


def test_model_pool_cost_order_is_checked():
    clean, splits = sample_inputs()

    train_ids = splits.loc[
        splits["split"] == "train",
        "prompt_id",
    ]

    mask = clean["prompt_id"].isin(train_ids)

    clean.loc[mask, f"{POOL[0]}_cost"] = 0.01

    with pytest.raises(ValueError, match="ordered"):
        build_capability_profiles(clean, splits)
