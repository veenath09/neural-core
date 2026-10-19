
"""C3-13 tests for the workload-shifted test dataset.

Checks:
- Strict test-split isolation
- Episode compatibility
- Source distribution targets
- Correlated episode constraints
- Preserved relative budgets
- Deterministic generation
- Metadata and file output

All tests run offline.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.episodes import from_frame, to_frame
from data.routerbench import SOURCES

from scripts.make_shifted_test import (
    TARGET_SHARES,
    build_test_only_table,
    generate_shifted_skeletons,
    run,
    source_distribution,
    validate_shifted_skeletons,
    validate_targets,
)


@pytest.fixture(scope="module")
def dataset():
    """Load inputs and generate a test-only shifted workload."""

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    original_frame = pd.read_parquet(
        "data/episodes/skeletons_test.parquet"
    )

    # Critical: use a test-only table, not the full
    # table returned by _prepare_inputs().
    table = build_test_only_table(
        clean,
        splits,
    )

    original = from_frame(
        original_frame
    )

    shifted = generate_shifted_skeletons(
        original,
        table,
        seed=42,
    )

    return {
        "clean": clean,
        "splits": splits,
        "table": table,
        "original": original,
        "shifted": shifted,
        "original_frame": original_frame,
    }


def test_expected_episode_count(dataset):
    assert len(dataset["original"]) == 5000
    assert len(dataset["shifted"]) == 5000


def test_expected_step_count(dataset):
    frame = to_frame(
        dataset["shifted"]
    )

    assert len(frame) == 22259


def test_episode_ids_preserved(dataset):

    original_ids = [
        ep.episode_id
        for ep in dataset["original"]
    ]

    shifted_ids = [
        ep.episode_id
        for ep in dataset["shifted"]
    ]

    assert original_ids == shifted_ids
    assert len(set(shifted_ids)) == 5000


def test_episode_types_preserved(dataset):

    before = Counter(
        ep.ep_type
        for ep in dataset["original"]
    )

    after = Counter(
        ep.ep_type
        for ep in dataset["shifted"]
    )

    assert before == after

    assert after == {
        "correlated": 2000,
        "retry": 1250,
        "propagation": 1000,
        "random": 750,
    }


def test_episode_lengths_preserved(dataset):

    for before, after in zip(
        dataset["original"],
        dataset["shifted"],
    ):

        assert (
            len(before.prompt_ids)
            == len(after.prompt_ids)
        )

        assert 3 <= len(after.prompt_ids) <= 6


def test_only_test_prompts_used(dataset):
    """All generated prompts must belong to the test split."""

    splits = dataset["splits"]

    allowed = set(
        splits.loc[
            splits["split"].eq("test"),
            "prompt_id",
        ].astype(str)
    )

    actual = {
        prompt_id
        for episode in dataset["shifted"]
        for prompt_id in episode.prompt_ids
    }

    assert actual <= allowed


def test_training_prompts_excluded(dataset):
    """Explicitly reject any overlap with training prompts."""

    splits = dataset["splits"]

    training = set(
        splits.loc[
            splits["split"].eq("train"),
            "prompt_id",
        ].astype(str)
    )

    shifted_prompts = {
        prompt_id
        for episode in dataset["shifted"]
        for prompt_id in episode.prompt_ids
    }

    assert shifted_prompts.isdisjoint(
        training
    )


def test_no_duplicate_prompts_within_episodes(dataset):

    for episode in dataset["shifted"]:

        assert len(episode.prompt_ids) == len(
            set(episode.prompt_ids)
        )


def test_correlated_episode_constraints(dataset):

    table = dataset["table"]

    count = 0

    for episode in dataset["shifted"]:

        if episode.ep_type != "correlated":
            continue

        count += 1

        rows = np.asarray(
            [
                table.index[prompt_id]
                for prompt_id in episode.prompt_ids
            ],
            dtype=int,
        )

        sources = set(
            table.step_type[rows].tolist()
        )

        bands = set(
            table.band[rows].tolist()
        )

        assert len(sources) == 1
        assert len(bands) == 1

    assert count == 2000


def test_relative_budget_factors_preserved(dataset):

    table = dataset["table"]

    for before, after in zip(
        dataset["original"],
        dataset["shifted"],
    ):

        original_rows = np.asarray(
            [
                table.index[prompt_id]
                for prompt_id in before.prompt_ids
            ],
            dtype=int,
        )

        shifted_rows = np.asarray(
            [
                table.index[prompt_id]
                for prompt_id in after.prompt_ids
            ],
            dtype=int,
        )

        original_reference = float(
            table.cost[original_rows, -1].sum()
        )

        shifted_reference = float(
            table.cost[shifted_rows, -1].sum()
        )

        original_factor = (
            before.budget_usd
            / original_reference
        )

        shifted_factor = (
            after.budget_usd
            / shifted_reference
        )

        assert np.isclose(
            original_factor,
            shifted_factor,
            rtol=1e-10,
            atol=1e-12,
        )


def test_shifted_budgets_valid(dataset):

    for episode in dataset["shifted"]:

        assert np.isfinite(
            episode.budget_usd
        )

        assert episode.budget_usd > 0


def test_target_distribution_reached(dataset):

    observed = source_distribution(
        dataset["shifted"],
        dataset["table"],
    )

    assert set(observed) == set(
        SOURCES
    )

    for source in SOURCES:

        assert abs(
            observed[source]
            - TARGET_SHARES[source]
        ) <= 0.02


def test_workload_is_meaningfully_shifted(dataset):

    original_distribution = source_distribution(
        dataset["original"],
        dataset["table"],
    )

    shifted_distribution = source_distribution(
        dataset["shifted"],
        dataset["table"],
    )

    assert (
        original_distribution["mmlu"]
        - shifted_distribution["mmlu"]
    ) > 0.30

    assert (
        shifted_distribution["mbpp"]
        - original_distribution["mbpp"]
    ) > 0.20


def test_deterministic_generation(dataset):

    original = dataset["original"][:30]
    table = dataset["table"]

    first = generate_shifted_skeletons(
        original,
        table,
        seed=42,
    )

    second = generate_shifted_skeletons(
        original,
        table,
        seed=42,
    )

    assert first == second


def test_different_seeds_change_sampling(dataset):

    original = dataset["original"][:30]
    table = dataset["table"]

    first = generate_shifted_skeletons(
        original,
        table,
        seed=42,
    )

    second = generate_shifted_skeletons(
        original,
        table,
        seed=43,
    )

    assert any(
        first_ep.prompt_ids != second_ep.prompt_ids
        for first_ep, second_ep in zip(
            first,
            second,
        )
    )


def test_original_episodes_not_modified(dataset):

    original = dataset["original"][:20]
    table = dataset["table"]

    snapshot = tuple(
        original
    )

    generate_shifted_skeletons(
        original,
        table,
        seed=12,
    )

    assert tuple(original) == snapshot


def test_validation_accepts_shifted_workload(dataset):

    observed = validate_shifted_skeletons(
        dataset["original"],
        dataset["shifted"],
        dataset["table"],
        target_shares=TARGET_SHARES,
    )

    assert set(observed) == set(
        SOURCES
    )


def test_validation_rejects_changed_episode_id(dataset):

    original = dataset["original"]

    shifted = list(
        dataset["shifted"]
    )

    shifted[0] = replace(
        shifted[0],
        episode_id="invalid-episode-id",
    )

    with pytest.raises(
        RuntimeError,
        match="Original episode structure changed",
    ):
        validate_shifted_skeletons(
            original,
            shifted,
            dataset["table"],
            target_shares=TARGET_SHARES,
        )


@pytest.mark.parametrize(
    "invalid_targets",
    [
        {"mmlu": 1.0},
        {
            source: 0.0
            for source in SOURCES
        },
        {
            source: -0.2
            for source in SOURCES
        },
        {
            source: float("nan")
            for source in SOURCES
        },
    ],
)
def test_invalid_target_distributions_rejected(
    invalid_targets,
):

    with pytest.raises(
        ValueError
    ):
        validate_targets(
            invalid_targets
        )


def test_saved_schema_matches_original(dataset):

    shifted_frame = to_frame(
        dataset["shifted"]
    )

    original_frame = dataset[
        "original_frame"
    ]

    assert list(shifted_frame.columns) == list(
        original_frame.columns
    )

    assert len(shifted_frame) == len(
        original_frame
    )

    reconstructed = from_frame(
        shifted_frame
    )

    assert reconstructed == dataset["shifted"]


def test_generator_saves_output_and_metadata(
    dataset,
    tmp_path,
):
    """Verify the saved output also uses test-only prompts."""

    output = (
        tmp_path
        / "shifted_test.parquet"
    )

    run(
        output=output,
        seed=42,
    )

    metadata_path = (
        tmp_path
        / "shifted_test_metadata.json"
    )

    assert output.exists()
    assert metadata_path.exists()

    frame = pd.read_parquet(
        output
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(frame) == 22259

    assert metadata["episode_count"] == 5000
    assert metadata["step_count"] == 22259

    assert metadata["source_split"] == "test"
    assert metadata["test_only_table"] is True

    assert metadata["training_prompts_used"] is False
    assert metadata["validation_prompts_used"] is False

    allowed = set(
        dataset["splits"].loc[
            dataset["splits"]["split"].eq("test"),
            "prompt_id",
        ].astype(str)
    )

    assert set(
        frame["prompt_id"].astype(str)
    ) <= allowed

    for source in SOURCES:

        assert abs(
            metadata["achieved_source_shares"][source]
            - TARGET_SHARES[source]
        ) <= 0.02

    # No silent overwrite.
    with pytest.raises(
        FileExistsError
    ):
        run(
            output=output,
            seed=42,
        )
