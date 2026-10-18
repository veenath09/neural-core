
"""Tests for Pillar 3 C3-04: episode skeleton generation."""

from collections import Counter

import pandas as pd
import pytest

from data.episodes import (
    BUDGET_FACTOR,
    EPISODE_TYPES,
    MAX_STEPS,
    MIN_STEPS,
    from_frame,
)
from data.routerbench import POOL, SOURCES
from measurement import episodes as episode_module
from measurement.episodes import generate, save_skeletons


@pytest.fixture
def sample_inputs():
    """Build valid synthetic data with all 25 source/difficulty groups."""

    rows = []
    assignments = []

    for source_index, source in enumerate(SOURCES):
        for difficulty in range(5):
            for index in range(24):
                prompt_id = f"{source}-{difficulty}-{index}"

                row = {
                    "prompt_id": prompt_id,
                    "source": source,
                    "step_type": source_index,
                    "prompt_text": f"Question {prompt_id}",
                    "difficulty": difficulty,
                }

                for tier, model in enumerate(POOL):
                    row[f"{model}_correct"] = int(
                        tier >= difficulty
                    )

                    row[f"{model}_cost"] = (
                        (tier + 1) * 0.00001 *
                        (1 + index / 100)
                    )

                if index < 12:
                    split = "train"
                elif index < 18:
                    split = "val"
                else:
                    split = "test"

                rows.append(row)
                assignments.append({
                    "prompt_id": prompt_id,
                    "split": split,
                })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(assignments),
    )


def make_episodes(sample_inputs, split="train", n=100, **kwargs):
    clean, splits = sample_inputs

    return generate(
        split=split,
        n=n,
        clean=clean,
        splits=splits,
        **kwargs,
    )


def test_correct_episode_type_distribution(sample_inputs):
    episodes = make_episodes(sample_inputs, n=100)
    counts = Counter(e.ep_type for e in episodes)

    assert len(episodes) == 100
    assert counts == {
        "correlated": 40,
        "retry": 25,
        "propagation": 20,
        "random": 15,
    }


def test_episode_lengths_are_three_to_six(sample_inputs):
    episodes = make_episodes(sample_inputs, n=200)

    lengths = {
        len(e.prompt_ids)
        for e in episodes
    }

    assert lengths == {3, 4, 5, 6}

    for episode in episodes:
        assert MIN_STEPS <= len(episode.prompt_ids) <= MAX_STEPS


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_no_cross_split_prompt_leakage(sample_inputs, split):
    clean, splits = sample_inputs

    episodes = make_episodes(
        sample_inputs,
        split=split,
        n=100,
    )

    allowed = set(
        splits.loc[
            splits["split"] == split,
            "prompt_id",
        ]
    )

    for episode in episodes:
        assert set(episode.prompt_ids).issubset(allowed)
        assert episode.episode_id.startswith(f"{split}-")


def test_unique_episode_identifiers(sample_inputs):
    episodes = make_episodes(sample_inputs, n=100)

    ids = [e.episode_id for e in episodes]

    assert len(ids) == len(set(ids))


def test_no_duplicate_prompt_inside_episode(sample_inputs):
    episodes = make_episodes(sample_inputs, n=200)

    for episode in episodes:
        assert len(episode.prompt_ids) == len(
            set(episode.prompt_ids)
        )


def test_correlated_episodes_share_source_and_difficulty(
    sample_inputs,
):
    clean, _ = sample_inputs

    lookup = clean.set_index("prompt_id")

    episodes = make_episodes(sample_inputs, n=200)

    for episode in episodes:
        if episode.ep_type != "correlated":
            continue

        source_difficulty = {
            (
                lookup.loc[prompt_id, "source"],
                int(lookup.loc[prompt_id, "difficulty"]),
            )
            for prompt_id in episode.prompt_ids
        }

        assert len(source_difficulty) == 1


def test_budget_follows_strongest_model_cost(sample_inputs):
    clean, _ = sample_inputs

    strongest = clean.set_index("prompt_id")[
        f"{POOL[-1]}_cost"
    ]

    episodes = make_episodes(sample_inputs, n=200)

    for episode in episodes:
        strongest_total = sum(
            float(strongest.loc[prompt_id])
            for prompt_id in episode.prompt_ids
        )

        ratio = episode.budget_usd / strongest_total

        assert BUDGET_FACTOR[0] <= ratio < BUDGET_FACTOR[1]
        assert episode.budget_usd > 0


def test_budget_limited_episode_share(sample_inputs):
    episodes = make_episodes(sample_inputs, n=1000)

    clean, _ = sample_inputs

    strongest = clean.set_index("prompt_id")[
        f"{POOL[-1]}_cost"
    ]

    limited = 0

    for episode in episodes:
        full_cost = sum(
            float(strongest.loc[prompt_id])
            for prompt_id in episode.prompt_ids
        )

        if episode.budget_usd < full_cost:
            limited += 1

    proportion = limited / len(episodes)

    # Expected share is approximately one third.
    assert 0.28 <= proportion <= 0.39


def test_same_seed_generates_identical_episodes(sample_inputs):
    first = make_episodes(sample_inputs, n=100, seed=1234)
    second = make_episodes(sample_inputs, n=100, seed=1234)

    assert first == second


def test_different_seed_changes_episodes(sample_inputs):
    first = make_episodes(sample_inputs, n=100, seed=1234)
    second = make_episodes(sample_inputs, n=100, seed=5678)

    assert first != second


def test_custom_episode_mix(sample_inputs):
    mix = {
        "correlated": 0.50,
        "retry": 0.10,
        "propagation": 0.10,
        "random": 0.30,
    }

    episodes = make_episodes(
        sample_inputs,
        n=100,
        mix=mix,
    )

    assert Counter(e.ep_type for e in episodes) == {
        "correlated": 50,
        "retry": 10,
        "propagation": 10,
        "random": 30,
    }


def test_random_only_mix(sample_inputs):
    mix = {
        "correlated": 0.0,
        "retry": 0.0,
        "propagation": 0.0,
        "random": 1.0,
    }

    episodes = make_episodes(
        sample_inputs,
        n=20,
        mix=mix,
    )

    assert len(episodes) == 20
    assert all(e.ep_type == "random" for e in episodes)


@pytest.mark.parametrize(
    "invalid_count",
    [0, -1, True, 2.5],
)
def test_invalid_episode_count_is_rejected(
    sample_inputs,
    invalid_count,
):
    with pytest.raises(ValueError, match="positive"):
        make_episodes(
            sample_inputs,
            n=invalid_count,
        )


def test_unknown_split_is_rejected(sample_inputs):
    with pytest.raises(ValueError, match="Unknown dataset split"):
        make_episodes(
            sample_inputs,
            split="production",
            n=10,
        )


@pytest.mark.parametrize(
    "invalid_mix",
    [
        {"correlated": 0.5, "retry": 0.5},
        {
            "correlated": 0.5,
            "retry": 0.2,
            "propagation": 0.2,
            "random": 0.2,
        },
        {
            "correlated": -0.1,
            "retry": 0.4,
            "propagation": 0.3,
            "random": 0.4,
        },
    ],
)
def test_invalid_mix_is_rejected(sample_inputs, invalid_mix):
    with pytest.raises(ValueError):
        make_episodes(
            sample_inputs,
            n=20,
            mix=invalid_mix,
        )


def test_duplicate_split_ids_are_rejected(sample_inputs):
    clean, splits = sample_inputs

    changed = splits.copy()
    changed.loc[1, "prompt_id"] = changed.loc[0, "prompt_id"]

    with pytest.raises(ValueError, match="Duplicate split"):
        generate(
            "train",
            10,
            clean=clean,
            splits=changed,
        )


def test_mismatched_dataset_ids_are_rejected(sample_inputs):
    clean, splits = sample_inputs

    changed = splits.copy()
    changed.loc[0, "prompt_id"] = "unknown-prompt"

    with pytest.raises(ValueError, match="do not match"):
        generate(
            "train",
            10,
            clean=clean,
            splits=changed,
        )


def test_skeletons_can_be_saved_and_reloaded(
    sample_inputs,
    tmp_path,
    monkeypatch,
):
    clean, splits = sample_inputs

    clean_path = tmp_path / "clean.parquet"
    splits_path = tmp_path / "splits.parquet"
    output_dir = tmp_path / "episodes"

    clean.to_parquet(clean_path, index=False)
    splits.to_parquet(splits_path, index=False)

    monkeypatch.setattr(
        episode_module,
        "CLEAN_PATH",
        clean_path,
    )

    monkeypatch.setattr(
        episode_module,
        "SPLITS_PATH",
        splits_path,
    )

    saved = save_skeletons(
        split="test",
        n=40,
        seed=1234,
        output_dir=output_dir,
    )

    path = output_dir / "skeletons_test.parquet"

    assert path.is_file()

    loaded_frame = pd.read_parquet(path)
    pd.testing.assert_frame_equal(saved, loaded_frame)

    loaded_episodes = from_frame(loaded_frame)

    assert len(loaded_episodes) == 40
    assert all(
        episode.episode_id.startswith("test-")
        for episode in loaded_episodes
    )

    # Skeletons must not contain execution outcomes.
    assert "correct" not in loaded_frame.columns
    assert "verifier_ok" not in loaded_frame.columns
    assert "success" not in loaded_frame.columns
