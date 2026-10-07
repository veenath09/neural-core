import numpy as np
import pytest

from data.episodes import (
    BUDGET_FACTOR,
    EPISODE_TYPES,
    MAX_STEPS,
    MIN_STEPS,
    from_frame,
    generate_skeletons,
    to_frame,
    type_counts,
)


def test_type_counts_match_shares_exactly():
    assert type_counts(1000) == {"correlated": 400, "retry": 250, "propagation": 200, "random": 150}
    assert sum(type_counts(7).values()) == 7


def test_skeleton_shape_and_split_purity(table):
    ids = list(table.prompt_ids[:200])
    sks = generate_skeletons(table, ids, n=500, seed=0, split="train")
    assert len(sks) == 500 and len({s.episode_id for s in sks}) == 500
    assert {s.ep_type for s in sks} == set(EPISODE_TYPES)
    allowed = set(ids)
    for s in sks:
        assert MIN_STEPS <= len(s.prompt_ids) <= MAX_STEPS
        assert set(s.prompt_ids) <= allowed
        assert len(set(s.prompt_ids)) == len(s.prompt_ids)


def test_correlated_episodes_share_source_and_band(table):
    sks = generate_skeletons(table, list(table.prompt_ids), n=300, seed=0, split="train")
    for s in sks:
        if s.ep_type == "correlated":
            rows = [table.index[p] for p in s.prompt_ids]
            assert len(set(table.step_type[rows])) == 1
            assert len(set(table.band[rows])) == 1


def test_about_a_third_cannot_afford_strongest_every_step(table):
    sks = generate_skeletons(table, list(table.prompt_ids), n=3000, seed=0, split="train")
    short = [s.budget_usd < table.cost[[table.index[p] for p in s.prompt_ids], -1].sum() for s in sks]
    assert np.mean(short) == pytest.approx(1 / 3, abs=0.04)
    assert BUDGET_FACTOR == (0.7, 1.6)


def test_generation_is_deterministic(table):
    ids = list(table.prompt_ids)
    assert generate_skeletons(table, ids, 50, 3, "val") == generate_skeletons(table, ids, 50, 3, "val")
    assert generate_skeletons(table, ids, 50, 3, "val") != generate_skeletons(table, ids, 50, 4, "val")


def test_frame_round_trip(table):
    sks = generate_skeletons(table, list(table.prompt_ids), n=40, seed=0, split="test")
    df = to_frame(sks)
    assert len(df) == sum(len(s.prompt_ids) for s in sks)
    assert from_frame(df) == sks
