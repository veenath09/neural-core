"""Tests for Part 2 Task 7: simulated online experiments (eval/online_sim.py)."""

import numpy as np
import pytest

from eval.online_sim import cards_from_data, shifted_skeletons, stream, subset_data
from tests.test_train import tiny_data
from train.online import OnlineConfig
from train.train import train


def _trained(tmp_path, data):
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    return tmp_path / "full-s0"


def test_stream_is_prequential(tmp_path):
    data = tiny_data()
    init = _trained(tmp_path, data)
    cfg = OnlineConfig(update_every=4, replay_size=4, lr=1e-3)
    df = stream(init, data.train[:20], data, cfg, learn=True, lam_idx_fn=lambda i: i % 5, out_dir=tmp_path / "s")
    assert len(df) == 20 and df["episode"].tolist() == list(range(20))
    # each episode ran with the version published before it was observed: one update per 4 sample episodes
    expected = [i // 4 for i in range(20)]
    assert df["policy_version"].tolist() == expected
    assert {"ep_type", "lam", "mode", "quality", "cost_usd", "reward", "new_model_share", "rollback"} <= set(df)


def test_frozen_stream_never_updates(tmp_path):
    data = tiny_data()
    init = _trained(tmp_path, data)
    df = stream(init, data.train[:12], data, OnlineConfig(update_every=4), learn=False, lam_idx_fn=lambda i: 0,
                out_dir=tmp_path / "s")
    assert (df["policy_version"] == 0).all() and not df["rollback"].any()


def test_new_models_get_explored(tmp_path):
    data = tiny_data()
    ids = list(data.world.model_ids)
    cards = {c.model_id: c for c in cards_from_data(data)}
    start = subset_data(data, ids[:2])  # trained without models 2 and 3
    init = _trained(tmp_path, start)
    df = stream(init, data.train[:100], data, OnlineConfig(update_every=8, replay_size=8, lr=1e-3), learn=True,
                lam_idx_fn=lambda i: i % 5, add_model_at={0: [cards[ids[2]], cards[ids[3]]]}, out_dir=tmp_path / "s",
                per_model_share=True)
    assert df[f"share_{ids[2]}"].sum() > 0 and df[f"share_{ids[3]}"].sum() > 0
    assert df["new_model_share"].iloc[:100].mean() > 0


def test_shifted_skeletons_change_the_task_mix():
    data = tiny_data()
    table = data.world.table
    sks = shifted_skeletons(table, list(table.prompt_ids), n=200, seed=0, shares={0: 0.9, 1: 0.1}, by="step_type")
    types = np.array([table.step_type[table.index[p]] for sk in sks for p in sk.prompt_ids])
    assert (types == 0).mean() > 0.75  # the tiny table is 50/50; the shift pushes it to ~90/10
