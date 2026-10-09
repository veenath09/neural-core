"""Tests for the RL quality-cost frontier evaluation (C3)."""

from pathlib import Path

import pandas as pd

from eval.rl_frontier import LAMS, by_type, frontier
from tests.test_train import tiny_data
from train.rl import train_rl
from train.train import train


def _smoke_rl_dir(tmp_path):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"
    train_rl("full", seed=0, init_dir=run, updates=4, batch_size=8, eval_every=2, data=data, log=lambda row: None)
    return run / "rl", data


def test_frontier_rows_and_files(tmp_path):
    rl_dir, data = _smoke_rl_dir(tmp_path)

    df = frontier(rl_dir, data=data)
    by_type(rl_dir, data=data)

    assert set(df.columns) >= {"source", "lam", "tau", "halt_tau", "quality", "cost_usd", "reward"}
    assert (df["source"] == "rl").sum() == len(LAMS)
    assert (df["source"] == "lam_oracle").sum() == len(LAMS)
    # no tau_sweep.csv was written for this smoke run, so no supervised_tau rows
    assert (df["source"] == "supervised_tau").sum() == 0

    assert (rl_dir / "rl_frontier.csv").exists()
    assert (rl_dir / "rl_by_type.csv").exists()
    assert (rl_dir / "rl_frontier.png").exists()


def test_oracle_bounds_rl_reward(tmp_path):
    rl_dir, data = _smoke_rl_dir(tmp_path)

    df = frontier(rl_dir, data=data)

    rl_rows = df[df["source"] == "rl"].set_index("lam")
    oracle_rows = df[df["source"] == "lam_oracle"].set_index("lam")

    for lam in LAMS:
        assert oracle_rows.loc[lam, "reward"] >= rl_rows.loc[lam, "reward"] - 1e-9


def test_by_type_table(tmp_path):
    rl_dir, data = _smoke_rl_dir(tmp_path)

    df = by_type(rl_dir, data=data)

    assert {"ep_type", "lam", "quality", "cost_usd", "reward"} <= set(df.columns)
    assert set(df["ep_type"].unique()) == {s.ep_type for s in data.val}
    assert len(df) == len(LAMS) * df["ep_type"].nunique()


def test_frontier_includes_supervised_tau_when_present(tmp_path):
    rl_dir, data = _smoke_rl_dir(tmp_path)
    init_dir = rl_dir.parent
    tau_sweep = pd.DataFrame([
        {"tau": 0.5, "halt_tau": 0.1, "quality": 0.5, "cost_usd": 0.001, "gap_closed": 0.1,
         "escalation_rate": 0.1, "choice_entropy": 0.5, "top_choice_share": 0.5},
        {"tau": 0.9, "halt_tau": 0.3, "quality": 0.7, "cost_usd": 0.002, "gap_closed": 0.3,
         "escalation_rate": 0.2, "choice_entropy": 0.4, "top_choice_share": 0.6},
    ])
    tau_sweep.to_csv(init_dir / "tau_sweep.csv", index=False)

    df = frontier(rl_dir, data=data)

    sup = df[df["source"] == "supervised_tau"]
    assert len(sup) == 2
    assert set(sup["tau"]) == {0.5, 0.9}
    assert sup["quality"].notna().all() and sup["cost_usd"].notna().all()
    assert sup["lam"].isna().all() and sup["reward"].isna().all()
