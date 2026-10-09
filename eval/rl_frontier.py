"""Quality-cost frontier evaluation for the simulator RL policy (plan C3).

Compares, at each lambda in the RL grid:
  - "rl": the trained RL policy in serve (greedy) mode, evaluate_rl (train.rl).
  - "lam_oracle": the exact lambda-weighted oracle on the same val skeletons
    (measurement/rl_reward.lam_oracle_points) -- a per-episode upper bound on reward,
    so lam_oracle.reward >= rl.reward for every lambda.
  - "supervised_tau": the supervised tau-sweep curve (eval/tau_sweep.py's tau_sweep.csv,
    written to the init_dir named by rl_dir/metrics.json), if present. Those rows have no
    lambda or reward (the tau sweep isn't lambda-conditioned), only quality/cost/tau/halt_tau.

Also tabulates quality/cost/reward per episode type (correlated/retry/propagation/random) per
lambda, by_type(rl_dir); run it on both a "full" and a "context_free" config's rl_dir and
compare the two tables for the null-context check.

    python -m eval.rl_frontier runs/routerbench/full-s0/rl [more_rl_dir ...] [--max-val N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from measurement.rl_reward import LAMS, lam_oracle_points
from train.rl import evaluate_rl
from train.train import Data, load_data, load_model

__all__ = ["LAMS", "frontier", "by_type"]


def _load_rl_context(rl_dir: Path, data: Data | None, max_val: int | None):
    rl_dir = Path(rl_dir)
    summary = json.loads((rl_dir / "metrics.json").read_text())
    if data is None:
        data = load_data(Path(summary["data_dir"]))
    model = load_model(rl_dir, data)
    val_sk = data.val[:max_val] if max_val else data.val
    cost_ref = summary["cost_ref"]
    return summary, data, model, val_sk, cost_ref


def frontier(rl_dir: Path, lams=LAMS, max_val: int | None = None, data: Data | None = None) -> pd.DataFrame:
    """Quality-cost frontier: rl vs lam_oracle (same val skeletons) vs supervised_tau (if present).

    Columns: source, lam, tau, halt_tau, quality, cost_usd, reward -- NaN where a field
    doesn't apply to that source. Writes rl_dir/rl_frontier.csv and rl_dir/rl_frontier.png.
    """
    rl_dir = Path(rl_dir)
    summary, data, model, val_sk, cost_ref = _load_rl_context(rl_dir, data, max_val)

    rows = []
    for lam_idx, lam in enumerate(lams):
        r = evaluate_rl(model, val_sk, data.world, lam_idx, cost_ref)
        rows.append({"source": "rl", "lam": lam, "tau": np.nan, "halt_tau": np.nan,
                     "quality": r["quality"], "cost_usd": r["cost_usd"], "reward": r["reward"]})

    oracle_cache_dir = data.data_dir if data.data_dir is not None else None
    oracle_df = lam_oracle_points(val_sk, data.world.table, data.world.model_ids, cost_ref, lams=lams,
                                  cache_dir=oracle_cache_dir)
    for _, r in oracle_df.iterrows():
        rows.append({"source": "lam_oracle", "lam": r["lam"], "tau": np.nan, "halt_tau": np.nan,
                     "quality": r["quality"], "cost_usd": r["cost_usd"], "reward": r["reward"]})

    init_dir = Path(summary["init_dir"])
    tau_sweep_path = init_dir / "tau_sweep.csv"
    if tau_sweep_path.exists():
        tau_df = pd.read_csv(tau_sweep_path)
        for _, r in tau_df.iterrows():
            rows.append({"source": "supervised_tau", "lam": np.nan, "tau": r["tau"], "halt_tau": r["halt_tau"],
                         "quality": r["quality"], "cost_usd": r["cost_usd"], "reward": np.nan})

    df = pd.DataFrame(rows, columns=["source", "lam", "tau", "halt_tau", "quality", "cost_usd", "reward"])
    df.to_csv(rl_dir / "rl_frontier.csv", index=False)
    _plot_frontier(df, rl_dir / "rl_frontier.png")
    return df


def _plot_frontier(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rl": "#0d7a6f", "lam_oracle": "#3a55a4", "supervised_tau": "#b2552a"}
    markers = {"rl": "o", "lam_oracle": "s", "supervised_tau": "^"}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for source, g in df.groupby("source"):
        g = g.sort_values("cost_usd")
        ax.plot(g["cost_usd"], g["quality"], color=colors.get(source, "0.4"),
               marker=markers.get(source, "o"), label=source)
    ax.set_xlabel("cost per episode (USD)")
    ax.set_ylabel("quality")
    ax.set_title("Quality-cost frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_type(rl_dir: Path, lams=LAMS, max_val: int | None = None, data: Data | None = None) -> pd.DataFrame:
    """Quality/cost/reward per ep_type per lambda for the rl policy. Writes rl_dir/rl_by_type.csv."""
    rl_dir = Path(rl_dir)
    summary, data, model, val_sk, cost_ref = _load_rl_context(rl_dir, data, max_val)

    rows = []
    for lam_idx, lam in enumerate(lams):
        r = evaluate_rl(model, val_sk, data.world, lam_idx, cost_ref)
        for ep_type, vals in r["by_type"].items():
            rows.append({"ep_type": ep_type, "lam": lam, "quality": vals["quality"],
                         "cost_usd": vals["cost_usd"], "reward": vals["reward"]})

    df = pd.DataFrame(rows, columns=["ep_type", "lam", "quality", "cost_usd", "reward"])
    df.to_csv(rl_dir / "rl_by_type.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rl_dirs", nargs="+", type=Path)
    parser.add_argument("--max-val", type=int)
    a = parser.parse_args()
    for rl_dir in a.rl_dirs:
        frontier(rl_dir, max_val=a.max_val)
        by_type(rl_dir, max_val=a.max_val)
        print(rl_dir)


if __name__ == "__main__":
    main()
