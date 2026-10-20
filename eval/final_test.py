"""
Purpose:  The frozen final evaluation on the TEST split (docs/analysis_plan.md): each supervised run at the
          thresholds it chose on val, the baselines and the oracle, and the paired full vs context_free contrast,
          aggregated over seeds with 95% t-intervals.
Owner:    C3 Measurement (K V A Bethmi)
Contract: v1
Inputs:   runs/<pool>/<config>-s<seed>/ (model.pt, metrics.json, tau.json from eval.tau_sweep).
Outputs:  <out>/per_run.csv (one row per run: quality, cost, reward-free metrics, per episode type),
          <out>/per_episode.parquet (paired contrasts), <out>/summary.json (seed means and CIs, contrasts,
          baselines, oracle).
Notes:    Test episodes are evaluated once, greedily, at each run's val-selected tau / halt_tau; nothing here is
          tuned. CI over seeds: mean +- t(0.975, n-1) * sd / sqrt(n). The paired contrast is per test episode,
          same seed, then averaged over seeds with the same t-interval on the per-seed mean differences.

    python -m eval.final_test --pool-dir runs/routerbench --configs full context_free --seeds 0 1 2 \
        --out results/final/routerbench
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from data.episodes import load_skeletons
from eval.policies import AlwaysCheapest, AlwaysStrongest, RandomPolicy
from eval.policies import evaluate as evaluate_policies
from train.rollout import rollout
from train.train import load_model, load_run_data, make_episodes, reference_points


def t_interval(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return {"mean": float(x.mean()), "ci95": None, "n": int(len(x))}
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return {"mean": float(x.mean()), "ci95": [float(x.mean() - half), float(x.mean() + half)], "n": int(len(x))}


def run_on_test(run_dir: Path, batch_size: int = 512) -> pd.DataFrame:
    """Per-episode quality and cost of one run on the test split at its val-selected thresholds."""
    summary = json.loads((run_dir / "metrics.json").read_text())
    tau = json.loads((run_dir / "tau.json").read_text())
    data = load_run_data(summary)
    model = load_model(run_dir, data).eval()
    test = load_skeletons("test", data.data_dir)
    rows = []
    with torch.no_grad():
        for s in range(0, len(test), batch_size):
            chunk = test[s:s + batch_size]
            out = rollout(model, make_episodes(chunk, data.world), data.world, tau=tau["tau"], halt_tau=tau["halt_tau"])
            rows += [{"episode_id": r.episode_id, "ep_type": r.ep_type, "quality": r.quality, "cost_usd": r.cost_usd}
                     for r in out.results]
    return pd.DataFrame(rows).assign(config=summary["config"], seed=summary["seed"], tau=tau["tau"],
                                     halt_tau=tau["halt_tau"])


def baselines_on_test(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "metrics.json").read_text())
    data = load_run_data(summary)
    test = load_skeletons("test", data.data_dir)
    df = evaluate_policies(test, data.world.table, data.world.model_ids,
                           [AlwaysCheapest(), AlwaysStrongest(), RandomPolicy(0)], hard_p=data.world.hard_p)
    out = {p: {"quality": float(g["quality"].mean()), "cost_usd": float(g["cost_usd"].mean())}
           for p, g in df.groupby("policy")}
    ref = reference_points(test, data.world, data.data_dir)
    out["oracle"] = {"quality": ref["oracle_quality"], "cost_usd": ref["oracle_cost"]}
    return out


def evaluate(pool_dir: Path, configs: list[str], seeds: list[int], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    per_ep = pd.concat([run_on_test(pool_dir / f"{c}-s{s}") for c in configs for s in seeds
                        if (pool_dir / f"{c}-s{s}" / "tau.json").exists()], ignore_index=True)
    per_ep.to_parquet(out / "per_episode.parquet", index=False)
    per_run = per_ep.groupby(["config", "seed"]).agg(quality=("quality", "mean"), cost_usd=("cost_usd", "mean"),
                                                     tau=("tau", "first"), halt_tau=("halt_tau", "first")).reset_index()
    by_type = per_ep.pivot_table(index=["config", "seed"], columns="ep_type", values="quality").reset_index()
    per_run.merge(by_type, on=["config", "seed"]).to_csv(out / "per_run.csv", index=False)
    summary = {"pool_dir": str(pool_dir), "seeds": sorted(per_ep["seed"].unique().tolist()), "configs": {}}
    for c, g in per_run.groupby("config"):
        summary["configs"][c] = {"quality": t_interval(g["quality"]), "cost_usd": t_interval(g["cost_usd"]),
                                 "by_type": {t: t_interval(by_type[by_type.config == c][t])
                                             for t in by_type.columns if t not in ("config", "seed")}}
    if {"full", "context_free"} <= set(per_ep["config"]):
        wide = per_ep.pivot_table(index=["seed", "episode_id", "ep_type"], columns="config", values="quality").dropna()
        wide["diff"] = wide["full"] - wide["context_free"]
        per_seed = wide.groupby("seed")["diff"].mean()
        summary["full_minus_context_free"] = {
            "overall": t_interval(per_seed.to_numpy()),
            "per_seed": {int(k): float(v) for k, v in per_seed.items()},
            "by_type": {t: t_interval(g.groupby("seed")["diff"].mean().to_numpy())
                        for t, g in wide.reset_index().groupby("ep_type")}}
        costs = per_ep.pivot_table(index=["seed", "episode_id"], columns="config", values="cost_usd").dropna()
        summary["full_minus_context_free"]["cost_usd"] = t_interval(
            (costs["full"] - costs["context_free"]).groupby("seed").mean().to_numpy())
    first = next(p for p in (pool_dir / f"{c}-s{s}" for c in configs for s in seeds) if (p / "tau.json").exists())
    summary["baselines"] = baselines_on_test(first)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", default=["full", "context_free"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.pool_dir, a.configs, a.seeds, a.out), indent=2))


if __name__ == "__main__":
    main()
