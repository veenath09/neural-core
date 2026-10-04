"""
Purpose:  Choose the decision thresholds of a trained controller on validation episodes, before any test
          number is looked at, and save the whole quality-cost curve.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   runs/<run>/ (model.pt, metrics.json from train.train), the run's pool (data_dir in metrics.json).
Outputs:  runs/<run>/tau_sweep.csv  one row per (tau, halt_tau): val quality, cost, gap closed, choice spread
          runs/<run>/tau.json       the chosen thresholds and the rule that chose them
Notes:    tau: pick the cheapest model whose predicted success is >= tau (else the most likely one).
          halt_tau: escalate after round 1 when the halt head's probability exceeds it.
          Rule, fixed before test: highest val quality; among settings within QUALITY_TOL of it, the cheapest.
          Gap closed = (quality - always-cheapest) / (oracle - always-cheapest) on the same episodes.

    python -m eval.tau_sweep runs/depth/full-s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.policies import AlwaysCheapest
from eval.policies import evaluate as evaluate_policies
from train.train import evaluate, load_data, load_model, reference_points

TAUS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
HALT_TAUS = (0.1, 0.2, 0.3, 0.5, 0.7)
QUALITY_TOL = 0.002


def choose(curve: pd.DataFrame, tol: float = QUALITY_TOL) -> pd.Series:
    near = curve[curve["quality"] >= curve["quality"].max() - tol]
    return near.sort_values(["cost_usd", "tau", "halt_tau"]).iloc[0]


def sweep(run_dir: Path, taus=TAUS, halt_taus=HALT_TAUS) -> pd.DataFrame:
    summary = json.loads((run_dir / "metrics.json").read_text())
    data = load_data(Path(summary["data_dir"]))
    model = load_model(run_dir, data)
    ref = reference_points(data.val, data.world, data.data_dir)
    floor = float(evaluate_policies(data.val, data.world.table, data.world.model_ids, [AlwaysCheapest()])["quality"].mean())
    rows = []
    for tau in taus:
        for halt_tau in halt_taus:
            r = evaluate(model, data.val, data.world, tau=tau, halt_tau=halt_tau)
            rows.append({"tau": tau, "halt_tau": halt_tau, "quality": r["quality"], "cost_usd": r["cost_usd"],
                         "gap_closed": (r["quality"] - floor) / max(1e-9, ref["oracle_quality"] - floor),
                         "escalation_rate": r["escalation_rate"], "choice_entropy": r["choice_entropy"],
                         "top_choice_share": max(r["choice_share"])})
            print(json.dumps(rows[-1]), flush=True)
    curve = pd.DataFrame(rows)
    curve.to_csv(run_dir / "tau_sweep.csv", index=False)
    best = choose(curve)
    (run_dir / "tau.json").write_text(json.dumps({
        "tau": float(best["tau"]), "halt_tau": float(best["halt_tau"]), "val_quality": float(best["quality"]),
        "val_cost_usd": float(best["cost_usd"]), "rule": f"max val quality, ties within {QUALITY_TOL} -> cheapest",
        "oracle": ref, "always_cheapest_quality": floor}, indent=2))
    return curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    for d in ap.parse_args().run_dirs:
        sweep(d)
        print(d, (d / "tau.json").read_text())
