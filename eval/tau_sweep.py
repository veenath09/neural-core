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
          Rule, fixed before test (decided 2026-10-21): the highest val quality among settings whose val cost
          is no higher than the random-option baseline's on the same episodes; ties within QUALITY_TOL go to
          the cheaper setting. If no setting is that cheap, the cheapest setting. Report the whole curve too.
          Gap closed = (quality - always-cheapest) / (oracle - always-cheapest) on the same episodes.

    python -m eval.tau_sweep runs/depth/full-s0
    python -m eval.tau_sweep --reselect runs/*/*-s0     # re-apply the rule to saved curves, no new rollouts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.policies import AlwaysCheapest, RandomPolicy
from eval.policies import evaluate as evaluate_policies
from train.train import evaluate, load_data, load_model, reference_points

TAUS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
HALT_TAUS = (0.1, 0.2, 0.3, 0.5, 0.7)
QUALITY_TOL = 0.002


RULE = "max val quality with val cost <= random-option baseline; ties within {tol} -> cheapest"


def choose(curve: pd.DataFrame, cost_cap: float = float("inf"), tol: float = QUALITY_TOL) -> pd.Series:
    affordable = curve[curve["cost_usd"] <= cost_cap]
    if affordable.empty:
        return curve.sort_values(["cost_usd", "tau", "halt_tau"]).iloc[0]
    near = affordable[affordable["quality"] >= affordable["quality"].max() - tol]
    return near.sort_values(["cost_usd", "tau", "halt_tau"]).iloc[0]


def baselines(data) -> dict:
    df = evaluate_policies(data.val, data.world.table, data.world.model_ids, [AlwaysCheapest(), RandomPolicy(0)])
    g = df.groupby("policy")[["quality", "cost_usd"]].mean()
    return {"always_cheapest_quality": float(g.loc["always_cheapest", "quality"]),
            "random_quality": float(g.loc["random", "quality"]), "random_cost_usd": float(g.loc["random", "cost_usd"])}


def write_choice(run_dir: Path, curve: pd.DataFrame, ref: dict, base: dict) -> dict:
    best = choose(curve, base["random_cost_usd"])
    out = {"tau": float(best["tau"]), "halt_tau": float(best["halt_tau"]), "val_quality": float(best["quality"]),
           "val_cost_usd": float(best["cost_usd"]), "rule": RULE.format(tol=QUALITY_TOL), "oracle": ref} | base
    (run_dir / "tau.json").write_text(json.dumps(out, indent=2))
    return out


def sweep(run_dir: Path, taus=TAUS, halt_taus=HALT_TAUS) -> pd.DataFrame:
    summary = json.loads((run_dir / "metrics.json").read_text())
    data = load_data(Path(summary["data_dir"]))
    model = load_model(run_dir, data)
    ref = reference_points(data.val, data.world, data.data_dir)
    base = baselines(data)
    floor = base["always_cheapest_quality"]
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
    write_choice(run_dir, curve, ref, base)
    return curve


def reselect(run_dir: Path) -> dict:
    """Apply the current rule to a saved curve (the curve itself does not depend on the rule)."""
    data = load_data(Path(json.loads((run_dir / "metrics.json").read_text())["data_dir"]))
    return write_choice(run_dir, pd.read_csv(run_dir / "tau_sweep.csv"),
                        reference_points(data.val, data.world, data.data_dir), baselines(data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--reselect", action="store_true", help="re-apply the rule to saved tau_sweep.csv files")
    ap.add_argument("--track", action="store_true", help="log the choice and curve to the run's MLflow run")
    a = ap.parse_args()
    for d in a.run_dirs:
        reselect(d) if a.reselect else sweep(d)
        print(d, (d / "tau.json").read_text())
        run_id = json.loads((d / "metrics.json").read_text()).get("mlflow_run_id")
        if a.track and run_id:
            from train.tracking import Tracker

            t = Tracker("", run_id=run_id)
            t.log({"tau": json.loads((d / "tau.json").read_text())})
            t.artifact(d / "tau_sweep.csv")
            t.artifact(d / "tau.json")
            t.end()
