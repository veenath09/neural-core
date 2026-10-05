"""
Purpose:  Load runs that finished before MLflow tracking existed into MLflow, from their logs, so the
          MLflow UI and the Grafana dashboard show the whole history.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   runs/<pool>/<config>-s<seed>.log (+ metrics.json, tau.json) for supervised runs;
          runs/<pool>/<config>-s<seed>/<rl dir>.log for simulator RL runs.
Outputs:  one MLflow run per finished run, tagged stage / pool / config / seed (and rl_version for RL).
Notes:    Idempotent: each run carries a backfill_key tag and is skipped if that key already exists.
          Metric steps are the original epochs (supervised) or update numbers (RL).

    python -m monitoring.backfill
"""

from __future__ import annotations

import json
from pathlib import Path

from train.tracking import EXPERIMENT, Tracker, default_uri

ROOT = Path(__file__).resolve().parents[1]
POOLS = {"routerbench": "routerbench", "depth": "depth_pool", "depth_aug": "depth_pool_aug"}
RL_DIRS = ("rl", "rl_v3")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def _existing_keys() -> set[str]:
    import mlflow

    mlflow.set_tracking_uri(default_uri())
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return set()
    runs = mlflow.search_runs([exp.experiment_id], output_format="list", max_results=50000)
    return {r.data.tags.get("backfill_key") for r in runs}


def backfill_supervised(pool_dir: Path, pool: str, done: set[str]) -> int:
    n = 0
    for log in sorted(pool_dir.glob("*-s*.log")):
        run_dir = pool_dir / log.stem
        key = f"supervised:{pool_dir.name}/{log.stem}"
        if key in done or not (run_dir / "metrics.json").exists():
            continue
        summary = json.loads((run_dir / "metrics.json").read_text())
        config, seed = summary["config"], summary["seed"]
        t = Tracker(f"{pool_dir.name}/{log.stem}",
                    params={"config": config, "seed": seed, "data_dir": summary.get("data_dir"),
                            "model": summary.get("model_config", {})},
                    tags={"stage": "supervised", "pool": pool, "config": config, "seed": seed, "backfill_key": key})
        for row in _rows(log):
            if "reference" in row:
                t.log(row, step=0)
            elif "epoch" in row:
                t.log(row, step=row["epoch"])
        t.log({"best_val_loss": summary["best_val_loss"], "epochs_run": len(summary["history"]),
               "params": summary["params"]})
        if (run_dir / "tau.json").exists():
            t.log({"tau": json.loads((run_dir / "tau.json").read_text())})
        for name in ("metrics.json", "tau.json", "tau_sweep.csv"):
            t.artifact(run_dir / name)
        t.end()
        n += 1
    return n


def backfill_rl(pool_dir: Path, pool: str, done: set[str]) -> int:
    n = 0
    for run_dir in sorted(p for p in pool_dir.glob("*-s*") if p.is_dir()):
        for rl in RL_DIRS:
            log, out = run_dir / f"{rl}.log", run_dir / rl
            key = f"rl:{pool_dir.name}/{run_dir.name}/{rl}"
            if key in done or not log.exists() or not (out / "metrics.json").exists():
                continue
            summary = json.loads((out / "metrics.json").read_text())
            version = "v3" if rl == "rl_v3" else "v1"
            t = Tracker(f"{pool_dir.name}/{run_dir.name}/{rl}",
                        params={"config": summary["config"], "seed": summary["seed"], "lams": summary.get("lams"),
                                "cost_ref": summary.get("cost_ref"), "init_dir": summary.get("init_dir")},
                        tags={"stage": "rl_sim", "rl_version": version, "pool": pool, "config": summary["config"],
                              "seed": summary["seed"], "backfill_key": key})
            for row in _rows(log):
                if "update" in row:
                    t.log(row, step=row["update"])
                elif "calibrated_init" in row:
                    t.log({"calibrated_init": {k: v for k, v in row["calibrated_init"].items() if k != "fit"}}, step=0)
            for name in ("metrics.json", "rl_frontier.csv", "rl_frontier.png", "rl_by_type.csv"):
                t.artifact(out / name)
            t.end()
            n += 1
    return n


def main() -> None:
    done = _existing_keys()
    total = 0
    for name, pool in POOLS.items():
        d = ROOT / "runs" / name
        if d.exists():
            total += backfill_supervised(d, pool, done) + backfill_rl(d, pool, done)
    print(f"backfilled {total} runs into {default_uri()}")


if __name__ == "__main__":
    main()
