"""
Purpose:  Send training and evaluation numbers to MLflow, so runs can be compared in the MLflow UI and
          plotted live in Grafana (monitoring/), without changing what the scripts compute.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   rows the scripts already log (dicts of numbers, lists and an "alarms" list), run parameters.
Outputs:  an MLflow run per training run in monitoring/mlflow/mlflow.db (SQLite), artifacts in
          monitoring/mlflow/artifacts. Override with MLFLOW_TRACKING_URI.
Notes:    Opt-in: train.train and eval.tau_sweep create a Tracker only from their CLIs (--track), so
          tests and library calls never write a tracking database. If mlflow is not installed every
          method is a no-op. Lists become one metric per index (val_choice_share -> val_choice_share_3),
          nested dicts are flattened with "." and alarms are counted and kept as a tag.

    python -m train.train --config full --seed 0 --track
    mlflow ui --backend-store-uri sqlite:///monitoring/mlflow/mlflow.db
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "monitoring" / "mlflow"
EXPERIMENT = "neural-core"


def default_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{STORE / 'mlflow.db'}")


def flatten(row: Mapping, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a logged row, as MLflow metric names."""
    out: dict[str, float] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                out[name] = float(value)
        elif isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, (int, float)) for v in value):
            out.update({f"{name}_{i}": float(v) for i, v in enumerate(value)})
    return out


class Tracker:
    def __init__(self, run_name: str, params: Mapping | None = None, tags: Mapping | None = None,
                 run_id: str | None = None, experiment: str = EXPERIMENT, uri: str | None = None):
        try:
            import mlflow
        except ImportError:
            self.mlflow = None
            self.run_id = None
            return
        self.mlflow = mlflow
        uri = uri or default_uri()
        local_store = uri == f"sqlite:///{STORE / 'mlflow.db'}"
        if local_store:
            STORE.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            # artifacts sit next to a SQLite store (never in the working directory's ./mlruns)
            db = Path(uri[len("sqlite:///"):]) if uri.startswith("sqlite:///") else None
            mlflow.create_experiment(experiment, artifact_location=(db.parent / "artifacts").as_uri() if db else None)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name)
        self.run_id = run.info.run_id
        self._alarms: list[str] = []
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in flatten_params(params).items()})
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log(self, row: Mapping, step: int | None = None) -> None:
        if self.mlflow is None:
            return
        metrics = flatten({k: v for k, v in row.items() if k != "alarms"})
        if metrics:
            self.mlflow.log_metrics(metrics, step=step)
        alarms = row.get("alarms")
        if alarms is not None:
            self.mlflow.log_metric("alarm_count", float(len(alarms)), step=step)
            if alarms:
                self._alarms += [f"{step}: {a}" for a in alarms]
                self.mlflow.set_tag("alarms", " | ".join(self._alarms)[-4000:])

    def artifact(self, path: Path) -> None:
        if self.mlflow is not None and Path(path).exists():
            self.mlflow.log_artifact(str(path))

    def end(self, status: str = "FINISHED") -> None:
        if self.mlflow is not None:
            self.mlflow.end_run(status=status)


def flatten_params(params: Mapping, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, Mapping):
            out.update(flatten_params(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out
