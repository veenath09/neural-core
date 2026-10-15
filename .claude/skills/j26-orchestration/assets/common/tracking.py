"""
Purpose:  Minimal experiment tracking that every training and evaluation script uses,
          so the dashboard and the final figures read one consistent format.
Owner:    Shared
Contract: v1
Outputs:  runs/{run_id}/run.json            run metadata and status
          runs/{run_id}/metrics-*.parquet   long-format metric stream
          runs/{run_id}/events-*.parquet    rollbacks, early stops, alarms
Notes:    Long format (one row per metric value) so new metrics never change the schema.
          Files are written in parts so a crashed run still leaves readable data.
          Optional TensorBoard mirroring if `tensorboard` is installed.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

STATUSES = {"running", "done", "failed", "stopped"}


class RunLogger:
    def __init__(
        self,
        root: str | os.PathLike,
        run_id: str,
        component: str,          # C1 | C2 | C3 | C4
        config_name: str,        # full | gate_off | gate_sigmoid | memory_off | context_free | baseline name
        stage: str,              # supervised | rl | eval | data
        seed: int,
        config: dict[str, Any],
        flush_every: int = 500,
        tensorboard: bool = False,
    ) -> None:
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.flush_every = flush_every
        self._rows: list[dict] = []
        self._events: list[dict] = []
        self._part = 0
        self._t0 = time.time()
        self.meta = {
            "run_id": run_id, "component": component, "config_name": config_name,
            "stage": stage, "seed": seed, "config": config, "status": "running",
            "started_at": _now(), "ended_at": None, "contract_version": "v1",
        }
        self._write_meta()
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.dir / "tb"))
            except Exception:
                self._tb = None

    # -- metrics ------------------------------------------------------------
    def log(self, step: int, metrics: dict[str, float], split: str = "train") -> None:
        """Record scalar metrics at a training step. split: train | val | test."""
        wall = time.time() - self._t0
        for name, value in metrics.items():
            self._rows.append({
                "run_id": self.run_id, "step": int(step), "split": split,
                "metric": name, "value": float(value), "wall_s": wall,
            })
            if self._tb is not None:
                self._tb.add_scalar(f"{split}/{name}", float(value), int(step))
        if len(self._rows) >= self.flush_every:
            self.flush()

    def event(self, step: int, kind: str, detail: str = "") -> None:
        """Record a discrete event: rollback, early_stop, alarm, checkpoint, note."""
        self._events.append({
            "run_id": self.run_id, "step": int(step), "kind": kind,
            "detail": detail, "time": _now(),
        })

    # -- lifecycle ----------------------------------------------------------
    def flush(self) -> None:
        if self._rows:
            pd.DataFrame(self._rows).to_parquet(self.dir / f"metrics-{self._part:05d}.parquet", index=False)
            self._rows.clear()
        if self._events:
            pd.DataFrame(self._events).to_parquet(self.dir / f"events-{self._part:05d}.parquet", index=False)
            self._events.clear()
        self._part += 1

    def close(self, status: str = "done") -> None:
        assert status in STATUSES
        self.flush()
        self.meta["status"] = status
        self.meta["ended_at"] = _now()
        self._write_meta()
        if self._tb is not None:
            self._tb.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close("failed" if exc_type else "done")

    def _write_meta(self) -> None:
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=2, default=str))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
