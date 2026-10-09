"""Tests for the RL simulator loop, safety guard, and checkpoint compatibility (C2-07)."""

import json
import math

import numpy as np
import pytest
import torch

from measurement.rl_reward import load_cost_ref
from tests.test_train import tiny_data
from train.baseline import BucketedBaseline
from train.rl import train_rl
from train.safety import SafetyGuard
from train.train import load_model, train


def test_safety_guard():
    guard = SafetyGuard(reference_quality=0.7, tol=0.05)
    assert guard.ok({0: 0.66, 1: 0.71}) is True
    assert guard.ok({0: 0.64}) is False


def test_load_old_checkpoint_without_lam_adj(tmp_path):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"

    sd = torch.load(run / "model.pt", weights_only=True)
    old_sd = {k: v for k, v in sd.items() if not k.startswith("heads.lam_adj.")}
    torch.save(old_sd, run / "model.pt")

    # strict=False loads fine when only heads.lam_adj.* is missing
    model = load_model(run, data, strict=False)
    assert model is not None

    # strict=True still raises on the same missing keys
    with pytest.raises(RuntimeError):
        load_model(run, data, strict=True)

    # deleting an unrelated key must raise even under strict=False
    bad_sd = {k: v for k, v in old_sd.items() if k != "heads.halt.weight"}
    torch.save(bad_sd, run / "model.pt")
    with pytest.raises(RuntimeError):
        load_model(run, data, strict=False)


def test_train_rl_smoke(tmp_path):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"

    result = train_rl("full", seed=0, init_dir=run, updates=4, batch_size=8, eval_every=2, data=data,
                       log=lambda row: None)

    rl_dir = run / "rl"
    assert (rl_dir / "model.pt").exists()
    assert (rl_dir / "baseline.pt").exists()
    assert (rl_dir / "metrics.json").exists()

    metrics = json.loads((rl_dir / "metrics.json").read_text())
    assert metrics["stage"] == "rl_sim"
    assert metrics["data_dir"] is not None and isinstance(metrics["data_dir"], str)

    loaded = load_model(rl_dir, data)  # must load strictly
    assert loaded is not None

    def check_finite(x):
        if isinstance(x, dict):
            for v in x.values():
                check_finite(v)
        elif isinstance(x, list):
            for v in x:
                check_finite(v)
        elif isinstance(x, (int, float)):
            assert math.isfinite(x), f"non-finite value: {x}"

    check_finite(result["history"])


def test_rollback_restores_weights_and_halves_lr(tmp_path):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"

    start_sd = {k: v.clone() for k, v in torch.load(run / "model.pt", weights_only=True).items()}

    rows = []
    result = train_rl("full", seed=0, init_dir=run, updates=4, batch_size=8, eval_every=2, data=data,
                       reference_quality=1.0, log=rows.append)

    check_rows = [r for r in result["history"] if "val" in r]
    assert check_rows, "expected at least one check row"
    lrs = [r["lr"] for r in check_rows]
    for prev, cur in zip(lrs, lrs[1:]):
        assert cur == pytest.approx(prev / 2)
    for r in check_rows:
        assert "rollback" in r["alarms"]

    final_sd = torch.load(run / "rl" / "model.pt", weights_only=True)
    for k in start_sd:
        assert torch.allclose(start_sd[k], final_sd[k]), f"{k} changed despite guard never passing"
