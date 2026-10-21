"""Tests for the RL simulator loop, safety guard, and checkpoint compatibility (C2-07)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import train.rl as rl_module
from measurement.rl_reward import load_cost_ref
from tests.test_train import tiny_data
from train.baseline import BucketedBaseline
from train.rl import train_rl
from train.safety import SafetyGuard
from train.train import load_model, train


def test_safety_guard():
    guard = SafetyGuard(reference_reward=0.7, tol=0.05)
    assert guard.ok({0: 0.66, 1: 0.71}) is True
    assert guard.ok({0: 0.64}) is False


def test_safety_guard_per_lambda_dict_reference():
    guard = SafetyGuard(reference_reward={0: 0.6, 1: 0.9}, tol=0.05)
    # lambda 0's own reference (0.6) is what it's checked against, not lambda 1's (0.9)
    assert guard.ok({0: 0.56}) is True
    assert guard.ok({1: 0.84}) is False
    assert guard.ok({0: 0.56, 1: 0.86}) is True
    assert guard.ok({0: 0.5, 1: 0.86}) is False


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


def test_guard_uses_reward_not_quality(tmp_path, monkeypatch):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"

    real_evaluate_rl = rl_module.evaluate_rl
    calls = {"n": 0}

    def fake_evaluate_rl(model, skeletons, world, lam_idx, cost_ref, batch_size=512):
        out = real_evaluate_rl(model, skeletons, world, lam_idx, cost_ref, batch_size=batch_size)
        if calls["n"] >= len(rl_module.LAMS):
            # Quality far below the (measured) reference, reward far above it: must not roll back.
            out = dict(out, quality=0.0, reward=10.0)
        calls["n"] += 1
        return out

    monkeypatch.setattr(rl_module, "evaluate_rl", fake_evaluate_rl)
    result = train_rl("full", seed=0, init_dir=run, updates=2, batch_size=8, eval_every=2, data=data,
                       log=lambda row: None)
    check_rows = [r for r in result["history"] if "val" in r]
    assert len(check_rows) == 2  # update 0 (reference measurement) and update 2
    assert "rollback" not in check_rows[-1]["alarms"], \
        "low quality but high reward must not trigger a rollback"

    calls["n"] = 0

    def fake_evaluate_rl_high_quality_low_reward(model, skeletons, world, lam_idx, cost_ref, batch_size=512):
        out = real_evaluate_rl(model, skeletons, world, lam_idx, cost_ref, batch_size=batch_size)
        if calls["n"] >= len(rl_module.LAMS):
            # Quality far above the reference, reward far below it: must roll back.
            out = dict(out, quality=1.0, reward=-10.0)
        calls["n"] += 1
        return out

    monkeypatch.setattr(rl_module, "evaluate_rl", fake_evaluate_rl_high_quality_low_reward)
    result2 = train_rl("full", seed=0, init_dir=run, updates=2, batch_size=8, eval_every=2, data=data,
                        log=lambda row: None)
    check_rows2 = [r for r in result2["history"] if "val" in r]
    assert len(check_rows2) == 2
    assert "rollback" in check_rows2[-1]["alarms"], \
        "high quality but low reward must trigger a rollback"


def test_train_rl_reads_data_dir_from_init_dir(tmp_path, monkeypatch):
    data = tiny_data()
    pool_dir = tmp_path / "some_pool_dir"
    pool_dir.mkdir()
    data.data_dir = pool_dir  # tiny_data() has no real files on disk; only used as a cache dir here
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"

    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["data_dir"] == str(data.data_dir)

    captured = {}
    real_load_data = rl_module.load_data

    def fake_load_data(data_dir):
        captured["data_dir"] = Path(data_dir)
        return data  # reuse the same tiny_data() Data so the rest of the loop still runs

    monkeypatch.setattr(rl_module, "load_data", fake_load_data)
    train_rl("full", seed=0, init_dir=run, updates=1, batch_size=8, eval_every=1, data=None, log=lambda row: None)

    assert captured["data_dir"] == Path(metrics["data_dir"])
    rl_module.load_data = real_load_data  # not strictly needed (monkeypatch undoes it), but explicit


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
    # No override given: the reference is measured per-lambda at update 0, so that check
    # always passes by construction and best_update/best_mean_val_reward are set from it.
    assert metrics["reference_metric"] == "reward"
    assert set(metrics["reference_by_lam"].keys()) == {str(i) for i in range(5)}
    assert metrics["best_update"] is not None
    assert math.isfinite(metrics["best_mean_val_reward"])

    loaded = load_model(rl_dir, data)  # must load strictly
    assert loaded is not None

    # train_rl's return value also carries the live final state (not written to metrics.json),
    # used by callers (and tests) that need to tell "restored to last_good" from "kept training".
    assert set(result["final_model_state"].keys()) == set(loaded.state_dict().keys())
    assert "table" in result["final_baseline_state"]

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
    start_baseline = BucketedBaseline()  # zeros at the start, matching train_rl's own initial baseline

    rows = []
    result = train_rl("full", seed=0, init_dir=run, updates=4, batch_size=8, eval_every=2, data=data,
                       reference_reward=1.0, log=rows.append)

    check_rows = [r for r in result["history"] if "val" in r]
    assert check_rows, "expected at least one check row"
    lrs = [r["lr"] for r in check_rows]
    for prev, cur in zip(lrs, lrs[1:]):
        assert cur == pytest.approx(prev / 2)
    for r in check_rows:
        assert "rollback" in r["alarms"]

    # The checkpoint on disk is the supervised start (expected, since no check ever passed).
    final_sd = torch.load(run / "rl" / "model.pt", weights_only=True)
    for k in start_sd:
        assert torch.allclose(start_sd[k], final_sd[k]), f"{k} changed despite guard never passing"

    # The *live* model/baseline at the end of training must also equal the supervised start:
    # this is the part a deleted restore-on-rollback would break without the checkpoint test
    # above catching it (the checkpoint is written once at the start regardless).
    for k in start_sd:
        assert torch.allclose(start_sd[k], result["final_model_state"][k]), \
            f"live model {k} changed despite guard never passing"
    assert torch.allclose(result["final_baseline_state"]["table"], start_baseline.table)
