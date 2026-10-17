"""Tests for the calibrated RL starting point (pilot v3 deviation, train/policy_init.py)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from measurement.rl_reward import LAMS
from tests.test_train import tiny_data
from train.policy_init import calibrate_policy_init, set_policy_init
from train.rl import evaluate_rl
from train.train import load_model, train


def _model(tmp_path):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    return load_model(tmp_path / "full-s0", data, strict=False), data


def test_set_policy_init_gives_a_price_weight_linear_in_log_lambda(tmp_path):
    model, _ = _model(tmp_path)
    set_policy_init(model, price=(0.2, -0.5), halt=(1.0, 0.25))
    d = model.heads.lam_adj.in_features - 2
    z = torch.randn(3, d)
    for lam in (0.1, 1.0, 2.0):
        raw = model.heads.lam_adj(torch.cat([z, torch.full((3, 1), math.log(lam)), torch.rand(3, 1)], dim=-1))
        assert torch.allclose(raw[:, 0], torch.full((3,), 0.2 - 0.5 * math.log(lam)), atol=1e-6)
        assert torch.allclose(raw[:, 1], torch.full((3,), 1.0 + 0.25 * math.log(lam)), atol=1e-6)


def test_calibration_never_starts_worse_than_the_default(tmp_path):
    model, data = _model(tmp_path)
    cost_ref = 1e-3
    sks = data.train[:32]
    default = np.mean([evaluate_rl(model, sks, data.world, i, cost_ref)["reward"] for i in range(len(LAMS))])
    report = calibrate_policy_init(model, sks, data.world, cost_ref, grid_w=(0.05, 0.5), grid_h=(0.0, 2.0))
    after = np.mean([evaluate_rl(model, sks, data.world, i, cost_ref)["reward"] for i in range(len(LAMS))])
    assert after >= default - 1e-9
    assert report["mean_reward"] == after or abs(report["mean_reward"] - after) < 1e-9
    assert set(report["per_lam"]) == set(range(len(LAMS)))
    # the price weight it reports is what the head applies
    lam = LAMS[-1]
    raw0 = model.heads.lam_adj.bias[0] + model.heads.lam_adj.weight[0, -2] * math.log(lam)
    assert abs(lam * F.softplus(raw0).item() - report["fitted_price_weight"][len(LAMS) - 1]) < 1e-5


def test_train_rl_with_calibrated_init_records_it(tmp_path):
    from train.rl import train_rl

    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    rows = []
    summary = train_rl("full", 0, tmp_path / "full-s0", updates=2, batch_size=16, eval_every=1, max_val=16, data=data,
                       log=rows.append, calibrate_init=True, calibrate_episodes=16)
    assert summary["calibrated_init"] is not None and "fit" in summary["calibrated_init"]
    assert any("calibrated_init" in r for r in rows)


def test_old_checkpoint_without_sharpness_loads_as_one(tmp_path):
    model, data = _model(tmp_path)
    sd = {k: v for k, v in model.state_dict().items() if k != "heads.log_sharpness"}
    fresh = load_model(tmp_path / "full-s0", data, strict=False)
    fresh.load_state_dict(sd)  # strict load of a pre-v4 state dict
    assert float(fresh.heads.log_sharpness) == 0.0


def test_sharpness_changes_sampling_not_serving(tmp_path):
    model, data = _model(tmp_path)
    sks, cost_ref = data.train[:32], 1e-3
    z = torch.randn(4, model.heads.lam_adj.in_features - 2)
    args = (z, torch.randn(4, 3), torch.randn(4), torch.randn(4), torch.ones(4), torch.tensor([0.0, 1.0, 2.0]))
    p1, h1 = model.heads.policy(*args)
    base = [evaluate_rl(model, sks, data.world, i, cost_ref)["reward"] for i in range(len(LAMS))]
    with torch.no_grad():
        model.heads.log_sharpness.fill_(math.log(10.0))
    p10, h10 = model.heads.policy(*args)
    assert torch.allclose(p10, 10 * p1, atol=1e-5) and torch.allclose(h10, 10 * h1, atol=1e-5)
    sharp = [evaluate_rl(model, sks, data.world, i, cost_ref)["reward"] for i in range(len(LAMS))]
    assert base == sharp  # greedy (serve) decisions are scale-free


def test_calibration_picks_the_smallest_sharpness_near_serve(tmp_path):
    model, data = _model(tmp_path)
    report = calibrate_policy_init(model, data.train[:32], data.world, 1e-3, grid_w=(0.05, 0.5), grid_h=(0.0,),
                                   grid_k=(1.0, 3.0, 10.0))
    k = math.exp(float(model.heads.log_sharpness))
    assert abs(k - report["sharpness"]) < 1e-6 and k in (1.0, 3.0, 10.0)
    gaps = report["sharpness_gap"]
    assert gaps[str(k)] <= 0.02 or k == 10.0
    smaller = [kk for kk in (1.0, 3.0, 10.0) if kk < k]
    assert all(gaps[str(kk)] > 0.02 for kk in smaller)


def test_guard_reference_is_the_supervised_start_and_bad_calibration_is_rejected(tmp_path, monkeypatch):
    import train.policy_init as pi
    from train.rl import train_rl

    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"
    supervised = load_model(run, data, strict=False)
    cost_ref = 1.756663098e-3  # conf/score.yaml (tiny data has no cost_ref.json)
    expected = {i: evaluate_rl(supervised, data.val[:16], data.world, i, cost_ref)["reward"] for i in range(len(LAMS))}
    default_bias = supervised.heads.lam_adj.bias.detach().clone()

    def sabotage(model, skeletons, world, cost_ref, **kw):  # only the cheapest (weakest) model, never escalate
        with torch.no_grad():
            model.heads.lam_adj.weight.zero_()
            model.heads.lam_adj.bias.copy_(torch.tensor([50.0, -50.0]))
        return {"kept_default": False, "default_mean_reward": 0.0, "mean_reward": 0.0, "fit": None,
                "fitted_price_weight": None, "lams_below_default": [], "sharpness": 1.0, "sharpness_gap": {},
                "sharpness_at_grid_edge": False}

    monkeypatch.setattr(pi, "calibrate_policy_init", sabotage)
    rows = []
    summary = train_rl("full", 0, run, updates=1, batch_size=8, eval_every=1, max_val=16, data=data,
                       log=rows.append, calibrate_init=True, calibrate_episodes=8, out_dir=tmp_path / "rl")
    assert {int(k): v for k, v in summary["reference_by_lam"].items()} == pytest.approx(expected)
    init = [r for r in rows if "calibrated_init" in r][0]
    assert init["calibrated_init"]["calibration_rejected"] and "calibration_rejected" in init["alarms"]
    saved = torch.load(tmp_path / "rl" / "model.pt", weights_only=True)
    assert torch.allclose(saved["heads.lam_adj.bias"], default_bias)  # reverted to the supervised start
