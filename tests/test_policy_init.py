"""Tests for the calibrated RL starting point (pilot v3 deviation, train/policy_init.py)."""

import math

import numpy as np
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
