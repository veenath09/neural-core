import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from data.episodes import generate_skeletons
from tests.conftest import make_table
from tests.test_env import IDS
from train.rollout import World
from train.train import Data, ema_decay, load_model, make_ema, train


def tiny_data():
    table = make_table(n=200, seed=11)
    world = World(table, IDS, np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4,
                  {p: i for i, p in enumerate(table.prompt_ids)}, {m: i for i, m in enumerate(IDS)})
    sks = generate_skeletons(table, list(table.prompt_ids), n=96, seed=0, split="t")
    torch.manual_seed(0)
    return Data(world, sks[:64], sks[64:], torch.randn(200, 384), torch.rand(4, 6), n_types=2)


def test_training_writes_checkpoint_and_metrics(tmp_path):
    data = tiny_data()
    summary = train("full", seed=0, epochs=2, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"
    assert len(summary["history"]) == 2 and (run / "model.pt").exists()
    assert json.loads((run / "metrics.json").read_text())["params"] == summary["params"]
    assert all(np.isfinite(h["val_loss"]) and 0 <= h["val_quality"] <= 1 for h in summary["history"])
    model = load_model(run, data)
    assert model.cfg.gate == "cfc"


def test_training_reduces_loss(tmp_path):
    summary = train("memory_off", seed=0, epochs=6, batch_size=32, lr=3e-3, patience=10, data=tiny_data(),
                    out_dir=tmp_path, log=lambda row: None)
    history = summary["history"]
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_ema_decay_has_warmup_and_caps_at_0_999():
    assert ema_decay(0) == pytest.approx(0.1)
    assert ema_decay(10) == pytest.approx(11 / 20)
    assert ema_decay(10_000) == pytest.approx(0.999)


def test_ema_update_uses_warmup_decay_not_fixed_0_999():
    torch.manual_seed(0)
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    ema = make_ema(model)
    ema.update_parameters(model)  # initial copy (n_averaged 0 -> 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 5.0]]))
    ema.update_parameters(model)  # first real average: t=0, decay=0.1
    expected = torch.tensor([[1.0, 2.0]]) + (torch.tensor([[3.0, 5.0]]) - torch.tensor([[1.0, 2.0]])) * (1 - 0.1)
    assert torch.allclose(ema.module.weight, expected)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[7.0, 11.0]]))
    ema.update_parameters(model)  # second real average: t=1, decay=2/11
    expected2 = expected + (torch.tensor([[7.0, 11.0]]) - expected) * (1 - 2 / 11)
    assert torch.allclose(ema.module.weight, expected2)
