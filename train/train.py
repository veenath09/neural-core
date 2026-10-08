"""Supervised training (plan §6, D1, D11).

    python -m train.train --config full --seed 0

Episodes are rolled out live under a behaviour mixture; validation and checkpoints
use an EMA of the weights."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from controller.core import bp_schedule
from controller.model import CONFIGS, Controller, n_parameters
from data.embed import load_cache
from data.env import Episode
from data.episodes import Skeleton, load_skeletons
from data.routerbench import DATA_DIR, OutcomeTable, load_prepared
from data.routerbench import SOURCES as BENCH_SOURCES
from train.rollout import World, rollout

EMA_DECAY = 0.999


@dataclass
class Data:
    world: World
    train: list[Skeleton]
    val: list[Skeleton]
    embeddings: torch.Tensor
    cards: torch.Tensor
    n_types: int


def load_data(data_dir: Path = DATA_DIR) -> Data:
    prompts, cards = load_prepared(data_dir)
    emb, emb_index = load_cache(data_dir)
    ids = [c.model_id for c in cards]
    world = World(OutcomeTable.from_frame(prompts), ids, np.array([c.mean_cost_usd for c in cards]), emb_index,
                  {m: i for i, m in enumerate(ids)})
    return Data(world, load_skeletons("train", data_dir), load_skeletons("val", data_dir),
                torch.from_numpy(np.array(emb)), torch.tensor([c.capability for c in cards]), len(BENCH_SOURCES))


def make_episodes(skeletons: Sequence[Skeleton], world: World) -> list[Episode]:
    return [Episode(sk, world.table, world.model_ids) for sk in skeletons]


def evaluate(model, skeletons: Sequence[Skeleton], world: World, tau: float = 0.5, batch_size: int = 512) -> dict:
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            results += rollout(model, make_episodes(skeletons[start:start + batch_size], world), world,
                               tau=tau).results
    return {"quality": float(np.mean([r.quality for r in results])),
            "cost_usd": float(np.mean([r.cost_usd for r in results]))}


def validation_loss(model, skeletons: Sequence[Skeleton], world: World, batch_size: int = 512) -> float:
    model.eval()
    rng = np.random.default_rng(1234)  # same behaviour draws every epoch
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            chunk = skeletons[start:start + batch_size]
            out = rollout(model, make_episodes(chunk, world), world, policy="mixture", rng=rng, compute_loss=True)
            total += out.loss.item() * len(chunk)
            count += len(chunk)
    return total / count


def train(config: str, seed: int = 0, epochs: int = 20, batch_size: int = 256, lr: float = 3e-4,
          weight_decay: float = 0.01, patience: int = 3, bp_min: int = 2, max_train: int | None = None,
          max_val: int | None = None, out_dir: Path = Path("runs"), data: Data | None = None,
          log: Callable[[dict], None] = print) -> dict:
    data = data or load_data()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Controller(CONFIGS[config], data.embeddings, data.cards, data.n_types)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_sk = data.train[:max_train] if max_train else data.train
    val_sk = data.val[:max_val] if max_val else data.val
    total_steps = epochs * math.ceil(len(train_sk) / batch_size)
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    run_dir = Path(out_dir) / f"{config}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale, step = [], math.inf, 0, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_sk))
        losses, bp = [], bp_min
        for start in range(0, len(order), batch_size):
            episodes = make_episodes([train_sk[k] for k in order[start:start + batch_size]], data.world)
            bp = bp_schedule(step, total_steps, bp_min=bp_min)
            out = rollout(model, episodes, data.world, policy="mixture", model_share=epoch > 0, rng=rng,
                          bp_steps=bp, compute_loss=True)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            losses.append(out.loss.item())
            step += 1
        val_loss = validation_loss(ema.module, val_sk, data.world)
        val = evaluate(ema.module, val_sk, data.world)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_quality": val["quality"], "val_cost": val["cost_usd"], "bp_steps": bp}
        history.append(row)
        log(row)
        if val_loss < best - 1e-4:
            best, stale = val_loss, 0
            torch.save(ema.module.state_dict(), run_dir / "model.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    summary = {"config": config, "seed": seed, "params": n_parameters(model), "best_val_loss": best,
               "model_config": asdict(CONFIGS[config]), "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(run_dir: Path, data: Data | None = None) -> Controller:
    data = data or load_data()
    summary = json.loads((Path(run_dir) / "metrics.json").read_text())
    model = Controller(CONFIGS[summary["config"]], data.embeddings, data.cards, data.n_types)
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", weights_only=True))
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--bp-min", type=int, default=2)
    a = parser.parse_args()
    train(a.config, a.seed, a.epochs, a.batch_size, bp_min=a.bp_min, max_train=a.max_train, max_val=a.max_val,
          log=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
