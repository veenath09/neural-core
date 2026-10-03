"""Supervised training (plan §6, D1, D11).

    python -m train.train --config full --seed 0
    python -m train.train --config full --seed 0 --data-dir data/processed/depth_pool --out-dir runs/depth

Episodes are rolled out live under a behaviour mixture; validation and checkpoints
use an EMA of the weights. Each epoch logs one JSON line with per-head losses, validation quality
and cost, the share of the oracle gap closed, how choices spread over the pool, and alarms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel

from controller.core import bp_schedule
from controller.model import CONFIGS, Controller, n_parameters
from data.embed import load_cache
from data.env import Episode
from data.episodes import Skeleton, load_skeletons
from data.routerbench import DATA_DIR, OutcomeTable, load_prepared
from eval.policies import AlwaysCheapest, EpisodeOracle
from eval.policies import evaluate as evaluate_policies
from train.rollout import World, rollout

EMA_DECAY = 0.999
COLLAPSE_SHARE = 0.85  # alarm if one option takes this share of first-round choices


def ema_decay(t: int) -> float:
    """Warm-up EMA decay: min(EMA_DECAY, (1+t)/(10+t)), t = updates already averaged (D11)."""
    return min(EMA_DECAY, (1 + t) / (10 + t))


def make_ema(model) -> AveragedModel:
    """AveragedModel with warm-up EMA decay (plan D11).

    torch's AveragedModel.update_parameters does the initial parameter copy on the call where
    n_averaged == 0 (no avg_fn call that time), then calls avg_fn with num_averaged = n_averaged
    (pre-increment) on every later call. So the first real averaging update after the initial
    copy is called with num_averaged == 1, and that call must use t=0 -- hence t = num_averaged - 1.
    """

    def avg_fn(averaged_param, model_param, num_averaged):
        t = int(num_averaged) - 1
        return averaged_param + (model_param - averaged_param) * (1 - ema_decay(t))

    return AveragedModel(model, avg_fn=avg_fn)


@dataclass
class Data:
    world: World
    train: list[Skeleton]
    val: list[Skeleton]
    embeddings: torch.Tensor
    cards: torch.Tensor
    n_types: int
    data_dir: Path | None = None  # set when loaded from disk; enables the oracle cache


def load_data(data_dir: Path = DATA_DIR) -> Data:
    prompts, cards = load_prepared(data_dir)
    emb, emb_index = load_cache(data_dir)
    ids = [c.model_id for c in cards]
    world = World(OutcomeTable.from_frame(prompts), ids, np.array([c.mean_cost_usd for c in cards]), emb_index,
                  {m: i for i, m in enumerate(ids)})
    return Data(world, load_skeletons("train", data_dir), load_skeletons("val", data_dir),
                torch.from_numpy(np.array(emb)), torch.tensor([c.capability for c in cards]),
                int(prompts["step_type"].max()) + 1, Path(data_dir))


def make_episodes(skeletons: Sequence[Skeleton], world: World) -> list[Episode]:
    return [Episode(sk, world.table, world.model_ids) for sk in skeletons]


def evaluate(model, skeletons: Sequence[Skeleton], world: World, tau: float = 0.5, batch_size: int = 512) -> dict:
    model.eval()
    results, firsts, n_halt, n_escalate = [], [], 0, 0
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            out = rollout(model, make_episodes(skeletons[start:start + batch_size], world), world, tau=tau)
            results += out.results
            for actions in out.decision_actions:
                firsts += [a["model"] for a in actions if a["phase"] == "choose" and a["model"] is not None]
                n_halt += sum(a["phase"] == "halt" for a in actions)
                n_escalate += sum(a["halt"] == "escalate" for a in actions)
    share = np.bincount(firsts, minlength=len(world.model_ids)) / max(1, len(firsts))
    nz = share[share > 0]
    return {"quality": float(np.mean([r.quality for r in results])),
            "cost_usd": float(np.mean([r.cost_usd for r in results])),
            "choice_share": share.round(4).tolist(),
            "choice_entropy": max(0.0, float(-(nz * np.log(nz)).sum() / math.log(len(share)))),  # 0 = one option, 1 = uniform
            "escalation_rate": n_escalate / max(1, n_halt)}


def reference_points(skeletons: Sequence[Skeleton], world: World, cache_dir: Path | None) -> dict:
    """Episode-oracle quality and cost on these skeletons, cached (keyed by the episode ids) because the
    exact search takes minutes on a large pool."""
    key = hashlib.blake2b("|".join(sk.episode_id for sk in skeletons).encode(), digest_size=6).hexdigest()
    cache = Path(cache_dir) / f"oracle_{key}.json" if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    df = evaluate_policies(skeletons, world.table, world.model_ids, [EpisodeOracle()])
    ref = {"n": len(skeletons), "oracle_quality": float(df["quality"].mean()), "oracle_cost": float(df["cost_usd"].mean())}
    if cache:
        cache.write_text(json.dumps(ref))
    return ref


def alarms(row: dict, model_ids: Sequence[str]) -> list[str]:
    out = []
    top = int(np.argmax(row["val_choice_share"]))
    if row["val_choice_share"][top] > COLLAPSE_SHARE:
        out.append(f"collapse: {model_ids[top]} takes {row['val_choice_share'][top]:.0%} of choices")
    if not all(math.isfinite(row[k]) for k in ("train_loss", "val_loss", "grad_norm")):
        out.append("non-finite loss or gradient")
    if row["val_loss"] - row["train_loss"] > 0.1:
        out.append(f"train/val gap {row['val_loss'] - row['train_loss']:.3f}")
    return out


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
    ema = make_ema(model)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_sk = data.train[:max_train] if max_train else data.train
    val_sk = data.val[:max_val] if max_val else data.val
    total_steps = epochs * math.ceil(len(train_sk) / batch_size)
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    run_dir = Path(out_dir) / f"{config}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale, step = [], math.inf, 0, 0
    ref = reference_points(val_sk, data.world, data.data_dir)
    floor = evaluate_policies(val_sk, data.world.table, data.world.model_ids, [AlwaysCheapest()])
    ref_floor = float(floor["quality"].mean())  # always-cheapest: 0% of the gap, oracle: 100%
    log({"reference": ref | {"always_cheapest_quality": ref_floor,
                             "always_cheapest_cost": float(floor["cost_usd"].mean())}})
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_sk))
        losses, parts, norms, bp = [], [], [], bp_min
        for start in range(0, len(order), batch_size):
            episodes = make_episodes([train_sk[k] for k in order[start:start + batch_size]], data.world)
            bp = bp_schedule(step, total_steps, bp_min=bp_min)
            out = rollout(model, episodes, data.world, policy="mixture", model_share=epoch > 0, rng=rng,
                          bp_steps=bp, compute_loss=True)
            opt.zero_grad()
            out.loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            parts.append(out.parts)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            losses.append(out.loss.item())
            step += 1
        val_loss = validation_loss(ema.module, val_sk, data.world)
        val = evaluate(ema.module, val_sk, data.world)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               **{f"train_{k}": float(np.mean([p[k] for p in parts])) for k in parts[0]},
               "val_quality": val["quality"], "val_cost": val["cost_usd"],
               "val_gap_closed": (val["quality"] - ref_floor) / max(1e-9, ref["oracle_quality"] - ref_floor),
               "val_escalation_rate": val["escalation_rate"], "val_choice_entropy": val["choice_entropy"],
               "val_choice_share": val["choice_share"], "grad_norm": float(np.mean(norms)),
               "lr": sched.get_last_lr()[0], "bp_steps": bp}
        row["alarms"] = alarms(row, data.world.model_ids)
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
               "data_dir": str(data.data_dir or DATA_DIR), "reference": ref,
               "model_config": asdict(CONFIGS[config]), "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(run_dir: Path, data: Data | None = None) -> Controller:
    summary = json.loads((Path(run_dir) / "metrics.json").read_text())
    data = data or load_data(Path(summary.get("data_dir", DATA_DIR)))
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
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR), help="pool: RouterBench (default) or depth_pool")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    a = parser.parse_args()
    train(a.config, a.seed, a.epochs, a.batch_size, bp_min=a.bp_min, max_train=a.max_train, max_val=a.max_val,
          out_dir=a.out_dir, data=load_data(a.data_dir), log=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
