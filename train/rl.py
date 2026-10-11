"""Monte Carlo policy-gradient update, simulator RL loop and CLI (plan C2-05, C2-06, C2-07).

Loss = -mean(A.detach() * logp) over choice slots
       - beta * mean(entropy) over choice slots
       + COMPLEXITY_AUX * mean((complexity - band_target)**2) over valid slots
       + ANCHOR * (ANCHOR_MODEL * mean(sup_model) over valid slots
                   + ANCHOR_HALT * mean(sup_halt) over halt slots)   [anchor=True, Task 9]

The anchor keeps the supervised model/halt BCE losses on during RL (plan D13
deviation, recorded after the seed-0 pilot): without it, RL rewrote the
lambda-independent success/halt logits and collapsed to cheap models at every
lambda. rl_loss(..., anchor=False) drops the anchor term entirely (Part 2's
online replay, which has no labels live, calls it this way).

A = G - baseline[band, lam_idx], G = returns_to_go(reward), normalised over
choice slots with population std ((A - mean) / (std + 1e-8)).

load_cost_ref is Task 1's (measurement/rl_reward.py); re-exported here so
callers that only import from train.rl still get it.

train_rl starts from a supervised checkpoint (init_dir), rolls out batches of
episodes under the "sample" policy, applies rl_update, and every eval_every
updates checks greedy "serve" val quality at each lambda against that
lambda's own reference (SafetyGuard): the supervised start's own serve-mode
val quality at that lambda, measured once at update 0 (so the update-0 check
always passes by construction, unless reference_quality is overridden). A
check that passes snapshots the weights and baseline as last_good and saves
them if they give the best mean val reward so far; a check that fails
restores last_good, rebuilds the optimizer at half the lr (dropping stale
Adam moments from the rejected trajectory), and adds the alarm "rollback".

    python -m train.rl --config full --seed 0 --init runs/routerbench/full-s0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from controller.model import CONFIGS
from data.episodes import Skeleton
from data.routerbench import DATA_DIR
from measurement.rl_reward import LAMS, episode_reward, load_cost_ref  # re-exported; do not duplicate
from measurement.scorer import CompositeScorer
from train.baseline import BucketedBaseline
from train.rollout import LOSS_WEIGHTS, RLTrace, World, rollout
from train.safety import SafetyGuard
from train.train import Data, load_data, load_model, make_episodes

COMPLEXITY_AUX = 0.2
LOG_EVERY = 25

# Supervised anchor during simulator RL (plan D13 deviation, Task 9, seed-0 pilot):
# without it, RL rewrote the lambda-independent success/halt logits and collapsed
# to cheap models at every lambda. ANCHOR_MODEL/ANCHOR_HALT reuse the supervised
# weights (train/rollout.py:LOSS_WEIGHTS) rather than re-typing them.
ANCHOR = 1.0
ANCHOR_MODEL = LOSS_WEIGHTS["model"]
ANCHOR_HALT = LOSS_WEIGHTS["halt"]

__all__ = [
    "COMPLEXITY_AUX",
    "ANCHOR",
    "ANCHOR_MODEL",
    "ANCHOR_HALT",
    "load_cost_ref",
    "returns_to_go",
    "entropy_beta",
    "rl_loss",
    "rl_update",
    "evaluate_rl",
    "train_rl",
]


def returns_to_go(reward: torch.Tensor) -> torch.Tensor:
    """Undiscounted reward-to-go G_t = sum_{k>=t} r_k, computed right-to-left over T."""
    return torch.flip(torch.cumsum(torch.flip(reward, dims=[-1]), dim=-1), dims=[-1])


def entropy_beta(update: int, total: int, start: float = 0.01, end: float = 0.001) -> float:
    """Linear decay from start to end over [0, total], clamped outside that range."""
    if total <= 0:
        return end

    frac = update / total
    frac = max(0.0, min(1.0, frac))
    return start + (end - start) * frac


def rl_loss(
    trace: RLTrace, baseline: BucketedBaseline, beta: float, anchor: bool = True
) -> tuple[torch.Tensor, dict]:
    device = trace.logp.device

    logp = trace.logp
    entropy = trace.entropy
    complexity = trace.complexity
    reward = trace.reward.to(device)
    band = trace.band.to(device)
    band_target = trace.band_target.to(device)
    lam_idx = trace.lam_idx.to(device)
    choice = trace.choice.to(device)
    valid = trace.valid.to(device)

    returns = returns_to_go(reward)

    # Broadcast lam_idx [n] to [n, T] to index the baseline per decision slot.
    # baseline.value moves band/lam_idx to the baseline table's device internally
    # (which may differ from the trace's device); move the result back here.
    lam_idx_bt = lam_idx[:, None].expand_as(band)
    baseline_values = baseline.value(band, lam_idx_bt).to(device)

    advantage = returns - baseline_values

    if choice.any():
        chosen_adv = advantage[choice]
        mean = chosen_adv.mean()
        std = chosen_adv.std(unbiased=False)
        norm_advantage = (advantage - mean) / (std + 1e-8)

        pg_terms = -norm_advantage.detach() * logp
        pg_loss = pg_terms[choice].mean()

        entropy_term = -beta * entropy[choice].mean()
    else:
        pg_loss = torch.zeros((), device=device)
        entropy_term = torch.zeros((), device=device)

    if valid.any():
        complexity_mse = ((complexity - band_target) ** 2)[valid].mean()
    else:
        complexity_mse = torch.zeros((), device=device)

    anchor_term = torch.zeros((), device=device)

    if anchor:
        sup_model = trace.sup_model.to(device)
        sup_halt = trace.sup_halt.to(device)
        halt_slot = trace.halt_slot.to(device)

        mean_sup_model = sup_model[valid].mean() if valid.any() else torch.zeros((), device=device)
        mean_sup_halt = sup_halt[halt_slot].mean() if halt_slot.any() else torch.zeros((), device=device)

        anchor_term = ANCHOR * (ANCHOR_MODEL * mean_sup_model + ANCHOR_HALT * mean_sup_halt)

    loss = pg_loss + entropy_term + COMPLEXITY_AUX * complexity_mse + anchor_term

    parts = {
        "pg_loss": float(pg_loss.detach()),
        "complexity_mse": float(complexity_mse.detach()),
        "entropy": float(entropy[choice].mean().detach()) if choice.any() else 0.0,
        "episode_return": float(returns[:, 0].mean().detach()) if returns.shape[1] > 0 else 0.0,
        "anchor": float(anchor_term.detach()),
    }

    return loss, parts


def rl_update(
    model,
    trace: RLTrace,
    baseline: BucketedBaseline,
    opt: torch.optim.Optimizer,
    beta: float,
    anchor: bool = True,
) -> dict:
    device = trace.logp.device

    loss, parts = rl_loss(trace, baseline, beta, anchor=anchor)

    opt.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    reward = trace.reward.to(device)
    band = trace.band.to(device)
    lam_idx = trace.lam_idx.to(device)
    choice = trace.choice.to(device)

    returns = returns_to_go(reward)
    lam_idx_bt = lam_idx[:, None].expand_as(band)

    if choice.any():
        # baseline.update moves these to the baseline table's device internally.
        baseline.update(band[choice], lam_idx_bt[choice], returns[choice].detach())

    return {
        "loss": float(loss.detach()),
        "pg_loss": parts["pg_loss"],
        "complexity_mse": parts["complexity_mse"],
        "episode_return": parts["episode_return"],
        "entropy": parts["entropy"],
        "anchor": parts["anchor"],
        "grad_norm": float(grad_norm),
    }


def evaluate_rl(model, skeletons: list[Skeleton], world: World, lam_idx: int, cost_ref: float,
                batch_size: int = 512) -> dict:
    """Greedy ("serve") rollout of skeletons at a single lambda, scored with the composite
    scorer at that lambda. by_type groups quality/cost/reward by ep_type; complexity_mae is
    mean |sigmoid(complexity_logit) - band_target| over valid decision slots."""
    model.eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    lam = LAMS[lam_idx]
    qualities, costs, rewards, ep_types = [], [], [], []
    firsts, n_halt, n_escalate = [], 0, 0
    complexity_abs_err, n_valid = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            chunk = skeletons[start:start + batch_size]
            episodes = make_episodes(chunk, world)
            lam_idx_arr = np.full(len(chunk), lam_idx, dtype=int)
            out = rollout(model, episodes, world, policy="serve", lam_idx=lam_idx_arr, cost_ref=cost_ref)
            for r, ep in zip(out.results, episodes):
                qualities.append(r.quality)
                costs.append(r.cost_usd)
                ep_types.append(r.ep_type)
                rewards.append(episode_reward(scorer, r.quality, r.cost_usd, ep.n_steps, lam))
            for actions in out.decision_actions:
                firsts += [a["model"] for a in actions if a["phase"] == "choose" and a["model"] is not None]
                n_halt += sum(a["phase"] == "halt" for a in actions)
                n_escalate += sum(a["halt"] == "escalate" for a in actions)
            rl = out.rl
            if rl is not None and rl.valid.any():
                complexity = torch.sigmoid(rl.complexity)
                err = (complexity - rl.band_target).abs()[rl.valid]
                complexity_abs_err += float(err.sum())
                n_valid += int(rl.valid.sum())
    share = np.bincount(firsts, minlength=len(world.model_ids)) / max(1, len(firsts))
    by_type: dict[str, dict] = {}
    ep_types_arr = np.array(ep_types)
    qualities_arr, costs_arr, rewards_arr = np.array(qualities), np.array(costs), np.array(rewards)
    for t in sorted(set(ep_types)):
        mask = ep_types_arr == t
        by_type[t] = {"quality": float(qualities_arr[mask].mean()), "cost_usd": float(costs_arr[mask].mean()),
                      "reward": float(rewards_arr[mask].mean())}
    return {"quality": float(qualities_arr.mean()), "cost_usd": float(costs_arr.mean()),
            "reward": float(rewards_arr.mean()), "escalation_rate": n_escalate / max(1, n_halt),
            "choice_share": share.round(4).tolist(),
            "complexity_mae": complexity_abs_err / max(1, n_valid), "by_type": by_type}


def train_rl(config: str, seed: int, init_dir: Path, updates: int = 1000, batch_size: int = 256, lr: float = 1e-4,
            eval_every: int = 250, max_val: int | None = None, out_dir: Path | None = None, data: Data | None = None,
            reference_quality: float | dict[int, float] | None = None, log=print) -> dict:
    init_dir = Path(init_dir)
    out_dir = Path(out_dir) if out_dir is not None else init_dir / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        init_summary = json.loads((init_dir / "metrics.json").read_text())
        data = load_data(Path(init_summary.get("data_dir", DATA_DIR)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = load_model(init_dir, data, strict=False)
    model.train()
    baseline = BucketedBaseline()

    def make_opt(learning_rate: float) -> torch.optim.Optimizer:
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    opt = make_opt(lr)
    cost_ref = load_cost_ref()

    val_sk = data.val[:max_val] if max_val else data.val
    train_sk = data.train

    last_good = {"model": copy.deepcopy(model.state_dict()), "baseline": copy.deepcopy(baseline.state_dict())}
    best_mean_val_reward = None
    best_update = None

    def save_best(state_model: dict, state_baseline: dict) -> None:
        torch.save(state_model, out_dir / "model.pt")
        torch.save(state_baseline, out_dir / "baseline.pt")

    # Ensure a checkpoint exists from the start (supervised start), overwritten if a later
    # check passes and improves on it.
    save_best(last_good["model"], last_good["baseline"])

    history: list[dict] = []

    def measure_val(lam_idx: int) -> dict:
        return evaluate_rl(model, val_sk, data.world, lam_idx, cost_ref)

    def run_check(update: int) -> dict:
        nonlocal best_mean_val_reward, best_update, opt
        model.eval()
        val_by_lam = {lam_idx: measure_val(lam_idx) for lam_idx in range(len(LAMS))}
        model.train()
        quality_by_lam = {lam_idx: v["quality"] for lam_idx, v in val_by_lam.items()}
        mean_val_reward = float(np.mean([v["reward"] for v in val_by_lam.values()]))
        row = {"update": update, "val": val_by_lam, "mean_val_reward": mean_val_reward,
               "lr": opt.param_groups[0]["lr"], "alarms": []}
        if guard.ok(quality_by_lam):
            last_good["model"] = copy.deepcopy(model.state_dict())
            last_good["baseline"] = copy.deepcopy(baseline.state_dict())
            if best_mean_val_reward is None or mean_val_reward > best_mean_val_reward:
                best_mean_val_reward = mean_val_reward
                best_update = update
                save_best(last_good["model"], last_good["baseline"])
        else:
            model.load_state_dict(last_good["model"])
            baseline.load_state_dict(last_good["baseline"])
            new_lr = opt.param_groups[0]["lr"] / 2
            opt = make_opt(new_lr)  # fresh Adam moments at the halved lr
            row["lr"] = new_lr
            row["alarms"].append("rollback")
        return row

    # The per-lambda reference is the supervised start's own serve-mode val quality at each
    # lambda, measured once here (before the guard exists, so run_check can't be reused for
    # this first pass) -- unless reference_quality overrides it. With no override, update 0's
    # check below always passes by construction (it's measuring against itself).
    model.eval()
    measured_reference_by_lam = {lam_idx: measure_val(lam_idx)["quality"] for lam_idx in range(len(LAMS))}
    model.train()
    guard = SafetyGuard(reference_quality if reference_quality is not None else measured_reference_by_lam, tol=0.05)
    # reference_by_lam is what metrics.json records: the guard's actual per-lambda reference,
    # whether measured here or supplied as an override (a float override applies to every lambda).
    if isinstance(guard.reference_quality, dict):
        reference_by_lam = dict(guard.reference_quality)
    else:
        reference_by_lam = {lam_idx: guard.reference_quality for lam_idx in range(len(LAMS))}

    row0 = run_check(0)
    history.append(row0)
    log(row0)

    for update in range(1, updates + 1):
        idx = rng.choice(len(train_sk), size=batch_size, replace=False)
        batch = [train_sk[i] for i in idx]
        lam_idx = rng.integers(len(LAMS), size=batch_size)
        episodes = make_episodes(batch, data.world)
        out = rollout(model, episodes, data.world, policy="sample", rng=rng, bp_steps=5, lam_idx=lam_idx,
                      cost_ref=cost_ref)
        beta = entropy_beta(update, updates)
        step = rl_update(model, out.rl, baseline, opt, beta)

        if update % LOG_EVERY == 0:
            row = {"update": update, "train_return": step["episode_return"], "pg_loss": step["pg_loss"],
                   "complexity_mse": step["complexity_mse"], "entropy": step["entropy"], "anchor": step["anchor"],
                   "beta": beta, "lr": opt.param_groups[0]["lr"], "grad_norm": step["grad_norm"]}
            history.append(row)
            log(row)

        if update % eval_every == 0 or update == updates:
            row = run_check(update)
            history.append(row)
            log(row)

    summary = {"config": config, "seed": seed, "stage": "rl_sim", "init_dir": str(init_dir),
               "data_dir": str(data.data_dir or DATA_DIR), "lams": list(LAMS),
               "cost_ref": cost_ref, "reference_quality": reference_by_lam, "best_update": best_update,
               "best_mean_val_reward": best_mean_val_reward, "history": history,
               "final_model_state": model.state_dict(), "final_baseline_state": baseline.state_dict()}
    metrics_only = {k: v for k, v in summary.items() if k not in ("final_model_state", "final_baseline_state")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_only, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", required=True, type=Path, dest="init_dir")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-val", type=int)
    a = parser.parse_args()
    train_rl(a.config, a.seed, a.init_dir, updates=a.updates, batch_size=a.batch_size, max_val=a.max_val,
            log=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
