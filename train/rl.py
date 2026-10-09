"""Monte Carlo policy-gradient update, simulator RL loop and CLI (plan C2-05, C2-06, C2-07).

Loss = -mean(A.detach() * logp) over choice slots
       - beta * mean(entropy) over choice slots
       + COMPLEXITY_AUX * mean((complexity - band_target)**2) over valid slots

A = G - baseline[band, lam_idx], G = returns_to_go(reward), normalised over
choice slots with population std ((A - mean) / (std + 1e-8)).

load_cost_ref is Task 1's (measurement/rl_reward.py); re-exported here so
callers that only import from train.rl still get it.

train_rl starts from a supervised checkpoint (init_dir), rolls out batches of
episodes under the "sample" policy, applies rl_update, and every eval_every
updates checks greedy "serve" val quality at each lambda against the
supervised reference (SafetyGuard). A check that passes snapshots the weights
and baseline as last_good and saves them if they give the best mean val
reward so far; a check that fails restores last_good and halves the lr.

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
from train.rollout import RLTrace, World, rollout
from train.safety import SafetyGuard
from train.train import Data, evaluate, load_data, load_model, make_episodes

COMPLEXITY_AUX = 0.2
LOG_EVERY = 25

__all__ = [
    "COMPLEXITY_AUX",
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


def rl_loss(trace: RLTrace, baseline: BucketedBaseline, beta: float) -> tuple[torch.Tensor, dict]:
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

    loss = pg_loss + entropy_term + COMPLEXITY_AUX * complexity_mse

    parts = {
        "pg_loss": float(pg_loss.detach()),
        "complexity_mse": float(complexity_mse.detach()),
        "entropy": float(entropy[choice].mean().detach()) if choice.any() else 0.0,
        "episode_return": float(returns[:, 0].mean().detach()) if returns.shape[1] > 0 else 0.0,
    }

    return loss, parts


def rl_update(model, trace: RLTrace, baseline: BucketedBaseline, opt: torch.optim.Optimizer, beta: float) -> dict:
    device = trace.logp.device

    loss, parts = rl_loss(trace, baseline, beta)

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
            reference_quality: float | None = None, log=print) -> dict:
    init_dir = Path(init_dir)
    out_dir = Path(out_dir) if out_dir is not None else init_dir / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = data or load_data()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = load_model(init_dir, data, strict=False)
    model.train()
    baseline = BucketedBaseline()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    cost_ref = load_cost_ref()

    val_sk = data.val[:max_val] if max_val else data.val
    train_sk = data.train

    tau_path = init_dir / "tau.json"
    if tau_path.exists():
        tau_cfg = json.loads(tau_path.read_text())
        tau, halt_tau = tau_cfg.get("tau", 0.5), tau_cfg.get("halt_tau", 0.5)
    else:
        tau, halt_tau = 0.5, 0.5

    if reference_quality is None:
        reference_quality = evaluate(model, val_sk, data.world, tau=tau, halt_tau=halt_tau)["quality"]
    guard = SafetyGuard(reference_quality, tol=0.05)

    last_good = {"model": copy.deepcopy(model.state_dict()), "baseline": copy.deepcopy(baseline.state_dict())}
    best_mean_val_reward = None

    def save_best(state_model: dict, state_baseline: dict) -> None:
        torch.save(state_model, out_dir / "model.pt")
        torch.save(state_baseline, out_dir / "baseline.pt")

    # Ensure a checkpoint exists from the start (supervised start), overwritten if a later
    # check passes and improves on it.
    save_best(last_good["model"], last_good["baseline"])

    history: list[dict] = []

    def run_check(update: int) -> dict:
        nonlocal best_mean_val_reward
        model.eval()
        val_by_lam: dict[int, dict] = {}
        for lam_idx in range(len(LAMS)):
            val_by_lam[lam_idx] = evaluate_rl(model, val_sk, data.world, lam_idx, cost_ref)
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
                save_best(last_good["model"], last_good["baseline"])
        else:
            model.load_state_dict(last_good["model"])
            baseline.load_state_dict(last_good["baseline"])
            for g in opt.param_groups:
                g["lr"] = g["lr"] / 2
            row["lr"] = opt.param_groups[0]["lr"]
            row["alarms"].append("rollback")
        return row

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
                   "complexity_mse": step["complexity_mse"], "entropy": step["entropy"], "beta": beta,
                   "lr": opt.param_groups[0]["lr"], "grad_norm": step["grad_norm"]}
            history.append(row)
            log(row)

        if update % eval_every == 0 or update == updates:
            row = run_check(update)
            history.append(row)
            log(row)

    summary = {"config": config, "seed": seed, "stage": "rl_sim", "init_dir": str(init_dir),
               "data_dir": str(data.data_dir or DATA_DIR), "lams": list(LAMS),
               "cost_ref": cost_ref, "reference_quality": reference_quality, "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
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
