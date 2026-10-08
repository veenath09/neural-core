"""Monte Carlo policy-gradient update with a bucketed baseline and complexity
auxiliary (plan C2-05, C2-06).

Loss = -mean(A.detach() * logp) over choice slots
       - beta * mean(entropy) over choice slots
       + COMPLEXITY_AUX * mean((complexity - band_target)**2) over valid slots

A = G - baseline[band, lam_idx], G = returns_to_go(reward), normalised over
choice slots with population std ((A - mean) / (std + 1e-8)).

load_cost_ref is Task 1's (measurement/rl_reward.py); re-exported here so
callers that only import from train.rl still get it.
"""

from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

from measurement.rl_reward import load_cost_ref  # re-exported; do not duplicate
from train.baseline import BucketedBaseline
from train.rollout import RLTrace

COMPLEXITY_AUX = 0.2

__all__ = [
    "COMPLEXITY_AUX",
    "load_cost_ref",
    "returns_to_go",
    "entropy_beta",
    "rl_loss",
    "rl_update",
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
        baseline.update(band[choice].cpu(), lam_idx_bt[choice].cpu(), returns[choice].detach().cpu())

    return {
        "loss": float(loss.detach()),
        "pg_loss": parts["pg_loss"],
        "complexity_mse": parts["complexity_mse"],
        "episode_return": parts["episode_return"],
        "entropy": parts["entropy"],
        "grad_norm": float(grad_norm),
    }
