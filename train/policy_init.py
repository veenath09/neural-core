"""Calibrated starting point for simulator RL (plan D13, deviation: seed-0 pilot v3).

The D13 policy is policy_logits = success_logits - lam * softplus(raw0) * ln(card_cost / cost_ref)
and halt_policy = halt_logit + raw1, with [raw0, raw1] = lam_adj([z_H, ln lam, c]). The default
init (raw0 = softplus^-1(0.1), raw1 = 0) starts RL at the expensive corner: nearly always the
strongest model, escalation only when p(halt) > 0.5. On the seed-0 pilot a search over just these
two numbers, with no training, beat both the supervised tau rule and 1,000 RL updates at every lambda.

calibrate_policy_init grid-searches, per lambda, the price weight w = lam * softplus(raw0) and the
halt offset h by serve-mode reward on the given skeletons (training episodes, so the val guard and
model selection stay untouched), fits raw0 and raw1 as linear functions of ln lam, and writes them
into lam_adj (weights on z_H and c stay zero). If the fitted start is worse than the default on
those skeletons, the default is kept.

Pilot v4: it then sets the policy sharpness k (controller/heads.py) to the smallest value in
grid_k whose mean sample-mode reward is within SHARPNESS_TOL of the serve-mode reward, so RL
learns from behaviour close to what it is judged on while keeping as much exploration as
possible. k does not change serve decisions, so the fitted start above is unaffected.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from measurement.rl_reward import LAMS, episode_reward, scorer_for
from train.rl import evaluate_rl
from train.rollout import rollout
from train.train import make_episodes

GRID_W = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)
GRID_H = (0.0, 1.0, 2.0, 3.0)
GRID_K = (1.0, 2.0, 3.0, 5.0, 10.0)
SHARPNESS_TOL = 0.02


def _softplus_inv(v: float) -> float:
    return math.log(math.expm1(v))


def set_policy_init(model, price: tuple[float, float], halt: tuple[float, float]) -> None:
    """raw0 = price[0] + price[1] * ln lam and raw1 = halt[0] + halt[1] * ln lam; other weights zero."""
    lin = model.heads.lam_adj
    log_lam_col = lin.in_features - 2  # input is [z_H (d), ln lam, c]
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.tensor([price[0], halt[0]]))
        lin.weight[0, log_lam_col] = price[1]
        lin.weight[1, log_lam_col] = halt[1]


def sample_reward(model, skeletons, world, cost_ref, lam_idx: int, seed: int = 0, batch_size: int = 512) -> float:
    """Mean episode reward of the sample-mode policy at one lambda (no gradient)."""
    model.eval()
    scorer = scorer_for(cost_ref)
    rng, rewards = np.random.default_rng(seed), []
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            eps = make_episodes(skeletons[start:start + batch_size], world)
            out = rollout(model, eps, world, policy="sample", rng=rng, lam_idx=np.full(len(eps), lam_idx),
                          cost_ref=cost_ref)
            rewards += [episode_reward(scorer, r.quality, r.cost_usd, e.n_steps, LAMS[lam_idx])
                        for r, e in zip(out.results, eps)]
    return float(np.mean(rewards))


def calibrate_sharpness(model, skeletons, world, cost_ref, serve: dict[int, float], lams=LAMS, grid_k=GRID_K,
                        tol: float = SHARPNESS_TOL) -> tuple[float, dict[str, float]]:
    """Smallest k in grid_k with mean(serve - sample reward) <= tol (else the largest k); sets it."""
    gaps = {}
    for k in sorted(grid_k):
        with torch.no_grad():
            model.heads.log_sharpness.fill_(math.log(k))
        gaps[str(k)] = float(np.mean([serve[i] - sample_reward(model, skeletons, world, cost_ref, i)
                                      for i in range(len(lams))]))
        if gaps[str(k)] <= tol:
            return k, gaps
    return float(max(grid_k)), gaps


def _rewards(model, skeletons, world, cost_ref, lams) -> dict[int, dict]:
    return {i: evaluate_rl(model, skeletons, world, i, cost_ref) for i in range(len(lams))}


def calibrate_policy_init(model, skeletons, world, cost_ref: float, lams=LAMS, grid_w=GRID_W,
                          grid_h=GRID_H, grid_k=GRID_K, floor_tol: float | None = 0.05, refits: int = 3) -> dict:
    """See the module docstring. With floor_tol, a fit that leaves some lambda more than floor_tol below the
    default start on these skeletons is refitted with 4x weight on the failing lambdas, up to `refits` times
    (a straight line in ln lambda can miss one end of the grid: the seed-0 pilot lost 0.1 at lambda 2)."""
    lin = model.heads.lam_adj
    default_state = {k: v.clone() for k, v in lin.state_dict().items()}
    default = _rewards(model, skeletons, world, cost_ref, lams)
    default_mean = float(np.mean([r["reward"] for r in default.values()]))

    best = {}
    for i, lam in enumerate(lams):
        for w in grid_w:
            for h in grid_h:
                set_policy_init(model, price=(_softplus_inv(w / lam), 0.0), halt=(h, 0.0))
                r = evaluate_rl(model, skeletons, world, i, cost_ref)
                if i not in best or r["reward"] > best[i]["reward"]:
                    best[i] = {"w": w, "h": h, "reward": r["reward"], "quality": r["quality"], "cost_usd": r["cost_usd"]}

    x = np.log(np.asarray(lams, dtype=float))
    y_price = np.array([_softplus_inv(best[i]["w"] / lam) for i, lam in enumerate(lams)])
    y_halt = np.array([best[i]["h"] for i in range(len(lams))])
    weights = np.ones(len(lams))
    for it in range(refits + 1):
        b_price, a_price = np.polyfit(x, y_price, 1, w=weights)
        b_halt, a_halt = np.polyfit(x, y_halt, 1, w=weights)
        set_policy_init(model, price=(float(a_price), float(b_price)), halt=(float(a_halt), float(b_halt)))
        fitted = _rewards(model, skeletons, world, cost_ref, lams)
        below = [i for i in range(len(lams)) if fitted[i]["reward"] < default[i]["reward"] - (floor_tol or 0)]
        if floor_tol is None or not below:
            break
        weights[below] *= 4
    fitted_mean = float(np.mean([r["reward"] for r in fitted.values()]))

    kept_default = fitted_mean < default_mean
    if kept_default:
        lin.load_state_dict(default_state)
    final = default if kept_default else fitted
    k, gaps = calibrate_sharpness(model, skeletons, world, cost_ref, {i: final[i]["reward"] for i in final}, lams,
                                  grid_k)
    with torch.no_grad():
        b0, w0 = lin.bias[0].item(), lin.weight[0, lin.in_features - 2].item()
    raw0 = lambda lam: b0 + w0 * math.log(lam)  # noqa: E731
    return {
        "kept_default": kept_default,
        "default_mean_reward": default_mean,
        "mean_reward": default_mean if kept_default else fitted_mean,
        "grid_best": best,
        "fit": {"price": [float(a_price), float(b_price)], "halt": [float(a_halt), float(b_halt)]},
        "fit_weights": weights.tolist(), "lams_below_default": [] if kept_default else below,
        "per_lam": {i: {"reward": final[i]["reward"], "quality": final[i]["quality"], "cost_usd": final[i]["cost_usd"]}
                    for i in range(len(lams))},
        "fitted_price_weight": [lam * math.log1p(math.exp(raw0(lam))) for lam in lams],
        "sharpness": k,
        "sharpness_gap": gaps,
        "sharpness_at_grid_edge": k == max(grid_k),
    }
