"""Safety guard for the simulator RL loop (plan C2-07).

Every eval_every updates, train_rl checks greedy ("serve") val reward at each
lambda against that lambda's own reference: the supervised start's own
serve-mode val reward at that lambda, measured once at update 0. If any
lambda's reward has fallen more than tol below its reference, the guard
fails and the caller restores the last good weights/baseline and halves the
learning rate.

Deviation (seed-0 pilot v2): the guard compares val reward, not quality,
because high-lambda policies trade quality for cost by design (the
lambda-oracle's own quality falls as lambda grows), so a quality-based guard
rejected good high-lambda policies. The guard must measure the thing being
optimised.

reference_reward may be a single float (the same reference for every lambda,
e.g. the reference_reward override argument of train_rl) or a dict mapping
lam_idx -> reference reward (the per-lambda measurement from update 0).
"""

from __future__ import annotations


class SafetyGuard:
    def __init__(self, reference_reward: float | dict[int, float], tol: float = 0.05):
        self.reference_reward = reference_reward
        self.tol = tol

    def _reference(self, lam_idx: int) -> float:
        if isinstance(self.reference_reward, dict):
            return self.reference_reward[lam_idx]
        return self.reference_reward

    def ok(self, reward_by_lam: dict[int, float]) -> bool:
        return all(r >= self._reference(lam_idx) - self.tol for lam_idx, r in reward_by_lam.items())
