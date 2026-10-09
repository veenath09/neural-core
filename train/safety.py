"""Safety guard for the simulator RL loop (plan C2-07).

Every eval_every updates, train_rl checks greedy val quality at each lambda
in serve mode against the supervised reference. If any lambda's quality has
fallen more than tol below the reference, the guard fails and the caller
restores the last good weights/baseline and halves the learning rate.
"""

from __future__ import annotations


class SafetyGuard:
    def __init__(self, reference_quality: float, tol: float = 0.05):
        self.reference_quality = reference_quality
        self.tol = tol

    def ok(self, quality_by_lam: dict[int, float]) -> bool:
        return all(q >= self.reference_quality - self.tol for q in quality_by_lam.values())
