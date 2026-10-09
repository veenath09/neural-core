"""Safety guard for the simulator RL loop (plan C2-07).

Every eval_every updates, train_rl checks greedy ("serve") val quality at each
lambda against that lambda's own reference: the supervised start's own
serve-mode val quality at that lambda, measured once at update 0. If any
lambda's quality has fallen more than tol below its reference, the guard
fails and the caller restores the last good weights/baseline and halves the
learning rate.

reference_quality may be a single float (the same reference for every lambda,
e.g. the reference_quality override argument of train_rl) or a dict mapping
lam_idx -> reference quality (the per-lambda measurement from update 0).
"""

from __future__ import annotations


class SafetyGuard:
    def __init__(self, reference_quality: float | dict[int, float], tol: float = 0.05):
        self.reference_quality = reference_quality
        self.tol = tol

    def _reference(self, lam_idx: int) -> float:
        if isinstance(self.reference_quality, dict):
            return self.reference_quality[lam_idx]
        return self.reference_quality

    def ok(self, quality_by_lam: dict[int, float]) -> bool:
        return all(q >= self._reference(lam_idx) - self.tol for lam_idx, q in quality_by_lam.items())
