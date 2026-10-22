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

ValidationGate, RollbackManager, CanaryEvaluator and validate_or_rollback serve the
online runner (train/online_runner.py).
"""

from __future__ import annotations

import copy


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



class ValidationGate:
    def __init__(
        self,
        max_degradation: float = 0.05,
    ):
        if max_degradation < 0.0:
            raise ValueError(
                "max_degradation must be non-negative"
            )

        self.max_degradation = max_degradation

    def accept(
        self,
        baseline_score: float,
        candidate_score: float,
    ) -> bool:
        degradation = (
            baseline_score - candidate_score
        )

        return degradation <= self.max_degradation


class RollbackManager:
    def __init__(self):
        self._state = None
        self._optimizer_state = None

    def capture(
        self,
        module,
        optimizer=None,
    ) -> None:
        self._state = {
            name: tensor.detach().clone()
            for name, tensor in module.state_dict().items()
        }

        self._optimizer_state = (
            copy.deepcopy(optimizer.state_dict())
            if optimizer is not None
            else None
        )

    def restore(
        self,
        module,
        optimizer=None,
    ) -> None:
        if self._state is None:
            raise ValueError(
                "no rollback state has been captured"
            )

        module.load_state_dict(
            self._state
        )

        if optimizer is not None:
            if self._optimizer_state is None:
                raise ValueError(
                    "no optimizer rollback state has been captured"
                )

            optimizer.load_state_dict(
                self._optimizer_state
            )


class CanaryEvaluator:
    def __init__(
        self,
        gate: ValidationGate,
    ):
        self.gate = gate

    def evaluate(
        self,
        baseline_scores: list[float],
        candidate_scores: list[float],
    ) -> dict:
        if not baseline_scores or not candidate_scores:
            raise ValueError(
                "canary score lists must not be empty"
            )

        if len(baseline_scores) != len(candidate_scores):
            raise ValueError(
                "baseline_scores and candidate_scores "
                "must have the same length"
            )

        baseline_mean = (
            sum(baseline_scores)
            / len(baseline_scores)
        )

        candidate_mean = (
            sum(candidate_scores)
            / len(candidate_scores)
        )

        accepted = self.gate.accept(
            baseline_score=baseline_mean,
            candidate_score=candidate_mean,
        )

        return {
            "accepted": accepted,
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "n_canary": len(candidate_scores),
        }


def validate_or_rollback(
    module,
    gate: ValidationGate,
    rollback: RollbackManager,
    baseline_score: float,
    candidate_score: float,
    optimizer=None,
) -> bool:
    accepted = gate.accept(
        baseline_score=baseline_score,
        candidate_score=candidate_score,
    )

    if not accepted:
        rollback.restore(
            module,
            optimizer=optimizer,
        )

    return accepted
