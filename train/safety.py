from __future__ import annotations

import copy


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