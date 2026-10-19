from __future__ import annotations


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

    def capture(self, module) -> None:
        self._state = {
            name: tensor.detach().clone()
            for name, tensor in module.state_dict().items()
        }

    def restore(self, module) -> None:
        if self._state is None:
            raise ValueError(
                "no rollback state has been captured"
            )

        module.load_state_dict(
            self._state
        )


def validate_or_rollback(
        module,
        gate: ValidationGate,
        rollback: RollbackManager,
        baseline_score: float,
        candidate_score: float,
    ) -> bool:
        accepted = gate.accept(
            baseline_score=baseline_score,
            candidate_score=candidate_score,
        )

        if not accepted:
            rollback.restore(module)

        return accepted