
"""Pillar 3 C3-05: RouterBench-backed simulator interface.

Wraps the existing data.env.Episode without duplicating its
retry, propagation, escalation, budget, or verifier logic.

TaskRep and TraceItem expose observable information.
True correctness is reserved for privileged evaluation
and oracle methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from controller import contracts
from controller.contracts import Strategy, TaskRep, TraceItem
from data.env import Episode, EnvState, VERIFIER_FLIP_P
from data.episodes import Skeleton
from data.routerbench import OutcomeTable


SimulationStatus = Literal[
    "running",
    "awaiting_escalation",
    "completed",
    "budget_exhausted",
]


@dataclass(frozen=True)
class SimulationTransition:
    """Observable result of applying one controller strategy."""

    task: TaskRep | None
    trace_item: TraceItem | None
    done: bool
    status: SimulationStatus


class Simulator:
    """Provide the Controller-facing interface to an existing Episode.

    Strategy.round is zero-based:
        0 = first model attempt
        1 = optional escalation attempt

    TraceItem.round is one-based, matching data.env.Episode.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        table: OutcomeTable,
        model_ids: Sequence[str],
        *,
        lam: float | None = None,
        flip_p: float = VERIFIER_FLIP_P,
        seed: int = 0,
        trace_window: int | None = None,
    ) -> None:
        ids = list(model_ids)

        if len(ids) != table.n_models:
            raise ValueError(
                "Model ID count must match OutcomeTable model count"
            )

        if len(set(ids)) != len(ids):
            raise ValueError("Model IDs must be unique")

        if not all(
            isinstance(model_id, str) and model_id
            for model_id in ids
        ):
            raise ValueError("Model IDs must be nonempty strings")

        if not math.isfinite(float(skeleton.budget_usd)):
            raise ValueError("Episode budget must be finite")

        if skeleton.budget_usd < 0:
            raise ValueError("Episode budget cannot be negative")

        if lam is not None:
            if isinstance(lam, bool) or not math.isfinite(lam):
                raise ValueError("Lambda must be finite")

        if not math.isfinite(flip_p) or not 0 <= flip_p <= 1:
            raise ValueError("Verifier flip probability must be 0..1")

        if trace_window is None:
            trace_window = getattr(contracts, "K_TRACE", 8)

        if (
            isinstance(trace_window, bool)
            or not isinstance(trace_window, int)
            or trace_window < 1
        ):
            raise ValueError("Trace window must be positive")

        self.skeleton = skeleton
        self.model_ids = ids
        self.model_index = {
            model_id: index
            for index, model_id in enumerate(ids)
        }

        self.lam = lam
        self.trace_window = trace_window

        self._episode = Episode(
            skeleton=skeleton,
            table=table,
            model_ids=ids,
            flip_p=flip_p,
            seed=seed,
        )

        self._state = self._episode.initial_state()

    def reset(self) -> TaskRep:
        """Restart the episode from its initial state."""

        self._state = self._episode.initial_state()

        task = self.current_task()

        if task is None:
            raise ValueError("Episode has no active task")

        return task

    def _step_id(self, state: EnvState) -> str:
        """Stable identifier for one logical execution step."""

        return (
            f"{self.skeleton.episode_id}:"
            f"step:{state.step_index}"
        )

    def current_task(self) -> TaskRep | None:
        """Return the observation visible to the Controller."""

        state = self._state

        if state.done:
            return None

        row_index = self._episode.row(state)
        prompt_id = str(
            self._episode.table.prompt_ids[row_index]
        )

        # These are lookup keys, not computed embeddings.
        return TaskRep(
            episode_id=self.skeleton.episode_id,
            step_id=self._step_id(state),
            round=state.rounds,
            task_emb_key=self.skeleton.episode_id,
            step_type=int(
                self._episode.table.step_type[row_index]
            ),
            step_emb_key=prompt_id,
            trace=list(state.trace[-self.trace_window:]),
            budget_total_usd=float(
                self.skeleton.budget_usd
            ),
            budget_left_usd=float(state.budget_left),
            spent_usd=float(state.spent),
            step_index=state.step_index,
            expected_steps=self._episode.n_steps,
            lam=self.lam,
        )

    def counterfactual(self) -> dict[str, bool]:
        """Oracle-only success outcomes given the realised history.

        These are not exposed through TaskRep or a transition.
        They describe correctness, not affordability.
        """

        if self._state.done:
            raise ValueError(
                "No counterfactuals after episode completion"
            )

        outcomes = self._episode.succeeds(self._state)

        return {
            model_id: bool(outcomes[index])
            for index, model_id in enumerate(self.model_ids)
        }

    def affordable_models(self) -> list[str]:
        """List models affordable for the next execution round."""

        state = self._state

        if state.done:
            return []

        if state.rounds == 0:
            indices = range(len(self.model_ids))
        else:
            indices = self._episode.escalation_options(state)

        return [
            self.model_ids[index]
            for index in indices
            if self._episode.affordable(state, index)
        ]

    def apply(self, strategy: Strategy) -> SimulationTransition:
        """Execute one Strategy through the existing environment."""

        state = self._state

        if state.done:
            raise ValueError("Episode is already complete")

        task = self.current_task()

        if task is None:
            raise ValueError("No active task")

        if strategy.step_id != task.step_id:
            raise ValueError("Strategy step ID mismatch")

        if strategy.round != task.round:
            raise ValueError("Strategy round mismatch")

        if strategy.model_id not in self.model_index:
            raise ValueError("Unknown model ID")

        if strategy.halt not in ("stop", "escalate"):
            raise ValueError("Unsupported halt decision")

        if state.rounds == 1 and strategy.halt != "stop":
            raise ValueError(
                "Second round must stop after escalation"
            )

        model_index = self.model_index[
            strategy.model_id
        ]

        before_trace_length = len(state.trace)

        # The existing Episode enforces budget limits,
        # tier ordering, and the maximum round count.
        next_state = self._episode.play(
            state,
            model_index,
        )

        trace_item = (
            next_state.trace[-1]
            if len(next_state.trace) > before_trace_length
            else None
        )

        if next_state.done:
            # An unaffordable round is not executed or charged.
            status: SimulationStatus = "budget_exhausted"

        elif strategy.halt == "escalate":
            # The next Strategy must choose a strictly stronger,
            # affordable model on the same logical step.
            if not self._episode.escalation_options(next_state):
                raise ValueError(
                    "No affordable stronger model for escalation"
                )

            status = "awaiting_escalation"

        else:
            # Completing the step applies retry or propagation
            # rules through the existing Episode.
            next_state = self._episode.end_step(next_state)

            status = (
                "completed"
                if next_state.done
                else "running"
            )

        self._state = next_state

        return SimulationTransition(
            task=self.current_task(),
            trace_item=trace_item,
            done=next_state.done,
            status=status,
        )

    def final_metrics(self) -> dict[str, float | int | str]:
        """Provide ground-truth metrics for evaluation only.

        Call after completion, never as Controller observation.
        Composite reward is deferred to C3-07.
        """

        state = self._state

        if not state.done:
            raise ValueError(
                "Final metrics require a completed episode"
            )

        return {
            "episode_id": self.skeleton.episode_id,
            "quality": float(self._episode.quality(state)),
            "n_correct": int(state.n_correct),
            "expected_steps": int(self._episode.n_steps),
            "cost_usd": float(state.spent),
            "budget_left_usd": float(state.budget_left),
        }
