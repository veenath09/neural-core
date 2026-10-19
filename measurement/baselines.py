
"""Pillar 3 C3-08: Simple baseline ControllerAPI policies.

Baselines:
    1. AlwaysCheapest
    2. AlwaysStrongest
    3. RandomBaseline
    4. EpisodeOracleBaseline

All policies expose:
    decide(TaskRep) -> Strategy

The first three are ordinary non-oracle policies.

The episode oracle is a privileged evaluation ceiling:
it must be explicitly bound to a simulation episode
before making decisions. Its ground-truth information
must not be passed to the learned Controller.
"""

from __future__ import annotations

import math
import random
from typing import Protocol, Sequence, runtime_checkable

from controller.contracts import ModelCard, Strategy, TaskRep
from data.env import Episode
from measurement.oracle import exact_episode_oracle


@runtime_checkable
class ControllerAPI(Protocol):
    """Minimal Controller interface for C3-09 evaluation."""

    def decide(self, task: TaskRep) -> Strategy:
        """Return a strategy for the current task."""
        ...


def _validate_pool(
    pool: Sequence[ModelCard],
) -> tuple[ModelCard, ...]:
    """Validate the available model cards."""

    cards = tuple(pool)

    if not cards:
        raise ValueError("Model pool cannot be empty")

    seen = set()

    for card in cards:
        model_id = card.model_id

        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                "Model IDs must be nonempty strings"
            )

        if model_id in seen:
            raise ValueError(
                "Model IDs must be unique"
            )

        seen.add(model_id)

        cost = card.mean_cost_usd

        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError(
                "Model mean costs must be finite and nonnegative"
            )

    return cards


def _single_round_strategy(
    task: TaskRep,
    model_id: str,
) -> Strategy:
    """Create a strategy that executes once and stops.

    Simple baselines never request escalation, so they
    normally receive only round-zero observations.
    """

    if task.round != 0:
        raise ValueError(
            "Single-round baseline expected round 0"
        )

    return Strategy(
        step_id=task.step_id,
        round=task.round,
        model_id=model_id,
        halt="stop",
    )


class AlwaysCheapest:
    """Always select the lowest mean-cost model."""

    def __init__(
        self,
        pool: Sequence[ModelCard],
    ) -> None:
        self.pool = _validate_pool(pool)

        self.model_id = min(
            self.pool,
            key=lambda card: card.mean_cost_usd,
        ).model_id

    def decide(self, task: TaskRep) -> Strategy:
        return _single_round_strategy(
            task,
            self.model_id,
        )


class AlwaysStrongest:
    """Always select the highest-cost pool model.

    The four-model RouterBench pool is ordered by cost
    tier. This baseline uses the most expensive model
    as the strongest-model reference.
    """

    def __init__(
        self,
        pool: Sequence[ModelCard],
    ) -> None:
        self.pool = _validate_pool(pool)

        self.model_id = max(
            self.pool,
            key=lambda card: card.mean_cost_usd,
        ).model_id

    def decide(self, task: TaskRep) -> Strategy:
        return _single_round_strategy(
            task,
            self.model_id,
        )


class RandomBaseline:
    """Choose a model reproducibly using a fixed seed.

    The random choice is derived from the seed and task
    identifiers. Repeated evaluation of the same task
    therefore produces the same choice.
    """

    def __init__(
        self,
        pool: Sequence[ModelCard],
        seed: int = 1234,
    ) -> None:
        self.pool = _validate_pool(pool)

        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError(
                "Random seed must be an integer"
            )

        self.seed = seed

        self.model_ids = tuple(
            card.model_id
            for card in self.pool
        )

    def decide(self, task: TaskRep) -> Strategy:
        if task.round != 0:
            raise ValueError(
                "Random baseline expected round 0"
            )

        key = (
            f"{self.seed}:"
            f"{task.episode_id}:"
            f"{task.step_id}:"
            f"{task.round}"
        )

        rng = random.Random(key)

        model_id = rng.choice(
            self.model_ids
        )

        return _single_round_strategy(
            task,
            model_id,
        )


class EpisodeOracleBaseline:
    """Follow the exact episode Oracle's privileged plan.

    This policy must be bound to a data.env.Episode
    before evaluation.

    It converts the oracle's action sequence:

        play(model), stop
        play(model), play(stronger), stop

    into Controller Strategy objects:

        Strategy(model, halt="stop")
        Strategy(model, halt="escalate")
        Strategy(stronger, halt="stop")

    The Episode Oracle is an evaluation ceiling only.
    """

    def __init__(
        self,
        model_ids: Sequence[str],
    ) -> None:
        ids = tuple(model_ids)

        if not ids:
            raise ValueError(
                "Model pool cannot be empty"
            )

        if (
            any(
                not isinstance(model_id, str)
                or not model_id
                for model_id in ids
            )
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "Model IDs must be unique nonempty strings"
            )

        self.model_ids = ids

        self._episode_id: str | None = None

        self._decisions: dict[
            tuple[int, int],
            tuple[str, str, str],
        ] = {}

    def bind_episode(
        self,
        episode: Episode,
    ) -> None:
        """Prepare an exact plan for one evaluation episode."""

        if tuple(episode.model_ids) != self.model_ids:
            raise ValueError(
                "Episode model order does not match baseline"
            )

        oracle = exact_episode_oracle(episode)

        actions = oracle.actions

        state = episode.initial_state()

        planned: dict[
            tuple[int, int],
            tuple[str, str, str],
        ] = {}

        for index, action in enumerate(actions):
            action_name, model_index = action

            if state.done:
                raise RuntimeError(
                    "Oracle plan contains actions after completion"
                )

            if action_name == "play":
                if model_index is None:
                    raise RuntimeError(
                        "Oracle play action has no model"
                    )

                if not 0 <= model_index < len(self.model_ids):
                    raise RuntimeError(
                        "Oracle selected an invalid model"
                    )

                key = (
                    state.step_index,
                    state.rounds,
                )

                if key in planned:
                    raise RuntimeError(
                        "Oracle plan contains duplicate decisions"
                    )

                # Two consecutive play actions represent
                # the initial round followed by escalation.
                next_is_play = (
                    index + 1 < len(actions)
                    and actions[index + 1][0] == "play"
                )

                halt = (
                    "escalate"
                    if next_is_play
                    else "stop"
                )

                row_index = episode.row(state)

                prompt_id = str(
                    episode.table.prompt_ids[row_index]
                )

                planned[key] = (
                    self.model_ids[model_index],
                    halt,
                    prompt_id,
                )

            state = episode.apply(
                state,
                action,
            )

        if not state.done:
            raise RuntimeError(
                "Oracle plan did not finish the episode"
            )

        self._episode_id = episode.skeleton.episode_id
        self._decisions = planned

    def decide(
        self,
        task: TaskRep,
    ) -> Strategy:
        """Return the next action from the prepared plan."""

        if self._episode_id is None:
            raise ValueError(
                "Oracle baseline must be bound to an episode"
            )

        if task.episode_id != self._episode_id:
            raise ValueError(
                "Task belongs to a different episode"
            )

        key = (
            task.step_index,
            task.round,
        )

        if key not in self._decisions:
            raise ValueError(
                "No Oracle decision for this step and round"
            )

        model_id, halt, prompt_id = (
            self._decisions[key]
        )

        if task.step_emb_key != prompt_id:
            raise ValueError(
                "Oracle task prompt does not match planned prompt"
            )

        expected_step_id = (
            f"{self._episode_id}:step:{task.step_index}"
        )

        if task.step_id != expected_step_id:
            raise ValueError(
                "Oracle task step ID mismatch"
            )

        return Strategy(
            step_id=task.step_id,
            round=task.round,
            model_id=model_id,
            halt=halt,
        )
