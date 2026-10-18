
"""Pillar 3 C3-06: Per-step and exact episode oracles.

Reuses the team's existing quality-first oracle.

Adds a separately optimized RL oracle for:
    episode_quality - lam * simulated_cost

Oracle information is privileged and must never be
included in observations given to the Controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from data.env import Action, Episode, EnvState
from data.oracle import halt_label, plan, step_labels


@dataclass(frozen=True)
class PerStepOracleResult:
    model_id: str
    model_tier: int
    succeeds: tuple[bool, ...]
    difficulty: int
    affordable: bool


@dataclass(frozen=True)
class EpisodeOracleResult:
    episode_id: str
    correct_steps: int
    expected_steps: int
    quality: float
    cost_usd: float
    budget_left_usd: float
    actions: tuple[Action, ...]
    status: str


@dataclass(frozen=True)
class WeightedEpisodeOracleResult:
    episode_id: str
    lam: float
    correct_steps: int
    expected_steps: int
    quality: float
    cost_usd: float
    budget_left_usd: float
    objective: float
    actions: tuple[Action, ...]
    status: str


def per_step_oracle(
    episode: Episode,
    state: EnvState | None = None,
) -> PerStepOracleResult:
    """Cheapest successful model, or cheapest if all fail."""

    if state is None:
        state = episode.initial_state()

    if state.done:
        raise ValueError(
            "Per-step oracle requires an active episode"
        )

    if state.rounds != 0:
        raise ValueError(
            "Per-step oracle requires a new step"
        )

    labels = step_labels(episode, state)
    tier = int(labels.oracle_model)

    if not 0 <= tier < len(episode.model_ids):
        raise ValueError("Invalid oracle model tier")

    return PerStepOracleResult(
        model_id=episode.model_ids[tier],
        model_tier=tier,
        succeeds=tuple(
            bool(value) for value in labels.succeeds
        ),
        difficulty=int(labels.band),
        affordable=bool(
            episode.affordable(state, tier)
        ),
    )


def escalation_oracle(
    episode: Episode,
    state: EnvState,
) -> bool:
    """Whether a successful affordable escalation exists."""

    if state.done:
        raise ValueError(
            "Escalation oracle requires an active episode"
        )

    if state.rounds != 1:
        raise ValueError(
            "Escalation oracle requires exactly one played round"
        )

    return bool(halt_label(episode, state))


def _terminal_status(actions: tuple[Action, ...]) -> str:
    """A terminal play indicates an unaffordable attempted round."""

    return (
        "budget_exhausted"
        if actions and actions[-1][0] == "play"
        else "completed"
    )


def exact_episode_oracle(
    episode: Episode,
    state: EnvState | None = None,
) -> EpisodeOracleResult:
    """Maximize episode quality, then minimize total cost."""

    if state is None:
        state = episode.initial_state()

    if state.done:
        raise ValueError(
            "Exact oracle requires an active episode"
        )

    oracle_plan = plan(episode, state)

    final_state = state

    for action in oracle_plan.actions:
        final_state = episode.apply(final_state, action)

    if not final_state.done:
        raise RuntimeError(
            "Oracle plan did not complete the episode"
        )

    if final_state.n_correct != oracle_plan.n_correct:
        raise RuntimeError(
            "Oracle correctness does not match replay"
        )

    if not math.isclose(
        final_state.spent,
        oracle_plan.spent,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Oracle cost does not match replay"
        )

    return EpisodeOracleResult(
        episode_id=episode.skeleton.episode_id,
        correct_steps=int(final_state.n_correct),
        expected_steps=int(episode.n_steps),
        quality=float(episode.quality(final_state)),
        cost_usd=float(final_state.spent),
        budget_left_usd=float(final_state.budget_left),
        actions=tuple(oracle_plan.actions),
        status=_terminal_status(oracle_plan.actions),
    )


def _continuation_key(
    episode: Episode,
    state: EnvState,
) -> tuple:
    """Identify equivalent futures at an episode step boundary.

    Among equivalent futures, the cheaper path dominates
    because it leaves at least as much available budget.
    """

    last_failure = (
        state.last_fail_tier
        if episode.skeleton.ep_type == "propagation"
        else None
    )

    return (
        state.n_correct,
        state.slot,
        state.step_index,
        state.retried,
        state.done,
        last_failure,
    )


def _weighted_finish_options(
    episode: Episode,
    state: EnvState,
) -> list[tuple[EnvState, tuple[Action, ...]]]:
    """Enumerate ways to finish one step, pruning dominated costs.

    Uses the environment's own legal actions and transitions.
    """

    cheapest: dict[
        tuple,
        tuple[EnvState, tuple[Action, ...]],
    ] = {}

    def walk(
        current: EnvState,
        actions: tuple[Action, ...],
    ) -> None:
        for action in episode.legal_actions(current):
            following = episode.apply(current, action)
            path = actions + (action,)

            if following.done or action[0] == "stop":
                key = _continuation_key(
                    episode, following
                )

                previous = cheapest.get(key)

                if (
                    previous is None
                    or following.spent < previous[0].spent
                ):
                    cheapest[key] = (following, path)

            else:
                walk(following, path)

    walk(state, ())

    return list(cheapest.values())


def weighted_episode_oracle(
    episode: Episode,
    lam: float,
    state: EnvState | None = None,
) -> WeightedEpisodeOracleResult:
    """Find the best legal plan for quality - lam * raw cost.

    lam must be finite and nonnegative.

    Ties are resolved by lower cost, then higher quality.

    This is a separate objective from the normalized,
    latency-aware composite scorer planned for C3-07.
    """

    if (
        isinstance(lam, bool)
        or not isinstance(lam, Real)
        or not math.isfinite(lam)
        or lam < 0
    ):
        raise ValueError(
            "Lambda must be finite and nonnegative"
        )

    weight = float(lam)

    if state is None:
        state = episode.initial_state()

    if state.done:
        raise ValueError(
            "Weighted oracle requires an active episode"
        )

    memo: dict[
        tuple,
        tuple[int, float, tuple[Action, ...]],
    ] = {}

    def solve(
        current: EnvState,
    ) -> tuple[int, float, tuple[Action, ...]]:
        if current.done:
            return (
                int(current.n_correct),
                float(current.spent),
                (),
            )

        # This compact key avoids storing entire trace histories
        # in the search memo.
        key = (
            _continuation_key(episode, current),
            current.rounds,
            current.cur_model,
            current.cur_correct,
            current.spent,
            current.budget_left,
        )

        if key in memo:
            return memo[key]

        best_result = None
        best_rank = None

        for following, actions in _weighted_finish_options(
            episode, current
        ):
            correct, cost, remaining = solve(following)

            quality = correct / episode.n_steps

            objective = quality - weight * cost

            rank = (
                objective,
                -cost,
                correct,
            )

            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_result = (
                    correct,
                    cost,
                    actions + remaining,
                )

        if best_result is None:
            raise RuntimeError(
                "Oracle has no legal continuation"
            )

        memo[key] = best_result
        return best_result

    correct, spent, actions = solve(state)

    # Verify the winning plan using the actual environment.
    final_state = state

    for action in actions:
        final_state = episode.apply(final_state, action)

    if not final_state.done:
        raise RuntimeError(
            "Weighted oracle plan is not terminal"
        )

    if final_state.n_correct != correct:
        raise RuntimeError(
            "Weighted oracle correctness mismatch"
        )

    if not math.isclose(
        final_state.spent,
        spent,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Weighted oracle cost mismatch"
        )

    quality = float(episode.quality(final_state))

    return WeightedEpisodeOracleResult(
        episode_id=episode.skeleton.episode_id,
        lam=weight,
        correct_steps=int(final_state.n_correct),
        expected_steps=int(episode.n_steps),
        quality=quality,
        cost_usd=float(final_state.spent),
        budget_left_usd=float(final_state.budget_left),
        objective=quality - weight * final_state.spent,
        actions=actions,
        status=_terminal_status(actions),
    )
