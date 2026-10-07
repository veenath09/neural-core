"""Supervised labels and the exact episode oracle (plan D5, D6, D7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.env import Action, Episode, EnvState


@dataclass(frozen=True)
class StepLabels:
    succeeds: np.ndarray  # bool per pool model, given the history
    oracle_model: int  # cheapest model that succeeds, else the cheapest
    band: int  # pool models that fail this step given the history (0..4)


def step_labels(ep: Episode, s: EnvState) -> StepLabels:
    ok = ep.succeeds(s)
    winners = np.flatnonzero(ok)
    return StepLabels(succeeds=ok, oracle_model=int(winners[0]) if len(winners) else 0, band=int((~ok).sum()))


def halt_label(ep: Episode, s: EnvState) -> int:
    """1 if the round's answer is wrong and a strictly more expensive, affordable model succeeds (D5)."""
    if s.rounds != 1 or s.cur_correct:
        return 0
    ok = ep.succeeds(s)
    return int(any(ok[m] for m in ep.escalation_options(s)))


@dataclass(frozen=True)
class OraclePlan:
    n_correct: int
    spent: float
    actions: tuple[Action, ...]


def _future_key(ep: Episode, s: EnvState) -> tuple:
    """Everything about a step-boundary state that affects the future, except spend."""
    last_fail = s.last_fail_tier if ep.skeleton.ep_type == "propagation" else None
    return s.n_correct, s.slot, s.step_index, s.retried, s.done, last_fail


def _finish_step(ep: Episode, s: EnvState) -> list[tuple[EnvState, tuple[Action, ...]]]:
    """Every way to finish the current step from s, keeping the cheapest per future key."""
    best: dict[tuple, tuple[EnvState, tuple[Action, ...]]] = {}

    def walk(state: EnvState, path: tuple[Action, ...]) -> None:
        for a in ep.legal_actions(state):
            nxt = ep.apply(state, a)
            if nxt.done or a[0] == "stop":
                key = _future_key(ep, nxt)
                if key not in best or nxt.spent < best[key][0].spent:
                    best[key] = (nxt, path + (a,))
            else:
                walk(nxt, path + (a,))

    walk(s, ())
    return list(best.values())


def plan(ep: Episode, s: EnvState) -> OraclePlan:
    """Exact search: maximise correct steps, then minimise spend (D6)."""
    memo: dict[tuple, tuple[int, float, tuple[Action, ...]]] = {}

    def best(state: EnvState) -> tuple[int, float, tuple[Action, ...]]:
        if state.done:
            return state.n_correct, state.spent, ()
        key = (_future_key(ep, state), state.rounds, state.cur_model, state.cur_correct, state.spent)
        if key not in memo:
            result = None
            for nxt, path in _finish_step(ep, state):
                n, spent, rest = best(nxt)
                if result is None or (n, -spent) > (result[0], -result[1]):
                    result = (n, spent, path + rest)
            memo[key] = result
        return memo[key]

    n, spent, actions = best(s)
    return OraclePlan(n_correct=n, spent=spent, actions=actions)
