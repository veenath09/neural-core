"""Baseline policies rolled through the simulator (plan §7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from data.env import Action, Episode, EnvState
from data.episodes import Skeleton
from data.oracle import plan
from data.routerbench import OutcomeTable


class Policy(Protocol):
    name: str

    def act(self, ep: Episode, s: EnvState) -> Action: ...


class AlwaysCheapest:
    name = "always_cheapest"

    def act(self, ep: Episode, s: EnvState) -> Action:
        return ("play", 0) if s.rounds == 0 else ("stop", None)


class AlwaysStrongest:
    name = "always_strongest"

    def act(self, ep: Episode, s: EnvState) -> Action:
        return ("play", ep.n_models - 1) if s.rounds == 0 else ("stop", None)


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, ep: Episode, s: EnvState) -> Action:
        return ("play", int(self.rng.integers(ep.n_models))) if s.rounds == 0 else ("stop", None)


class EpisodeOracle:
    """Follows the exact episode plan; re-plans if the state is not the one it expected."""

    name = "oracle"

    def __init__(self):
        self._queue: list[Action] = []
        self._expect: EnvState | None = None

    def act(self, ep: Episode, s: EnvState) -> Action:
        if not self._queue or self._expect != s:
            self._queue = list(plan(ep, s).actions)
        a = self._queue.pop(0)
        self._expect = ep.apply(s, a)
        return a


@dataclass
class EpisodeResult:
    episode_id: str
    ep_type: str
    quality: float
    cost_usd: float


def run_episode(ep: Episode, policy: Policy) -> EpisodeResult:
    s = ep.initial_state()
    while not s.done:
        s = ep.apply(s, policy.act(ep, s))
    return EpisodeResult(ep.skeleton.episode_id, ep.skeleton.ep_type, ep.quality(s), s.spent)


def evaluate(skeletons: Sequence[Skeleton], table: OutcomeTable, model_ids: Sequence[str],
             policies: Sequence[Policy], **episode_kw) -> pd.DataFrame:
    """episode_kw go to every Episode (e.g. hard_p, model_costs), so baselines see the same simulator."""
    rows = []
    for policy in policies:
        for sk in skeletons:
            result = run_episode(Episode(sk, table, model_ids, **episode_kw), policy)
            rows.append({"policy": policy.name, **asdict(result)})
    return pd.DataFrame(rows)
