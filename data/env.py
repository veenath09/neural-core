"""Workflow simulator over RouterBench outcomes (plan D1, D3, D4, D5, D10).

States are immutable, so the episode oracle can search the simulator exactly.
A step is one or two rounds: play a model, optionally escalate to a strictly
more expensive one, then stop. The last round's answer is the step's answer.

"More expensive" is by pool index (pools are cost-ordered) unless the episode is given
model_costs (the card costs), in which case it is a strictly higher model_cost: a pool that
grows at run time appends joiners at the end whatever their price.

Optional hidden episode difficulty (hard_p > 0; default 0 = the original simulator): with probability
hard_p (fixed per episode by a hash), the episode is "hard" and the cheaper half of the models fail every
step of it, whatever the prompt. Nothing in the current prompt reveals it; only earlier results in the same
episode do. This is the dependency a workflow-aware controller should exploit and a per-query router cannot."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from controller.contracts import MAX_ROUNDS, TraceItem
from data.episodes import Skeleton
from data.routerbench import OutcomeTable

VERIFIER_FLIP_P = 0.15

Action = tuple[str, int | None]  # ("play", model) or ("stop", None)


@dataclass(frozen=True)
class EnvState:
    step_index: int = 0  # steps completed so far
    slot: int = 0  # position of the current item in skeleton.prompt_ids
    retried: bool = False  # the current item is a retry of the previous step
    rounds: int = 0  # rounds played in the current step
    cur_model: int | None = None  # model that gave the current answer
    cur_correct: bool = False
    last_fail_tier: int | None = None  # tier whose answer failed the previous step
    budget_left: float = 0.0
    spent: float = 0.0
    n_correct: int = 0
    done: bool = False
    trace: tuple[TraceItem, ...] = ()


class Episode:
    def __init__(self, skeleton: Skeleton, table: OutcomeTable, model_ids: Sequence[str],
                 flip_p: float = VERIFIER_FLIP_P, seed: int = 0, model_costs: np.ndarray | None = None,
                 hard_p: float = 0.0):
        self.skeleton = skeleton
        self.table = table
        self.model_ids = list(model_ids)
        self.flip_p = flip_p
        self.seed = seed
        self.rows = np.array([table.index[p] for p in skeleton.prompt_ids])
        self.n_steps = len(self.rows)
        self.n_models = table.n_models
        self.model_costs = None if model_costs is None else np.asarray(model_costs, dtype=float)
        if self.model_costs is not None and self.model_costs.shape != (self.n_models,):
            raise ValueError(f"model_costs shape {self.model_costs.shape} != ({self.n_models},)")
        self.hard_p = hard_p
        key = f"hard|{seed}|{skeleton.episode_id}"
        u = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest()) / 2**64
        self.hard = bool(hard_p > 0 and u < hard_p)
        order = np.argsort(self.model_costs, kind="stable") if self.model_costs is not None else np.arange(self.n_models)
        self.cheap = np.zeros(self.n_models, dtype=bool)
        self.cheap[order[: self.n_models // 2]] = True  # the cheaper half fails throughout a hard episode

    def initial_state(self) -> EnvState:
        return EnvState(budget_left=self.skeleton.budget_usd)

    def row(self, s: EnvState) -> int:
        return int(self.rows[s.slot])

    def succeeds(self, s: EnvState) -> np.ndarray:
        """Which pool models answer the current step correctly, given the history (D3)."""
        ok = self.table.correct[self.row(s)].copy()
        if self.hard:
            ok &= ~self.cheap
        if self.skeleton.ep_type == "propagation" and s.last_fail_tier is not None:
            ok &= np.arange(self.n_models) > s.last_fail_tier
        return ok

    def cost(self, s: EnvState, m: int) -> float:
        return float(self.table.cost[self.row(s), m])

    def affordable(self, s: EnvState, m: int) -> bool:
        return self.cost(s, m) <= s.budget_left

    def pricier(self, cur: int) -> list[int]:
        """Models an escalation from cur may use: strictly higher model_cost, or higher index without model_costs."""
        if self.model_costs is None:
            return list(range(cur + 1, self.n_models))
        return [m for m in range(self.n_models) if self.model_costs[m] > self.model_costs[cur]]

    def escalation_options(self, s: EnvState) -> list[int]:
        if s.done or s.rounds != 1:
            return []
        return [m for m in self.pricier(s.cur_model) if self.affordable(s, m)]

    def legal_actions(self, s: EnvState) -> list[Action]:
        if s.done:
            return []
        if s.rounds == 0:
            return [("play", m) for m in range(self.n_models)]
        stop: list[Action] = [("stop", None)]
        if s.rounds < MAX_ROUNDS:
            return stop + [("play", m) for m in self.pricier(s.cur_model)]
        return stop

    def _verifier_ok(self, s: EnvState, m: int, correct: bool) -> bool:
        key = f"{self.seed}|{self.skeleton.episode_id}|{s.step_index}|{s.rounds + 1}|{m}"
        # blake2b, not crc32: crc32 is linear, which made flips across models, rounds and seeds correlated
        u = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest()) / 2**64
        flipped = u < self.flip_p
        return correct != flipped

    def play(self, s: EnvState, m: int) -> EnvState:
        if s.done:
            raise ValueError("episode is over")
        if s.rounds >= MAX_ROUNDS:
            raise ValueError("no rounds left in this step")
        if s.rounds == 1 and m not in self.pricier(s.cur_model):
            raise ValueError("escalation must use a strictly more expensive model")
        cost = self.cost(s, m)
        if cost > s.budget_left:  # D3: round not run, step scores 0, episode ends
            return replace(s, cur_correct=False, done=True)
        correct = bool(self.succeeds(s)[m])
        item = TraceItem(model_id=self.model_ids[m], round=s.rounds + 1,
                         verifier_ok=self._verifier_ok(s, m, correct), cost_usd=cost, latency_ms=0.0,
                         tokens_out=0, escalated=s.rounds == 1, output_emb_key=None,
                         step_index=s.step_index, step_emb_key=self.skeleton.prompt_ids[s.slot])
        return replace(s, rounds=s.rounds + 1, cur_model=m, cur_correct=correct,
                       budget_left=s.budget_left - cost, spent=s.spent + cost, trace=s.trace + (item,))

    def end_step(self, s: EnvState) -> EnvState:
        if s.done or s.rounds == 0:
            raise ValueError("no round has been played in this step")
        failed = not s.cur_correct
        # Retry on the verifier's flag, not true correctness: s.trace[-1] is this step's final
        # round (rounds > 0 here), so using true correctness would leak noiseless ground truth
        # to a returning prompt, bypassing the verifier noise (D4).
        flagged = not s.trace[-1].verifier_ok
        retry = self.skeleton.ep_type == "retry" and flagged and not s.retried
        step_index = s.step_index + 1
        return replace(s, step_index=step_index, slot=s.slot if retry else s.slot + 1, retried=retry,
                       rounds=0, cur_model=None, cur_correct=False,
                       last_fail_tier=s.cur_model if failed else None,
                       n_correct=s.n_correct + int(s.cur_correct), done=step_index >= self.n_steps)

    def apply(self, s: EnvState, a: Action) -> EnvState:
        kind, m = a
        if kind == "play":
            return self.play(s, m)
        if kind == "stop":
            return self.end_step(s)
        raise ValueError(f"unknown action {a!r}")

    def quality(self, s: EnvState) -> float:
        return s.n_correct / self.n_steps
