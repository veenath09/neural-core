"""Part 2 Task 3: step environments for online RL.

Two StepEnvironment implementations drive a controller session one decision
at a time:

- SimStepEnv wraps data.env.Episode. Retry and propagation dynamics stay in
  Episode; this is just a step-at-a-time facade over Episode.play/end_step.
- LiveStepEnv wraps execution.engine.ExecutionEngine with a real verifier
  (ExactMatchVerifier or JudgeModelVerifier), for the live execution side.

Both expose the same protocol so a single driver (Task 4) can run either.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from controller.contracts import Strategy, TraceItem
from data.env import Episode, EnvState
from execution.engine import ExecutionEngine

VERIFIER_QUALITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class RoundOutcome:
    trace_item: TraceItem | None
    quality: float
    cost_usd: float


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    step_key: str
    step_text: str | None
    step_type: int


class StepEnvironment(Protocol):
    ep_type: str

    def start(self) -> tuple[str, str | None, float, int]:
        """Return (task_key, task_text, budget_usd, expected_steps)."""
        ...

    def current_step(self) -> StepSpec | None:
        """Return the current step's spec, or None when the episode is over."""
        ...

    def play(self, model_id: str, round: int) -> RoundOutcome:
        ...

    def end_step(self) -> None:
        ...


class SimStepEnv:
    """Step-at-a-time facade over data.env.Episode (the RouterBench simulator)."""

    def __init__(self, episode: Episode, quality_source: str = "verifier"):
        if quality_source not in ("verifier", "truth"):
            raise ValueError(f"unknown quality_source {quality_source!r}")
        self.episode = episode
        self.quality_source = quality_source
        self.ep_type = episode.skeleton.ep_type
        self._state: EnvState = episode.initial_state()

    @property
    def done(self) -> bool:
        return self._state.done

    @property
    def state(self) -> EnvState:
        return self._state

    @property
    def verifier_flip_p(self) -> float:
        """Known flip probability of the quality signal (the simulator's verifier), 0 for ground truth."""
        return self.episode.flip_p if self.quality_source == "verifier" else 0.0

    @property
    def quality_truth(self) -> float:
        return self.episode.quality(self._state)

    def start(self) -> tuple[str, str | None, float, int]:
        return (
            self.episode.skeleton.prompt_ids[0],
            None,
            self.episode.skeleton.budget_usd,
            self.episode.n_steps,
        )

    def current_step(self) -> StepSpec | None:
        s = self._state
        if s.done:
            return None
        row = self.episode.row(s)
        return StepSpec(
            step_id=f"{self.episode.skeleton.episode_id}-{s.step_index}",
            step_key=self.episode.skeleton.prompt_ids[s.slot],
            step_text=None,
            step_type=int(self.episode.table.step_type[row]),
        )

    def play(self, model_id: str, round: int) -> RoundOutcome:
        m = self.episode.model_ids.index(model_id)
        before = self._state
        after = self.episode.play(before, m)
        self._state = after
        if after.done and len(after.trace) == len(before.trace):
            # Unaffordable play: episode ends with no trace item appended, no cost.
            return RoundOutcome(trace_item=None, quality=0.0, cost_usd=0.0)
        item = after.trace[-1]
        if self.quality_source == "truth":
            quality = float(after.cur_correct)
        else:
            quality = float(item.verifier_ok)
        return RoundOutcome(trace_item=item, quality=quality, cost_usd=item.cost_usd)

    def end_step(self) -> None:
        if self._state.done:
            return
        self._state = self.episode.end_step(self._state)


@dataclass(frozen=True)
class LiveStep:
    text: str
    step_type: int
    gold: str | None = None
    rubric: str | None = None


def _step_emb_key(text: str) -> str:
    """Stable hash key for an unseen live step's text, so it embeds once."""
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


class LiveStepEnv:
    """Step-at-a-time facade over the execution engine with a real verifier."""

    def __init__(self, engine: ExecutionEngine, steps: Sequence[LiveStep], verifier,
                 budget_usd: float, ep_type: str = "live"):
        self.engine = engine
        self.steps = list(steps)
        self.verifier = verifier
        self.budget_total = float(budget_usd)
        self.budget_left = float(budget_usd)
        self.ep_type = ep_type
        self.n_steps = len(self.steps)
        self.step_index = 0
        self.spent = 0.0
        self._done = self.n_steps == 0
        self._execution_count = 0

    @property
    def done(self) -> bool:
        return self._done

    def start(self) -> tuple[str, str | None, float, int]:
        first = self.steps[0] if self.steps else None
        task_key = _step_emb_key(first.text) if first is not None else ""
        task_text = first.text if first is not None else None
        return (task_key, task_text, self.budget_total, self.n_steps)

    def current_step(self) -> StepSpec | None:
        if self._done or self.step_index >= self.n_steps:
            return None
        step = self.steps[self.step_index]
        key = _step_emb_key(step.text)
        return StepSpec(
            step_id=f"live-{self.step_index}",
            step_key=key,
            step_text=step.text,
            step_type=int(step.step_type),
        )

    def play(self, model_id: str, round: int) -> RoundOutcome:
        step = self.steps[self.step_index]
        strategy = Strategy(step_id=f"live-{self.step_index}", round=round, model_id=model_id,
                            halt="stop")
        self._execution_count += 1
        execution_id = f"{strategy.step_id}-r{round}-{self._execution_count}"
        result = self.engine.execute(strategy, step.text, execution_id)
        cost = sum(s.cost_usd for s in result.trace.steps)

        if step.rubric is not None:
            score = self.verifier.verify(question=step.text, answer=result.output_text,
                                         rubric=step.rubric, reference=step.gold,
                                         generation_cost_usd=cost)
        else:
            score = self.verifier.verify(answer=result.output_text, reference=step.gold,
                                         cost_usd=cost)
        quality = float(score.quality)

        item = TraceItem(
            model_id=model_id,
            round=round,
            verifier_ok=quality >= VERIFIER_QUALITY_THRESHOLD,
            cost_usd=cost,
            latency_ms=0.0,
            tokens_out=0,
            escalated=round == 2,
            output_emb_key=None,
            step_index=self.step_index,
            step_emb_key=_step_emb_key(step.text),
        )

        self.spent += cost
        self.budget_left -= cost
        if self.budget_left < 0:
            self._done = True

        return RoundOutcome(trace_item=item, quality=quality, cost_usd=cost)

    def end_step(self) -> None:
        if self._done:
            return
        self.step_index += 1
        if self.step_index >= self.n_steps:
            self._done = True
