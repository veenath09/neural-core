"""
Purpose:  The only interfaces components may use to call each other.
Owner:    Shared (changes need agreement from all four members)
Contract: v1
Notes:    The simulator (C3) and the live executor (C4) both satisfy EnvironmentAPI,
          so training and evaluation code is identical in simulation and live runs.
          Storage protocols have a Parquet/in-memory backend (Phases 0-2) and a
          PostgreSQL/Redis backend (Phase 3).
"""
from __future__ import annotations

from typing import Any, Literal, Protocol

from common.contracts import (
    CheckpointMeta, DecisionRecord, EpisodeSkeleton, ExecutionTrace, ModelCard,
    Score, Strategy, TaskRep, ToolCard, TraceItem,
)

Mode = Literal["greedy", "sample", "serve"]


class ControllerAPI(Protocol):  # C1
    def decide(
        self, rep: TaskRep, pool: list[ModelCard], tools: list[ToolCard], mode: Mode
    ) -> tuple[Strategy, DecisionRecord]:
        """Choose the strategy for one round of one step."""

    def observe(self, episode_id: str, item: TraceItem) -> None:
        """Update the episode's memory W from the round that just finished."""

    def reset(self, episode_id: str) -> None:
        """Clear episode state (W, trace). Called at episode start and end."""


class EnvironmentAPI(Protocol):  # C3 simulator, C4 live executor
    def reset(self, skeleton: EpisodeSkeleton, lam: float) -> TaskRep:
        """Start an episode and return the first TaskRep."""

    def step(self, strategy: Strategy) -> tuple[TaskRep | None, TraceItem, Score, bool]:
        """Run one round. Returns (next TaskRep or None, trace item, score, done)."""

    def counterfactual(self, rep: TaskRep, pool: list[ModelCard]) -> dict[str, bool]:
        """Simulator only: would each pool model succeed on this step, given history?
        Live executor raises NotImplementedError."""


class ExecutorAPI(Protocol):  # C4
    def run(self, strategy: Strategy, request: dict[str, Any]) -> tuple[ExecutionTrace, TraceItem, dict]:
        """Execute a strategy against providers. Returns trace, item for C1, OpenAI response."""


class ScorerAPI(Protocol):  # C3
    def score(self, trace: ExecutionTrace, reference: str | None, lam: float) -> Score:
        """Score one executed round."""


class LearnerAPI(Protocol):  # C2
    def train_supervised(self, cfg: dict) -> CheckpointMeta: ...
    def train_rl(self, cfg: dict, init: CheckpointMeta) -> CheckpointMeta: ...
    def promote(self, ckpt: CheckpointMeta) -> bool: ...


class CatalogueAPI(Protocol):  # C4 owns, C3 updates profiles, C1 reads
    def active_pool(self) -> list[ModelCard]: ...
    def register(self, card: ModelCard) -> None: ...
    def update_profile(self, model_id: str, step_type: str, quality: float) -> None: ...
    def retire(self, model_id: str) -> None: ...


class SessionStore(Protocol):  # per-episode state: dict in Phases 0-2, Redis in Phase 3
    def get(self, episode_id: str) -> dict[str, Any] | None: ...
    def put(self, episode_id: str, state: dict[str, Any]) -> None: ...
    def delete(self, episode_id: str) -> None: ...


class RecordStore(Protocol):  # Parquet in Phases 0-2, PostgreSQL in Phase 3
    def write_decision(self, rec: DecisionRecord) -> None: ...
    def write_execution(self, trace: ExecutionTrace) -> None: ...
    def write_score(self, score: Score) -> None: ...
    def flush(self) -> None: ...
