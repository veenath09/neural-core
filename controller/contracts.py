"""Shared contracts v1 (plan §1a D8, D10). Other components build against these types;
change them only together with a version bump."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

VERSION = "v1"
K_TRACE = 2  # trace items shown explicitly; older history goes through the memory (D2)
MAX_ROUNDS = 2  # one escalation per step (D5)
STATE_DIM = 128

Halt = Literal["stop", "escalate"]
Memory = Literal["none", "session", "retrieval"]
Depth = Literal["low", "medium", "high"]
Verification = Literal["none", "self_check", "cross_model"]
Topology = Literal["single", "parallel", "sub_agents"]


@dataclass
class ModelCard:
    model_id: str
    provider: str
    tier: int  # 0 = cheapest in the pool
    mean_cost_usd: float  # mean per-call cost on the train split
    price_source: str
    price_date: str  # date the prices refer to
    capability: list[float]  # train accuracy per step type, then log10(mean_cost_usd)
    version: str = VERSION


@dataclass
class TraceItem:
    model_id: str
    round: int
    verifier_ok: bool  # noisy verifier signal, never ground truth (D4)
    cost_usd: float
    latency_ms: float
    tokens_out: int
    escalated: bool
    output_emb_key: str | None  # cache key; None in the prototype


@dataclass
class TaskRep:
    episode_id: str
    step_id: str
    round: int
    task_emb_key: str
    step_type: int
    step_emb_key: str
    trace: list[TraceItem]  # last K_TRACE rounds
    budget_total_usd: float
    budget_left_usd: float
    spent_usd: float
    step_index: int
    expected_steps: int
    lam: float | None = None  # RL only
    version: str = VERSION


@dataclass
class Strategy:
    step_id: str
    round: int
    model_id: str
    halt: Halt  # filled from the post-round forward pass (D10)
    tools: list[str] = field(default_factory=list)
    memory: Memory = "none"
    depth: Depth = "medium"
    verification: Verification = "none"
    topology: Topology = "single"
    complexity: float = 0.0
    confidence: float = 0.0
    version: str = VERSION


@dataclass
class DecisionRecord:
    step_id: str
    round: int
    strategy: Strategy
    log_probs: dict[str, float]
    entropy: dict[str, float]
    complexity_band: int  # 0..4
    lam: float | None
    decision_state: list[float]  # len STATE_DIM
    state_version: str = VERSION


def validate_strategy(s: Strategy, pool: list[ModelCard]) -> None:
    errors = []
    if s.model_id not in {c.model_id for c in pool}:
        errors.append(f"model_id {s.model_id!r} not in pool")
    for name, allowed in [("halt", Halt), ("memory", Memory), ("depth", Depth),
                          ("verification", Verification), ("topology", Topology)]:
        if getattr(s, name) not in get_args(allowed):
            errors.append(f"{name} {getattr(s, name)!r} not in {get_args(allowed)}")
    for name in ("complexity", "confidence"):
        if not 0.0 <= getattr(s, name) <= 1.0:
            errors.append(f"{name} {getattr(s, name)} outside [0, 1]")
    if not 1 <= s.round <= MAX_ROUNDS:
        errors.append(f"round {s.round} outside 1..{MAX_ROUNDS}")
    elif s.round == MAX_ROUNDS and s.halt == "escalate":
        errors.append(f"halt cannot be 'escalate' in round {MAX_ROUNDS}")
    if s.version != VERSION:
        errors.append(f"version {s.version!r} != {VERSION!r}")
    if errors:
        raise ValueError("; ".join(errors))


def validate_task_rep(rep: TaskRep) -> None:
    errors = []
    if len(rep.trace) > K_TRACE:
        errors.append(f"trace has {len(rep.trace)} items, max {K_TRACE}")
    if not 1 <= rep.round <= MAX_ROUNDS:
        errors.append(f"round {rep.round} outside 1..{MAX_ROUNDS}")
    if not 0.0 <= rep.budget_left_usd <= rep.budget_total_usd + 1e-12:
        errors.append(f"budget_left_usd {rep.budget_left_usd} outside [0, budget_total_usd]")
    if rep.spent_usd < 0:
        errors.append(f"spent_usd {rep.spent_usd} negative")
    if not 0 <= rep.step_index < rep.expected_steps:
        errors.append(f"step_index {rep.step_index} outside 0..{rep.expected_steps - 1}")
    if rep.version != VERSION:
        errors.append(f"version {rep.version!r} != {VERSION!r}")
    if errors:
        raise ValueError("; ".join(errors))


def strategy_from_dict(d: dict) -> Strategy:
    return Strategy(**d)
