"""
Purpose:  Shared data contracts passed between the four components.
Owner:    Shared (changes need agreement from all four members)
Contract: v1
Inputs:   n/a
Outputs:  Dataclasses plus to_json / from_json helpers.
Notes:    Plain dataclasses keep the contracts dependency-free. Pydantic is used only
          at the HTTP edge in Phase 3. Every class carries `version`; any field change
          bumps it. Long text (prompts, outputs) is never stored inline: use *_key or
          *_ref fields that point into a cache or file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date, datetime
from typing import Any, Literal

CONTRACT_VERSION = "v1"

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
Memory = Literal["none", "session", "retrieval"]
Depth = Literal["low", "medium", "high"]
Verification = Literal["none", "self_check", "cross_model"]
Topology = Literal["single", "parallel", "sub_agents"]  # sub_agents reserved in v1
Halt = Literal["stop", "escalate"]
CardStatus = Literal["registered", "warmup", "active", "retired"]
EpisodeType = Literal["correlated", "retry", "propagation", "random"]
Split = Literal["train", "val", "test"]
Stage = Literal["supervised", "rl"]


# ---------------------------------------------------------------------------
# Catalogue (owned by C4, profiled by C3, read by C1)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ModelCard:
    model_id: str
    provider: str
    endpoint: str
    price_in_per_m: float            # USD per 1M input tokens
    price_out_per_m: float           # USD per 1M output tokens
    price_date: date                 # prices are always reported with their date
    context_limit: int
    max_output_tokens: int
    supports_tools: bool
    supports_reasoning_effort: bool
    capability: dict[str, float]     # step type -> score in [0, 1]
    probe_version: str               # which probe set produced the prior
    outcome_counts: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    status: CardStatus = "registered"
    version: str = CONTRACT_VERSION


@dataclass(slots=True)
class ToolCard:
    tool_name: str
    description: str
    description_emb_key: str
    version: str = CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Controller input (built by C4 or the C3 simulator, read by C1)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TraceItem:
    step_id: str
    round: int                       # 0 = first attempt, 1 = escalation
    model_id: str
    verifier_ok: bool                # noisy verifier signal, NEVER ground truth
    cost_usd: float
    latency_ms: float
    tokens_in: int
    tokens_out: int
    escalated: bool
    tool_error: bool
    output_emb_key: str | None       # None in the simulator
    dt: float = 1.0                  # time since previous item; constant 1.0 in the simulator
    version: str = CONTRACT_VERSION


@dataclass(slots=True)
class TaskRep:
    episode_id: str
    step_id: str
    round: int
    task_emb_key: str
    step_type: str
    step_emb_key: str
    trace: list[TraceItem]           # last k items, oldest first
    budget_total_usd: float
    budget_left_usd: float
    spent_usd: float
    step_index: int                  # 0-based
    expected_steps: int
    tools_offered: list[str] = field(default_factory=list)
    lam: float | None = None         # cost weight; set in RL and serving
    version: str = CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Controller output (C1 -> C4 and C1 -> C2)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Strategy:
    step_id: str
    round: int
    model_id: str
    halt: Halt
    tools: list[str] = field(default_factory=list)
    memory: Memory = "none"
    depth: Depth = "medium"
    verification: Verification = "none"
    topology: Topology = "single"
    complexity: float = 0.0          # [0, 1]
    confidence: float = 0.0          # calibrated success probability of chosen model
    controller_version: str = ""
    version: str = CONTRACT_VERSION


@dataclass(slots=True)
class DecisionRecord:
    step_id: str
    round: int
    strategy: Strategy
    log_probs: dict[str, float]      # per head
    entropy: dict[str, float]        # per head
    complexity_band: int             # 0..4, used for RL baseline buckets
    lam: float | None
    decision_state: list[float]      # final z_H, length d
    diagnostics: dict[str, float] = field(default_factory=dict)  # surprise_norm, gate_rate, latency_ms
    version: str = CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Execution and measurement (C4 -> C3, C3 -> C2)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ExecutionTrace:
    step_id: str
    round: int
    model_id: str
    request_hash: str
    tools_called: list[str]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    finish_reason: str
    error: str | None
    output_text_ref: str             # key into output storage, never inline text
    version: str = CONTRACT_VERSION


@dataclass(slots=True)
class Score:
    step_id: str
    round: int
    quality: float                   # [0, 1], uses ground truth
    verifier_ok: bool
    cost_usd: float
    latency_ms: float
    lam: float
    reward: float
    components: dict[str, float]     # quality, cost_term, latency_term, penalty
    scorer_version: str
    version: str = CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Simulation and training
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EpisodeSkeleton:
    episode_id: str
    episode_type: EpisodeType
    prompt_ids: list[str]            # ordered; retries are generated live, not stored here
    budget_usd: float
    split: Split
    seed: int
    version: str = CONTRACT_VERSION


@dataclass(slots=True)
class CheckpointMeta:
    checkpoint_id: str
    config_name: str                 # full | gate_off | gate_stateless | memory_off | context_free | *_attn
    stage: Stage
    seed: int
    parent_id: str | None
    path: str
    temperature: float
    norm_stats_ref: str
    contract_version: str
    val_metrics: dict[str, float]
    promoted: bool
    created_at: datetime
    version: str = CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
_NESTED: dict[type, dict[str, tuple[str, type]]] = {
    TaskRep: {"trace": ("list", TraceItem)},
    DecisionRecord: {"strategy": ("one", Strategy)},
}
_DATES: dict[type, dict[str, type]] = {
    ModelCard: {"price_date": date},
    CheckpointMeta: {"created_at": datetime},
}


def _default(o: Any) -> Any:
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"Not serialisable: {type(o)}")


def to_dict(obj: Any) -> dict:
    assert is_dataclass(obj), "to_dict expects a contract dataclass"
    return asdict(obj)


def to_json(obj: Any) -> str:
    return json.dumps(to_dict(obj), default=_default, sort_keys=True)


def from_dict(cls: type, data: dict) -> Any:
    data = dict(data)
    for name, (kind, sub) in _NESTED.get(cls, {}).items():
        if name in data and data[name] is not None:
            data[name] = (
                [from_dict(sub, x) for x in data[name]] if kind == "list" else from_dict(sub, data[name])
            )
    for name, typ in _DATES.get(cls, {}).items():
        if isinstance(data.get(name), str):
            data[name] = typ.fromisoformat(data[name])
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    if data.get("version", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise ValueError(f"{cls.__name__}: version {data['version']} != {CONTRACT_VERSION}")
    return cls(**data)


def from_json(cls: type, s: str) -> Any:
    return from_dict(cls, json.loads(s))


ALL_CONTRACTS = [
    ModelCard, ToolCard, TraceItem, TaskRep, Strategy, DecisionRecord,
    ExecutionTrace, Score, EpisodeSkeleton, CheckpointMeta,
]
