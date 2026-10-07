# Week 1: Contracts, Data and Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the v1 contracts and stub, a leakage-safe RouterBench table, cached prompt embeddings, episode skeletons, the workflow simulator, both oracles and the four baseline policies with real test-set numbers.

**Architecture:** `controller/contracts.py` fixes the shared types. `data/routerbench.py` turns the RouterBench 0-shot pickle into one Parquet table (one row per prompt, pool columns ordered cheapest first so model index = tier) plus `pool.json` ModelCards. `data/episodes.py` writes policy-independent skeletons; `data/env.py` is a pure-functional simulator (`EnvState` is frozen, `Episode.apply(state, action)` returns a new state) so `data/oracle.py` can search it exactly. `eval/policies.py` rolls policies through the simulator.

**Tech Stack:** Python 3.12, numpy, pandas, pyarrow, huggingface_hub, sentence-transformers (bge-small-en-v1.5), pytest.

**Spec:** `docs/IMPLEMENTATION_PLAN_neural_core.md` (v1.1 with §1a D1–D10). D10 records the review fixes this plan implements.

## Global Constraints

- Python 3.12 (`requires-python = ">=3.12,<3.13"`), venv at `.venv/` created with `uv venv --python 3.12 .venv`.
- Work directly on `main`; one commit per task.
- Packages at repo root: `controller/`, `data/`, `train/`, `eval/`. No `models/` directory.
- Contracts version string `"v1"`; trace length `K_TRACE = 2`; `MAX_ROUNDS = 2`; decision state length 128.
- Kept sources (D3, verified on the data): `mmlu` (all `mmlu-*` evals grouped), `hellaswag`, `arc-challenge`, `winogrande`, `mbpp`. `grade-school-math` is dropped (scores in 0.25 steps), MT-Bench is graded, the remaining small evals are not benchmark sources. `n_step_types = 5`.
- Pool, cheapest first: `mistralai/mistral-7b-chat`, `mistralai/mixtral-8x7b-chat`, `gpt-3.5-turbo-1106`, `gpt-4-1106-preview`. Model index == tier.
- "Cheaper/more expensive" always means lower/higher tier.
- Split 80/20 train/test, then 10% of train to val, stratified by source, seed 0, after de-duplicating prompt text.
- Episode type shares 40/25/20/15 (correlated/retry/propagation/random), 3–6 steps, budget = U(0.7, 1.6) × cost of the strongest model on the skeleton's prompts (so ≈1/3 cannot afford it every step).
- Verifier flip probability 0.15, deterministic per (seed, episode, step, round, model).
- Generated artefacts (`*.npy`, `*.parquet`, `data/cache/`, `runs/`, `checkpoints/`) are never committed.

## Review Focus

1. Duplicate prompt texts across splits — dedupe must run before splitting (test in Task 3).
2. A kept source that suddenly contains a non-0/1 score — must raise, not silently binarise (test in Task 3).
3. Always-strongest under a budget it cannot afford — the unaffordable round must end the episode with that step scored 0, not crash or overspend (test in Task 6).
4. Retry of an item that fails again — it must not loop forever; an item is retried at most once (test in Task 6).
5. Oracle search on a propagation episode with 6 steps — must stay exact and finish quickly (brute-force comparison and timing test in Task 7).

---

### Task 1: Scaffold and plan v1.1

**Files:**
- Create: `pyproject.toml`, `README.md` (overwrite empty), `requirements.txt` (overwrite empty), `.gitignore` (overwrite), `controller/__init__.py`, `data/__init__.py`, `eval/__init__.py`, `train/__init__.py`, `tests/__init__.py`
- Modify: `docs/IMPLEMENTATION_PLAN_neural_core.md` (replace with v1.1 from `docs/guide.md` + new D10)
- Delete: `docs/guide.md`, `models/`

**Interfaces:**
- Produces: importable empty packages; `pytest` runs from repo root with `pythonpath = ["."]`.

- [ ] **Step 1: Create the venv and remove old artefacts**

```bash
rm -rf venv models
uv venv --python 3.12 .venv
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "neural-core"
version = "0.1.0"
description = "C1 adaptive orchestration controller prototype"
requires-python = ">=3.12,<3.13"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"
```

- [ ] **Step 3: Write `requirements.txt`**

```
numpy>=2.0
pandas>=2.2
pyarrow>=16
huggingface_hub>=0.24
sentence-transformers>=3.0
torch>=2.4
pytest>=8
```

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
*.npy
*.parquet
data/cache/
checkpoints/
runs/
hf_cache/
```

- [ ] **Step 5: Write `README.md`**

```markdown
# neural-core

Prototype of the C1 adaptive orchestration controller (J26-DS-359). Plan: `docs/IMPLEMENTATION_PLAN_neural_core.md`.

## Setup

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

## Data pipeline

    .venv/bin/python -m data.routerbench   # table + split + pool.json -> data/cache/
    .venv/bin/python -m data.embed         # bge-small embeddings -> data/cache/embeddings.npy
    .venv/bin/python -m data.episodes      # episode skeletons -> data/cache/episodes_{split}.parquet
    .venv/bin/python -m eval.baselines     # baseline numbers on test -> runs/baselines_test.csv

## Tests

    .venv/bin/python -m pytest
```

- [ ] **Step 6: Install and create packages**

```bash
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p controller data eval train tests
touch controller/__init__.py data/__init__.py eval/__init__.py train/__init__.py tests/__init__.py
```

- [ ] **Step 7: Replace the plan with v1.1 and add D10**

Copy `docs/guide.md` over `docs/IMPLEMENTATION_PLAN_neural_core.md`, delete `docs/guide.md`, and insert this section directly before `### D9. Engineering` (it becomes D9's sibling; keep D9 text but change "Branch `feat/controller-prototype`" to "Work on `main`"):

```markdown
### D10. Review fixes (v1.1 → v1.2)

- **Control-row fairness.** `context_free` masks earlier steps' trace items, the budget token and the memory readout, but still sees the **current step's own round-1 result**. Escalating within a step is not workflow context; masking it would hand `full` a gain on plain random chains.
- **Halt timing.** A `Strategy` is emitted per round. `model_id` comes from the forward pass before the round; `halt` is filled from a second forward pass on the post-round `TaskRep` (trace includes that round). `halt = "escalate"` is invalid in round `MAX_ROUNDS = 2`.
- **Retry episodes.** The skeleton lists `n` prompts; the episode always has `n` steps. In a retry episode, if a step's final answer is wrong and the item is not already a retry, the next step is the same item (marked retried); otherwise the next listed prompt. Items are retried at most once.
- **Step outcome.** A step's outcome is its last round's answer. Propagation uses the tier of the model that gave that final wrong answer.
- **TTT surprise.** `surprise = value(obs) − W·key(obs)`. The gate rate is one scalar per decision. At an episode's first decision there is no outcome to store, so the memory update is skipped. Gradients flow through `W` across the whole episode.
- **Sources (checked on the data).** Kept: MMLU (grouped), HellaSwag, ARC-Challenge, Winogrande, MBPP → 5 step types. Dropped: grade-school-math (0.25-step scores), MT-Bench (graded), small miscellaneous evals. Duplicate prompt texts (16) are removed before splitting.
- **Pool.** mistral-7b-chat, mixtral-8x7b-chat, gpt-3.5-turbo-1106, gpt-4-1106-preview (mean cost per call rises with tier). Costs are RouterBench's per-prompt `total_cost`, prices as of the RouterBench release (2024-03).
- **ModelCard v1** (until `model_onboarding.md` lands): `model_id, provider, tier, mean_cost_usd, price_source, price_date, capability` where `capability` = train accuracy per step type + log10(mean cost).
- **Sequencing.** `env.py` and the episode oracle move to Day 5 (D1 needs them for the behaviour policy).
```

- [ ] **Step 8: Verify pytest runs**

Run: `.venv/bin/python -m pytest`
Expected: `no tests ran` (exit code 5), no import errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore README.md controller data eval train tests docs
git rm -q --cached -r models 2>/dev/null || true
git commit -m "chore: python 3.12 scaffold and plan v1.1 with review fixes"
```

---

### Task 2: Contracts v1 and stub

**Files:**
- Create: `controller/contracts.py`, `controller/stub.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: `VERSION = "v1"`, `K_TRACE = 2`, `MAX_ROUNDS = 2`, `STATE_DIM = 128`; dataclasses `ModelCard`, `TraceItem`, `TaskRep`, `Strategy`, `DecisionRecord` (fields exactly as D8/D10); `validate_strategy(s: Strategy, pool: list[ModelCard]) -> None` (raises `ValueError` naming the bad field); `validate_task_rep(rep: TaskRep) -> None`; `strategy_from_dict(d: dict) -> Strategy`; `stub_strategy(rep: TaskRep, pool: list[ModelCard], rng: random.Random | None = None) -> Strategy`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_contracts.py
import json
import random
from dataclasses import asdict

import pytest

from controller.contracts import (
    K_TRACE,
    ModelCard,
    Strategy,
    TaskRep,
    TraceItem,
    strategy_from_dict,
    validate_strategy,
    validate_task_rep,
)
from controller.stub import stub_strategy


def make_cards():
    return [
        ModelCard(model_id=f"m{i}", provider="test", tier=i, mean_cost_usd=10.0 ** (i - 4),
                  price_source="test", price_date="2024-03", capability=[0.5, 0.5, i - 4.0])
        for i in range(4)
    ]


def make_rep(round_=1, trace=(), budget_left=0.5):
    return TaskRep(episode_id="e0", step_id="e0-s0", round=round_, task_emb_key="p0", step_type=0,
                   step_emb_key="p0", trace=list(trace), budget_total_usd=1.0, budget_left_usd=budget_left,
                   spent_usd=1.0 - budget_left, step_index=0, expected_steps=3)


def make_item(round_=1):
    return TraceItem(model_id="m0", round=round_, verifier_ok=False, cost_usd=0.1, latency_ms=0.0,
                     tokens_out=0, escalated=False, output_emb_key=None)


@pytest.mark.parametrize("round_", [1, 2])
def test_stub_returns_valid_strategy(round_):
    pool, rng = make_cards(), random.Random(0)
    for _ in range(200):
        s = stub_strategy(make_rep(round_), pool, rng)
        validate_strategy(s, pool)
        assert (s.step_id, s.round) == ("e0-s0", round_)


def test_stub_never_escalates_in_last_round():
    pool, rng = make_cards(), random.Random(0)
    assert {stub_strategy(make_rep(2), pool, rng).halt for _ in range(100)} == {"stop"}


@pytest.mark.parametrize("field,value", [
    ("model_id", "nope"), ("halt", "maybe"), ("memory", "disk"), ("depth", "max"),
    ("verification", "vote"), ("topology", "mesh"), ("complexity", 1.5), ("confidence", -0.1),
    ("round", 3), ("version", "v0"),
])
def test_validate_strategy_rejects_bad_field(field, value):
    s = Strategy(step_id="x", round=1, model_id="m0", halt="stop")
    setattr(s, field, value)
    with pytest.raises(ValueError, match=field):
        validate_strategy(s, make_cards())


def test_validate_strategy_rejects_escalate_in_last_round():
    s = Strategy(step_id="x", round=2, model_id="m0", halt="escalate")
    with pytest.raises(ValueError, match="halt"):
        validate_strategy(s, make_cards())


def test_strategy_round_trips_through_json():
    s = Strategy(step_id="x", round=1, model_id="m1", halt="escalate", tools=["search"],
                 complexity=0.25, confidence=0.9)
    assert strategy_from_dict(json.loads(json.dumps(asdict(s)))) == s


def test_validate_task_rep_limits_trace():
    validate_task_rep(make_rep(trace=[make_item(), make_item()]))
    with pytest.raises(ValueError, match="trace"):
        validate_task_rep(make_rep(trace=[make_item()] * (K_TRACE + 1)))


def test_validate_task_rep_rejects_budget_above_total():
    with pytest.raises(ValueError, match="budget"):
        validate_task_rep(make_rep(budget_left=2.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_contracts.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'controller.contracts'`

- [ ] **Step 3: Write `controller/contracts.py`**

```python
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
```

- [ ] **Step 4: Write `controller/stub.py`**

```python
"""Random but schema-valid controller, published on day 1 so other components can integrate."""

from __future__ import annotations

import random

from controller.contracts import MAX_ROUNDS, ModelCard, Strategy, TaskRep


def stub_strategy(rep: TaskRep, pool: list[ModelCard], rng: random.Random | None = None) -> Strategy:
    rng = rng or random.Random()
    halt = rng.choice(("stop", "escalate")) if rep.round < MAX_ROUNDS else "stop"
    return Strategy(step_id=rep.step_id, round=rep.round, model_id=rng.choice(pool).model_id,
                    halt=halt, complexity=rng.random(), confidence=rng.random())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_contracts.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add controller tests/test_contracts.py
git commit -m "feat(contracts): v1 contracts, validators and random stub"
```

---

### Task 3: RouterBench table, pool and split

**Files:**
- Create: `data/routerbench.py`
- Test: `tests/test_routerbench.py`

**Interfaces:**
- Consumes: `controller.contracts.ModelCard`.
- Produces: constants `SOURCES`, `POOL`, `CACHE_DIR = Path("data/cache")`; `build_table(raw: pd.DataFrame, pool=POOL, sources=SOURCES) -> pd.DataFrame` with columns `prompt_id, source, step_type, text, correct_0..3 (bool), cost_0..3 (float), band (int)`; `split_prompts(table, seed=0, test_frac=0.2, val_frac=0.1) -> pd.Series` of `"train"|"val"|"test"`; `make_model_cards(table, pool=POOL, sources=SOURCES) -> list[ModelCard]` (needs a `split` column); `prepare(cache_dir=CACHE_DIR, raw_path=None, seed=0)`; `load_prepared(cache_dir=CACHE_DIR) -> tuple[pd.DataFrame, list[ModelCard]]`; class `OutcomeTable(prompt_ids, step_type, correct, cost)` with `.index: dict[str, int]`, `.n_models`, `.band`, `OutcomeTable.from_frame(df)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_routerbench.py
import numpy as np
import pandas as pd
import pytest

from data.routerbench import POOL, OutcomeTable, build_table, make_model_cards, split_prompts

COSTS = (1e-5, 2e-5, 3e-5, 4e-5)


def row(sid, eval_name, prompt, scores, costs=COSTS):
    r = {"sample_id": sid, "eval_name": eval_name, "prompt": prompt}
    for m, s, c in zip(POOL, scores, costs):
        r[m] = s
        r[f"{m}|total_cost"] = c
    return r


def raw_frame(rows):
    return pd.DataFrame(rows)


def test_build_table_groups_mmlu_and_drops_other_sources():
    raw = raw_frame([
        row("a", "mmlu-anatomy", "['q1']", (0, 0, 1, 1)),
        row("b", "mmlu-law", "['q2']", (1, 1, 1, 1)),
        row("c", "grade-school-math", "['q3']", (0.25, 0.5, 0.75, 1)),
        row("d", "hellaswag", "['q4']", (0, 0, 0, 0)),
    ])
    t = build_table(raw)
    assert list(t.prompt_id) == ["a", "b", "d"]
    assert list(t.source) == ["mmlu", "mmlu", "hellaswag"]
    assert list(t.step_type) == [0, 0, 1]
    assert list(t.band) == [2, 0, 4]
    assert t.correct_3.dtype == bool


def test_build_table_parses_list_prompts():
    t = build_table(raw_frame([row("a", "mbpp", "['part one', 'part two']", (1, 1, 1, 1))]))
    assert t.text.iloc[0] == "part one\npart two"


def test_build_table_drops_missing_results():
    t = build_table(raw_frame([
        row("a", "mbpp", "['x']", (1, np.nan, 1, 1)),
        row("b", "mbpp", "['y']", (1, 1, 1, 1)),
    ]))
    assert list(t.prompt_id) == ["b"]


def test_build_table_rejects_non_binary_scores_in_kept_source():
    with pytest.raises(ValueError, match="non-binary"):
        build_table(raw_frame([row("a", "winogrande", "['x']", (0.5, 1, 1, 1))]))


def test_build_table_dedupes_prompt_text():
    t = build_table(raw_frame([
        row("a", "mbpp", "['same']", (1, 1, 1, 1)),
        row("b", "mbpp", "['same']", (0, 1, 1, 1)),
    ]))
    assert list(t.prompt_id) == ["a"]


def synthetic_table(n=1000):
    rng = np.random.default_rng(0)
    raw = raw_frame([
        row(f"s{i}", "hellaswag" if i % 3 else "arc-challenge", f"['prompt {i}']",
            tuple(int(x) for x in rng.random(4) < (0.3, 0.5, 0.7, 0.9)))
        for i in range(n)
    ])
    t = build_table(raw)
    t["split"] = split_prompts(t, seed=0)
    return t


def test_split_fractions_and_coverage():
    t = synthetic_table()
    counts = t.split.value_counts(normalize=True)
    assert set(t.split) == {"train", "val", "test"}
    assert counts["test"] == pytest.approx(0.2, abs=0.01)
    assert counts["val"] == pytest.approx(0.08, abs=0.01)


def test_split_is_deterministic_and_text_disjoint():
    t = synthetic_table()
    assert (split_prompts(t, seed=0) == t.split).all()
    assert not (split_prompts(t, seed=1) == t.split).all()
    assert t.groupby("text").split.nunique().max() == 1


def test_model_cards_ordered_by_tier_with_capability():
    t = synthetic_table()
    cards = make_model_cards(t)
    assert [c.model_id for c in cards] == list(POOL)
    assert [c.tier for c in cards] == [0, 1, 2, 3]
    assert len(cards[0].capability) == 5 + 1
    assert cards[3].capability[1] > cards[0].capability[1]  # hellaswag accuracy rises with tier


def test_outcome_table_from_frame():
    t = synthetic_table(30)
    ot = OutcomeTable.from_frame(t)
    assert ot.correct.shape == (len(t), 4) and ot.n_models == 4
    assert ot.index[t.prompt_id.iloc[5]] == 5
    assert (ot.band == t.band.to_numpy()).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_routerbench.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.routerbench'`

- [ ] **Step 3: Write `data/routerbench.py`**

```python
"""RouterBench 0-shot -> one row per prompt with pool outcomes, a leakage-safe split and ModelCards
(plan §4.1, D3, D10). Pool columns are ordered cheapest first, so model index == tier."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from controller.contracts import ModelCard

HF_REPO = "withmartian/routerbench"
HF_FILE = "routerbench_0shot.pkl"
CACHE_DIR = Path("data/cache")

# Sources with 0/1 scores (D3). grade-school-math scores come in 0.25 steps and MT-Bench is graded.
SOURCES = ("mmlu", "hellaswag", "arc-challenge", "winogrande", "mbpp")
POOL = (
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
)
PRICE_SOURCE = "RouterBench 0-shot per-prompt total_cost"
PRICE_DATE = "2024-03"


@dataclass
class OutcomeTable:
    """Array view of the prompt table used by the simulator."""

    prompt_ids: np.ndarray
    step_type: np.ndarray
    correct: np.ndarray  # bool [n_prompts, n_models]
    cost: np.ndarray  # float [n_prompts, n_models]
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.prompt_ids)}

    @property
    def n_models(self) -> int:
        return self.correct.shape[1]

    @property
    def band(self) -> np.ndarray:
        return (~self.correct).sum(axis=1)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> OutcomeTable:
        n = sum(c.startswith("correct_") for c in df.columns)
        return cls(
            prompt_ids=df["prompt_id"].to_numpy(),
            step_type=df["step_type"].to_numpy(),
            correct=df[[f"correct_{i}" for i in range(n)]].to_numpy(bool),
            cost=df[[f"cost_{i}" for i in range(n)]].to_numpy(float),
        )


def source_of(eval_name: pd.Series) -> pd.Series:
    return eval_name.where(~eval_name.str.startswith("mmlu-"), "mmlu")


def parse_prompt(raw) -> str:
    """RouterBench stores prompts as a stringified list of message parts."""
    try:
        parts = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return str(raw)
    if isinstance(parts, (list, tuple)):
        return "\n".join(str(p) for p in parts)
    return str(parts)


def build_table(raw: pd.DataFrame, pool=POOL, sources=SOURCES) -> pd.DataFrame:
    src = source_of(raw["eval_name"])
    df = raw[src.isin(sources)].assign(source=src)
    df = df.dropna(subset=list(pool) + [f"{m}|total_cost" for m in pool])
    if not np.isin(df[list(pool)].to_numpy(), (0.0, 1.0)).all():
        raise ValueError("non-binary scores in a kept source")
    out = pd.DataFrame({
        "prompt_id": df["sample_id"].astype(str).to_numpy(),
        "source": df["source"].to_numpy(),
        "step_type": df["source"].map({s: i for i, s in enumerate(sources)}).to_numpy(),
        "text": df["prompt"].map(parse_prompt).to_numpy(),
    })
    for i, m in enumerate(pool):
        out[f"correct_{i}"] = df[m].to_numpy().astype(bool)
        out[f"cost_{i}"] = df[f"{m}|total_cost"].to_numpy(float)
    out = out.drop_duplicates("text", keep="first").reset_index(drop=True)
    out["band"] = (~out[[f"correct_{i}" for i in range(len(pool))]].to_numpy()).sum(axis=1)
    return out


def split_prompts(table: pd.DataFrame, seed: int = 0, test_frac: float = 0.2,
                  val_frac: float = 0.1) -> pd.Series:
    """Split prompts once, stratified by source, before any episode is built (§4.1 step 4)."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=table.index, dtype=object)
    for _, idx in sorted(table.groupby("source").groups.items()):
        idx = rng.permutation(np.asarray(idx))
        n_test = round(len(idx) * test_frac)
        n_val = round((len(idx) - n_test) * val_frac)
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def make_model_cards(table: pd.DataFrame, pool=POOL, sources=SOURCES) -> list[ModelCard]:
    train = table[table["split"] == "train"]
    cards = []
    for i, m in enumerate(pool):
        acc = [float(train.loc[train["step_type"] == t, f"correct_{i}"].mean()) for t in range(len(sources))]
        mean_cost = float(train[f"cost_{i}"].mean())
        cards.append(ModelCard(model_id=m, provider=m.split("/")[0] if "/" in m else "openai", tier=i,
                               mean_cost_usd=mean_cost, price_source=PRICE_SOURCE, price_date=PRICE_DATE,
                               capability=acc + [math.log10(mean_cost)]))
    costs = [c.mean_cost_usd for c in cards]
    if costs != sorted(costs):
        raise ValueError(f"pool must be ordered cheapest first, got mean costs {costs}")
    return cards


def prepare(cache_dir: Path = CACHE_DIR, raw_path: str | None = None, seed: int = 0):
    if raw_path is None:
        from huggingface_hub import hf_hub_download

        raw_path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    table = build_table(pd.read_pickle(raw_path))
    table["split"] = split_prompts(table, seed)
    cards = make_model_cards(table)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_dir / "prompts.parquet", index=False)
    (cache_dir / "pool.json").write_text(json.dumps([asdict(c) for c in cards], indent=2))
    return table, cards


def load_prepared(cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, list[ModelCard]]:
    cache_dir = Path(cache_dir)
    table = pd.read_parquet(cache_dir / "prompts.parquet")
    cards = [ModelCard(**d) for d in json.loads((cache_dir / "pool.json").read_text())]
    return table, cards


if __name__ == "__main__":
    table, cards = prepare()
    print(table.groupby(["source", "split"]).size().unstack())
    for c in cards:
        print(f"{c.tier} {c.model_id:32s} mean_cost={c.mean_cost_usd:.2e} acc={[round(a, 3) for a in c.capability[:-1]]}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_routerbench.py`
Expected: all pass.

- [ ] **Step 5: Build the real table**

Run: `.venv/bin/python -m data.routerbench`
Expected: ~27,2xx prompts over 5 sources and 3 splits; tiers 0–3 with mean cost rising.

- [ ] **Step 6: Commit**

```bash
git add data/routerbench.py tests/test_routerbench.py
git commit -m "feat(data): RouterBench table, 4-model pool and leakage-safe split"
```

---

### Task 4: Cached bge-small embeddings

**Files:**
- Create: `data/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `data.routerbench.CACHE_DIR`, `load_prepared`.
- Produces: `MODEL_NAME = "BAAI/bge-small-en-v1.5"`, `DIM = 384`; `build_cache(ids, texts, encode, cache_dir=CACHE_DIR, batch_size=1024, dim=DIM) -> np.ndarray`; `load_cache(cache_dir=CACHE_DIR) -> tuple[np.ndarray, dict[str, int]]` (memory-mapped float32 `[n, 384]`, id → row).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embed.py
import time

import numpy as np
import pytest

from data.embed import DIM, build_cache, load_cache


def fake_encode(texts):
    return np.stack([np.full(DIM, len(t), dtype=np.float32) for t in texts])


def test_cache_round_trip_keeps_id_order(tmp_path):
    ids, texts = ["a", "b", "c"], ["x", "yy", "zzz"]
    build_cache(ids, texts, fake_encode, cache_dir=tmp_path, batch_size=2)
    emb, index = load_cache(tmp_path)
    assert emb.shape == (3, DIM) and emb.dtype == np.float32
    assert emb[index["c"], 0] == 3.0


def test_build_cache_rejects_wrong_dim(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        build_cache(["a"], ["x"], lambda t: np.zeros((len(t), 7)), cache_dir=tmp_path)


def test_load_is_fast(tmp_path):
    n = 30_000
    build_cache([str(i) for i in range(n)], ["t"] * n,
                lambda t: np.zeros((len(t), DIM), np.float32), cache_dir=tmp_path, batch_size=10_000)
    start = time.perf_counter()
    emb, index = load_cache(tmp_path)
    assert time.perf_counter() - start < 1.0
    assert emb.shape == (n, DIM) and len(index) == n
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_embed.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.embed'`

- [ ] **Step 3: Write `data/embed.py`**

```python
"""Embed every prompt once with frozen bge-small and cache to .npy (plan §4.1 step 5)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from data.routerbench import CACHE_DIR, load_prepared

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

Encoder = Callable[[list[str]], np.ndarray]


def sentence_transformer_encoder(name: str = MODEL_NAME, device: str | None = None) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)


def build_cache(ids: Sequence[str], texts: Sequence[str], encode: Encoder, cache_dir: Path = CACHE_DIR,
                batch_size: int = 1024, dim: int = DIM) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encode(list(texts[start:start + batch_size])), dtype=np.float32))
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    emb = np.concatenate(chunks)
    if emb.shape != (len(ids), dim):
        raise ValueError(f"embedding shape {emb.shape} != {(len(ids), dim)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", emb)
    (cache_dir / "embedding_ids.json").write_text(json.dumps(list(ids)))
    return emb


def load_cache(cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, dict[str, int]]:
    cache_dir = Path(cache_dir)
    emb = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    ids = json.loads((cache_dir / "embedding_ids.json").read_text())
    return emb, {p: i for i, p in enumerate(ids)}


if __name__ == "__main__":
    table, _ = load_prepared()
    build_cache(table["prompt_id"].tolist(), table["text"].tolist(), sentence_transformer_encoder())
    start = time.perf_counter()
    emb, index = load_cache()
    print(f"loaded {emb.shape} in {time.perf_counter() - start:.3f}s")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_embed.py`
Expected: all pass.

- [ ] **Step 5: Build the real cache**

Run: `.venv/bin/python -m data.embed`
Expected: `loaded (27xxx, 384) in 0.0xs`.

- [ ] **Step 6: Commit**

```bash
git add data/embed.py tests/test_embed.py
git commit -m "feat(data): cached bge-small prompt embeddings"
```

---

### Task 5: Episode skeletons

**Files:**
- Create: `data/episodes.py`
- Create: `tests/conftest.py`
- Test: `tests/test_episodes.py`

**Interfaces:**
- Consumes: `OutcomeTable`, `CACHE_DIR`, `load_prepared`.
- Produces: `EPISODE_TYPES = ("correlated", "retry", "propagation", "random")`, `TYPE_SHARES`, `MIN_STEPS = 3`, `MAX_STEPS = 6`, `BUDGET_FACTOR = (0.7, 1.6)`; frozen dataclass `Skeleton(episode_id: str, ep_type: str, prompt_ids: tuple[str, ...], budget_usd: float)`; `type_counts(n) -> dict[str, int]`; `generate_skeletons(table, prompt_ids, n, seed, split) -> list[Skeleton]`; `to_frame(skeletons) -> pd.DataFrame` (one row per step); `from_frame(df) -> list[Skeleton]`; `load_skeletons(split, cache_dir=CACHE_DIR) -> list[Skeleton]`; test helper `tests.conftest.make_table(n, n_types, seed) -> OutcomeTable`.

- [ ] **Step 1: Write the shared fixture**

```python
# tests/conftest.py
import numpy as np
import pytest

from data.routerbench import OutcomeTable


def make_table(n=300, n_types=2, seed=0) -> OutcomeTable:
    """Synthetic outcome table: stronger (higher-tier) models succeed more often and cost more."""
    rng = np.random.default_rng(seed)
    correct = rng.random((n, 4)) < np.array([0.3, 0.5, 0.7, 0.9])
    cost = np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4 * rng.uniform(0.5, 1.5, (n, 1))
    return OutcomeTable(prompt_ids=np.array([f"p{i}" for i in range(n)]),
                        step_type=np.arange(n) % n_types, correct=correct, cost=cost)


@pytest.fixture
def table():
    return make_table()
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_episodes.py
import numpy as np
import pytest

from data.episodes import (
    BUDGET_FACTOR,
    EPISODE_TYPES,
    MAX_STEPS,
    MIN_STEPS,
    from_frame,
    generate_skeletons,
    to_frame,
    type_counts,
)


def test_type_counts_match_shares_exactly():
    assert type_counts(1000) == {"correlated": 400, "retry": 250, "propagation": 200, "random": 150}
    assert sum(type_counts(7).values()) == 7


def test_skeleton_shape_and_split_purity(table):
    ids = list(table.prompt_ids[:200])
    sks = generate_skeletons(table, ids, n=500, seed=0, split="train")
    assert len(sks) == 500 and len({s.episode_id for s in sks}) == 500
    assert {s.ep_type for s in sks} == set(EPISODE_TYPES)
    allowed = set(ids)
    for s in sks:
        assert MIN_STEPS <= len(s.prompt_ids) <= MAX_STEPS
        assert set(s.prompt_ids) <= allowed
        assert len(set(s.prompt_ids)) == len(s.prompt_ids)


def test_correlated_episodes_share_source_and_band(table):
    sks = generate_skeletons(table, list(table.prompt_ids), n=300, seed=0, split="train")
    for s in sks:
        if s.ep_type == "correlated":
            rows = [table.index[p] for p in s.prompt_ids]
            assert len(set(table.step_type[rows])) == 1
            assert len(set(table.band[rows])) == 1


def test_about_a_third_cannot_afford_strongest_every_step(table):
    sks = generate_skeletons(table, list(table.prompt_ids), n=3000, seed=0, split="train")
    short = [s.budget_usd < table.cost[[table.index[p] for p in s.prompt_ids], -1].sum() for s in sks]
    assert np.mean(short) == pytest.approx(1 / 3, abs=0.04)
    assert BUDGET_FACTOR == (0.7, 1.6)


def test_generation_is_deterministic(table):
    ids = list(table.prompt_ids)
    assert generate_skeletons(table, ids, 50, 3, "val") == generate_skeletons(table, ids, 50, 3, "val")
    assert generate_skeletons(table, ids, 50, 3, "val") != generate_skeletons(table, ids, 50, 4, "val")


def test_frame_round_trip(table):
    sks = generate_skeletons(table, list(table.prompt_ids), n=40, seed=0, split="test")
    df = to_frame(sks)
    assert len(df) == sum(len(s.prompt_ids) for s in sks)
    assert from_frame(df) == sks
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_episodes.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.episodes'`

- [ ] **Step 4: Write `data/episodes.py`**

```python
"""Policy-independent episode skeletons (plan §4.2–4.3, D1, D10). Outcomes are rolled out later
by data/env.py, because they depend on the models chosen earlier in the episode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.routerbench import CACHE_DIR, OutcomeTable, load_prepared

EPISODE_TYPES = ("correlated", "retry", "propagation", "random")
TYPE_SHARES = {"correlated": 0.40, "retry": 0.25, "propagation": 0.20, "random": 0.15}
MIN_STEPS, MAX_STEPS = 3, 6
# Budget = factor x cost of the strongest model on every listed prompt.
# P(factor < 1) = 0.3 / 0.9 = 1/3, so about a third cannot afford it on every step.
BUDGET_FACTOR = (0.7, 1.6)
SIZES = {"train": 20_000, "val": 2_000, "test": 5_000}


@dataclass(frozen=True)
class Skeleton:
    episode_id: str
    ep_type: str
    prompt_ids: tuple[str, ...]
    budget_usd: float


def type_counts(n: int) -> dict[str, int]:
    """Exact shares by largest remainder."""
    raw = {t: n * TYPE_SHARES[t] for t in EPISODE_TYPES}
    counts = {t: int(v) for t, v in raw.items()}
    by_remainder = sorted(EPISODE_TYPES, key=lambda t: raw[t] - counts[t], reverse=True)
    for t in by_remainder[: n - sum(counts.values())]:
        counts[t] += 1
    return counts


def _correlated_groups(table: OutcomeTable, rows: np.ndarray) -> list[np.ndarray]:
    """Prompt rows sharing a source and a difficulty band, large enough for any episode length."""
    key = table.step_type[rows] * 10 + table.band[rows]
    groups = [rows[key == k] for k in np.unique(key)]
    return [g for g in groups if len(g) >= MAX_STEPS]


def generate_skeletons(table: OutcomeTable, prompt_ids: Sequence[str], n: int, seed: int,
                       split: str) -> list[Skeleton]:
    rng = np.random.default_rng(seed)
    rows = np.array([table.index[p] for p in prompt_ids])
    groups = _correlated_groups(table, rows)
    sizes = np.array([len(g) for g in groups], dtype=float)
    counts = type_counts(n)
    types = rng.permutation(np.repeat(EPISODE_TYPES, [counts[t] for t in EPISODE_TYPES]))
    out = []
    for i, ep_type in enumerate(types):
        n_steps = int(rng.integers(MIN_STEPS, MAX_STEPS + 1))
        source = groups[rng.choice(len(groups), p=sizes / sizes.sum())] if ep_type == "correlated" else rows
        chosen = rng.choice(source, size=n_steps, replace=False)
        budget = float(rng.uniform(*BUDGET_FACTOR) * table.cost[chosen, -1].sum())
        out.append(Skeleton(f"{split}-{i:06d}", str(ep_type), tuple(str(p) for p in table.prompt_ids[chosen]), budget))
    return out


def to_frame(skeletons: Sequence[Skeleton]) -> pd.DataFrame:
    return pd.DataFrame([
        {"episode_id": s.episode_id, "ep_type": s.ep_type, "slot": j, "prompt_id": p,
         "budget_usd": s.budget_usd, "n_steps": len(s.prompt_ids)}
        for s in skeletons for j, p in enumerate(s.prompt_ids)
    ])


def from_frame(df: pd.DataFrame) -> list[Skeleton]:
    out = []
    for eid, g in df.sort_values(["episode_id", "slot"]).groupby("episode_id", sort=True):
        out.append(Skeleton(str(eid), str(g["ep_type"].iloc[0]), tuple(g["prompt_id"].astype(str)),
                            float(g["budget_usd"].iloc[0])))
    return out


def load_skeletons(split: str, cache_dir: Path = CACHE_DIR) -> list[Skeleton]:
    return from_frame(pd.read_parquet(Path(cache_dir) / f"episodes_{split}.parquet"))


def build_all(cache_dir: Path = CACHE_DIR, seed: int = 0, sizes: dict[str, int] = SIZES) -> None:
    prompts, _ = load_prepared(cache_dir)
    table = OutcomeTable.from_frame(prompts)
    for i, (split, n) in enumerate(sizes.items()):
        ids = prompts.loc[prompts["split"] == split, "prompt_id"].tolist()
        skeletons = generate_skeletons(table, ids, n, seed + i, split)
        to_frame(skeletons).to_parquet(Path(cache_dir) / f"episodes_{split}.parquet", index=False)
        print(split, n, pd.Series([s.ep_type for s in skeletons]).value_counts().to_dict())


if __name__ == "__main__":
    build_all()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_episodes.py`
Expected: all pass.

- [ ] **Step 6: Generate real skeletons and eyeball 100**

Run: `.venv/bin/python -m data.episodes` then
`.venv/bin/python -c "import pandas as pd; df=pd.read_parquet('data/cache/episodes_train.parquet'); print(df.head(100).to_string())"`
Expected: type counts 8000/5000/4000/3000 for train; rows look sane.

- [ ] **Step 7: Commit**

```bash
git add data/episodes.py tests/conftest.py tests/test_episodes.py
git commit -m "feat(data): episode skeleton generator with four types and budget pressure"
```

---

### Task 6: Workflow simulator

**Files:**
- Create: `data/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Skeleton`, `OutcomeTable`, `controller.contracts.TraceItem`, `MAX_ROUNDS`.
- Produces: `VERIFIER_FLIP_P = 0.15`; `Action = tuple[str, int | None]` (`("play", m)` or `("stop", None)`); frozen dataclass `EnvState(step_index, slot, retried, rounds, cur_model, cur_correct, last_fail_tier, budget_left, spent, n_correct, done, trace)`; class `Episode(skeleton, table, model_ids, flip_p=VERIFIER_FLIP_P, seed=0)` with `.n_steps`, `.n_models`, `initial_state()`, `succeeds(s) -> np.ndarray[bool]`, `cost(s, m)`, `affordable(s, m)`, `escalation_options(s) -> list[int]`, `legal_actions(s) -> list[Action]`, `play(s, m) -> EnvState`, `end_step(s) -> EnvState`, `apply(s, a) -> EnvState`, `quality(s) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_env.py
import numpy as np
import pytest

from data.env import Episode
from data.episodes import Skeleton
from data.routerbench import OutcomeTable

IDS = ["m0", "m1", "m2", "m3"]


def tiny_table(correct_rows):
    correct = np.array(correct_rows, dtype=bool)
    n = len(correct)
    cost = np.tile([1.0, 2.0, 4.0, 8.0], (n, 1))
    return OutcomeTable(np.array([f"p{i}" for i in range(n)]), np.zeros(n, int), correct, cost)


def episode(ep_type, correct_rows, budget=100.0, flip_p=0.0):
    t = tiny_table(correct_rows)
    sk = Skeleton("e0", ep_type, tuple(t.prompt_ids), budget)
    return Episode(sk, t, IDS, flip_p=flip_p)


def run(ep, models):
    """Play one round per step with the given models; return final state."""
    s = ep.initial_state()
    for m in models:
        s = ep.end_step(ep.play(s, m))
    return s


def test_quality_and_cost_accumulate():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    s = run(ep, [0, 0, 3])
    assert s.done and s.n_correct == 2 and ep.quality(s) == pytest.approx(2 / 3)
    assert s.spent == 1 + 1 + 8 and s.budget_left == 100 - 10


def test_propagation_requires_stronger_model_after_failure():
    ep = episode("propagation", [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))  # tier 0 fails step 0
    assert list(ep.succeeds(s)) == [False, True, True, True]
    s = ep.end_step(ep.play(s, 1))  # tier 1 succeeds -> no propagation
    assert list(ep.succeeds(s)) == [True, True, True, True]


def test_propagation_does_not_apply_to_other_types():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))
    assert ep.succeeds(s).all()


def test_retry_repeats_failed_item_once():
    ep = episode("retry", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))  # fail item 0
    assert (s.slot, s.retried) == (0, True)
    s = ep.end_step(ep.play(s, 3))  # fail again: not retried a second time
    assert (s.slot, s.retried) == (1, False)
    s = ep.end_step(ep.play(s, 0))
    assert s.done and s.step_index == 3 and s.n_correct == 1


def test_successful_step_is_not_retried():
    ep = episode("retry", [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))
    assert (s.slot, s.retried) == (1, False)


def test_escalation_must_be_strictly_more_expensive_and_replaces_answer():
    ep = episode("random", [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.play(ep.initial_state(), 1)
    assert ep.escalation_options(s) == [2, 3]
    with pytest.raises(ValueError, match="strictly more expensive"):
        ep.play(s, 1)
    s = ep.play(s, 2)
    assert s.cur_correct and s.spent == 2 + 4 and ep.escalation_options(s) == []
    with pytest.raises(ValueError, match="no rounds left"):
        ep.play(s, 3)
    assert ep.end_step(s).n_correct == 1


def test_unaffordable_round_ends_episode_with_step_scored_zero():
    ep = episode("random", [[1, 1, 1, 1]] * 3, budget=10.0)
    s = ep.end_step(ep.play(ep.initial_state(), 3))  # spend 8, budget left 2
    s = ep.play(s, 3)  # costs 8 > 2: not run
    assert s.done and s.n_correct == 1 and s.spent == 8 and ep.quality(s) == pytest.approx(1 / 3)


def test_legal_actions():
    ep = episode("random", [[0, 0, 0, 0]] * 3, budget=6.0)
    s0 = ep.initial_state()
    assert ep.legal_actions(s0) == [("play", m) for m in range(4)]
    s1 = ep.play(s0, 0)
    assert ep.legal_actions(s1) == [("stop", None), ("play", 1), ("play", 2), ("play", 3)]
    assert ep.escalation_options(s1) == [1, 2]  # 8 > 5 left: not affordable


def test_verifier_flips_at_configured_rate_and_is_deterministic():
    rows = [[1, 1, 1, 1]] * 6
    flips = []
    for i in range(400):
        t = tiny_table(rows)
        sk = Skeleton(f"e{i}", "random", tuple(t.prompt_ids), 1e9)
        ep = Episode(sk, t, IDS, flip_p=0.15, seed=0)
        s = run(ep, [0] * 6)
        flips += [not item.verifier_ok for item in s.trace]
        assert run(ep, [0] * 6).trace == s.trace
    assert np.mean(flips) == pytest.approx(0.15, abs=0.02)


def test_trace_items_record_rounds():
    ep = episode("random", [[0, 1, 1, 1]] * 3)
    s = ep.play(ep.play(ep.initial_state(), 0), 2)
    assert [(i.model_id, i.round, i.escalated, i.cost_usd) for i in s.trace] == [
        ("m0", 1, False, 1.0), ("m2", 2, True, 4.0)]
    assert [i.verifier_ok for i in s.trace] == [False, True]  # flip_p = 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.env'`

- [ ] **Step 3: Write `data/env.py`**

```python
"""Workflow simulator over RouterBench outcomes (plan D1, D3, D4, D5, D10).

States are immutable, so the episode oracle can search the simulator exactly.
A step is one or two rounds: play a model, optionally escalate to a strictly
more expensive one, then stop. The last round's answer is the step's answer."""

from __future__ import annotations

import zlib
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
                 flip_p: float = VERIFIER_FLIP_P, seed: int = 0):
        self.skeleton = skeleton
        self.table = table
        self.model_ids = list(model_ids)
        self.flip_p = flip_p
        self.seed = seed
        self.rows = np.array([table.index[p] for p in skeleton.prompt_ids])
        self.n_steps = len(self.rows)
        self.n_models = table.n_models

    def initial_state(self) -> EnvState:
        return EnvState(budget_left=self.skeleton.budget_usd)

    def row(self, s: EnvState) -> int:
        return int(self.rows[s.slot])

    def succeeds(self, s: EnvState) -> np.ndarray:
        """Which pool models answer the current step correctly, given the history (D3)."""
        ok = self.table.correct[self.row(s)].copy()
        if self.skeleton.ep_type == "propagation" and s.last_fail_tier is not None:
            ok &= np.arange(self.n_models) > s.last_fail_tier
        return ok

    def cost(self, s: EnvState, m: int) -> float:
        return float(self.table.cost[self.row(s), m])

    def affordable(self, s: EnvState, m: int) -> bool:
        return self.cost(s, m) <= s.budget_left

    def escalation_options(self, s: EnvState) -> list[int]:
        if s.done or s.rounds != 1:
            return []
        return [m for m in range(s.cur_model + 1, self.n_models) if self.affordable(s, m)]

    def legal_actions(self, s: EnvState) -> list[Action]:
        if s.done:
            return []
        if s.rounds == 0:
            return [("play", m) for m in range(self.n_models)]
        stop: list[Action] = [("stop", None)]
        if s.rounds < MAX_ROUNDS:
            return stop + [("play", m) for m in range(s.cur_model + 1, self.n_models)]
        return stop

    def _verifier_ok(self, s: EnvState, m: int, correct: bool) -> bool:
        key = f"{self.seed}|{self.skeleton.episode_id}|{s.step_index}|{s.rounds + 1}|{m}"
        flipped = zlib.crc32(key.encode()) / 2**32 < self.flip_p
        return correct != flipped

    def play(self, s: EnvState, m: int) -> EnvState:
        if s.done:
            raise ValueError("episode is over")
        if s.rounds >= MAX_ROUNDS:
            raise ValueError("no rounds left in this step")
        if s.rounds == 1 and m <= s.cur_model:
            raise ValueError("escalation must use a strictly more expensive model")
        cost = self.cost(s, m)
        if cost > s.budget_left:  # D3: round not run, step scores 0, episode ends
            return replace(s, cur_correct=False, done=True)
        correct = bool(self.succeeds(s)[m])
        item = TraceItem(model_id=self.model_ids[m], round=s.rounds + 1,
                         verifier_ok=self._verifier_ok(s, m, correct), cost_usd=cost, latency_ms=0.0,
                         tokens_out=0, escalated=s.rounds == 1, output_emb_key=None)
        return replace(s, rounds=s.rounds + 1, cur_model=m, cur_correct=correct,
                       budget_left=s.budget_left - cost, spent=s.spent + cost, trace=s.trace + (item,))

    def end_step(self, s: EnvState) -> EnvState:
        if s.done or s.rounds == 0:
            raise ValueError("no round has been played in this step")
        failed = not s.cur_correct
        retry = self.skeleton.ep_type == "retry" and failed and not s.retried
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_env.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add data/env.py tests/test_env.py
git commit -m "feat(data): workflow simulator with retry, propagation, budget and noisy verifier"
```

---

### Task 7: Labels and episode oracle

**Files:**
- Create: `data/oracle.py`
- Test: `tests/test_oracle.py`

**Interfaces:**
- Consumes: `Episode`, `EnvState`, `Action`.
- Produces: frozen dataclass `StepLabels(succeeds: np.ndarray, oracle_model: int, band: int)`; `step_labels(ep, s) -> StepLabels`; `halt_label(ep, s) -> int`; frozen dataclass `OraclePlan(n_correct: int, spent: float, actions: tuple[Action, ...])`; `plan(ep, s) -> OraclePlan` (max correct steps, ties broken by lower spend, from any non-terminal state).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_oracle.py
import time

import numpy as np
import pytest

from data.env import Episode
from data.episodes import Skeleton, generate_skeletons
from data.oracle import halt_label, plan, step_labels
from tests.conftest import make_table
from tests.test_env import IDS, episode


def brute_force(ep, s):
    """Best (n_correct, -spent) over every legal action sequence, no pruning."""
    if s.done:
        return s.n_correct, -s.spent
    return max(brute_force(ep, ep.apply(s, a)) for a in ep.legal_actions(s))


def test_step_labels_pick_cheapest_success_else_cheapest():
    ep = episode("random", [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    lab = step_labels(ep, ep.initial_state())
    assert list(lab.succeeds) == [False, False, True, True] and lab.oracle_model == 2 and lab.band == 2
    s = ep.end_step(ep.play(ep.initial_state(), 0))
    assert step_labels(ep, s).oracle_model == 0 and step_labels(ep, s).band == 4


def test_halt_label():
    ep = episode("random", [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], budget=5.0)
    s = ep.play(ep.initial_state(), 0)  # wrong; tier 2 succeeds, costs 4 <= 4 left
    assert halt_label(ep, s) == 1
    assert halt_label(ep, ep.play(ep.initial_state(), 2)) == 0  # already right
    tight = episode("random", [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], budget=4.0)
    assert halt_label(tight, tight.play(tight.initial_state(), 0)) == 0  # 4 > 3 left
    nobody = episode("random", [[0, 0, 0, 0]] * 3)
    assert halt_label(nobody, nobody.play(nobody.initial_state(), 0)) == 0


def test_plan_prefers_cheapest_correct_route():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    p = plan(ep, ep.initial_state())
    assert p.n_correct == 3 and p.spent == 2 + 1 + 8
    assert p.actions == (("play", 1), ("stop", None), ("play", 0), ("stop", None), ("play", 3), ("stop", None))


def test_plan_skips_hopeless_step_to_save_budget():
    # Budget 9: solving step 0 (tier 3, cost 8) leaves 1, so neither later step (tier 2, cost 4) is affordable.
    ep = episode("random", [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]], budget=9.0)
    p = plan(ep, ep.initial_state())
    assert p.n_correct == 2 and p.spent == 1 + 4 + 4
    assert p.actions[0] == ("play", 0)


@pytest.mark.parametrize("ep_type", ["correlated", "retry", "propagation", "random"])
def test_plan_matches_brute_force(ep_type):
    table = make_table(n=60, seed=1)
    sks = generate_skeletons(table, list(table.prompt_ids), n=200, seed=2, split="t")
    sks = [s for s in sks if s.ep_type == ep_type and len(s.prompt_ids) <= 4][:15]
    for sk in sks:
        ep = Episode(sk, table, IDS)
        p = plan(ep, ep.initial_state())
        n, neg_spent = brute_force(ep, ep.initial_state())
        assert p.n_correct == n and p.spent == pytest.approx(-neg_spent)
        s = ep.initial_state()
        for a in p.actions:
            s = ep.apply(s, a)
        assert s.done and s.n_correct == p.n_correct and s.spent == pytest.approx(p.spent)


def test_plan_from_mid_step_state():
    ep = episode("random", [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.play(ep.initial_state(), 0)
    p = plan(ep, s)
    assert p.actions[0] == ("play", 2) and p.n_correct == 3


def test_plan_is_fast_on_six_step_propagation():
    table = make_table(n=300, seed=3)
    sk = Skeleton("e", "propagation", tuple(table.prompt_ids[:6]), 1.0)
    ep = Episode(sk, table, IDS)
    start = time.perf_counter()
    plan(ep, ep.initial_state())
    assert time.perf_counter() - start < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_oracle.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.oracle'`

- [ ] **Step 3: Write `data/oracle.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_oracle.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add data/oracle.py tests/test_oracle.py
git commit -m "feat(data): per-step labels, halt labels and exact episode oracle"
```

---

### Task 8: Baseline policies and test-set numbers

**Files:**
- Create: `eval/policies.py`, `eval/baselines.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: `Episode`, `EnvState`, `Action`, `plan`, `load_skeletons`, `load_prepared`, `OutcomeTable`.
- Produces: protocol `Policy` with `name: str` and `act(ep, s) -> Action`; classes `AlwaysCheapest`, `AlwaysStrongest`, `RandomPolicy(seed=0)`, `EpisodeOracle`; dataclass `EpisodeResult(episode_id, ep_type, quality, cost_usd)`; `run_episode(ep, policy) -> EpisodeResult`; `evaluate(skeletons, table, model_ids, policies) -> pd.DataFrame` (one row per policy × episode, columns `policy, episode_id, ep_type, quality, cost_usd`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_policies.py
import pandas as pd

from data.episodes import generate_skeletons
from eval.policies import AlwaysCheapest, AlwaysStrongest, EpisodeOracle, RandomPolicy, evaluate, run_episode
from tests.conftest import make_table
from tests.test_env import IDS, episode


def test_always_cheapest_and_strongest():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    cheap = run_episode(ep, AlwaysCheapest())
    strong = run_episode(ep, AlwaysStrongest())
    assert (cheap.quality, cheap.cost_usd) == (1 / 3, 3.0)
    assert (strong.quality, strong.cost_usd) == (1.0, 24.0)


def test_always_strongest_under_tight_budget_stops_early():
    ep = episode("random", [[1, 1, 1, 1]] * 3, budget=10.0)
    r = run_episode(ep, AlwaysStrongest())
    assert r.quality == 1 / 3 and r.cost_usd == 8.0


def test_oracle_policy_reaches_plan_value():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    r = run_episode(ep, EpisodeOracle())
    assert (r.quality, r.cost_usd) == (1.0, 11.0)


def test_oracle_beats_every_baseline_on_every_episode():
    table = make_table(n=200, seed=5)
    sks = generate_skeletons(table, list(table.prompt_ids), n=120, seed=0, split="t")
    df = evaluate(sks, table, IDS, [AlwaysCheapest(), AlwaysStrongest(), RandomPolicy(0), EpisodeOracle()])
    wide = df.pivot(index="episode_id", columns="policy", values="quality")
    for name in ("always_cheapest", "always_strongest", "random"):
        assert (wide["oracle"] >= wide[name]).all()
    assert len(df) == 4 * 120 and set(df.ep_type) == {"correlated", "retry", "propagation", "random"}


def test_random_policy_is_seeded():
    table = make_table(n=100, seed=6)
    sks = generate_skeletons(table, list(table.prompt_ids), n=30, seed=0, split="t")
    a = evaluate(sks, table, IDS, [RandomPolicy(1)])
    b = evaluate(sks, table, IDS, [RandomPolicy(1)])
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_policies.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.policies'`

- [ ] **Step 3: Write `eval/policies.py`**

```python
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
             policies: Sequence[Policy]) -> pd.DataFrame:
    rows = []
    for policy in policies:
        for sk in skeletons:
            result = run_episode(Episode(sk, table, model_ids), policy)
            rows.append({"policy": policy.name, **asdict(result)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Write `eval/baselines.py`**

```python
"""Baseline numbers on a split: python -m eval.baselines [split]"""

from __future__ import annotations

import sys
from pathlib import Path

from data.episodes import load_skeletons
from data.routerbench import OutcomeTable, load_prepared
from eval.policies import AlwaysCheapest, AlwaysStrongest, EpisodeOracle, RandomPolicy, evaluate


def main(split: str = "test", out_dir: Path = Path("runs")) -> None:
    prompts, cards = load_prepared()
    table = OutcomeTable.from_frame(prompts)
    df = evaluate(load_skeletons(split), table, [c.model_id for c in cards],
                  [AlwaysCheapest(), AlwaysStrongest(), RandomPolicy(0), EpisodeOracle()])
    wide = df.pivot(index="episode_id", columns="policy", values="quality")
    for name in ("always_cheapest", "always_strongest", "random"):
        assert (wide["oracle"] >= wide[name]).all(), f"oracle below {name} on some episode"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / f"baselines_{split}.csv", index=False)
    print(df.groupby("policy")[["quality", "cost_usd"]].mean().to_string(float_format="%.4f"))
    print(df.pivot_table(index="ep_type", columns="policy", values="quality").to_string(float_format="%.3f"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest`
Expected: whole suite passes.

- [ ] **Step 6: Produce the real baseline numbers**

Run: `.venv/bin/python -m eval.baselines test`
Expected: table printed; oracle quality highest; `runs/baselines_test.csv` written.

- [ ] **Step 7: Commit**

```bash
git add eval/policies.py eval/baselines.py tests/test_policies.py
git commit -m "feat(eval): baseline policies and test-set baseline numbers"
```
