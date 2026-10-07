# Week 2: Controller Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train the C1 controller: feature builder, H/L core, liquid-gated episode memory, encoder, heads, batched rollout engine and supervised training loop. Then train `memory_off`, `gate_off`, `full` and `gate_stateless` once each, with residual and gate diagnostics.

**Architecture:**
- **Features.** `controller/features.py` turns simulator states into tensors without reading ground truth.
- **Core.** `controller/core.py` follows HRM-Text's H/L schedule and staircase gradient horizon.
- **Memory.** `controller/memory.py` is the delta-rule memory with forgetting, plus three gates: CfC, stateless and constant.
- **Model.** `controller/model.py` wires encoder, memory, core and heads.
- **Rollout.** `train/rollout.py` plays batches of episodes in lockstep through `data/env.py`, one forward pass per decision, and is shared by training and evaluation.
- **Training.** `train/train.py` trains with a behaviour-policy mixture and keeps an EMA of the weights.

**Tech Stack:** Python 3.12, PyTorch 2.x, numpy, pandas, matplotlib, pytest.

**Spec:** `docs/IMPLEMENTATION_PLAN_neural_core.md`, v1.3. D11 is binding and wins over §5 and D2. Also D1, D4, D5, D7, D10.

## Global Constraints

- Work directly on `main`, one commit per task. Run commands from the repo root with `.venv/bin/python`.
- Width `d = 128`; trace length `K_TRACE = 2`; `MAX_ROUNDS = 2`; 2 H-cycles × 3 L-steps.
- Gradient horizon `bp = bp_min + ⌊min(1, step/(0.2·total))·(5 − bp_min)⌋` with `bp_min = 2`, split `H_bp = min(2, bp−1)`, `L_bp = bp − H_bp`.
- Memory: `k` unit-norm, `s = v − W k`, `r ∈ (0,1)`, `α = σ(·)` with bias −4, `W ← W₀ + (1−α)(W−W₀) + r·s kᵀ`, `W₀` learned and zero-initialised, readout `RMSNorm(W q)`, `Δt = 1/n_steps` per write.
- CfC gate: 16 units, backbone 32, LeCun-tanh backbone, state reset per episode.
- Behaviour mixture per step: per-step oracle 50%, random 25%, current model 25% (oracle instead of model in epoch 1). Random escalation probability 0.3.
- Optimiser AdamW (lr 3e-4, weight decay 0.01), cosine decay, batch 256 episodes, at most 20 epochs, early stop on validation loss with patience 3, gradient clip 1.0, EMA 0.999.
- Configs (in `controller/model.py`): `full`, `gate_off`, `gate_stateless`, `memory_off`, `context_free`.
- Features never read `cur_correct`, `n_correct` or the outcome table; only targets do.
- Generated artefacts (`runs/`, `data/processed/`) are never committed.
- Every adapted or re-implemented file names its sources in a header (D11 licences).

## Review Focus

1. Ground truth leaking into the controller's inputs: a feature that changes when only `cur_correct` changes. Tested in Task 2.
2. `context_free` seeing earlier steps, the task prompt or the budget: its outputs must not change when those are perturbed, but must change when the current step's own round-1 item changes. Tested in Task 5.
3. Memory blowing up over a full episode of writes: 12 writes with random inputs must stay finite and bounded. Tested in Task 4.
4. Lockstep batching mixing up episodes' memories: an episode's greedy result must not depend on which other episodes share its batch. Tested in Task 6.
5. Escalation when no stronger model is affordable by card cost: the step must stop, not crash. Tested in Task 6.

---

### Task 1: Contract, simulator and metadata fixes

**Files:**
- Modify: `controller/contracts.py` (TraceItem), `data/env.py:104-106` (fill new fields), `data/routerbench.py` (`PRICE_DATE`), `requirements.txt` (matplotlib)
- Create: `LICENSE-HRM-Text` (verbatim copy of HRM-Text's LICENSE)
- Test: `tests/test_env.py`, `tests/test_routerbench.py`

**Interfaces:**
- Produces: `TraceItem.step_index: int | None = None`, `TraceItem.step_emb_key: str | None = None`; env fills both with the step's index and prompt id; `PRICE_DATE = "not stated by RouterBench"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_env.py`:

```python
def test_trace_items_record_step_and_prompt():
    ep = episode("retry", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    s = ep.end_step(ep.play(ep.initial_state(), 0))  # fails, so item p0 comes back
    s = ep.play(s, 1)
    assert [(i.step_index, i.step_emb_key) for i in s.trace] == [(0, "p0"), (1, "p0")]
```

In `tests/test_routerbench.py`, in `test_model_cards_ordered_by_tier_with_capability`, add as the last line:

```python
    assert cards[0].price_date == "not stated by RouterBench"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py tests/test_routerbench.py`
Expected: 2 failures (`AttributeError: 'TraceItem' object has no attribute 'step_index'`, and the price_date assertion).

- [ ] **Step 3: Implement**

In `controller/contracts.py`, replace the last field of `TraceItem`:

```python
    output_emb_key: str | None  # cache key; None in the prototype
    step_index: int | None = None  # episode step this round belongs to (D11)
    step_emb_key: str | None = None  # prompt of that step, so a retried item is visible (D11)
```

In `data/env.py` `play`, replace the `TraceItem(...)` call:

```python
        item = TraceItem(model_id=self.model_ids[m], round=s.rounds + 1,
                         verifier_ok=self._verifier_ok(s, m, correct), cost_usd=cost, latency_ms=0.0,
                         tokens_out=0, escalated=s.rounds == 1, output_emb_key=None,
                         step_index=s.step_index, step_emb_key=self.skeleton.prompt_ids[s.slot])
```

In `data/routerbench.py`: `PRICE_DATE = "not stated by RouterBench"`.

Append `matplotlib>=3.8` to `requirements.txt`, then:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
cp /private/tmp/claude-501/-Users-veenathwickramaarachchi-Documents-SLIIT-final-research-neural-core/0328af69-bc87-44aa-9340-b29ba816a29e/scratchpad/refs/HRM-Text/LICENSE LICENSE-HRM-Text
.venv/bin/python -m data.routerbench
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest`
Expected: all pass. `data/processed/pool.json` now shows `"price_date": "not stated by RouterBench"`.

- [ ] **Step 5: Commit**

```bash
git add controller/contracts.py data/env.py data/routerbench.py requirements.txt LICENSE-HRM-Text tests/test_env.py tests/test_routerbench.py
git commit -m "feat(contracts): trace items carry step and prompt; honest price date"
```

---

### Task 2: Feature builder

**Files:**
- Create: `controller/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `Episode`, `EnvState` (data/env.py), `halt_label` (data/oracle.py), `K_TRACE`.
- Produces:
  - Constants: `CHOOSE = 0`, `HALT = 1`, `BUDGET_FEATS = 6`, `TRACE_FEATS = 7`.
  - Dataclass `View` with fields: `task_row`, `step_row`, `step_type`, `round`, `phase`, `budget`, `trace_row`, `trace_model`, `trace_feats`, `trace_present`, `trace_earlier`, `write`, `write_dt`.
  - `view(ep, s, phase, n_written, emb_index, model_index) -> View`.
  - Dataclass `Batch` (same field names, as tensors).
  - `collate(views, device="cpu") -> Batch`.
  - Dataclass `Target(succeeds: list[float], halt: float, band: float)`.
  - `target(ep, s, phase) -> Target`.
  - `collate_targets(targets, device="cpu") -> tuple[Tensor, Tensor, Tensor]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
from dataclasses import replace

import torch

from controller.features import BUDGET_FEATS, CHOOSE, HALT, TRACE_FEATS, collate, collate_targets, target, view
from tests.test_env import IDS, episode

EMB = {f"p{i}": i for i in range(10)}
MIDX = {m: i for i, m in enumerate(IDS)}
ROWS = [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]


def test_first_view_has_empty_trace_and_nothing_to_write():
    ep = episode("random", ROWS)
    v = view(ep, ep.initial_state(), CHOOSE, 0, EMB, MIDX)
    assert v.trace_present == [False, False] and not v.write and v.write_dt == 0.0
    assert (v.task_row, v.step_row, v.round, v.phase) == (0, 0, 1, CHOOSE)
    assert v.budget[:4] == [1.0, 0.0, 0.0, 1 / 3]


def test_post_round_view_writes_newest_round_of_current_step():
    ep = episode("random", ROWS)
    s = ep.play(ep.initial_state(), 0)
    v = view(ep, s, HALT, 0, EMB, MIDX)
    assert v.write and v.write_dt == 1 / 3
    assert v.trace_present == [True, False] and v.trace_earlier == [False, False]
    assert (v.trace_model[0], v.round, v.phase) == (0, 1, HALT)


def test_next_step_lists_earlier_rounds_newest_first():
    ep = episode("random", ROWS)
    s = ep.end_step(ep.play(ep.play(ep.initial_state(), 0), 2))
    v = view(ep, s, CHOOSE, 1, EMB, MIDX)
    assert v.trace_model == [2, 0] and v.trace_earlier == [True, True]
    assert v.write and v.step_row == 1 and v.trace_row == [0, 0]


def test_view_ignores_ground_truth():
    ep = episode("random", ROWS)
    s = ep.play(ep.initial_state(), 0)
    flipped = replace(s, cur_correct=not s.cur_correct, n_correct=7)
    assert view(ep, s, HALT, 0, EMB, MIDX) == view(ep, flipped, HALT, 0, EMB, MIDX)


def test_targets():
    ep = episode("random", [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], budget=5.0)
    s = ep.play(ep.initial_state(), 0)
    t = target(ep, s, HALT)
    assert (t.succeeds, t.halt, t.band) == ([0.0, 0.0, 1.0, 1.0], 1.0, 0.5)
    assert target(ep, ep.initial_state(), CHOOSE).halt == 0.0


def test_collate_shapes_and_types():
    ep = episode("random", ROWS)
    s = ep.play(ep.initial_state(), 0)
    b = collate([view(ep, ep.initial_state(), CHOOSE, 0, EMB, MIDX), view(ep, s, HALT, 0, EMB, MIDX)])
    assert b.trace_feats.shape == (2, 2, TRACE_FEATS) and b.budget.shape == (2, BUDGET_FEATS)
    assert b.write.dtype == torch.bool and b.task_row.dtype == torch.long and len(b) == 2
    succ, halt, band = collate_targets([target(ep, s, HALT)] * 3)
    assert succ.shape == (3, 4) and halt.shape == (3,) and band.shape == (3,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_features.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'controller.features'`

- [ ] **Step 3: Write `controller/features.py`**

```python
"""Controller inputs and training targets from simulator states (plan D2, D4, D10, D11).

Inputs use only what a deployed controller could see: the prompts, the trace with
its noisy verifier signal, and the budget. Ground truth (`cur_correct`, `n_correct`,
the outcome table's correctness) is read only by `target`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch

from controller.contracts import K_TRACE
from data.env import Episode, EnvState
from data.oracle import halt_label

CHOOSE, HALT = 0, 1  # pass before a round (pick a model) / after round 1 (escalate or stop)
BUDGET_FEATS = 6
TRACE_FEATS = 7


def _log_cost(usd: float) -> float:
    return math.log10(usd + 1e-9) / 4 + 1


@dataclass
class View:
    task_row: int
    step_row: int
    step_type: int
    round: int
    phase: int
    budget: list[float]
    trace_row: list[int]
    trace_model: list[int]
    trace_feats: list[list[float]]
    trace_present: list[bool]
    trace_earlier: list[bool]  # from an earlier step: masked in context_free
    write: bool  # a round outcome not yet stored in memory
    write_dt: float


def view(ep: Episode, s: EnvState, phase: int, n_written: int, emb_index: dict[str, int],
         model_index: dict[str, int]) -> View:
    sk, n = ep.skeleton, ep.n_steps
    total = sk.budget_usd
    budget = [s.budget_left / total, s.spent / total, s.step_index / n, 1.0 / n, _log_cost(s.budget_left), 0.0]
    rows, models, feats, present, earlier = [], [], [], [], []
    for item in reversed(s.trace[-K_TRACE:]):  # newest first
        same_step = item.step_index == s.step_index
        rows.append(emb_index[item.step_emb_key])
        models.append(model_index[item.model_id])
        feats.append([1.0 if item.verifier_ok else -1.0, float(item.escalated), float(item.round - 1),
                      _log_cost(item.cost_usd), float(same_step), (s.step_index - item.step_index) / n, 1.0])
        present.append(True)
        earlier.append(not same_step)
    while len(rows) < K_TRACE:
        rows.append(0)
        models.append(0)
        feats.append([0.0] * TRACE_FEATS)
        present.append(False)
        earlier.append(False)
    write = len(s.trace) > n_written
    return View(task_row=emb_index[sk.prompt_ids[0]], step_row=emb_index[sk.prompt_ids[s.slot]],
                step_type=int(ep.table.step_type[ep.row(s)]),
                round=s.rounds + 1 if phase == CHOOSE else s.rounds, phase=phase, budget=budget,
                trace_row=rows, trace_model=models, trace_feats=feats, trace_present=present,
                trace_earlier=earlier, write=write, write_dt=1.0 / n if write else 0.0)


@dataclass
class Batch:
    task_row: torch.Tensor
    step_row: torch.Tensor
    step_type: torch.Tensor
    round: torch.Tensor
    phase: torch.Tensor
    budget: torch.Tensor
    trace_row: torch.Tensor
    trace_model: torch.Tensor
    trace_feats: torch.Tensor
    trace_present: torch.Tensor
    trace_earlier: torch.Tensor
    write: torch.Tensor
    write_dt: torch.Tensor

    def __len__(self) -> int:
        return self.task_row.shape[0]


_DTYPES = {"budget": torch.float32, "trace_feats": torch.float32, "write_dt": torch.float32,
           "trace_present": torch.bool, "trace_earlier": torch.bool, "write": torch.bool}


def collate(views: Sequence[View], device: str | torch.device = "cpu") -> Batch:
    return Batch(**{f.name: torch.tensor([getattr(v, f.name) for v in views],
                                         dtype=_DTYPES.get(f.name, torch.long), device=device)
                    for f in fields(Batch)})


@dataclass
class Target:
    succeeds: list[float]
    halt: float
    band: float


def target(ep: Episode, s: EnvState, phase: int) -> Target:
    ok = ep.succeeds(s)
    return Target(succeeds=[float(x) for x in ok], halt=float(halt_label(ep, s)) if phase == HALT else 0.0,
                  band=float((~ok).sum()) / len(ok))


def collate_targets(targets: Sequence[Target], device: str | torch.device = "cpu"):
    return (torch.tensor([t.succeeds for t in targets], dtype=torch.float32, device=device),
            torch.tensor([t.halt for t in targets], dtype=torch.float32, device=device),
            torch.tensor([t.band for t in targets], dtype=torch.float32, device=device))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_features.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add controller/features.py tests/test_features.py
git commit -m "feat(controller): feature builder from simulator states without ground truth"
```

---

### Task 3: H/L core

**Files:**
- Create: `controller/core.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces:
  - `rms_norm(x) -> Tensor` (weightless).
  - `SwiGLU(d, expansion=4.0)`.
  - `ReasoningModule(d, n_blocks=2)`.
  - `bp_schedule(step, total_steps, bp_min=2, bp_max=5, warmup_ratio=0.2) -> int`.
  - `bp_split(bp_steps, h_cycles) -> tuple[int, int]`.
  - `HLCore(d=128, h_cycles=2, l_cycles=3)`, with `forward(h, x, bp_steps=5, record=False) -> tuple[Tensor, list[tuple[str, float]]]` and buffer `zL_init`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_core.py
import torch

from controller.core import HLCore, ReasoningModule, bp_schedule, bp_split


def test_bp_schedule_is_hrm_text_staircase():
    assert [bp_schedule(s, 1000) for s in (0, 66, 67, 134, 199, 200, 999)] == [2, 2, 3, 4, 4, 5, 5]
    assert bp_schedule(0, 1000, bp_min=3) == 3 and bp_schedule(999, 1000, bp_min=3) == 5


def test_bp_split_matches_hrm_text():
    assert [bp_split(b, 2) for b in (2, 3, 4, 5)] == [(1, 1), (2, 1), (2, 2), (2, 3)]


def test_module_output_is_rms_normalised():
    y = ReasoningModule(16)(torch.randn(4, 16) * 10)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


def test_core_shapes_record_and_fixed_init():
    core = HLCore(d=32)
    z, rec = core(torch.randn(5, 32), torch.randn(5, 32), record=True)
    assert z.shape == (5, 32)
    assert [k for k, _ in rec] == ["L", "L", "L", "H", "L", "L", "L", "H"]
    assert "zL_init" not in dict(core.named_parameters()) and "zL_init" in core.state_dict()


def test_gradient_reaches_both_inputs_at_shortest_horizon():
    core = HLCore(d=32)
    h = torch.randn(3, 32, requires_grad=True)
    x = torch.randn(3, 32, requires_grad=True)
    z, _ = core(h, x, bp_steps=2)
    z.sum().backward()
    assert h.grad.abs().sum() > 0 and x.grad.abs().sum() > 0


def test_horizon_controls_which_l_calls_build_a_graph():
    core = HLCore(d=32)
    calls = []
    forward = core.L.forward

    def spy(inp):
        calls.append(torch.is_grad_enabled())
        return forward(inp)

    core.L.forward = spy
    core(torch.randn(2, 32), torch.randn(2, 32), bp_steps=4)
    assert calls == [False, False, False, False, True, True]
    calls.clear()
    with torch.no_grad():
        core(torch.randn(2, 32), torch.randn(2, 32), bp_steps=5)
    assert calls == [False] * 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_core.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'controller.core'`

- [ ] **Step 3: Write `controller/core.py`**

```python
"""H/L reasoning core (plan §5, D11).

Adapted from sapientinc/HRM-Text, models/baselines/hrm_nocarry_bp_warmup.py and
models/transformer.py (Apache-2.0; see LICENSE-HRM-Text).
Kept: 2 H-cycles x 3 L-steps; MagicNorm (pre-norm blocks capped by a weightless
RMSNorm); a fixed, non-trainable z_L init; the staircase gradient-horizon warmup
and its H/L split.
Changed: SwiGLU MLP blocks on one vector per module instead of attention over a
token sequence; input x injected at every L-step (as in HRM and TRM); the H input h
re-injected at every H-step so the task/budget encoders get gradient under
truncated backprop; z_H initialised from pooled H tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class SwiGLU(nn.Module):
    def __init__(self, d: int, expansion: float = 4.0):
        super().__init__()
        hidden = -(-round(expansion * d * 2 / 3) // 16) * 16
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class ReasoningModule(nn.Module):
    """Pre-norm residual SwiGLU blocks capped by a weightless RMSNorm (MagicNorm)."""

    def __init__(self, d: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(SwiGLU(d) for _ in range(n_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(rms_norm(x))
        return rms_norm(x)


def bp_schedule(step: int, total_steps: int, bp_min: int = 2, bp_max: int = 5,
                warmup_ratio: float = 0.2) -> int:
    """Backprop horizon in module calls, widening in steps over the warmup (HRM-Text)."""
    progress = min(1.0, step / max(1.0, warmup_ratio * total_steps))
    return bp_min + int(progress * (bp_max - bp_min))


def bp_split(bp_steps: int, h_cycles: int) -> tuple[int, int]:
    h_bp = min(h_cycles, bp_steps - 1)
    return h_bp, bp_steps - h_bp


class HLCore(nn.Module):
    def __init__(self, d: int = 128, h_cycles: int = 2, l_cycles: int = 3):
        super().__init__()
        self.h_cycles, self.l_cycles = h_cycles, l_cycles
        self.H = ReasoningModule(d)
        self.L = ReasoningModule(d)
        self.register_buffer("zL_init", nn.init.trunc_normal_(torch.empty(d), std=1.0, a=-2.0, b=2.0))

    def forward(self, h: torch.Tensor, x: torch.Tensor, bp_steps: int = 5,
                record: bool = False) -> tuple[torch.Tensor, list[tuple[str, float]]]:
        grad = torch.is_grad_enabled()
        h_bp, l_bp = bp_split(bp_steps, self.h_cycles)
        n_l = self.h_cycles * self.l_cycles
        z_h, z_l = h, self.zL_init.expand_as(x)
        residuals: list[tuple[str, float]] = []
        i = 0
        for cycle in range(self.h_cycles):
            for _ in range(self.l_cycles):
                i += 1
                with torch.set_grad_enabled(grad and i > n_l - l_bp):
                    new = self.L(z_l + z_h + x)
                if record:
                    residuals.append(("L", (new - z_l).norm(dim=-1).mean().item()))
                z_l = new
            with torch.set_grad_enabled(grad and cycle >= self.h_cycles - h_bp):
                new = self.H(z_h + z_l + h)
            if record:
                residuals.append(("H", (new - z_h).norm(dim=-1).mean().item()))
            z_h = new
        return z_h, residuals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_core.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add controller/core.py tests/test_core.py
git commit -m "feat(controller): H/L core with MagicNorm blocks and staircase gradient horizon"
```

---

### Task 4: Episode memory and gates

**Files:**
- Create: `controller/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `rms_norm` from `controller.core`.
- Produces:
  - Dataclass `MemoryState(W: Tensor[B,d,d], c: Tensor[B,units])` with `select(idx) -> MemoryState` and `merge(idx, part) -> MemoryState`.
  - Gates, each with `units: int` and `forward(u, c, dt) -> (rate[B], c)`:
    - `CfCGate(in_dim, units=16, backbone=32)`
    - `StatelessGate(in_dim, hidden=32)`
    - `ConstGate()`
  - `EpisodeMemory(d=128, gate="cfc", gate_units=16)`, with:
    - `init_state(batch) -> MemoryState`
    - `write(state, obs, mask, dt) -> (MemoryState, {"surprise", "rate", "alpha"})`
    - `read(state, z) -> Tensor[B,d]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
import pytest
import torch
import torch.nn.functional as F

from controller.memory import CfCGate, ConstGate, EpisodeMemory, MemoryState, StatelessGate

ON = lambda n: torch.ones(n, dtype=torch.bool)  # noqa: E731


def test_one_write_moves_memory_a_fraction_rate_toward_value():
    torch.manual_seed(0)
    mem = EpisodeMemory(d=16, gate="const")
    state, obs = mem.init_state(3), torch.randn(3, 16)
    new, stats = mem.write(state, obs, ON(3), torch.full((3,), 0.25))
    k, v = F.normalize(mem.key(obs), dim=-1), mem.value(obs)
    s_after = v - torch.bmm(new.W, k[..., None]).squeeze(-1)
    assert torch.allclose(s_after, (1 - stats["rate"])[:, None] * v, atol=1e-5)  # W starts at W0 = 0


def test_masked_rows_are_untouched():
    mem = EpisodeMemory(d=16, gate="cfc")
    state = mem.init_state(2)
    new, _ = mem.write(state, torch.randn(2, 16), torch.tensor([True, False]), torch.full((2,), 0.25))
    assert not torch.equal(new.W[0], state.W[0]) and torch.equal(new.W[1], state.W[1])
    assert not torch.equal(new.c[0], state.c[0]) and torch.equal(new.c[1], state.c[1])


def test_forgetting_decays_toward_learned_prior():
    mem = EpisodeMemory(d=8, gate="const")
    with torch.no_grad():
        mem.W0.fill_(0.5)
        mem.forget.bias.fill_(20.0)  # alpha ~ 1
        mem.gate.c.fill_(-30.0)  # rate ~ 0
    new, _ = mem.write(MemoryState(torch.zeros(1, 8, 8), torch.zeros(1, 0)), torch.randn(1, 8), ON(1), torch.ones(1))
    assert torch.allclose(new.W, torch.full((1, 8, 8), 0.5), atol=1e-3)


@pytest.mark.parametrize("gate", ["cfc", "stateless", "const"])
def test_rates_in_unit_interval_and_forgetting_starts_near_zero(gate):
    mem = EpisodeMemory(d=16, gate=gate)
    _, stats = mem.write(mem.init_state(4), torch.randn(4, 16), ON(4), torch.full((4,), 0.2))
    assert ((stats["rate"] > 0) & (stats["rate"] < 1)).all() and (stats["alpha"] < 0.05).all()


def test_twelve_writes_stay_finite_and_bounded():
    torch.manual_seed(1)
    mem = EpisodeMemory(d=32, gate="cfc")
    state = mem.init_state(4)
    for _ in range(12):
        state, _ = mem.write(state, torch.randn(4, 32) * 3, ON(4), torch.full((4,), 1 / 6))
    assert torch.isfinite(state.W).all() and state.W.norm(dim=(1, 2)).max() < 100


def test_cfc_gate_is_stateful_and_time_dependent():
    torch.manual_seed(0)
    gate = CfCGate(in_dim=6, units=4)
    u, c0 = torch.randn(2, 6), torch.zeros(2, 4)
    r1, c1 = gate(u, c0, torch.full((2,), 0.2))
    r2, _ = gate(u, c1, torch.full((2,), 0.2))
    r3, _ = gate(u, c0, torch.full((2,), 1.0))
    assert c1.shape == (2, 4) and not torch.allclose(r1, r2) and not torch.allclose(r1, r3)


@pytest.mark.parametrize("gate", [StatelessGate(6), ConstGate()])
def test_other_gates_carry_no_state(gate):
    rate, c = gate(torch.randn(3, 6), torch.zeros(3, 0), torch.ones(3))
    assert rate.shape == (3,) and c.shape == (3, 0)


def test_read_is_zero_when_empty_and_normalised_after_a_write():
    mem = EpisodeMemory(d=16)
    state = mem.init_state(2)
    assert torch.allclose(mem.read(state, torch.randn(2, 16)), torch.zeros(2, 16))
    new, _ = mem.write(state, torch.randn(2, 16), ON(2), torch.full((2,), 0.2))
    assert torch.allclose(mem.read(new, torch.randn(2, 16)).pow(2).mean(-1), torch.ones(2), atol=1e-3)


def test_select_and_merge_rows():
    mem = EpisodeMemory(d=8)
    state = mem.init_state(3)
    part, _ = mem.write(state.select(torch.tensor([2])), torch.randn(1, 8), ON(1), torch.ones(1))
    merged = state.merge(torch.tensor([2]), part)
    assert torch.equal(merged.W[2], part.W[0]) and torch.equal(merged.W[0], state.W[0])


def test_unknown_gate_is_rejected():
    with pytest.raises(ValueError, match="gate"):
        EpisodeMemory(d=8, gate="lstm")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_memory.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'controller.memory'`

- [ ] **Step 3: Write `controller/memory.py`**

```python
"""Episode memory: a fast weight W written once per round outcome (plan D2, D11).

Re-implemented from the equations, no code copied:
- inner model and associative loss: TTT-Linear (Sun et al. 2024, arXiv 2407.04620;
  test-time-training/ttt-lm-pytorch, MIT), one explicit delta-rule step;
- forgetting toward a learned prior: Titans (Behrouz et al. 2025, arXiv 2501.00663;
  lucidrains/titans-pytorch, MIT), without momentum;
- CfC gate: closed-form continuous-time cell, Eq. 10 (Hasani et al. 2022,
  arXiv 2106.13898), following mlech26l/ncps torch/cfc_cell.py default mode (Apache-2.0).
Gating the write on the surprise s (not on the input) is this project's addition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from controller.core import rms_norm


@dataclass
class MemoryState:
    W: torch.Tensor  # [B, d, d]
    c: torch.Tensor  # [B, gate units]; width 0 for gates without state

    def select(self, idx: torch.Tensor) -> MemoryState:
        return MemoryState(self.W[idx], self.c[idx])

    def merge(self, idx: torch.Tensor, part: MemoryState) -> MemoryState:
        return MemoryState(self.W.index_copy(0, idx, part.W), self.c.index_copy(0, idx, part.c))


def lecun_tanh(x: torch.Tensor) -> torch.Tensor:
    return 1.7159 * torch.tanh(0.666 * x)


class CfCGate(nn.Module):
    """Write rate from a CfC cell whose state persists across an episode's writes."""

    def __init__(self, in_dim: int, units: int = 16, backbone: int = 32):
        super().__init__()
        self.units = units
        self.backbone = nn.Linear(in_dim + units, backbone)
        self.ff1 = nn.Linear(backbone, units)
        self.ff2 = nn.Linear(backbone, units)
        self.time_a = nn.Linear(backbone, units)
        self.time_b = nn.Linear(backbone, units)
        self.readout = nn.Linear(units, 1)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)
            elif name == "readout.bias":
                nn.init.zeros_(p)

    def forward(self, u: torch.Tensor, c: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = lecun_tanh(self.backbone(torch.cat([u, c], dim=-1)))
        f1, f2 = torch.tanh(self.ff1(b)), torch.tanh(self.ff2(b))
        t = torch.sigmoid(self.time_a(b) * dt[:, None] + self.time_b(b))
        c = f1 * (1 - t) + t * f2
        return torch.sigmoid(self.readout(c)).squeeze(-1), c


class StatelessGate(nn.Module):
    """Ablation: same input as the CfC gate, no state and no time."""

    units = 0

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, u, c, dt):
        return torch.sigmoid(self.net(u)).squeeze(-1), c


class ConstGate(nn.Module):
    """Ablation gate_off: one learned rate."""

    units = 0

    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, u, c, dt):
        return torch.sigmoid(self.c).expand(u.shape[0]), c


class EpisodeMemory(nn.Module):
    def __init__(self, d: int = 128, gate: str = "cfc", gate_units: int = 16):
        super().__init__()
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(d, d, bias=False)
        self.W0 = nn.Parameter(torch.zeros(d, d))
        gate_in = 2 * d + 1  # [s/|s|, log|s|, obs]
        if gate == "cfc":
            self.gate = CfCGate(gate_in, units=gate_units)
        elif gate == "stateless":
            self.gate = StatelessGate(gate_in)
        elif gate == "const":
            self.gate = ConstGate()
        else:
            raise ValueError(f"unknown gate {gate!r}")
        self.forget = nn.Linear(d + 1, 1)
        nn.init.zeros_(self.forget.weight)
        nn.init.constant_(self.forget.bias, -4.0)

    def init_state(self, batch: int) -> MemoryState:
        return MemoryState(self.W0.unsqueeze(0).expand(batch, -1, -1),
                           torch.zeros(batch, self.gate.units, device=self.W0.device))

    def write(self, state: MemoryState, obs: torch.Tensor, mask: torch.Tensor,
              dt: torch.Tensor) -> tuple[MemoryState, dict[str, torch.Tensor]]:
        k = F.normalize(self.key(obs), dim=-1)
        v = self.value(obs)
        s = v - torch.bmm(state.W, k.unsqueeze(-1)).squeeze(-1)
        s_norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        log_s = s_norm.log()
        rate, c_new = self.gate(torch.cat([s / s_norm, log_s, obs], dim=-1), state.c, dt)
        alpha = torch.sigmoid(self.forget(torch.cat([log_s, obs], dim=-1))).squeeze(-1)
        W0 = self.W0.unsqueeze(0)
        W_new = W0 + (1 - alpha)[:, None, None] * (state.W - W0) + rate[:, None, None] * s.unsqueeze(-1) * k.unsqueeze(1)
        W = torch.where(mask[:, None, None], W_new, state.W)
        c = torch.where(mask[:, None], c_new, state.c)
        return MemoryState(W, c), {"surprise": s_norm.squeeze(-1), "rate": rate, "alpha": alpha}

    def read(self, state: MemoryState, z: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query(z), dim=-1)
        return rms_norm(torch.bmm(state.W, q.unsqueeze(-1)).squeeze(-1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory.py`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add controller/memory.py tests/test_memory.py
git commit -m "feat(controller): delta-rule episode memory with forgetting and CfC, stateless and constant gates"
```

---

### Task 5: Encoder, heads and controller model

**Files:**
- Create: `controller/encoder.py`, `controller/heads.py`, `controller/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes:
  - `Batch`, `BUDGET_FEATS`, `TRACE_FEATS` (features);
  - `HLCore`, `rms_norm` (core);
  - `EpisodeMemory`, `MemoryState` (memory);
  - `STATE_DIM` (contracts).
- Produces:
  - `StepEncoder(d, emb_dim, n_types, card_dim)`, whose forward returns `Encoded(task, budget, step, trace, card)`.
  - `Heads(d)`, whose forward returns `(model_logits[B,M], halt_logit[B], complexity_logit[B])`.
  - `ControllerConfig` (frozen dataclass, fields `d`, `h_cycles`, `l_cycles`, `memory`, `gate`, `gate_units`, `context_free`).
  - `CONFIGS: dict[str, ControllerConfig]`.
  - `Output` dataclass with fields `model_logits`, `halt_logit`, `complexity_logit`, `z_h`, `memory`, `stats`, `residuals`.
  - `Controller(cfg, embeddings, cards, n_types)`, with:
    - `.init_memory(batch) -> MemoryState | None`
    - `forward(batch, memory=None, bp_steps=5, record=False) -> Output`
  - `n_parameters(model) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py
from dataclasses import replace

import pytest
import torch

from controller.features import CHOOSE, HALT, collate, view
from controller.model import CONFIGS, Controller, n_parameters
from tests.test_env import IDS, episode

EMB = {f"p{i}": i for i in range(10)}
MIDX = {m: i for i, m in enumerate(IDS)}


def make_model(name="full", n_cards=4, seed=0):
    torch.manual_seed(seed)
    return Controller(CONFIGS[name], torch.randn(10, 384), torch.rand(n_cards, 6), n_types=2)


def sample_batch():
    """Rows: first pass of an episode; after round 1; first pass of step 2 (after an escalation)."""
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]])
    s0 = ep.initial_state()
    s1 = ep.play(s0, 0)
    s2 = ep.end_step(ep.play(s1, 2))
    return collate([view(ep, s0, CHOOSE, 0, EMB, MIDX), view(ep, s1, HALT, 0, EMB, MIDX),
                    view(ep, s2, CHOOSE, 1, EMB, MIDX)])


def perturb_context(b):
    return replace(b, task_row=torch.full_like(b.task_row, 7), budget=torch.rand_like(b.budget),
                   trace_row=torch.where(b.trace_earlier, torch.full_like(b.trace_row, 9), b.trace_row),
                   trace_feats=torch.where(b.trace_earlier[..., None], torch.rand_like(b.trace_feats), b.trace_feats))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_forward_shapes(name):
    m = make_model(name)
    out = m(sample_batch(), m.init_memory(3))
    assert out.model_logits.shape == (3, 4) and out.halt_logit.shape == (3,)
    assert out.complexity_logit.shape == (3,) and out.z_h.shape == (3, 128)


def test_pool_is_a_list_of_cards():
    m = make_model(n_cards=5)
    assert m(sample_batch(), m.init_memory(3)).model_logits.shape == (3, 5)


def test_parameter_count():
    n = n_parameters(make_model("full"))
    print(f"\nfull controller: {n:,} trainable parameters")
    assert 200_000 < n < 1_500_000


def test_memory_written_only_where_a_new_outcome_exists():
    m = make_model("full")
    empty = m.init_memory(3)
    out = m(sample_batch(), empty)
    assert torch.equal(out.memory.W[0], empty.W[0])
    assert not torch.equal(out.memory.W[1], empty.W[1]) and not torch.equal(out.memory.W[2], empty.W[2])
    assert out.stats["rate"].shape == (3,)
    assert make_model("memory_off").init_memory(3) is None and make_model("context_free").init_memory(3) is None


def test_context_free_ignores_task_budget_and_earlier_steps():
    m = make_model("context_free").eval()
    b = sample_batch()
    with torch.no_grad():
        assert torch.allclose(m(b).model_logits, m(perturb_context(b)).model_logits, atol=1e-6)


def test_context_free_still_sees_current_steps_round_one():
    m = make_model("context_free").eval()
    b = sample_batch()
    flipped = replace(b, trace_feats=b.trace_feats.clone())
    flipped.trace_feats[1, 0, 0] = -flipped.trace_feats[1, 0, 0]  # row 1: round-1 verifier signal
    with torch.no_grad():
        assert not torch.allclose(m(b).halt_logit[1], m(flipped).halt_logit[1])


def test_full_model_reads_context():
    m = make_model("full").eval()
    b = sample_batch()
    with torch.no_grad():
        assert not torch.allclose(m(b, m.init_memory(3)).model_logits,
                                  m(perturb_context(b), m.init_memory(3)).model_logits)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'controller.model'`

- [ ] **Step 3: Write `controller/encoder.py`**

```python
"""StepEncoder: Batch -> H tokens (task, budget/state) and L tokens (step, trace -1, -2) (plan §3, D11)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from controller.features import BUDGET_FEATS, TRACE_FEATS, Batch


@dataclass
class Encoded:
    task: torch.Tensor  # [B, d]
    budget: torch.Tensor  # [B, d]
    step: torch.Tensor  # [B, d]
    trace: torch.Tensor  # [B, K, d], zero where absent
    card: torch.Tensor  # [M, d]


def _mlp(n_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, d), nn.SiLU(), nn.Linear(d, d))


class StepEncoder(nn.Module):
    def __init__(self, d: int, emb_dim: int, n_types: int, card_dim: int):
        super().__init__()
        self.text = nn.Linear(emb_dim, d)  # projects frozen bge-small vectors
        self.step_type = nn.Embedding(n_types, d)
        self.round = nn.Embedding(3, d)
        self.phase = nn.Embedding(2, d)
        self.budget = _mlp(BUDGET_FEATS, d)
        self.trace = _mlp(TRACE_FEATS, d)
        self.card = _mlp(card_dim, d)

    def forward(self, b: Batch, emb: torch.Tensor, cards: torch.Tensor) -> Encoded:
        card = self.card(cards)
        step = (self.text(emb[b.step_row]) + self.step_type(b.step_type) + self.round(b.round)
                + self.phase(b.phase))
        trace = self.text(emb[b.trace_row]) + self.trace(b.trace_feats) + card[b.trace_model]
        return Encoded(task=self.text(emb[b.task_row]), budget=self.budget(b.budget), step=step,
                       trace=trace * b.trace_present[..., None], card=card)
```

- [ ] **Step 4: Write `controller/heads.py`**

```python
"""Heads on the final z_H (plan D7, D11). The model head matches a requirement vector
against each model card, so the pool is a list, not fixed output slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Heads(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.need = nn.Linear(d, d)
        self.match = nn.Linear(d, 1)
        self.halt = nn.Linear(d, 1)
        self.complexity = nn.Linear(d, 1)

    def forward(self, z_h: torch.Tensor, card: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.match(F.silu(self.need(z_h)[:, None, :] + card[None])).squeeze(-1)
        return model, self.halt(z_h).squeeze(-1), self.complexity(z_h).squeeze(-1)
```

- [ ] **Step 5: Write `controller/model.py`**

```python
"""Controller = encoder -> episode memory -> H/L core -> heads (plan §5, D2, D10, D11)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from controller.contracts import STATE_DIM
from controller.core import HLCore, rms_norm
from controller.encoder import StepEncoder
from controller.features import Batch
from controller.heads import Heads
from controller.memory import EpisodeMemory, MemoryState


@dataclass(frozen=True)
class ControllerConfig:
    d: int = STATE_DIM
    h_cycles: int = 2
    l_cycles: int = 3
    memory: bool = True
    gate: str = "cfc"  # cfc | stateless | const
    gate_units: int = 16
    context_free: bool = False


CONFIGS = {
    "full": ControllerConfig(),
    "gate_off": ControllerConfig(gate="const"),
    "gate_stateless": ControllerConfig(gate="stateless"),
    "memory_off": ControllerConfig(memory=False),
    # Masking the memory readout is the same as having no memory, so none is built.
    "context_free": ControllerConfig(memory=False, context_free=True),
}


@dataclass
class Output:
    model_logits: torch.Tensor  # [B, M]
    halt_logit: torch.Tensor  # [B]
    complexity_logit: torch.Tensor  # [B]
    z_h: torch.Tensor  # [B, d], the DecisionRecord decision_state
    memory: MemoryState | None
    stats: dict[str, torch.Tensor] = field(default_factory=dict)
    residuals: list[tuple[str, float]] = field(default_factory=list)


class Controller(nn.Module):
    def __init__(self, cfg: ControllerConfig, embeddings: torch.Tensor, cards: torch.Tensor, n_types: int):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("emb", embeddings.float(), persistent=False)  # frozen bge-small vectors
        self.register_buffer("cards", cards.float(), persistent=False)  # capability vectors of the pool
        self.encoder = StepEncoder(cfg.d, embeddings.shape[1], n_types, cards.shape[1])
        self.memory = EpisodeMemory(cfg.d, cfg.gate, cfg.gate_units) if cfg.memory else None
        self.core = HLCore(cfg.d, cfg.h_cycles, cfg.l_cycles)
        self.heads = Heads(cfg.d)

    def init_memory(self, batch: int) -> MemoryState | None:
        return self.memory.init_state(batch) if self.memory is not None else None

    def forward(self, b: Batch, memory: MemoryState | None = None, bp_steps: int = 5,
                record: bool = False) -> Output:
        e = self.encoder(b, self.emb, self.cards)
        task, budget, trace, present = e.task, e.budget, e.trace, b.trace_present
        if self.cfg.context_free:  # D10/D11: keep only the current step's own rounds
            present = present & ~b.trace_earlier
            trace = trace * present[..., None]
            task, budget = torch.zeros_like(task), torch.zeros_like(budget)
        h = rms_norm((task + budget) / 2)
        x = (e.step + trace.sum(1)) / (1 + present.sum(1, keepdim=True).float())
        stats: dict[str, torch.Tensor] = {}
        if self.memory is not None:
            memory, stats = self.memory.write(memory, e.trace[:, 0], b.write, b.write_dt)
            x = x + self.memory.read(memory, h)
        z_h, residuals = self.core(h, x, bp_steps, record)
        model_logits, halt_logit, complexity_logit = self.heads(z_h, e.card)
        return Output(model_logits, halt_logit, complexity_logit, z_h, memory, stats, residuals)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model.py -s`
Expected: 11 passed; prints the measured parameter count.

- [ ] **Step 7: Commit**

```bash
git add controller/encoder.py controller/heads.py controller/model.py tests/test_model.py
git commit -m "feat(controller): encoder, heads and controller model with five configs"
```

---

### Task 6: Batched rollout engine

**Files:**
- Create: `train/rollout.py`
- Test: `tests/test_rollout.py`

**Interfaces:**
- Consumes:
  - `Controller`, `Output`;
  - `view`, `collate`, `target`, `collate_targets`, `CHOOSE`, `HALT`;
  - `Episode`, `step_labels`, `halt_label`, `EpisodeResult`, `MAX_ROUNDS`.
- Produces:
  - Dataclass `World(table, model_ids, card_costs: np.ndarray, emb_index, model_index)`.
  - `tau_choice(p, candidates, tau) -> int`.
  - Dataclass `RolloutOutput(loss, parts, results, log, residuals)`.
  - `rollout(model, episodes, world, *, policy="greedy", tau=0.5, model_share=True, rng=None, bp_steps=5, compute_loss=False, record=False) -> RolloutOutput`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rollout.py
import numpy as np
import pytest
import torch

from controller.model import CONFIGS, Controller
from data.env import Episode
from data.episodes import Skeleton, generate_skeletons
from tests.conftest import make_table
from tests.test_env import IDS
from train.rollout import World, rollout, tau_choice


def world_and_episodes(n=12):
    table = make_table(n=120, seed=7)
    world = World(table, IDS, np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4,
                  {p: i for i, p in enumerate(table.prompt_ids)}, {m: i for i, m in enumerate(IDS)})
    sks = generate_skeletons(table, list(table.prompt_ids), n=n, seed=0, split="t")
    return world, [Episode(sk, table, IDS) for sk in sks]


def tiny_model(name="full"):
    torch.manual_seed(0)
    return Controller(CONFIGS[name], torch.randn(120, 384), torch.rand(4, 6), n_types=2)


def test_tau_choice():
    p = np.array([0.2, 0.6, 0.9, 0.95])
    assert tau_choice(p, [0, 1, 2, 3], 0.5) == 1
    assert tau_choice(p, [0, 1, 2, 3], 0.99) == 3
    assert tau_choice(p, [2, 3], 0.5) == 2


@pytest.mark.parametrize("name", ["full", "gate_stateless", "memory_off", "context_free"])
def test_greedy_rollout_finishes_every_episode(name):
    world, eps = world_and_episodes()
    with torch.no_grad():
        out = rollout(tiny_model(name).eval(), eps, world)
    assert [r.episode_id for r in out.results] == [ep.skeleton.episode_id for ep in eps]
    assert all(0 <= r.quality <= 1 and r.cost_usd >= 0 for r in out.results) and out.loss is None


def test_mixture_loss_trains_the_gate_through_the_episode():
    world, eps = world_and_episodes()
    m = tiny_model("full")
    out = rollout(m, eps, world, policy="mixture", rng=np.random.default_rng(0), compute_loss=True, bp_steps=3)
    assert torch.isfinite(out.loss) and set(out.parts) == {"model", "halt", "complexity"}
    out.loss.backward()
    assert m.memory.gate.readout.weight.grad.abs().sum() > 0


def test_episode_result_does_not_depend_on_batch_mates():
    world, eps = world_and_episodes()
    m = tiny_model("full").eval()
    with torch.no_grad():
        together = rollout(m, eps, world).results
        alone = [rollout(m, [ep], world).results[0] for ep in eps]
    assert [(r.quality, round(r.cost_usd, 12)) for r in together] == [(r.quality, round(r.cost_usd, 12)) for r in alone]


def test_no_affordable_stronger_model_means_stop():
    world, _ = world_and_episodes()
    table = world.table
    sk = Skeleton("tight", "random", tuple(table.prompt_ids[:3]), float(table.cost[:3, 0].sum()) * 1.01)
    m = tiny_model("full").eval()
    with torch.no_grad():
        nn_bias = m.heads.halt.bias
        nn_bias.fill_(50.0)  # always wants to escalate
        out = rollout(m, [Episode(sk, table, IDS)], world, tau=0.0)  # tau 0: cheapest model each step
    assert out.results[0].cost_usd <= sk.budget_usd


def test_record_logs_memory_writes_and_residuals():
    world, eps = world_and_episodes(4)
    with torch.no_grad():
        out = rollout(tiny_model("full").eval(), eps, world, record=True)
    assert out.log and {"episode_id", "step_index", "rate", "surprise", "alpha", "verifier_ok"} <= set(out.log[0])
    assert out.residuals and len(out.residuals[0]) == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rollout.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'train.rollout'`

- [ ] **Step 3: Write `train/rollout.py`**

```python
"""Batched rollout of the controller through the simulator (plan D1, D5, D10, D11).

Training and evaluation share this loop. Episodes advance in lockstep: each active
episode makes one forward pass per iteration, either before a round (CHOOSE a model)
or after round 1 (HALT: escalate or stop). Memory rows follow their episode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from controller.contracts import MAX_ROUNDS
from controller.features import CHOOSE, HALT, collate, collate_targets, target, view
from data.env import Episode
from data.oracle import halt_label, step_labels
from data.routerbench import OutcomeTable
from eval.policies import EpisodeResult

SOURCES = ("oracle", "random", "model")
SHARES = (0.5, 0.25, 0.25)  # behaviour mixture per step (D1, D11)
RANDOM_ESCALATE_P = 0.3


@dataclass
class World:
    table: OutcomeTable
    model_ids: list[str]
    card_costs: np.ndarray  # mean cost per call from the model cards
    emb_index: dict[str, int]
    model_index: dict[str, int]


def tau_choice(p: np.ndarray, candidates: Sequence[int], tau: float) -> int:
    """Cheapest candidate with p >= tau, else the most likely to succeed (plan §5)."""
    above = [m for m in candidates if p[m] >= tau]
    return min(above) if above else max(candidates, key=lambda m: p[m])


@dataclass
class RolloutOutput:
    loss: torch.Tensor | None
    parts: dict[str, float]
    results: list[EpisodeResult]
    log: list[dict] = field(default_factory=list)
    residuals: list[list[tuple[str, float]]] = field(default_factory=list)


def rollout(model, episodes: Sequence[Episode], world: World, *, policy: str = "greedy", tau: float = 0.5,
            model_share: bool = True, rng: np.random.Generator | None = None, bp_steps: int = 5,
            compute_loss: bool = False, record: bool = False) -> RolloutOutput:
    rng = rng or np.random.default_rng(0)
    device = next(model.parameters()).device
    n = len(episodes)
    states = [ep.initial_state() for ep in episodes]
    written = [0] * n
    source = ["model"] * n
    memory = model.init_memory(n)
    terms: dict[str, list[torch.Tensor]] = {"model": [], "halt": [], "complexity": []}
    log: list[dict] = []
    residuals: list[list[tuple[str, float]]] = []
    while True:
        for i, ep in enumerate(episodes):
            if not states[i].done and states[i].rounds == MAX_ROUNDS:
                states[i] = ep.end_step(states[i])
        active = [i for i in range(n) if not states[i].done]
        if not active:
            break
        phases = [CHOOSE if states[i].rounds == 0 else HALT for i in active]
        views = [view(episodes[i], states[i], ph, written[i], world.emb_index, world.model_index)
                 for i, ph in zip(active, phases)]
        idx = torch.tensor(active, device=device)
        out = model(collate(views, device), memory.select(idx) if memory is not None else None,
                    bp_steps=bp_steps, record=record)
        if memory is not None:
            memory = memory.merge(idx, out.memory)
        if compute_loss:
            succ, halt_t, band = collate_targets(
                [target(episodes[i], states[i], ph) for i, ph in zip(active, phases)], device)
            terms["model"].append(F.binary_cross_entropy_with_logits(out.model_logits, succ, reduction="none").mean(1))
            terms["complexity"].append((torch.sigmoid(out.complexity_logit) - band) ** 2)
            is_halt = torch.tensor([ph == HALT for ph in phases], device=device)
            if is_halt.any():
                terms["halt"].append(F.binary_cross_entropy_with_logits(
                    out.halt_logit[is_halt], halt_t[is_halt], reduction="none"))
        if record:
            residuals.append(out.residuals)
            for j, (i, v) in enumerate(zip(active, views)):
                if v.write and out.stats:
                    newest = states[i].trace[-1]
                    log.append({"episode_id": episodes[i].skeleton.episode_id, "ep_type": episodes[i].skeleton.ep_type,
                                "step_index": newest.step_index, "round": newest.round, "model_id": newest.model_id,
                                "verifier_ok": newest.verifier_ok, "surprise": out.stats["surprise"][j].item(),
                                "rate": out.stats["rate"][j].item(), "alpha": out.stats["alpha"][j].item()})
        p_model = torch.sigmoid(out.model_logits).detach().cpu().numpy()
        p_halt = torch.sigmoid(out.halt_logit).detach().cpu().numpy()
        for j, (i, ph, v) in enumerate(zip(active, phases, views)):
            ep, s = episodes[i], states[i]
            if v.write:
                written[i] = len(s.trace)
            if ph == CHOOSE:
                if policy == "mixture":
                    src = str(rng.choice(SOURCES, p=SHARES))
                    source[i] = "oracle" if src == "model" and not model_share else src
                if source[i] == "oracle":
                    m = step_labels(ep, s).oracle_model
                elif source[i] == "random":
                    m = int(rng.integers(ep.n_models))
                else:
                    affordable = [k for k in range(ep.n_models) if world.card_costs[k] <= s.budget_left]
                    m = tau_choice(p_model[j], affordable or list(range(ep.n_models)), tau)
                states[i] = ep.play(s, m)
                continue
            options = [k for k in range(s.cur_model + 1, ep.n_models) if world.card_costs[k] <= s.budget_left]
            m2 = None
            if source[i] == "oracle":
                if halt_label(ep, s):
                    ok = ep.succeeds(s)
                    m2 = next(k for k in ep.escalation_options(s) if ok[k])
            elif source[i] == "random":
                if options and rng.random() < RANDOM_ESCALATE_P:
                    m2 = int(rng.choice(options))
            elif options and p_halt[j] > 0.5:
                m2 = tau_choice(p_model[j], options, tau)
            states[i] = ep.play(s, m2) if m2 is not None else ep.end_step(s)
    results = [EpisodeResult(ep.skeleton.episode_id, ep.skeleton.ep_type, ep.quality(s), s.spent)
               for ep, s in zip(episodes, states)]
    loss, parts = None, {}
    if compute_loss:
        means = {k: torch.cat(v).mean() if v else torch.zeros((), device=device) for k, v in terms.items()}
        loss = means["model"] + means["halt"] + means["complexity"]
        parts = {k: v.item() for k, v in means.items()}
    return RolloutOutput(loss, parts, results, log, residuals)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rollout.py`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add train/rollout.py tests/test_rollout.py
git commit -m "feat(train): batched lockstep rollout with behaviour mixture and losses"
```

---

### Task 7: Training loop

**Files:**
- Create: `train/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `rollout`, `World`, `Controller`, `CONFIGS`, `n_parameters`, `bp_schedule`;
  - `load_prepared`, `OutcomeTable`, `SOURCES` (as `BENCH_SOURCES`);
  - `load_cache`, `load_skeletons`, `Episode`.
- Produces:
  - Dataclass `Data(world, train, val, embeddings, cards, n_types)`.
  - `load_data(data_dir=DATA_DIR) -> Data`.
  - `make_episodes(skeletons, world) -> list[Episode]`.
  - `evaluate(model, skeletons, world, tau=0.5, batch_size=512) -> dict` with keys `quality` and `cost_usd`.
  - `validation_loss(model, skeletons, world, batch_size=512) -> float`.
  - `train(config, seed=0, epochs=20, batch_size=256, lr=3e-4, weight_decay=0.01, patience=3, bp_min=2, max_train=None, max_val=None, out_dir=Path("runs"), data=None, log=print) -> dict`.
  - `load_model(run_dir, data=None) -> Controller`.
  - CLI: `python -m train.train --config NAME --seed S [--epochs E] [--max-train N] [--max-val N] [--bp-min B]`.
  - Run output in `runs/<config>-s<seed>/`: `model.pt` (EMA weights) and `metrics.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train.py
import json

import numpy as np
import torch

from data.episodes import generate_skeletons
from tests.conftest import make_table
from tests.test_env import IDS
from train.rollout import World
from train.train import Data, load_model, train


def tiny_data():
    table = make_table(n=200, seed=11)
    world = World(table, IDS, np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4,
                  {p: i for i, p in enumerate(table.prompt_ids)}, {m: i for i, m in enumerate(IDS)})
    sks = generate_skeletons(table, list(table.prompt_ids), n=96, seed=0, split="t")
    torch.manual_seed(0)
    return Data(world, sks[:64], sks[64:], torch.randn(200, 384), torch.rand(4, 6), n_types=2)


def test_training_writes_checkpoint_and_metrics(tmp_path):
    data = tiny_data()
    summary = train("full", seed=0, epochs=2, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"
    assert len(summary["history"]) == 2 and (run / "model.pt").exists()
    assert json.loads((run / "metrics.json").read_text())["params"] == summary["params"]
    assert all(np.isfinite(h["val_loss"]) and 0 <= h["val_quality"] <= 1 for h in summary["history"])
    model = load_model(run, data)
    assert model.cfg.gate == "cfc"


def test_training_reduces_loss(tmp_path):
    summary = train("memory_off", seed=0, epochs=6, batch_size=32, lr=3e-3, patience=10, data=tiny_data(),
                    out_dir=tmp_path, log=lambda row: None)
    history = summary["history"]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'train.train'`

- [ ] **Step 3: Write `train/train.py`**

```python
"""Supervised training (plan §6, D1, D11).

    python -m train.train --config full --seed 0

Episodes are rolled out live under a behaviour mixture; validation and checkpoints
use an EMA of the weights."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from controller.core import bp_schedule
from controller.model import CONFIGS, Controller, n_parameters
from data.embed import load_cache
from data.env import Episode
from data.episodes import Skeleton, load_skeletons
from data.routerbench import DATA_DIR, OutcomeTable, load_prepared
from data.routerbench import SOURCES as BENCH_SOURCES
from train.rollout import World, rollout

EMA_DECAY = 0.999


@dataclass
class Data:
    world: World
    train: list[Skeleton]
    val: list[Skeleton]
    embeddings: torch.Tensor
    cards: torch.Tensor
    n_types: int


def load_data(data_dir: Path = DATA_DIR) -> Data:
    prompts, cards = load_prepared(data_dir)
    emb, emb_index = load_cache(data_dir)
    ids = [c.model_id for c in cards]
    world = World(OutcomeTable.from_frame(prompts), ids, np.array([c.mean_cost_usd for c in cards]), emb_index,
                  {m: i for i, m in enumerate(ids)})
    return Data(world, load_skeletons("train", data_dir), load_skeletons("val", data_dir),
                torch.from_numpy(np.array(emb)), torch.tensor([c.capability for c in cards]), len(BENCH_SOURCES))


def make_episodes(skeletons: Sequence[Skeleton], world: World) -> list[Episode]:
    return [Episode(sk, world.table, world.model_ids) for sk in skeletons]


def evaluate(model, skeletons: Sequence[Skeleton], world: World, tau: float = 0.5, batch_size: int = 512) -> dict:
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            results += rollout(model, make_episodes(skeletons[start:start + batch_size], world), world,
                               tau=tau).results
    return {"quality": float(np.mean([r.quality for r in results])),
            "cost_usd": float(np.mean([r.cost_usd for r in results]))}


def validation_loss(model, skeletons: Sequence[Skeleton], world: World, batch_size: int = 512) -> float:
    model.eval()
    rng = np.random.default_rng(1234)  # same behaviour draws every epoch
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(skeletons), batch_size):
            chunk = skeletons[start:start + batch_size]
            out = rollout(model, make_episodes(chunk, world), world, policy="mixture", rng=rng, compute_loss=True)
            total += out.loss.item() * len(chunk)
            count += len(chunk)
    return total / count


def train(config: str, seed: int = 0, epochs: int = 20, batch_size: int = 256, lr: float = 3e-4,
          weight_decay: float = 0.01, patience: int = 3, bp_min: int = 2, max_train: int | None = None,
          max_val: int | None = None, out_dir: Path = Path("runs"), data: Data | None = None,
          log: Callable[[dict], None] = print) -> dict:
    data = data or load_data()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Controller(CONFIGS[config], data.embeddings, data.cards, data.n_types)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_sk = data.train[:max_train] if max_train else data.train
    val_sk = data.val[:max_val] if max_val else data.val
    total_steps = epochs * math.ceil(len(train_sk) / batch_size)
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    run_dir = Path(out_dir) / f"{config}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale, step = [], math.inf, 0, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_sk))
        losses, bp = [], bp_min
        for start in range(0, len(order), batch_size):
            episodes = make_episodes([train_sk[k] for k in order[start:start + batch_size]], data.world)
            bp = bp_schedule(step, total_steps, bp_min=bp_min)
            out = rollout(model, episodes, data.world, policy="mixture", model_share=epoch > 0, rng=rng,
                          bp_steps=bp, compute_loss=True)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            losses.append(out.loss.item())
            step += 1
        val_loss = validation_loss(ema.module, val_sk, data.world)
        val = evaluate(ema.module, val_sk, data.world)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_quality": val["quality"], "val_cost": val["cost_usd"], "bp_steps": bp}
        history.append(row)
        log(row)
        if val_loss < best - 1e-4:
            best, stale = val_loss, 0
            torch.save(ema.module.state_dict(), run_dir / "model.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    summary = {"config": config, "seed": seed, "params": n_parameters(model), "best_val_loss": best,
               "model_config": asdict(CONFIGS[config]), "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(run_dir: Path, data: Data | None = None) -> Controller:
    data = data or load_data()
    summary = json.loads((Path(run_dir) / "metrics.json").read_text())
    model = Controller(CONFIGS[summary["config"]], data.embeddings, data.cards, data.n_types)
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", weights_only=True))
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--bp-min", type=int, default=2)
    a = parser.parse_args()
    train(a.config, a.seed, a.epochs, a.batch_size, bp_min=a.bp_min, max_train=a.max_train, max_val=a.max_val,
          log=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train.py`
Expected: 2 passed.

- [ ] **Step 5: Real-data smoke run**

Run: `.venv/bin/python -m train.train --config full --epochs 1 --max-train 512 --max-val 256`
Expected: one JSON line with finite `train_loss` and `val_loss`; `runs/full-s0/model.pt` written. Then `rm -rf runs/full-s0`.

- [ ] **Step 6: Commit**

```bash
git add train/train.py tests/test_train.py
git commit -m "feat(train): supervised training loop with EMA, staircase horizon and early stopping"
```

---

### Task 8: Residual and memory diagnostics

**Files:**
- Create: `eval/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `rollout`, `World`, `make_episodes`, `load_model`, `load_data`.
- Produces:
  - `residual_profile(model, skeletons, world, n=256) -> list[tuple[str, float]]`: 8 entries, the mean of every pass.
  - `plot_residuals(profile, path)`.
  - `memory_trace(model, skeletons, world, episode_id) -> pd.DataFrame`.
  - CLI: `python -m eval.diagnostics runs/<run>`, which writes `residuals.png`, `residuals.json` and `memory_trace.csv` into the run dir.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_diagnostics.py
from eval.diagnostics import memory_trace, plot_residuals, residual_profile
from tests.test_rollout import tiny_model, world_and_episodes


def test_residual_profile_has_six_l_and_two_h_steps(tmp_path):
    world, eps = world_and_episodes(8)
    profile = residual_profile(tiny_model("full").eval(), [ep.skeleton for ep in eps], world, n=8)
    assert [k for k, _ in profile] == ["L", "L", "L", "H", "L", "L", "L", "H"]
    assert all(v >= 0 for _, v in profile)
    plot_residuals(profile, tmp_path / "r.png")
    assert (tmp_path / "r.png").stat().st_size > 0


def test_memory_trace_lists_writes_of_one_episode():
    world, eps = world_and_episodes(8)
    sks = [ep.skeleton for ep in eps]
    df = memory_trace(tiny_model("full").eval(), sks, world, sks[0].episode_id)
    assert len(df) >= 1 and (df.episode_id == sks[0].episode_id).all()
    assert {"step_index", "verifier_ok", "surprise", "rate", "alpha"} <= set(df.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_diagnostics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.diagnostics'`

- [ ] **Step 3: Write `eval/diagnostics.py`**

```python
"""Residual and memory diagnostics (plan §7 outputs 4 and 5).

    python -m eval.diagnostics runs/full-s0"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.episodes import Skeleton
from train.rollout import World, rollout
from train.train import load_data, load_model, make_episodes


def residual_profile(model, skeletons: Sequence[Skeleton], world: World, n: int = 256) -> list[tuple[str, float]]:
    """Mean step-to-step change of z_L / z_H over the 8 inner updates, averaged over every pass."""
    with torch.no_grad():
        out = rollout(model, make_episodes(skeletons[:n], world), world, record=True)
    kinds = [k for k, _ in out.residuals[0]]
    values = np.mean([[v for _, v in r] for r in out.residuals], axis=0)
    return list(zip(kinds, values.tolist()))


def plot_residuals(profile: list[tuple[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, l_i, h_i = [], 0, 0
    for kind, _ in profile:
        if kind == "L":
            l_i += 1
            labels.append(f"L{l_i}")
        else:
            h_i += 1
            labels.append(f"H{h_i}")
    xs = np.arange(len(profile))
    ys = [v for _, v in profile]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, ys, color="0.6", lw=1)
    for x, (kind, y) in zip(xs, profile):
        ax.scatter(x, y, color="#0d7a6f" if kind == "L" else "#3a55a4", marker="o" if kind == "L" else "s", zorder=3)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("mean ‖Δz‖ per update")
    ax.set_title("Inner-loop residuals (L: circles, H: squares)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def memory_trace(model, skeletons: Sequence[Skeleton], world: World, episode_id: str) -> pd.DataFrame:
    sk = next(s for s in skeletons if s.episode_id == episode_id)
    with torch.no_grad():
        out = rollout(model, make_episodes([sk], world), world, record=True)
    return pd.DataFrame(out.log)


def main(run_dir: str) -> None:
    run = Path(run_dir)
    data = load_data()
    model = load_model(run, data)
    profile = residual_profile(model, data.val, data.world)
    plot_residuals(profile, run / "residuals.png")
    (run / "residuals.json").write_text(json.dumps(profile, indent=2))
    if model.memory is not None:
        with torch.no_grad():
            out = rollout(model, make_episodes(data.val[:512], data.world), data.world, record=True)
        pd.DataFrame(out.log).to_csv(run / "memory_trace.csv", index=False)
    print(json.dumps(profile))


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_diagnostics.py`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/diagnostics.py tests/test_diagnostics.py
git commit -m "feat(eval): residual profile plot and memory write trace"
```

---

### Task 9: First real model (`memory_off`) and residual diagnostic (Days 7–8)

**Files:**
- Modify: `README.md` (training commands)

**Interfaces:**
- Consumes: CLI from Tasks 7 and 8, and `python -m eval.baselines val` from week 1.

- [ ] **Step 1: Baseline numbers on validation**

Run: `.venv/bin/python -m eval.baselines val`
Expected: the table prints; note `always_cheapest` quality and cost on val.

- [ ] **Step 2: Train `memory_off` (background, logs to file)**

Run: `.venv/bin/python -m train.train --config memory_off --seed 0 > runs/memory_off-s0.log 2>&1`
Expected: one JSON line per epoch, early-stopped or finished at 20 epochs; `runs/memory_off-s0/model.pt` and `metrics.json` written.

- [ ] **Step 3: Check the Day 7 criterion**

Run: `.venv/bin/python -c "import json; h=json.load(open('runs/memory_off-s0/metrics.json'))['history']; print(max(r['val_quality'] for r in h))"`
Expected: higher than `always_cheapest` val quality from Step 1. If not, stop and debug (superpowers:systematic-debugging) before Task 10.

- [ ] **Step 4: Residual diagnostic (Day 8)**

Run: `.venv/bin/python -m eval.diagnostics runs/memory_off-s0`
Expected: `runs/memory_off-s0/residuals.png` written. L residuals shrink within each cycle (L1 > L2 > L3, L4 > L5 > L6) and jump after each H update. Record the numbers in the ledger.

- [ ] **Step 5: Document and commit**

Add under "Data pipeline" in `README.md`:

```markdown
## Training

    .venv/bin/python -m eval.baselines val                    # reference numbers on validation
    .venv/bin/python -m train.train --config memory_off --seed 0
    .venv/bin/python -m eval.diagnostics runs/memory_off-s0  # residuals.png, memory_trace.csv

Configs: `full`, `gate_off`, `gate_stateless`, `memory_off`, `context_free`.
```

```bash
git add README.md
git commit -m "docs: training and diagnostics commands"
```

---

### Task 10: Memory configs (`gate_off`, `full`, `gate_stateless`) (Days 9–10)

**Files:** none (runs and ledger only)

- [ ] **Step 1: Train the three memory configs, seed 0**

Run, one after another (each in the background with its own log):

```bash
.venv/bin/python -m train.train --config gate_off --seed 0 > runs/gate_off-s0.log 2>&1
.venv/bin/python -m train.train --config full --seed 0 > runs/full-s0.log 2>&1
.venv/bin/python -m train.train --config gate_stateless --seed 0 > runs/gate_stateless-s0.log 2>&1
```

Expected for each: finite losses every epoch, with no NaN or sudden spike above 2× the previous epoch's training loss.

- [ ] **Step 2: Diagnostics for each**

Run: `.venv/bin/python -m eval.diagnostics runs/<config>-s0` for all three.
Expected: `memory_trace.csv` has one row per write. In `full`, `rate` is not constant across writes. Compare its mean on writes where `verifier_ok` is False vs True, and record both in the ledger.

- [ ] **Step 3: Record results**

Ledger line per run: best `val_loss`, `val_quality` and `val_cost` at the best epoch, epochs run, parameter count, and the gate-rate summary. Compare `memory_off`, `gate_off`, `gate_stateless` and `full` on val; this is single-seed and indicative only. These runs produce artefacts in `runs/` only; there is nothing to commit.
