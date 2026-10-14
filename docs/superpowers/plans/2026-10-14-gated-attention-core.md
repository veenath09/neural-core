# Gated-Attention H/L Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HRM-Text-style token-level H/L core with gated self-attention as a second architecture (`arch="attn"`), alongside the existing MLP core, with three new configs and their seed-0 runs.

**Architecture:** The encoder's six role tokens `[task, budget, step, trace−1, trace−2, memory]` each get a learned slot embedding and are kept as a sequence instead of being averaged. Each H/L module becomes 2 pre-norm blocks of (gated self-attention → SwiGLU) capped by a weightless RMSNorm (MagicNorm). The H/L loop, fixed z_L init and staircase gradient horizon are unchanged and run over the token sequence; an attention key mask hides absent tokens and everything `context_free` must not see. The heads read the current-step token.

**Tech Stack:** Python 3.12, PyTorch (`F.scaled_dot_product_attention`), pytest.

**Spec:** `docs/IMPLEMENTATION_PLAN_neural_core.md` v1.3 (D2, D10, D11). This plan adds D12 (Task 1).

## Global Constraints

- Work directly on `main`. One commit per task. **Commit messages carry no co-author trailer.**
- Run from repo root with `.venv/bin/python`.
- The MLP core and its five configs must stay bit-for-bit unchanged in behaviour (existing tests unchanged and passing).
- Attention follows HRM-Text `models/layers.py` `Attention`: one bias-free projection to `[gate | query | key | value]` (heads-major split), scaled-dot-product attention, output `o_proj(sigmoid(gate) * attn)`. Blocks follow `models/transformer.py` pre-norm: `x + attn(norm(x))`, then `x + mlp(norm(x))`; module exit weightless RMSNorm.
- Width 128, 4 heads × 32, 2 blocks per module, 2 H-cycles × 3 L-steps, staircase horizon `bp = bp_min + ⌊min(1, step/(0.2·total))·(5 − bp_min)⌋`.
- No RoPE: tokens are roles, not a sequence; each slot gets a learned embedding (init N(0, 0.02)).
- Token order is fixed: `0 task, 1 budget, 2 step, 3 trace−1 (newest), 4 trace−2, 5 memory`. The heads read token 2.
- Key mask: token valid iff — task/budget: not `context_free`; step: always; trace j: `trace_present[j]` and (not `context_free` or not `trace_earlier[j]`); memory: a memory module exists. Invalid tokens are zeroed and excluded as keys.
- New configs: `full_attn`, `memory_off_attn`, `context_free_attn` (memory off, context_free on).
- Every new/adapted file names its sources in the header (HRM-Text, Apache-2.0, `LICENSE-HRM-Text`).

## Review Focus

1. `context_free_attn` leaking: its outputs must not change when task, budget, earlier-step trace tokens (features, rows and model ids) are perturbed — attention must not see masked keys.
2. Absent trace slots must not affect outputs (padding invariance).
3. Gradient-horizon gating must be identical in token mode (same calls with grad as the MLP core).
4. The MLP path must be untouched: all existing tests still pass with no edits.

---

### Task 1: Spec D12 and gated attention blocks

**Files:**
- Modify: `docs/IMPLEMENTATION_PLAN_neural_core.md` (insert `### D12` before `### D9. Engineering`)
- Create: `controller/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Consumes: `SwiGLU`, `rms_norm` from `controller/core.py`.
- Produces: `GatedAttention(d, n_heads=4)` with `forward(x[B,T,d], key_mask[B,T] bool) -> [B,T,d]`; `AttnBlock(d, n_heads=4)` with `forward(x, key_mask)`; `AttnModule(d, n_blocks=2, n_heads=4)` with `forward(x, key_mask)` returning RMS-normalised tokens.

- [ ] **Step 1: Add D12 to the spec** — insert this section immediately before `### D9. Engineering`:

```markdown
### D12. Gated-attention core variant (v1.3 → v1.4)

A second core architecture, `arch = "attn"`, alongside the MLP core of D11. The six role tokens `[task, budget, step, trace−1, trace−2, memory readout]` keep their identity (learned slot embedding each, no RoPE because they are roles, not a sequence) instead of being averaged, so the controller can compare the current step with earlier rounds directly (retry, propagation). Each H/L module is 2 pre-norm blocks of HRM-Text gated self-attention (fused bias-free gate/q/k/v projection, `o_proj(σ(gate) ⊙ attn)`, 4 heads × 32) followed by SwiGLU, capped by a weightless RMSNorm. The H/L schedule, fixed z_L init, input re-injection and staircase gradient horizon are those of D11, applied per token. An attention key mask hides absent trace slots and everything `context_free` must not see; the heads read the current-step token. Configs: `full_attn`, `memory_off_attn`, `context_free_attn`. The MLP configs are unchanged.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_attention.py
import math

import pytest
import torch
import torch.nn.functional as F

from controller.attention import AttnModule, GatedAttention


def reference_gated_attention(att: GatedAttention, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    """Explicit softmax implementation of HRM-Text's gated attention, for comparison."""
    B, T, d = x.shape
    h, hd = att.n_heads, att.head_dim
    g, q, k, v = att.gqkv(x).view(B, T, 4, h, hd).unbind(2)
    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(hd)
    scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    out = torch.einsum("bhts,bshd->bthd", scores.softmax(-1), v)
    return att.out((torch.sigmoid(g) * out).reshape(B, T, d))


def test_gated_attention_matches_reference():
    torch.manual_seed(0)
    att = GatedAttention(32, n_heads=4)
    x = torch.randn(3, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    assert torch.allclose(att(x, mask), reference_gated_attention(att, x, mask), atol=1e-5)


def test_masked_keys_do_not_affect_visible_tokens():
    torch.manual_seed(0)
    mod = AttnModule(32)
    x = torch.randn(2, 6, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0]], dtype=torch.bool)
    y = x.clone()
    y[~mask] = torch.randn(int((~mask).sum()), 32) * 5
    a, b = mod(x, mask), mod(y, mask)
    assert torch.allclose(a[mask], b[mask], atol=1e-5)


def test_module_output_is_rms_normalised_per_token():
    out = AttnModule(16)(torch.randn(4, 6, 16) * 10, torch.ones(4, 6, dtype=torch.bool))
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(4, 6), atol=1e-3)


def test_projection_is_bias_free_and_sized_for_gate_qkv():
    att = GatedAttention(128, n_heads=4)
    assert att.gqkv.bias is None and att.out.bias is None
    assert att.gqkv.weight.shape == (4 * 128, 128)


def test_heads_must_divide_width():
    with pytest.raises(ValueError, match="divisible"):
        GatedAttention(30, n_heads=4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_attention.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'controller.attention'`

- [ ] **Step 4: Write `controller/attention.py`**

```python
"""Gated self-attention blocks for the token-level H/L core (plan D12).

Adapted from sapientinc/HRM-Text, models/layers.py (Attention) and models/transformer.py
(TransformerBlock, Transformer) (Apache-2.0; see LICENSE-HRM-Text).
Kept: one bias-free projection to [gate | query | key | value] split heads-major; an
elementwise sigmoid gate on each head's attention output before the output projection;
pre-norm residual blocks (attention, then SwiGLU); a weightless RMSNorm at the module exit.
Changed: PyTorch scaled-dot-product attention over a few role tokens with a key mask instead
of FlashAttention over a text sequence; no RoPE (tokens are roles, not positions; the model
adds a learned slot embedding instead); no grouped-query attention and no KV cache."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from controller.core import SwiGLU, rms_norm


class GatedAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"width {d} is not divisible by {n_heads} heads")
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.gqkv = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d]; key_mask: [B, T] bool, True where a token may be attended to."""
        B, T, d = x.shape
        g, q, k, v = self.gqkv(x).view(B, T, 4, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, heads, T, head_dim]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        return self.out((torch.sigmoid(g) * attn.transpose(1, 2)).reshape(B, T, d))


class AttnBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.attn = GatedAttention(d, n_heads)
        self.mlp = SwiGLU(d)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), key_mask)
        return x + self.mlp(rms_norm(x))


class AttnModule(nn.Module):
    """Pre-norm gated-attention blocks capped by a weightless RMSNorm (MagicNorm)."""

    def __init__(self, d: int, n_blocks: int = 2, n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(AttnBlock(d, n_heads) for _ in range(n_blocks))

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_mask)
        return rms_norm(x)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_attention.py` → 5 passed. Then `.venv/bin/python -m pytest` → all pass.

- [ ] **Step 6: Commit**

```bash
git add docs/IMPLEMENTATION_PLAN_neural_core.md controller/attention.py tests/test_attention.py
git commit -m "feat(controller): HRM-Text gated self-attention blocks and spec D12"
```

---

### Task 2: Token-level H/L core

**Files:**
- Modify: `controller/core.py` (`HLCore`)
- Test: `tests/test_core.py` (append)

**Interfaces:**
- Consumes: `AttnModule` (Task 1).
- Produces: `HLCore(d=128, h_cycles=2, l_cycles=3, arch="mlp", n_heads=4)`; `forward(h, x, bp_steps=5, record=False, key_mask=None)`. For `arch="mlp"`: unchanged ([B,d] in and out, `key_mask` ignored). For `arch="attn"`: `h`, `x` are [B,T,d], `key_mask` [B,T] bool is required, returns z_H [B,T,d].

- [ ] **Step 1: Write the failing tests** (append to `tests/test_core.py`)

```python
def test_attn_core_shapes_and_record():
    core = HLCore(d=32, arch="attn")
    x = torch.randn(3, 6, 32)
    mask = torch.ones(3, 6, dtype=torch.bool)
    z, rec = core(x, x, record=True, key_mask=mask)
    assert z.shape == (3, 6, 32)
    assert [k for k, _ in rec] == ["L", "L", "L", "H", "L", "L", "L", "H"]


def test_attn_core_uses_same_gradient_horizon():
    core = HLCore(d=32, arch="attn")
    calls = []
    forward = core.L.forward

    def spy(inp, key_mask):
        calls.append(torch.is_grad_enabled())
        return forward(inp, key_mask)

    core.L.forward = spy
    x = torch.randn(2, 6, 32)
    core(x, x, bp_steps=4, key_mask=torch.ones(2, 6, dtype=torch.bool))
    assert calls == [False, False, False, False, True, True]


def test_attn_core_requires_key_mask():
    core = HLCore(d=32, arch="attn")
    x = torch.randn(1, 6, 32)
    with pytest.raises(ValueError, match="key_mask"):
        core(x, x)


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="arch"):
        HLCore(d=32, arch="rnn")
```

Add `import pytest` at the top of `tests/test_core.py` if not present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_core.py` → the 4 new tests fail (unexpected keyword `arch`).

- [ ] **Step 3: Implement** — in `controller/core.py`, replace the `HLCore` class with:

```python
class HLCore(nn.Module):
    def __init__(self, d: int = 128, h_cycles: int = 2, l_cycles: int = 3, arch: str = "mlp",
                 n_heads: int = 4):
        super().__init__()
        self.h_cycles, self.l_cycles, self.arch = h_cycles, l_cycles, arch
        if arch == "mlp":
            self.H, self.L = ReasoningModule(d), ReasoningModule(d)
        elif arch == "attn":
            from controller.attention import AttnModule

            self.H, self.L = AttnModule(d, n_heads=n_heads), AttnModule(d, n_heads=n_heads)
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.register_buffer("zL_init", nn.init.trunc_normal_(torch.empty(d), std=1.0, a=-2.0, b=2.0))

    def _step(self, module: nn.Module, inp: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
        return module(inp) if self.arch == "mlp" else module(inp, key_mask)

    def forward(self, h: torch.Tensor, x: torch.Tensor, bp_steps: int = 5, record: bool = False,
                key_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list[tuple[str, float]]]:
        if self.arch == "attn" and key_mask is None:
            raise ValueError("the attention core needs a key_mask")
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
                    new = self._step(self.L, z_l + z_h + x, key_mask)
                if record:
                    residuals.append(("L", (new - z_l).norm(dim=-1).mean().item()))
                z_l = new
            with torch.set_grad_enabled(grad and cycle >= self.h_cycles - h_bp):
                new = self._step(self.H, z_h + z_l + h, key_mask)
            if record:
                residuals.append(("H", (new - z_h).norm(dim=-1).mean().item()))
            z_h = new
        return z_h, residuals
```

Also append to the module docstring of `controller/core.py`: `arch="attn" (D12) swaps each module for HRM-Text gated-attention blocks over role tokens (controller/attention.py).`

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/test_core.py` → all pass (old and new); then the full suite.

- [ ] **Step 5: Commit**

```bash
git add controller/core.py tests/test_core.py
git commit -m "feat(controller): token-level H/L core option with gated attention"
```

---

### Task 3: Token assembly, masking and configs in the controller

**Files:**
- Modify: `controller/model.py`
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Consumes: `HLCore(..., arch, n_heads)` (Task 2).
- Produces: `ControllerConfig` gains `arch: str = "mlp"`, `n_heads: int = 4`; `CONFIGS` gains `full_attn`, `memory_off_attn`, `context_free_attn`; `Controller.forward` unchanged signature; `Output.z_h` stays [B,d] (the step token's z_H in attn mode); `SLOTS = 6`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model.py`)

```python
ATTN = ["full_attn", "memory_off_attn", "context_free_attn"]


def test_attn_configs_exist_and_use_attention():
    for name in ATTN:
        assert CONFIGS[name].arch == "attn"
    assert CONFIGS["context_free_attn"].context_free and not CONFIGS["context_free_attn"].memory
    assert all(CONFIGS[n].arch == "mlp" for n in ["full", "gate_off", "gate_stateless", "memory_off", "context_free"])


def test_attn_parameter_count():
    n = n_parameters(make_model("full_attn"))
    print(f"\nfull_attn controller: {n:,} trainable parameters")
    assert 800_000 < n < 1_500_000


def test_context_free_attn_ignores_task_budget_and_earlier_steps():
    m = make_model("context_free_attn").eval()
    b = sample_batch()
    stronger = replace(perturb_context(b),
                       trace_model=torch.where(b.trace_earlier, torch.full_like(b.trace_model, 3), b.trace_model))
    with torch.no_grad():
        assert torch.allclose(m(b).model_logits, m(stronger).model_logits, atol=1e-5)
        assert torch.allclose(m(b).halt_logit, m(stronger).halt_logit, atol=1e-5)


def test_context_free_attn_still_sees_current_steps_round_one():
    m = make_model("context_free_attn").eval()
    b = sample_batch()
    flipped = replace(b, trace_feats=b.trace_feats.clone())
    flipped.trace_feats[1, 0, 0] = -flipped.trace_feats[1, 0, 0]
    with torch.no_grad():
        assert not torch.allclose(m(b).halt_logit[1], m(flipped).halt_logit[1])


def test_attn_absent_trace_slots_do_not_matter():
    m = make_model("full_attn").eval()
    b = sample_batch()
    absent = ~b.trace_present
    noisy = replace(b, trace_feats=torch.where(absent[..., None], torch.rand_like(b.trace_feats) * 9, b.trace_feats),
                    trace_row=torch.where(absent, torch.full_like(b.trace_row, 8), b.trace_row))
    with torch.no_grad():
        assert torch.allclose(m(b, m.init_memory(3)).model_logits, m(noisy, m.init_memory(3)).model_logits, atol=1e-5)


def test_full_attn_reads_context_and_memory():
    m = make_model("full_attn").eval()
    b = sample_batch()
    with torch.no_grad():
        out = m(b, m.init_memory(3))
        assert not torch.allclose(out.model_logits, m(perturb_context(b), m.init_memory(3)).model_logits)
    assert out.z_h.shape == (3, 128) and out.stats["rate"].shape == (3,)
```

- [ ] **Step 2: Run tests to verify they fail** — `.venv/bin/python -m pytest tests/test_model.py` → new tests fail (`KeyError: 'full_attn'` / no `arch`).

- [ ] **Step 3: Implement** in `controller/model.py`:

(a) Add to `ControllerConfig` (after `context_free`):

```python
    arch: str = "mlp"  # mlp (D11) | attn (D12: gated attention over role tokens)
    n_heads: int = 4
```

(b) Add to `CONFIGS`:

```python
    "full_attn": ControllerConfig(arch="attn"),
    "memory_off_attn": ControllerConfig(arch="attn", memory=False),
    "context_free_attn": ControllerConfig(arch="attn", memory=False, context_free=True),
```

(c) Add module constant `SLOTS = 6  # task, budget, step, trace-1, trace-2, memory (D12)` and `STEP_SLOT = 2`.

(d) In `Controller.__init__`: build the core as `HLCore(cfg.d, cfg.h_cycles, cfg.l_cycles, arch=cfg.arch, n_heads=cfg.n_heads)`; and when `cfg.arch == "attn"`:

```python
            self.slot = nn.Parameter(torch.empty(SLOTS, cfg.d))
            nn.init.normal_(self.slot, std=0.02)
```

(e) In `forward`, keep the MLP path exactly as it is, and branch for attention after the `context_free` masking and the computation of `h`:

```python
        if self.cfg.arch == "attn":
            return self._forward_tokens(b, e, task, budget, trace, present, h, memory, bp_steps, record)
```

placed right after `h = rms_norm((task + budget) / 2)` (before `x` is computed), and add:

```python
    def _forward_tokens(self, b, e, task, budget, trace, present, h, memory, bp_steps, record) -> Output:
        n = task.shape[0]
        on = torch.ones(n, dtype=torch.bool, device=task.device)
        stats: dict[str, torch.Tensor] = {}
        mem_tok, mem_ok = torch.zeros_like(task), ~on
        if self.memory is not None:
            memory, stats = self.memory.write(memory, e.trace[:, 0], b.write, b.write_dt)
            mem_tok, mem_ok = self.memory.read(memory, h), on
        ctx = ~on if self.cfg.context_free else on
        tokens = torch.stack([task, budget, e.step, trace[:, 0], trace[:, 1], mem_tok], dim=1) + self.slot
        valid = torch.stack([ctx, ctx, on, present[:, 0], present[:, 1], mem_ok], dim=1)
        tokens = tokens * valid[..., None]
        z_seq, residuals = self.core(tokens, tokens, bp_steps, record, key_mask=valid)
        z_h = z_seq[:, STEP_SLOT]
        model_logits, halt_logit, complexity_logit = self.heads(z_h, e.card)
        return Output(model_logits, halt_logit, complexity_logit, z_h, memory, stats, residuals)
```

Update the module docstring: `Controller = encoder -> episode memory -> H/L core -> heads (plan §5, D2, D10, D11, D12). arch="attn" keeps the six role tokens as a sequence for the gated-attention core.`

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/test_model.py -s` → all pass, prints the `full_attn` parameter count; then the full suite (existing tests unchanged).

- [ ] **Step 5: Smoke-train** — `.venv/bin/python -m train.train --config full_attn --epochs 1 --max-train 512 --max-val 256 --out /tmp/attn_smoke` is not a valid flag; instead run `.venv/bin/python -c "from train.train import train; print(train('full_attn', epochs=1, max_train=512, max_val=256, out_dir=__import__('pathlib').Path('/tmp/attn_smoke'))['history'])"` and confirm finite losses.

- [ ] **Step 6: Commit**

```bash
git add controller/model.py tests/test_model.py
git commit -m "feat(controller): gated-attention configs over six role tokens"
```

---

### Task 4: Seed-0 runs (controller runs these)

- [ ] Train `context_free`, `full_attn`, `memory_off_attn`, `context_free_attn` at seed 0 in parallel: `.venv/bin/python -m train.train --config <name> --seed 0 > runs/<name>-s0.log 2>&1`.
- [ ] `.venv/bin/python -m eval.diagnostics runs/<name>-s0` for each.
- [ ] Measure decision latency for `full_attn` (batch 1, CPU, one thread) as in the MLP measurement.
- [ ] Report: per-config val quality, cost, per-episode-type quality, params, latency; `full` vs `context_free` and `full_attn` vs `context_free_attn` per type.
