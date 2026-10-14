"""H/L reasoning core (plan §5, D11).

Adapted from sapientinc/HRM-Text, models/baselines/hrm_nocarry_bp_warmup.py and
models/transformer.py (Apache-2.0; see LICENSE-HRM-Text).
Kept: 2 H-cycles x 3 L-steps; MagicNorm (pre-norm blocks capped by a weightless
RMSNorm); a fixed, non-trainable z_L init; the staircase gradient-horizon warmup
and its H/L split.
Changed: SwiGLU MLP blocks on one vector per module instead of attention over a
token sequence; input x injected at every L-step (as in HRM and TRM); the H input h
re-injected at every H-step so the task/budget encoders get gradient under
truncated backprop; z_H initialised from pooled H tokens.
arch="attn" (D12) swaps each module for HRM-Text gated-attention blocks over role tokens (controller/attention.py)."""

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
