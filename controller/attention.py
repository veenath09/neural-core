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
