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
