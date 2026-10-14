import torch
import pytest

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
