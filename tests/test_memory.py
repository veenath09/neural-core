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


def test_read_is_zero_when_empty_and_rms_normalised_after_a_write():
    torch.manual_seed(0)
    mem = EpisodeMemory(d=16)
    state = mem.init_state(2)
    z = torch.randn(2, 16)
    assert torch.allclose(mem.read(state, z), torch.zeros(2, 16))
    new, _ = mem.write(state, torch.randn(2, 16), ON(2), torch.full((2,), 0.2))
    raw = torch.bmm(new.W, F.normalize(mem.query(z), dim=-1).unsqueeze(-1)).squeeze(-1)
    m = raw.pow(2).mean(-1)
    assert (m > 0).all()
    assert torch.allclose(mem.read(new, z).pow(2).mean(-1), m / (m + 1e-6), atol=1e-5)


def test_select_and_merge_rows():
    mem = EpisodeMemory(d=8)
    state = mem.init_state(3)
    part, _ = mem.write(state.select(torch.tensor([2])), torch.randn(1, 8), ON(1), torch.ones(1))
    merged = state.merge(torch.tensor([2]), part)
    assert torch.equal(merged.W[2], part.W[0]) and torch.equal(merged.W[0], state.W[0])


def test_unknown_gate_is_rejected():
    with pytest.raises(ValueError, match="gate"):
        EpisodeMemory(d=8, gate="lstm")
