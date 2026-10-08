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
