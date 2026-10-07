"""D13: lambda- and complexity-conditioned policy outputs (C1). Policy logits shift the
model head toward cheaper models as lambda grows; halt policy shifts the halting head.
context_free sees lambda but still ignores context."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from controller.policy import log_price
from tests.test_model import make_model, perturb_context, sample_batch

LP = torch.log(torch.tensor([1.0, 2.0, 4.0, 8.0]))


def test_no_lam_means_no_policy_outputs():
    m = make_model("full")
    out = m(sample_batch(), m.init_memory(3))
    assert out.policy_logits is None and out.halt_policy_logit is None


def test_policy_at_init_is_success_minus_lam_times_price_init():
    m = make_model("full")
    lam = torch.full((3,), 0.6)
    out = m(sample_batch(), m.init_memory(3), lam=lam, log_price=LP)
    assert torch.allclose(out.policy_logits, out.model_logits - 0.6 * 0.1 * LP[None], atol=1e-6)
    assert torch.allclose(out.halt_policy_logit, out.halt_logit)


def test_higher_lam_prefers_cheaper_models():
    m = make_model("full")
    b = sample_batch()
    lo = m(b, m.init_memory(3), lam=torch.full((3,), 0.1), log_price=LP).policy_logits.softmax(-1)
    hi = m(b, m.init_memory(3), lam=torch.full((3,), 2.0), log_price=LP).policy_logits.softmax(-1)
    assert (hi[:, 0] > lo[:, 0]).all() and (hi[:, 3] < lo[:, 3]).all()


def test_policy_gradient_reaches_complexity_head():
    m = make_model("full")
    with torch.no_grad():
        m.heads.lam_adj.weight[:, -1] = 1.0
    out = m(sample_batch(), m.init_memory(3), lam=torch.full((3,), 1.0), log_price=LP)
    out.policy_logits.sum().backward()
    assert m.heads.complexity.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["context_free", "context_free_attn"])
def test_context_free_policy_ignores_context_but_sees_lam(name):
    m = make_model(name)
    b = sample_batch()
    lam = torch.full((3,), 1.0)
    a = m(b, m.init_memory(3), lam=lam, log_price=LP)
    c = m(perturb_context(b), m.init_memory(3), lam=lam, log_price=LP)
    assert torch.equal(a.policy_logits, c.policy_logits) and torch.equal(a.halt_policy_logit, c.halt_policy_logit)
    with torch.no_grad():
        m.heads.lam_adj.weight[:, -2] = 1.0  # the ln lambda column
    d = m(b, m.init_memory(3), lam=torch.full((3,), 2.0), log_price=LP)
    assert not torch.equal(a.policy_logits, d.policy_logits)


def test_log_price_is_natural_log_relative_to_cost_ref():
    assert torch.allclose(log_price(np.array([1e-3, 2e-3]), 1e-3), torch.tensor([0.0, math.log(2)]))
