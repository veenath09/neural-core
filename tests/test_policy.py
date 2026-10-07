"""D13: policy distributions and selection modes (C1). choose_step picks a model from
policy_logits under an allowed mask; halt_step decides stop-vs-escalate from the halt
policy logit and, on escalation, a model via choose_step. Both modes keep grad to the
input logits; sample mode is stochastic (generator-seeded), serve mode is deterministic
argmax / threshold-0.5."""

from __future__ import annotations

import math

import pytest
import torch

from controller.policy import choose_step, halt_step


def test_choose_never_picks_a_disallowed_model():
    logits = torch.zeros(1, 4)
    allowed = torch.tensor([[False, True, False, True]])
    gen = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(200):
        model, logp, entropy = choose_step(logits, allowed, mode="sample", gen=gen)
        seen.add(int(model.item()))
    assert seen == {1, 3}


def test_choose_logp_matches_masked_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    allowed = torch.tensor([[True, False, True, True]])
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy = choose_step(logits, allowed, mode="sample", gen=gen)

    masked = logits.masked_fill(~allowed, float("-inf"))
    expected_logp_all = torch.log_softmax(masked, dim=-1)
    expected_logp = expected_logp_all.gather(1, model[:, None]).squeeze(1)
    assert torch.allclose(logp, expected_logp, atol=1e-6)

    p = expected_logp_all.exp()
    expected_entropy = -(p[allowed] * expected_logp_all[allowed]).sum()
    assert torch.allclose(entropy, expected_entropy[None], atol=1e-6)


def test_choose_raises_when_nothing_allowed():
    logits = torch.zeros(1, 4)
    allowed = torch.zeros(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        choose_step(logits, allowed, mode="sample", gen=torch.Generator())


def test_serve_is_argmax_and_deterministic():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0], [9.0, 1.0, 2.0, 3.0]])
    allowed = torch.tensor([[True, True, True, True], [True, True, True, True]])
    model1, logp1, ent1 = choose_step(logits, allowed, mode="serve", gen=None)
    model2, logp2, ent2 = choose_step(logits, allowed, mode="serve", gen=None)
    assert torch.equal(model1, torch.tensor([1, 0]))
    assert torch.equal(model1, model2)
    assert torch.allclose(logp1, logp2)
    assert torch.allclose(ent1, ent2)


def test_serve_respects_allowed_mask():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    allowed = torch.tensor([[True, False, True, True]])
    model, logp, entropy = choose_step(logits, allowed, mode="serve", gen=None)
    assert model.item() == 3  # argmax among {0, 2, 3} is index 3 (logit 4.0)


def test_halt_escalate_logp_is_sum_of_parts():
    halt_logit = torch.tensor([20.0])
    policy_logits = torch.tensor([[1.0, 2.0, 3.0]])
    options = torch.tensor([[True, True, True]])
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy, has_choice = halt_step(halt_logit, policy_logits, options, mode="sample", gen=gen)

    assert has_choice.item() is True
    assert model.item() != -1
    log_sigma_x = torch.nn.functional.logsigmoid(halt_logit)
    expected_pi_m2 = torch.log_softmax(policy_logits, dim=-1)[0, model.item()]
    expected_logp = log_sigma_x[0] + expected_pi_m2
    assert torch.allclose(logp, expected_logp[None], atol=1e-5)


def test_halt_stop_logp():
    halt_logit = torch.tensor([-20.0])
    policy_logits = torch.tensor([[1.0, 2.0, 3.0]])
    options = torch.tensor([[True, True, True]])
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy, has_choice = halt_step(halt_logit, policy_logits, options, mode="sample", gen=gen)

    assert model.item() == -1
    expected_logp = torch.nn.functional.logsigmoid(-halt_logit)
    assert torch.allclose(logp, expected_logp, atol=1e-5)


def test_halt_serve_escalates_above_half():
    halt_logit = torch.tensor([5.0, -5.0])
    policy_logits = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    options = torch.tensor([[True, True], [True, True]])
    model, logp, entropy, has_choice = halt_step(halt_logit, policy_logits, options, mode="serve", gen=None)
    assert model[0].item() != -1
    assert model[1].item() == -1


def test_forced_stop_has_no_choice():
    halt_logit = torch.tensor([20.0])  # would escalate, but there are no options
    policy_logits = torch.tensor([[1.0, 2.0, 3.0]])
    options = torch.zeros(1, 3, dtype=torch.bool)
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy, has_choice = halt_step(halt_logit, policy_logits, options, mode="sample", gen=gen)

    assert model.item() == -1
    assert logp.item() == 0.0
    assert entropy.item() == 0.0
    assert has_choice.item() is False


def test_forced_stop_mixed_with_other_rows_no_nan():
    halt_logit = torch.tensor([20.0, -20.0, 20.0], requires_grad=False)
    policy_logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
    options = torch.tensor([[True, True, True], [True, True, True], [False, False, False]])
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy, has_choice = halt_step(halt_logit, policy_logits, options, mode="sample", gen=gen)

    assert not torch.isnan(logp).any()
    assert not torch.isnan(entropy).any()
    assert has_choice.tolist() == [True, True, False]
    assert model[2].item() == -1 and logp[2].item() == 0.0 and entropy[2].item() == 0.0

    logp.sum().backward()
    assert policy_logits.grad is not None
    assert not torch.isnan(policy_logits.grad).any()
    # the forced-stop row must not receive gradient through policy_logits
    assert torch.equal(policy_logits.grad[2], torch.zeros(3))


def test_choose_entropy_grad_is_not_nan_with_single_allowed_entry():
    # log_softmax of an all-but-one-masked row is exactly -inf at the disallowed
    # entries; entropy = -(p * logp) must not let that -inf leak into the backward
    # pass for the one surviving entry (0 * -inf = nan if not guarded).
    logits = torch.randn(3, 6, requires_grad=True)
    allowed = torch.tensor([
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
        [False, True, False, True, False, True],
    ])
    gen = torch.Generator().manual_seed(1)
    model, logp, entropy = choose_step(logits, allowed, mode="sample", gen=gen)
    (logp.sum() + entropy.sum()).backward()
    assert not torch.isnan(logits.grad).any()
    assert torch.equal(logits.grad[~allowed], torch.zeros_like(logits.grad[~allowed]))


def test_logp_has_grad():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    allowed = torch.ones(1, 4, dtype=torch.bool)
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy = choose_step(logits, allowed, mode="sample", gen=gen)
    logp.sum().backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.abs().sum() > 0


def test_halt_logp_has_grad_to_both_inputs():
    halt_logit = torch.tensor([20.0], requires_grad=True)
    policy_logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    options = torch.tensor([[True, True, True]])
    gen = torch.Generator().manual_seed(0)
    model, logp, entropy, has_choice = halt_step(halt_logit, policy_logits, options, mode="sample", gen=gen)
    logp.sum().backward()
    assert halt_logit.grad is not None and halt_logit.grad.abs().sum() > 0
    assert policy_logits.grad is not None and policy_logits.grad.abs().sum() > 0
