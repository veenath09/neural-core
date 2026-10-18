"""Tests for the supervised anchor during simulator RL (Task 9, plan D13 deviation)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from controller.contracts import MAX_ROUNDS
from controller.features import CHOOSE, HALT, target
from measurement.rl_reward import load_cost_ref
from measurement.scorer import CompositeScorer
from tests.test_rollout import tiny_model, world_and_episodes
from train.baseline import BucketedBaseline
from train.rl import ANCHOR, ANCHOR_HALT, ANCHOR_MODEL, rl_loss, rl_update
from train.rollout import LOSS_WEIGHTS, rollout


def _lam_idx(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(5, size=n)  # len(LAMS) == 5


def test_anchor_constants_match_supervised_weights():
    assert ANCHOR == 1.0
    assert ANCHOR_MODEL == LOSS_WEIGHTS["model"]
    assert ANCHOR_HALT == LOSS_WEIGHTS["halt"]


def test_sup_model_matches_manual_bce():
    # z_h is recorded per decision (out.decision_states) and e.card is a pure
    # function of the model's own fixed `cards` buffer (encoder.card(cards)),
    # independent of the view. So model.heads(z_h, card) exactly reproduces
    # the model_logits the rollout used to build sup_model, and recomputing
    # the BCE against target(...).succeeds at the replayed pre-action state
    # is an exact (not approximate) check of the brief's formula.
    world, eps = world_and_episodes(4)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    with torch.no_grad():
        out = rollout(
            model,
            eps,
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    rl = out.rl
    assert hasattr(rl, "sup_model")

    with torch.no_grad():
        card = model.encoder.card(model.cards)

    for i, ep in enumerate(eps):
        s = ep.initial_state()

        for t, action in enumerate(out.decision_actions[i]):
            if not s.done and s.rounds == MAX_ROUNDS:
                s = ep.end_step(s)

            phase = CHOOSE if action["phase"] == "choose" else HALT
            tgt = target(ep, s, phase)
            succ = torch.tensor([tgt.succeeds], dtype=torch.float32)

            z_h = out.decision_states[i][t].unsqueeze(0)

            with torch.no_grad():
                model_logits, _halt_logit, _complexity_logit = model.heads(z_h, card)

            expected = F.binary_cross_entropy_with_logits(
                model_logits, succ, reduction="none"
            ).mean(-1)

            got = float(rl.sup_model[i, t])
            assert got == pytest.approx(float(expected.squeeze()), abs=1e-5)

            if action["phase"] == "choose":
                s = ep.play(s, action["model"])
            elif action["halt"] == "escalate":
                s = ep.play(s, action["model"])
            else:
                s = ep.end_step(s)


def test_sup_halt_only_on_halt_slots():
    world, eps = world_and_episodes(4)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    with torch.no_grad():
        out = rollout(
            model,
            eps,
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    rl = out.rl
    assert hasattr(rl, "halt_slot")

    # halt_slot is a subset of valid.
    assert bool((rl.halt_slot & ~rl.valid).any()) is False

    for i in range(len(eps)):
        for t, action in enumerate(out.decision_actions[i]):
            is_halt = action["phase"] == "halt"
            assert bool(rl.halt_slot[i, t]) == is_halt

    # sup_halt is 0 wherever halt_slot is False.
    non_halt_mask = rl.valid & ~rl.halt_slot
    assert torch.allclose(rl.sup_halt[non_halt_mask], torch.zeros_like(rl.sup_halt[non_halt_mask]))


def test_anchor_reduces_supervised_bce():
    """With the anchor on, the supervised BCE drops clearly more than with it off, from the same start."""
    import copy

    world, eps = world_and_episodes(8)
    start = tiny_model("full")
    cost_ref = load_cost_ref()
    lam_idx = np.zeros(len(eps), dtype=int)

    def bce_after(anchor: bool) -> tuple[float, float]:
        model = copy.deepcopy(start)
        baseline = BucketedBaseline()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        def batch():
            return rollout(model, eps, world, policy="sample", rng=np.random.default_rng(0), lam_idx=lam_idx,
                           cost_ref=cost_ref)

        before = float(batch().rl.sup_model[batch().rl.valid].detach().mean())
        for _ in range(20):
            rl_update(model, batch().rl, baseline, opt, beta=0.0, anchor=anchor)
        out = batch()
        return before, float(out.rl.sup_model[out.rl.valid].detach().mean())

    before_on, after_on = bce_after(True)
    before_off, after_off = bce_after(False)
    assert before_on == before_off
    assert (before_on - after_on) > (before_off - after_off) + 0.01

def test_anchor_off_matches_previous_loss():
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    cost_ref = load_cost_ref()
    lam_idx = _lam_idx(len(eps))

    out = rollout(
        model,
        eps,
        world,
        policy="sample",
        rng=np.random.default_rng(3),
        lam_idx=lam_idx,
        cost_ref=cost_ref,
    )

    baseline = BucketedBaseline()
    loss_off, parts_off = rl_loss(out.rl, baseline, beta=0.01, anchor=False)

    # pg + entropy + complexity only, exactly as before the anchor existed.
    logp = out.rl.logp
    entropy = out.rl.entropy
    complexity = out.rl.complexity
    reward = out.rl.reward
    band = out.rl.band
    band_target = out.rl.band_target
    lam_idx_t = out.rl.lam_idx
    choice = out.rl.choice
    valid = out.rl.valid

    from train.rl import COMPLEXITY_AUX, returns_to_go

    returns = returns_to_go(reward)
    lam_idx_bt = lam_idx_t[:, None].expand_as(band)
    baseline_values = baseline.value(band, lam_idx_bt)
    advantage = returns - baseline_values

    chosen_adv = advantage[choice]
    mean = chosen_adv.mean()
    std = chosen_adv.std(unbiased=False)
    norm_advantage = (advantage - mean) / (std + 1e-8)
    pg_terms = -norm_advantage.detach() * logp
    pg_loss = pg_terms[choice].mean()
    entropy_term = -0.01 * entropy[choice].mean()
    complexity_mse = ((complexity - band_target) ** 2)[valid].mean()

    expected_loss = pg_loss + entropy_term + COMPLEXITY_AUX * complexity_mse

    assert torch.allclose(loss_off, expected_loss, atol=1e-6)
    assert "anchor" not in parts_off or parts_off.get("anchor", 0.0) == 0.0


def test_anchor_default_true_adds_to_loss():
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    cost_ref = load_cost_ref()
    lam_idx = _lam_idx(len(eps))

    out = rollout(
        model,
        eps,
        world,
        policy="sample",
        rng=np.random.default_rng(3),
        lam_idx=lam_idx,
        cost_ref=cost_ref,
    )

    baseline = BucketedBaseline()
    loss_on, parts_on = rl_loss(out.rl, baseline, beta=0.01, anchor=True)
    loss_off, parts_off = rl_loss(out.rl, baseline, beta=0.01, anchor=False)

    assert "anchor" in parts_on
    assert not torch.allclose(loss_on, loss_off)
    assert torch.allclose(loss_on, loss_off + parts_on["anchor"], atol=1e-5)


def test_rl_loss_default_anchor_is_true():
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    cost_ref = load_cost_ref()
    lam_idx = _lam_idx(len(eps))

    out = rollout(
        model,
        eps,
        world,
        policy="sample",
        rng=np.random.default_rng(3),
        lam_idx=lam_idx,
        cost_ref=cost_ref,
    )

    baseline = BucketedBaseline()
    loss_default, _ = rl_loss(out.rl, baseline, beta=0.01)
    loss_explicit_true, _ = rl_loss(out.rl, baseline, beta=0.01, anchor=True)

    assert torch.allclose(loss_default, loss_explicit_true)


def test_empty_mask_anchor_contributes_zero():
    n, T = 2, 1
    logp = torch.zeros(n, T, requires_grad=True)
    entropy = torch.zeros(n, T)
    complexity = torch.full((n, T), 0.5, requires_grad=True)
    reward = torch.zeros(n, T, dtype=torch.float32)
    band = torch.zeros(n, T, dtype=torch.long)
    band_target = torch.full((n, T), 0.5, dtype=torch.float32)
    sup_model = torch.zeros(n, T, dtype=torch.float32)
    sup_halt = torch.zeros(n, T, dtype=torch.float32)
    lam_idx = torch.zeros(n, dtype=torch.long)
    choice = torch.zeros(n, T, dtype=torch.bool)
    valid = torch.zeros(n, T, dtype=torch.bool)  # nothing valid at all
    halt_slot = torch.zeros(n, T, dtype=torch.bool)

    from train.rollout import RLTrace

    trace = RLTrace(
        logp=logp,
        entropy=entropy,
        complexity=complexity,
        reward=reward,
        band=band,
        band_target=band_target,
        lam_idx=lam_idx,
        choice=choice,
        valid=valid,
        sup_model=sup_model,
        sup_halt=sup_halt,
        halt_slot=halt_slot,
    )

    baseline = BucketedBaseline()
    loss, parts = rl_loss(trace, baseline, beta=0.01, anchor=True)

    assert torch.isfinite(loss)
    assert parts["anchor"] == pytest.approx(0.0)


def test_rl_update_returns_and_logs_anchor(monkeypatch):
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    cost_ref = load_cost_ref()
    lam_idx = _lam_idx(len(eps))

    out = rollout(
        model,
        eps,
        world,
        policy="sample",
        rng=np.random.default_rng(3),
        lam_idx=lam_idx,
        cost_ref=cost_ref,
    )

    baseline = BucketedBaseline()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    step = rl_update(model, out.rl, baseline, opt, beta=0.01)
    assert "anchor" in step
    assert np.isfinite(step["anchor"])
