import torch

from train.action_value_heads import ActionValueHeads
from train.safety import (
    ValidationGate,
    RollbackManager,
    CanaryEvaluator,
    validate_or_rollback,
)

def test_validation_gate_rejects_large_degradation():
    gate = ValidationGate(
        max_degradation=0.05,
    )

    baseline_score = 0.80
    candidate_score = 0.70

    accepted = gate.accept(
        baseline_score=baseline_score,
        candidate_score=candidate_score,
    )

    assert accepted is False

def test_validation_gate_accepts_small_degradation():
    gate = ValidationGate(
        max_degradation=0.05,
    )

    baseline_score = 0.80
    candidate_score = 0.77

    accepted = gate.accept(
        baseline_score=baseline_score,
        candidate_score=candidate_score,
    )

    assert accepted is True

def test_rollback_restores_previous_weights():
    torch.manual_seed(0)

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    rollback = RollbackManager()

    rollback.capture(heads)

    original_weight = (
        heads.halt.weight
        .detach()
        .clone()
    )

    # Simulate a bad training update.
    with torch.no_grad():
        heads.halt.weight.add_(1.0)

    assert not torch.allclose(
        heads.halt.weight,
        original_weight,
    )

    rollback.restore(heads)

    assert torch.allclose(
        heads.halt.weight,
        original_weight,
    )

def test_failed_validation_automatically_rolls_back():
    torch.manual_seed(0)

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    gate = ValidationGate(
        max_degradation=0.05,
    )

    rollback = RollbackManager()

    rollback.capture(heads)

    original_weight = (
        heads.halt.weight
        .detach()
        .clone()
    )

    # Simulate a candidate training update.
    with torch.no_grad():
        heads.halt.weight.add_(1.0)

    accepted = validate_or_rollback(
        module=heads,
        gate=gate,
        rollback=rollback,
        baseline_score=0.80,
        candidate_score=0.70,
    )

    assert accepted is False

    assert torch.allclose(
        heads.halt.weight,
        original_weight,
    )

def test_successful_validation_keeps_candidate_weights():
    torch.manual_seed(0)

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    gate = ValidationGate(
        max_degradation=0.05,
    )

    rollback = RollbackManager()

    rollback.capture(heads)

    # Simulate a candidate update.
    with torch.no_grad():
        heads.halt.weight.add_(0.1)

    candidate_weight = (
        heads.halt.weight
        .detach()
        .clone()
    )

    accepted = validate_or_rollback(
        module=heads,
        gate=gate,
        rollback=rollback,
        baseline_score=0.80,
        candidate_score=0.78,
    )

    assert accepted is True

    assert torch.allclose(
        heads.halt.weight,
        candidate_weight,
    )

def test_canary_rejects_degraded_candidate_batch():
    gate = ValidationGate(
        max_degradation=0.05,
    )

    evaluator = CanaryEvaluator(
        gate=gate,
    )

    baseline_scores = [
        0.80,
        0.82,
        0.78,
    ]

    candidate_scores = [
        0.68,
        0.70,
        0.72,
    ]

    result = evaluator.evaluate(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
    )

    assert result["accepted"] is False

    assert result["baseline_mean"] > result["candidate_mean"]

def test_canary_accepts_good_candidate_batch():
    gate = ValidationGate(
        max_degradation=0.05,
    )

    evaluator = CanaryEvaluator(
        gate=gate,
    )

    baseline_scores = [
        0.80,
        0.82,
        0.78,
    ]

    candidate_scores = [
        0.79,
        0.81,
        0.80,
    ]

    result = evaluator.evaluate(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
    )

    assert result["accepted"] is True

    assert result["candidate_mean"] >= 0.79

    assert result["n_canary"] == 3

def test_failed_canary_rolls_back_candidate_weights():
    torch.manual_seed(0)

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    gate = ValidationGate(
        max_degradation=0.05,
    )

    evaluator = CanaryEvaluator(
        gate=gate,
    )

    rollback = RollbackManager()
    rollback.capture(heads)

    original_weight = (
        heads.halt.weight
        .detach()
        .clone()
    )

    # Simulate candidate update.
    with torch.no_grad():
        heads.halt.weight.add_(1.0)

    baseline_scores = [
        0.80,
        0.82,
        0.78,
    ]

    candidate_scores = [
        0.68,
        0.70,
        0.72,
    ]

    result = evaluator.evaluate(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
    )

    accepted = validate_or_rollback(
        module=heads,
        gate=gate,
        rollback=rollback,
        baseline_score=result["baseline_mean"],
        candidate_score=result["candidate_mean"],
    )

    assert accepted is False

    assert torch.allclose(
        heads.halt.weight,
        original_weight,
    )