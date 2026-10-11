"""Tests for Part 2 Task 3: step environments and episode rewards.

SimStepEnv wraps data.env.Episode so a driver can take one decision at a time
(the same dynamics Episode.play/end_step already implement). LiveStepEnv wraps
execution.engine.ExecutionEngine with a real verifier, using a fake adapter so
no network is touched. episode_log_rewards assigns the per-decision reward
(measurement.rl_reward.step_reward) across a flat decision list, with the
ruling that each step's LAST decision also receives +quality/n.
"""

from __future__ import annotations

import pytest

from controller.contracts import ModelCard, Strategy, TraceItem
from data.env import Episode
from data.routerbench import OutcomeTable
from data.episodes import Skeleton
from execution.adapters.base import AdapterResult, ExecutionAdapter, ExecutionRequest
from execution.catalogue import ResourceCatalogue
from execution.engine import ExecutionEngine
from execution.trace import ExecutionStep, ExecutionTrace
from measurement.live_verifiers import ExactMatchVerifier
from measurement.oracle import weighted_episode_oracle
from measurement.rl_reward import episode_reward, episode_log_rewards, step_reward
from measurement.scorer import CompositeScorer
from measurement.online_env import (
    LiveStep,
    LiveStepEnv,
    RoundOutcome,
    SimStepEnv,
    StepSpec,
)

from tests.test_env import IDS, tiny_table

MODEL_IDS = IDS


def _scorer(cost_ref: float = 1.0) -> CompositeScorer:
    return CompositeScorer({
        "version": 1,
        "cost_ref_usd": cost_ref,
        "cost_ref_model": "test-model",
        "cost_ref_split": "train",
        "cost_ref_statistic": "mean_per_step_cost",
        "train_prompt_count": 1,
        "latency_ref_ms": None,
        "latency_source": "unavailable_in_routerbench",
    })


def _episode(ep_type, correct_rows, budget=100.0, flip_p=0.0) -> Episode:
    t = tiny_table(correct_rows)
    sk = Skeleton("e0", ep_type, tuple(t.prompt_ids), budget)
    return Episode(sk, t, MODEL_IDS, flip_p=flip_p)


# ---------------------------------------------------------------------------
# SimStepEnv
# ---------------------------------------------------------------------------


def test_sim_env_start_and_current_step():
    ep = _episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    env = SimStepEnv(ep, quality_source="truth")
    task_key, task_text, budget_usd, expected_steps = env.start()
    assert task_key == ep.skeleton.prompt_ids[0]
    assert task_text is None
    assert budget_usd == 100.0
    assert expected_steps == 3

    step = env.current_step()
    assert isinstance(step, StepSpec)
    assert step.step_id == f"{ep.skeleton.episode_id}-0"
    assert step.step_key == ep.skeleton.prompt_ids[0]
    assert step.step_text is None
    assert step.step_type == int(ep.table.step_type[ep.row(ep.initial_state())])


def test_sim_env_truth_matches_episode_quality_and_spend():
    """Driving SimStepEnv("truth") with fixed model choices matches Episode.play/end_step."""
    rows = [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]]
    models = [0, 0, 3]

    ep_ref = _episode("random", rows)
    s = ep_ref.initial_state()
    for m in models:
        s = ep_ref.end_step(ep_ref.play(s, m))

    ep = _episode("random", rows)
    env = SimStepEnv(ep, quality_source="truth")
    env.start()
    for m in models:
        step = env.current_step()
        assert step is not None
        outcome = env.play(MODEL_IDS[m], round=1)
        assert outcome.trace_item is not None
        env.end_step()

    assert env.done
    assert env.state.n_correct == s.n_correct
    assert env.state.spent == pytest.approx(s.spent)
    assert env.quality_truth == pytest.approx(s.n_correct / ep.n_steps)
    assert env.current_step() is None


def test_sim_env_truth_quality_matches_correctness():
    ep = _episode("random", [[0, 1, 1, 1]])
    env = SimStepEnv(ep, quality_source="truth")
    env.start()
    env.current_step()
    outcome = env.play(MODEL_IDS[0], round=1)  # model 0 fails this row
    assert outcome.quality == 0.0
    assert outcome.cost_usd == pytest.approx(1.0)


def test_sim_env_verifier_quality_is_noisy_signal():
    """With flip_p=1.0, verifier-sourced step quality is 1 - truth."""
    ep = _episode("random", [[0, 1, 1, 1]], flip_p=1.0)
    env = SimStepEnv(ep, quality_source="verifier")
    env.start()
    env.current_step()
    outcome = env.play(MODEL_IDS[0], round=1)  # true correctness for model 0 is False
    assert outcome.trace_item.verifier_ok is True  # flipped
    assert outcome.quality == 1.0  # verifier source reports noisy signal, not truth


def test_sim_env_unaffordable_play_returns_zero_outcome_and_ends_episode():
    ep = _episode("random", [[1, 1, 1, 1]] * 3, budget=10.0)
    env = SimStepEnv(ep, quality_source="truth")
    env.start()
    env.current_step()
    env.play(MODEL_IDS[3], round=1)  # costs 8
    env.end_step()
    env.current_step()
    outcome = env.play(MODEL_IDS[3], round=1)  # costs 8 > 2 left
    assert outcome.trace_item is None
    assert outcome.quality == 0.0
    assert outcome.cost_usd == 0.0
    assert env.done


# ---------------------------------------------------------------------------
# LiveStepEnv
# ---------------------------------------------------------------------------


def make_card(model_id: str = "model-a") -> ModelCard:
    return ModelCard(model_id=model_id, provider="test", tier=0, mean_cost_usd=0.001,
                     price_source="test", price_date="2026-01-01", capability=[0.5, -3.0])


class FakeAdapter(ExecutionAdapter):
    """Returns fixed text and cost. No network."""

    framework = "fake"
    adapter_version = "v1"

    def __init__(self, output_text: str, cost_usd: float):
        self.output_text = output_text
        self.cost_usd = cost_usd

    def execute(self, request: ExecutionRequest) -> AdapterResult:
        trace = ExecutionTrace(
            execution_id=request.execution_id,
            strategy_step_id=request.strategy.step_id,
            strategy_round=request.strategy.round,
            framework=self.framework,
            catalogue_version=request.catalogue_version,
            adapter_version=self.adapter_version,
            steps=(
                ExecutionStep(step_index=0, parent_step=None, model_id=request.resource.card.model_id,
                              tool=None, input_tokens=1, output_tokens=1, cost_usd=self.cost_usd,
                              wall_ms=0.0, status="ok"),
            ),
            final_status="ok",
        )
        return AdapterResult(output_text=self.output_text, trace=trace)


def make_engine(output_text: str, cost_usd: float, model_id: str = "model-a") -> ExecutionEngine:
    catalogue = ResourceCatalogue.from_model_cards([make_card(model_id)])
    return ExecutionEngine(catalogue, FakeAdapter(output_text, cost_usd))


def test_live_env_with_fake_adapter_exact_match():
    engine = make_engine("Paris", 0.002)
    verifier = ExactMatchVerifier(_scorer())
    steps = [LiveStep(text="What is the capital of France?", step_type=0, gold="Paris")]
    env = LiveStepEnv(engine, steps, verifier, budget_usd=1.0)

    task_key, task_text, budget_usd, expected_steps = env.start()
    assert budget_usd == 1.0
    assert expected_steps == 1

    step = env.current_step()
    assert step.step_text == "What is the capital of France?"
    assert step.step_type == 0

    outcome = env.play("model-a", round=1)
    assert outcome.cost_usd == pytest.approx(0.002)
    assert outcome.quality == 1.0
    assert outcome.trace_item.verifier_ok is True
    assert outcome.trace_item.cost_usd == pytest.approx(0.002)
    assert outcome.trace_item.round == 1
    assert outcome.trace_item.escalated is False
    assert outcome.trace_item.step_index == 0

    env.end_step()
    assert env.done
    assert env.current_step() is None


def test_live_env_quality_mismatch_scores_zero():
    engine = make_engine("Berlin", 0.001)
    verifier = ExactMatchVerifier(_scorer())
    steps = [LiveStep(text="capital of France?", step_type=0, gold="Paris")]
    env = LiveStepEnv(engine, steps, verifier, budget_usd=1.0)
    env.start()
    env.current_step()
    outcome = env.play("model-a", round=1)
    assert outcome.quality == 0.0
    assert outcome.trace_item.verifier_ok is False


def test_live_env_round2_marks_escalated():
    engine = make_engine("Paris", 0.001)
    verifier = ExactMatchVerifier(_scorer())
    steps = [LiveStep(text="capital of France?", step_type=0, gold="Paris")]
    env = LiveStepEnv(engine, steps, verifier, budget_usd=1.0)
    env.start()
    env.current_step()
    outcome = env.play("model-a", round=2)
    assert outcome.trace_item.round == 2
    assert outcome.trace_item.escalated is True


def test_live_env_tracks_budget_and_ends_when_exceeded():
    engine = make_engine("Paris", 0.6)
    verifier = ExactMatchVerifier(_scorer())
    steps = [
        LiveStep(text="q1", step_type=0, gold="Paris"),
        LiveStep(text="q2", step_type=0, gold="Paris"),
    ]
    env = LiveStepEnv(engine, steps, verifier, budget_usd=1.0)
    env.start()
    env.current_step()
    env.play("model-a", round=1)  # spends 0.6, budget_left 0.4
    env.end_step()
    assert not env.done
    step = env.current_step()
    assert step is not None
    env.play("model-a", round=1)  # spends another 0.6: budget_left goes negative
    assert env.done


def test_live_env_step_emb_key_is_stable_hash():
    engine = make_engine("Paris", 0.0)
    verifier = ExactMatchVerifier(_scorer())
    text = "capital of France?"
    steps = [LiveStep(text=text, step_type=0, gold="Paris")]
    env = LiveStepEnv(engine, steps, verifier, budget_usd=1.0)
    env.start()
    env.current_step()
    outcome = env.play("model-a", round=1)
    key1 = outcome.trace_item.step_emb_key

    env2 = LiveStepEnv(engine, steps, verifier, budget_usd=1.0)
    env2.start()
    env2.current_step()
    key2 = env2.play("model-a", round=1).trace_item.step_emb_key
    assert key1 == key2
    assert isinstance(key1, str) and key1


# ---------------------------------------------------------------------------
# episode_log_rewards
# ---------------------------------------------------------------------------


def test_episode_log_rewards_sum_equals_episode_reward():
    cost_ref = 1.0
    lam = 0.5
    n_steps = 3
    scorer = _scorer(cost_ref)

    # 3 steps, step 1 has two decisions (round 1 + round 2 escalation).
    round_costs = [1.0, 2.0, 1.0, 4.0]
    decision_step = [0, 1, 1, 2]
    step_quality = [1.0, 0.0, 1.0]  # per-step quality (e.g. verifier_ok as float)

    rewards = episode_log_rewards(round_costs, step_quality, decision_step, lam, cost_ref)
    assert len(rewards) == len(round_costs)

    expected = episode_reward(
        scorer=scorer,
        quality=sum(step_quality) / n_steps,
        spent=sum(round_costs),
        n_steps=n_steps,
        lam=lam,
    )
    assert sum(rewards) == pytest.approx(expected)


def test_episode_log_rewards_last_decision_per_step_gets_quality_share():
    cost_ref = 1.0
    lam = 0.0  # isolate the quality contribution
    round_costs = [0.0, 0.0]
    decision_step = [0, 1]
    step_quality = [1.0, 0.0]
    n = len(step_quality)

    rewards = episode_log_rewards(round_costs, step_quality, decision_step, lam, cost_ref)
    assert rewards[0] == pytest.approx(step_quality[0] / n)
    assert rewards[1] == pytest.approx(step_quality[1] / n)


def test_episode_log_rewards_only_last_decision_of_a_step_gets_quality():
    cost_ref = 1.0
    lam = 0.0
    # Step 0 has two decisions (round 1 play, round 2 escalate-then-stop).
    round_costs = [1.0, 2.0]
    decision_step = [0, 0]
    step_quality = [1.0]

    rewards = episode_log_rewards(round_costs, step_quality, decision_step, lam, cost_ref)
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(1.0)


def test_episode_log_rewards_matches_per_decision_step_reward_formula():
    """Each decision's reward should equal step_reward(d_quality, d_spent, n, lam, cost_ref)
    for its own round cost, plus the step's quality share on its last decision."""
    cost_ref = 2.0
    lam = 0.3
    round_costs = [1.5, 0.5]
    decision_step = [0, 1]
    step_quality = [1.0, 0.0]
    n = len(step_quality)

    rewards = episode_log_rewards(round_costs, step_quality, decision_step, lam, cost_ref)

    expected0 = step_reward(0.0, round_costs[0], n, lam, cost_ref) + step_quality[0] / n
    expected1 = step_reward(0.0, round_costs[1], n, lam, cost_ref) + step_quality[1] / n
    assert rewards[0] == pytest.approx(expected0)
    assert rewards[1] == pytest.approx(expected1)
