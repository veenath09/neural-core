from train.online_runner import (
    run_online_update,
    run_online_update_from_scores,
)
from types import SimpleNamespace

import torch
from measurement.scorer import Score
from train.online_learner import OnlineLearner
from train.action_value_heads import ActionValueHeads
from train.replay_buffer import ReplayBuffer

from train.safety import (
    ValidationGate,
    RollbackManager,
)

class FakeReplayBuffer:
    def __len__(self):
        return 2


class FakeLearner:
    def __init__(self):
        self.calls = []

    def train_from_rollout(
        self,
        rollout_output,
        rewards,
        cards,
        heads,
        optimizer,
        anchor_lambda=0.0,
    ):
        self.calls.append("fresh")

        return {
            "loss": 0.5,
            "episodes": 1,
            "decisions": 2,
        }

    def train_from_replay(
        self,
        replay_buffer,
        sample_size,
        cards,
        heads,
        optimizer,
        anchor_lambda=0.0,
    ):
        self.calls.append("replay")

        return {
            "loss": 0.4,
            "replay_episodes": 2,
            "decisions": 4,
        }

    def store_rollout_in_replay(
        self,
        rollout_output,
        rewards,
        replay_buffer,
    ):
        self.calls.append("store")
        return 1


def test_online_update_trains_old_replay_before_storing_fresh_rollout():
    learner = FakeLearner()
    replay_buffer = FakeReplayBuffer()

    result = run_online_update(
        learner=learner,
        rollout_output=object(),
        rewards=[0.8],
        cards=object(),
        heads=object(),
        optimizer=object(),
        replay_buffer=replay_buffer,
        replay_sample_size=4,
    )

    assert learner.calls == [
        "fresh",
        "replay",
        "store",
    ]

    assert result["fresh"]["episodes"] == 1
    assert result["replay"]["replay_episodes"] == 2
    assert result["stored_episodes"] == 1


class EmptyReplayBuffer:
    def __len__(self):
        return 0


def test_online_update_skips_replay_when_buffer_is_empty():
    learner = FakeLearner()
    replay_buffer = EmptyReplayBuffer()

    result = run_online_update(
        learner=learner,
        rollout_output=object(),
        rewards=[0.8],
        cards=object(),
        heads=object(),
        optimizer=object(),
        replay_buffer=replay_buffer,
        replay_sample_size=4,
    )

    assert learner.calls == [
        "fresh",
        "store",
    ]

    assert result["fresh"]["episodes"] == 1
    assert result["replay"] is None
    assert result["stored_episodes"] == 1

def test_online_update_uses_learner_episode_count_once():
    torch.manual_seed(0)

    learner = OnlineLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    replay_buffer = ReplayBuffer(
        capacity=10,
    )

    rollout_output = SimpleNamespace(
        results=[
            SimpleNamespace(
                ep_type="domain_a",
            )
        ],
        decision_states=[
            [
                torch.randn(128),
                torch.randn(128),
            ]
        ],
        decision_actions=[
            [
                {
                    "phase": "choose",
                    "model": 0,
                    "halt": None,
                },
                {
                    "phase": "halt",
                    "model": None,
                    "halt": "stop",
                },
            ]
        ],
    )

    cards = torch.randn(3, 6)

    result = run_online_update(
        learner=learner,
        rollout_output=rollout_output,
        rewards=[0.8],
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        replay_sample_size=4,
    )

    assert learner.episode_count == 1
    assert len(replay_buffer) == 1

    assert result["fresh"]["episodes"] == 1
    assert result["replay"] is None
    assert result["stored_episodes"] == 1


class MutatingLearner(FakeLearner):
    def train_from_rollout(
        self,
        rollout_output,
        rewards,
        cards,
        heads,
        optimizer,
        anchor_lambda=0.0,
    ):
        self.calls.append("fresh")

        # Simulate a candidate model update.
        with torch.no_grad():
            heads.halt.weight.add_(1.0)

        return {
            "loss": 0.5,
            "episodes": 1,
            "decisions": 2,
        }


def test_online_update_rolls_back_rejected_candidate():
    learner = MutatingLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.SGD(
    heads.parameters(),
    lr=0.01,
    )

    original_weight = (
        heads.halt.weight
        .detach()
        .clone()
    )

    gate = ValidationGate(
        max_degradation=0.05,
    )

    rollback = RollbackManager()

    scores = iter([
        0.80,  # baseline before training
        0.70,  # candidate after training
    ])

    def validation_fn(module):
        return next(scores)

    result = run_online_update(
        learner=learner,
        rollout_output=object(),
        rewards=[0.8],
        cards=object(),
        heads=heads,
        optimizer=optimizer,
        replay_buffer=EmptyReplayBuffer(),
        replay_sample_size=4,
        validation_gate=gate,
        rollback_manager=rollback,
        validation_fn=validation_fn,
    )

    assert result["validation"]["accepted"] is False
    assert result["validation"]["baseline_score"] == 0.80
    assert result["validation"]["candidate_score"] == 0.70

    assert torch.allclose(
        heads.halt.weight,
        original_weight,
    )


def test_online_update_keeps_accepted_candidate():
    learner = MutatingLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.SGD(
    heads.parameters(),
    lr=0.01,
    )

    original_weight = (
        heads.halt.weight
        .detach()
        .clone()
    )

    gate = ValidationGate(
        max_degradation=0.05,
    )

    rollback = RollbackManager()

    scores = iter([
        0.80,  # baseline before training
        0.78,  # candidate after training
    ])

    def validation_fn(module):
        return next(scores)

    result = run_online_update(
        learner=learner,
        rollout_output=object(),
        rewards=[0.8],
        cards=object(),
        heads=heads,
        optimizer=optimizer,
        replay_buffer=EmptyReplayBuffer(),
        replay_sample_size=4,
        validation_gate=gate,
        rollback_manager=rollback,
        validation_fn=validation_fn,
    )

    assert result["validation"]["accepted"] is True
    assert result["validation"]["baseline_score"] == 0.80
    assert result["validation"]["candidate_score"] == 0.78

    assert not torch.allclose(
        heads.halt.weight,
        original_weight,
    )

class OptimizerMutatingLearner(FakeLearner):
    def train_from_rollout(
        self,
        rollout_output,
        rewards,
        cards,
        heads,
        optimizer,
        anchor_lambda=0.0,
    ):
        self.calls.append("fresh")

        optimizer.zero_grad()

        loss = heads.halt.weight.sum()
        loss.backward()

        optimizer.step()

        return {
            "loss": float(loss.detach()),
            "episodes": 1,
            "decisions": 1,
        }


def test_rejected_candidate_restores_optimizer_state():
    learner = OptimizerMutatingLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.Adam(
        heads.parameters(),
        lr=0.01,
    )

    assert len(optimizer.state) == 0

    gate = ValidationGate(
        max_degradation=0.05,
    )

    rollback = RollbackManager()

    scores = iter([
        0.80,  # baseline
        0.60,  # rejected candidate
    ])

    def validation_fn(module):
        return next(scores)

    result = run_online_update(
        learner=learner,
        rollout_output=object(),
        rewards=[0.8],
        cards=object(),
        heads=heads,
        optimizer=optimizer,
        replay_buffer=EmptyReplayBuffer(),
        replay_sample_size=4,
        validation_gate=gate,
        rollback_manager=rollback,
        validation_fn=validation_fn,
    )

    assert result["validation"]["accepted"] is False

    # A rejected candidate must not leave Adam's
    # momentum/variance state behind.
    assert len(optimizer.state) == 0


class RewardCapturingLearner(FakeLearner):
    def __init__(self):
        super().__init__()
        self.seen_rewards = None

    def train_from_rollout(
        self,
        rollout_output,
        rewards,
        cards,
        heads,
        optimizer,
        anchor_lambda=0.0,
    ):
        self.seen_rewards = list(rewards)

        return super().train_from_rollout(
            rollout_output=rollout_output,
            rewards=rewards,
            cards=cards,
            heads=heads,
            optimizer=optimizer,
            anchor_lambda=anchor_lambda,
        )


def test_online_update_from_scores_passes_c3_rewards_to_learner():
    learner = RewardCapturingLearner()

    rollout_output = SimpleNamespace(
        results=[
            object(),
            object(),
        ]
    )

    scores = [
        Score(
            quality=0.8,
            cost_usd=0.01,
            latency_ms=None,
            penalty=0.0,
            lam=0.0,
            mu=0.0,
            cost_ref_usd=0.01,
            latency_ref_ms=None,
            normalized_cost=1.0,
            normalized_latency=None,
            reward=0.75,
            latency_included=False,
        ),
        Score(
            quality=0.7,
            cost_usd=0.01,
            latency_ms=None,
            penalty=0.0,
            lam=0.0,
            mu=0.0,
            cost_ref_usd=0.01,
            latency_ref_ms=None,
            normalized_cost=1.0,
            normalized_latency=None,
            reward=-0.10,
            latency_included=False,
        ),
    ]

    run_online_update_from_scores(
        learner=learner,
        rollout_output=rollout_output,
        scores=scores,
        cards=object(),
        heads=object(),
        optimizer=object(),
        replay_buffer=EmptyReplayBuffer(),
        replay_sample_size=4,
    )

    assert learner.seen_rewards == [
        0.75,
        -0.10,
    ]