from train.online_runner import run_online_update
from types import SimpleNamespace

import torch

from train.online_learner import OnlineLearner
from train.action_value_heads import ActionValueHeads
from train.replay_buffer import ReplayBuffer

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