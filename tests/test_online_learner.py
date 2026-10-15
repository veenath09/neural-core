import pytest
import torch

from train.online_learner import OnlineLearner
from train.branching_heads import BranchingValueHeads


def test_learner_updates_episode_count():
    learner = OnlineLearner()

    result = learner.update(0.8)

    assert result["episode"] == 1
    assert result["reward"] == 0.8
    assert result["returns"] == [0.8]


def test_monte_carlo_returns_same_reward_when_gamma_is_one():
    learner = OnlineLearner(gamma=1.0)

    returns = learner.monte_carlo_returns(
        reward=0.8,
        n_decisions=3,
    )

    assert returns == [0.8, 0.8, 0.8]


def test_monte_carlo_returns_are_discounted():
    learner = OnlineLearner(gamma=0.9)

    returns = learner.monte_carlo_returns(
        reward=1.0,
        n_decisions=3,
    )

    assert returns == pytest.approx([0.81, 0.9, 1.0])


def test_monte_carlo_returns_reject_zero_decisions():
    learner = OnlineLearner()

    with pytest.raises(ValueError):
        learner.monte_carlo_returns(
            reward=1.0,
            n_decisions=0,
        )

def test_monte_carlo_loss_matches_observed_return():
    learner = OnlineLearner()

    predicted = torch.tensor([0.5, 0.7])
    returns = torch.tensor([0.8, 0.8])

    loss = learner.monte_carlo_loss(predicted, returns)

    expected = ((0.5 - 0.8) ** 2 + (0.7 - 0.8) ** 2) / 2

    assert loss.item() == pytest.approx(expected)




def test_monte_carlo_loss_updates_branching_heads():
    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
        },
    )

    learner = OnlineLearner()

    z_h = torch.randn(2, 128)
    values = heads(z_h)

    actions = {
        "memory": torch.tensor([0, 2]),
        "depth": torch.tensor([1, 0]),
    }

    returns = torch.tensor([0.8, 0.4])

    loss = learner.branching_loss(
        values=values,
        actions=actions,
        returns=returns,
    )

    loss.backward()

    assert loss.requires_grad

    for parameter in heads.parameters():
        assert parameter.grad is not None

def test_online_step_updates_branching_head_weights():
    torch.manual_seed(0)

    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
        },
    )

    learner = OnlineLearner()

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.1,
    )

    z_h = torch.randn(2, 128)

    actions = {
        "memory": torch.tensor([0, 2]),
        "depth": torch.tensor([1, 0]),
    }

    returns = torch.tensor([0.8, 0.4])

    before = [
        parameter.detach().clone()
        for parameter in heads.parameters()
    ]

    loss = learner.train_branching_step(
        heads=heads,
        optimizer=optimizer,
        z_h=z_h,
        actions=actions,
        returns=returns,
    )

    after = list(heads.parameters())

    changed = any(
        not torch.allclose(old, new)
        for old, new in zip(before, after)
    )

    assert changed
    assert loss >= 0.0