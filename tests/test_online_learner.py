import pytest

from train.online_learner import OnlineLearner


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