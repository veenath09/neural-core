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


@pytest.mark.parametrize(
    "gamma",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_online_learner_rejects_invalid_gamma(gamma):
    with pytest.raises(
        ValueError,
        match="gamma must be a finite value between 0 and 1",
    ):
        OnlineLearner(gamma=gamma)


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

def test_branching_loss_rejects_mismatched_batch_sizes():
    learner = OnlineLearner()

    values = {
        "memory": torch.randn(2, 3),
    }

    actions = {
        "memory": torch.tensor([0]),
    }

    returns = torch.tensor([0.8, 0.4])

    with pytest.raises(
        ValueError,
        match="branch batch sizes must match returns",
    ):
        learner.branching_loss(
            values=values,
            actions=actions,
            returns=returns,
        )


def test_branching_loss_rejects_out_of_range_action_indices():
    learner = OnlineLearner()

    values = {
        "memory": torch.randn(2, 3),
    }

    actions = {
        "memory": torch.tensor([0, 3]),
    }

    returns = torch.tensor([0.8, 0.4])

    with pytest.raises(
        ValueError,
        match="branch action indices must be within range",
    ):
        learner.branching_loss(
            values=values,
            actions=actions,
            returns=returns,
        )

def test_branching_loss_rejects_invalid_branch_value_rank():
    learner = OnlineLearner()

    values = {
        "memory": torch.randn(2),
    }

    actions = {
        "memory": torch.tensor([0, 0]),
    }

    returns = torch.tensor([0.8, 0.4])

    with pytest.raises(
        ValueError,
        match="branch values must be a 2D tensor",
    ):
        learner.branching_loss(
            values=values,
            actions=actions,
            returns=returns,
        )

def test_branching_loss_rejects_invalid_action_rank():
    learner = OnlineLearner()

    values = {
        "memory": torch.randn(2, 3),
    }

    actions = {
        "memory": torch.tensor([
            [0],
            [1],
        ]),
    }

    returns = torch.tensor([0.8, 0.4])

    with pytest.raises(
        ValueError,
        match="branch actions must be a 1D tensor",
    ):
        learner.branching_loss(
            values=values,
            actions=actions,
            returns=returns,
        )

def test_branching_loss_rejects_non_integer_actions():
    learner = OnlineLearner()

    values = {
        "memory": torch.randn(2, 3),
    }

    actions = {
        "memory": torch.tensor(
            [0.0, 1.0],
            dtype=torch.float32,
        ),
    }

    returns = torch.tensor([0.8, 0.4])

    with pytest.raises(
        ValueError,
        match="branch actions must use an integer dtype",
    ):
        learner.branching_loss(
            values=values,
            actions=actions,
            returns=returns,
        )

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

def test_prepare_episode_links_states_to_observed_reward():
    learner = OnlineLearner(gamma=1.0)

    decision_states = [
        torch.randn(128),
        torch.randn(128),
        torch.randn(128),
    ]

    states, returns = learner.prepare_episode(
        decision_states=decision_states,
        reward=0.8,
    )

    assert states.shape == (3, 128)
    assert returns.shape == (3,)
    assert returns.tolist() == pytest.approx([0.8, 0.8, 0.8])

def test_prepare_rollout_links_each_episode_to_its_reward():
    learner = OnlineLearner(gamma=1.0)

    decision_states = [
        [
            torch.randn(128),
            torch.randn(128),
        ],
        [
            torch.randn(128),
        ],
    ]

    rewards = [0.8, 0.4]

    states, returns = learner.prepare_rollout(
        decision_states=decision_states,
        rewards=rewards,
    )

    assert states.shape == (3, 128)
    assert returns.shape == (3,)
    assert returns.tolist() == pytest.approx([
        0.8,
        0.8,
        0.4,
    ])


def test_prepare_rollout_batch_aligns_states_actions_and_returns():
    learner = OnlineLearner(gamma=1.0)

    decision_states = [
        [
            torch.randn(128),
            torch.randn(128),
        ],
        [
            torch.randn(128),
        ],
    ]

    decision_actions = [
        [
            {
                "phase": "choose",
                "model": 1,
                "halt": None,
            },
            {
                "phase": "halt",
                "model": None,
                "halt": "stop",
            },
        ],
        [
            {
                "phase": "choose",
                "model": 2,
                "halt": None,
            },
        ],
    ]

    rewards = [0.8, 0.4]

    states, actions, returns = learner.prepare_rollout_batch(
        decision_states=decision_states,
        decision_actions=decision_actions,
        rewards=rewards,
    )

    assert states.shape == (3, 128)
    assert len(actions) == 3
    assert returns.shape == (3,)

    assert actions[0]["phase"] == "choose"
    assert actions[0]["model"] == 1

    assert actions[1]["phase"] == "halt"
    assert actions[1]["halt"] == "stop"

    assert actions[2]["model"] == 2

    assert returns.tolist() == pytest.approx([
        0.8,
        0.8,
        0.4,
    ])

def test_split_actions_separates_choose_and_halt():
    learner = OnlineLearner()

    actions = [
        {"phase": "choose", "model": 1, "halt": None},
        {"phase": "halt", "model": None, "halt": "stop"},
        {"phase": "choose", "model": 2, "halt": None},
        {"phase": "halt", "model": 3, "halt": "escalate"},
    ]

    choose, halt = learner.split_actions(actions)

    assert choose == [
        (0, 1),
        (2, 2),
    ]

    assert halt == [
        (1, 0),   # stop
        (3, 1),   # escalate
    ]

def test_extract_action_batches():
    learner = OnlineLearner()

    states = torch.arange(
        4 * 128,
        dtype=torch.float32,
    ).reshape(4, 128)

    actions = [
        {"phase": "choose", "model": 1, "halt": None},
        {"phase": "halt", "model": None, "halt": "stop"},
        {"phase": "choose", "model": 2, "halt": None},
        {"phase": "halt", "model": 3, "halt": "escalate"},
    ]

    returns = torch.tensor(
        [1.0, 0.8, 0.6, 0.4],
        dtype=torch.float32,
    )

    choose_batch, halt_batch = learner.extract_action_batches(
        states=states,
        actions=actions,
        returns=returns,
    )

    assert choose_batch["states"].shape == (2, 128)
    assert torch.equal(
        choose_batch["states"],
        states[[0, 2]],
    )
    assert torch.equal(
        choose_batch["actions"],
        torch.tensor([1, 2]),
    )
    assert torch.equal(
        choose_batch["returns"],
        torch.tensor([1.0, 0.6]),
    )

    assert halt_batch["states"].shape == (2, 128)
    assert torch.equal(
        halt_batch["states"],
        states[[1, 3]],
    )
    assert torch.equal(
        halt_batch["actions"],
        torch.tensor([0, 1]),
    )
    assert torch.equal(
        halt_batch["returns"],
        torch.tensor([0.8, 0.4]),
    )