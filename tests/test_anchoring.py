import torch

from train.action_value_heads import ActionValueHeads
from train.online_learner import OnlineLearner
from train.replay_buffer import ReplayBuffer
from train.rollout import rollout
from tests.test_rollout import world_and_episodes, tiny_model


def test_anchor_penalty_detects_weight_drift():
    torch.manual_seed(0)

    learner = OnlineLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    # Save the current stable policy as the anchor.
    learner.set_anchor(heads)

    # Immediately after anchoring, there should be no drift.
    penalty_before = learner.anchor_penalty(heads)

    assert torch.isclose(
        penalty_before,
        torch.tensor(0.0),
    )

    # Simulate learning that moves one set of weights.
    with torch.no_grad():
        heads.halt.weight.add_(0.1)

    penalty_after = learner.anchor_penalty(heads)

    assert penalty_after.item() > 0.0

def test_anchor_penalty_is_added_to_training_loss():
    torch.manual_seed(0)

    learner = OnlineLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    learner.set_anchor(heads)

    # Move weights away from the stable anchor.
    with torch.no_grad():
        heads.halt.weight.add_(0.1)

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.0,
    )

    cards = torch.randn(4, 6)

    choose_batch = {
        "states": torch.randn(2, 128),
        "actions": torch.tensor([1, 2]),
        "returns": torch.tensor([0.8, 0.6]),
    }

    halt_batch = {
        "states": torch.randn(2, 128),
        "actions": torch.tensor([0, 1]),
        "returns": torch.tensor([0.7, 0.5]),
    }

    loss_without_anchor = learner.train_action_value_step(
        heads=heads,
        optimizer=optimizer,
        cards=cards,
        choose_batch=choose_batch,
        halt_batch=halt_batch,
        anchor_lambda=0.0,
    )

    loss_with_anchor = learner.train_action_value_step(
        heads=heads,
        optimizer=optimizer,
        cards=cards,
        choose_batch=choose_batch,
        halt_batch=halt_batch,
        anchor_lambda=1.0,
    )

    assert loss_with_anchor > loss_without_anchor



def test_replay_training_accepts_anchor_lambda():
    torch.manual_seed(0)

    learner = OnlineLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    learner.set_anchor(heads)

    replay_buffer = ReplayBuffer(
        capacity=10,
        per_domain_capacity=5,
    )

    replay_buffer.add({
        "states": torch.randn(2, 128),
        "actions": [
            {"phase": "choose", "model": 1, "halt": None},
            {"phase": "halt", "model": None, "halt": "stop"},
        ],
        "returns": torch.tensor([0.8, 0.8]),
        "domain": "domain_a",
        "age": 2,
    })

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    cards = torch.randn(4, 6)

    result = learner.train_from_replay(
        replay_buffer=replay_buffer,
        sample_size=1,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        anchor_lambda=0.5,
    )

    assert isinstance(result["loss"], float)
    assert result["replay_episodes"] == 1

def test_rollout_training_accepts_anchor_lambda():
    torch.manual_seed(0)

    world, episodes = world_and_episodes(4)

    controller = tiny_model("full").eval()

    with torch.no_grad():
        rollout_output = rollout(
            controller,
            episodes,
            world,
        )

    learner = OnlineLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=controller.cards.shape[1],
    )

    learner.set_anchor(heads)

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    rewards = [0.8] * len(rollout_output.results)

    result = learner.train_from_rollout(
        rollout_output=rollout_output,
        rewards=rewards,
        cards=controller.cards,
        heads=heads,
        optimizer=optimizer,
        anchor_lambda=0.5,
    )

    assert isinstance(result["loss"], float)
    assert result["episodes"] == len(episodes)


def test_consolidation_resets_anchor_after_training():
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

    learner.set_anchor(heads)

    # Create some weight drift first.
    with torch.no_grad():
        heads.halt.weight.add_(0.1)

    penalty_before = learner.anchor_penalty(heads)

    assert penalty_before.item() > 0.0

    replay_buffer = ReplayBuffer(
        capacity=10
    )

    replay_buffer.add(
        {
            "states": torch.randn(2, 128),
            "actions": [
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
            ],
            "returns": torch.tensor(
                [0.8, 0.8]
            ),
            "domain": "domain_a",
            "age": 1,
        }
    )

    replay_buffer.add(
        {
            "states": torch.randn(2, 128),
            "actions": [
                {
                    "phase": "choose",
                    "model": 1,
                    "halt": None,
                },
                {
                    "phase": "halt",
                    "model": 1,
                    "halt": "escalate",
                },
            ],
            "returns": torch.tensor(
                [0.6, 0.6]
            ),
            "domain": "domain_b",
            "age": 4,
        }
    )

    cards = torch.randn(4, 6)

    result = learner.consolidate(
        replay_buffer=replay_buffer,
        sample_size=2,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        anchor_lambda=0.5,
    )

    penalty_after = learner.anchor_penalty(
        heads
    )

    assert isinstance(
        result["loss"],
        float,
    )

    assert result["replay_episodes"] == 2

    assert result["anchor_reset"] is True

    assert torch.isclose(
        penalty_after,
        torch.tensor(0.0),
        atol=1e-8,
    )


def test_checkpoint_restores_anchor_state(tmp_path):
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

    # Save the stable parameters as the anchor.
    learner.set_anchor(heads)

    # Simulate later learning that moves the current weights.
    with torch.no_grad():
        heads.halt.weight.add_(0.1)

    checkpoint_path = tmp_path / "learner_checkpoint.pt"

    learner.save_checkpoint(
        checkpoint_path,
        heads,
        optimizer,
    )

    restored_learner = OnlineLearner()

    restored_heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    restored_optimizer = torch.optim.SGD(
        restored_heads.parameters(),
        lr=0.01,
    )

    restored_learner.load_checkpoint(
        checkpoint_path,
        restored_heads,
        restored_optimizer,
    )

    assert restored_learner.anchor_state is not None

    penalty = restored_learner.anchor_penalty(
        restored_heads
    )

    assert penalty.item() > 0.0


def test_consolidation_aligns_replay_tensors_with_heads():
    learner = OnlineLearner()

    replay_buffer = ReplayBuffer(
        capacity=10,
        per_domain_capacity=5,
    )

    replay_buffer.add({
        "states": torch.randn(2, 128),
        "actions": [
            {"phase": "choose", "model": 1, "halt": None},
            {"phase": "halt", "model": None, "halt": "stop"},
        ],
        "returns": torch.tensor([0.8, 0.8]),
        "domain": "domain_a",
        "age": 3,
    })

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    ).double()

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    cards = torch.randn(
        4,
        6,
        dtype=torch.float64,
    )

    result = learner.consolidate(
        replay_buffer=replay_buffer,
        sample_size=1,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
    )

    assert isinstance(result["loss"], float)
    assert result["anchor_reset"] is True

    