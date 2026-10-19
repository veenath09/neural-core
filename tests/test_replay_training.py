import torch

from train.online_learner import OnlineLearner
from train.replay_buffer import ReplayBuffer
from train.rollout import rollout
from tests.test_rollout import world_and_episodes, tiny_model
from train.action_value_heads import ActionValueHeads


def test_online_learner_stores_real_rollout_in_replay():
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

    replay_buffer = ReplayBuffer(
        capacity=10,
        per_domain_capacity=5,
    )

    # Temporary rewards.
    # Later these will come from C3.
    rewards = [0.8] * len(rollout_output.results)

    stored_count = learner.store_rollout_in_replay(
        rollout_output=rollout_output,
        rewards=rewards,
        replay_buffer=replay_buffer,
    )

    assert stored_count == len(rollout_output.results)
    assert len(replay_buffer) == len(rollout_output.results)

    stored = replay_buffer.sample(
        len(rollout_output.results)
    )

    for index, episode in enumerate(stored):
        assert episode["domain"] == rollout_output.results[index].ep_type
        assert episode["age"] == 0

        assert episode["states"].shape[1] == 128

        assert len(episode["actions"]) == len(
            episode["states"]
        )

        assert len(episode["returns"]) == len(
            episode["states"]
        )

def test_online_learner_trains_from_replay_buffer():
    torch.manual_seed(0)

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
        "age": 2,
    })

    replay_buffer.add({
        "states": torch.randn(2, 128),
        "actions": [
            {"phase": "choose", "model": 2, "halt": None},
            {"phase": "halt", "model": 3, "halt": "escalate"},
        ],
        "returns": torch.tensor([0.6, 0.6]),
        "domain": "domain_b",
        "age": 6,
    })

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.05,
    )

    cards = torch.randn(4, 6)

    model_before = heads.model_match.weight.detach().clone()
    halt_before = heads.halt.weight.detach().clone()

    result = learner.train_from_replay(
        replay_buffer=replay_buffer,
        sample_size=2,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
    )

    assert isinstance(result["loss"], float)
    assert result["replay_episodes"] == 2
    assert result["decisions"] == 4

    assert not torch.equal(
        model_before,
        heads.model_match.weight.detach(),
    )

    assert not torch.equal(
        halt_before,
        heads.halt.weight.detach(),
    )