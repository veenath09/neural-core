import torch

from train.action_value_heads import ActionValueHeads
from train.online_learner import OnlineLearner
from train.rollout import rollout
from tests.test_rollout import world_and_episodes, tiny_model


def test_action_value_heads_output_shapes():
    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    z_h = torch.randn(3, 128)
    cards = torch.randn(4, 6)

    values = heads(
        z_h=z_h,
        cards=cards,
    )

    assert set(values.keys()) == {"model", "halt"}

    assert values["model"].shape == (3, 4)
    assert values["halt"].shape == (3, 2)


def test_model_value_head_supports_dynamic_model_pool():
    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    z_h = torch.randn(2, 128)

    cards_4 = torch.randn(4, 6)
    cards_5 = torch.randn(5, 6)

    values_4 = heads(
        z_h=z_h,
        cards=cards_4,
    )

    values_5 = heads(
        z_h=z_h,
        cards=cards_5,
    )

    assert values_4["model"].shape == (2, 4)
    assert values_5["model"].shape == (2, 5)

    def test_action_value_heads_support_gradients():
        heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    z_h = torch.randn(3, 128)
    cards = torch.randn(4, 6)

    values = heads(
        z_h=z_h,
        cards=cards,
    )

    loss = (
        values["model"].mean()
        + values["halt"].mean()
    )

    loss.backward()

    assert heads.need.weight.grad is not None
    assert heads.model_match.weight.grad is not None
    assert heads.halt.weight.grad is not None
    assert heads.card[0].weight.grad is not None



def test_online_learner_trains_real_action_value_heads():
    torch.manual_seed(0)

    learner = OnlineLearner()

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.05,
    )

    states = torch.randn(4, 128)

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

    cards = torch.randn(4, 6)

    choose_batch, halt_batch = learner.extract_action_batches(
        states=states,
        actions=actions,
        returns=returns,
    )

    model_before = heads.model_match.weight.detach().clone()
    halt_before = heads.halt.weight.detach().clone()

    loss = learner.train_action_value_step(
        heads=heads,
        optimizer=optimizer,
        cards=cards,
        choose_batch=choose_batch,
        halt_batch=halt_batch,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0

    assert not torch.equal(
        model_before,
        heads.model_match.weight.detach(),
    )

    assert not torch.equal(
        halt_before,
        heads.halt.weight.detach(),
    )

def test_online_learner_trains_from_real_rollout():
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

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.05,
    )

    # Temporary external rewards.
    # Later these will come from C3 measurement.
    rewards = [0.8] * len(rollout_output.results)

    model_before = heads.model_match.weight.detach().clone()

    result = learner.train_from_rollout(
        rollout_output=rollout_output,
        rewards=rewards,
        cards=controller.cards,
        heads=heads,
        optimizer=optimizer,
    )

    assert isinstance(result["loss"], float)

    assert result["episodes"] == len(episodes)

    assert result["decisions"] == sum(
        len(actions)
        for actions in rollout_output.decision_actions
    )

    assert learner.episode_count == len(episodes)

    assert not torch.equal(
        model_before,
        heads.model_match.weight.detach(),
    )