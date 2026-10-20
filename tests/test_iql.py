import torch

from train.iql import (
    expectile_loss,
    advantage_weights,
    weighted_behavior_cloning_loss,
    iql_loss_components,
    SparseIQLHeads,
    train_sparse_iql_step,
)

def test_expectile_loss_penalizes_underestimation_more():
    prediction = torch.tensor([
        0.2,
        0.8,
    ])

    target = torch.tensor([
        0.8,
        0.2,
    ])

    loss = expectile_loss(
        prediction=prediction,
        target=target,
        expectile=0.7,
    )

    assert loss.item() > 0.0


def test_expectile_loss_penalizes_underestimation_more_than_overestimation():
    under_prediction = torch.tensor([0.2])
    under_target = torch.tensor([0.8])

    over_prediction = torch.tensor([0.8])
    over_target = torch.tensor([0.2])

    under_loss = expectile_loss(
        prediction=under_prediction,
        target=under_target,
        expectile=0.7,
    )

    over_loss = expectile_loss(
        prediction=over_prediction,
        target=over_target,
        expectile=0.7,
    )

    assert under_loss.item() > over_loss.item()

def test_advantage_weights_favour_better_actions():
    q_values = torch.tensor([
        0.9,
        0.4,
    ])

    v_values = torch.tensor([
        0.5,
        0.5,
    ])

    weights = advantage_weights(
        q_values=q_values,
        v_values=v_values,
        beta=2.0,
    )

    assert weights.shape == q_values.shape

    # First action has positive advantage:
    # 0.9 - 0.5 = +0.4
    #
    # Second action has negative advantage:
    # 0.4 - 0.5 = -0.1
    assert weights[0].item() > weights[1].item()

def test_weighted_behavior_cloning_loss_is_positive():
    logits = torch.tensor([
        [2.0, 0.5, 0.1],
        [0.2, 1.5, 0.3],
    ])

    actions = torch.tensor([
        0,
        1,
    ])

    weights = torch.tensor([
        2.0,
        0.5,
    ])

    loss = weighted_behavior_cloning_loss(
        logits=logits,
        actions=actions,
        weights=weights,
    )

    assert loss.item() > 0.0

def test_weighted_behavior_cloning_gives_more_importance_to_high_weight_sample():
    logits = torch.tensor([
        [0.1, 2.0],
        [2.0, 0.1],
    ])

    actions = torch.tensor([
        0,
        1,
    ])

    low_weights = torch.tensor([
        1.0,
        1.0,
    ])

    high_weights = torch.tensor([
        5.0,
        1.0,
    ])

    low_weight_loss = weighted_behavior_cloning_loss(
        logits=logits,
        actions=actions,
        weights=low_weights,
    )

    high_weight_loss = weighted_behavior_cloning_loss(
        logits=logits,
        actions=actions,
        weights=high_weights,
    )

    assert high_weight_loss.item() > low_weight_loss.item()

def test_iql_loss_components_are_computed_for_logged_actions():
    q_values = torch.tensor(
        [
            [0.2, 0.8, 0.4],
            [0.7, 0.3, 0.5],
        ],
        requires_grad=True,
    )

    value_predictions = torch.tensor(
        [0.5, 0.4],
        requires_grad=True,
    )

    policy_logits = torch.tensor(
        [
            [0.1, 1.5, 0.2],
            [1.2, 0.3, 0.1],
        ],
        requires_grad=True,
    )

    actions = torch.tensor([
        1,
        0,
    ])

    returns = torch.tensor([
        0.9,
        0.8,
    ])

    losses = iql_loss_components(
        q_values=q_values,
        value_predictions=value_predictions,
        policy_logits=policy_logits,
        actions=actions,
        returns=returns,
        expectile=0.7,
        beta=3.0,
    )

    assert losses["q_loss"].item() >= 0.0
    assert losses["value_loss"].item() >= 0.0
    assert losses["policy_loss"].item() >= 0.0

    total_loss = (
        losses["q_loss"]
        + losses["value_loss"]
        + losses["policy_loss"]
    )

    total_loss.backward()

    assert q_values.grad is not None
    assert value_predictions.grad is not None
    assert policy_logits.grad is not None


def test_sparse_iql_heads_output_shapes():
    torch.manual_seed(0)

    heads = SparseIQLHeads(
        state_dim=128,
        num_actions=3,
    )

    states = torch.randn(
        4,
        128,
    )

    outputs = heads(states)

    assert outputs["q_values"].shape == (4, 3)
    assert outputs["value"].shape == (4,)
    assert outputs["policy_logits"].shape == (4, 3)

def test_sparse_iql_training_step_updates_weights():
    torch.manual_seed(0)

    heads = SparseIQLHeads(
        state_dim=128,
        num_actions=3,
    )

    optimizer = torch.optim.Adam(
        heads.parameters(),
        lr=0.01,
    )

    states = torch.randn(
        8,
        128,
    )

    actions = torch.tensor([
        0,
        1,
        2,
        1,
        0,
        2,
        1,
        0,
    ])

    returns = torch.tensor([
        0.8,
        0.6,
        0.9,
        0.7,
        0.5,
        0.85,
        0.65,
        0.75,
    ])

    before = (
        heads.q_head.weight
        .detach()
        .clone()
    )

    result = train_sparse_iql_step(
        heads=heads,
        optimizer=optimizer,
        states=states,
        actions=actions,
        returns=returns,
        expectile=0.7,
        beta=3.0,
    )

    after = (
        heads.q_head.weight
        .detach()
        .clone()
    )

    assert isinstance(
        result["loss"],
        float,
    )

    assert result["q_loss"] >= 0.0
    assert result["value_loss"] >= 0.0
    assert result["policy_loss"] >= 0.0

    assert not torch.allclose(
        before,
        after,
    )