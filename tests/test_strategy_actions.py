import torch
import pytest

from controller.contracts import Strategy
from train.branching_heads import BranchingValueHeads
from train.online_learner import OnlineLearner
from train.strategy_actions import (
    encode_strategy_actions,
    encode_tool_actions,
)

def test_encode_strategy_actions_uses_contract_values():
    strategy = Strategy(
    step_id="step-1",
    round=0,
    model_id="model-a",
    halt="stop",
    memory="session",
    depth="high",
    verification="cross_model",
    topology="parallel",
)

    actions = encode_strategy_actions(strategy)

    assert actions["memory"] == 1
    assert actions["depth"] == 2
    assert actions["verification"] == 2
    assert actions["topology"] == 1


def test_encoded_strategy_actions_train_branching_heads():
    torch.manual_seed(0)

    strategy = Strategy(
        step_id="step-1",
        round=0,
        model_id="model-a",
        halt="stop",
        memory="session",
        depth="high",
        verification="cross_model",
        topology="parallel",
    )

    encoded = encode_strategy_actions(
        strategy
    )

    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
            "verification": 3,
            "topology": 3,
        },
    )

    learner = OnlineLearner()

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    z_h = torch.randn(
        1,
        128,
    )

    returns = torch.tensor([
        0.8,
    ])

    actions = {
        name: torch.tensor([index])
        for name, index in encoded.items()
    }

    before = (
        heads.heads["memory"].weight
        .detach()
        .clone()
    )

    loss = learner.train_branching_step(
        heads=heads,
        optimizer=optimizer,
        z_h=z_h,
        actions=actions,
        returns=returns,
    )

    after = (
        heads.heads["memory"].weight
        .detach()
        .clone()
    )

    assert isinstance(loss, float)

    assert not torch.allclose(
        before,
        after,
    )

def test_online_learner_trains_from_strategy_contract():
    torch.manual_seed(0)

    strategy = Strategy(
        step_id="step-1",
        round=0,
        model_id="model-a",
        halt="stop",
        memory="retrieval",
        depth="high",
        verification="self_check",
        topology="sub_agents",
    )

    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
            "verification": 3,
            "topology": 3,
        },
    )

    learner = OnlineLearner()

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    z_h = torch.randn(
        1,
        128,
    )

    returns = torch.tensor([
        0.8,
    ])

    before = (
        heads.heads["depth"].weight
        .detach()
        .clone()
    )

    loss = learner.train_strategy_step(
        heads=heads,
        optimizer=optimizer,
        z_h=z_h,
        strategy=strategy,
        returns=returns,
    )

    after = (
        heads.heads["depth"].weight
        .detach()
        .clone()
    )

    assert isinstance(loss, float)

    assert not torch.allclose(
        before,
        after,
    )

def test_online_learner_trains_multiple_strategy_decisions():
    torch.manual_seed(0)

    strategies = [
        Strategy(
            step_id="step-1",
            round=0,
            model_id="model-a",
            halt="escalate",
            memory="session",
            depth="medium",
            verification="self_check",
            topology="single",
        ),
        Strategy(
            step_id="step-1",
            round=1,
            model_id="model-b",
            halt="stop",
            memory="retrieval",
            depth="high",
            verification="cross_model",
            topology="sub_agents",
        ),
    ]

    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
            "verification": 3,
            "topology": 3,
        },
    )

    learner = OnlineLearner(
        gamma=0.9,
    )

    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=0.01,
    )

    z_h = torch.randn(
        2,
        128,
    )

    returns = torch.tensor(
        learner.monte_carlo_returns(
            reward=0.8,
            n_decisions=2,
        )
    )

    before = (
        heads.heads["memory"].weight
        .detach()
        .clone()
    )

    loss = learner.train_strategy_batch(
        heads=heads,
        optimizer=optimizer,
        z_h=z_h,
        strategies=strategies,
        returns=returns,
    )

    after = (
        heads.heads["memory"].weight
        .detach()
        .clone()
    )

    assert isinstance(loss, float)

    assert returns.tolist() == pytest.approx([
        0.72,
        0.8,
    ])

    assert not torch.allclose(
        before,
        after,
    )

def test_encode_tool_actions_uses_dynamic_tool_catalog():
    strategy = Strategy(
        step_id="step-1",
        round=1,
        model_id="model-a",
        halt="stop",
        tools=[
            "search",
            "retrieval",
        ],
    )

    tool_catalog = [
        "search",
        "calculator",
        "retrieval",
    ]

    encoded = encode_tool_actions(
        strategy=strategy,
        tool_catalog=tool_catalog,
    )

    assert encoded == [
        1,
        0,
        1,
    ]

def test_encode_tool_actions_rejects_unknown_tool():
    strategy = Strategy(
        step_id="step-1",
        round=1,
        model_id="model-a",
        halt="stop",
        tools=["unknown-tool"],
    )

    tool_catalog = [
        "search",
        "calculator",
    ]

    with pytest.raises(
        ValueError,
        match="tools not in catalog",
    ):
        encode_tool_actions(
            strategy=strategy,
            tool_catalog=tool_catalog,
        )


def test_encode_tool_actions_rejects_duplicate_catalog():
    strategy = Strategy(
        step_id="step-1",
        round=1,
        model_id="model-a",
        halt="stop",
        tools=["search"],
    )

    tool_catalog = [
        "search",
        "search",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate tools",
    ):
        encode_tool_actions(
            strategy=strategy,
            tool_catalog=tool_catalog,
        )