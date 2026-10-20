import torch

from train.exploration import CostBoundedExplorer
from train.action_value_heads import ActionValueHeads


def test_explorer_never_selects_model_above_budget():
    explorer = CostBoundedExplorer(
        epsilon=1.0,
        seed=0,
    )

    values = torch.tensor([
        0.2,
        0.4,
        0.9,
    ])

    costs = torch.tensor([
        0.01,
        0.03,
        0.10,
    ])

    budget = 0.05

    for _ in range(50):
        action = explorer.select_model(
            values=values,
            costs=costs,
            budget=budget,
        )

        assert costs[action].item() <= budget

def test_explorer_exploits_best_affordable_model():
    explorer = CostBoundedExplorer(
        epsilon=0.0,
        seed=0,
    )

    values = torch.tensor([
        0.2,
        0.7,
        0.9,
    ])

    costs = torch.tensor([
        0.01,
        0.03,
        0.10,
    ])

    budget = 0.05

    action = explorer.select_model(
        values=values,
        costs=costs,
        budget=budget,
    )

    assert action == 1

import pytest


def test_explorer_raises_when_no_model_is_affordable():
    explorer = CostBoundedExplorer(
        epsilon=0.5,
        seed=0,
    )

    values = torch.tensor([
        0.2,
        0.7,
        0.9,
    ])

    costs = torch.tensor([
        0.06,
        0.08,
        0.10,
    ])

    budget = 0.05

    with pytest.raises(
        ValueError,
        match="no model is affordable within the budget",
    ):
        explorer.select_model(
            values=values,
            costs=costs,
            budget=budget,
        )
def test_explorer_works_with_action_value_head_scores():
    torch.manual_seed(0)

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    z_h = torch.randn(1, 128)

    cards = torch.randn(3, 6)

    model_values = heads(
        z_h=z_h,
        cards=cards,
    )["model"][0]

    costs = torch.tensor([
        0.01,
        0.03,
        0.10,
    ])

    budget = 0.05

    explorer = CostBoundedExplorer(
        epsilon=0.0,
        seed=0,
    )

    action = explorer.select_model(
        values=model_values,
        costs=costs,
        budget=budget,
    )

    expected_action = int(
        torch.argmax(
            model_values[:2]
        ).item()
    )

    assert action == expected_action
    assert costs[action].item() <= budget

@pytest.mark.parametrize(
    "invalid_cost",
    [
        -0.01,
        float("nan"),
        float("inf"),
    ],
)
def test_explorer_rejects_invalid_model_costs(invalid_cost):
    explorer = CostBoundedExplorer(
        epsilon=0.0,
        seed=0,
    )

    values = torch.tensor([
        0.2,
        0.8,
    ])

    costs = torch.tensor([
        0.01,
        invalid_cost,
    ])

    with pytest.raises(
        ValueError,
        match="costs must contain finite non-negative values",
    ):
        explorer.select_model(
            values=values,
            costs=costs,
            budget=0.05,
        )

@pytest.mark.parametrize(
    "invalid_budget",
    [
        -0.01,
        float("nan"),
        float("inf"),
    ],
)
def test_explorer_rejects_invalid_budget(invalid_budget):
    explorer = CostBoundedExplorer(
        epsilon=0.0,
        seed=0,
    )

    values = torch.tensor([
        0.2,
        0.8,
    ])

    costs = torch.tensor([
        0.01,
        0.03,
    ])

    with pytest.raises(
        ValueError,
        match="budget must be a finite non-negative value",
    ):
        explorer.select_model(
            values=values,
            costs=costs,
            budget=invalid_budget,
        )