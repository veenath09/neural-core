import torch

from train.branching_heads import BranchingValueHeads


def test_branching_heads_output_shapes():
    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
            "verification": 3,
            "topology": 3,
        },
    )

    z_h = torch.randn(4, 128)

    values = heads(z_h)

    assert set(values.keys()) == {
        "memory",
        "depth",
        "verification",
        "topology",
    }

    assert values["memory"].shape == (4, 3)
    assert values["depth"].shape == (4, 3)
    assert values["verification"].shape == (4, 3)
    assert values["topology"].shape == (4, 3)


def test_branching_heads_support_gradients():
    heads = BranchingValueHeads(
        d=128,
        dimensions={
            "memory": 3,
            "depth": 3,
        },
    )

    z_h = torch.randn(2, 128, requires_grad=True)

    values = heads(z_h)

    loss = values["memory"].mean() + values["depth"].mean()
    loss.backward()

    assert z_h.grad is not None