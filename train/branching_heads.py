from __future__ import annotations

import torch
from torch import nn


class BranchingValueHeads(nn.Module):
    """One value head per factored strategy dimension."""

    def __init__(self, d: int, dimensions: dict[str, int]):
        super().__init__()

        if not dimensions:
            raise ValueError("dimensions must not be empty")

        if any(n_choices < 1 for n_choices in dimensions.values()):
            raise ValueError("each dimension must have at least one choice")

        self.heads = nn.ModuleDict({
            name: nn.Linear(d, n_choices)
            for name, n_choices in dimensions.items()
        })

    def forward(self, z_h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: head(z_h)
            for name, head in self.heads.items()
        }