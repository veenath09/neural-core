"""Heads on the final z_H (plan D7, D11). The model head matches a requirement vector
against each model card, so the pool is a list, not fixed output slots."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Heads(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.need = nn.Linear(d, d)
        self.match = nn.Linear(d, 1)
        self.halt = nn.Linear(d, 1)
        self.complexity = nn.Linear(d, 1)

    def forward(self, z_h: torch.Tensor, card: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.match(F.silu(self.need(z_h)[:, None, :] + card[None])).squeeze(-1)
        return model, self.halt(z_h).squeeze(-1), self.complexity(z_h).squeeze(-1)
