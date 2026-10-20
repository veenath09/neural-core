from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ActionValueHeads(nn.Module):
    """
    Monte Carlo value heads for the real rollout actions.

    model:
        Produces one value for every model in the current model pool.

    halt:
        Produces two values:
        0 = stop
        1 = escalate
    """

    def __init__(
        self,
        d: int,
        card_dim: int,
    ):
        super().__init__()

        self.card = nn.Sequential(
            nn.Linear(card_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.need = nn.Linear(d, d)
        self.model_match = nn.Linear(d, 1)

        self.halt = nn.Linear(d, 2)

    def forward(
        self,
        z_h: torch.Tensor,
        cards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        card_features = self.card(cards)

        model_values = self.model_match(
            F.silu(
                self.need(z_h)[:, None, :]
                + card_features[None, :, :]
            )
        ).squeeze(-1)

        halt_values = self.halt(z_h)

        return {
            "model": model_values,
            "halt": halt_values,
        }