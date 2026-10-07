"""Heads on the final z_H (plan D7, D11). The model head matches a requirement vector
against each model card, so the pool is a list, not fixed output slots. D13 adds a
lambda/complexity-conditioned policy adapter (Monte Carlo RL); it never touches the
supervised outputs, only reshapes them into policy_logits / halt_policy_logit."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from controller.policy import PRICE_INIT


class Heads(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.need = nn.Linear(d, d)
        self.match = nn.Linear(d, 1)
        self.halt = nn.Linear(d, 1)
        self.complexity = nn.Linear(d, 1)
        self.lam_adj = nn.Linear(d + 2, 2)
        with torch.no_grad():
            self.lam_adj.weight.zero_()
            self.lam_adj.bias.copy_(torch.tensor([math.log(math.expm1(PRICE_INIT)), 0.0]))

    def forward(self, z_h: torch.Tensor, card: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.match(F.silu(self.need(z_h)[:, None, :] + card[None])).squeeze(-1)
        return model, self.halt(z_h).squeeze(-1), self.complexity(z_h).squeeze(-1)

    def policy(self, z_h: torch.Tensor, model_logits: torch.Tensor, halt_logit: torch.Tensor,
               complexity_logit: torch.Tensor, lam: torch.Tensor,
               log_price: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """lambda/complexity-conditioned policy outputs (plan D13).

        z_h [B,d], model_logits [B,M], halt_logit [B], complexity_logit [B], lam [B],
        log_price [M] -> (policy_logits [B,M], halt_policy_logit [B]).
        """
        c = torch.sigmoid(complexity_logit)
        raw = self.lam_adj(torch.cat([z_h, torch.log(lam)[:, None], c[:, None]], dim=-1))
        policy = model_logits - (lam * F.softplus(raw[:, 0]))[:, None] * log_price[None]
        halt = halt_logit + raw[:, 1]
        return policy, halt
