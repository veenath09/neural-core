"""StepEncoder: Batch -> H tokens (task, budget/state) and L tokens (step, trace -1, -2) (plan §3, D11)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from controller.features import BUDGET_FEATS, TRACE_FEATS, Batch


@dataclass
class Encoded:
    task: torch.Tensor  # [B, d]
    budget: torch.Tensor  # [B, d]
    step: torch.Tensor  # [B, d]
    trace: torch.Tensor  # [B, K, d], zero where absent
    card: torch.Tensor  # [M, d]


def _mlp(n_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, d), nn.SiLU(), nn.Linear(d, d))


class StepEncoder(nn.Module):
    def __init__(self, d: int, emb_dim: int, n_types: int, card_dim: int):
        super().__init__()
        self.text = nn.Linear(emb_dim, d)  # projects frozen bge-small vectors
        self.step_type = nn.Embedding(n_types, d)
        self.round = nn.Embedding(3, d)
        self.phase = nn.Embedding(2, d)
        self.budget = _mlp(BUDGET_FEATS, d)
        self.trace = _mlp(TRACE_FEATS, d)
        self.card = _mlp(card_dim, d)

    def forward(self, b: Batch, emb: torch.Tensor, cards: torch.Tensor) -> Encoded:
        card = self.card(cards)
        step = (self.text(emb[b.step_row]) + self.step_type(b.step_type) + self.round(b.round)
                + self.phase(b.phase))
        trace = self.text(emb[b.trace_row]) + self.trace(b.trace_feats) + card[b.trace_model]
        return Encoded(task=self.text(emb[b.task_row]), budget=self.budget(b.budget), step=step,
                       trace=trace * b.trace_present[..., None], card=card)
