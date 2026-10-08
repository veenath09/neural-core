"""Episode memory: a fast weight W written once per round outcome (plan D2, D11).

Re-implemented from the equations, no code copied:
- inner model and associative loss: TTT-Linear (Sun et al. 2024, arXiv 2407.04620;
  test-time-training/ttt-lm-pytorch, MIT), one explicit delta-rule step;
- forgetting toward a learned prior: Titans (Behrouz et al. 2025, arXiv 2501.00663;
  lucidrains/titans-pytorch, MIT), without momentum;
- CfC gate: closed-form continuous-time cell, Eq. 10 (Hasani et al. 2022,
  arXiv 2106.13898), following mlech26l/ncps torch/cfc_cell.py default mode (Apache-2.0).
Gating the write on the surprise s (not on the input) is this project's addition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from controller.core import rms_norm


@dataclass
class MemoryState:
    W: torch.Tensor  # [B, d, d]
    c: torch.Tensor  # [B, gate units]; width 0 for gates without state

    def select(self, idx: torch.Tensor) -> MemoryState:
        return MemoryState(self.W[idx], self.c[idx])

    def merge(self, idx: torch.Tensor, part: MemoryState) -> MemoryState:
        return MemoryState(self.W.index_copy(0, idx, part.W), self.c.index_copy(0, idx, part.c))


def lecun_tanh(x: torch.Tensor) -> torch.Tensor:
    return 1.7159 * torch.tanh(0.666 * x)


class CfCGate(nn.Module):
    """Write rate from a CfC cell whose state persists across an episode's writes."""

    def __init__(self, in_dim: int, units: int = 16, backbone: int = 32):
        super().__init__()
        self.units = units
        self.backbone = nn.Linear(in_dim + units, backbone)
        self.ff1 = nn.Linear(backbone, units)
        self.ff2 = nn.Linear(backbone, units)
        self.time_a = nn.Linear(backbone, units)
        self.time_b = nn.Linear(backbone, units)
        self.readout = nn.Linear(units, 1)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)
            elif name == "readout.bias":
                nn.init.zeros_(p)

    def forward(self, u: torch.Tensor, c: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = lecun_tanh(self.backbone(torch.cat([u, c], dim=-1)))
        f1, f2 = torch.tanh(self.ff1(b)), torch.tanh(self.ff2(b))
        t = torch.sigmoid(self.time_a(b) * dt[:, None] + self.time_b(b))
        c = f1 * (1 - t) + t * f2
        return torch.sigmoid(self.readout(c)).squeeze(-1), c


class StatelessGate(nn.Module):
    """Ablation: same input as the CfC gate, no state and no time."""

    units = 0

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, u, c, dt):
        return torch.sigmoid(self.net(u)).squeeze(-1), c


class ConstGate(nn.Module):
    """Ablation gate_off: one learned rate."""

    units = 0

    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, u, c, dt):
        return torch.sigmoid(self.c).expand(u.shape[0]), c


class EpisodeMemory(nn.Module):
    def __init__(self, d: int = 128, gate: str = "cfc", gate_units: int = 16):
        super().__init__()
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(d, d, bias=False)
        self.W0 = nn.Parameter(torch.zeros(d, d))
        gate_in = 2 * d + 1  # [s/|s|, log|s|, obs]
        if gate == "cfc":
            self.gate = CfCGate(gate_in, units=gate_units)
        elif gate == "stateless":
            self.gate = StatelessGate(gate_in)
        elif gate == "const":
            self.gate = ConstGate()
        else:
            raise ValueError(f"unknown gate {gate!r}")
        self.forget = nn.Linear(d + 1, 1)
        nn.init.zeros_(self.forget.weight)
        nn.init.constant_(self.forget.bias, -4.0)

    def init_state(self, batch: int) -> MemoryState:
        return MemoryState(self.W0.unsqueeze(0).expand(batch, -1, -1),
                           torch.zeros(batch, self.gate.units, device=self.W0.device))

    def write(self, state: MemoryState, obs: torch.Tensor, mask: torch.Tensor,
              dt: torch.Tensor) -> tuple[MemoryState, dict[str, torch.Tensor]]:
        k = F.normalize(self.key(obs), dim=-1)
        v = self.value(obs)
        s = v - torch.bmm(state.W, k.unsqueeze(-1)).squeeze(-1)
        s_norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        log_s = s_norm.log()
        rate, c_new = self.gate(torch.cat([s / s_norm, log_s, obs], dim=-1), state.c, dt)
        alpha = torch.sigmoid(self.forget(torch.cat([log_s, obs], dim=-1))).squeeze(-1)
        W0 = self.W0.unsqueeze(0)
        W_new = W0 + (1 - alpha)[:, None, None] * (state.W - W0) + rate[:, None, None] * s.unsqueeze(-1) * k.unsqueeze(1)
        W = torch.where(mask[:, None, None], W_new, state.W)
        c = torch.where(mask[:, None], c_new, state.c)
        return MemoryState(W, c), {"surprise": s_norm.squeeze(-1), "rate": rate, "alpha": alpha}

    def read(self, state: MemoryState, z: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query(z), dim=-1)
        return rms_norm(torch.bmm(state.W, q.unsqueeze(-1)).squeeze(-1))
