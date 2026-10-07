"""Policy constants for Monte Carlo RL (plan D13). Owned by controller/: no import from
measurement/, so only PRICE_INIT and log_price live here (LAMS stays in measurement.rl_reward)."""

from __future__ import annotations

import numpy as np
import torch

PRICE_INIT = 0.1


def log_price(card_costs: np.ndarray, cost_ref: float) -> torch.Tensor:
    """ln(card_cost / cost_ref) per model card, float32 [M]."""
    return torch.log(torch.as_tensor(card_costs, dtype=torch.float32) / cost_ref)
