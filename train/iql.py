from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    expectile: float = 0.7,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

    if not 0.0 < expectile < 1.0:
        raise ValueError(
            "expectile must be between 0 and 1"
        )

    difference = target - prediction

    weights = torch.where(
        difference > 0,
        expectile,
        1.0 - expectile,
    )

    return torch.mean(
        weights * difference.pow(2)
    )

def advantage_weights(
    q_values: torch.Tensor,
    v_values: torch.Tensor,
    beta: float = 3.0,
    max_weight: float = 100.0,
) -> torch.Tensor:
    if q_values.shape != v_values.shape:
        raise ValueError(
            "q_values and v_values must have the same shape"
        )

    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(
            "beta must be a finite positive value"
    )

    if not math.isfinite(max_weight) or max_weight <= 0.0:
        raise ValueError(
            "max_weight must be a finite positive value"
    )

    advantages = q_values - v_values

    max_log_weight = math.log(max_weight)

    scaled_advantages = torch.clamp(
        beta * advantages,
        max=max_log_weight,
    )

    return torch.exp(scaled_advantages)

def weighted_behavior_cloning_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(
            "logits must be a 2D tensor"
        )

    if actions.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            "actions and weights must be 1D tensors"
        )

    if len(logits) != len(actions) or len(logits) != len(weights):
        raise ValueError(
            "logits, actions and weights must have the same batch size"
        )

    per_sample_loss = F.cross_entropy(
        logits,
        actions,
        reduction="none",
    )

    detached_weights = weights.detach()

    return torch.mean(
        detached_weights * per_sample_loss
    )

def iql_loss_components(
    q_values: torch.Tensor,
    value_predictions: torch.Tensor,
    policy_logits: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    expectile: float = 0.7,
    beta: float = 3.0,
) -> dict[str, torch.Tensor]:
    if q_values.ndim != 2:
        raise ValueError(
            "q_values must be a 2D tensor"
        )

    if policy_logits.ndim != 2:
        raise ValueError(
            "policy_logits must be a 2D tensor"
        )

    if value_predictions.ndim != 1:
        raise ValueError(
            "value_predictions must be a 1D tensor"
        )

    if actions.ndim != 1 or returns.ndim != 1:
        raise ValueError(
            "actions and returns must be 1D tensors"
        )

    batch_size = len(actions)

    if (
        len(q_values) != batch_size
        or len(value_predictions) != batch_size
        or len(policy_logits) != batch_size
        or len(returns) != batch_size
    ):
        raise ValueError(
            "all IQL inputs must have the same batch size"
        )

    # Q-value for the action that was actually logged.
    chosen_q = q_values.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    # One-step / contextual-bandit Q target:
    # realised Monte Carlo return.
    q_loss = F.mse_loss(
        chosen_q,
        returns,
    )

    # Learn V(s) using expectile regression against Q(s, a).
    # Detach Q so this loss updates the value prediction only.
    value_loss = expectile_loss(
        prediction=value_predictions,
        target=chosen_q.detach(),
        expectile=expectile,
    )

    # High-advantage logged actions receive larger policy weights.
    weights = advantage_weights(
        q_values=chosen_q.detach(),
        v_values=value_predictions.detach(),
        beta=beta,
    )

    policy_loss = weighted_behavior_cloning_loss(
        logits=policy_logits,
        actions=actions,
        weights=weights,
    )

    return {
        "q_loss": q_loss,
        "value_loss": value_loss,
        "policy_loss": policy_loss,
    }

class SparseIQLHeads(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError(
                "state_dim must be positive"
            )

        if num_actions <= 1:
            raise ValueError(
                "num_actions must be greater than 1"
            )

        self.shared = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
        )

        self.q_head = nn.Linear(
            state_dim,
            num_actions,
        )

        self.value_head = nn.Linear(
            state_dim,
            1,
        )

        self.policy_head = nn.Linear(
            state_dim,
            num_actions,
        )

    def forward(
        self,
        states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.shared(
            states
        )

        q_values = self.q_head(
            features
        )

        value = self.value_head(
            features
        ).squeeze(-1)

        policy_logits = self.policy_head(
            features
        )

        return {
            "q_values": q_values,
            "value": value,
            "policy_logits": policy_logits,
        }

def train_sparse_iql_step(
    heads,
    optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    expectile: float = 0.7,
    beta: float = 3.0,
) -> dict:
    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = heads(
        states.detach()
    )

    losses = iql_loss_components(
        q_values=outputs["q_values"],
        value_predictions=outputs["value"],
        policy_logits=outputs["policy_logits"],
        actions=actions,
        returns=returns,
        expectile=expectile,
        beta=beta,
    )

    total_loss = (
        losses["q_loss"]
        + losses["value_loss"]
        + losses["policy_loss"]
    )

    total_loss.backward()
    optimizer.step()

    return {
        "loss": float(
            total_loss.detach().item()
        ),
        "q_loss": float(
            losses["q_loss"].detach().item()
        ),
        "value_loss": float(
            losses["value_loss"].detach().item()
        ),
        "policy_loss": float(
            losses["policy_loss"].detach().item()
        ),
    }