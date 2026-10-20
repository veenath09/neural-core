from __future__ import annotations

import random
import math
import torch


class CostBoundedExplorer:
    def __init__(
        self,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def select_model(
        self,
        values: torch.Tensor,
        costs: torch.Tensor,
        budget: float,
    ) -> int:
        if values.ndim != 1 or costs.ndim != 1:
            raise ValueError(
                "values and costs must be 1D tensors"
            )

        if len(values) != len(costs):
            raise ValueError(
                "values and costs must have the same length"
            )

        if not torch.isfinite(values).all():
            raise ValueError(
                "values must contain finite values"
        )

        if not math.isfinite(budget) or budget < 0:
            raise ValueError(
                "budget must be a finite non-negative value"
        )


        if (
            not torch.isfinite(costs).all()
            or torch.any(costs < 0)
        ):
            raise ValueError(
                "costs must contain finite non-negative values"
        )

        affordable_indices = [
            index
            for index, cost in enumerate(costs)
            if float(cost.item()) <= budget
        ]

        if not affordable_indices:
            raise ValueError(
                "no model is affordable within the budget"
            )

        # Explore only among affordable models.
        if self.rng.random() < self.epsilon:
            return self.rng.choice(
                affordable_indices
            )

        # Exploit: choose the highest-value affordable model.
        best_index = max(
            affordable_indices,
            key=lambda index: float(
                values[index].item()
            ),
        )

        return int(best_index)