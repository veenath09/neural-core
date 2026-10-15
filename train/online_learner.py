import torch
import torch.nn.functional as F


class OnlineLearner:
    def __init__(self, gamma: float = 1.0):
        self.episode_count = 0
        self.gamma = gamma

    def monte_carlo_returns(
        self,
        reward: float,
        n_decisions: int,
    ) -> list[float]:
        if n_decisions < 1:
            raise ValueError("n_decisions must be at least 1")

        return [
            (self.gamma ** (n_decisions - 1 - i)) * reward
            for i in range(n_decisions)
        ]

    def monte_carlo_loss(
        self,
        predicted: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        if predicted.shape != returns.shape:
            raise ValueError("predicted and returns must have the same shape")

        return F.mse_loss(predicted, returns)

    def update(
        self,
        reward: float,
        n_decisions: int = 1,
    ):
        self.episode_count += 1

        returns = self.monte_carlo_returns(
            reward,
            n_decisions,
        )

        return {
            "episode": self.episode_count,
            "reward": reward,
            "returns": returns,
        }