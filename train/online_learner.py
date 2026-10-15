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

    def branching_loss(
        self,
        values: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        returns: torch.Tensor,
    ) -> torch.Tensor:
        losses = []

        for name, branch_values in values.items():
            if name not in actions:
                raise ValueError(f"missing action for branch {name!r}")

            chosen = branch_values.gather(
                1,
                actions[name].unsqueeze(1),
            ).squeeze(1)

            losses.append(
                F.mse_loss(chosen, returns)
            )

        if not losses:
            raise ValueError("values must contain at least one branch")

        return torch.stack(losses).mean()

    def train_branching_step(
        self,
        heads,
        optimizer,
        z_h: torch.Tensor,
        actions: dict[str, torch.Tensor],
        returns: torch.Tensor,
    ) -> float:
        optimizer.zero_grad()

        values = heads(z_h.detach())

        loss = self.branching_loss(
        values=values,
        actions=actions,
        returns=returns,
        )

        loss.backward()
        optimizer.step()

        return float(loss.detach().item())

    def prepare_episode(
        self,
        decision_states: list[torch.Tensor],
        reward: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not decision_states:
            raise ValueError("decision_states must not be empty")

        states = torch.stack(decision_states)

        returns = torch.tensor(
            self.monte_carlo_returns(
                reward=reward,
                n_decisions=len(decision_states),
        ),
        dtype=states.dtype,
        device=states.device,
    )

        return states, returns

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