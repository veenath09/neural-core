from numpy.ma import choose
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

    def prepare_rollout(
        self,
        decision_states: list[list[torch.Tensor]],
        rewards: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(decision_states) != len(rewards):
            raise ValueError(
                "decision_states and rewards must have the same length"
        )

        state_batches = []
        return_batches = []

        for episode_states, reward in zip(
            decision_states,
            rewards,
        ):
            states, returns = self.prepare_episode(
                decision_states=episode_states,
                    reward=reward,
            )

            state_batches.append(states)
            return_batches.append(returns)

        if not state_batches:
            raise ValueError("rollout must contain at least one episode")

        return (
            torch.cat(state_batches, dim=0),
            torch.cat(return_batches, dim=0),
        )

    def prepare_rollout_batch(
        self,
        decision_states: list[list[torch.Tensor]],
        decision_actions: list[list[dict]],
        rewards: list[float],
    ) -> tuple[torch.Tensor, list[dict], torch.Tensor]:
        if not (
            len(decision_states)
            == len(decision_actions)
            == len(rewards)
        ):
            raise ValueError(
                "decision_states, decision_actions and rewards "
                "must have the same number of episodes"
        )

        state_batches = []
        action_batch = []
        return_batches = []

        for episode_states, episode_actions, reward in zip(
            decision_states,
            decision_actions,
            rewards,
        ):
            if len(episode_states) != len(episode_actions):
                raise ValueError(
                    "each decision state must have a matching action"
                )

            states, returns = self.prepare_episode(
                decision_states=episode_states,
                reward=reward,
            )

            state_batches.append(states)
            action_batch.extend(episode_actions)
            return_batches.append(returns)

        if not state_batches:
            raise ValueError(
            "rollout must contain at least one episode"
        )

        return (
            torch.cat(state_batches, dim=0),
            action_batch,
            torch.cat(return_batches, dim=0),
        )

    def split_actions(
        self,
        actions: list[dict],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        choose = []
        halt = []

        for index, action in enumerate(actions):
            phase = action["phase"]

            if phase == "choose":
                choose.append(
                (index, int(action["model"]))
            )

            elif phase == "halt":
                halt_value = 1 if action["halt"] == "escalate" else 0
                halt.append(
                    (index, halt_value)
            )

            else:
                raise ValueError(
                    f"unknown decision phase {phase!r}"
            )

        return choose, halt

    def extract_action_batches(
        self,
        states: torch.Tensor,
        actions: list[dict],
        returns: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if len(states) != len(actions) or len(states) != len(returns):
            raise ValueError(
            "states, actions and returns must have the same length"
        )

        choose, halt = self.split_actions(actions)

        choose_indices = torch.tensor(
            [index for index, _ in choose],
            dtype=torch.long,
            device=states.device,
        )

        choose_actions = torch.tensor(
            [action for _, action in choose],
            dtype=torch.long,
            device=states.device,
        )

        halt_indices = torch.tensor(
            [index for index, _ in halt],
            dtype=torch.long,
            device=states.device,
        )

        halt_actions = torch.tensor(
            [action for _, action in halt],
            dtype=torch.long,
            device=states.device,
        )

        choose_batch = {
            "states": states[choose_indices],
            "actions": choose_actions,
            "returns": returns[choose_indices],
        }

        halt_batch = {
            "states": states[halt_indices],
            "actions": halt_actions,
            "returns": returns[halt_indices],
        }

        return choose_batch, halt_batch 

    def train_action_value_step(
        self,
        heads,
        optimizer,
        cards: torch.Tensor,
        choose_batch: dict[str, torch.Tensor],
        halt_batch: dict[str, torch.Tensor],
    ) -> float:
        optimizer.zero_grad(set_to_none=True)

        losses = []

        # Train the real model-choice value head.
        if len(choose_batch["actions"]) > 0:
            choose_values = heads(
                z_h=choose_batch["states"].detach(),
                cards=cards,
            )["model"]

            selected_model_values = choose_values.gather(
                1,
                choose_batch["actions"].unsqueeze(1),
            ).squeeze(1)

            model_loss = F.mse_loss(
                selected_model_values,
                choose_batch["returns"],
            )

            losses.append(model_loss)

        # Train the real halt value head.
        if len(halt_batch["actions"]) > 0:
            halt_values = heads(
                z_h=halt_batch["states"].detach(),
                cards=cards,
            )["halt"]

            selected_halt_values = halt_values.gather(
                1,
                halt_batch["actions"].unsqueeze(1),
            ).squeeze(1)

            halt_loss = F.mse_loss(
                selected_halt_values,
                halt_batch["returns"],
            )

            losses.append(halt_loss)

        if not losses:
            raise ValueError("no CHOOSE or HALT decisions to train")

        loss = torch.stack(losses).mean()

        loss.backward()
        optimizer.step()

        return float(loss.detach())

    
       
    
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