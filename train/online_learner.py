from numpy.ma import choose
import torch
import torch.nn.functional as F
from train.strategy_actions import (
    encode_strategy_actions,
    encode_factored_strategy_actions,
)


class OnlineLearner:
    def __init__(self, gamma: float = 1.0):
        self.episode_count = 0
        self.gamma = gamma
        self.anchor_state = None

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

    def train_strategy_step(
        self,
        heads,
        optimizer,
        z_h: torch.Tensor,
        strategy,
        returns: torch.Tensor,
    ) -> float:
        if len(z_h) != 1 or len(returns) != 1:
            raise ValueError(
                "train_strategy_step expects exactly one strategy decision"
        )

        encoded_actions = encode_strategy_actions(
            strategy
        )

        actions = {
            name: torch.tensor(
                [index],
                dtype=torch.long,
                device=z_h.device,
            )
            for name, index in encoded_actions.items()
    }

        aligned_returns = returns.to(
            device=z_h.device,
            dtype=z_h.dtype,
        )

        return self.train_branching_step(
            heads=heads,
            optimizer=optimizer,
            z_h=z_h,
            actions=actions,
            returns=aligned_returns,
        )

    def train_strategy_batch(
        self,
        heads,
        optimizer,
        z_h: torch.Tensor,
        strategies: list,
        returns: torch.Tensor,
    ) -> float:
        if len(z_h) != len(strategies) or len(z_h) != len(returns):
            raise ValueError(
                "z_h, strategies and returns must have the same length"
        )

        encoded = [
            encode_strategy_actions(strategy)
            for strategy in strategies
        ]

        actions = {
            name: torch.tensor(
                [
                    strategy_actions[name]
                    for strategy_actions in encoded
            ],
            dtype=torch.long,
            device=z_h.device,
        )
            for name in encoded[0]
    }

        aligned_returns = returns.to(
            device=z_h.device,
            dtype=z_h.dtype,
    )

        return self.train_branching_step(
            heads=heads,
            optimizer=optimizer,
            z_h=z_h,
            actions=actions,
            returns=aligned_returns,
    )
    
    def train_factored_strategy_step(
        self,
        heads,
        optimizer,
        z_h: torch.Tensor,
        strategy,
        tool_catalog: list[str],
        returns: torch.Tensor,
    ) -> float:
        if len(z_h) != 1 or len(returns) != 1:
            raise ValueError(
                "train_factored_strategy_step expects exactly one strategy decision"
            )

        encoded_actions = encode_factored_strategy_actions(
            strategy=strategy,
            tool_catalog=tool_catalog,
        )

        actions = {
            name: torch.tensor(
                [index],
                dtype=torch.long,
                device=z_h.device,
            )
            for name, index in encoded_actions.items()
        }

        aligned_returns = returns.to(
            device=z_h.device,
            dtype=z_h.dtype,
        )

        return self.train_branching_step(
            heads=heads,
            optimizer=optimizer,
            z_h=z_h,
            actions=actions,
            returns=aligned_returns,
        )

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
        anchor_lambda: float = 0.0,
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

        if anchor_lambda > 0.0:
            loss = loss + (
                    anchor_lambda * self.anchor_penalty(heads)
                    )

        loss.backward()
        optimizer.step()

        return float(loss.detach())

    def train_from_rollout(
        self,
        rollout_output,
        rewards: list[float],
        cards: torch.Tensor,
        heads,
        optimizer,
        anchor_lambda: float = 0.0,
    ) -> dict:
        if len(rollout_output.results) != len(rewards):
            raise ValueError(
                "rollout results and rewards must have the same length"
            )

        states, actions, returns = self.prepare_rollout_batch(
            decision_states=rollout_output.decision_states,
            decision_actions=rollout_output.decision_actions,
            rewards=rewards,
        )

        choose_batch, halt_batch = self.extract_action_batches(
            states=states,
            actions=actions,
            returns=returns,
        )

        loss = self.train_action_value_step(
            heads=heads,
            optimizer=optimizer,
            cards=cards,
            choose_batch=choose_batch,
            halt_batch=halt_batch,
            anchor_lambda=anchor_lambda,
        )

        self.episode_count += len(rewards)

        return {
            "loss": loss,
            "episodes": len(rewards),
            "decisions": len(actions),
        }

    def store_rollout_in_replay(
        self,
        rollout_output,
        rewards: list[float],
        replay_buffer,
    ) -> int:
        if len(rollout_output.results) != len(rewards):
            raise ValueError(
                "rollout results and rewards must have the same length"
            )

        if (
            len(rollout_output.decision_states)
            != len(rollout_output.results)
            or len(rollout_output.decision_actions)
            != len(rollout_output.results)
        ):
            raise ValueError(
                "rollout states, actions and results must align"
            )

        # Existing replay episodes become one generation older.
        replay_buffer.increment_ages()

        stored_count = 0

        for index, reward in enumerate(rewards):
            decision_states = rollout_output.decision_states[index]
            decision_actions = rollout_output.decision_actions[index]

            if len(decision_states) != len(decision_actions):
                raise ValueError(
                    "decision states and actions must have the same length"
                )

            states, returns = self.prepare_episode(
                decision_states=decision_states,
                reward=reward,
            )

            replay_episode = {
                "states": states.detach().cpu(),
                "actions": list(decision_actions),
                "returns": returns.detach().cpu(),
                "domain": rollout_output.results[index].ep_type,
                "age": 0,
            }

            replay_buffer.add(replay_episode)
            stored_count += 1

        return stored_count

    def train_from_replay(
        self,
        replay_buffer,
        sample_size: int,
        cards: torch.Tensor,
        heads,
        optimizer,
        anchor_lambda: float = 0.0,
    ) -> dict:
        replay_episodes = replay_buffer.sample_age_banded(
            size=sample_size
        )

        if not replay_episodes:
            raise ValueError("replay buffer is empty")

        state_batches = []
        action_batch = []
        return_batches = []

        for episode in replay_episodes:
            states = episode["states"]
            actions = episode["actions"]
            returns = episode["returns"]

            if len(states) != len(actions) or len(states) != len(returns):
                raise ValueError(
                    "replay states, actions and returns must align"
                )

            state_batches.append(states)
            action_batch.extend(actions)
            return_batches.append(returns)

        states = torch.cat(
            state_batches,
            dim=0,
        )

        returns = torch.cat(
            return_batches,
            dim=0,
        )

        choose_batch, halt_batch = self.extract_action_batches(
            states=states,
            actions=action_batch,
            returns=returns,
        )

        loss = self.train_action_value_step(
            heads=heads,
            optimizer=optimizer,
            cards=cards,
            choose_batch=choose_batch,
            halt_batch=halt_batch,
            anchor_lambda=anchor_lambda,
        )

        return {
            "loss": loss,
            "replay_episodes": len(replay_episodes),
            "decisions": len(action_batch),
        }

    def consolidate(
        self,
        replay_buffer,
        sample_size: int,
        cards: torch.Tensor,
        heads,
        optimizer,
        anchor_lambda: float = 0.0,
    ) -> dict:
        replay_episodes = replay_buffer.sample_domain_balanced(
            size=sample_size
        )

        if not replay_episodes:
            raise ValueError("replay buffer is empty")

        state_batches = []
        action_batch = []
        return_batches = []

        for episode in replay_episodes:
            states = episode["states"]
            actions = episode["actions"]
            returns = episode["returns"]

            if len(states) != len(actions) or len(states) != len(returns):
                raise ValueError(
                    "replay states, actions and returns must align"
                )

            state_batches.append(states)
            action_batch.extend(actions)
            return_batches.append(returns)

        states = torch.cat(
            state_batches,
            dim=0,
        )

        returns = torch.cat(
            return_batches,
            dim=0,
        )

        choose_batch, halt_batch = self.extract_action_batches(
            states=states,
            actions=action_batch,
            returns=returns,
        )

        loss = self.train_action_value_step(
            heads=heads,
            optimizer=optimizer,
            cards=cards,
            choose_batch=choose_batch,
            halt_batch=halt_batch,
            anchor_lambda=anchor_lambda,
        )

        # Consolidation completed successfully.
        # The newly updated parameters become the new stable anchor.
        self.set_anchor(heads)

        return {
            "loss": loss,
            "replay_episodes": len(replay_episodes),
            "decisions": len(action_batch),
            "anchor_reset": True,
        }

    def set_anchor(self, heads) -> None:
        self.anchor_state = {
            name: parameter.detach().clone()
            for name, parameter in heads.named_parameters()
        }

    def anchor_penalty(self, heads) -> torch.Tensor:
        if self.anchor_state is None:
            raise ValueError("anchor has not been set")

        penalties = []

        for name, parameter in heads.named_parameters():
            anchor_parameter = self.anchor_state[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

            penalties.append(
                torch.mean(
                    (parameter - anchor_parameter) ** 2
                )
            )

        return torch.stack(penalties).mean()
    
    def save_checkpoint(
        self,
        path,
        heads,
        optimizer,
    ) -> None:
        checkpoint = {
            "heads_state_dict": heads.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
            "episode_count": self.episode_count,
            "gamma": self.gamma,
            "anchor_state": (
               None
               if self.anchor_state is None
               else {
                  name: parameter.detach().cpu().clone()
                  for name, parameter in self.anchor_state.items()
              }
        ),
}

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path,
        heads,
        optimizer,
    ) -> None:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        heads.load_state_dict(
            checkpoint["heads_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.episode_count = checkpoint["episode_count"]
        self.gamma = checkpoint["gamma"]
        anchor_state = checkpoint.get("anchor_state")

        self.anchor_state = (
            None
            if anchor_state is None
            else {
                name: parameter.detach().clone()
                for name, parameter in anchor_state.items()
        }
    )

        
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