from __future__ import annotations

from train.experiment_logging import append_experiment_record
from train.reward_adapter import rewards_from_scores
from train.safety import validate_or_rollback


def run_online_update(
    learner,
    rollout_output,
    rewards,
    cards,
    heads,
    optimizer,
    replay_buffer,
    replay_sample_size: int,
    anchor_lambda: float = 0.0,
    validation_gate=None,
    rollback_manager=None,
    validation_fn=None,
    log_path=None,
) -> dict:
    validation_result = None

    safety_enabled = (
        validation_gate is not None
        and rollback_manager is not None
        and validation_fn is not None
    )

    if safety_enabled:
        # Evaluate the currently accepted model
        # before candidate training.
        baseline_score = validation_fn(heads)

        # Capture both model and optimizer state.
        rollback_manager.capture(
            heads,
            optimizer=optimizer,
        )

    # 1. Train immediately on the fresh rollout.
    fresh_result = learner.train_from_rollout(
        rollout_output=rollout_output,
        rewards=rewards,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        anchor_lambda=anchor_lambda,
    )

    # 2. Train only on replay episodes that existed
    # before the fresh rollout is stored.
    replay_result = None

    if len(replay_buffer) > 0:
        replay_result = learner.train_from_replay(
            replay_buffer=replay_buffer,
            sample_size=replay_sample_size,
            cards=cards,
            heads=heads,
            optimizer=optimizer,
            anchor_lambda=anchor_lambda,
        )

    # 3. Validate the trained candidate.
    if safety_enabled:
        candidate_score = validation_fn(heads)

        accepted = validate_or_rollback(
            module=heads,
            gate=validation_gate,
            rollback=rollback_manager,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            optimizer=optimizer,
        )

        validation_result = {
            "accepted": accepted,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
        }

    # 4. Store the fresh experience only after
    # replay training has finished.
    stored_episodes = learner.store_rollout_in_replay(
        rollout_output=rollout_output,
        rewards=rewards,
        replay_buffer=replay_buffer,
    )

    # 5. Persist one research record if logging is enabled.
    logged = False

    if log_path is not None:
        episode_count = getattr(
            learner,
            "episode_count",
            fresh_result.get("episodes", 0),
        )

        reward_mean = (
            sum(rewards) / len(rewards)
            if rewards
            else None
        )

        accepted = (
            None
            if validation_result is None
            else validation_result["accepted"]
        )

        record = {
            "episode_count": episode_count,
            "fresh_loss": fresh_result.get("loss"),
            "replay_loss": (
                None
                if replay_result is None
                else replay_result.get("loss")
            ),
            "reward_mean": reward_mean,
            "stored_episodes": stored_episodes,
            "accepted": accepted,
        }

        if validation_result is not None:
            record["baseline_score"] = (
                validation_result["baseline_score"]
            )
            record["candidate_score"] = (
                validation_result["candidate_score"]
            )

        append_experiment_record(
            log_path,
            record,
        )

        logged = True

    return {
        "fresh": fresh_result,
        "replay": replay_result,
        "stored_episodes": stored_episodes,
        "validation": validation_result,
        "logged": logged,
    }


def run_online_update_from_scores(
    learner,
    rollout_output,
    scores,
    cards,
    heads,
    optimizer,
    replay_buffer,
    replay_sample_size: int,
    anchor_lambda: float = 0.0,
    validation_gate=None,
    rollback_manager=None,
    validation_fn=None,
    log_path=None,
) -> dict:
    """Run one C2 update using final C3 Score rewards."""

    rewards = rewards_from_scores(
        scores,
        expected_count=len(rollout_output.results),
    )

    return run_online_update(
        learner=learner,
        rollout_output=rollout_output,
        rewards=rewards,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        replay_sample_size=replay_sample_size,
        anchor_lambda=anchor_lambda,
        validation_gate=validation_gate,
        rollback_manager=rollback_manager,
        validation_fn=validation_fn,
        log_path=log_path,
    )