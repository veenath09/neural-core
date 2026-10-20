from __future__ import annotations

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
) -> dict:
    validation_result = None

    safety_enabled = (
        validation_gate is not None
        and rollback_manager is not None
        and validation_fn is not None
    )

    if safety_enabled:
        # Evaluate the currently accepted model.
        baseline_score = validation_fn(heads)

        # Save both model weights and optimizer state
        # before candidate training.
        rollback_manager.capture(
            heads,
            optimizer=optimizer,
        )

    # 1. Train on the fresh rollout.
    fresh_result = learner.train_from_rollout(
        rollout_output=rollout_output,
        rewards=rewards,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        anchor_lambda=anchor_lambda,
    )

    # 2. Train only on replay data that already existed.
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

    # 4. Store fresh experience after replay training.
    stored_episodes = learner.store_rollout_in_replay(
        rollout_output=rollout_output,
        rewards=rewards,
        replay_buffer=replay_buffer,
    )

    return {
        "fresh": fresh_result,
        "replay": replay_result,
        "stored_episodes": stored_episodes,
        "validation": validation_result,
    }