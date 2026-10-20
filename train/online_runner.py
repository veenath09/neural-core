from __future__ import annotations


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
) -> dict:
    # 1. Learn immediately from the fresh rollout.
    fresh_result = learner.train_from_rollout(
        rollout_output=rollout_output,
        rewards=rewards,
        cards=cards,
        heads=heads,
        optimizer=optimizer,
        anchor_lambda=anchor_lambda,
    )

    # 2. Train only from replay episodes that existed
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

    # 3. Store the fresh rollout only after replay training.
    stored_episodes = learner.store_rollout_in_replay(
        rollout_output=rollout_output,
        rewards=rewards,
        replay_buffer=replay_buffer,
    )

    return {
        "fresh": fresh_result,
        "replay": replay_result,
        "stored_episodes": stored_episodes,
    }