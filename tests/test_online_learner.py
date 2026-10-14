from train.online_learner import OnlineLearner


def test_learner_updates_episode_count():
    learner = OnlineLearner()

    result = learner.update(0.8)

    assert result["episode"] == 1
    assert result["reward"] == 0.8