from train.replay_buffer import ReplayBuffer


def test_replay_buffer_stores_training_episode():
    buffer = ReplayBuffer(capacity=3)

    episode = {
        "states": ["state_1", "state_2"],
        "actions": [
            {"phase": "choose", "model": 1, "halt": None},
            {"phase": "halt", "model": None, "halt": "stop"},
        ],
        "returns": [0.8, 0.8],
        "domain": "summarisation",
        "age": 0,
    }

    buffer.add(episode)

    assert len(buffer) == 1

    stored = buffer.sample(1)[0]

    assert stored["domain"] == "summarisation"
    assert stored["age"] == 0
    assert len(stored["states"]) == 2
    assert len(stored["actions"]) == 2
    assert len(stored["returns"]) == 2

def test_replay_buffer_evicts_oldest_when_full():
    buffer = ReplayBuffer(capacity=2)

    episode_1 = {
        "states": ["s1"],
        "actions": [{"phase": "choose", "model": 0, "halt": None}],
        "returns": [0.2],
        "domain": "domain_a",
        "age": 0,
    }

    episode_2 = {
        "states": ["s2"],
        "actions": [{"phase": "choose", "model": 1, "halt": None}],
        "returns": [0.5],
        "domain": "domain_b",
        "age": 0,
    }

    episode_3 = {
        "states": ["s3"],
        "actions": [{"phase": "choose", "model": 2, "halt": None}],
        "returns": [0.9],
        "domain": "domain_c",
        "age": 0,
    }

    buffer.add(episode_1)
    buffer.add(episode_2)
    buffer.add(episode_3)

    assert len(buffer) == 2

    stored = buffer.sample(2)

    assert episode_1 not in stored
    assert episode_2 in stored
    assert episode_3 in stored


def test_replay_buffer_increments_episode_age():
    buffer = ReplayBuffer(capacity=3)

    episode_1 = {
        "states": ["s1"],
        "actions": [{"phase": "choose", "model": 0, "halt": None}],
        "returns": [0.2],
        "domain": "domain_a",
        "age": 0,
    }

    episode_2 = {
        "states": ["s2"],
        "actions": [{"phase": "choose", "model": 1, "halt": None}],
        "returns": [0.5],
        "domain": "domain_b",
        "age": 0,
    }

    buffer.add(episode_1)
    buffer.increment_ages()

    assert buffer.sample(1)[0]["age"] == 1

    buffer.add(episode_2)
    buffer.increment_ages()

    stored = buffer.sample(2)

    assert stored[0]["age"] == 2
    assert stored[1]["age"] == 1

def test_replay_buffer_age_bands():
    buffer = ReplayBuffer(capacity=10)

    assert buffer.age_band(0) == "recent"
    assert buffer.age_band(2) == "recent"

    assert buffer.age_band(3) == "medium"
    assert buffer.age_band(5) == "medium"

    assert buffer.age_band(6) == "old"
    assert buffer.age_band(20) == "old"

def test_replay_buffer_samples_across_age_bands():
    buffer = ReplayBuffer(capacity=10)

    episodes = [
        {"id": "recent_1", "age": 0, "domain": "a"},
        {"id": "recent_2", "age": 2, "domain": "a"},
        {"id": "medium_1", "age": 3, "domain": "b"},
        {"id": "medium_2", "age": 5, "domain": "b"},
        {"id": "old_1", "age": 6, "domain": "c"},
        {"id": "old_2", "age": 10, "domain": "c"},
    ]

    for episode in episodes:
        buffer.add(episode)

    sampled = buffer.sample_age_banded(size=3)

    assert len(sampled) == 3

    sampled_bands = {
        buffer.age_band(episode["age"])
        for episode in sampled
    }

    assert sampled_bands == {
        "recent",
        "medium",
        "old",
    }