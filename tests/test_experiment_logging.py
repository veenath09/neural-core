import json

from train.experiment_logging import append_experiment_record


def test_append_experiment_record_persists_jsonl_history(tmp_path):
    log_path = tmp_path / "c2_updates.jsonl"

    first = {
        "episode_count": 1,
        "fresh_loss": 0.5,
        "reward_mean": 0.8,
        "accepted": True,
    }

    second = {
        "episode_count": 2,
        "fresh_loss": 0.4,
        "reward_mean": 0.7,
        "accepted": False,
    }

    append_experiment_record(
        log_path,
        first,
    )

    append_experiment_record(
        log_path,
        second,
    )

    lines = log_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 2

    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second
    