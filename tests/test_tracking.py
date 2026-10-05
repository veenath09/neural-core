import json

import pytest

from tests.test_train import tiny_data
from train.tracking import Tracker, flatten
from train.train import train


def test_flatten_keeps_numbers_lists_and_nested_values():
    row = {"epoch": 2, "val_quality": 0.7, "ok": True, "val_choice_share": [0.25, 0.75], "name": "x",
           "reference": {"oracle_quality": 0.9}, "bad": float("nan"), "mixed": [1, "a"]}
    assert flatten(row) == {"epoch": 2.0, "val_quality": 0.7, "ok": 1.0, "val_choice_share_0": 0.25,
                            "val_choice_share_1": 0.75, "reference.oracle_quality": 0.9}


def test_training_logs_metrics_params_and_alarms(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = Tracker("test/full-s0", params={"config": "full", "model": {"d": 128}}, tags={"pool": "toy"}, uri=uri,
                      experiment="test")
    summary = train("full", seed=0, epochs=2, batch_size=32, data=tiny_data(), out_dir=tmp_path, log=lambda r: None,
                    tracker=tracker)
    tracker.end()
    assert summary["mlflow_run_id"] == tracker.run_id
    assert json.loads((tmp_path / "full-s0" / "metrics.json").read_text())["mlflow_run_id"] == tracker.run_id
    client = mlflow.tracking.MlflowClient(uri)
    run = client.get_run(tracker.run_id)
    assert run.data.params["model.d"] == "128" and run.data.tags["pool"] == "toy"
    steps = [m.step for m in client.get_metric_history(tracker.run_id, "val_quality")]
    assert steps == [0, 1]
    assert "val_choice_share_0" in run.data.metrics and "alarm_count" in run.data.metrics
    assert "reference.oracle_quality" in run.data.metrics


def test_training_without_tracker_is_unchanged(tmp_path):
    summary = train("full", seed=0, epochs=1, batch_size=32, data=tiny_data(), out_dir=tmp_path, log=lambda r: None)
    assert "mlflow_run_id" not in summary
