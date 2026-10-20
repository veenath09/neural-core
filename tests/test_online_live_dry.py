"""Task 8 dry run: the live loop completes, logs every episode, learns, and spends nothing (no network)."""

import json

import pandas as pd

from data.depth_collect import Arm
from scripts.online_live import Spend, SpendCapReached, fake_caller, run_live
from tests.test_train import tiny_data
from train.online import OnlineConfig
from train.train import train


def _setup(tmp_path):
    data = tiny_data()
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda r: None)
    pids = sorted({p for sk in data.train[:20] for p in sk.prompt_ids})
    questions = pd.DataFrame({"text": [f"q {p}" for p in pids], "source": "mmlu", "gold": "A"}, index=pids)
    arms = {m: Arm(m, "fake", m, "off", 1.0, 2.0) for m in data.world.model_ids}
    return data, questions, arms


def test_online_live_dry_run(tmp_path):
    data, questions, arms = _setup(tmp_path)
    spend = Spend(cap=float("inf"))
    out = tmp_path / "live"
    summary = run_live(tmp_path / "full-s0", data, list(data.world.model_ids), [], data.train[:20], questions,
                       fake_caller(questions), spend, out, OnlineConfig(update_every=4, replay_size=2),
                       explore_share=1.0, arms=arms)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert sum(r["type"] == "episode" for r in rows) == 20 and sum(r["type"] == "update" for r in rows) >= 1
    assert summary["spent_usd"] == 0.0 and summary["episodes"] == 20 and (out / "pool.json").exists()


def test_spend_cap_stops_cleanly_and_saves_state(tmp_path):
    data, questions, arms = _setup(tmp_path)

    def paid(arm, text):  # 1M input tokens at $1/M: $1 per call
        return "Answer: A", 1_000_000, 0, 0

    spend = Spend(cap=3.0)
    summary = run_live(tmp_path / "full-s0", data, list(data.world.model_ids), [], data.train[:20], questions, paid,
                       spend, tmp_path / "live", OnlineConfig(update_every=4), explore_share=1.0, arms=arms)
    assert summary["stopped"] and summary["spent_usd"] <= 3.0 and (tmp_path / "live" / "pool.json").exists()
    try:
        Spend(cap=1.0, spent=0.9).preflight(0.2)
        raise AssertionError("preflight should refuse")
    except SpendCapReached:
        pass
