import json
import urllib.error

import pandas as pd
import pytest

from data.depth_collect import (
    ARMS,
    Arm,
    Budget,
    call_cost,
    collect,
    done_keys,
    parse_answer,
    parse_number,
    recover_gold,
    score,
    spent,
)

ARM = Arm("toy@low", "groq", "toy", "low", 1.0, 10.0, {"reasoning_effort": "low"})


@pytest.mark.parametrize("text,choices,want", [
    ("C", "ABCD", "C"),
    ("['B']", "ABCD", "B"),
    ("The best option is A. Answer: D", "ABCD", "D"),
    ("**Answer:** (B)", "ABCD", "B"),
    ("I think A, but on reflection B", "ABCD", "B"),
    ("Option C", "AB", None),
    ("", "ABCD", None),
    (None, "ABCD", None),
    ("no letter here", "ABCD", None),
])
def test_parse_answer(text, choices, want):
    assert parse_answer(text, choices) == want


def test_recover_gold_uses_only_correct_models_and_needs_agreement():
    assert recover_gold(["A", "B", "['A']"], [True, False, True]) == "A"
    assert recover_gold(["A", "B"], [True, True]) is None  # disagreement
    assert recover_gold(["A", "B"], [False, False]) is None  # nobody correct
    assert recover_gold(["hmm", None], [True, True]) is None  # unparsable


def test_call_cost_counts_reasoning_tokens_at_output_price():
    assert call_cost(ARM, 1_000_000, 0, 0) == pytest.approx(1.0)
    assert call_cost(ARM, 0, 100_000, 100_000) == pytest.approx(2.0)


def test_arms_are_cheapest_first_and_unique():
    names = [a.name for a in ARMS]
    assert len(names) == len(set(names))
    assert ARMS[-1].model == "gemini-3.1-pro-preview"
    assert all(a.provider in {"groq", "gemini", "openrouter", "kimi", "openai", "anthropic"} for a in ARMS)


def test_budget_stops_at_cap():
    b = Budget(1.0, already=0.4)
    assert b.can_spend()
    b.add(0.7)
    assert not b.can_spend()


def prompts_frame(n=6):
    return pd.DataFrame({"prompt_id": [f"p{i}" for i in range(n)], "split": "train", "source": "mmlu",
                         "text": [f"q{i}" for i in range(n)], "gold": "A"})


def test_collect_writes_outcomes_resumes_and_respects_cap(tmp_path):
    out = tmp_path / "o.jsonl"
    calls = []

    def fake(arm, text, env):
        calls.append((arm.name, text))
        return "Answer: A", 1000, 10, 90  # cost = (1000*1 + 100*10)/1e6 = 0.002

    arms = [ARM, Arm("toy@high", "groq", "toy", "high", 1.0, 10.0)]
    collect(prompts_frame(), arms, cap=1.0, out_path=out, env={}, call=fake, workers_per_provider=2)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 12 and all(r["correct"] and r["reasoning_tokens"] == 90 for r in rows)
    assert spent(out) == pytest.approx(12 * 0.002)
    assert len(done_keys(out)) == 12
    calls.clear()
    collect(prompts_frame(), arms, cap=1.0, out_path=out, env={}, call=fake)  # resume: nothing left to do
    assert calls == []

    out2 = tmp_path / "o2.jsonl"
    collect(prompts_frame(), arms, cap=0.005, out_path=out2, env={}, call=fake, workers_per_provider=1)
    assert spent(out2) <= 0.005 + 0.002  # at most one call past the cap
    assert len(out2.read_text().splitlines()) < 12


def test_failed_calls_are_logged_free_and_retried_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("data.depth_collect.time.sleep", lambda s: None)
    out = tmp_path / "o.jsonl"

    def broken(arm, text, env):
        raise urllib.error.HTTPError("u", 400, "bad", {}, None)

    collect(prompts_frame(2), [ARM], cap=1.0, out_path=out, env={}, call=broken)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["error"] and r["cost_usd"] == 0.0 and not r["correct"] for r in rows)
    assert done_keys(out) == set()  # errors are not "done", so a resume retries them


def test_sdk_client_errors_are_not_retried(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("data.depth_collect.time.sleep", lambda s: sleeps.append(s))

    class LimitError(Exception):
        status_code = 400

    calls = []

    def limited(arm, text, env):
        calls.append(1)
        raise LimitError("usage limit reached")

    collect(prompts_frame(1), [ARM], cap=1.0, out_path=tmp_path / "o.jsonl", env={}, call=limited)
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("text,want", [
    ("48 + 24 = 72 clips. Answer: 72", "72"),
    ("The total is $1,250.", "1250"),
    ("so 3 + 4 = 7\n#### 7", "7"),
    ("\\boxed{18}", "18"),
    ("The answer is 5.0", "5"),
    ("first 10, then 12, Answer: -3", "-3"),
    ("The baker must arrive by 2:00 am. Answer: 2:00", "2"),
    ("total \\boxed{2{,}870}", "2870"),
    ("so 2\\,870 fish", "2870"),
    ("no number", None),
    (None, None),
])
def test_parse_number(text, want):
    assert parse_number(text) == want


def test_score_dispatches_by_task():
    assert score("Answer: 72", "gsm8k", "72") == ("72", True)
    assert score("Answer: 72.00", "gsm8k", "72")[1]
    assert score("Answer: B", "mmlu", "B") == ("B", True)
    assert score("Answer: B", "winogrande", "A") == ("B", False)
