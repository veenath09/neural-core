import pandas as pd

from data.depth_dataset import wide_table


def test_wide_table_keeps_only_complete_prompts():
    ok = pd.DataFrame({
        "prompt_id": ["p0", "p0", "p1"], "arm": ["a", "b", "a"], "gold": "A",
        "correct": [True, False, True], "cost_usd": [0.1, 0.2, 0.1], "reasoning_tokens": [0, 50, 0],
    })
    prompts = pd.DataFrame({"prompt_id": ["p0", "p1"], "split": "train", "source": "mmlu",
                            "step_type": "qa", "band": [1, 2]})
    t = wide_table(ok, prompts)
    assert t["prompt_id"].tolist() == ["p0"]
    assert t.loc[0, "correct_a"] and not t.loc[0, "correct_b"] and t.loc[0, "think_b"] == 50
