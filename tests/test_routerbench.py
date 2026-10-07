import numpy as np
import pandas as pd
import pytest

from data.routerbench import POOL, OutcomeTable, build_table, make_model_cards, split_prompts

COSTS = (1e-5, 2e-5, 3e-5, 4e-5)


def row(sid, eval_name, prompt, scores, costs=COSTS):
    r = {"sample_id": sid, "eval_name": eval_name, "prompt": prompt}
    for m, s, c in zip(POOL, scores, costs):
        r[m] = s
        r[f"{m}|total_cost"] = c
    return r


def raw_frame(rows):
    return pd.DataFrame(rows)


def test_build_table_groups_mmlu_and_drops_other_sources():
    raw = raw_frame([
        row("a", "mmlu-anatomy", "['q1']", (0, 0, 1, 1)),
        row("b", "mmlu-law", "['q2']", (1, 1, 1, 1)),
        row("c", "grade-school-math", "['q3']", (0.25, 0.5, 0.75, 1)),
        row("d", "hellaswag", "['q4']", (0, 0, 0, 0)),
    ])
    t = build_table(raw)
    assert list(t.prompt_id) == ["a", "b", "d"]
    assert list(t.source) == ["mmlu", "mmlu", "hellaswag"]
    assert list(t.step_type) == [0, 0, 1]
    assert list(t.band) == [2, 0, 4]
    assert t.correct_3.dtype == bool


def test_build_table_parses_list_prompts():
    t = build_table(raw_frame([row("a", "mbpp", "['part one', 'part two']", (1, 1, 1, 1))]))
    assert t.text.iloc[0] == "part one\npart two"


def test_build_table_drops_missing_results():
    t = build_table(raw_frame([
        row("a", "mbpp", "['x']", (1, np.nan, 1, 1)),
        row("b", "mbpp", "['y']", (1, 1, 1, 1)),
    ]))
    assert list(t.prompt_id) == ["b"]


def test_build_table_rejects_non_binary_scores_in_kept_source():
    with pytest.raises(ValueError, match="non-binary"):
        build_table(raw_frame([row("a", "winogrande", "['x']", (0.5, 1, 1, 1))]))


def test_build_table_dedupes_prompt_text():
    t = build_table(raw_frame([
        row("a", "mbpp", "['same']", (1, 1, 1, 1)),
        row("b", "mbpp", "['same']", (0, 1, 1, 1)),
    ]))
    assert list(t.prompt_id) == ["a"]


def synthetic_table(n=1000):
    rng = np.random.default_rng(0)
    raw = raw_frame([
        row(f"s{i}", "hellaswag" if i % 3 else "arc-challenge", f"['prompt {i}']",
            tuple(int(x) for x in rng.random(4) < (0.3, 0.5, 0.7, 0.9)))
        for i in range(n)
    ])
    t = build_table(raw)
    t["split"] = split_prompts(t, seed=0)
    return t


def test_split_fractions_and_coverage():
    t = synthetic_table()
    counts = t.split.value_counts(normalize=True)
    assert set(t.split) == {"train", "val", "test"}
    assert counts["test"] == pytest.approx(0.2, abs=0.01)
    assert counts["val"] == pytest.approx(0.08, abs=0.01)


def test_split_is_deterministic_and_text_disjoint():
    t = synthetic_table()
    assert (split_prompts(t, seed=0) == t.split).all()
    assert not (split_prompts(t, seed=1) == t.split).all()
    assert t.groupby("text").split.nunique().max() == 1


def test_model_cards_ordered_by_tier_with_capability():
    t = synthetic_table()
    cards = make_model_cards(t)
    assert [c.model_id for c in cards] == list(POOL)
    assert [c.tier for c in cards] == [0, 1, 2, 3]
    assert len(cards[0].capability) == 5 + 1
    assert cards[3].capability[1] > cards[0].capability[1]  # hellaswag accuracy rises with tier


def test_outcome_table_from_frame():
    t = synthetic_table(30)
    ot = OutcomeTable.from_frame(t)
    assert ot.correct.shape == (len(t), 4) and ot.n_models == 4
    assert ot.index[t.prompt_id.iloc[5]] == 5
    assert (ot.band == t.band.to_numpy()).all()


def test_fetch_raw_reuses_local_file_without_downloading(tmp_path, monkeypatch):
    import huggingface_hub

    from data.routerbench import HF_FILE, fetch_raw

    (tmp_path / HF_FILE).write_bytes(b"x")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("should not download"))
    assert fetch_raw(tmp_path) == tmp_path / HF_FILE


def test_fetch_raw_downloads_into_project_dir_not_hf_cache(tmp_path, monkeypatch):
    import huggingface_hub

    from data.routerbench import HF_FILE, fetch_raw

    def fake_download(repo_id, filename, repo_type, local_dir):
        assert local_dir == tmp_path
        (local_dir / filename).write_bytes(b"x")
        return str(local_dir / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert fetch_raw(tmp_path) == tmp_path / HF_FILE
