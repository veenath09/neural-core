"""
Purpose:  GSM8K (grade-school maths) prompts for the reasoning-depth dataset, with exact numeric gold
          answers, so depth is also measured on multi-step reasoning, not only multiple choice.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   data/raw/routerbench_0shot.pkl (RouterBench "grade-school-math" prompts, kept word for word),
          openai/gsm8k on Hugging Face (MIT licence; downloaded to data/raw/gsm8k) for the gold answers.
Outputs:  data/processed/gsm8k_prompts.parquet: prompt_id, source="gsm8k", text, gold (number as text),
          split (72/8/20, seed 0, same rule as data.routerbench.split_prompts).
Notes:    RouterBench scores this task in 0.25 steps that do not mark correctness (gpt-4 scores 0.75 on
          most prompts), so it is not in prompts.parquet and its scores are not used. Gold is the number
          after "####" in the original GSM8K answer, matched to RouterBench by the question text. There
          is no RouterBench difficulty band for these prompts.

    python -m data.gsm8k
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from data.routerbench import DATA_DIR, RAW_DIR, parse_prompt, split_prompts

GSM8K_REPO = "openai/gsm8k"
GSM8K_FILES = ("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet")
OUT_FILE = "gsm8k_prompts.parquet"


def fetch_gold(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """question -> gold number (as text) from the original GSM8K train and test files."""
    import huggingface_hub

    local = Path(raw_dir) / "gsm8k"
    frames = [pd.read_parquet(huggingface_hub.hf_hub_download(GSM8K_REPO, f, repo_type="dataset", local_dir=local))
              for f in GSM8K_FILES]
    d = pd.concat(frames, ignore_index=True)
    d["gold"] = d["answer"].str.split("####").str[-1].str.strip().str.replace(",", "")
    return d.assign(key=d["question"].str.strip())[["key", "gold"]].drop_duplicates("key")


def question_of(raw_prompt) -> str:
    """RouterBench stores [instructions + examples, question]; the last part is the GSM8K question."""
    try:
        parts = ast.literal_eval(raw_prompt)
        return str(parts[-1]).strip()
    except (ValueError, SyntaxError, IndexError, TypeError):
        return ""


def build(raw_dir: Path = RAW_DIR, data_dir: Path = DATA_DIR, seed: int = 0) -> pd.DataFrame:
    raw = pd.read_pickle(Path(raw_dir) / "routerbench_0shot.pkl")
    g = raw[raw["eval_name"] == "grade-school-math"]
    table = pd.DataFrame({
        "prompt_id": g["sample_id"].astype(str).to_numpy(),
        "source": "gsm8k",
        "text": g["prompt"].map(parse_prompt).to_numpy(),
        "key": g["prompt"].map(question_of).to_numpy(),
    })
    table = table.merge(fetch_gold(raw_dir), on="key", how="inner").drop(columns="key")
    table = table.drop_duplicates("text").reset_index(drop=True)
    table["split"] = split_prompts(table, seed)
    out = Path(data_dir) / OUT_FILE
    table.to_parquet(out, index=False)
    return table


if __name__ == "__main__":
    t = build()
    print(f"{len(t)} GSM8K prompts with gold", t["split"].value_counts().to_dict())
