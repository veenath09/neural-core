"""
Purpose:  Turn raw depth-collection calls into the reasoning-depth dataset, and record every parameter
          that produced it in a manifest, so the dataset is reproducible and citable.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   data/raw/depth/outcomes.jsonl and outcomes_gsm8k.jsonl (data.depth_collect),
          data/processed/prompts.parquet and gsm8k_prompts.parquet.
Outputs:  data/processed/depth/outcomes.parquet  one row per (prompt, option): latest successful call
          data/processed/depth/table.parquet     one row per complete prompt, wide: correct_<option>,
                                                 cost_<option>, think_<option>; plus split, source, band, gold
                                                 (band is empty for GSM8K: RouterBench has no usable scores)
          data/processed/depth/manifest.json     options (provider, model, depth, request params, list
                                                 prices, price date), collection settings, counts, spend
Notes:    A prompt is "complete" when every option in the run answered it; only complete prompts enter
          the wide table, because episodes need an outcome for every option. Prompts keep the RouterBench
          split, so there is no train/test leakage.

    python -m data.depth_dataset --skip-provider groq   # table without gpt-oss (Groq daily limit)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data.depth_collect import (
    ARMS,
    MAX_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS_BY_PROVIDER,
    MC_SOURCES,
    OUT_PATHS,
    PRICE_DATE,
    score,
)
from data.routerbench import DATA_DIR

DEPTH_DIR = Path(DATA_DIR) / "depth"


def latest_successful(paths=tuple(OUT_PATHS.values())) -> tuple[pd.DataFrame, pd.DataFrame]:
    lines = [line for p in map(Path, paths) if p.exists() for line in p.read_text().splitlines()]
    rows = pd.DataFrame([json.loads(line) for line in lines])
    if "response" in rows:  # re-score saved replies with the current parser; older rows keep their score
        has = rows["response"].notna() & rows["error"].isna()
        scored = [score(r, src, g) for r, src, g in zip(rows.loc[has, "response"], rows.loc[has, "source"],
                                                        rows.loc[has, "gold"])]
        rows.loc[has, "answer"] = [a for a, _ in scored]
        rows.loc[has, "correct"] = [c for _, c in scored]
    ok = rows[rows["error"].isna()].sort_values("ts").drop_duplicates(["prompt_id", "arm"], keep="last")
    return rows, ok.reset_index(drop=True)


def wide_table(ok: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    arms = sorted(ok["arm"].unique())
    per_prompt = ok.groupby("prompt_id")["arm"].nunique()
    complete = per_prompt[per_prompt == len(arms)].index
    sub = ok[ok["prompt_id"].isin(complete)]
    parts = [sub.pivot(index="prompt_id", columns="arm", values=col).add_prefix(prefix)
             for col, prefix in [("correct", "correct_"), ("cost_usd", "cost_"), ("reasoning_tokens", "think_")]]
    meta = prompts.set_index("prompt_id").reindex(columns=["split", "source", "step_type", "band"]).loc[complete]
    gold = sub.groupby("prompt_id")["gold"].first()
    return pd.concat([meta, gold] + parts, axis=1).reset_index()


def all_prompts(data_dir: Path = Path(DATA_DIR)) -> pd.DataFrame:
    """Prompt metadata for both tasks; GSM8K has no step_type or band."""
    mc = pd.read_parquet(data_dir / "prompts.parquet", columns=["prompt_id", "split", "source", "step_type", "band"])
    gsm = data_dir / "gsm8k_prompts.parquet"
    parts = [mc] + ([pd.read_parquet(gsm, columns=["prompt_id", "split", "source"])] if gsm.exists() else [])
    return pd.concat(parts, ignore_index=True)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build(out_dir: Path = DEPTH_DIR, skip_providers: tuple[str, ...] = ()) -> dict:
    """outcomes.parquet keeps every successful call; table.parquet uses the options not skipped."""
    rows, ok = latest_successful()
    prompts = all_prompts()
    table = wide_table(ok[~ok["provider"].isin(skip_providers)], prompts)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok.to_parquet(out_dir / "outcomes.parquet", index=False)
    table.to_parquet(out_dir / "table.parquet", index=False)
    used = set(ok["arm"])
    manifest = {
        "dataset": "reasoning-depth outcomes on RouterBench multiple-choice prompts",
        "created_from_commit": git_commit(),
        "python": platform.python_version(),
        "collection": {
            "tasks": list(MC_SOURCES) + ["gsm8k"],
            "prompt_source": "RouterBench 0-shot prompts, word for word; our 72/8/20 split (seed 0) "
                             "(data/processed/prompts.parquet, gsm8k_prompts.parquet)",
            "prompt_selection": "multiple choice: stratified by split and task, seed 0, only prompts whose gold "
                                "letter is recoverable; gsm8k: stratified by split, seed 0",
            "gold_answer": {"multiple_choice": "letter given by the RouterBench models scored correct (no disagreements)",
                            "gsm8k": "final number after '####' in openai/gsm8k (MIT), matched by question text"},
            "scoring": {"multiple_choice": "exact match of the final answer letter (A-D; A-B for winogrande)",
                        "gsm8k": "numeric equality of the final number ('Answer: N', '#### N', boxed, else last number)"},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_output_tokens_by_provider": MAX_OUTPUT_TOKENS_BY_PROVIDER,
            "price_date": PRICE_DATE,
            "cost": "tokens x list price (input, output incl. thinking); free-tier calls are still priced",
            "single_sample": "one call per (prompt, option), provider default sampling",
        },
        "options": [asdict(a) | {"collected": a.name in used} for a in ARMS],
        "counts": {
            "calls": int(len(rows)),
            "successful": int(len(ok)),
            "failed_calls": int(rows["error"].notna().sum()),
            "prompts_touched": int(ok["prompt_id"].nunique()),
            "complete_prompts": int(len(table)),
            "complete_by_split": table["split"].value_counts().to_dict() if len(table) else {},
            "complete_by_task": table["source"].value_counts().to_dict() if len(table) else {},
            "options_collected": sorted(used),
            "options_in_table": sorted(set(ok.loc[~ok["provider"].isin(skip_providers), "arm"])),
            "providers_left_out_of_table": list(skip_providers),
        },
        "spend_usd_list_price": round(float(rows["cost_usd"].sum()), 4),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-provider", nargs="*", default=[], help="providers left out of table.parquet")
    m = build(skip_providers=tuple(ap.parse_args().skip_provider))
    print(json.dumps(m["counts"], indent=2), f"\nspend ${m['spend_usd_list_price']}")
