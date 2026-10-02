"""
Purpose:  Evaluate the reasoning-depth outcome dataset: data quality, per-option accuracy and cost,
          and whether thinking harder helps (overall and on hard questions).
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   data/raw/depth/outcomes*.jsonl (from data.depth_collect), prompt metadata via data.depth_dataset.
Outputs:  printed tables; runs/depth_report.csv (per-option summary).
Notes:    Uses the latest successful call per (prompt, option). "Complete" prompts are those with an
          answer from every option in the run, which is what episode building needs. Difficulty band =
          number of the 4 RouterBench pool models that got the prompt wrong (0 easy .. 4 hardest).

    python -m eval.depth_report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.depth_dataset import all_prompts, latest_successful

DEPTH_ORDER = {"off": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4}


def summary(ok: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    calls = rows.groupby("arm").size().rename("calls")
    errors = rows[rows["error"].notna()].groupby("arm").size().rename("errors")
    g = ok.groupby("arm")
    out = pd.DataFrame({
        "model": g["model"].first(), "depth": g["depth"].first(), "n": g.size(),
        "accuracy": g["correct"].mean(), "unparsed": g["answer"].apply(lambda s: s.isna().mean()),
        "think_tokens": g["reasoning_tokens"].mean(), "out_tokens": g["output_tokens"].mean(),
        "latency_s": g["latency_ms"].mean() / 1000, "cost_per_q": g["cost_usd"].mean(),
    }).join(calls).join(errors).fillna({"errors": 0})
    out["depth_rank"] = out["depth"].map(DEPTH_ORDER)
    return out.sort_values(["cost_per_q"])


def main() -> None:
    rows, ok = latest_successful()
    bands = all_prompts().set_index("prompt_id")["band"]
    ok = ok.assign(band=ok["prompt_id"].map(bands))
    n_arms = ok["arm"].nunique()
    per_prompt = ok.groupby("prompt_id")["arm"].nunique()
    complete = per_prompt[per_prompt == n_arms].index
    s = summary(ok, rows)

    print(f"calls {len(rows)} · successful {len(ok)} · options {n_arms} · prompts {per_prompt.size} "
          f"(complete {len(complete)}) · spent ${ok['cost_usd'].sum():.3f} at list price\n")
    print("1) per option (cheapest first)")
    cols = ["n", "accuracy", "unparsed", "think_tokens", "latency_s", "cost_per_q", "errors"]
    print(s[cols].to_string(float_format=lambda v: f"{v:.4f}"))

    lo = s.loc[s.groupby("model")["depth_rank"].idxmin()].index
    hi = s.loc[s.groupby("model")["depth_rank"].idxmax()].index
    print("\n2) does thinking help? accuracy by depth within each model")
    piv = ok.pivot_table(index="model", columns="depth", values="correct", aggfunc="mean")
    print(piv[[d for d in DEPTH_ORDER if d in piv.columns]].to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n2b) accuracy by depth per task (all models): shallowest vs deepest level")
    task = ok.assign(level=ok["arm"].map(lambda a: "shallowest" if a in lo else "deepest" if a in hi else None))
    print(task.dropna(subset=["level"]).pivot_table(index="source", columns="level", values="correct",
                                                    aggfunc="mean")[["shallowest", "deepest"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n3) by difficulty band (0 easy .. 4 hardest; multiple choice only): shallowest vs deepest level")
    shallow = ok[ok["arm"].isin(lo)].groupby("band")["correct"].mean().rename("shallowest")
    deep = ok[ok["arm"].isin(hi)].groupby("band")["correct"].mean().rename("deepest")
    counts = ok[ok["arm"].isin(lo)].groupby("band")["prompt_id"].nunique().rename("prompts")
    print(pd.concat([counts, shallow, deep], axis=1).to_string(float_format=lambda v: f"{v:.3f}"))

    oracle = ok[ok["prompt_id"].isin(complete)].groupby("prompt_id")["correct"].max().mean() if len(complete) else float("nan")
    print(f"\n4) ceiling: on complete prompts, {oracle:.3f} of questions are solved by at least one option")
    Path("runs").mkdir(exist_ok=True)
    s.to_csv("runs/depth_report.csv")


if __name__ == "__main__":
    main()
