"""
Purpose:  Turn the reasoning-depth dataset into a simulator pool: each (model, depth) option is one model
          card, so the unchanged simulator, controller and trainer can learn when deeper thinking pays.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   data/processed/depth/table.parquet and manifest.json (data.depth_dataset), prompt texts from
          data/processed/prompts.parquet and gsm8k_prompts.parquet.
Outputs:  data/processed/depth_pool/ with the same files as the RouterBench pool, so every tool takes it
          through --data-dir:
            prompts.parquet   prompt_id, source, step_type, text, correct_i, cost_i, band (0..4), split
            pool.json         one ModelCard per option, cheapest first
            embeddings.npy, embedding_ids.json   frozen bge-small, same encoder as the RouterBench pool
            episodes_{train,val,test}.parquet    same generator, mix and budget rule as the RouterBench pool
Notes:    Options are ordered by mean train cost, so "escalate" means a strictly more expensive option:
          the same model thinking deeper, or a stronger model. Card capability = train accuracy per step
          type, log10(mean cost), then two depth features (depth level / 4, log10(1 + mean thinking
          tokens) / 4). Band = how many of the options fail, bucketed to 0..4 (0; 1-2; 3-6; 7-12; 13+)
          because correlated episodes group prompts by step type x band. Costs are the measured list-price
          cost of each call (price date in the manifest), not RouterBench's.

    python -m data.depth_pool
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from controller.contracts import ModelCard
from data.routerbench import DATA_DIR

DEPTH_SOURCES = ("mmlu", "hellaswag", "arc-challenge", "winogrande", "gsm8k")
DEPTH_LEVEL = {"off": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4}
BAND_EDGES = (1, 3, 7, 13)  # options failing: 0 | 1-2 | 3-6 | 7-12 | 13+
POOL_DIR = Path(DATA_DIR) / "depth_pool"


def band_bucket(n_wrong: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(BAND_EDGES), n_wrong, side="right")


def prompt_texts(data_dir: Path = Path(DATA_DIR)) -> pd.Series:
    parts = [pd.read_parquet(data_dir / f, columns=["prompt_id", "text"])
             for f in ("prompts.parquet", "gsm8k_prompts.parquet")]
    return pd.concat(parts).drop_duplicates("prompt_id").set_index("prompt_id")["text"]


def build_table(wide: pd.DataFrame, options: list[dict], texts: pd.Series) -> tuple[pd.DataFrame, list[ModelCard]]:
    names = [c[len("correct_"):] for c in wide.columns if c.startswith("correct_")]
    train = wide[wide["split"] == "train"]
    names = sorted(names, key=lambda a: (train[f"cost_{a}"].mean(), a))  # cheapest first
    meta = {o["name"]: o for o in options}
    out = pd.DataFrame({
        "prompt_id": wide["prompt_id"].astype(str).to_numpy(),
        "source": wide["source"].to_numpy(),
        "step_type": wide["source"].map({s: i for i, s in enumerate(DEPTH_SOURCES)}).to_numpy(),
        "text": wide["prompt_id"].map(texts).to_numpy(),
    })
    if out["text"].isna().any() or out["step_type"].isna().any():
        raise ValueError("prompts without text or with an unknown source")
    for i, a in enumerate(names):
        out[f"correct_{i}"] = wide[f"correct_{a}"].astype(bool).to_numpy()
        out[f"cost_{i}"] = wide[f"cost_{a}"].astype(float).to_numpy()
    n_wrong = (~out[[f"correct_{i}" for i in range(len(names))]].to_numpy()).sum(axis=1)
    out["band"] = band_bucket(n_wrong)
    out["split"] = wide["split"].to_numpy()

    is_train = out["split"] == "train"
    cards = []
    for i, a in enumerate(names):
        acc = [float(out.loc[is_train & (out["step_type"] == t), f"correct_{i}"].mean())
               for t in range(len(DEPTH_SOURCES))]
        mean_cost = float(out.loc[is_train, f"cost_{i}"].mean())
        think = float(train[f"think_{a}"].mean())
        o = meta[a]
        depth = DEPTH_LEVEL.get(o["depth"], 4)
        cards.append(ModelCard(model_id=a, provider=o["provider"], tier=i, mean_cost_usd=mean_cost,
                               price_source="measured tokens x provider list price", price_date=o.get(
                                   "price_date", "") or "",
                               capability=acc + [math.log10(mean_cost), depth / 4, math.log10(1 + think) / 4]))
    return out, cards


def build(data_dir: Path = Path(DATA_DIR), out_dir: Path = POOL_DIR, encode=None, seed: int = 0) -> None:
    from data.embed import build_cache, sentence_transformer_encoder
    from data.episodes import build_all

    wide = pd.read_parquet(data_dir / "depth" / "table.parquet")
    manifest = json.loads((data_dir / "depth" / "manifest.json").read_text())
    price_date = manifest["collection"]["price_date"]
    options = [o | {"price_date": price_date} for o in manifest["options"]]
    table, cards = build_table(wide, options, prompt_texts(data_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "prompts.parquet", index=False)
    (out_dir / "pool.json").write_text(json.dumps([asdict(c) for c in cards], indent=2))
    build_cache(table["prompt_id"].tolist(), table["text"].tolist(), encode or sentence_transformer_encoder(),
                data_dir=out_dir)
    build_all(data_dir=out_dir, seed=seed)
    print(table.groupby(["source", "split"]).size().unstack().to_string())
    print("band", table["band"].value_counts().sort_index().to_dict())
    for c in cards:
        print(f"{c.tier:2d} {c.model_id:32s} cost={c.mean_cost_usd:.2e} acc={[round(x, 3) for x in c.capability[:5]]}")


if __name__ == "__main__":
    build()
