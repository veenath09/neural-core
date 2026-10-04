"""
Purpose:  Grow the depth pool's training split with synthetic questions from the same distribution, by
          hot-deck imputation: an unused RouterBench train question borrows the 19-option outcomes of the
          most similar real question we collected.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   data/processed/depth_pool/ (real pool, data.depth_pool), data/raw/routerbench_0shot.pkl
          (11 RouterBench model scores per question), data/processed/prompts.parquet,
          gsm8k_prompts.parquet, embeddings.npy (RouterBench prompt embeddings).
Outputs:  data/processed/depth_pool_aug/: same files as depth_pool. prompts.parquet adds `synthetic` and
          `donor_id`; val and test are the real questions only, unchanged, so their episodes are identical
          to depth_pool's. pool.json is copied from the real pool (cards use real data only).
          synth_report.json: how well imputed outcomes predict real ones (leave-one-out on real train).
Notes:    Donor = a real train question of the same task whose 11 RouterBench model scores are closest
          (L1), ties broken by bge text similarity. Leave-one-out on real train: P(real right | borrowed
          right) - P(real right | borrowed wrong) = +0.18, vs +0.03 when matching on the mean-score bin and
          -0.005 for a random same-task donor (synth_report.json). Copying a
          whole outcome row keeps the joint pattern across options and the link between question text
          and difficulty. Synthetic rows are training data only and are labelled; every reported number
          uses real val and test questions. Costs are the donor's measured costs.

    python -m data.depth_synth --factor 10
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from data.depth_pool import DEPTH_SOURCES, POOL_DIR
from data.routerbench import DATA_DIR, RAW_DIR

AUG_DIR = Path(DATA_DIR) / "depth_pool_aug"


def routerbench_scores(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """sample_id -> the 11 RouterBench model scores (0/1, GSM8K in 0.25 steps)."""
    raw = pd.read_pickle(Path(raw_dir) / "routerbench_0shot.pkl")
    models = [c for c in raw.columns if "|" not in c and c not in ("prompt", "eval_name", "oracle_model_to_route_to", "sample_id")]
    return raw.set_index("sample_id")[models].astype(float)


TEXT_TIEBREAK = 1e-3


def find_donors(q_source: np.ndarray, q_scores: np.ndarray, q_emb: np.ndarray, d_source: np.ndarray,
                d_scores: np.ndarray, d_emb: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Index of the donor for each query: same source, closest 11-model score pattern, then most similar
    text (embeddings are unit-normalised, so the dot product is the cosine)."""
    dist = np.abs(np.nan_to_num(q_scores)[:, None, :] - np.nan_to_num(d_scores)[None, :, :]).sum(-1)
    score = -dist + TEXT_TIEBREAK * (q_emb @ d_emb.T)
    score = np.where(q_source[:, None] == d_source[None, :], score, -np.inf)
    if exclude_self:
        np.fill_diagonal(score, -np.inf)
    return score.argmax(axis=1)


def agreement(real: np.ndarray, imputed: np.ndarray) -> dict:
    """How much imputed outcomes say about real ones, per option, averaged."""
    p_hit = [real[imputed[:, i], i].mean() if imputed[:, i].any() else np.nan for i in range(real.shape[1])]
    p_miss = [real[~imputed[:, i], i].mean() if (~imputed[:, i]).any() else np.nan for i in range(real.shape[1])]
    return {"exact_row_match": float((real == imputed).all(axis=1).mean()),
            "cell_agreement": float((real == imputed).mean()),
            "p_real_right_if_imputed_right": float(np.nanmean(p_hit)),
            "p_real_right_if_imputed_wrong": float(np.nanmean(p_miss)),
            "all_right_real": float(real.all(axis=1).mean()), "all_right_imputed": float(imputed.all(axis=1).mean())}


def validate(real: pd.DataFrame, emb: np.ndarray, scores: pd.DataFrame, n_opt: int, seed: int = 0) -> dict:
    """Leave-one-out on the real train questions: impute each from the others, compare with its real row."""
    tr = real[real["split"] == "train"].reset_index(drop=True)
    e = emb[tr.index.to_numpy()] if len(emb) == len(tr) else emb
    y = tr[[f"correct_{i}" for i in range(n_opt)]].to_numpy(bool)
    src, sc = tr["source"].to_numpy(), scores.loc[tr["prompt_id"]].to_numpy()
    donor = find_donors(src, sc, e, src, sc, e, exclude_self=True)
    rng = np.random.default_rng(seed)
    rand = np.array([rng.choice(np.flatnonzero((src == s) & (np.arange(len(tr)) != k))) for k, s in enumerate(src)])
    return {"n": len(tr), "matched_donor": agreement(y, y[donor]), "random_same_task_donor": agreement(y, y[rand])}


def build(factor: int = 10, seed: int = 0, pool_dir: Path = POOL_DIR, out_dir: Path = AUG_DIR,
          data_dir: Path = Path(DATA_DIR), encode=None) -> dict:
    from data.embed import build_cache, load_cache, sentence_transformer_encoder
    from data.episodes import build_all

    real = pd.read_parquet(pool_dir / "prompts.parquet")
    n_opt = sum(c.startswith("correct_") for c in real.columns)
    real_emb, real_idx = load_cache(pool_dir)
    real_emb = np.asarray(real_emb)[[real_idx[p] for p in real["prompt_id"]]]
    scores = routerbench_scores()

    train_mask = (real["split"] == "train").to_numpy()
    report = validate(real[train_mask].reset_index(drop=True), real_emb[train_mask], scores, n_opt, seed)

    # candidates: train-split questions of the same tasks that we never sent to the options
    mc = pd.read_parquet(data_dir / "prompts.parquet", columns=["prompt_id", "source", "text", "split"])
    gsm = pd.read_parquet(data_dir / "gsm8k_prompts.parquet", columns=["prompt_id", "source", "text", "split"])
    cand = pd.concat([mc, gsm], ignore_index=True)
    cand = cand[(cand["split"] == "train") & cand["source"].isin(DEPTH_SOURCES) & ~cand["prompt_id"].isin(real["prompt_id"])]
    cand = cand[cand["prompt_id"].isin(scores.index)]
    want = (real.loc[train_mask, "source"].value_counts() * factor).astype(int)  # same task mix as real train
    rng = np.random.default_rng(seed)
    picked = pd.concat([cand[cand["source"] == s].sample(min(n, int((cand["source"] == s).sum())), random_state=rng)
                        for s, n in want.items()], ignore_index=True)

    encode = encode or sentence_transformer_encoder()
    syn_emb = np.asarray(encode(picked["text"].tolist()), dtype=np.float32)
    donors_tr = real[train_mask].reset_index(drop=True)
    d_idx = find_donors(picked["source"].to_numpy(), scores.loc[picked["prompt_id"]].to_numpy(), syn_emb,
                        donors_tr["source"].to_numpy(), scores.loc[donors_tr["prompt_id"]].to_numpy(),
                        real_emb[train_mask])
    donor_rows = donors_tr.iloc[d_idx].reset_index(drop=True)
    outcome_cols = [c for c in real.columns if c.startswith(("correct_", "cost_"))] + ["band"]
    syn = pd.concat([picked[["prompt_id", "source", "text"]].reset_index(drop=True), donor_rows[outcome_cols]], axis=1)
    syn["step_type"] = syn["source"].map({s: i for i, s in enumerate(DEPTH_SOURCES)})
    syn["split"], syn["synthetic"], syn["donor_id"] = "train", True, donor_rows["prompt_id"].to_numpy()

    aug = pd.concat([real.assign(synthetic=False, donor_id=None), syn[real.columns.tolist() + ["synthetic", "donor_id"]]],
                    ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    aug.to_parquet(out_dir / "prompts.parquet", index=False)
    shutil.copy(pool_dir / "pool.json", out_dir / "pool.json")
    emb = np.concatenate([real_emb, syn_emb])
    build_cache(aug["prompt_id"].tolist(), list(emb), lambda texts: np.asarray(texts, dtype=np.float32), data_dir=out_dir)
    build_all(data_dir=out_dir, seed=seed)

    report |= {"factor": factor, "synthetic_questions": len(syn), "real_train": int(train_mask.sum()),
               "synthetic_by_task": syn["source"].value_counts().to_dict(),
               "distinct_donors_used": int(syn["donor_id"].nunique())}
    (out_dir / "synth_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=10, help="synthetic questions per real train question")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    if a.validate_only:
        real = pd.read_parquet(POOL_DIR / "prompts.parquet")
        from data.embed import load_cache
        e, idx = load_cache(POOL_DIR)
        tr = real[real["split"] == "train"].reset_index(drop=True)
        n = sum(c.startswith("correct_") for c in real.columns)
        print(json.dumps(validate(tr, np.asarray(e)[[idx[p] for p in tr["prompt_id"]]], routerbench_scores(), n), indent=2))
    else:
        print(json.dumps(build(a.factor), indent=2))
