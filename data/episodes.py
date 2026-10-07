"""Policy-independent episode skeletons (plan §4.2–4.3, D1, D10). Outcomes are rolled out later
by data/env.py, because they depend on the models chosen earlier in the episode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.routerbench import DATA_DIR, OutcomeTable, load_prepared

EPISODE_TYPES = ("correlated", "retry", "propagation", "random")
TYPE_SHARES = {"correlated": 0.40, "retry": 0.25, "propagation": 0.20, "random": 0.15}
MIN_STEPS, MAX_STEPS = 3, 6
# Budget = factor x cost of the strongest model on every listed prompt.
# P(factor < 1) = 0.3 / 0.9 = 1/3, so about a third cannot afford it on every step.
BUDGET_FACTOR = (0.7, 1.6)
SIZES = {"train": 20_000, "val": 2_000, "test": 5_000}


@dataclass(frozen=True)
class Skeleton:
    episode_id: str
    ep_type: str
    prompt_ids: tuple[str, ...]
    budget_usd: float


def type_counts(n: int) -> dict[str, int]:
    """Exact shares by largest remainder."""
    raw = {t: n * TYPE_SHARES[t] for t in EPISODE_TYPES}
    counts = {t: int(v) for t, v in raw.items()}
    by_remainder = sorted(EPISODE_TYPES, key=lambda t: raw[t] - counts[t], reverse=True)
    for t in by_remainder[: n - sum(counts.values())]:
        counts[t] += 1
    return counts


def _correlated_groups(table: OutcomeTable, rows: np.ndarray) -> list[np.ndarray]:
    """Prompt rows sharing a source and a difficulty band, large enough for any episode length."""
    key = table.step_type[rows] * 10 + table.band[rows]
    groups = [rows[key == k] for k in np.unique(key)]
    return [g for g in groups if len(g) >= MAX_STEPS]


def generate_skeletons(table: OutcomeTable, prompt_ids: Sequence[str], n: int, seed: int,
                       split: str) -> list[Skeleton]:
    rng = np.random.default_rng(seed)
    rows = np.array([table.index[p] for p in prompt_ids])
    groups = _correlated_groups(table, rows)
    sizes = np.array([len(g) for g in groups], dtype=float)
    counts = type_counts(n)
    types = rng.permutation(np.repeat(EPISODE_TYPES, [counts[t] for t in EPISODE_TYPES]))
    out = []
    for i, ep_type in enumerate(types):
        n_steps = int(rng.integers(MIN_STEPS, MAX_STEPS + 1))
        source = groups[rng.choice(len(groups), p=sizes / sizes.sum())] if ep_type == "correlated" else rows
        chosen = rng.choice(source, size=n_steps, replace=False)
        budget = float(rng.uniform(*BUDGET_FACTOR) * table.cost[chosen, -1].sum())
        out.append(Skeleton(f"{split}-{i:06d}", str(ep_type), tuple(str(p) for p in table.prompt_ids[chosen]), budget))
    return out


def to_frame(skeletons: Sequence[Skeleton]) -> pd.DataFrame:
    return pd.DataFrame([
        {"episode_id": s.episode_id, "ep_type": s.ep_type, "slot": j, "prompt_id": p,
         "budget_usd": s.budget_usd, "n_steps": len(s.prompt_ids)}
        for s in skeletons for j, p in enumerate(s.prompt_ids)
    ])


def from_frame(df: pd.DataFrame) -> list[Skeleton]:
    out = []
    for eid, g in df.sort_values(["episode_id", "slot"]).groupby("episode_id", sort=True):
        out.append(Skeleton(str(eid), str(g["ep_type"].iloc[0]), tuple(g["prompt_id"].astype(str)),
                            float(g["budget_usd"].iloc[0])))
    return out


def load_skeletons(split: str, data_dir: Path = DATA_DIR) -> list[Skeleton]:
    return from_frame(pd.read_parquet(Path(data_dir) / f"episodes_{split}.parquet"))


def build_all(data_dir: Path = DATA_DIR, seed: int = 0, sizes: dict[str, int] = SIZES) -> None:
    prompts, _ = load_prepared(data_dir)
    table = OutcomeTable.from_frame(prompts)
    for i, (split, n) in enumerate(sizes.items()):
        ids = prompts.loc[prompts["split"] == split, "prompt_id"].tolist()
        skeletons = generate_skeletons(table, ids, n, seed + i, split)
        to_frame(skeletons).to_parquet(Path(data_dir) / f"episodes_{split}.parquet", index=False)
        print(split, n, pd.Series([s.ep_type for s in skeletons]).value_counts().to_dict())


if __name__ == "__main__":
    build_all()
