
"""Pillar 3 C3: RL reward and lambda-weighted oracle on the composite scorer.

Episode reward:
    R = quality - lam * (spent / n_steps) / cost_ref

Cost is the per-step mean because cost_ref is a per-step statistic
(mean_per_step_cost). This equals CompositeScorer.score(quality=q,
cost_usd=spent/n_steps, lam=lam).reward with mu=0, penalty=0.

The lambda-oracle reuses measurement/oracle.py's weighted_episode_oracle,
which maximises quality - lam_raw * raw_cost. Passing
lam_raw = lam / (n_steps * cost_ref) makes its objective equal our reward
exactly, without changing measurement/oracle.py.

LAMS is the RL lambda grid. It lives here, not in the controller package,
so the controller imports it from measurement.rl_reward.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from data.env import Episode
from data.episodes import Skeleton
from data.routerbench import OutcomeTable
from measurement.oracle import WeightedEpisodeOracleResult, weighted_episode_oracle
from measurement.scorer import CompositeScorer

LAMS: tuple[float, ...] = (0.1, 0.3, 0.6, 1.0, 2.0)


def load_cost_ref(path: str | Path = "conf/score.yaml") -> float:
    """Read the frozen per-step cost reference used to scale reward cost."""

    return CompositeScorer.from_yaml(path).cfg.cost_ref_usd


def episode_reward(
    scorer: CompositeScorer,
    quality: float,
    spent: float,
    n_steps: int,
    lam: float,
) -> float:
    """Episode reward: quality minus lam times normalized per-step cost."""

    return scorer.score(
        quality=quality,
        cost_usd=spent / n_steps,
        lam=lam,
    ).reward


def step_reward(
    d_quality: float,
    d_spent: float,
    n_steps: int,
    lam: float,
    cost_ref: float,
) -> float:
    """Per-decision reward: r_t = d_quality/n_steps - lam * d_spent/(n_steps*cost_ref).

    Summed over a full episode's decisions, this equals episode_reward exactly.
    """

    return d_quality / n_steps - lam * d_spent / (n_steps * cost_ref)


def lam_oracle(
    ep: Episode,
    lam: float,
    cost_ref: float,
) -> WeightedEpisodeOracleResult:
    """Lambda-weighted oracle whose objective equals our episode reward.

    weighted_episode_oracle maximises quality - lam_raw * raw_cost; passing
    lam_raw = lam / (n_steps * cost_ref) makes that objective equal
    quality - lam * (cost / n_steps) / cost_ref, i.e. episode_reward.
    """

    return weighted_episode_oracle(ep, lam / (ep.n_steps * cost_ref))


def lam_oracle_points(
    skeletons: Sequence[Skeleton],
    table: OutcomeTable,
    model_ids: Sequence[str],
    cost_ref: float,
    lams: Sequence[float] = LAMS,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Mean quality, cost and reward of the lambda-oracle over these episodes.

    Cached by a hash of the episode ids and the lambda grid, because the
    exact search takes minutes on a large pool.
    """

    key_source = "|".join(sk.episode_id for sk in skeletons) + "|" + ",".join(
        repr(lam) for lam in lams
    )
    key = hashlib.blake2b(key_source.encode(), digest_size=6).hexdigest()
    cache = Path(cache_dir) / f"lam_oracle_{key}.json" if cache_dir else None

    if cache and cache.exists():
        return pd.DataFrame(json.loads(cache.read_text()))

    episodes = [Episode(sk, table, model_ids) for sk in skeletons]
    scorer = CompositeScorer(
        {
            "version": 1,
            "cost_ref_usd": cost_ref,
            "cost_ref_model": "lam_oracle_points",
            "cost_ref_split": "train",
            "cost_ref_statistic": "mean_per_step_cost",
            "train_prompt_count": 1,
            "latency_ref_ms": None,
            "latency_source": "unavailable_in_routerbench",
        }
    )

    rows = []

    for lam in lams:
        qualities = []
        costs = []
        rewards = []

        for ep in episodes:
            result = lam_oracle(ep, lam, cost_ref)
            qualities.append(result.quality)
            costs.append(result.cost_usd)
            rewards.append(
                episode_reward(
                    scorer=scorer,
                    quality=result.quality,
                    spent=result.cost_usd,
                    n_steps=ep.n_steps,
                    lam=lam,
                )
            )

        rows.append(
            {
                "lam": lam,
                "quality": sum(qualities) / len(qualities),
                "cost_usd": sum(costs) / len(costs),
                "reward": sum(rewards) / len(rewards),
            }
        )

    df = pd.DataFrame(rows)

    if cache:
        cache.write_text(df.to_json(orient="records"))

    return df
