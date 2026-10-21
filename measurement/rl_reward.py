
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
from functools import lru_cache
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


def cost_ref_for(data_dir: str | Path | None) -> float:
    """The pool's own per-step cost reference: <data_dir>/cost_ref.json when the pool ships one (the depth
    pools: strongest option's mean per-step cost on the train split, the statistic conf/score.yaml uses for
    the RouterBench pool), else the frozen RouterBench value."""

    path = Path(data_dir) / "cost_ref.json" if data_dir is not None else None
    if path is not None and path.exists():
        return float(json.loads(path.read_text())["cost_ref_usd"])
    return load_cost_ref()


@lru_cache(maxsize=None)
def scorer_for(cost_ref: float) -> CompositeScorer:
    """Composite scorer at this pool's per-step cost reference (cost_ref_for(data_dir)), built once per value.

    Every RL reward (training rollouts, evaluate_rl, calibration, canary, online paired evaluation) must use
    the same cost_ref as log_price: the RouterBench value in conf/score.yaml is only right for that pool."""

    return CompositeScorer(
        {
            "version": 1,
            "cost_ref_usd": float(cost_ref),
            "cost_ref_model": "pool cost_ref",
            "cost_ref_split": "train",
            "cost_ref_statistic": "mean_per_step_cost",
            "train_prompt_count": 1,
            "latency_ref_ms": None,
            "latency_source": "unavailable_in_routerbench",
        }
    )


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


def episode_log_rewards(
    round_costs: list[float],
    step_quality: list[float],
    decision_step: list[int],
    lam: float,
    cost_ref: float,
) -> list[float]:
    """Per-decision reward across a flat list of an episode's decisions.

    round_costs[t] is the cost of the call decision t triggered (0.0 for a
    stop); decision_step[t] is the step index of decision t. Each decision
    gets -lam * (its round's cost) / (n * cost_ref) via step_reward; each
    step's LAST decision also gets +step_quality[step] / n, with
    n = len(step_quality).

    The sum equals episode_reward(scorer, mean(step_quality),
    sum(round_costs), n, lam): the quality terms sum to mean(step_quality)
    because each step contributes its quality exactly once (on its last
    decision), and the cost terms sum to -lam * sum(round_costs) /
    (n * cost_ref) by linearity of step_reward's cost term.
    """

    n = len(step_quality)
    last_decision_of_step: dict[int, int] = {}
    for t, step in enumerate(decision_step):
        last_decision_of_step[step] = t

    rewards = []
    for t, (cost, step) in enumerate(zip(round_costs, decision_step)):
        r = step_reward(d_quality=0.0, d_spent=cost, n_steps=n, lam=lam, cost_ref=cost_ref)
        if last_decision_of_step.get(step) == t:
            r += step_quality[step] / n
        rewards.append(r)
    return rewards


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
    scorer = scorer_for(cost_ref)

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
