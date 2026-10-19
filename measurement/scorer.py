
"""Pillar 3 C3-07: Composite episode scorer.

Reward:
    quality
    - lam * cost_usd / cost_ref_usd
    - mu * latency_ms / latency_ref_ms
    - penalty

RouterBench has no measured latency. When mu is zero,
latency is not required and is never fabricated.

The cost reference is frozen from the training split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Validate an ordinary finite numeric value."""

    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(
            f"{name} must be a finite number"
        )

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite"
        )

    if number < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}"
        )

    if strictly_positive and number == 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    if maximum is not None and number > maximum:
        raise ValueError(
            f"{name} must not exceed {maximum}"
        )

    return number


@dataclass(frozen=True)
class ScoreConfig:
    """Frozen scoring references and their provenance."""

    cost_ref_usd: float
    cost_ref_model: str
    cost_ref_split: str
    cost_ref_statistic: str
    train_prompt_count: int
    latency_ref_ms: float | None
    latency_source: str
    version: int = 1

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "ScoreConfig":
        if values.get("version") != 1:
            raise ValueError(
                "Unsupported score config version"
            )

        if values.get("cost_ref_split") != "train":
            raise ValueError(
                "Cost reference must come from train split"
            )

        if (
            values.get("cost_ref_statistic")
            != "mean_per_step_cost"
        ):
            raise ValueError(
                "Unexpected cost reference statistic"
            )

        model = values.get("cost_ref_model")

        if not isinstance(model, str) or not model:
            raise ValueError(
                "Cost reference model must be specified"
            )

        count = values.get("train_prompt_count")

        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(
                "Training prompt count must be positive"
            )

        cost_ref = _finite_number(
            values.get("cost_ref_usd"),
            "cost_ref_usd",
            strictly_positive=True,
        )

        latency_ref = values.get("latency_ref_ms")

        if latency_ref is not None:
            latency_ref = _finite_number(
                latency_ref,
                "latency_ref_ms",
                strictly_positive=True,
            )

        latency_source = values.get(
            "latency_source"
        )

        if (
            not isinstance(latency_source, str)
            or not latency_source
        ):
            raise ValueError(
                "Latency source must be specified"
            )

        return cls(
            cost_ref_usd=cost_ref,
            cost_ref_model=model,
            cost_ref_split="train",
            cost_ref_statistic="mean_per_step_cost",
            train_prompt_count=count,
            latency_ref_ms=latency_ref,
            latency_source=latency_source,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = "conf/score.yaml",
    ) -> "ScoreConfig":
        """Load the frozen reference configuration."""

        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to read score.yaml"
            ) from exc

        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:
            values = yaml.safe_load(file)

        if not isinstance(values, dict):
            raise ValueError(
                "Score configuration must be a mapping"
            )

        return cls.from_mapping(values)


@dataclass(frozen=True)
class Score:
    """Composite scoring result.

    A missing latency is represented by None, not 0.
    """

    quality: float
    cost_usd: float
    latency_ms: float | None
    penalty: float
    lam: float
    mu: float

    cost_ref_usd: float
    latency_ref_ms: float | None

    normalized_cost: float
    normalized_latency: float | None

    reward: float
    latency_included: bool


class CompositeScorer:
    """Compute a normalized, configurable episode reward."""

    def __init__(
        self,
        cfg: ScoreConfig | Mapping[str, Any],
    ) -> None:
        if isinstance(cfg, ScoreConfig):
            self.cfg = cfg

        elif isinstance(cfg, Mapping):
            self.cfg = ScoreConfig.from_mapping(cfg)

        else:
            raise TypeError(
                "cfg must be ScoreConfig or a mapping"
            )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = "conf/score.yaml",
    ) -> "CompositeScorer":
        return cls(ScoreConfig.from_yaml(path))

    def score(
        self,
        *,
        quality: float,
        cost_usd: float,
        latency_ms: float | None = None,
        lam: float = 0.0,
        mu: float = 0.0,
        penalty: float = 0.0,
    ) -> Score:
        """Score an episode using the research reward formula.

        quality: Correct steps / expected steps.
        cost_usd: Total episode execution cost.
        latency_ms: Measured episode latency, if available.
        lam: Nonnegative cost weight.
        mu: Nonnegative latency weight.
        penalty: Explicit, nonnegative penalty.

        If mu > 0, real latency and a reference are required.
        """

        q = _finite_number(
            quality,
            "quality",
            maximum=1.0,
        )

        cost = _finite_number(
            cost_usd,
            "cost_usd",
        )

        cost_weight = _finite_number(
            lam,
            "lam",
        )

        latency_weight = _finite_number(
            mu,
            "mu",
        )

        applied_penalty = _finite_number(
            penalty,
            "penalty",
        )

        observed_latency = None

        if latency_ms is not None:
            observed_latency = _finite_number(
                latency_ms,
                "latency_ms",
            )

        cost_ratio = (
            cost / self.cfg.cost_ref_usd
        )

        latency_ratio = None

        if latency_weight > 0:
            if observed_latency is None:
                raise ValueError(
                    "Measured latency is required when mu > 0"
                )

            if self.cfg.latency_ref_ms is None:
                raise ValueError(
                    "A measured latency reference is required "
                    "when mu > 0"
                )

            latency_ratio = (
                observed_latency
                / self.cfg.latency_ref_ms
            )

            latency_cost = (
                latency_weight * latency_ratio
            )

        else:
            # No latency contribution is calculated.
            # Missing latency remains None.
            latency_cost = 0.0

        reward = (
            q
            - cost_weight * cost_ratio
            - latency_cost
            - applied_penalty
        )

        return Score(
            quality=q,
            cost_usd=cost,
            latency_ms=observed_latency,
            penalty=applied_penalty,
            lam=cost_weight,
            mu=latency_weight,
            cost_ref_usd=self.cfg.cost_ref_usd,
            latency_ref_ms=self.cfg.latency_ref_ms,
            normalized_cost=cost_ratio,
            normalized_latency=latency_ratio,
            reward=reward,
            latency_included=latency_weight > 0,
        )
