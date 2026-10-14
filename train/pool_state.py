"""
Purpose:  The live model pool for online RL: models can join (one or several at once) or retire while the
          controller keeps serving and learning, without ever renumbering existing models.
Owner:    C2 Learning (R R H J Rathnayaka) + C3 Measurement (K V A Bethmi)
Contract: v1
Inputs:   ModelCard list (the starting pool), ProfileUpdater (C3-12, prior weight 50), EpisodeLogs.
Outputs:  model_ids, costs, capability_tensor [M, F], bonus [M], available [M] for the session, learner and
          canary; to_json / from_json for <online_dir>/pool.json.
Notes:    Status per card: warmup (new, until warmup_uses rounds) -> active; retired masks a model out of the
          CHOOSE and HALT masks but keeps its index, so old logs still replay. Exploration bonus for warmup
          models: bonus0 / max(1, k_warmup) * max(0, 1 - uses / warmup_uses), split across models warming
          up at the same time. Card accuracy per step type is re-estimated from the prior card and the
          cumulative verifier successes (one update_many on the totals, never repeated priors), and
          mean_cost_usd is the running mean of observed round costs (prior cost until the first use).
          With a verifier that flips the true outcome with known probability flip_p (the simulator's 0.15),
          the observed success rate is de-biased first, (observed - flip_p) / (1 - 2 flip_p), otherwise every
          card drifts toward 0.5 (a true 0.90 reads as 0.78). Cards wider than the RouterBench layout
          (5 accuracies + log cost, e.g. the depth pool's extra depth features) get the same w-weighted
          update directly, since the C3 updater only accepts that layout; extra features are left as they are.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np
import torch

from controller.contracts import ModelCard
from data.routerbench import SOURCES
from measurement.profile_updater import ProfileUpdater

WARMUP, ACTIVE, RETIRED = "warmup", "active", "retired"
N_ACC = len(SOURCES)  # accuracy features first, then log10 cost at index N_ACC (both layouts)


class PoolState:
    def __init__(self, cards: list[ModelCard], updater: ProfileUpdater, warmup_uses: int = 200,
                 bonus0: float = 1.0, flip_p: float = 0.0):
        if not 0 <= flip_p < 0.5:
            raise ValueError("flip_p must be in [0, 0.5)")
        self.updater, self.warmup_uses, self.bonus0, self.flip_p = updater, warmup_uses, bonus0, flip_p
        self.prior: list[ModelCard] = list(cards)  # cards before any observation
        self.cards: list[ModelCard] = list(cards)
        self.status = {c.model_id: ACTIVE for c in cards}
        self.sim_known = {c.model_id: True for c in cards}
        self.uses = {c.model_id: 0 for c in cards}
        self.successes: dict[str, dict[int, float]] = {c.model_id: {} for c in cards}
        self.counts: dict[str, dict[int, int]] = {c.model_id: {} for c in cards}
        self.cost_sum = {c.model_id: 0.0 for c in cards}

    # ---- views used by the session, learner and canary ----

    def model_ids(self) -> list[str]:
        return [c.model_id for c in self.cards]

    def costs(self) -> np.ndarray:
        return np.array([c.mean_cost_usd for c in self.cards], dtype=float)

    def capability_tensor(self) -> torch.Tensor:
        return torch.tensor([c.capability for c in self.cards], dtype=torch.float32)

    def available(self) -> np.ndarray:
        return np.array([self.status[m] != RETIRED for m in self.model_ids()], dtype=bool)

    def bonus(self) -> np.ndarray:
        warming = [m for m in self.model_ids() if self.status[m] == WARMUP]
        share = self.bonus0 / max(1, len(warming))
        return np.array([share * max(0.0, 1 - self.uses[m] / self.warmup_uses) if self.status[m] == WARMUP else 0.0
                         for m in self.model_ids()], dtype=float)

    def sim_model_ids(self) -> list[str]:
        return [m for m in self.model_ids() if self.sim_known[m] and self.status[m] != RETIRED]

    def sim_indices(self) -> list[int]:
        keep = set(self.sim_model_ids())
        return [i for i, m in enumerate(self.model_ids()) if m in keep]

    # ---- changes ----

    def add_models(self, cards: list[ModelCard], sim_known: bool = False) -> None:
        for c in cards:
            if c.model_id in self.status:
                raise ValueError(f"model {c.model_id!r} is already in the pool")
            self.prior.append(c)
            self.cards.append(c)
            self.status[c.model_id] = WARMUP
            self.sim_known[c.model_id] = sim_known
            self.uses[c.model_id] = 0
            self.successes[c.model_id], self.counts[c.model_id] = {}, {}
            self.cost_sum[c.model_id] = 0.0

    def retire(self, model_id: str) -> None:
        if model_id not in self.status:
            raise KeyError(model_id)
        self.status[model_id] = RETIRED

    def record(self, log) -> None:
        """Count uses, re-estimate accuracy per step type from verifier outcomes, update the running cost."""
        touched = set()
        for model_id, step_type, quality, cost in log.used:
            self.uses[model_id] += 1
            self.successes[model_id][step_type] = self.successes[model_id].get(step_type, 0.0) + float(quality)
            self.counts[model_id][step_type] = self.counts[model_id].get(step_type, 0) + 1
            self.cost_sum[model_id] += float(cost)
            touched.add(model_id)
        for model_id in touched:
            i = self.model_ids().index(model_id)
            obs = {t: (self._debias(self.successes[model_id][t] / n), n) for t, n in self.counts[model_id].items()}
            updated = self._update(self.prior[i], obs)
            mean_cost = self.cost_sum[model_id] / self.uses[model_id]
            capability = list(updated.capability)
            if mean_cost > 0:
                capability[N_ACC] = math.log10(mean_cost)
            self.cards[i] = replace(updated, mean_cost_usd=mean_cost if mean_cost > 0 else updated.mean_cost_usd,
                                    capability=capability)
            if self.status[model_id] == WARMUP and self.uses[model_id] >= self.warmup_uses:
                self.status[model_id] = ACTIVE

    def _debias(self, rate: float) -> float:
        if self.flip_p == 0:
            return rate
        return min(1.0, max(0.0, (rate - self.flip_p) / (1 - 2 * self.flip_p)))

    def _update(self, prior: ModelCard, obs: dict[int, tuple[float, int]]) -> ModelCard:
        if len(prior.capability) == N_ACC + 1:  # the C3-12 layout: use the C3 updater
            return self.updater.update_many(prior, obs)
        w, capability = self.updater.prior_weight, list(prior.capability)
        for t, (acc, n) in obs.items():
            capability[t] = (w * capability[t] + n * acc) / (w + n)
        return replace(prior, capability=capability)

    # ---- persistence ----

    def to_json(self) -> dict:
        return {"warmup_uses": self.warmup_uses, "bonus0": self.bonus0, "prior_weight": self.updater.prior_weight,
                "flip_p": self.flip_p,
                "prior": [asdict(c) for c in self.prior], "cards": [asdict(c) for c in self.cards],
                "status": self.status, "sim_known": self.sim_known, "uses": self.uses,
                "successes": {m: {str(t): v for t, v in d.items()} for m, d in self.successes.items()},
                "counts": {m: {str(t): v for t, v in d.items()} for m, d in self.counts.items()},
                "cost_sum": self.cost_sum}

    @classmethod
    def from_json(cls, d: dict) -> PoolState:
        pool = cls([ModelCard(**c) for c in d["prior"]], ProfileUpdater(prior_weight=d["prior_weight"]),
                   warmup_uses=d["warmup_uses"], bonus0=d["bonus0"], flip_p=d.get("flip_p", 0.0))
        pool.cards = [ModelCard(**c) for c in d["cards"]]
        pool.status, pool.sim_known, pool.uses = dict(d["status"]), dict(d["sim_known"]), dict(d["uses"])
        pool.successes = {m: {int(t): v for t, v in x.items()} for m, x in d["successes"].items()}
        pool.counts = {m: {int(t): v for t, v in x.items()} for m, x in d["counts"].items()}
        pool.cost_sum = dict(d["cost_sum"])
        return pool
