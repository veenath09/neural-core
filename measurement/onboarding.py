"""
Purpose:  Build the ModelCard for a model joining the pool: cold-start accuracy priors from the models
          already known, then its own training-probe observations with the w = 50 update (C3-12).
Owner:    C3 Measurement (K V A Bethmi)
Contract: v1
Inputs:   the joining model's card, the existing pool's cards, its training probes (probe profiler format).
Outputs:  (onboarded card, observed capability vector).
Notes:    Moved unchanged from scripts/evaluate_leave_one_out.py so online RL (train/pool_state.py) and the
          leave-one-out experiment share one implementation; the script imports it from here.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from controller.contracts import ModelCard
from data.routerbench import SOURCES
from measurement.probe_profiler import ProbeProfiler
from measurement.profile_updater import PRIOR_WEIGHT, ProfileUpdater


def build_onboarded_card(
    *,
    held_out: ModelCard,
    existing_cards: list[ModelCard],
    probes: pd.DataFrame,
) -> tuple[ModelCard, list[float]]:
    """Profile and onboard the unseen model.

    Initial accuracy priors come from the three
    already-known models.

    The held-out model's observed accuracy and cost
    come only from its training probes.
    """

    if len(existing_cards) != 3:
        raise ValueError(
            "Expected three existing model cards"
        )

    profiler = ProbeProfiler()

    observed_vector = profiler.run(
        held_out,
        probes,
    )

    # Build cold-start accuracy priors from
    # the existing models, not the held-out
    # model's full C3-03 capability estimates.
    prior_accuracies = [
        sum(
            card.capability[index]
            for card in existing_cards
        ) / len(existing_cards)
        for index in range(len(SOURCES))
    ]

    # The sixth feature represents cost.
    # Estimate it from the held-out model's
    # training probes rather than from test data.
    observed_cost = 10 ** observed_vector[-1]

    cold_card = replace(
        held_out,
        mean_cost_usd=observed_cost,
        capability=[
            *prior_accuracies,
            observed_vector[-1],
        ],
    )

    # Each source gets its own observed accuracy
    # and count, following the required w=50 rule.
    observations = {}

    for index, source in enumerate(SOURCES):
        count = int(
            probes["source"].eq(source).sum()
        )

        observations[index] = (
            observed_vector[index],
            count,
        )

    updated = ProfileUpdater(
        prior_weight=PRIOR_WEIGHT,
    ).update_many(
        cold_card,
        observations,
    )

    # Cost and model identity must remain consistent.
    if updated.model_id != held_out.model_id:
        raise RuntimeError(
            "Held-out model identity changed"
        )

    if updated.capability[-1] != cold_card.capability[-1]:
        raise RuntimeError(
            "Cost capability changed during update"
        )

    return updated, observed_vector
