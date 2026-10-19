
"""C3-12: Model capability profile updater.

Updates ModelCard accuracy estimates using the
research-required weighted update formula:

    new = (w * prior + n * observed) / (w + n)

Default prior weight:
    w = 50

Capability layout:
    0: accuracy_mmlu
    1: accuracy_hellaswag
    2: accuracy_arc-challenge
    3: accuracy_winogrande
    4: accuracy_mbpp
    5: log10_mean_cost_usd

Only the five accuracy dimensions are updated.

The original ModelCard is never modified.

The updater accepts aggregated observations from
a training/probe set. Callers must enforce the
appropriate data split before calculating them.

For multiple observations of the same step type,
combine their successes and counts before updating,
rather than repeatedly applying the prior weight.
"""

from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import replace

from controller.contracts import ModelCard
from data.routerbench import SOURCES


PRIOR_WEIGHT = 50.0

PROFILE_UPDATE_VERSION = "rb_profile_update_v1"


class ProfileUpdater:
    """Apply weighted updates to model accuracy profiles."""

    def __init__(
        self,
        *,
        prior_weight: float = PRIOR_WEIGHT,
    ) -> None:
        """Configure the strength of the existing prior."""

        if isinstance(prior_weight, bool):
            raise ValueError(
                "prior_weight must be positive and finite"
            )

        try:
            weight = float(prior_weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "prior_weight must be positive and finite"
            ) from exc

        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                "prior_weight must be positive and finite"
            )

        self.prior_weight = weight

    @staticmethod
    def _validate_card(card: ModelCard) -> None:
        """Check compatibility with the C3-03 contract."""

        if not isinstance(card, ModelCard):
            raise TypeError(
                "card must be a ModelCard"
            )

        expected_features = len(SOURCES) + 1

        if len(card.capability) != expected_features:
            raise ValueError(
                "ModelCard must have six capability features"
            )

        for index, value in enumerate(card.capability):

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Invalid capability at index {index}"
                )

            if index < len(SOURCES) and not 0 <= value <= 1:
                raise ValueError(
                    "Accuracy capabilities must be "
                    "between zero and one"
                )

    @staticmethod
    def _validate_observation(
        step_type: int,
        observed: float,
        n: int,
    ) -> tuple[float, int]:
        """Validate one observed accuracy and sample count."""

        if (
            isinstance(step_type, bool)
            or not isinstance(step_type, int)
            or not 0 <= step_type < len(SOURCES)
        ):
            raise ValueError(
                "step_type must identify a valid task type"
            )

        if (
            isinstance(n, bool)
            or not isinstance(n, int)
            or n < 1
        ):
            raise ValueError(
                "n must be a positive integer"
            )

        if isinstance(observed, bool):
            raise ValueError(
                "observed accuracy must be between 0 and 1"
            )

        try:
            accuracy = float(observed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "observed accuracy must be between 0 and 1"
            ) from exc

        if (
            not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
        ):
            raise ValueError(
                "observed accuracy must be between 0 and 1"
            )

        return accuracy, n

    def update(
        self,
        card: ModelCard,
        *,
        step_type: int,
        observed: float,
        n: int,
    ) -> ModelCard:
        """Update one task-type capability.

        Returns a new ModelCard without modifying card.

        Example:
            prior = 0.70
            observed = 0.90
            n = 10
            w = 50

            updated = 0.733333...
        """

        return self.update_many(
            card,
            {
                step_type: (observed, n),
            },
        )

    def update_many(
        self,
        card: ModelCard,
        observations: Mapping[int, tuple[float, int]],
    ) -> ModelCard:
        """Update multiple task types independently.

        observations format:

            {
                step_type: (observed_accuracy, sample_count)
            }

        Each step type receives one aggregated observation.

        All updates use the same original card as the prior.
        An unobserved task type retains its prior estimate.

        Model identity, cost, metadata and version are preserved.
        """

        self._validate_card(card)

        if not isinstance(observations, Mapping):
            raise TypeError(
                "observations must be a mapping"
            )

        if not observations:
            raise ValueError(
                "At least one observation is required"
            )

        updated_capability = list(
            card.capability
        )

        # Validate all observations before applying changes.
        validated = {}

        for step_type, values in observations.items():

            if (
                not isinstance(values, (tuple, list))
                or len(values) != 2
            ):
                raise ValueError(
                    "Each observation must contain "
                    "(observed_accuracy, sample_count)"
                )

            observed, n = values

            accuracy, count = self._validate_observation(
                step_type,
                observed,
                n,
            )

            validated[step_type] = (
                accuracy,
                count,
            )

        # Apply the research-required weighted update.
        for step_type, (accuracy, count) in validated.items():

            prior = float(
                card.capability[step_type]
            )

            weight = self.prior_weight

            updated_accuracy = (
                weight * prior
                + count * accuracy
            ) / (
                weight + count
            )

            if (
                not math.isfinite(updated_accuracy)
                or not 0 <= updated_accuracy <= 1
            ):
                raise RuntimeError(
                    "Updated accuracy is invalid"
                )

            updated_capability[step_type] = (
                updated_accuracy
            )

        # Preserve the original cost feature exactly.
        updated_capability[len(SOURCES)] = (
            card.capability[len(SOURCES)]
        )

        # dataclasses.replace creates a new ModelCard
        # and preserves all other card attributes.
        return replace(
            card,
            capability=updated_capability,
        )
