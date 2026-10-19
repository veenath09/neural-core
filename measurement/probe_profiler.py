
"""C3-12: Leakage-safe model probe profiler.

Profiles one ModelCard using historical RouterBench
observations from an explicitly selected training probe set.

Capability layout:
    [
        accuracy_mmlu,
        accuracy_hellaswag,
        accuracy_arc_challenge,
        accuracy_winogrande,
        accuracy_mbpp,
        log10_mean_cost_usd,
    ]

This module does NOT make live model API calls.

It does NOT modify the original ModelCard.

Only training observations are accepted to prevent
validation/test outcome leakage into model profiles.
"""

from __future__ import annotations

import math

import pandas as pd

from controller.contracts import ModelCard
from data.routerbench import SOURCES


PROBE_VERSION = "rb_probe_v1"


class ProbeProfiler:
    """Estimate model capabilities from a training probe set."""

    def __init__(
        self,
        *,
        require_all_step_types: bool = True,
    ) -> None:
        if not isinstance(require_all_step_types, bool):
            raise TypeError(
                "require_all_step_types must be boolean"
            )

        self.require_all_step_types = (
            require_all_step_types
        )

    def run(
        self,
        card: ModelCard,
        probe_set: pd.DataFrame,
    ) -> list[float]:
        """Return the six-dimensional capability vector.

        Expected probe_set columns:
            prompt_id
            source
            step_type
            split
            <model_id>_correct
            <model_id>_cost

        All rows must belong to the training split.

        For each step type, accuracy is the mean of
        binary correctness observations.

        The final capability feature is the log10
        of the observed mean execution cost.
        """

        # --------------------------------------------------
        # Validate input types
        # --------------------------------------------------

        if not isinstance(card, ModelCard):
            raise TypeError(
                "card must be a ModelCard"
            )

        if not isinstance(probe_set, pd.DataFrame):
            raise TypeError(
                "probe_set must be a pandas DataFrame"
            )

        if probe_set.empty:
            raise ValueError(
                "Probe set cannot be empty"
            )

        if len(card.capability) != len(SOURCES) + 1:
            raise ValueError(
                "ModelCard must contain six "
                "capability features"
            )

        # --------------------------------------------------
        # Validate required columns
        # --------------------------------------------------

        correct_column = (
            f"{card.model_id}_correct"
        )

        cost_column = (
            f"{card.model_id}_cost"
        )

        required = {
            "prompt_id",
            "source",
            "step_type",
            "split",
            correct_column,
            cost_column,
        }

        missing = required.difference(
            probe_set.columns
        )

        if missing:
            raise ValueError(
                "Missing probe columns: "
                f"{sorted(missing)}"
            )

        # --------------------------------------------------
        # Enforce training-only probes
        # --------------------------------------------------

        if probe_set["split"].isna().any():
            raise ValueError(
                "Probe split contains missing values"
            )

        if not probe_set["split"].eq(
            "train"
        ).all():
            raise ValueError(
                "ProbeProfiler accepts training "
                "probes only"
            )

        # --------------------------------------------------
        # Validate prompt identities
        # --------------------------------------------------

        if probe_set["prompt_id"].isna().any():
            raise ValueError(
                "Missing probe prompt IDs"
            )

        if probe_set["prompt_id"].duplicated().any():
            raise ValueError(
                "Duplicate probe prompt IDs"
            )

        # --------------------------------------------------
        # Validate step types
        # --------------------------------------------------

        source_to_type = {
            source: index
            for index, source in enumerate(SOURCES)
        }

        if not probe_set["source"].isin(
            SOURCES
        ).all():
            raise ValueError(
                "Unknown probe source"
            )

        expected_types = probe_set[
            "source"
        ].map(source_to_type)

        if not expected_types.eq(
            probe_set["step_type"]
        ).all():
            raise ValueError(
                "Probe source and step type mismatch"
            )

        # --------------------------------------------------
        # Validate binary correctness
        # --------------------------------------------------

        correctness = pd.to_numeric(
            probe_set[correct_column],
            errors="raise",
        )

        if correctness.isna().any():
            raise ValueError(
                "Missing correctness observations"
            )

        if not correctness.isin(
            [0, 1]
        ).all():
            raise ValueError(
                "Correctness observations "
                "must be binary"
            )

        # --------------------------------------------------
        # Validate execution cost
        # --------------------------------------------------

        costs = pd.to_numeric(
            probe_set[cost_column],
            errors="raise",
        )

        if costs.isna().any():
            raise ValueError(
                "Missing probe cost observations"
            )

        if not all(
            math.isfinite(float(value))
            for value in costs
        ):
            raise ValueError(
                "Non-finite probe costs"
            )

        if (costs < 0).any():
            raise ValueError(
                "Negative probe costs"
            )

        mean_cost = float(
            costs.mean()
        )

        if not math.isfinite(mean_cost) or mean_cost <= 0:
            raise ValueError(
                "Mean probe cost must be "
                "positive and finite"
            )

        # --------------------------------------------------
        # Calculate task-type accuracies
        # --------------------------------------------------

        capabilities: list[float] = []

        for source in SOURCES:

            mask = probe_set[
                "source"
            ].eq(source)

            count = int(
                mask.sum()
            )

            if count == 0:

                if self.require_all_step_types:
                    raise ValueError(
                        "Probe set has no observations "
                        f"for source: {source}"
                    )

                # No observed data for this type.
                # Preserve the original prior rather
                # than inventing an observed accuracy.
                source_index = source_to_type[
                    source
                ]

                prior = float(
                    card.capability[source_index]
                )

                if (
                    not math.isfinite(prior)
                    or not 0 <= prior <= 1
                ):
                    raise ValueError(
                        "Invalid prior capability"
                    )

                capabilities.append(
                    prior
                )

                continue

            accuracy = float(
                correctness.loc[mask].mean()
            )

            if not math.isfinite(accuracy):
                raise ValueError(
                    "Invalid probe accuracy"
                )

            capabilities.append(
                accuracy
            )

        # --------------------------------------------------
        # Append log10 observed mean cost
        # --------------------------------------------------

        capabilities.append(
            math.log10(mean_cost)
        )

        # --------------------------------------------------
        # Validate output representation
        # --------------------------------------------------

        if len(capabilities) != len(SOURCES) + 1:
            raise RuntimeError(
                "Unexpected capability dimension"
            )

        if not all(
            math.isfinite(value)
            for value in capabilities
        ):
            raise RuntimeError(
                "Non-finite capability vector"
            )

        return capabilities
