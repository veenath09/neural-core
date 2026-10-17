
"""Pillar 3 C3-03: Training-only model capability profiles.

Build ModelCards using only RouterBench training prompts.
Preserve the existing six-feature Controller contract.

Latency is unavailable in RouterBench and is recorded separately
as missing metadata rather than as a measured value.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from controller.contracts import ModelCard
from data.routerbench import (
    POOL,
    SOURCES,
    PRICE_DATE,
    PRICE_SOURCE,
)


PROBE_VERSION = "rb_train_v1"

CLEAN_PATH = Path("data/clean/routerbench_binary.parquet")
SPLITS_PATH = Path("data/clean/splits.parquet")

CARDS_PATH = Path("data/clean/model_cards.json")
METADATA_PATH = Path("data/clean/capability_metadata.json")


def build_capability_profiles(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[list[ModelCard], dict]:
    """Calculate capability profiles using training prompts only."""

    required_clean = {
        "prompt_id",
        "source",
        "step_type",
    }

    for model in POOL:
        required_clean.add(f"{model}_correct")
        required_clean.add(f"{model}_cost")

    required_splits = {"prompt_id", "split"}

    missing_clean = required_clean.difference(clean.columns)
    missing_splits = required_splits.difference(splits.columns)

    if missing_clean:
        raise ValueError(
            f"Missing clean columns: {sorted(missing_clean)}"
        )

    if missing_splits:
        raise ValueError(
            f"Missing split columns: {sorted(missing_splits)}"
        )

    if clean.empty or splits.empty:
        raise ValueError("Input datasets must not be empty")

    if clean["prompt_id"].isna().any():
        raise ValueError("Clean dataset contains missing prompt IDs")

    if splits[["prompt_id", "split"]].isna().any().any():
        raise ValueError("Splits contain missing values")

    if clean["prompt_id"].duplicated().any():
        raise ValueError("Duplicate clean prompt IDs")

    if splits["prompt_id"].duplicated().any():
        raise ValueError("Duplicate split prompt IDs")

    if not splits["split"].isin(
        ["train", "val", "test"]
    ).all():
        raise ValueError("Unknown split label")

    clean_ids = set(clean["prompt_id"])
    split_ids = set(splits["prompt_id"])

    if clean_ids != split_ids:
        raise ValueError(
            "Clean and split prompt IDs do not match"
        )

    # Select the train split BEFORE computing any statistics.
    train_ids = splits.loc[
        splits["split"] == "train",
        ["prompt_id"],
    ]

    train = clean.merge(
        train_ids,
        on="prompt_id",
        how="inner",
        validate="one_to_one",
    )

    if train.empty:
        raise ValueError("Training split must not be empty")

    expected_types = {
        source: index
        for index, source in enumerate(SOURCES)
    }

    if not train["source"].isin(SOURCES).all():
        raise ValueError("Unknown training source")

    if not (
        train["source"].map(expected_types).to_numpy()
        == train["step_type"].to_numpy()
    ).all():
        raise ValueError("Source and step type mismatch")

    source_counts = {
        source: int((train["source"] == source).sum())
        for source in SOURCES
    }

    if any(count == 0 for count in source_counts.values()):
        raise ValueError(
            "Every step type must have training examples"
        )

    cards: list[ModelCard] = []
    profile_details: list[dict] = []

    for tier, model in enumerate(POOL):
        correct_column = f"{model}_correct"
        cost_column = f"{model}_cost"

        correctness = pd.to_numeric(
            train[correct_column],
            errors="raise",
        )

        costs = pd.to_numeric(
            train[cost_column],
            errors="raise",
        )

        if correctness.isna().any():
            raise ValueError(
                f"Missing correctness values for {model}"
            )

        if not correctness.isin([0, 1]).all():
            raise ValueError(
                f"Non-binary correctness values for {model}"
            )

        if not np.isfinite(costs.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Non-finite costs for {model}"
            )

        if (costs < 0).any():
            raise ValueError(
                f"Negative costs for {model}"
            )

        mean_cost = float(costs.mean())

        if mean_cost <= 0:
            raise ValueError(
                f"Mean cost must be positive for {model}"
            )

        accuracy_by_source = {
            source: float(
                correctness.loc[
                    train["source"] == source
                ].mean()
            )
            for source in SOURCES
        }

        # Match the existing Controller contract:
        # five accuracies + log10(mean cost).
        capability = [
            accuracy_by_source[source]
            for source in SOURCES
        ]

        capability.append(math.log10(mean_cost))

        provider = (
            model.split("/")[0]
            if "/" in model
            else "openai"
        )

        card = ModelCard(
            model_id=model,
            provider=provider,
            tier=tier,
            mean_cost_usd=mean_cost,
            price_source=PRICE_SOURCE,
            price_date=PRICE_DATE,
            capability=capability,
        )

        cards.append(card)

        profile_details.append({
            "model_id": model,
            "tier": tier,
            "probe_version": PROBE_VERSION,
            "training_prompt_count": len(train),
            "accuracy_by_source": accuracy_by_source,
            "mean_cost_usd": mean_cost,
            "log10_mean_cost_usd": math.log10(mean_cost),
            "mean_latency_ms": None,
            "latency_available": False,
            "log_latency_feature": 0.0,
            "latency_feature_is_placeholder": True,
        })

    mean_costs = [
        card.mean_cost_usd
        for card in cards
    ]

    if mean_costs != sorted(mean_costs):
        raise ValueError(
            "Model pool must be ordered by training mean cost"
        )

    metadata = {
        "probe_version": PROBE_VERSION,
        "training_only": True,
        "training_prompt_count": len(train),
        "source_counts": source_counts,
        "controller_capability_layout": [
            *[
                f"accuracy_{source}"
                for source in SOURCES
            ],
            "log10_mean_cost_usd",
        ],
        "latency_status": "unavailable",
        "latency_note": (
            "RouterBench has no latency measurement. "
            "The zero placeholder is metadata only and "
            "does not represent measured zero latency."
        ),
        "price_source": PRICE_SOURCE,
        "price_date": PRICE_DATE,
        "profiles": profile_details,
    }

    return cards, metadata


def save_capability_profiles(
    clean_path: str | Path = CLEAN_PATH,
    splits_path: str | Path = SPLITS_PATH,
    cards_path: str | Path = CARDS_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> tuple[list[ModelCard], dict]:
    """Build and save reproducible C3-03 model profiles."""

    clean = pd.read_parquet(clean_path)
    splits = pd.read_parquet(splits_path)

    cards, metadata = build_capability_profiles(
        clean,
        splits,
    )

    cards_path = Path(cards_path)
    metadata_path = Path(metadata_path)

    cards_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cards_path.write_text(
        json.dumps(
            [asdict(card) for card in cards],
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return cards, metadata


if __name__ == "__main__":
    cards, metadata = save_capability_profiles()

    print("\nC3-03 CAPABILITY PROFILES GENERATED")
    print(
        "Training prompts:",
        metadata["training_prompt_count"],
    )

    for card in cards:
        print(
            f"\nTier {card.tier}: {card.model_id}"
        )
        print(
            f"Mean cost: ${card.mean_cost_usd:.8f}"
        )
        print(
            "Capability:",
            [round(value, 5) for value in card.capability],
        )

    print("\nProbe version:", PROBE_VERSION)
    print("Latency: unavailable")
    print("ModelCards saved to:", CARDS_PATH)
    print("Metadata saved to:", METADATA_PATH)
