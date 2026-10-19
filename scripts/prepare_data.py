
"""Prepare the official Pillar 3 RouterBench dataset (C3-01).

Run from the repository root:
    python -m scripts.prepare_data
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.routerbench import (
    HF_FILE,
    POOL,
    PRICE_DATE,
    PRICE_SOURCE,
    RAW_DIR,
)
from measurement.routerbench import save_clean_routerbench


EXPECTED_SHA256 = (
    "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
)

CLEAN_DIR = Path("data/clean")
CLEAN_FILE = CLEAN_DIR / "routerbench_binary.parquet"
METADATA_FILE = CLEAN_DIR / "pool_metadata.json"


def verify_checksum(path: Path) -> None:
    """Verify the expected raw pickle before loading it."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download it first."
        )

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    if sha256.hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            "RouterBench checksum mismatch. "
            "Do not load this pickle file."
        )


def main() -> None:
    raw_path = RAW_DIR / HF_FILE

    print("Verifying RouterBench dataset...")
    verify_checksum(raw_path)
    print("Checksum verified.")

    print("Cleaning RouterBench records...")

    clean = save_clean_routerbench(
        raw_path=raw_path,
        output_path=CLEAN_FILE,
    )

    model_costs = [
        float(clean[f"{model}_cost"].mean())
        for model in POOL
    ]

    if not all(
        model_costs[i] < model_costs[i + 1]
        for i in range(len(model_costs) - 1)
    ):
        raise ValueError(
            "Model pool is not ordered by increasing average cost"
        )

    models = [
        {
            "model_id": model,
            "tier": index,
            "mean_cost_usd": model_costs[index],
        }
        for index, model in enumerate(POOL)
    ]

    metadata = {
        "schema_version": "c3-01-v1",
        "dataset": "withmartian/routerbench",
        "raw_file": HF_FILE,
        "raw_sha256": EXPECTED_SHA256,
        "prompt_count": len(clean),
        "source_counts": {
            str(source): int(count)
            for source, count in clean["source"].value_counts().items()
        },
        "price_source": PRICE_SOURCE,
        "price_date": PRICE_DATE,
        "models": models,
    }

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nC3-01 DATASET PREPARATION COMPLETE")
    print(f"Clean prompts: {len(clean)}")
    print(f"Sources: {clean['source'].nunique()}")
    print(f"Models: {len(POOL)}")
    print(f"Difficulty range: "
          f"{clean['difficulty'].min()}-"
          f"{clean['difficulty'].max()}")

    print("\nModel cost ordering:")
    for model in models:
        print(
            f"Tier {model['tier']}: "
            f"{model['model_id']} "
            f"USD {model['mean_cost_usd']:.8f}"
        )

    print(f"\nClean dataset: {CLEAN_FILE}")
    print(f"Model metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
