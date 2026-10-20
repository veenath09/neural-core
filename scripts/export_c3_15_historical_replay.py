
"""C3-15: Export historical RouterBench response replay.

Preserves original RouterBench model responses, scores,
and costs alongside the simulator's observed outcomes.

Simulator and historical correctness are deliberately
separate: stochastic simulation may change outcomes.

This exporter performs no live model execution or API calls.

Only load the raw pickle file from a trusted source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path(
    "results/c3-15/profile_aware_after_manifest.jsonl"
)

DEFAULT_RAW = Path(
    "data/raw/routerbench_0shot.pkl"
)

DEFAULT_OUTPUT = Path(
    "results/c3-15/historical_response_replay.jsonl"
)


def parse_response(value: object) -> str:
    """Decode a stringified response without executing code."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Historical response must be nonempty text"
        )

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value

    if isinstance(parsed, str):
        text = parsed

    elif isinstance(parsed, (list, tuple)):
        if (
            not parsed
            or not all(
                isinstance(part, str)
                for part in parsed
            )
        ):
            raise ValueError(
                "Unexpected response list structure"
            )

        text = "\n".join(parsed)

    else:
        raise ValueError(
            "Unsupported historical response structure"
        )

    if not text.strip():
        raise ValueError(
            "Parsed historical response is empty"
        )

    return text


def validate_cost(value: object, name: str) -> float:
    """Accept only nonnegative, finite cost values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid {name}")

    try:
        cost = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc

    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"Invalid {name}")

    return cost


def validate_correctness(
    value: object,
    name: str,
) -> bool:
    """Validate a binary RouterBench correctness score."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc

    if (
        not math.isfinite(numeric)
        or numeric not in (0.0, 1.0)
    ):
        raise ValueError(f"Invalid {name}")

    return bool(numeric)


def export_historical_replay(
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_path: Path = DEFAULT_RAW,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Export recorded historical model responses.

    Correctness fields:
        simulator_correct:
            Outcome observed in the simulated episode.

        historical_correct:
            Original RouterBench model score.

        correctness_agreement:
            Whether those two outcomes agree.

    These values must not be interpreted as new live
    model-execution observations.
    """

    manifest = pd.read_json(
        manifest_path,
        lines=True,
        dtype={
            "episode_id": "string",
            "step_id": "string",
            "prompt_id": "string",
            "model_id": "string",
        },
    )

    raw = pd.read_pickle(raw_path)

    required_manifest = {
        "policy",
        "onboarding_phase",
        "split",
        "seed",
        "episode_id",
        "decision_index",
        "step_id",
        "round",
        "prompt_id",
        "step_type",
        "source",
        "model_id",
        "prompt_text",
        "selected_model_correct",
        "execution_cost_usd",
        "measurement_source",
    }

    missing = required_manifest.difference(
        manifest.columns
    )

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError("Manifest is empty")

    if not manifest[
        "measurement_source"
    ].eq("simulator").all():
        raise ValueError(
            "Manifest contains non-simulator records"
        )

    attempt_keys = [
        "episode_id",
        "decision_index",
    ]

    if manifest[attempt_keys].isna().any().any():
        raise ValueError(
            "Missing execution attempt identifiers"
        )

    if manifest.duplicated(
        subset=attempt_keys
    ).any():
        raise ValueError(
            "Duplicate simulator execution attempts"
        )

    if "sample_id" not in raw.columns:
        raise ValueError(
            "Original RouterBench sample_id is missing"
        )

    if raw["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate RouterBench sample IDs"
        )

    joined = manifest.merge(
        raw,
        left_on="prompt_id",
        right_on="sample_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not joined["_merge"].eq("both").all():
        raise ValueError(
            "Some manifest prompts are missing "
            "from the original RouterBench dataset"
        )

    replay_rows = []

    for record in joined.to_dict(orient="records"):

        model = record["model_id"]

        response_column = (
            f"{model}|model_response"
        )

        cost_column = (
            f"{model}|total_cost"
        )

        for column in (
            model,
            response_column,
            cost_column,
        ):
            if column not in joined.columns:
                raise ValueError(
                    f"Missing RouterBench column: {column}"
                )

        historical_response_raw = record[
            response_column
        ]

        historical_response_text = parse_response(
            historical_response_raw
        )

        simulator_correct = validate_correctness(
            record["selected_model_correct"],
            "simulator correctness",
        )

        historical_correct = validate_correctness(
            record[model],
            "historical correctness",
        )

        simulator_cost = validate_cost(
            record["execution_cost_usd"],
            "simulator cost",
        )

        historical_cost = validate_cost(
            record[cost_column],
            "historical cost",
        )

        # Cost inconsistencies indicate a possible join,
        # transformation, or data-integrity problem.
        if not math.isclose(
            simulator_cost,
            historical_cost,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Historical cost differs from simulator cost "
                f"for {record['prompt_id']} / {model}"
            )

        # Correctness mismatches are intentionally allowed.
        # They are observations to analyze, not overwrite.
        agreement = (
            simulator_correct == historical_correct
        )

        replay_rows.append(
            {
                "policy": record["policy"],
                "onboarding_phase": (
                    record["onboarding_phase"]
                ),
                "split": record["split"],
                "seed": record["seed"],
                "episode_id": record["episode_id"],
                "decision_index": int(
                    record["decision_index"]
                ),
                "step_id": record["step_id"],
                "round": int(record["round"]),
                "prompt_id": record["prompt_id"],
                "step_type": int(
                    record["step_type"]
                ),
                "source": record["source"],
                "model_id": model,
                "prompt_text": record["prompt_text"],
                "historical_response_raw": (
                    historical_response_raw
                ),
                "historical_response_text": (
                    historical_response_text
                ),
                "simulator_correct": (
                    simulator_correct
                ),
                "historical_correct": (
                    historical_correct
                ),
                "correctness_agreement": agreement,
                "simulator_cost_usd": (
                    simulator_cost
                ),
                "historical_cost_usd": (
                    historical_cost
                ),
                "measurement_source": (
                    "historical_replay"
                ),
                "response_source": (
                    "routerbench_0shot.pkl"
                ),
                "verification_source": (
                    "routerbench_recorded_score"
                ),
                "cost_source": (
                    "routerbench_recorded_total_cost"
                ),
            }
        )

    replay = pd.DataFrame(replay_rows)

    replay = replay.sort_values(
        [
            "episode_id",
            "decision_index",
        ],
        kind="stable",
    ).reset_index(drop=True)

    matches = int(
        replay["correctness_agreement"].sum()
    )

    mismatches = len(replay) - matches

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
        double_precision=15,
    )

    digest = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    print("C3-15 HISTORICAL RESPONSE REPLAY")
    print("-" * 42)

    print(
        "Execution attempts:",
        len(replay),
    )

    print(
        "Episodes:",
        replay["episode_id"].nunique(),
    )

    print(
        "Responses exported:",
        replay["historical_response_text"]
        .notna()
        .sum(),
    )

    print("\nCORRECTNESS COMPARISON")

    print(
        "Matches:",
        matches,
    )

    print(
        "Mismatches:",
        mismatches,
    )

    print(
        "Agreement rate:",
        f"{100 * matches / len(replay):.2f}%",
    )

    print(
        "Simulator success rate:",
        f"{100 * replay['simulator_correct'].mean():.2f}%",
    )

    print(
        "Historical success rate:",
        f"{100 * replay['historical_correct'].mean():.2f}%",
    )

    print("\nCOST COMPARISON")

    print(
        "Simulator cost USD:",
        f"{replay['simulator_cost_usd'].sum():.6f}",
    )

    print(
        "Historical cost USD:",
        f"{replay['historical_cost_usd'].sum():.6f}",
    )

    print("\nMODEL DISTRIBUTION")

    print(
        replay["model_id"]
        .value_counts()
        .to_string()
    )

    print("\nOUTPUT")

    print(output_path)

    print("SHA-256:", digest)

    print(
        "\nStatus: HISTORICAL REPLAY ONLY"
    )

    print(
        "No live model executions were performed."
    )

    return replay


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Export original RouterBench model responses "
            "without changing simulator outcomes"
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    export_historical_replay(
        manifest_path=args.manifest,
        raw_path=args.raw,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
