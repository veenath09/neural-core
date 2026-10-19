
"""C3-11: Guarded live LLM-as-router evaluation.

This runner is separate from scripts/evaluate.py so that
the working C3-09, C3-10, and mock C3-11 experiments remain
unchanged.

OFFLINE BY DEFAULT.

Live API access requires:
1. --enable-live-api
2. --acknowledge-billing BILLABLE_CALLS_APPROVED
3. OPENAI_API_KEY in the environment
4. Explicit provider model and pricing
5. Explicit spending and call-count limits

Pricing must be verified against the provider's official
documentation before a real run.

Spending limits are based on estimated request costs.
They cannot guarantee an exact provider billing ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

from pathlib import Path

import pandas as pd

from data.episodes import from_frame
from data.routerbench import POOL

from measurement.evaluate import (
    evaluate_episode,
    load_predictions,
    save_predictions,
)

from measurement.llm_router import RouterPricing

from measurement.openai_router_backend import (
    OpenAIResponsesBackend,
)

from measurement.routing_spend_guard import (
    BudgetGuardedLLMRouter,
    RoutingSpendGuard,
)

from scripts.evaluate import (
    AlwaysCheapest,
    CompositeScorer,
    _prepare_inputs,
    make_model_cards,
    sample_ids,
)


LIVE_POLICY = "llm-router-openai"

BASELINE_POLICY = "always-cheapest"

BILLING_CONFIRMATION = "BILLABLE_CALLS_APPROVED"


def positive_float(value: str) -> float:
    """Parse a strictly positive, finite number."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Expected a positive number"
        ) from exc

    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(
            "Expected a positive finite number"
        )

    return result


def positive_int(value: str) -> int:
    """Parse a strictly positive integer."""

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Expected a positive integer"
        ) from exc

    if result < 1:
        raise argparse.ArgumentTypeError(
            "Expected a positive integer"
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    """Create the safe live-evaluation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "C3-11 guarded live LLM router evaluation"
        )
    )

    parser.add_argument(
        "--eval-id",
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=1,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
    )

    parser.add_argument(
        "--lam",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model ID used as the router",
    )

    parser.add_argument(
        "--input-usd-per-million",
        type=positive_float,
    )

    parser.add_argument(
        "--output-usd-per-million",
        type=positive_float,
    )

    parser.add_argument(
        "--price-source",
        type=str,
        help="Source used to verify provider token prices",
    )

    parser.add_argument(
        "--max-total-usd",
        type=positive_float,
    )

    parser.add_argument(
        "--max-request-usd",
        type=positive_float,
    )

    parser.add_argument(
        "--max-calls",
        type=positive_int,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        default=128,
    )

    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
    )

    parser.add_argument(
        "--enable-live-api",
        action="store_true",
        help="Explicitly enable billable API requests",
    )

    parser.add_argument(
        "--acknowledge-billing",
        default=None,
        help="Required confirmation for live calls",
    )

    return parser


def validate_configuration(args) -> bool:
    """Validate all settings before any API client exists.

    Returns True only when the user explicitly authorizes
    live provider access.
    """

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        args.eval_id,
    ):
        raise ValueError(
            "Invalid evaluation ID"
        )

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(
            "Evaluation seeds must be unique"
        )

    if not math.isfinite(args.lam):
        raise ValueError(
            "Lambda must be finite"
        )

    if not 0 <= args.lam <= 100:
        raise ValueError(
            "Lambda must be between 0 and 100"
        )

    if not args.model.strip():
        raise ValueError(
            "Provider model cannot be empty"
        )

    if not args.enable_live_api:

        if args.acknowledge_billing is not None:
            raise ValueError(
                "Billing confirmation was supplied "
                "without --enable-live-api"
            )

        return False

    if (
        args.acknowledge_billing
        != BILLING_CONFIRMATION
    ):
        raise PermissionError(
            "Live API access requires explicit "
            "billing confirmation"
        )

    required = {
        "input_usd_per_million":
            args.input_usd_per_million,
        "output_usd_per_million":
            args.output_usd_per_million,
        "max_total_usd":
            args.max_total_usd,
        "max_request_usd":
            args.max_request_usd,
        "max_calls":
            args.max_calls,
        "price_source":
            args.price_source,
    }

    missing = [
        name
        for name, value in required.items()
        if value is None or value == ""
    ]

    if missing:
        raise ValueError(
            "Missing live spending configuration: "
            + ", ".join(missing)
        )

    if args.max_request_usd > args.max_total_usd:
        raise ValueError(
            "Per-request limit cannot exceed "
            "the total spending limit"
        )

    if (
        not args.price_source.startswith("https://")
        or "openai.com" not in args.price_source
    ):
        raise ValueError(
            "Provide an official HTTPS OpenAI "
            "pricing source"
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not configured"
        )

    return True


def load_inputs(episodes: int):
    """Load held-out test inputs and real C3-03 cards."""

    print(
        "Loading cleaned RouterBench data..."
    )

    clean = pd.read_parquet(
        "data/clean/routerbench_binary.parquet"
    )

    splits = pd.read_parquet(
        "data/clean/splits.parquet"
    )

    episode_frame = pd.read_parquet(
        "data/episodes/skeletons_test.parquet"
    )

    table, _ = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    cards = make_model_cards(
        clean,
        splits,
    )

    selected_ids = sample_ids(
        episode_frame,
        episodes,
    )

    selected = episode_frame.loc[
        episode_frame["episode_id"].isin(
            selected_ids
        )
    ]

    skeletons = from_frame(
        selected
    )

    if len(skeletons) != episodes:
        raise RuntimeError(
            "Selected episode count does not match "
            "the requested episode count"
        )

    prompt_text_by_id = dict(
        zip(
            clean["prompt_id"].astype(str),
            clean["prompt_text"].astype(str),
        )
    )

    difficulty_by_prompt = dict(
        zip(
            clean["prompt_id"],
            clean["difficulty"],
        )
    )

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    return (
        skeletons,
        table,
        cards,
        prompt_text_by_id,
        difficulty_by_prompt,
        scorer,
    )


def make_guarded_router(
    *,
    args,
    cards,
    prompt_text_by_id,
    client,
    guard,
):
    """Create a live router with mandatory spending checks."""

    pricing = RouterPricing(
        input_usd_per_million=(
            args.input_usd_per_million
        ),
        output_usd_per_million=(
            args.output_usd_per_million
        ),
        price_source=args.price_source,
    )

    backend = OpenAIResponsesBackend(
        client=client,
        model=args.model,
        max_output_tokens=(
            args.max_output_tokens
        ),
    )

    return BudgetGuardedLLMRouter(
        spending_guard=guard,
        cards=cards,
        prompt_text_by_id=prompt_text_by_id,
        backend=backend,
        pricing=pricing,
    )


def write_run_status(
    directory: Path,
    *,
    status: str,
    guard,
    router,
    completed_evaluations: int,
    error: str | None = None,
):
    """Save provider usage and run status, never API keys."""

    records = []

    for record in router.call_records:
        records.append(
            {
                "episode_id": record.episode_id,
                "step_id": record.step_id,
                "round": record.round,
                "router_model": record.router_model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "routing_cost_usd":
                    record.routing_cost_usd,
                "usage_source": record.usage_source,
                "selected_model_id":
                    record.selected_model_id,
                "valid_choice": record.valid_choice,
            }
        )

    summary = {
        "status": status,
        "completed_policy_evaluations":
            completed_evaluations,
        "provider_calls_started":
            guard.calls_started,
        "reported_routing_spend_usd":
            guard.actual_spent_usd,
        "remaining_guard_budget_usd":
            guard.remaining_run_budget_usd,
        "usage_uncertain":
            guard.usage_uncertain,
        "blocked":
            guard.blocked,
        "estimate_exceeded":
            guard.estimate_exceeded,
        "error": error,
        "billing_note": (
            "Provider usage can be uncertain after "
            "errors or timeouts. Guard limits are "
            "not a guaranteed billing ceiling."
        ),
    }

    with (
        directory / "run_status.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    with (
        directory / "router_calls.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=2,
        )


def run(args) -> Path | None:
    """Run an offline preflight or a guarded evaluation."""

    live = validate_configuration(
        args
    )

    directory = (
        Path("results")
        / args.eval_id
    )

    if directory.exists():
        raise FileExistsError(
            f"Evaluation directory already exists: "
            f"{directory}"
        )

    (
        skeletons,
        table,
        cards,
        prompt_text_by_id,
        difficulty_by_prompt,
        scorer,
    ) = load_inputs(
        args.episodes
    )

    print(
        "\nC3-11 evaluation configuration"
    )
    print(
        "------------------------------"
    )
    print(
        "Evaluation ID:", args.eval_id
    )
    print(
        "Episodes:", len(skeletons)
    )
    print(
        "Seeds:", args.seeds
    )
    print(
        "Router model:", args.model
    )
    print(
        "Split: test"
    )
    print(
        "Capability cards:", len(cards)
    )
    print(
        "Live API enabled:", live
    )

    if not live:
        print(
            "\nPASS: Offline configuration verified."
        )
        print(
            "No API client was created."
        )
        print(
            "No API requests were made."
        )
        print(
            "No provider costs were incurred "
            "by this script."
        )
        print(
            "No evaluation results were saved."
        )
        return None

    # Import and initialize the provider SDK only
    # after explicit billing authorization.
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        max_retries=0,
        timeout=args.timeout_seconds,
    )

    guard = RoutingSpendGuard(
        max_total_usd=(
            args.max_total_usd
        ),
        max_request_usd=(
            args.max_request_usd
        ),
        max_calls=args.max_calls,
        enabled=True,
    )

    router = make_guarded_router(
        args=args,
        cards=cards,
        prompt_text_by_id=(
            prompt_text_by_id
        ),
        client=client,
        guard=guard,
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    all_frames = []
    completed = 0

    try:
        for seed in args.seeds:

            print(
                f"\nEvaluating seed {seed}..."
            )

            baseline = AlwaysCheapest(
                cards
            )

            for index, skeleton in enumerate(
                skeletons,
                start=1,
            ):

                for name, policy in (
                    (BASELINE_POLICY, baseline),
                    (LIVE_POLICY, router),
                ):

                    frame = evaluate_episode(
                        skeleton,
                        table,
                        POOL,
                        policy,
                        policy_name=name,
                        scorer=scorer,
                        seed=seed,
                        lam=args.lam,
                        split="test",
                        difficulty_by_prompt=(
                            difficulty_by_prompt
                        ),
                    )

                    all_frames.append(
                        frame
                    )

                    completed += 1

                    # Preserve every completed episode.
                    pd.concat(
                        all_frames,
                        ignore_index=True,
                    ).to_parquet(
                        directory
                        / "partial_predictions.parquet",
                        index=False,
                    )

                    write_run_status(
                        directory,
                        status="running",
                        guard=guard,
                        router=router,
                        completed_evaluations=completed,
                    )

                print(
                    f"Completed {index}/{len(skeletons)} "
                    f"episodes for seed {seed}"
                )

        combined = pd.concat(
            all_frames,
            ignore_index=True,
        )

        expected = (
            len(skeletons)
            * len(args.seeds)
            * 2
        )

        actual = int(
            combined["terminal"].sum()
        )

        if actual != expected:
            raise RuntimeError(
                "Incomplete evaluation: "
                f"expected {expected}, got {actual}"
            )

        saved_path = save_predictions(
            combined,
            directory
            / "predictions.parquet",
        )

        saved = load_predictions(
            saved_path
        )

        if int(saved["terminal"].sum()) != expected:
            raise RuntimeError(
                "Persisted prediction count mismatch"
            )

        live_rows = saved.loc[
            saved["policy"].eq(
                LIVE_POLICY
            )
        ]

        if (
            live_rows.empty
            or not live_rows[
                "routing_usage_source"
            ].eq("provider").all()
        ):
            raise RuntimeError(
                "Live evaluation is missing "
                "provider usage records"
            )

        terminal = live_rows.loc[
            live_rows["terminal"]
        ]

        differences = (
            terminal["episode_cost_usd"].astype(float)
            - terminal[
                "episode_routing_cost_usd"
            ].astype(float)
            - terminal[
                "episode_execution_cost_usd"
            ].astype(float)
        )

        if not differences.abs().le(
            1e-8
        ).all():
            raise RuntimeError(
                "Routing and execution costs "
                "do not reconcile"
            )

        write_run_status(
            directory,
            status="completed",
            guard=guard,
            router=router,
            completed_evaluations=completed,
        )

        print(
            "\nPASS: Live evaluation completed."
        )
        print(
            "Predictions:", saved_path
        )
        print(
            "Completed policy episodes:", actual
        )
        print(
            "Reported routing cost (USD):",
            guard.actual_spent_usd,
        )

        return saved_path

    except Exception as exc:

        write_run_status(
            directory,
            status="failed",
            guard=guard,
            router=router,
            completed_evaluations=completed,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        print(
            "\nEvaluation stopped safely."
        )
        print(
            "Completed predictions, if any, are "
            "in partial_predictions.parquet"
        )
        print(
            "Review run_status.json before retrying."
        )

        raise


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
