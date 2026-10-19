
"""RouterBench evaluation runner for C3-09, C3-10, and C3-11.

C3-09: Four original baseline policies.
C3-10: Optional trained context-free router.
C3-11: Offline LLM mock router with routing-cost accounting.

The mock backend uses SYNTHETIC token counts and prices. It makes no API
calls and is not a benchmark of real LLM routing performance.

Prediction records are saved to results/<eval-id>/predictions.parquet.
Existing evaluation outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd

from controller.contracts import ModelCard
from data.episodes import from_frame
from data.routerbench import POOL
from measurement.baselines import (
    AlwaysCheapest,
    AlwaysStrongest,
    EpisodeOracleBaseline,
    RandomBaseline,
)
from measurement.context_free_router import ContextFreeRouter
from measurement.episodes import _prepare_inputs
from measurement.evaluate import (
    evaluate_policies,
    load_predictions,
    save_predictions,
)
from measurement.llm_router import LLMRouter, MockLLMBackend, RouterPricing
from measurement.scorer import CompositeScorer
from scripts.validate_oracle import sample_ids


# ============================================================
# 1. POLICY DEFINITIONS
# ============================================================

# The four original C3-09 policies remain the CLI default.
POLICY_NAMES = (
    "always-cheapest",
    "always-strongest",
    "random",
    "episode-oracle",
)

CONTEXT_FREE_POLICY = "context-free-router"

LLM_MOCK_POLICY = "llm-router-mock"

ALL_POLICY_NAMES = (
    POLICY_NAMES
    + (CONTEXT_FREE_POLICY, LLM_MOCK_POLICY)
)


# ============================================================
# 2. MOCK LLM CONFIGURATION
# ============================================================

# Synthetic offline test values.
# These are NOT OpenAI provider prices.

MOCK_INPUT_TOKENS = 120
MOCK_OUTPUT_TOKENS = 20

MOCK_INPUT_USD_PER_MILLION = 1.0
MOCK_OUTPUT_USD_PER_MILLION = 2.0


# ============================================================
# 3. C3-03 CAPABILITY CONFIGURATION
# ============================================================

CAPABILITY_LAYOUT = [
    "accuracy_mmlu",
    "accuracy_hellaswag",
    "accuracy_arc-challenge",
    "accuracy_winogrande",
    "accuracy_mbpp",
    "log10_mean_cost_usd",
]

CAPABILITY_SOURCES = (
    "mmlu",
    "hellaswag",
    "arc-challenge",
    "winogrande",
    "mbpp",
)


# ============================================================
# 4. CREATE MOCK LLM ROUTER
# ============================================================

def make_mock_llm_router(
    cards: list[ModelCard],
    prompt_text_by_id: Mapping[str, str],
) -> LLMRouter:
    """Build the deterministic offline mock router.

    No OpenAI API or network connection is required.
    """

    backend = MockLLMBackend(
        input_tokens=MOCK_INPUT_TOKENS,
        output_tokens=MOCK_OUTPUT_TOKENS,
    )

    pricing = RouterPricing(
        input_usd_per_million=(
            MOCK_INPUT_USD_PER_MILLION
        ),
        output_usd_per_million=(
            MOCK_OUTPUT_USD_PER_MILLION
        ),
        price_source="synthetic-offline-test-pricing",
    )

    return LLMRouter(
        cards=cards,
        prompt_text_by_id=prompt_text_by_id,
        backend=backend,
        pricing=pricing,
    )


# ============================================================
# 5. LOAD ORIGINAL C3-03 MODEL CARDS
# ============================================================

def make_model_cards(
    clean: pd.DataFrame,
    splits: pd.DataFrame,
) -> list[ModelCard]:
    """Load the original C3-03 training-derived cards.

    Unlike the earlier mock runner, this method never
    substitutes placeholder zeros for capability vectors.

    Model-card and profile values are validated against
    the active training split and RouterBench model pool.

    Metadata provides recorded provenance; it is not
    independent proof that generation was leak-free.
    """

    cards_path = Path(
        "data/clean/model_cards.json"
    )

    metadata_path = Path(
        "data/clean/capability_metadata.json"
    )

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not cards_path.is_file():
        raise FileNotFoundError(
            f"Missing C3-03 model cards: {cards_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing C3-03 metadata: {metadata_path}"
        )

    # --------------------------------------------------------
    # Load JSON artifacts
    # --------------------------------------------------------

    with cards_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_cards = json.load(file)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # --------------------------------------------------------
    # Validate file structures
    # --------------------------------------------------------

    if (
        not isinstance(raw_cards, list)
        or len(raw_cards) != len(POOL)
    ):
        raise ValueError(
            "C3-03 model-card list does not match "
            "the RouterBench model pool"
        )

    if not isinstance(metadata, dict):
        raise ValueError(
            "C3-03 capability metadata must be a JSON object"
        )

    # --------------------------------------------------------
    # Validate training-only provenance
    # --------------------------------------------------------

    if metadata.get("training_only") is not True:
        raise ValueError(
            "C3-03 capability metadata must declare "
            "training_only"
        )

    if metadata.get("probe_version") != "rb_train_v1":
        raise ValueError(
            "Unexpected C3-03 probe version"
        )

    if metadata.get(
        "controller_capability_layout"
    ) != CAPABILITY_LAYOUT:
        raise ValueError(
            "Unexpected C3-03 capability feature layout"
        )

    # --------------------------------------------------------
    # Validate training prompt count
    # --------------------------------------------------------

    train_ids = set(
        splits.loc[
            splits["split"].eq("train"),
            "prompt_id",
        ].astype(str)
    )

    clean_ids = set(
        clean["prompt_id"].astype(str)
    )

    if not train_ids.issubset(clean_ids):
        raise ValueError(
            "Training prompts are missing "
            "from the clean dataset"
        )

    if len(train_ids) != metadata.get(
        "training_prompt_count"
    ):
        raise ValueError(
            "C3-03 training count differs "
            "from the loaded split"
        )

    # --------------------------------------------------------
    # Validate capability profiles
    # --------------------------------------------------------

    profiles = metadata.get(
        "profiles"
    )

    if (
        not isinstance(profiles, list)
        or len(profiles) != len(POOL)
    ):
        raise ValueError(
            "C3-03 model capability profiles "
            "are incomplete"
        )

    if any(
        not isinstance(profile, dict)
        for profile in profiles
    ):
        raise ValueError(
            "Invalid C3-03 capability profile format"
        )

    profile_lookup = {
        profile["model_id"]: profile
        for profile in profiles
    }

    if (
        len(profile_lookup) != len(POOL)
        or set(profile_lookup) != set(POOL)
    ):
        raise ValueError(
            "C3-03 profile model IDs do not "
            "match RouterBench"
        )

    # --------------------------------------------------------
    # Validate each model card
    # --------------------------------------------------------

    cards: list[ModelCard] = []

    for tier, (model_id, raw) in enumerate(
        zip(POOL, raw_cards)
    ):

        if (
            not isinstance(raw, dict)
            or raw.get("model_id") != model_id
        ):
            raise ValueError(
                f"Unexpected model card or order: {model_id}"
            )

        if raw.get("tier") != tier:
            raise ValueError(
                f"Incorrect model tier: {model_id}"
            )

        card = ModelCard(**raw)

        cost = float(
            card.mean_cost_usd
        )

        if not math.isfinite(cost) or cost <= 0:
            raise ValueError(
                f"Invalid model cost: {model_id}"
            )

        capabilities = list(
            card.capability
        )

        if len(capabilities) != 6:
            raise ValueError(
                f"Expected six capability values: {model_id}"
            )

        if not all(
            math.isfinite(float(value))
            for value in capabilities
        ):
            raise ValueError(
                f"Nonfinite capability values: {model_id}"
            )

        # The first five features are benchmark accuracies.
        if not all(
            0.0 <= float(value) <= 1.0
            for value in capabilities[:5]
        ):
            raise ValueError(
                f"Invalid task accuracies: {model_id}"
            )

        # The sixth feature is log10(mean_cost_usd).
        if not math.isclose(
            float(capabilities[5]),
            math.log10(cost),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Capability cost feature mismatch: {model_id}"
            )

        # ----------------------------------------------------
        # Compare with original C3-03 metadata
        # ----------------------------------------------------

        profile = profile_lookup[
            model_id
        ]

        if profile.get("tier") != tier:
            raise ValueError(
                f"C3-03 profile tier mismatch: {model_id}"
            )

        if profile.get(
            "training_prompt_count"
        ) != len(train_ids):
            raise ValueError(
                f"C3-03 profile training count "
                f"mismatch: {model_id}"
            )

        if not math.isclose(
            cost,
            float(profile["mean_cost_usd"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"C3-03 mean cost mismatch: {model_id}"
            )

        source_accuracies = profile[
            "accuracy_by_source"
        ]

        for index, source in enumerate(
            CAPABILITY_SOURCES
        ):

            expected_accuracy = float(
                source_accuracies[source]
            )

            if not math.isclose(
                float(capabilities[index]),
                expected_accuracy,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"C3-03 accuracy mismatch: "
                    f"{model_id}, {source}"
                )

        cards.append(
            card
        )

    print(
        "PASS: Loaded four C3-03 training-derived "
        "model capability cards"
    )

    return cards


# ============================================================
# 6. LOAD TRAINED C3-10 CONTEXT-FREE ROUTER
# ============================================================

def load_context_free_router(
    router_model: str | Path,
) -> ContextFreeRouter:
    """Load a trusted C3-10 ContextFreeRouter artifact.

    Only load joblib files you trust because loading
    a joblib artifact can execute Python code.
    """

    path = Path(
        router_model
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Trained router model not found: {path}"
        )

    print(
        "Loading trained context-free router..."
    )

    loaded = joblib.load(
        path
    )

    if not isinstance(
        loaded,
        ContextFreeRouter,
    ):
        raise TypeError(
            "The supplied artifact is not "
            "a ContextFreeRouter"
        )

    if tuple(loaded.model_ids) != tuple(POOL):
        raise ValueError(
            "Trained router's model pool "
            "does not match RouterBench"
        )

    print(
        f"PASS: Loaded context-free router: {path}"
    )

    return loaded


# ============================================================
# 7. CREATE SELECTED POLICIES
# ============================================================

def make_policies(
    cards: list[ModelCard],
    names: list[str],
    seed: int,
    router: ContextFreeRouter | None = None,
    prompt_text_by_id: Mapping[str, str] | None = None,
) -> dict:
    """Create requested C3-09, C3-10, and C3-11 policies."""

    if (
        CONTEXT_FREE_POLICY in names
        and router is None
    ):
        raise ValueError(
            "context-free-router requires "
            "a trained router model"
        )

    if (
        LLM_MOCK_POLICY in names
        and prompt_text_by_id is None
    ):
        raise ValueError(
            "llm-router-mock requires "
            "a prompt text lookup"
        )

    available = {
        "always-cheapest": (
            lambda: AlwaysCheapest(cards)
        ),

        "always-strongest": (
            lambda: AlwaysStrongest(cards)
        ),

        "random": (
            lambda: RandomBaseline(
                cards,
                seed=seed,
            )
        ),

        "episode-oracle": (
            lambda: EpisodeOracleBaseline(
                POOL
            )
        ),
    }

    if router is not None:
        available[CONTEXT_FREE_POLICY] = (
            lambda: router
        )

    if prompt_text_by_id is not None:
        available[LLM_MOCK_POLICY] = (
            lambda: make_mock_llm_router(
                cards,
                prompt_text_by_id,
            )
        )

    unknown = [
        name
        for name in names
        if name not in available
    ]

    if unknown:
        raise ValueError(
            f"Unknown or unavailable policies: {unknown}"
        )

    return {
        name: available[name]()
        for name in names
    }


# ============================================================
# 8. RUN ROUTERBENCH EVALUATION
# ============================================================

def run_evaluation(
    *,
    eval_id: str,
    episodes: int,
    seeds: list[int],
    policy_names: list[str],
    lam: float,
    router_model: str | None = None,
) -> Path:
    """Run requested policies on held-out test episodes."""

    # --------------------------------------------------------
    # Validate evaluation configuration
    # --------------------------------------------------------

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        eval_id,
    ):
        raise ValueError(
            "Evaluation ID may contain only "
            "letters, numbers, hyphens and underscores"
        )

    if episodes < 1:
        raise ValueError(
            "Episode count must be positive"
        )

    if (
        not seeds
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "Specify at least one unique seed"
        )

    if (
        not policy_names
        or len(set(policy_names)) != len(policy_names)
    ):
        raise ValueError(
            "Specify at least one unique policy"
        )

    unknown = (
        set(policy_names)
        - set(ALL_POLICY_NAMES)
    )

    if unknown:
        raise ValueError(
            f"Unknown evaluation policies: {sorted(unknown)}"
        )

    if (
        not math.isfinite(lam)
        or not 0 <= lam <= 100
    ):
        raise ValueError(
            "Lambda must be finite "
            "and between 0 and 100"
        )

    # --------------------------------------------------------
    # Prevent existing evaluations from being overwritten
    # --------------------------------------------------------

    destination = (
        Path("results")
        / eval_id
        / "predictions.parquet"
    )

    if destination.exists():
        raise FileExistsError(
            f"Evaluation already exists: {destination}"
        )

    # --------------------------------------------------------
    # Load optional C3-10 router
    # --------------------------------------------------------

    router = None

    if CONTEXT_FREE_POLICY in policy_names:

        if router_model is None:
            raise ValueError(
                "context-free-router requires --router-model"
            )

        router = load_context_free_router(
            router_model
        )

    # --------------------------------------------------------
    # Load RouterBench dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prepare evaluation environment
    # --------------------------------------------------------

    table, _ = _prepare_inputs(
        clean,
        splits,
        "test",
    )

    # Load actual train-derived C3-03 model cards.
    cards = make_model_cards(
        clean,
        splits,
    )

    scorer = CompositeScorer.from_yaml(
        "conf/score.yaml"
    )

    # --------------------------------------------------------
    # Build prompt text lookup
    # --------------------------------------------------------

    # Prompt text is observable input data.
    # Ground-truth labels are not included.
    prompt_text_by_id = dict(
        zip(
            clean["prompt_id"].astype(str),
            clean["prompt_text"].astype(str),
        )
    )

    # Difficulty is evaluation metadata only.
    difficulty_by_prompt = dict(
        zip(
            clean["prompt_id"],
            clean["difficulty"],
        )
    )

    # --------------------------------------------------------
    # Select held-out test episodes
    # --------------------------------------------------------

    selected_ids = sample_ids(
        episode_frame,
        episodes,
    )

    selected_frame = episode_frame.loc[
        episode_frame["episode_id"].isin(
            selected_ids
        )
    ]

    skeletons = from_frame(
        selected_frame
    )

    if not skeletons:
        raise ValueError(
            "No test episode skeletons were selected"
        )

    # --------------------------------------------------------
    # Print evaluation configuration
    # --------------------------------------------------------

    print(
        "\nEvaluation configuration"
    )

    print(
        "------------------------"
    )

    print(
        "Evaluation ID:",
        eval_id,
    )

    print(
        "Episodes:",
        len(skeletons),
    )

    print(
        "Policies:",
        ", ".join(policy_names),
    )

    print(
        "Seeds:",
        seeds,
    )

    print(
        "Lambda:",
        lam,
    )

    print(
        "Split: test"
    )

    print(
        "Episode types:",
        selected_frame.drop_duplicates(
            "episode_id"
        )["ep_type"].value_counts().to_dict(),
    )

    if router is not None:
        print(
            "Context-free router:",
            router_model,
        )

    if LLM_MOCK_POLICY in policy_names:

        print(
            "\nC3-11 OFFLINE MOCK ROUTER ENABLED"
        )

        print(
            "No live API calls will be made."
        )

        print(
            "Token counts and router prices are SYNTHETIC."
        )

        print(
            "Real train-derived capability cards "
            "are loaded from C3-03."
        )

        print(
            "Mock model selection: cheapest affordable."
        )

        print(
            "Mock results are NOT real LLM routing benchmarks."
        )

    # --------------------------------------------------------
    # Evaluate every requested seed
    # --------------------------------------------------------

    all_predictions = []

    for seed in seeds:

        print(
            f"\nEvaluating seed {seed}..."
        )

        policies = make_policies(
            cards,
            policy_names,
            seed,
            router=router,
            prompt_text_by_id=(
                prompt_text_by_id
                if LLM_MOCK_POLICY in policy_names
                else None
            ),
        )

        frame = evaluate_policies(
            skeletons,
            table,
            POOL,
            policies,
            scorer=scorer,
            seed=seed,
            lam=lam,
            split="test",
            difficulty_by_prompt=(
                difficulty_by_prompt
            ),
        )

        all_predictions.append(
            frame
        )

        print(
            f"Seed {seed}: evaluation completed"
        )

    # --------------------------------------------------------
    # Combine all prediction records
    # --------------------------------------------------------

    combined = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save predictions to Parquet
    # --------------------------------------------------------

    saved_path = save_predictions(
        combined,
        destination,
    )

    # Always verify from persisted predictions.
    saved = load_predictions(
        saved_path
    )

    # --------------------------------------------------------
    # Verify completed episodes
    # --------------------------------------------------------

    expected_terminal = (
        len(skeletons)
        * len(policy_names)
        * len(seeds)
    )

    actual_terminal = int(
        saved["terminal"].sum()
    )

    if actual_terminal != expected_terminal:
        raise RuntimeError(
            "Unexpected number of terminal policy episodes"
        )

    if set(
        saved["policy"].unique()
    ) != set(policy_names):
        raise RuntimeError(
            "Saved predictions are missing "
            "requested policies"
        )

    if {
        int(s)
        for s in saved["seed"].unique()
    } != set(seeds):
        raise RuntimeError(
            "Saved predictions are missing "
            "evaluation seeds"
        )

    # --------------------------------------------------------
    # Verify C3-11 mock routing costs
    # --------------------------------------------------------

    if LLM_MOCK_POLICY in policy_names:

        required = {
            "routing_cost_usd",
            "routing_input_tokens",
            "routing_output_tokens",
            "routing_usage_source",
            "episode_routing_cost_usd",
            "episode_execution_cost_usd",
            "episode_cost_usd",
        }

        missing = (
            required
            - set(saved.columns)
        )

        if missing:
            raise RuntimeError(
                f"Missing C3-11 cost fields: "
                f"{sorted(missing)}"
            )

        mock = saved.loc[
            saved["policy"].eq(
                LLM_MOCK_POLICY
            )
        ]

        if (
            mock.empty
            or not mock[
                "routing_usage_source"
            ].eq("mock").all()
        ):
            raise RuntimeError(
                "Missing or incorrectly labeled "
                "mock router rows"
            )

        if not mock[
            "routing_cost_usd"
        ].gt(0).all():
            raise RuntimeError(
                "Mock routing charges "
                "were not recorded"
            )

        terminal = mock.loc[
            mock["terminal"]
        ]

        difference = (
            terminal[
                "episode_cost_usd"
            ].astype(float)
            - terminal[
                "episode_routing_cost_usd"
            ].astype(float)
            - terminal[
                "episode_execution_cost_usd"
            ].astype(float)
        )

        if not difference.abs().le(
            1e-8
        ).all():
            raise RuntimeError(
                "C3-11 routing and execution "
                "costs do not reconcile"
            )

        print(
            "\nPASS: Saved mock routing and "
            "execution costs reconcile"
        )

    # --------------------------------------------------------
    # Print final evaluation results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "ROUTERBENCH EVALUATION SAVED"
    )

    print(
        "=" * 60
    )

    print(
        "Output:",
        saved_path,
    )

    print(
        "Prediction rows:",
        len(saved),
    )

    print(
        "Completed policy episodes:",
        actual_terminal,
    )

    print(
        "Policies:",
        sorted(
            saved["policy"].unique()
        ),
    )

    print(
        "Seeds:",
        sorted(
            {
                int(s)
                for s in saved["seed"].unique()
            }
        ),
    )

    print(
        "PASS: Predictions saved and verified"
    )

    if LLM_MOCK_POLICY in policy_names:

        print(
            "NOTE: Routing tokens, prices "
            "and choices are synthetic."
        )

        print(
            "NOTE: These results do NOT measure "
            "a real LLM router."
        )

    print(
        "NOTE: Historical RouterBench "
        "execution latency is unmeasured."
    )

    return saved_path


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    """CLI entry point.

    Original C3-09 baseline policies remain the default.
    """

    parser = argparse.ArgumentParser(
        description=(
            "C3-09 / C3-10 / C3-11 "
            "RouterBench evaluation runner"
        )
    )

    parser.add_argument(
        "--eval-id",
        required=True,
        help="Unique evaluation ID",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=ALL_POLICY_NAMES,
        default=list(POLICY_NAMES),
        help=(
            "Policies to evaluate "
            "(defaults to the original four baselines)"
        ),
    )

    parser.add_argument(
        "--lam",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--router-model",
        type=str,
        default=None,
        help=(
            "Path to trained C3-10 "
            "ContextFreeRouter joblib artifact"
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        eval_id=args.eval_id,
        episodes=args.episodes,
        seeds=args.seeds,
        policy_names=args.policies,
        lam=args.lam,
        router_model=args.router_model,
    )


if __name__ == "__main__":
    main()
