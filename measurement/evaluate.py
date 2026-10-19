
"""Pillar 3: Reproducible evaluation harness.

C3-09:
    Evaluate ControllerAPI policies against RouterBench episodes.

C3-10:
    Support the trained context-free router.

C3-11:
    Account for LLM routing expenses separately from model
    execution expenses.

Predictions are recorded per routing/execution decision.
Final episode metrics appear on the terminal row.

Research reports must be calculated from persisted
predictions.parquet, not from in-memory summaries.

Historical RouterBench execution costs are simulated.
Live execution latency is unavailable and is not fabricated.

Mock LLM routing costs are synthetic, not provider bills.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from controller.contracts import Strategy

from data.env import Episode, VERIFIER_FLIP_P
from data.episodes import Skeleton
from data.routerbench import OutcomeTable

from measurement.baselines import (
    ControllerAPI,
    EpisodeOracleBaseline,
)

from measurement.llm_router import LLMRouter
from measurement.scorer import CompositeScorer
from measurement.simulator import Simulator


# ============================================================
# 1. EVALUATE ONE EPISODE
# ============================================================

def evaluate_episode(
    skeleton: Skeleton,
    table: OutcomeTable,
    model_ids: Sequence[str],
    policy: ControllerAPI,
    *,
    policy_name: str,
    scorer: CompositeScorer,
    seed: int = 1234,
    flip_p: float = VERIFIER_FLIP_P,
    lam: float = 0.25,
    split: str = "test",
    difficulty_by_prompt: Mapping[str, int] | None = None,
    max_decisions: int = 64,
) -> pd.DataFrame:
    """Evaluate one episode and return prediction records.

    C3-09 and C3-10 controllers:
        Use the normal simulator and execution costs.

    C3-11 LLMRouter:
        Record routing tokens and costs.
        Charge routing costs before model execution.
        Include routing expenses in total episode cost.
        Include routing expenses in composite reward.
        Record invalid responses and budget overruns.

    Evaluation-only correctness is captured after decide().
    It is never passed to the controller.
    """

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    if not isinstance(policy_name, str) or not policy_name:
        raise ValueError(
            "Policy name must be a nonempty string"
        )

    if not isinstance(policy, ControllerAPI):
        raise TypeError(
            "Policy must implement ControllerAPI"
        )

    if max_decisions < 1:
        raise ValueError(
            "Maximum decisions must be positive"
        )

    model_ids = tuple(model_ids)

    # --------------------------------------------------------
    # INITIALIZE SIMULATOR
    # --------------------------------------------------------

    simulator = Simulator(
        skeleton,
        table,
        model_ids,
        lam=lam,
        flip_p=flip_p,
        seed=seed,
    )

    # Only the privileged oracle receives the actual
    # environment that contains correctness outcomes.
    if isinstance(policy, EpisodeOracleBaseline):
        oracle_environment = Episode(
            skeleton,
            table,
            model_ids,
            flip_p=flip_p,
            seed=seed,
        )

        policy.bind_episode(oracle_environment)

    is_llm_router = isinstance(
        policy,
        LLMRouter,
    )

    rows: list[dict] = []

    # These counters belong to this episode only.
    # A policy object may be reused across episodes.
    episode_routing_cost = 0.0
    episode_execution_cost = 0.0

    # --------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------

    for decision_index in range(max_decisions):

        task = simulator.current_task()

        if task is None:
            break

        strategy = None
        routing_record = None
        router_error = None

        routing_cost = 0.0

        # Full routing invoice not applied to simulator
        # state because it exceeds remaining budget.
        routing_budget_overrun = 0.0

        # ----------------------------------------------------
        # A. RECORD ROUTER STATE BEFORE THE DECISION
        # ----------------------------------------------------

        records_before = (
            len(policy.call_records)
            if is_llm_router
            else 0
        )

        # ----------------------------------------------------
        # B. ASK THE CONTROLLER TO CHOOSE A MODEL
        # ----------------------------------------------------

        try:
            strategy = policy.decide(
                task
            )

        except ValueError as exc:

            # A completed LLM call can produce invalid JSON
            # or an invalid model selection.
            #
            # LLMRouter records the completed call even
            # when its response fails validation.
            #
            # For ordinary policies, keep the existing
            # behavior of propagating the exception.
            #
            # Provider failures with unknown usage cannot
            # be billed accurately and must not be treated
            # as free completed evaluations.
            if (
                not is_llm_router
                or len(policy.call_records)
                != records_before + 1
            ):
                raise

            router_error = str(exc)

        # ----------------------------------------------------
        # C. ACCOUNT FOR THE LLM ROUTING CHARGE
        # ----------------------------------------------------

        if is_llm_router:

            records_after = len(
                policy.call_records
            )

            if records_after != records_before + 1:
                raise RuntimeError(
                    "LLM router must produce exactly "
                    "one call record per decision"
                )

            routing_record = policy.call_records[-1]

            # Verify the audit record belongs to the
            # actual task being evaluated.
            if (
                routing_record.episode_id
                != task.episode_id
                or routing_record.step_id
                != task.step_id
                or routing_record.round
                != task.round
            ):
                raise RuntimeError(
                    "Router call record does not match "
                    "the current task"
                )

            routing_cost = float(
                routing_record.routing_cost_usd
            )

            if (
                not math.isfinite(routing_cost)
                or routing_cost < 0
            ):
                raise ValueError(
                    "Invalid recorded routing cost"
                )

            # Always record the full completed call cost,
            # including calls with invalid responses.
            episode_routing_cost += routing_cost

            # If the routing call is affordable, deduct
            # the charge from the simulator's budget
            # before executing the selected model.
            if routing_cost <= task.budget_left_usd:

                simulator.charge_routing_cost(
                    routing_cost
                )

            else:

                # The call has already been made.
                #
                # It may be billable even though the
                # remaining simulated budget is smaller.
                #
                # Record the complete invoice separately,
                # terminate the episode and allow the
                # resulting balance to become negative.
                #
                # A future LIVE backend must prevent this
                # scenario with a pre-call budget guard.
                routing_budget_overrun = (
                    routing_cost
                )

            # An invalid audit record must not accompany
            # a supposedly valid routing decision.
            if (
                router_error is None
                and not routing_record.valid_choice
            ):
                raise RuntimeError(
                    "LLM router returned a strategy "
                    "for an invalid call record"
                )

            # Verify the selected model agrees with
            # the completed router call.
            if (
                router_error is None
                and strategy is not None
                and routing_record.selected_model_id
                != strategy.model_id
            ):
                raise RuntimeError(
                    "Router strategy and call record "
                    "contain different models"
                )

        # ----------------------------------------------------
        # D. VALIDATE THE RETURNED STRATEGY
        # ----------------------------------------------------

        if (
            strategy is not None
            and not isinstance(strategy, Strategy)
        ):
            raise TypeError(
                "Controller must return a Strategy"
            )

        if (
            strategy is None
            and router_error is None
        ):
            raise TypeError(
                "Controller must return a Strategy"
            )

        # ----------------------------------------------------
        # E. INITIALIZE EXECUTION RESULTS
        # ----------------------------------------------------

        selected_correct = None
        trace_item = None
        executed = False

        # ----------------------------------------------------
        # F. HANDLE ROUTING BUDGET OVERRUN
        # ----------------------------------------------------

        if routing_budget_overrun > 0:

            # No execution model is invoked.
            simulator.terminate()

            done = True
            status = "routing_budget_overrun"

        # ----------------------------------------------------
        # G. HANDLE INVALID LLM RESPONSE
        # ----------------------------------------------------

        elif router_error is not None:

            # Routing charges have already been applied.
            # Terminate without executing a model.
            simulator.terminate()

            done = True
            status = "router_invalid_response"

        # ----------------------------------------------------
        # H. NORMAL MODEL EXECUTION
        # ----------------------------------------------------

        else:

            # The controller has already selected a
            # strategy before ground truth is accessed.
            oracle_outcomes = (
                simulator.counterfactual()
            )

            selected_correct = (
                oracle_outcomes.get(
                    strategy.model_id
                )
            )

            transition = simulator.apply(
                strategy
            )

            trace_item = (
                transition.trace_item
            )

            executed = (
                trace_item is not None
            )

            # An unaffordable execution round is not
            # actually run or charged.
            if not executed:
                selected_correct = None

            else:
                execution_cost = float(
                    trace_item.cost_usd
                )

                episode_execution_cost += (
                    execution_cost
                )

            done = transition.done
            status = transition.status

        # ----------------------------------------------------
        # I. CALCULATE FINAL EPISODE METRICS
        # ----------------------------------------------------

        final_metrics = (
            simulator.final_metrics()
            if done
            else None
        )

        episode_reward = None
        episode_quality = None
        episode_cost = None
        episode_correct = None

        final_routing_cost = None
        final_execution_cost = None

        if final_metrics is not None:

            # Successful routing charges were already
            # added to the simulator's spent amount.
            #
            # An unaffordable routing invoice could not
            # be applied to the simulator, so we include
            # it separately in the final total.
            total_cost = (
                float(
                    final_metrics["cost_usd"]
                )
                + routing_budget_overrun
            )

            expected_total = (
                episode_routing_cost
                + episode_execution_cost
            )

            # The total should equal routing expenses
            # plus actual execution expenses.
            if not math.isclose(
                total_cost,
                expected_total,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Routing and execution costs "
                    "do not reconcile"
                )

            # Composite scoring includes BOTH the
            # router cost and selected execution cost.
            score = scorer.score(
                quality=final_metrics["quality"],
                cost_usd=total_cost,
                lam=lam,
                mu=0.0,
            )

            episode_reward = score.reward
            episode_quality = score.quality
            episode_cost = score.cost_usd

            episode_correct = (
                final_metrics["n_correct"]
            )

            final_routing_cost = (
                episode_routing_cost
            )

            final_execution_cost = (
                episode_execution_cost
            )

            # On an overrun, preserve the actual
            # negative balance instead of disguising it.
            budget_after = (
                float(
                    final_metrics["budget_left_usd"]
                )
                - routing_budget_overrun
            )

            spent_after = total_cost

        else:

            next_task = (
                simulator.current_task()
            )

            if next_task is None:
                raise RuntimeError(
                    "Nonterminal simulator state "
                    "has no next task"
                )

            budget_after = (
                next_task.budget_left_usd
            )

            spent_after = (
                next_task.spent_usd
            )

        # ----------------------------------------------------
        # J. ATTACH EVALUATION-ONLY DIFFICULTY METADATA
        # ----------------------------------------------------

        difficulty = None

        if difficulty_by_prompt is not None:

            difficulty = (
                difficulty_by_prompt.get(
                    task.step_emb_key
                )
            )

        # ----------------------------------------------------
        # K. EXPOSE CONFIDENCE ONLY WHEN DECLARED
        # ----------------------------------------------------

        # Existing baselines do not provide calibrated
        # confidence estimates.
        #
        # LLMRouter explicitly sets
        # provides_confidence = False.
        confidence = (
            float(
                strategy.confidence
            )
            if (
                strategy is not None
                and getattr(
                    policy,
                    "provides_confidence",
                    False,
                )
            )
            else None
        )

        # ----------------------------------------------------
        # L. BUILD PREDICTION RECORD
        # ----------------------------------------------------

        rows.append(
            {
                # --------------------------------------------
                # Original evaluation identifiers
                # --------------------------------------------

                "policy": policy_name,

                "split": split,

                "seed": seed,

                "episode_id": task.episode_id,

                "episode_type": skeleton.ep_type,

                "decision_index": decision_index,

                "step_id": task.step_id,

                "step_index": task.step_index,

                "round": task.round,

                "prompt_id": task.step_emb_key,

                "step_type": task.step_type,

                "difficulty": difficulty,

                # --------------------------------------------
                # Selected strategy
                # --------------------------------------------

                "model_id": (
                    strategy.model_id
                    if strategy is not None
                    else None
                ),

                "halt": (
                    strategy.halt
                    if strategy is not None
                    else None
                ),

                "confidence": confidence,

                # --------------------------------------------
                # Budget and spending
                # --------------------------------------------

                "budget_total_usd": (
                    task.budget_total_usd
                ),

                "budget_left_before_usd": (
                    task.budget_left_usd
                ),

                "budget_left_after_usd": (
                    budget_after
                ),

                "spent_after_usd": (
                    spent_after
                ),

                # --------------------------------------------
                # Model execution results
                # --------------------------------------------

                "executed": executed,

                "verifier_ok": (
                    trace_item.verifier_ok
                    if executed
                    else None
                ),

                "selected_model_correct": (
                    bool(selected_correct)
                    if selected_correct is not None
                    else None
                ),

                "execution_cost_usd": (
                    float(
                        trace_item.cost_usd
                    )
                    if executed
                    else 0.0
                ),

                # --------------------------------------------
                # C3-11: Routing cost accounting
                # --------------------------------------------

                "routing_cost_usd": (
                    routing_cost
                ),

                "routing_input_tokens": (
                    routing_record.input_tokens
                    if routing_record is not None
                    else None
                ),

                "routing_output_tokens": (
                    routing_record.output_tokens
                    if routing_record is not None
                    else None
                ),

                "routing_model": (
                    routing_record.router_model
                    if routing_record is not None
                    else None
                ),

                "routing_usage_source": (
                    routing_record.usage_source
                    if routing_record is not None
                    else None
                ),

                "routing_valid_choice": (
                    routing_record.valid_choice
                    if routing_record is not None
                    else None
                ),

                "routing_error": (
                    router_error
                ),

                # The complete invoice for a routing
                # call that exceeded remaining budget.
                "routing_budget_overrun_usd": (
                    routing_budget_overrun
                ),

                # Final episode-level cost breakdown.
                # Only populated on the terminal row.
                "episode_routing_cost_usd": (
                    final_routing_cost
                ),

                "episode_execution_cost_usd": (
                    final_execution_cost
                ),

                # --------------------------------------------
                # Latency
                # --------------------------------------------

                # Historical RouterBench does not have
                # measured end-to-end execution latency.
                "latency_ms": None,

                # --------------------------------------------
                # Episode status and final metrics
                # --------------------------------------------

                "status": status,

                "terminal": done,

                "expected_steps": (
                    task.expected_steps
                ),

                "episode_n_correct": (
                    episode_correct
                ),

                "episode_quality": (
                    episode_quality
                ),

                "episode_cost_usd": (
                    episode_cost
                ),

                "episode_reward": (
                    episode_reward
                ),
            }
        )

        if done:
            break

    # --------------------------------------------------------
    # ENSURE THE EPISODE REACHED A TERMINAL STATE
    # --------------------------------------------------------

    if not rows or not rows[-1]["terminal"]:
        raise RuntimeError(
            "Evaluation did not reach a terminal state"
        )

    return pd.DataFrame.from_records(
        rows
    )


# ============================================================
# 2. EVALUATE MULTIPLE POLICIES AND EPISODES
# ============================================================

def evaluate_policies(
    skeletons: Sequence[Skeleton],
    table: OutcomeTable,
    model_ids: Sequence[str],
    policies: Mapping[str, ControllerAPI],
    *,
    scorer: CompositeScorer,
    seed: int = 1234,
    flip_p: float = VERIFIER_FLIP_P,
    lam: float = 0.25,
    split: str = "test",
    difficulty_by_prompt: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Evaluate policies on the same episode skeletons.

    Each policy receives the same test episode inputs.

    A policy object may be reused across episodes.
    The evaluation function keeps episode-specific
    routing and execution expenses separate.
    """

    if not skeletons:
        raise ValueError(
            "At least one episode is required"
        )

    if not policies:
        raise ValueError(
            "At least one policy is required"
        )

    frames = []

    for skeleton in skeletons:

        for policy_name, policy in policies.items():

            frame = evaluate_episode(
                skeleton,
                table,
                model_ids,
                policy,
                policy_name=policy_name,
                scorer=scorer,
                seed=seed,
                flip_p=flip_p,
                lam=lam,
                split=split,
                difficulty_by_prompt=(
                    difficulty_by_prompt
                ),
            )

            frames.append(
                frame
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# 3. SAVE PREDICTIONS
# ============================================================

def save_predictions(
    predictions: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Persist prediction-level evaluation records.

    Required filename:
        results/<eval_id>/predictions.parquet

    Previous evaluations are never overwritten.

    C3-11 routing expenses are stored alongside the
    original episode and model-execution metrics.
    """

    destination = Path(
        path
    )

    if destination.name != "predictions.parquet":
        raise ValueError(
            "Output filename must be predictions.parquet"
        )

    if predictions.empty:
        raise ValueError(
            "Cannot save empty predictions"
        )

    if destination.exists():
        raise FileExistsError(
            f"Evaluation already exists: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        destination,
        index=False,
    )

    return destination


# ============================================================
# 4. LOAD SAVED PREDICTIONS
# ============================================================

def load_predictions(
    path: str | Path,
) -> pd.DataFrame:
    """Load persisted predictions for reporting.

    Research figures and summaries must be computed
    from these saved records.
    """

    return pd.read_parquet(
        path
    )
