
"""Pillar 3 C3-09: Reproducible evaluation harness.

Runs ControllerAPI policies through the existing simulator.

Predictions are recorded per attempted model execution.
Final episode metrics appear on the terminal prediction row.

All report metrics must later be computed from the
saved predictions.parquet, not from in-memory summaries.

Historical RouterBench costs are simulated. Live model
latency is unavailable and must not be fabricated.
"""

from __future__ import annotations

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
from measurement.scorer import CompositeScorer
from measurement.simulator import Simulator


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
    """Evaluate one episode and return its prediction records.

    Evaluation-only correctness is captured after decide()
    and is never passed to the Controller.

    Episode quality and reward are saved on the final row.
    """

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

    simulator = Simulator(
        skeleton,
        table,
        model_ids,
        lam=lam,
        flip_p=flip_p,
        seed=seed,
    )

    # Only the privileged episode oracle receives
    # the underlying environment containing ground truth.
    if isinstance(policy, EpisodeOracleBaseline):
        oracle_environment = Episode(
            skeleton,
            table,
            model_ids,
            flip_p=flip_p,
            seed=seed,
        )

        policy.bind_episode(oracle_environment)

    rows: list[dict] = []

    for decision_index in range(max_decisions):
        task = simulator.current_task()

        if task is None:
            break

        # Make the decision using the observable TaskRep.
        strategy = policy.decide(task)

        if not isinstance(strategy, Strategy):
            raise TypeError(
                "Controller must return a Strategy"
            )

        # Ground truth is captured only after the
        # Controller has already selected its model.
        # This value is evaluation-only.
        oracle_outcomes = simulator.counterfactual()

        selected_correct = oracle_outcomes.get(
            strategy.model_id
        )

        transition = simulator.apply(strategy)

        trace_item = transition.trace_item
        executed = trace_item is not None

        # An unaffordable attempted round does not
        # execute a model or produce a verifier result.
        if not executed:
            selected_correct = None

        final_metrics = (
            simulator.final_metrics()
            if transition.done
            else None
        )

        episode_reward = None
        episode_quality = None
        episode_cost = None
        episode_correct = None

        if final_metrics is not None:
            score = scorer.score(
                quality=final_metrics["quality"],
                cost_usd=final_metrics["cost_usd"],
                lam=lam,
                mu=0.0,
            )

            episode_reward = score.reward
            episode_quality = score.quality
            episode_cost = score.cost_usd
            episode_correct = final_metrics[
                "n_correct"
            ]

        if transition.task is not None:
            budget_after = (
                transition.task.budget_left_usd
            )

            spent_after = (
                transition.task.spent_usd
            )

        else:
            assert final_metrics is not None

            budget_after = final_metrics[
                "budget_left_usd"
            ]

            spent_after = final_metrics[
                "cost_usd"
            ]

        difficulty = None

        if difficulty_by_prompt is not None:
            difficulty = difficulty_by_prompt.get(
                task.step_emb_key
            )

        # Baselines do not supply meaningful calibrated
        # confidence, despite Strategy's default of 0.0.
        # A future learned-controller adapter can explicitly
        # declare that it provides confidence predictions.
        confidence = (
            float(strategy.confidence)
            if getattr(policy, "provides_confidence", False)
            else None
        )

        rows.append(
            {
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
                "model_id": strategy.model_id,
                "halt": strategy.halt,
                "confidence": confidence,
                "budget_total_usd": (
                    task.budget_total_usd
                ),
                "budget_left_before_usd": (
                    task.budget_left_usd
                ),
                "budget_left_after_usd": (
                    budget_after
                ),
                "spent_after_usd": spent_after,
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
                    trace_item.cost_usd
                    if executed
                    else 0.0
                ),
                # No measured end-to-end model latency
                # exists in historical RouterBench data.
                "latency_ms": None,
                "status": transition.status,
                "terminal": transition.done,
                "expected_steps": task.expected_steps,
                "episode_n_correct": episode_correct,
                "episode_quality": episode_quality,
                "episode_cost_usd": episode_cost,
                "episode_reward": episode_reward,
            }
        )

        if transition.done:
            break

    if not rows or not rows[-1]["terminal"]:
        raise RuntimeError(
            "Evaluation did not reach a terminal state"
        )

    return pd.DataFrame.from_records(rows)


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
    """Evaluate all supplied policies on the same episodes."""

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

            frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def save_predictions(
    predictions: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save prediction records without overwriting old runs.

    Standard output:
        results/<eval_id>/predictions.parquet
    """

    destination = Path(path)

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


def load_predictions(
    path: str | Path,
) -> pd.DataFrame:
    """Load the persisted predictions for reporting."""

    return pd.read_parquet(path)
