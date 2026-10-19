
"""Pillar 3 C3-09: Evaluation harness tests.

Verify prediction records, policy execution, episode metrics,
ground-truth separation, reproducibility, and Parquet storage.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from controller.contracts import Strategy
from measurement.baselines import (
    AlwaysCheapest,
    AlwaysStrongest,
    RandomBaseline,
    EpisodeOracleBaseline,
)
from measurement.evaluate import (
    evaluate_episode,
    evaluate_policies,
    save_predictions,
    load_predictions,
)
from measurement.oracle import exact_episode_oracle
from measurement.scorer import CompositeScorer

from tests.test_measurement_baselines import (
    IDS,
    make_environment,
    make_pool,
)


@pytest.fixture
def scorer():
    return CompositeScorer.from_yaml("conf/score.yaml")


def evaluate_cheapest(scorer, **kwargs):
    episode, _ = make_environment()

    return evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        AlwaysCheapest(make_pool()),
        policy_name="always-cheapest",
        scorer=scorer,
        flip_p=0.0,
        **kwargs,
    )


def test_evaluation_creates_prediction_records(scorer):
    frame = evaluate_cheapest(scorer)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert frame["decision_index"].tolist() == [0, 1, 2]


def test_prediction_schema_contains_required_fields(scorer):
    frame = evaluate_cheapest(scorer)

    required = {
        "policy",
        "split",
        "seed",
        "episode_id",
        "episode_type",
        "decision_index",
        "step_id",
        "step_index",
        "round",
        "prompt_id",
        "step_type",
        "model_id",
        "halt",
        "confidence",
        "executed",
        "verifier_ok",
        "selected_model_correct",
        "execution_cost_usd",
        "budget_total_usd",
        "budget_left_before_usd",
        "budget_left_after_usd",
        "spent_after_usd",
        "terminal",
        "episode_n_correct",
        "episode_quality",
        "episode_cost_usd",
        "episode_reward",
        "latency_ms",
    }

    assert required.issubset(frame.columns)


def test_policy_and_episode_identifiers(scorer):
    frame = evaluate_cheapest(scorer)

    assert set(frame["policy"]) == {"always-cheapest"}
    assert set(frame["episode_id"]) == {"baseline-demo"}
    assert set(frame["episode_type"]) == {"random"}
    assert set(frame["split"]) == {"test"}
    assert set(frame["seed"]) == {1234}


def test_model_decisions_are_recorded(scorer):
    frame = evaluate_cheapest(scorer)

    assert frame["model_id"].tolist() == [
        "m0", "m0", "m0"
    ]

    assert frame["halt"].tolist() == [
        "stop", "stop", "stop"
    ]


def test_single_terminal_prediction_row(scorer):
    frame = evaluate_cheapest(scorer)

    assert int(frame["terminal"].sum()) == 1
    assert bool(frame.iloc[-1]["terminal"]) is True
    assert frame.iloc[-1]["status"] == "completed"


def test_final_metrics_appear_only_on_terminal_row(scorer):
    frame = evaluate_cheapest(scorer)

    assert frame.iloc[:-1]["episode_quality"].isna().all()
    assert frame.iloc[:-1]["episode_reward"].isna().all()
    assert frame.iloc[:-1]["episode_cost_usd"].isna().all()

    assert pd.notna(frame.iloc[-1]["episode_quality"])
    assert pd.notna(frame.iloc[-1]["episode_reward"])
    assert pd.notna(frame.iloc[-1]["episode_cost_usd"])


def test_episode_quality_and_cost_are_correct(scorer):
    frame = evaluate_cheapest(scorer)
    final = frame.iloc[-1]

    assert final["episode_n_correct"] == 1
    assert final["expected_steps"] == 3
    assert final["episode_quality"] == pytest.approx(1 / 3)
    assert final["episode_cost_usd"] == pytest.approx(3.0)


def test_composite_reward_matches_scorer(scorer):
    frame = evaluate_cheapest(scorer)
    final = frame.iloc[-1]

    expected = scorer.score(
        quality=1 / 3,
        cost_usd=3.0,
        lam=0.25,
        mu=0.0,
    )

    assert final["episode_reward"] == pytest.approx(
        expected.reward
    )


def test_ground_truth_is_recorded_after_decisions(scorer):
    frame = evaluate_cheapest(scorer)

    assert frame["selected_model_correct"].tolist() == [
        False, True, False
    ]

    assert frame["executed"].all()


def test_missing_confidence_is_not_fabricated(scorer):
    frame = evaluate_cheapest(scorer)

    assert frame["confidence"].isna().all()


def test_missing_latency_is_not_fabricated(scorer):
    frame = evaluate_cheapest(scorer)

    assert frame["latency_ms"].isna().all()


def test_difficulty_metadata_is_recorded(scorer):
    frame = evaluate_cheapest(
        scorer,
        difficulty_by_prompt={
            "p0": 3,
            "p1": 1,
            "p2": 4,
        },
    )

    assert frame["difficulty"].tolist() == [3, 1, 4]


def test_budget_decreases_after_execution(scorer):
    frame = evaluate_cheapest(scorer)

    assert frame["budget_left_before_usd"].tolist() == [
        100.0, 99.0, 98.0
    ]

    assert frame["budget_left_after_usd"].tolist() == [
        99.0, 98.0, 97.0
    ]


def test_execution_costs_sum_to_episode_cost(scorer):
    frame = evaluate_cheapest(scorer)

    total = frame["execution_cost_usd"].sum()
    final = frame.iloc[-1]["episode_cost_usd"]

    assert total == pytest.approx(final)


def test_strongest_baseline_runs_through_harness(scorer):
    episode, _ = make_environment()

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        AlwaysStrongest(make_pool()),
        policy_name="always-strongest",
        scorer=scorer,
        flip_p=0.0,
    )

    final = frame.iloc[-1]

    assert len(frame) == 3
    assert final["episode_quality"] == 1.0
    assert final["episode_cost_usd"] == pytest.approx(24.0)


def test_random_baseline_is_reproducible(scorer):
    episode, _ = make_environment()

    def run():
        return evaluate_episode(
            episode.skeleton,
            episode.table,
            episode.model_ids,
            RandomBaseline(make_pool(), seed=1234),
            policy_name="random",
            scorer=scorer,
            flip_p=0.0,
        )

    pd.testing.assert_frame_equal(run(), run())


def test_oracle_baseline_matches_exact_oracle(scorer):
    episode, _ = make_environment()

    expected = exact_episode_oracle(episode)

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        EpisodeOracleBaseline(IDS),
        policy_name="episode-oracle",
        scorer=scorer,
        flip_p=0.0,
    )

    final = frame.iloc[-1]

    assert final["episode_n_correct"] == expected.correct_steps
    assert final["episode_quality"] == pytest.approx(
        expected.quality
    )
    assert final["episode_cost_usd"] == pytest.approx(
        expected.cost_usd
    )


def test_unaffordable_round_is_recorded(scorer):
    episode, _ = make_environment(budget=0.5)

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        AlwaysCheapest(make_pool()),
        policy_name="always-cheapest",
        scorer=scorer,
        flip_p=0.0,
    )

    assert len(frame) == 1

    row = frame.iloc[0]

    assert bool(row["executed"]) is False
    assert bool(row["terminal"]) is True
    assert row["status"] == "budget_exhausted"
    assert row["execution_cost_usd"] == 0.0
    assert pd.isna(row["selected_model_correct"])
    assert pd.isna(row["verifier_ok"])

    assert row["episode_quality"] == 0.0
    assert row["episode_cost_usd"] == 0.0


def test_retry_decisions_are_recorded(scorer):
    """Verify that retry episodes repeat a rejected prompt.

    In the existing environment, a retry advances the
    execution index but repeats the same prompt slot.

    Therefore, an episode with three execution steps can
    still produce three prediction records, even when one
    of the prompts is retried.
    """

    episode, _ = make_environment(
        rows=((1, 1, 1, 1),) * 3,
        ep_type="retry",
        flip_p=1.0,
    )

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        AlwaysCheapest(make_pool()),
        policy_name="always-cheapest",
        scorer=scorer,
        flip_p=1.0,
    )

    # Three execution slots are used.
    assert len(frame) == 3

    # The rejected first prompt is repeated.
    assert frame["prompt_id"].tolist() == [
        "p0", "p0", "p1"
    ]

    # The verifier rejects every execution
    # because flip_p=1.0 flips all true outcomes.
    assert frame["verifier_ok"].tolist() == [
        False, False, False
    ]

    # The episode ends with exactly one terminal row.
    assert int(frame["terminal"].sum()) == 1
    assert bool(frame.iloc[-1]["terminal"]) is True


def test_escalation_decisions_are_recorded(scorer):
    episode, _ = make_environment()

    class EscalatingPolicy:
        def decide(self, task):
            if task.step_index == 0 and task.round == 0:
                model, halt = "m0", "escalate"
            elif task.step_index == 0 and task.round == 1:
                model, halt = "m1", "stop"
            else:
                model, halt = "m0", "stop"

            return Strategy(
                step_id=task.step_id,
                round=task.round,
                model_id=model,
                halt=halt,
            )

    frame = evaluate_episode(
        episode.skeleton,
        episode.table,
        episode.model_ids,
        EscalatingPolicy(),
        policy_name="escalating",
        scorer=scorer,
        flip_p=0.0,
    )

    assert len(frame) == 4
    assert frame["round"].tolist() == [0, 1, 0, 0]
    assert frame["halt"].tolist() == [
        "escalate", "stop", "stop", "stop"
    ]

    assert frame.iloc[-1]["episode_quality"] == pytest.approx(
        2 / 3
    )


def test_evaluate_multiple_policies_and_episodes(scorer):
    first, _ = make_environment(
        episode_id="episode-one"
    )

    second, _ = make_environment(
        episode_id="episode-two"
    )

    policies = {
        "cheap": AlwaysCheapest(make_pool()),
        "strong": AlwaysStrongest(make_pool()),
    }

    frame = evaluate_policies(
        [first.skeleton, second.skeleton],
        first.table,
        IDS,
        policies,
        scorer=scorer,
        flip_p=0.0,
    )

    terminal = frame.loc[frame["terminal"]]

    assert len(terminal) == 4
    assert set(terminal["policy"]) == {"cheap", "strong"}
    assert set(terminal["episode_id"]) == {
        "episode-one", "episode-two"
    }


def test_evaluate_policies_rejects_empty_episodes(scorer):
    episode, _ = make_environment()

    with pytest.raises(ValueError, match="episode"):
        evaluate_policies(
            [],
            episode.table,
            IDS,
            {"cheap": AlwaysCheapest(make_pool())},
            scorer=scorer,
        )


def test_evaluate_policies_rejects_empty_policies(scorer):
    episode, _ = make_environment()

    with pytest.raises(ValueError, match="policy"):
        evaluate_policies(
            [episode.skeleton],
            episode.table,
            IDS,
            {},
            scorer=scorer,
        )


def test_empty_policy_name_is_rejected(scorer):
    episode, _ = make_environment()

    with pytest.raises(ValueError, match="Policy name"):
        evaluate_episode(
            episode.skeleton,
            episode.table,
            IDS,
            AlwaysCheapest(make_pool()),
            policy_name="",
            scorer=scorer,
        )


def test_invalid_policy_is_rejected(scorer):
    episode, _ = make_environment()

    with pytest.raises(TypeError, match="ControllerAPI"):
        evaluate_episode(
            episode.skeleton,
            episode.table,
            IDS,
            object(),
            policy_name="invalid",
            scorer=scorer,
        )


def test_decision_limit_is_enforced(scorer):
    with pytest.raises(RuntimeError, match="terminal"):
        evaluate_cheapest(
            scorer,
            max_decisions=2,
        )


def test_save_and_load_predictions(tmp_path, scorer):
    frame = evaluate_cheapest(scorer)

    path = (
        tmp_path
        / "eval-test"
        / "predictions.parquet"
    )

    saved = save_predictions(frame, path)

    assert saved == path
    assert path.exists()

    loaded = load_predictions(path)

    pd.testing.assert_frame_equal(
        loaded,
        frame,
        check_dtype=False,
    )


def test_existing_prediction_file_is_not_overwritten(
    tmp_path, scorer
):
    frame = evaluate_cheapest(scorer)
    path = tmp_path / "predictions.parquet"

    save_predictions(frame, path)

    with pytest.raises(FileExistsError):
        save_predictions(frame, path)


def test_invalid_prediction_filename_is_rejected(
    tmp_path, scorer
):
    frame = evaluate_cheapest(scorer)

    with pytest.raises(
        ValueError, match="predictions.parquet"
    ):
        save_predictions(
            frame,
            tmp_path / "wrong.parquet",
        )


def test_empty_predictions_cannot_be_saved(tmp_path):
    path = tmp_path / "predictions.parquet"

    with pytest.raises(ValueError, match="empty"):
        save_predictions(
            pd.DataFrame(),
            path,
        )


def test_scores_are_finite(scorer):
    frame = evaluate_cheapest(scorer)
    reward = frame.iloc[-1]["episode_reward"]

    assert math.isfinite(reward)
