
"""C3-06: Tests for the cost-weighted episode oracle.

Compare optimized oracle planning against exhaustive search
on small episodes with known outcomes.
"""

import math

import numpy as np
import pytest

from data.env import Episode
from data.episodes import Skeleton
from data.routerbench import OutcomeTable
from measurement.oracle import (
    exact_episode_oracle,
    weighted_episode_oracle,
)


MODELS = ["m0", "m1", "m2", "m3"]

ROWS = (
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 0, 1),
)


def make_episode(
    rows=ROWS,
    ep_type="random",
    budget=100.0,
    flip_p=0.0,
):
    correct = np.asarray(rows, dtype=bool)
    n = len(rows)

    table = OutcomeTable(
        prompt_ids=np.asarray(
            [f"p{i}" for i in range(n)]
        ),
        step_type=np.zeros(n, dtype=int),
        correct=correct,
        cost=np.tile(
            [1.0, 2.0, 4.0, 8.0],
            (n, 1),
        ),
    )

    skeleton = Skeleton(
        episode_id="weighted-test",
        ep_type=ep_type,
        prompt_ids=tuple(table.prompt_ids),
        budget_usd=budget,
    )

    return Episode(
        skeleton,
        table,
        MODELS,
        flip_p=flip_p,
        seed=1234,
    )


def exhaustive_objective(episode, state, lam):
    """Enumerate all legal action sequences without pruning."""

    if state.done:
        return (
            episode.quality(state) - lam * state.spent,
            -state.spent,
            state.n_correct,
        )

    return max(
        exhaustive_objective(
            episode,
            episode.apply(state, action),
            lam,
        )
        for action in episode.legal_actions(state)
    )


def replay_plan(episode, state, actions):
    """Verify that oracle actions are executable."""

    for action in actions:
        state = episode.apply(state, action)

    assert state.done
    return state


def test_zero_lambda_matches_quality_first_oracle():
    episode = make_episode()

    quality_first = exact_episode_oracle(episode)
    weighted = weighted_episode_oracle(
        episode, lam=0.0
    )

    assert weighted.correct_steps == quality_first.correct_steps
    assert weighted.quality == pytest.approx(
        quality_first.quality
    )
    assert weighted.cost_usd == pytest.approx(
        quality_first.cost_usd
    )
    assert weighted.objective == pytest.approx(
        quality_first.quality
    )


def test_small_lambda_favors_full_quality():
    episode = make_episode()

    result = weighted_episode_oracle(
        episode, lam=0.005
    )

    assert result.correct_steps == 3
    assert result.quality == pytest.approx(1.0)
    assert result.cost_usd == pytest.approx(11.0)


def test_high_lambda_favors_lower_cost():
    episode = make_episode()

    result = weighted_episode_oracle(
        episode, lam=0.4
    )

    assert result.correct_steps == 1
    assert result.quality == pytest.approx(1 / 3)
    assert result.cost_usd == pytest.approx(3.0)
    assert result.objective == pytest.approx(
        1 / 3 - 0.4 * 3
    )


def test_objective_formula_is_correct():
    episode = make_episode()

    result = weighted_episode_oracle(
        episode, lam=0.05
    )

    expected = (
        result.quality
        - 0.05 * result.cost_usd
    )

    assert result.objective == pytest.approx(expected)


def test_weighted_plan_replays_correctly():
    episode = make_episode()

    result = weighted_episode_oracle(
        episode, lam=0.1
    )

    final_state = replay_plan(
        episode,
        episode.initial_state(),
        result.actions,
    )

    assert final_state.n_correct == result.correct_steps
    assert final_state.spent == pytest.approx(
        result.cost_usd
    )
    assert episode.quality(final_state) == pytest.approx(
        result.quality
    )


@pytest.mark.parametrize(
    "episode_type",
    ["random", "correlated", "propagation", "retry"],
)
def test_supports_all_episode_types(episode_type):
    episode = make_episode(
        rows=((1, 1, 1, 1),) * 3,
        ep_type=episode_type,
    )

    result = weighted_episode_oracle(
        episode, lam=0.0
    )

    assert result.correct_steps == 3
    assert result.quality == pytest.approx(1.0)
    assert result.cost_usd == pytest.approx(3.0)
    assert result.status == "completed"


def test_weighted_oracle_from_mid_step():
    episode = make_episode()

    state = episode.play(
        episode.initial_state(),
        0,
    )

    result = weighted_episode_oracle(
        episode,
        lam=0.0,
        state=state,
    )

    assert result.correct_steps == 3
    assert result.quality == pytest.approx(1.0)
    assert result.cost_usd == pytest.approx(12.0)

    final_state = replay_plan(
        episode, state, result.actions
    )

    assert final_state.n_correct == 3


def test_weighted_oracle_is_deterministic():
    episode = make_episode()

    first = weighted_episode_oracle(
        episode, lam=0.05
    )
    second = weighted_episode_oracle(
        episode, lam=0.05
    )

    assert first == second


def test_budget_exhaustion_is_handled():
    episode = make_episode(budget=0.5)

    result = weighted_episode_oracle(
        episode, lam=0.2
    )

    assert result.status == "budget_exhausted"
    assert result.correct_steps == 0
    assert result.cost_usd == 0.0
    assert result.objective == 0.0


@pytest.mark.parametrize(
    "invalid_lambda",
    [-0.1, float("nan"), float("inf"), True, None, "0.2"],
)
def test_invalid_lambda_is_rejected(invalid_lambda):
    episode = make_episode()

    with pytest.raises(
        ValueError,
        match="Lambda",
    ):
        weighted_episode_oracle(
            episode,
            lam=invalid_lambda,
        )


def test_completed_episode_is_rejected():
    episode = make_episode()

    state = episode.initial_state()

    for model in (1, 0, 3):
        state = episode.end_step(
            episode.play(state, model)
        )

    assert state.done

    with pytest.raises(
        ValueError,
        match="active episode",
    ):
        weighted_episode_oracle(
            episode,
            lam=0.1,
            state=state,
        )


@pytest.mark.parametrize(
    "episode_type,budget,lam",
    [
        ("random", 100.0, 0.0),
        ("random", 100.0, 0.4),
        ("correlated", 9.0, 0.03),
        ("propagation", 9.0, 0.1),
    ],
)
def test_weighted_oracle_matches_exhaustive_search(
    episode_type,
    budget,
    lam,
):
    episode = make_episode(
        ep_type=episode_type,
        budget=budget,
    )

    result = weighted_episode_oracle(
        episode,
        lam=lam,
    )

    brute = exhaustive_objective(
        episode,
        episode.initial_state(),
        lam,
    )

    assert result.objective == pytest.approx(
        brute[0]
    )
    assert result.cost_usd == pytest.approx(
        -brute[1]
    )
    assert result.correct_steps == brute[2]


def test_retry_oracle_matches_exhaustive_search():
    # Two prompts keep the exhaustive search small.
    episode = make_episode(
        rows=(
            (1, 1, 1, 1),
            (0, 1, 1, 1),
        ),
        ep_type="retry",
        budget=7.0,
        flip_p=1.0,
    )

    lam = 0.1

    result = weighted_episode_oracle(
        episode,
        lam=lam,
    )

    brute = exhaustive_objective(
        episode,
        episode.initial_state(),
        lam,
    )

    assert result.objective == pytest.approx(
        brute[0]
    )
    assert result.cost_usd == pytest.approx(
        -brute[1]
    )


def test_ground_truth_not_added_to_controller_contract():
    """Oracle results must remain separate from TaskRep."""

    from controller.contracts import TaskRep

    assert "correct_steps" not in (
        TaskRep.__dataclass_fields__
    )
    assert "objective" not in (
        TaskRep.__dataclass_fields__
    )


def test_objective_is_finite():
    episode = make_episode()

    result = weighted_episode_oracle(
        episode,
        lam=0.2,
    )

    assert math.isfinite(result.objective)
    assert math.isfinite(result.cost_usd)
