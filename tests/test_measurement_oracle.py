
"""Tests for Pillar 3 C3-06: per-step and exact episode oracles."""

import numpy as np
import pytest

from data.env import Episode
from data.episodes import Skeleton
from data.routerbench import OutcomeTable
from measurement.oracle import (
    per_step_oracle,
    escalation_oracle,
    exact_episode_oracle,
)


MODELS = ["m0", "m1", "m2", "m3"]

DEFAULT_ROWS = (
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 0, 1),
)


def make_episode(
    rows=DEFAULT_ROWS,
    ep_type="random",
    budget=100.0,
    flip_p=0.0,
):
    """Create a small environment with known outcomes."""

    correct = np.asarray(rows, dtype=bool)
    n = len(rows)

    costs = np.tile(
        [1.0, 2.0, 4.0, 8.0],
        (n, 1),
    )

    prompt_ids = np.asarray(
        [f"p{i}" for i in range(n)]
    )

    table = OutcomeTable(
        prompt_ids=prompt_ids,
        step_type=np.zeros(n, dtype=int),
        correct=correct,
        cost=costs,
    )

    skeleton = Skeleton(
        episode_id="oracle-test",
        ep_type=ep_type,
        prompt_ids=tuple(prompt_ids),
        budget_usd=budget,
    )

    return Episode(
        skeleton=skeleton,
        table=table,
        model_ids=MODELS,
        flip_p=flip_p,
        seed=1234,
    )


def test_per_step_oracle_selects_cheapest_success():
    episode = make_episode()

    result = per_step_oracle(episode)

    assert result.model_id == "m1"
    assert result.model_tier == 1
    assert result.difficulty == 1
    assert result.affordable is True

    assert result.succeeds == (
        False, True, True, True
    )


def test_per_step_oracle_falls_back_to_cheapest():
    episode = make_episode(
        rows=((0, 0, 0, 0),) * 3
    )

    result = per_step_oracle(episode)

    assert result.model_id == "m0"
    assert result.model_tier == 0
    assert result.difficulty == 4
    assert result.succeeds == (False,) * 4


def test_per_step_reports_affordability_separately():
    episode = make_episode(budget=0.5)

    result = per_step_oracle(episode)

    assert result.model_id == "m1"
    assert result.affordable is False


def test_per_step_oracle_accounts_for_propagation():
    episode = make_episode(
        ep_type="propagation",
        rows=(
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        ),
    )

    state = episode.initial_state()
    state = episode.end_step(
        episode.play(state, 0)
    )

    result = per_step_oracle(episode, state)

    assert result.model_tier == 1
    assert result.difficulty == 1
    assert result.succeeds == (
        False, True, True, True
    )


def test_per_step_requires_new_step():
    episode = make_episode()

    state = episode.play(
        episode.initial_state(), 0
    )

    with pytest.raises(
        ValueError, match="new step"
    ):
        per_step_oracle(episode, state)


def test_per_step_rejects_completed_episode():
    episode = make_episode()

    state = episode.initial_state()

    for model in (1, 0, 3):
        state = episode.end_step(
            episode.play(state, model)
        )

    assert state.done

    with pytest.raises(
        ValueError, match="active episode"
    ):
        per_step_oracle(episode, state)


def test_escalation_oracle_recommends_successful_model():
    episode = make_episode(budget=5.0)

    state = episode.play(
        episode.initial_state(), 0
    )

    assert escalation_oracle(episode, state) is True


def test_escalation_oracle_respects_budget():
    episode = make_episode(budget=2.0)

    state = episode.play(
        episode.initial_state(), 0
    )

    assert escalation_oracle(episode, state) is False


def test_escalation_oracle_does_not_escalate_correct_answer():
    episode = make_episode()

    state = episode.play(
        episode.initial_state(), 1
    )

    assert escalation_oracle(episode, state) is False


def test_escalation_requires_first_completed_round():
    episode = make_episode()

    with pytest.raises(
        ValueError, match="exactly one played round"
    ):
        escalation_oracle(
            episode,
            episode.initial_state(),
        )


def test_escalation_rejects_second_round():
    episode = make_episode()

    state = episode.initial_state()
    state = episode.play(state, 0)
    state = episode.play(state, 2)

    with pytest.raises(
        ValueError, match="exactly one played round"
    ):
        escalation_oracle(episode, state)


def test_exact_oracle_finds_full_quality_plan():
    episode = make_episode()

    result = exact_episode_oracle(episode)

    assert result.episode_id == "oracle-test"
    assert result.correct_steps == 3
    assert result.expected_steps == 3
    assert result.quality == pytest.approx(1.0)
    assert result.cost_usd == pytest.approx(11.0)
    assert result.budget_left_usd == pytest.approx(89.0)
    assert result.status == "completed"

    assert result.actions == (
        ("play", 1),
        ("stop", None),
        ("play", 0),
        ("stop", None),
        ("play", 3),
        ("stop", None),
    )


def test_exact_oracle_optimizes_under_budget():
    episode = make_episode(
        rows=(
            (0, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
        ),
        budget=9.0,
    )

    result = exact_episode_oracle(episode)

    assert result.correct_steps == 2
    assert result.quality == pytest.approx(2 / 3)
    assert result.cost_usd == pytest.approx(9.0)
    assert result.status == "completed"


def test_exact_oracle_handles_budget_exhaustion():
    episode = make_episode(budget=0.5)

    result = exact_episode_oracle(episode)

    assert result.correct_steps == 0
    assert result.quality == 0.0
    assert result.cost_usd == 0.0
    assert result.status == "budget_exhausted"


def test_exact_oracle_can_continue_mid_step():
    episode = make_episode()

    state = episode.play(
        episode.initial_state(), 0
    )

    result = exact_episode_oracle(episode, state)

    assert result.correct_steps == 3
    assert result.quality == 1.0
    assert result.cost_usd == pytest.approx(12.0)
    assert result.actions[0] == ("play", 1)


def test_exact_oracle_does_not_mutate_input_state():
    episode = make_episode()

    initial = episode.initial_state()

    first = exact_episode_oracle(
        episode, initial
    )

    second = exact_episode_oracle(
        episode, initial
    )

    assert first == second
    assert initial.step_index == 0
    assert initial.spent == 0.0
    assert initial.trace == ()


def test_exact_oracle_rejects_completed_episode():
    episode = make_episode()

    state = episode.initial_state()

    for model in (1, 0, 3):
        state = episode.end_step(
            episode.play(state, model)
        )

    with pytest.raises(
        ValueError, match="active episode"
    ):
        exact_episode_oracle(episode, state)


@pytest.mark.parametrize(
    "episode_type",
    ["correlated", "retry", "propagation", "random"],
)
def test_exact_oracle_supports_all_episode_types(
    episode_type,
):
    episode = make_episode(
        rows=((1, 1, 1, 1),) * 3,
        ep_type=episode_type,
    )

    result = exact_episode_oracle(episode)

    assert result.status == "completed"
    assert result.correct_steps == 3
    assert result.quality == 1.0
    assert result.cost_usd == pytest.approx(3.0)


def test_exact_oracle_actions_replay_to_terminal_state():
    episode = make_episode(
        ep_type="propagation",
    )

    result = exact_episode_oracle(episode)

    state = episode.initial_state()

    for action in result.actions:
        state = episode.apply(state, action)

    assert state.done
    assert state.n_correct == result.correct_steps
    assert state.spent == pytest.approx(result.cost_usd)
    assert episode.quality(state) == pytest.approx(
        result.quality
    )
