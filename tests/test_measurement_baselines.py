
"""Pillar 3 C3-08: Tests for simple baseline controllers.

Checks ControllerAPI compatibility, model selection,
reproducible randomness, oracle planning, and integration
with the existing Measurement simulator.
"""

from dataclasses import replace

import numpy as np
import pytest

from controller.contracts import ModelCard, Strategy
from data.env import Episode
from data.episodes import Skeleton
from data.routerbench import OutcomeTable
from measurement.baselines import (
    ControllerAPI,
    AlwaysCheapest,
    AlwaysStrongest,
    RandomBaseline,
    EpisodeOracleBaseline,
)
from measurement.oracle import exact_episode_oracle
from measurement.simulator import Simulator


IDS = ["m0", "m1", "m2", "m3"]

ROWS = (
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 0, 1),
)


def make_pool():
    """Create synthetic model cards with increasing costs."""

    return [
        ModelCard(
            model_id=model_id,
            provider="test",
            tier=index,
            mean_cost_usd=float(cost),
            price_source="synthetic",
            price_date="2026-10-18",
            capability=[0.0] * 6,
        )
        for index, (model_id, cost) in enumerate(
            zip(IDS, [1.0, 2.0, 4.0, 8.0])
        )
    ]


def make_environment(
    rows=ROWS,
    ep_type="random",
    budget=100.0,
    episode_id="baseline-demo",
    flip_p=0.0,
    seed=1234,
):
    """Create matching oracle and simulator environments."""

    correct = np.asarray(rows, dtype=bool)
    n = len(correct)

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
        episode_id=episode_id,
        ep_type=ep_type,
        prompt_ids=tuple(table.prompt_ids),
        budget_usd=budget,
    )

    episode = Episode(
        skeleton,
        table,
        IDS,
        flip_p=flip_p,
        seed=seed,
    )

    simulator = Simulator(
        skeleton,
        table,
        IDS,
        flip_p=flip_p,
        seed=seed,
    )

    return episode, simulator


def run_policy(policy, simulator):
    """Run a baseline through the Controller-facing simulator."""

    transitions = []

    for _ in range(30):
        task = simulator.current_task()

        if task is None:
            return transitions

        strategy = policy.decide(task)

        assert isinstance(strategy, Strategy)
        assert strategy.step_id == task.step_id
        assert strategy.round == task.round

        transition = simulator.apply(strategy)
        transitions.append(transition)

        if transition.done:
            return transitions

    raise AssertionError(
        "Baseline did not finish within 30 decisions"
    )


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda pool: AlwaysCheapest(pool),
        lambda pool: AlwaysStrongest(pool),
        lambda pool: RandomBaseline(pool),
    ],
)
def test_baseline_implements_controller_api(policy_factory):
    policy = policy_factory(make_pool())

    assert isinstance(policy, ControllerAPI)


def test_episode_oracle_implements_controller_api():
    policy = EpisodeOracleBaseline(IDS)

    assert isinstance(policy, ControllerAPI)


def test_always_cheapest_selects_lowest_cost_model():
    _, sim = make_environment()

    policy = AlwaysCheapest(make_pool())
    strategy = policy.decide(sim.current_task())

    assert strategy.model_id == "m0"
    assert strategy.halt == "stop"
    assert strategy.round == 0


def test_always_strongest_selects_highest_cost_model():
    _, sim = make_environment()

    policy = AlwaysStrongest(make_pool())
    strategy = policy.decide(sim.current_task())

    assert strategy.model_id == "m3"
    assert strategy.halt == "stop"
    assert strategy.round == 0


def test_model_selection_does_not_depend_on_card_order():
    pool = list(reversed(make_pool()))

    assert AlwaysCheapest(pool).model_id == "m0"
    assert AlwaysStrongest(pool).model_id == "m3"


def test_cheapest_policy_runs_through_simulator():
    _, sim = make_environment()

    transitions = run_policy(
        AlwaysCheapest(make_pool()),
        sim,
    )

    metrics = sim.final_metrics()

    assert len(transitions) == 3
    assert transitions[-1].done
    assert metrics["n_correct"] == 1
    assert metrics["quality"] == pytest.approx(1 / 3)
    assert metrics["cost_usd"] == pytest.approx(3.0)


def test_strongest_policy_runs_through_simulator():
    _, sim = make_environment()

    transitions = run_policy(
        AlwaysStrongest(make_pool()),
        sim,
    )

    metrics = sim.final_metrics()

    assert len(transitions) == 3
    assert transitions[-1].done
    assert metrics["n_correct"] == 3
    assert metrics["quality"] == 1.0
    assert metrics["cost_usd"] == pytest.approx(24.0)


def test_random_policy_returns_valid_model():
    _, sim = make_environment()

    strategy = RandomBaseline(
        make_pool(),
        seed=1234,
    ).decide(sim.current_task())

    assert strategy.model_id in IDS
    assert strategy.halt == "stop"


def test_random_policy_is_reproducible():
    _, sim = make_environment()

    first = RandomBaseline(make_pool(), seed=1234)
    second = RandomBaseline(make_pool(), seed=1234)

    task = sim.current_task()

    assert first.decide(task) == second.decide(task)


def test_random_policy_is_stable_for_repeated_task():
    _, sim = make_environment()

    policy = RandomBaseline(make_pool(), seed=77)
    task = sim.current_task()

    choices = [
        policy.decide(task).model_id
        for _ in range(10)
    ]

    assert len(set(choices)) == 1


def test_random_policy_runs_through_simulator():
    _, sim = make_environment()

    transitions = run_policy(
        RandomBaseline(make_pool(), seed=1234),
        sim,
    )

    assert transitions[-1].done
    assert sim.final_metrics()["expected_steps"] == 3


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda pool: AlwaysCheapest(pool),
        lambda pool: AlwaysStrongest(pool),
        lambda pool: RandomBaseline(pool),
    ],
)
def test_single_round_policies_reject_escalation_round(
    policy_factory,
):
    _, sim = make_environment()

    task = replace(sim.current_task(), round=1)
    policy = policy_factory(make_pool())

    with pytest.raises(ValueError, match="round 0"):
        policy.decide(task)


@pytest.mark.parametrize(
    "policy_class",
    [AlwaysCheapest, AlwaysStrongest, RandomBaseline],
)
def test_empty_model_pool_is_rejected(policy_class):
    with pytest.raises(ValueError, match="empty"):
        policy_class([])


def test_duplicate_model_ids_are_rejected():
    pool = make_pool()
    pool.append(pool[0])

    with pytest.raises(ValueError, match="unique"):
        AlwaysCheapest(pool)


@pytest.mark.parametrize(
    "invalid_cost",
    [-1.0, float("nan"), float("inf")],
)
def test_invalid_model_cost_is_rejected(invalid_cost):
    pool = make_pool()

    pool[0] = replace(
        pool[0],
        mean_cost_usd=invalid_cost,
    )

    with pytest.raises(ValueError, match="cost"):
        AlwaysStrongest(pool)


@pytest.mark.parametrize("seed", [True, 1.5, "1234"])
def test_invalid_random_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        RandomBaseline(make_pool(), seed=seed)


def test_episode_oracle_requires_binding():
    _, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)

    with pytest.raises(ValueError, match="bound"):
        oracle.decide(sim.current_task())


def test_episode_oracle_selects_optimal_first_model():
    episode, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    strategy = oracle.decide(sim.current_task())

    assert strategy.model_id == "m1"
    assert strategy.halt == "stop"
    assert strategy.round == 0


def test_episode_oracle_matches_exact_plan():
    episode, sim = make_environment()

    expected = exact_episode_oracle(episode)

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    transitions = run_policy(oracle, sim)

    metrics = sim.final_metrics()

    assert transitions[-1].done
    assert metrics["n_correct"] == expected.correct_steps
    assert metrics["quality"] == pytest.approx(
        expected.quality
    )
    assert metrics["cost_usd"] == pytest.approx(
        expected.cost_usd
    )


def test_episode_oracle_completes_all_steps():
    episode, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    run_policy(oracle, sim)

    metrics = sim.final_metrics()

    assert metrics["n_correct"] == 3
    assert metrics["quality"] == 1.0
    assert metrics["cost_usd"] == pytest.approx(11.0)


def test_episode_oracle_handles_limited_budget():
    episode, sim = make_environment(
        rows=(
            (0, 0, 0, 1),
            (0, 0, 1, 1),
            (0, 0, 1, 1),
        ),
        budget=9.0,
    )

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    run_policy(oracle, sim)

    metrics = sim.final_metrics()

    assert metrics["n_correct"] == 2
    assert metrics["quality"] == pytest.approx(2 / 3)
    assert metrics["cost_usd"] == pytest.approx(9.0)


@pytest.mark.parametrize(
    "episode_type",
    ["random", "correlated", "retry", "propagation"],
)
def test_episode_oracle_supports_all_episode_types(
    episode_type,
):
    episode, sim = make_environment(
        rows=((1, 1, 1, 1),) * 3,
        ep_type=episode_type,
    )

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    run_policy(oracle, sim)

    assert sim.final_metrics()["quality"] == 1.0
    assert sim.final_metrics()["cost_usd"] == pytest.approx(3.0)


def test_episode_oracle_rejects_wrong_episode():
    episode, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    wrong_task = replace(
        sim.current_task(),
        episode_id="different-episode",
    )

    with pytest.raises(ValueError, match="different episode"):
        oracle.decide(wrong_task)


def test_episode_oracle_rejects_wrong_prompt():
    episode, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    wrong_task = replace(
        sim.current_task(),
        step_emb_key="wrong-prompt",
    )

    with pytest.raises(ValueError, match="prompt"):
        oracle.decide(wrong_task)


def test_episode_oracle_rejects_wrong_step_id():
    episode, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    wrong_task = replace(
        sim.current_task(),
        step_id="invalid-step",
    )

    with pytest.raises(ValueError, match="step ID"):
        oracle.decide(wrong_task)


def test_episode_oracle_rejects_wrong_model_order():
    episode, _ = make_environment()

    oracle = EpisodeOracleBaseline(
        list(reversed(IDS))
    )

    with pytest.raises(ValueError, match="order"):
        oracle.bind_episode(episode)


def test_episode_oracle_rejects_duplicate_model_ids():
    with pytest.raises(ValueError, match="unique"):
        EpisodeOracleBaseline(
            ["m0", "m0", "m2", "m3"]
        )


def test_episode_oracle_can_rebind_to_new_episode():
    first_episode, first_sim = make_environment(
        episode_id="episode-one",
    )

    second_episode, second_sim = make_environment(
        episode_id="episode-two",
    )

    oracle = EpisodeOracleBaseline(IDS)

    oracle.bind_episode(first_episode)
    assert oracle.decide(first_sim.current_task()).model_id == "m1"

    oracle.bind_episode(second_episode)
    assert oracle.decide(second_sim.current_task()).model_id == "m1"

    with pytest.raises(ValueError, match="different episode"):
        oracle.decide(first_sim.current_task())


def test_episode_oracle_does_not_mutate_environment():
    episode, _ = make_environment()

    original = episode.initial_state()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    assert episode.initial_state() == original


def test_strategy_does_not_expose_oracle_ground_truth():
    episode, sim = make_environment()

    oracle = EpisodeOracleBaseline(IDS)
    oracle.bind_episode(episode)

    strategy = oracle.decide(sim.current_task())

    assert isinstance(strategy, Strategy)
    assert not hasattr(strategy, "counterfactual")
    assert not hasattr(strategy, "ground_truth")
