import pandas as pd

from data.episodes import generate_skeletons
from eval.policies import AlwaysCheapest, AlwaysStrongest, EpisodeOracle, RandomPolicy, evaluate, run_episode
from tests.conftest import make_table
from tests.test_env import IDS, episode


def test_always_cheapest_and_strongest():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    cheap = run_episode(ep, AlwaysCheapest())
    strong = run_episode(ep, AlwaysStrongest())
    assert (cheap.quality, cheap.cost_usd) == (1 / 3, 3.0)
    assert (strong.quality, strong.cost_usd) == (1.0, 24.0)


def test_always_strongest_under_tight_budget_stops_early():
    ep = episode("random", [[1, 1, 1, 1]] * 3, budget=10.0)
    r = run_episode(ep, AlwaysStrongest())
    assert r.quality == 1 / 3 and r.cost_usd == 8.0


def test_oracle_policy_reaches_plan_value():
    ep = episode("random", [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]])
    r = run_episode(ep, EpisodeOracle())
    assert (r.quality, r.cost_usd) == (1.0, 11.0)


def test_oracle_beats_every_baseline_on_every_episode():
    table = make_table(n=200, seed=5)
    sks = generate_skeletons(table, list(table.prompt_ids), n=120, seed=0, split="t")
    df = evaluate(sks, table, IDS, [AlwaysCheapest(), AlwaysStrongest(), RandomPolicy(0), EpisodeOracle()])
    wide = df.pivot(index="episode_id", columns="policy", values="quality")
    for name in ("always_cheapest", "always_strongest", "random"):
        assert (wide["oracle"] >= wide[name]).all()
    assert len(df) == 4 * 120 and set(df.ep_type) == {"correlated", "retry", "propagation", "random"}


def test_random_policy_is_seeded():
    table = make_table(n=100, seed=6)
    sks = generate_skeletons(table, list(table.prompt_ids), n=30, seed=0, split="t")
    a = evaluate(sks, table, IDS, [RandomPolicy(1)])
    b = evaluate(sks, table, IDS, [RandomPolicy(1)])
    pd.testing.assert_frame_equal(a, b)
