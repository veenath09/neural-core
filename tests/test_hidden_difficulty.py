"""Optional hidden episode difficulty in the simulator (data.env.Episode hard_p)."""

import numpy as np

from data.env import Episode
from tests.test_rollout import world_and_episodes


def test_default_simulator_is_unchanged():
    world, eps = world_and_episodes(50)
    for ep in eps:
        e = Episode(ep.skeleton, world.table, world.model_ids, hard_p=0.0)
        assert not e.hard and np.array_equal(e.succeeds(e.initial_state()), ep.succeeds(ep.initial_state()))


def test_hard_episodes_fail_the_cheaper_half_on_every_step():
    world, eps = world_and_episodes(400)
    hard = [Episode(ep.skeleton, world.table, world.model_ids, model_costs=world.card_costs, hard_p=0.3) for ep in eps]
    share = np.mean([e.hard for e in hard])
    assert 0.2 < share < 0.4
    for e in hard:
        s = e.initial_state()
        while not s.done:
            ok = e.succeeds(s)
            if e.hard:
                assert not ok[:2].any()  # models 0 and 1 are the cheaper half of this cost-ordered pool
            s = e.end_step(e.play(s, 3)) if e.affordable(s, 3) else e.play(s, 3)


def test_hard_flag_is_fixed_per_episode_and_seed():
    world, eps = world_and_episodes(20)
    a = [Episode(ep.skeleton, world.table, world.model_ids, hard_p=0.5).hard for ep in eps]
    b = [Episode(ep.skeleton, world.table, world.model_ids, hard_p=0.5).hard for ep in eps]
    assert a == b
