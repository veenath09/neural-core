import numpy as np
import pytest
import torch

from data.episodes import Skeleton
from measurement.rl_reward import LAMS, episode_reward
from measurement.scorer import CompositeScorer
from tests.test_env import IDS
from tests.test_rollout import tiny_model, world_and_episodes
from train.rollout import rollout


def _lam_idx(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(len(LAMS), size=n)


def test_sample_rollout_trace_is_aligned():
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    lam_idx = _lam_idx(len(eps))

    out = rollout(
        model,
        eps,
        world,
        policy="sample",
        rng=np.random.default_rng(1),
        lam_idx=lam_idx,
        cost_ref=scorer.cfg.cost_ref_usd,
    )

    rl = out.rl
    assert rl is not None
    assert rl.valid.shape[1] <= 12

    for i in range(len(eps)):
        assert int(rl.valid[i].sum()) == len(out.decision_actions[i])
        # choice is a subset of valid
        assert bool((rl.choice[i] & ~rl.valid[i]).any()) is False


def test_rewards_sum_to_measurement_reward():
    world, eps = world_and_episodes(4)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    with torch.no_grad():
        out = rollout(
            model,
            eps,
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    rl = out.rl
    for i, (ep, r) in enumerate(zip(eps, out.results)):
        expected = episode_reward(scorer, r.quality, r.cost_usd, ep.n_steps, LAMS[lam_idx[i]])
        got = float(rl.reward[i].sum())
        assert got == pytest.approx(expected, abs=1e-5)


def test_rewards_sum_when_nothing_is_affordable():
    world, _ = world_and_episodes()
    table = world.table
    budget = 0.25 * world.card_costs.min()
    sk = Skeleton("broke", "random", tuple(table.prompt_ids[:3]), float(budget))
    from data.env import Episode

    ep = Episode(sk, table, IDS)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = np.array([0])

    with torch.no_grad():
        out = rollout(
            model,
            [ep],
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    rl = out.rl
    assert int(rl.valid[0].sum()) == 1
    assert float(rl.reward[0].sum()) == pytest.approx(0.0, abs=1e-5)
    expected = episode_reward(scorer, out.results[0].quality, out.results[0].cost_usd, ep.n_steps, LAMS[0])
    assert float(rl.reward[0].sum()) == pytest.approx(expected, abs=1e-5)


def test_forced_stop_is_not_a_choice_but_keeps_its_reward():
    world, _ = world_and_episodes()
    table = world.table
    sk = Skeleton("tight", "random", tuple(table.prompt_ids[:3]), float(table.cost[:3, 0].sum()) * 1.01)
    from data.env import Episode

    ep = Episode(sk, table, IDS)
    model = tiny_model("full").eval()
    with torch.no_grad():
        model.heads.halt.bias.fill_(50.0)  # always wants to escalate

    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = np.array([0])

    with torch.no_grad():
        out = rollout(
            model,
            [ep],
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    rl = out.rl
    for action, choice, valid in zip(out.decision_actions[0], rl.choice[0], rl.valid[0]):
        if action["phase"] == "halt":
            assert bool(valid)
            assert bool(choice) is False

    expected = episode_reward(scorer, out.results[0].quality, out.results[0].cost_usd, ep.n_steps, LAMS[0])
    assert float(rl.reward[0].sum()) == pytest.approx(expected, abs=1e-5)


def test_band_target_is_the_exact_fail_fraction():
    from controller.contracts import MAX_ROUNDS
    from controller.features import CHOOSE, HALT, target

    world, eps = world_and_episodes(4)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    with torch.no_grad():
        out = rollout(
            model,
            eps,
            world,
            policy="serve",
            lam_idx=lam_idx,
            cost_ref=cost_ref,
        )

    rl = out.rl

    # Independently replay each episode's recorded action sequence to recompute the
    # expected band_target per decision from features.target at the pre-action state.
    for i, ep in enumerate(eps):
        s = ep.initial_state()
        expected_targets = []

        for action in out.decision_actions[i]:
            if not s.done and s.rounds == MAX_ROUNDS:
                s = ep.end_step(s)

            phase = CHOOSE if action["phase"] == "choose" else HALT
            expected_targets.append(target(ep, s, phase).band)

            if action["phase"] == "choose":
                s = ep.play(s, action["model"])
            elif action["halt"] == "escalate":
                s = ep.play(s, action["model"])
            else:
                s = ep.end_step(s)

        n_valid = int(rl.valid[i].sum())
        assert n_valid == len(out.decision_actions[i]) == len(expected_targets)

        for t, expected in enumerate(expected_targets):
            bt = float(rl.band_target[i, t])
            assert 0.0 <= bt <= 1.0
            assert bt == pytest.approx(expected, abs=1e-9)


def test_policy_gradient_reaches_memory_gate():
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    out = rollout(
        model,
        eps,
        world,
        policy="sample",
        rng=np.random.default_rng(2),
        lam_idx=lam_idx,
        cost_ref=cost_ref,
    )

    rl = out.rl
    loss = rl.logp[rl.choice].sum()
    loss.backward()
    assert model.memory.gate.readout.weight.grad.abs().sum() > 0


def test_serve_is_deterministic():
    world, eps = world_and_episodes(4)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    with torch.no_grad():
        out1 = rollout(model, eps, world, policy="serve", lam_idx=lam_idx, cost_ref=cost_ref)
        out2 = rollout(model, eps, world, policy="serve", lam_idx=lam_idx, cost_ref=cost_ref)

    assert [(r.quality, r.cost_usd) for r in out1.results] == [(r.quality, r.cost_usd) for r in out2.results]
    assert torch.equal(out1.rl.choice, out2.rl.choice)
    assert torch.allclose(out1.rl.reward, out2.rl.reward)


def test_sample_requires_lam_and_cost_ref():
    world, eps = world_and_episodes(2)
    model = tiny_model("full").eval()
    scorer = CompositeScorer.from_yaml("conf/score.yaml")
    cost_ref = scorer.cfg.cost_ref_usd
    lam_idx = _lam_idx(len(eps))

    with pytest.raises(ValueError):
        rollout(model, eps, world, policy="sample", lam_idx=lam_idx)

    with pytest.raises(ValueError):
        rollout(model, eps, world, policy="sample", cost_ref=cost_ref)

    with pytest.raises(ValueError):
        rollout(model, eps, world, policy="serve", lam_idx=lam_idx)

    with pytest.raises(ValueError):
        rollout(model, eps, world, policy="serve", cost_ref=cost_ref)
