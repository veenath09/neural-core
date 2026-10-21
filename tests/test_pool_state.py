"""Tests for Part 2 Task 6: run-time model onboarding (train/pool_state.py)."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from controller.contracts import ModelCard
from controller.embedding_cache import EmbeddingCache
from controller.policy import log_price
from controller.session import ControllerSession
from measurement.online_env import SimStepEnv
from measurement.profile_updater import ProfileUpdater
from measurement.rl_reward import LAMS, load_cost_ref
from tests.test_env import IDS
from tests.test_rollout import tiny_model, world_and_episodes
from train.baseline import BucketedBaseline
from train.online import EpisodeLog, make_canary, replay, run_episode
from train.pool_state import PoolState


def card(model_id: str, tier: int, cost: float = 1e-4, acc: float = 0.7) -> ModelCard:
    return ModelCard(model_id=model_id, provider="x", tier=tier, mean_cost_usd=cost, price_source="test",
                     price_date="2026-10-22", capability=[acc] * 5 + [float(np.log10(cost))])


def pool4() -> PoolState:
    costs = [1e-4, 2e-4, 4e-4, 8e-4]
    return PoolState([card(m, i, c) for i, (m, c) in enumerate(zip(IDS, costs))], ProfileUpdater(prior_weight=50))


def fake_log(uses: list[tuple[str, int, float, float]]) -> EpisodeLog:
    return EpisodeLog(episode_id="e", ep_type="random", lam_idx=0, mode="serve", decisions=[], quality=0.0,
                      cost_usd=0.0, n_steps=1, policy_version=0, used=uses)


def test_bonus_decays_to_zero_and_status_flips():
    pool = pool4()
    pool.add_models([card("new", 4)])
    assert pool.bonus().tolist() == [0, 0, 0, 0, 1.0]
    pool.record(fake_log([("new", 0, 1.0, 1e-4)] * 100))
    assert pool.bonus()[4] == pytest.approx(0.5) and pool.status["new"] == "warmup"
    pool.record(fake_log([("new", 0, 1.0, 1e-4)] * 100))
    assert pool.bonus()[4] == 0.0 and pool.status["new"] == "active"


def test_bonus_is_split_across_simultaneous_arrivals():
    pool = pool4()
    pool.add_models([card("a", 4), card("b", 5)])
    assert pool.bonus()[4:].tolist() == [0.5, 0.5]
    pool.record(fake_log([("a", 0, 1.0, 1e-4)] * 200 + [("b", 0, 1.0, 1e-4)] * 50))
    assert pool.status["a"] == "active"
    assert pool.bonus()[4] == 0.0 and pool.bonus()[5] == pytest.approx(1.0 * (1 - 50 / 200))


def test_add_model_keeps_existing_indices():
    pool = pool4()
    before = pool.model_ids()
    pool.add_models([card("new", 4)])
    assert pool.model_ids()[:4] == before and pool.model_ids()[4] == "new"
    assert pool.capability_tensor().shape == (5, 6) and len(pool.costs()) == 5


def test_record_updates_card_with_prior_weight_50():
    pool = pool4()
    pool.record(fake_log([(IDS[0], 0, 1.0, 1e-4)] * 9 + [(IDS[0], 0, 0.0, 1e-4)]))
    assert pool.cards[0].capability[0] == pytest.approx((50 * 0.7 + 10 * 0.9) / 60, abs=1e-4)  # 0.7333
    assert pool.cards[0].capability[1] == pytest.approx(0.7)  # step type 1 unseen: prior kept


def test_retired_model_is_masked_not_deleted():
    world, eps = world_and_episodes(8)
    model = tiny_model("full")
    pool = pool4()
    cost_ref = load_cost_ref()
    session = ControllerSession(model, lambda: world.card_costs, cost_ref,
                                EmbeddingCache(model, dict(world.emb_index), embed_fn=None),
                                pool.model_ids, pool.available, lams=LAMS)
    old = [run_episode(session, SimStepEnv(ep), 0, "sample", pool.bonus, None, 0) for ep in eps]
    pool.retire(IDS[3])
    assert pool.model_ids() == list(IDS) and pool.available().tolist() == [True, True, True, False]
    logs = [run_episode(session, SimStepEnv(ep), k % 5, "sample", pool.bonus, None, 0) for k, ep in enumerate(eps)]
    assert all(d.action != 3 for log in logs for d in log.decisions)
    assert all(not d.mask[3] for log in logs for d in log.decisions)
    trace, behaviour = replay(model, old, lambda m: log_price(world.card_costs[:m], cost_ref))
    assert torch.isfinite(trace.logp).all()


def test_replay_survives_pool_growth():
    world, eps = world_and_episodes(10)
    model = tiny_model("full")
    pool = pool4()
    cost_ref = load_cost_ref()
    model.set_cards(pool.capability_tensor())
    session = ControllerSession(copy.deepcopy(model), pool.costs, cost_ref,
                                EmbeddingCache(model, dict(world.emb_index), embed_fn=None),
                                pool.model_ids, pool.available, lams=LAMS)
    logs = [run_episode(session, SimStepEnv(ep), k % 5, "sample", pool.bonus, None, 0) for k, ep in enumerate(eps)]
    pool.add_models([card("new", 4, cost=1.6e-3)])
    model.set_cards(pool.capability_tensor())  # 5 cards now
    trace, behaviour = replay(model, logs, lambda m: log_price(pool.costs()[:m], cost_ref))
    c = trace.choice
    assert torch.isfinite(trace.logp).all()
    assert torch.allclose(trace.logp[c].detach(), behaviour[c], atol=1e-5)
    assert model.cards.shape[0] == 5  # replay restores the full pool afterwards


def test_baseline_shape_unchanged_when_pool_grows():
    world, eps = world_and_episodes(8)
    model = tiny_model("full")
    pool = pool4()
    cost_ref = load_cost_ref()
    pool.add_models([card("a", 4, 1.6e-3), card("b", 5, 3.2e-3), card("c", 6, 6.4e-3)])
    model.set_cards(pool.capability_tensor())
    session = ControllerSession(copy.deepcopy(model), pool.costs, cost_ref,
                                EmbeddingCache(model, dict(world.emb_index), embed_fn=None),
                                pool.model_ids, pool.available, lams=LAMS)
    # 7 models in the pool, but the simulator only knows 4: keep the new ones out of the episodes
    pool.retire("a"), pool.retire("b"), pool.retire("c")
    logs = [run_episode(session, SimStepEnv(ep), k % 5, "sample", pool.bonus, None, 0) for k, ep in enumerate(eps)]
    trace, _ = replay(model, logs, lambda m: log_price(pool.costs()[:m], cost_ref))
    baseline = BucketedBaseline()
    baseline.update(trace.band[trace.choice], trace.lam_idx[:, None].expand_as(trace.band)[trace.choice],
                    trace.reward[trace.choice])
    assert baseline.table.shape == (5, 5) and ((trace.band >= 0) & (trace.band <= 4)).all()


def test_canary_masks_models_without_sim_outcomes():
    world, eps = world_and_episodes(8)
    model = tiny_model("full")
    pool = pool4()
    pool.add_models([card("live-only", 4, 1.6e-3)], sim_known=False)
    model.set_cards(pool.capability_tensor())
    assert pool.sim_model_ids() == list(IDS)
    canary = make_canary([e.skeleton for e in eps], lambda: world, load_cost_ref(),
                         cards_fn=lambda: pool.capability_tensor()[pool.sim_indices()])
    quality = canary(model)
    assert set(quality) == set(range(len(LAMS))) and all(0 <= q <= 1 for q in quality.values())
    assert model.cards.shape[0] == 5  # restored after the canary


def test_pool_state_round_trips_json():
    pool = pool4()
    pool.add_models([card("new", 4)], sim_known=False)
    pool.record(fake_log([("new", 1, 1.0, 2e-4)] * 30))
    pool.retire(IDS[2])
    back = PoolState.from_json(pool.to_json())
    assert back.model_ids() == pool.model_ids() and back.status == pool.status
    assert back.bonus().tolist() == pool.bonus().tolist() and back.sim_model_ids() == pool.sim_model_ids()
    assert np.allclose(back.capability_tensor(), pool.capability_tensor())


def test_load_data_excludes_model(tmp_path):
    from tests.test_train import tiny_data  # noqa: F401  (fixture style only)
    from train.train import exclude_from

    import pandas as pd
    from data.routerbench import OutcomeTable

    frame = pd.DataFrame({"prompt_id": ["p0", "p1"], "source": "mmlu", "step_type": 0, "band": [0, 1],
                          "split": "train", **{f"correct_{i}": [True, i % 2 == 0] for i in range(4)},
                          **{f"cost_{i}": [float(i + 1), float(i + 1)] for i in range(4)}})
    cards = [card(m, i) for i, m in enumerate(IDS)]
    frame2, cards2 = exclude_from(frame, cards, (IDS[1],))
    table = OutcomeTable.from_frame(frame2)
    assert table.n_models == 3 and [c.model_id for c in cards2] == [IDS[0], IDS[2], IDS[3]]
    assert frame2["cost_1"].tolist() == [3.0, 3.0] and [c.tier for c in cards2] == [0, 1, 2]
    with pytest.raises(ValueError):
        exclude_from(frame, cards, ("nope",))


def test_record_debiases_a_noisy_verifier():
    costs = [1e-4, 2e-4, 4e-4, 8e-4]
    pool = PoolState([card(m, i, c) for i, (m, c) in enumerate(zip(IDS, costs))], ProfileUpdater(prior_weight=50),
                     flip_p=0.15)
    # true accuracy 0.9 seen through a 15% flip: observed 0.9*0.85 + 0.1*0.15 = 0.78
    pool.record(fake_log([(IDS[0], 0, 1.0, 1e-4)] * 78 + [(IDS[0], 0, 0.0, 1e-4)] * 22))
    assert pool.cards[0].capability[0] == pytest.approx((50 * 0.7 + 100 * 0.9) / 150, abs=1e-6)


def test_record_handles_depth_cards_and_keeps_extra_features():
    depth_card = ModelCard(model_id="luna@high", provider="openai", tier=0, mean_cost_usd=1e-4, price_source="t",
                           price_date="2026-10-22", capability=[0.7] * 5 + [-4.0, 1.0, 0.5])
    pool = PoolState([depth_card], ProfileUpdater(prior_weight=50))
    pool.record(fake_log([("luna@high", 4, 1.0, 1e-3)] * 50))
    cap = pool.cards[0].capability
    assert cap[4] == pytest.approx((50 * 0.7 + 50 * 1.0) / 100)  # gsm8k (step type 4) updated
    assert cap[5] == pytest.approx(-3.0)  # cost slot: log10 of the running mean cost
    assert cap[6:] == [1.0, 0.5]  # depth features untouched


def test_exclude_models_flow_from_metrics(tmp_path, monkeypatch):
    import json

    import train.rl as rl_module
    import train.train as train_module
    from tests.test_train import tiny_data
    from train.train import Data, load_model, train

    full = tiny_data()
    ids = list(full.world.model_ids)

    def fake_load_data(data_dir, exclude_models=()):
        keep = [m for m in ids if m not in exclude_models]
        from eval.online_sim import subset_data

        d = subset_data(full, keep)
        return Data(d.world, d.train, d.val, d.embeddings, d.cards, d.n_types, tmp_path, tuple(exclude_models))

    monkeypatch.setattr(train_module, "load_data", fake_load_data)
    monkeypatch.setattr(rl_module, "load_data", fake_load_data)
    data = fake_load_data(tmp_path, (ids[3],))
    train("full", seed=0, epochs=1, batch_size=32, data=data, out_dir=tmp_path, log=lambda row: None)
    run = tmp_path / "full-s0"
    assert json.loads((run / "metrics.json").read_text())["exclude_models"] == [ids[3]]
    assert load_model(run, strict=False).cards.shape[0] == 3  # rebuilt without the excluded model
    with pytest.raises(ValueError):
        rl_module.train_rl("full", 0, run, updates=1, exclude_models=(ids[2],))
    summary = rl_module.train_rl("full", 0, run, updates=1, batch_size=8, eval_every=1, max_val=8,
                                 out_dir=tmp_path / "rl", log=lambda r: None)
    assert summary["exclude_models"] == [ids[3]]


def test_baseline_refuses_a_wrong_shape():
    b = BucketedBaseline()
    bad = b.state_dict() | {"table": torch.zeros(7, 5)}
    with pytest.raises(ValueError):
        b.load_state_dict(bad)
