"""Tests for Part 2 Task 5: OnlineLearner (train/online.py)."""

import copy
import time

import numpy as np
import pytest
import torch

from controller.embedding_cache import EmbeddingCache
from controller.model import CONFIGS, Controller
from controller.policy import log_price
from controller.session import ControllerSession
from data.env import Episode
from data.episodes import generate_skeletons
from data.routerbench import OutcomeTable
from measurement.online_env import SimStepEnv
from measurement.rl_reward import LAMS, load_cost_ref
from tests.conftest import make_table
from tests.test_env import IDS
from tests.test_rollout import tiny_model, world_and_episodes
from train.baseline import BucketedBaseline
from train.online import OnlineConfig, OnlineLearner, run_episode
from train.rollout import World, rollout
from train.rl import entropy_beta, rl_update


def _session(model, world, cost_ref):
    emb = EmbeddingCache(model, dict(world.emb_index), embed_fn=None)
    avail = np.ones(len(world.model_ids), dtype=bool)
    return ControllerSession(model, lambda: world.card_costs, cost_ref, emb, lambda: list(world.model_ids),
                             lambda: avail, lams=LAMS)


def _price_fn(world, cost_ref):
    return lambda pool_size: log_price(world.card_costs[:pool_size], cost_ref)


def _learner(model, world, tmp_path, cfg=None, canary=None, teacher=None, seed=0):
    cost_ref = load_cost_ref()
    return OnlineLearner(model, teacher if teacher is not None else copy.deepcopy(model), BucketedBaseline(),
                         cfg or OnlineConfig(), canary, tmp_path, np.random.default_rng(seed),
                         log_price_fn=_price_fn(world, cost_ref))


def _logs(model, world, eps, mode, version=0):
    cost_ref = load_cost_ref()
    session = _session(copy.deepcopy(model), world, cost_ref)
    bonus = np.zeros(len(world.model_ids))
    return [run_episode(session, SimStepEnv(ep), k % len(LAMS), mode, lambda: bonus, None, version)
            for k, ep in enumerate(eps)]


def test_update_fires_every_16_sample_episodes(tmp_path):
    world, eps = world_and_episodes(16)
    model = tiny_model("full")
    learner = _learner(model, world, tmp_path)
    serve = _logs(model, world, eps[:5], "serve")
    sample = _logs(model, world, eps, "sample")
    for log in serve:
        assert learner.observe(log) is None
    rows = [learner.observe(log) for log in sample]
    assert all(r is None for r in rows[:15]) and rows[15] is not None
    row = rows[15]
    for key in ("update", "loss", "pg_loss", "distill", "entropy", "mean_rho", "clipped_frac", "episode_return",
                "grad_norm", "policy_version", "alarms"):
        assert key in row
    assert row["policy_version"] == 1 and learner.published()[1] == 1
    assert (tmp_path / "v1.pt").exists() and (tmp_path / "log.jsonl").exists()


def test_importance_ratio_is_clipped(tmp_path):
    world, eps = world_and_episodes(16)
    model = tiny_model("full")
    learner = _learner(model, world, tmp_path)
    logs = _logs(model, world, eps, "sample")
    for log in logs[:8]:  # behaviour much less likely than now -> ratio far above 2
        for d in log.decisions:
            if d.has_choice:
                d.behaviour_logp -= 5.0
    row = None
    for log in logs:
        row = learner.observe(log) or row
    assert row["clipped_frac"] > 0 and row["max_rho"] <= 2.0 + 1e-6


def test_distill_pulls_complexity_to_teacher(tmp_path):
    world, eps = world_and_episodes(16)
    model = tiny_model("full")
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        teacher.heads.complexity.bias.add_(3.0)  # teacher says "harder" everywhere
    cfg = OnlineConfig(lr=1e-2, beta=0.0, replay_size=0)
    learner = _learner(model, world, tmp_path, cfg=cfg, teacher=teacher)
    logs = _logs(model, world, eps, "sample")
    for log in logs:
        for d in log.decisions:
            d.reward = 0.0  # zero advantage everywhere: only the distillation term moves the weights
    from train.online import replay

    with torch.no_grad():
        before = replay(model, logs, _price_fn(world, load_cost_ref()))[0]
        target = replay(teacher, logs, _price_fn(world, load_cost_ref()))[0]
    gap_before = (before.complexity - target.complexity)[before.valid].abs().mean()
    for log in logs:
        learner.observe(log)
    with torch.no_grad():
        after = replay(model, logs, _price_fn(world, load_cost_ref()))[0]
    gap_after = (after.complexity - target.complexity)[after.valid].abs().mean()
    assert gap_after < gap_before


def test_publish_does_not_change_running_episode(tmp_path):
    world, eps = world_and_episodes(17)
    model = tiny_model("full")
    cfg = OnlineConfig(lr=1e-2)
    learner = _learner(model, world, tmp_path, cfg=cfg)
    serving = copy.deepcopy(model)
    session = _session(serving, world, load_cost_ref())
    state, version = learner.published()
    env = SimStepEnv(eps[16])
    task_key, task_text, budget, n = env.start()
    session.reset("running", task_key, task_text, budget, n, 0, "serve", state_dict=state, policy_version=version)
    spec = env.current_step()
    session.decide(spec.step_id, spec.step_key, spec.step_text, spec.step_type, np.zeros(4))
    snapshot = {k: v.clone() for k, v in serving.state_dict().items()}
    for log in _logs(model, world, eps[:16], "sample"):
        learner.observe(log)  # publishes version 1 while "running" is mid-episode
    assert learner.published()[1] == version + 1
    assert all(torch.equal(snapshot[k], v) for k, v in serving.state_dict().items())
    new_state, new_version = learner.published()
    session.reset("next", task_key, task_text, budget, n, 0, "serve", state_dict=new_state, policy_version=new_version)
    assert any(not torch.equal(snapshot[k], v) for k, v in serving.state_dict().items())


def test_canary_rollback_restores_weights_and_halves_lr(tmp_path):
    world, eps = world_and_episodes(16)
    model = tiny_model("full")
    teacher = copy.deepcopy(model)

    def canary(m):  # the frozen start (teacher) scores 1, the learner 0: learning made it worse
        return {i: 1.0 if m is teacher else 0.0 for i in range(len(LAMS))}

    cfg = OnlineConfig(lr=1e-2, canary_every=1)
    learner = _learner(model, world, tmp_path, cfg=cfg, canary=canary, teacher=teacher)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    base_start = learner.baseline.state_dict()["table"].clone()
    row = None
    for log in _logs(model, world, eps, "sample"):
        row = learner.observe(log) or row
    assert "rollback" in row["alarms"]
    assert all(torch.equal(start[k], v) for k, v in model.state_dict().items())
    assert torch.equal(learner.baseline.table, base_start)
    assert learner.lr == pytest.approx(5e-3)
    assert all(torch.equal(start[k], v) for k, v in learner.published()[0].items())


def test_canary_compares_with_frozen_start_under_same_cards(tmp_path):
    world, eps = world_and_episodes(16)
    model = tiny_model("full")

    def canary(m):  # cards drifted: both score low, but equally -> learning did no harm, no rollback
        return {i: 0.2 for i in range(len(LAMS))}

    learner = _learner(model, world, tmp_path, cfg=OnlineConfig(lr=1e-2, canary_every=1), canary=canary)
    row = None
    for log in _logs(model, world, eps, "sample"):
        row = learner.observe(log) or row
    assert row["alarms"] == [] and row["canary_reference"] == row["canary"]


def test_kl_anchor_pulls_decisions_back_to_teacher(tmp_path):
    from train.online import replay

    world, eps = world_and_episodes(16)
    model = tiny_model("full")
    teacher = copy.deepcopy(model)
    with torch.no_grad():  # the learner has drifted: its policy differs from the teacher's
        model.heads.lam_adj.bias.add_(torch.tensor([1.5, -1.0]))
    cfg = OnlineConfig(lr=1e-3, beta=0.0, distill=0.0, kl=1.0, replay_size=0)
    learner = _learner(model, world, tmp_path, cfg=cfg, teacher=teacher)
    logs = _logs(model, world, eps, "sample")
    for log in logs:
        for d in log.decisions:
            d.reward = 0.0  # zero advantage: only the anchor moves the weights
    price = _price_fn(world, load_cost_ref())

    def gap():
        with torch.no_grad():
            a, b = replay(model, logs, price)[0], replay(teacher, logs, price)[0]
        return ((a.logp - b.logp) ** 2)[a.choice].mean()

    before = gap()
    for log in logs:
        learner.observe(log)
    assert gap() < before


def test_fresh_logs_give_part1_policy_gradient():
    """With rho = 1 (logs from the current weights), the online PG term equals Part 1's rl_loss."""
    from train.online import online_loss, replay
    from train.rl import rl_loss

    world, eps = world_and_episodes(16)
    model = tiny_model("full")
    logs = _logs(model, world, eps, "sample")
    trace, behaviour = replay(model, logs, _price_fn(world, load_cost_ref()))
    baseline = BucketedBaseline()
    baseline.table = torch.rand(5, 5)
    _, online = online_loss(trace, behaviour, trace.complexity.detach(), baseline, OnlineConfig(beta=0.0, distill=0.0))
    _, part1 = rl_loss(trace, baseline, beta=0.0, anchor=False)
    assert online["pg_loss"] == pytest.approx(part1["pg_loss"], abs=1e-7)
    assert online["mean_rho"] == pytest.approx(1.0, abs=1e-6)


def test_monitor_flags_a_live_reward_drop(tmp_path):
    world, eps = world_and_episodes(4)
    model = tiny_model("full")
    learner = _learner(model, world, tmp_path, cfg=OnlineConfig(monitor_window=4))
    log = _logs(model, world, eps[:1], "serve")[0]
    good, bad = copy.deepcopy(log), copy.deepcopy(log)
    for d in good.decisions:
        d.reward = 0.5 / len(good.decisions)
    for d in bad.decisions:
        d.reward = 0.0
    assert all(learner.monitor(good) == [] for _ in range(4))
    alarms = [learner.monitor(bad) for _ in range(4)]
    assert "live_reward_drop" in alarms[-1]


def _easy_world(correct_col: int, cost_scale: np.ndarray):
    table = make_table(n=120, seed=7)
    correct = np.zeros_like(table.correct)
    correct[:, correct_col] = True
    cost = table.cost / np.array([1.0, 2.0, 4.0, 8.0]) * cost_scale
    t = OutcomeTable(prompt_ids=table.prompt_ids, step_type=table.step_type, correct=correct, cost=cost)
    world = World(t, IDS, cost_scale * 1e-4, {p: i for i, p in enumerate(t.prompt_ids)},
                  {m: i for i, m in enumerate(IDS)})
    sks = generate_skeletons(t, list(t.prompt_ids), n=64, seed=0, split="t")
    return world, [Episode(sk, t, IDS) for sk in sks]


def _serve_share(model, world, eps, m):
    model.eval()
    with torch.no_grad():
        out = rollout(model, eps, world, policy="serve", lam_idx=np.full(len(eps), 1), cost_ref=load_cost_ref())
    picks = [a["model"] for acts in out.decision_actions for a in acts if a["phase"] == "choose"]
    return sum(p == m for p in picks) / len(picks)


def test_online_improves_on_easy_drift(tmp_path):
    t0 = time.time()
    # Part 1-style start: only model 0 is right, and it is the cheapest.
    world0, eps0 = _easy_world(0, np.array([1.0, 2.0, 4.0, 8.0]))
    torch.manual_seed(0)
    model = Controller(CONFIGS["full"], torch.randn(120, 384), torch.rand(4, 6), n_types=2)
    baseline, rng = BucketedBaseline(), np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for u in range(60):
        out = rollout(model, eps0, world0, policy="sample", rng=rng, lam_idx=rng.integers(len(LAMS), size=len(eps0)),
                      cost_ref=load_cost_ref())
        rl_update(model, out.rl, baseline, opt, beta=entropy_beta(u, 60))
    # Drift: model 1 becomes the one that is always right, and cheaper than model 0.
    world1, eps1 = _easy_world(1, np.array([2.0, 1.0, 4.0, 8.0]))
    before = _serve_share(model, world1, eps1, 1)
    # lr 1e-2: at 3e-3, 40 updates are too few to unlearn a confident start (share stayed 0);
    # the live default stays at the plan's 1e-5, where traffic supplies many more updates.
    cfg = OnlineConfig(lr=1e-2, beta=0.001, replay_size=16)
    learner = OnlineLearner(model, copy.deepcopy(model), baseline, cfg, None, tmp_path, np.random.default_rng(1),
                            log_price_fn=_price_fn(world1, load_cost_ref()))
    serving = copy.deepcopy(model)
    session = _session(serving, world1, load_cost_ref())
    bonus = np.zeros(4)
    k = 0
    while learner.updates < 40:
        state, version = learner.published()
        ep = eps1[k % len(eps1)]
        learner.observe(run_episode(session, SimStepEnv(ep), k % len(LAMS), "sample", lambda: bonus, state, version))
        k += 1
    after = _serve_share(model, world1, eps1, 1)
    assert after > 0.8 and after > before, (before, after)
    assert time.time() - t0 < 90
