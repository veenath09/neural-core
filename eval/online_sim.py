"""
Purpose:  Simulated online RL experiments (plan Part 2, Task 7): does learning online beat a frozen
          controller when the workload shifts, or when new models join the pool mid-run?
Owner:    C3 Measurement (K V A Bethmi)
Contract: v1
Inputs:   a simulator RL checkpoint dir (model.pt, metrics.json, baseline.pt; e.g. runs/routerbench/full-s0/rl_v4),
          the pool's Data (train.train.load_data), test skeletons.
Outputs:  <out>/<experiment>/stream.csv (one row per episode, learn and frozen), summary.json, curve.png.
Notes:    Prequential: every episode is run with the weights published before the learner sees it, so its
          reward is an honest out-of-sample score. learn=False runs the same stream, same order and seeds,
          with frozen weights but the same card updates. Deviation: C3's scripts/make_shifted_test.py reads
          C3's data/clean pipeline, which uses a different prompt split from the trained controllers; the
          shift here is rebuilt on our own test split with the same target task mix (TARGET_SHARES).
          Models that join keep the simulator's outcomes, and the world's columns are reordered to the
          pool's order (starting models, then joiners), so pool index == simulator index.

    python -m eval.online_sim shift --init runs/routerbench/full-s0/rl_v4
    python -m eval.online_sim new-models --init runs/routerbench/full-x1-s0/rl_v4 --model <id> [--model <id2>]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controller.contracts import ModelCard
from data.env import VERIFIER_FLIP_P, Episode
from data.episodes import BUDGET_FACTOR, EPISODE_TYPES, MAX_STEPS, MIN_STEPS, Skeleton, type_counts
from data.routerbench import SOURCES, OutcomeTable
from measurement.online_env import SimStepEnv
from measurement.profile_updater import PRIOR_WEIGHT, ProfileUpdater
from measurement.rl_reward import LAMS, cost_ref_for
from train.online import (  # noqa: F401  (subset_* and cards_from_data re-exported for callers)
    OnlineConfig,
    cards_from_data,
    run_episode,
    start_online,
    subset_data,
    subset_world,
    sync_cards,
)
from train.pool_state import N_ACC, PoolState
from train.rollout import World
from train.train import Data, load_data, load_model

TARGET_SHARES = {"mmlu": 0.15, "hellaswag": 0.15, "arc-challenge": 0.20, "winogrande": 0.20, "mbpp": 0.30}  # C3-13
LAST = 1000  # summary window (episodes)
ROLL = 200  # rolling window for curves and "episodes until learn beats frozen"


# ---------- pools and data ----------

def probe_card(model_id: str, data: Data, existing: list[ModelCard], n_probe: int = 50, seed: int = 0) -> ModelCard:
    """Card for a joining model: accuracy priors = mean of the existing cards, then n_probe training-split probes
    per step type with the w = 50 update; cost = mean probe cost. Generalises measurement.onboarding's
    build_onboarded_card (which requires exactly three existing cards) to any pool size. Features after the cost
    slot (the depth pool's depth level and thinking tokens) describe the request itself and are visible from the
    probe calls, so they are taken from the model's own card."""
    t, i = data.world.table, data.world.model_ids.index(model_id)
    train_ids = {p for sk in data.train for p in sk.prompt_ids}
    rows = np.array([t.index[p] for p in t.prompt_ids if p in train_ids])
    rng = np.random.default_rng(seed)
    prior = [float(np.mean([c.capability[k] for c in existing])) for k in range(N_ACC)]
    obs, costs = {}, []
    for k in range(N_ACC):
        pool_k = rows[t.step_type[rows] == k]
        if len(pool_k) == 0:
            continue
        take = rng.choice(pool_k, size=min(n_probe, len(pool_k)), replace=False)
        obs[k] = (float(t.correct[take, i].mean()), len(take))
        costs += list(t.cost[take, i])
    cost = float(np.mean(costs))
    extra = [float(x) for x in data.cards[i][N_ACC + 1:]]
    cold = ModelCard(model_id=model_id, provider="simulator", tier=len(existing), mean_cost_usd=cost,
                     price_source="training probes", price_date="", capability=prior + [math.log10(cost)] + extra)
    pool = PoolState([cold], ProfileUpdater(prior_weight=PRIOR_WEIGHT))
    return pool._update(cold, obs)


# ---------- workload shift ----------

def shifted_skeletons(table: OutcomeTable, prompt_ids: Sequence[str], n: int, seed: int, shares: dict[int, float],
                      by: str = "step_type") -> list[Skeleton]:
    """Episodes with the usual type mix, lengths and budgets, but prompts drawn with step-type shares `shares`
    (C3-13 workload shift) instead of the split's own mix. Correlated episodes keep one (type, band) group."""
    if by != "step_type":
        raise ValueError("shares are keyed by step type")
    rng = np.random.default_rng(seed)
    rows = np.array([table.index[p] for p in prompt_ids])
    types = table.step_type[rows]
    frac = {k: float((types == k).mean()) for k in np.unique(types)}
    w = np.array([shares.get(int(k), 0.0) / frac[k] for k in types])
    key = types * 10 + table.band[rows]
    groups = [(np.flatnonzero(key == g), w[key == g].sum()) for g in np.unique(key) if (key == g).sum() >= MAX_STEPS]
    groups = [(g, s) for g, s in groups if s > 0]
    counts = type_counts(n)
    ep_types = rng.permutation(np.repeat(EPISODE_TYPES, [counts[t] for t in EPISODE_TYPES]))
    out = []
    for i, ep_type in enumerate(ep_types):
        n_steps = int(rng.integers(MIN_STEPS, MAX_STEPS + 1))
        if ep_type == "correlated":
            gw = np.array([s for _, s in groups])
            members = groups[rng.choice(len(groups), p=gw / gw.sum())][0]
            pick = rng.choice(members, size=n_steps, replace=False, p=w[members] / w[members].sum())
        else:
            pick = rng.choice(len(rows), size=n_steps, replace=False, p=w / w.sum())
        chosen = rows[pick]
        budget = float(rng.uniform(*BUDGET_FACTOR) * table.cost[chosen, -1].sum())
        out.append(Skeleton(f"shift-{i:06d}", str(ep_type), tuple(str(p) for p in table.prompt_ids[chosen]), budget))
    return out


# ---------- the stream ----------

def stream(init_dir: Path, skeletons: Sequence[Skeleton], data: Data, cfg: OnlineConfig, learn: bool,
           lam_idx_fn: Callable[[int], int], add_model_at: dict[int, list[ModelCard]] | None = None, seed: int = 0,
           explore_share: float = 1.0, canary_skeletons: Sequence[Skeleton] | None = None, out_dir: Path | None = None,
           per_model_share: bool = False, update_cards: bool = True,
           world_at: dict[int, Callable[[World], World]] | None = None) -> pd.DataFrame:
    """One prequential stream. update_cards=False with learn=False is the fully static controller. world_at maps an
    episode index to a change of the simulator's ground truth from then on (drift: price shock, quality drop);
    cards are never told, they only learn it from experience."""
    init_dir = Path(init_dir)
    world_at = world_at or {}
    add_model_at = add_model_at or {}
    new_ids = [c.model_id for cs in add_model_at.values() for c in cs]
    start_ids = [m for m in data.world.model_ids if m not in new_ids]
    full = subset_world(data.world, start_ids + new_ids)  # pool index == simulator index

    torch.manual_seed(seed)
    learner, session, pool = start_online(init_dir, Path(out_dir or init_dir / "online"), data, cfg,
                                          start_ids=start_ids, world=full, canary_skeletons=canary_skeletons,
                                          seed=seed, flip_p=VERIFIER_FLIP_P)
    if not learn:
        learner = None  # same wiring, weights never updated

    rng = np.random.default_rng(seed + 1)
    rows = []
    for i, sk in enumerate(skeletons):
        if i in world_at:
            full = world_at[i](full)
        if i in add_model_at:
            pool.add_models(add_model_at[i], sim_known=True)
            sync_cards(pool, session, learner)
        lam_idx = int(lam_idx_fn(i))
        mode = "sample" if rng.random() < explore_share else "serve"
        state, version = learner.published() if learner else (None, 0)
        episode = Episode(sk, full.table, full.model_ids, model_costs=pool.costs())  # cards fixed within an episode
        log = run_episode(session, SimStepEnv(episode), lam_idx, mode, pool.bonus,
                          state, version)
        used = [u[0] for u in log.used]
        row = {"episode": i, "ep_type": log.ep_type, "lam": LAMS[lam_idx], "mode": mode, "quality": log.quality,
               "cost_usd": log.cost_usd, "reward": sum(d.reward for d in log.decisions),
               "policy_version": log.policy_version,
               "new_model_share": (sum(m in new_ids for m in used) / len(used)) if used else 0.0, "rollback": False}
        if per_model_share:
            row |= {f"share_{m}": (used.count(m) / len(used)) if used else 0.0 for m in pool.model_ids()}
        if update_cards:
            pool.record(log)
            sync_cards(pool, session, learner)
        if learner:
            update = learner.observe(log)
            row["rollback"] = bool(update and "rollback" in update["alarms"])
            row["alarms"] = "|".join(learner.monitor(log))  # review finding 7: live-reward alarm in the rows
        rows.append(row)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "pool.json").write_text(json.dumps(pool.to_json(), indent=2))
    return pd.DataFrame(rows)


# ---------- experiments ----------

ARMS = {"learn": dict(learn=True, update_cards=True),  # weights and cards adapt
        "frozen": dict(learn=False, update_cards=True),  # cards adapt, weights fixed
        "static": dict(learn=False, update_cards=False)}  # nothing adapts


def _run_pair(init_dir: Path, sks, data: Data, cfg: OnlineConfig, out: Path, seed: int, arms=("learn", "frozen"),
              **kw) -> pd.DataFrame:
    frames = []
    for arm in arms:
        df = stream(init_dir, sks, data, cfg, lam_idx_fn=lambda i: i % len(LAMS), seed=seed, out_dir=out / arm,
                    **ARMS[arm], **kw)
        frames.append(df.assign(arm=arm, seed=seed))
    return pd.concat(frames, ignore_index=True)


# ---------- drift scenarios ----------

def price_shock(model_id: str, factor: float) -> Callable[[World], World]:
    """The model's true per-call cost is multiplied by factor (its card still shows the old price)."""
    def apply(world: World) -> World:
        i = world.model_ids.index(model_id)
        t = world.table
        cost = t.cost.copy()
        cost[:, i] *= factor
        table = OutcomeTable(prompt_ids=t.prompt_ids, step_type=t.step_type, correct=t.correct, cost=cost, band=t.band)
        return World(table, world.model_ids, world.card_costs, world.emb_index, world.model_index)
    return apply


def quality_drop(model_id: str, keep: float, seed: int = 0) -> Callable[[World], World]:
    """The model now fails a share (1 - keep) of the questions it used to answer right (fixed by seed)."""
    def apply(world: World) -> World:
        i = world.model_ids.index(model_id)
        t = world.table
        correct = t.correct.copy()
        lose = np.random.default_rng(seed).random(len(correct)) > keep
        correct[lose, i] = False
        table = OutcomeTable(prompt_ids=t.prompt_ids, step_type=t.step_type, correct=correct, cost=t.cost, band=t.band)
        return World(table, world.model_ids, world.card_costs, world.emb_index, world.model_index)
    return apply


def with_better_model(data: Data, quality_of: str, price_of: str, new_id: str) -> Data:
    """SYNTHETIC joiner: quality_of's outcomes at price_of's per-call costs (a cheaper twin of a strong model).
    Labelled synthetic: no recorded model has this profile; it is the clean case where adoption must pay."""
    w, t = data.world, data.world.table
    q, c = w.model_ids.index(quality_of), w.model_ids.index(price_of)
    table = OutcomeTable(prompt_ids=t.prompt_ids, step_type=t.step_type,
                         correct=np.concatenate([t.correct, t.correct[:, [q]]], axis=1),
                         cost=np.concatenate([t.cost, t.cost[:, [c]]], axis=1), band=t.band)
    ids = list(w.model_ids) + [new_id]
    world = World(table, ids, np.append(w.card_costs, w.card_costs[c]), w.emb_index, {m: i for i, m in enumerate(ids)})
    card = data.cards[q].clone()
    card[N_ACC] = data.cards[c][N_ACC]
    return Data(world, data.train, data.val, data.embeddings, torch.cat([data.cards, card[None]]), data.n_types,
                data.data_dir, data.exclude_models)


def _with_card_costs(world: World, costs: np.ndarray) -> World:
    return World(world.table, world.model_ids, np.asarray(costs, dtype=float), world.emb_index, world.model_index)


def arm_rewards(init_dir: Path, data: Data, world: World, skeletons, weights: Path | None, pool_json: Path,
                cost_ref: float) -> dict[int, np.ndarray]:
    """Per-episode greedy reward at every lambda for one arm's final state (weights, cards) in a given world."""
    pool = PoolState.from_json(json.loads(Path(pool_json).read_text()))
    ids = pool.model_ids()
    w = _with_card_costs(subset_world(world, ids), pool.costs())
    model = load_model(init_dir, subset_data(data, [m for m in ids if m in data.world.model_ids]), strict=False)
    if weights is not None:
        model.load_state_dict(torch.load(weights, weights_only=True))
    model.set_cards(pool.capability_tensor())
    return {i: episode_rewards(model, skeletons, w, i, cost_ref) for i in range(len(LAMS))}


def compare_arms(rewards: dict[str, dict[int, np.ndarray]], base: str = "static") -> dict:
    out = {}
    for arm, by_lam in rewards.items():
        out[arm] = {}
        for i, r in by_lam.items():
            d = r - rewards[base][i]
            half = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
            out[arm][f"{LAMS[i]:g}"] = {"mean": float(r.mean()), f"diff_vs_{base}": float(d.mean()),
                                        "ci95": [float(d.mean() - half), float(d.mean() + half)]}
    return out


def experiment_drift(init_dir: Path, kind: str, out_dir: Path, seeds=(0,), n_episodes: int = 5000, at: int = 1000,
                     cfg: OnlineConfig | None = None, target: str | None = None) -> dict:
    """kind: price_shock (target x10 at episode `at`), quality_drop (target keeps half its right answers at `at`),
    better_model (a SYNTHETIC cheaper twin of the strongest model joins at episode 0). target defaults to the
    model the controller uses most (price/quality) or the strongest model (better_model)."""
    init_dir, data, _ = _init(init_dir)
    cfg = cfg or OnlineConfig()
    out = Path(out_dir) / f"drift_{kind}"
    sks = _test_skeletons(data)[:n_episodes]
    ids = list(data.world.model_ids)
    kw = {"canary_skeletons": data.val[:cfg.canary_episodes], "per_model_share": True}
    if kind in ("price_shock", "quality_drop"):
        target = target or ids[1]
        change = price_shock(target, 10.0) if kind == "price_shock" else quality_drop(target, 0.5, seed=7)
        kw["world_at"] = {at: change}
        stream_data = data
    elif kind == "better_model":
        target = target or ids[-1]
        new_id = f"synthetic-cheap-{target.split('/')[-1]}"
        stream_data = with_better_model(data, target, ids[0], new_id)
        known = cards_from_data(stream_data)
        joiner = replace(probe_card(new_id, stream_data, [c for c in known if c.model_id != new_id]), tier=len(ids))
        kw["add_model_at"] = {0: [joiner]}
    else:
        raise ValueError(f"unknown drift {kind!r}")
    frames = [_run_pair(init_dir, sks, stream_data, cfg, out / f"s{seed}", seed, arms=tuple(ARMS), **kw)
              for seed in seeds]
    df = pd.concat(frames, ignore_index=True)
    summary = _write(df, out, new_models=kind == "better_model")
    # held-out comparison of each arm's FINAL state in the post-drift world
    world = stream_data.world
    if "world_at" in kw:
        world = kw["world_at"][at](world)
    cost_ref = cost_ref_for(data.data_dir)
    s0 = out / f"s{seeds[0]}"
    finals = {"learn": max((s0 / "learn").glob("v*.pt"), key=lambda p: int(p.stem[1:]), default=None),
              "frozen": None, "static": None}
    rewards = {arm: arm_rewards(init_dir, stream_data, world, data.val, finals[arm], s0 / arm / "pool.json", cost_ref)
               for arm in ARMS}
    summary |= {"drift": kind, "target": target, "at_episode": at if kind != "better_model" else 0,
                "held_out": compare_arms(rewards)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarise(df: pd.DataFrame) -> dict:
    last = df[df["episode"] >= df["episode"].max() + 1 - LAST]
    per_lam = last.pivot_table(index="lam", columns="arm", values="reward", aggfunc="mean")
    roll = {arm: g.sort_values("episode")["reward"].rolling(ROLL).mean().to_numpy()
            for arm, g in df[df["seed"] == df["seed"].min()].groupby("arm")}
    ahead = np.flatnonzero(roll["learn"] > roll["frozen"]) if "frozen" in roll else []
    return {"episodes": int(df["episode"].max() + 1), "seeds": sorted(df["seed"].unique().tolist()),
            "mean_reward_last": {f"{lam:g}": {a: float(per_lam.loc[lam, a]) for a in per_lam.columns}
                                 for lam in per_lam.index},
            "learn_beats_frozen_at_every_lambda": bool((per_lam["learn"] > per_lam["frozen"]).all()),
            "quality_last": last.groupby("arm")["quality"].mean().to_dict(),
            "cost_last": last.groupby("arm")["cost_usd"].mean().to_dict(),
            "rollbacks": int(df[df["arm"] == "learn"]["rollback"].sum()),
            "episodes_until_learn_beats_frozen": int(ahead[0]) if len(ahead) else None}


def plot(df: pd.DataFrame, path: Path, new_models: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if new_models else 1, figsize=(12 if new_models else 7, 4), squeeze=False)
    for arm, g in df[df["seed"] == df["seed"].min()].groupby("arm"):
        g = g.sort_values("episode")
        axes[0, 0].plot(g["episode"], g["reward"].rolling(ROLL).mean(), label=arm)
        if new_models:
            axes[0, 1].plot(g["episode"], g["new_model_share"].rolling(ROLL).mean(), label=arm)
    axes[0, 0].set(xlabel="episode", ylabel=f"reward (rolling {ROLL})", title="online learn vs frozen")
    axes[0, 0].legend()
    if new_models:
        axes[0, 1].set(xlabel="episode", ylabel="share of rounds on new models", title="new-model use")
        axes[0, 1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _init(init_dir: Path) -> tuple[Path, Data, dict]:
    summary = json.loads((Path(init_dir) / "metrics.json").read_text())
    return Path(init_dir), load_data(Path(summary["data_dir"])), summary


def experiment_shift(init_dir: Path, out_dir: Path, seeds=(0,), n_episodes: int = 5000,
                     cfg: OnlineConfig | None = None) -> dict:
    init_dir, data, _ = _init(init_dir)
    cfg = cfg or OnlineConfig()
    table = data.world.table
    test_ids = sorted({p for sk in _test_skeletons(data) for p in sk.prompt_ids})
    shares = {SOURCES.index(s): v for s, v in TARGET_SHARES.items()}
    out = Path(out_dir) / "shift"
    frames = [_run_pair(init_dir, shifted_skeletons(table, test_ids, n_episodes, 100 + seed, shares), data, cfg,
                        out / f"s{seed}", seed, canary_skeletons=data.val[:cfg.canary_episodes]) for seed in seeds]
    return _write(pd.concat(frames, ignore_index=True), out, new_models=False)


def experiment_new_models(init_dir: Path, new_models: list[str], out_dir: Path, seeds=(0,), n_episodes: int = 5000,
                          cfg: OnlineConfig | None = None) -> dict:
    init_dir, data, _ = _init(init_dir)
    cfg = cfg or OnlineConfig()
    known = cards_from_data(data)
    existing = [c for c in known if c.model_id not in new_models]
    joining = [replace(probe_card(m, data, existing), tier=len(existing) + k) for k, m in enumerate(new_models)]
    out = Path(out_dir) / ("new_model_" + ("single" if len(new_models) == 1 else "set"))
    sks = _test_skeletons(data)[:n_episodes]
    frames = [_run_pair(init_dir, sks, data, cfg, out / f"s{seed}", seed, add_model_at={0: joining},
                        per_model_share=True, canary_skeletons=data.val[:cfg.canary_episodes])  # review finding 5
              for seed in seeds]
    df = pd.concat(frames, ignore_index=True)
    summary = _write(df, out, new_models=True)
    truth = {c.model_id: c.capability[:N_ACC] for c in known}  # accuracy features only (depth cards have 8)
    final = PoolState.from_json(json.loads((out / f"s{seeds[0]}" / "learn" / "pool.json").read_text()))
    final_cards = {c.model_id: c for c in final.cards}
    learn = df[(df["arm"] == "learn") & (df["episode"] >= df["episode"].max() + 1 - LAST)]
    summary["new_models"] = {m: {"final_share": float(learn[f"share_{m}"].mean()),
                                 "probe_card_accuracy": c.capability[:N_ACC],
                                 "final_card_accuracy": final_cards[m].capability[:N_ACC],
                                 "true_train_accuracy": truth[m]}
                             for m, c in zip(new_models, joining)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _test_skeletons(data: Data) -> list[Skeleton]:
    from data.episodes import load_skeletons

    return load_skeletons("test", data.data_dir)


def _write(df: pd.DataFrame, out: Path, new_models: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stream.csv", index=False)
    summary = summarise(df)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(df, out / "curve.png", new_models=new_models)
    return summary


# ---------- paired evaluation of the final weights ----------

def episode_rewards(model, skeletons, world: World, lam_idx: int, cost_ref: float, batch_size: int = 512) -> np.ndarray:
    """Per-episode serve-mode reward at one lambda (greedy, so two models can be compared episode by episode)."""
    from measurement.rl_reward import episode_reward, scorer_for
    from train.rollout import rollout
    from train.train import make_episodes

    model.eval()
    scorer, out = scorer_for(cost_ref), []
    with torch.no_grad():
        for s in range(0, len(skeletons), batch_size):
            eps = make_episodes(skeletons[s:s + batch_size], world)
            res = rollout(model, eps, world, policy="serve", lam_idx=np.full(len(eps), lam_idx), cost_ref=cost_ref).results
            out += [episode_reward(scorer, r.quality, r.cost_usd, e.n_steps, LAMS[lam_idx]) for r, e in zip(res, eps)]
    return np.array(out)


def paired_eval(init_dir: Path, run_dir: Path, skeletons, n_boot: int = 0) -> dict:
    """Final learned weights vs the frozen start on the same held-out episodes, under the learn arm's final cards.

    run_dir is a stream's learn output (v{n}.pt files, pool.json). Returns, per lambda, the mean reward of each
    and the paired difference with a 95% normal CI (1.96 * sd / sqrt(n))."""
    init_dir, run_dir = Path(init_dir), Path(run_dir)
    _, data, _ = _init(init_dir)
    pool = PoolState.from_json(json.loads((run_dir / "pool.json").read_text()))
    ids = pool.model_ids()
    world = subset_world(data.world, ids)
    # cards are not part of a checkpoint; both models get the final pool's cards below
    frozen = load_model(init_dir, subset_data(data, ids), strict=False)
    learned = copy.deepcopy(frozen)
    last = max(run_dir.glob("v*.pt"), key=lambda p: int(p.stem[1:]))
    learned.load_state_dict(torch.load(last, weights_only=True))
    for m in (frozen, learned):
        m.set_cards(pool.capability_tensor())
    cost_ref = cost_ref_for(data.data_dir)
    out = {"final_version": int(last.stem[1:]), "episodes": len(skeletons), "per_lambda": {}}
    for i, lam in enumerate(LAMS):
        a, b = episode_rewards(learned, skeletons, world, i, cost_ref), episode_rewards(frozen, skeletons, world, i, cost_ref)
        d = a - b
        half = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
        out["per_lambda"][f"{lam:g}"] = {"learned": float(a.mean()), "frozen": float(b.mean()), "diff": float(d.mean()),
                                         "ci95": [float(d.mean() - half), float(d.mean() + half)]}
    out["learned_better_at_every_lambda"] = all(v["ci95"][0] > 0 for v in out["per_lambda"].values())
    out["learned_worse_somewhere"] = any(v["ci95"][1] < 0 for v in out["per_lambda"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["shift", "new-models", "paired", "drift"])
    ap.add_argument("--kind", choices=["price_shock", "quality_drop", "better_model"], help="drift scenario")
    ap.add_argument("--target", help="drift: model to shock / degrade / copy (default: see experiment_drift)")
    ap.add_argument("--at", type=int, default=1000, help="drift: episode at which the world changes")
    ap.add_argument("--run", type=Path, help="paired: a stream's learn dir (v*.pt, pool.json)")
    ap.add_argument("--held-out", choices=["shift", "val"], default="val", help="paired: which held-out episodes")
    ap.add_argument("--init", type=Path, required=True, help="simulator RL checkpoint dir")
    ap.add_argument("--model", action="append", default=[], help="model joining at episode 0 (new-models)")
    ap.add_argument("--out", type=Path, default=Path("results/online_sim"))
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--lr", type=float, default=OnlineConfig.lr)
    a = ap.parse_args()
    cfg = OnlineConfig(lr=a.lr)
    seeds = tuple(a.seed or [0])
    if a.experiment == "paired":
        _, data, _ = _init(a.init)
        held = data.val
        if a.held_out == "shift":
            ids = sorted({p for sk in _test_skeletons(data) for p in sk.prompt_ids})
            held = shifted_skeletons(data.world.table, ids, 2000, 999, {SOURCES.index(k): v for k, v in TARGET_SHARES.items()})
        result = paired_eval(a.init, a.run, held)
        (a.run.parent.parent / "paired.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif a.experiment == "drift":
        print(json.dumps(experiment_drift(a.init, a.kind, a.out, seeds, a.episodes, a.at, cfg, a.target), indent=2))
    elif a.experiment == "shift":
        print(json.dumps(experiment_shift(a.init, a.out, seeds, a.episodes, cfg), indent=2))
    else:
        print(json.dumps(experiment_new_models(a.init, a.model, a.out, seeds, a.episodes, cfg), indent=2))


if __name__ == "__main__":
    main()
