"""
Purpose:  Live online RL pilot (plan Part 2, Task 8): the controller serves real questions through real
          providers, learns online, and new models can join live, under a hard spend cap.
Owner:    C4 Execution (S A Gunerathne) + C3 Measurement (K V A Bethmi)
Contract: v1
Inputs:   a depth-pool simulator-RL checkpoint (e.g. runs/depth_x_kimi/full-s0/rl_v5, trained without the
          joiners), real test questions with gold answers (data/processed/depth/table.parquet), API keys (.env).
Outputs:  results/online_live/<run_id>/log.jsonl (one row per episode and per update), v{n}.pt, pool.json,
          summary.json (spend, live reward, updates, rollbacks, live-vs-recorded agreement).
Notes:    Deviation: real calls go through data.depth_collect.call_arm (the tested collector, with each option's
          provider thinking parameters) instead of C4's ExecutionEngine, which has no adapters for these 2026
          options. Verification is exact match against the gold answer (deterministic: the same answer always
          gets the same verdict, so it cannot be re-rolled by escalating). Steps run in order (no simulator
          retry or propagation rules live). The spend cap is estimate-based: before a call it refuses if spent +
          3 x the option's mean cost would pass the cap, after a call it charges the actual list-price cost; like
          C3's RoutingSpendGuard it cannot guarantee a provider's billing to the cent. --dry-run uses a fake
          caller and makes no network call.

    python scripts/online_live.py --init runs/depth_x_kimi/full-s0/rl_v5 --join kimi-k2.6@on --episodes 200 \
        --spend-cap-usd 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from controller.contracts import TraceItem
from data.depth_collect import ARMS, call_arm, call_cost, load_env, score
from data.episodes import Skeleton, load_skeletons
from measurement.online_env import RoundOutcome, StepSpec
from measurement.rl_reward import LAMS
from train.online import OnlineConfig, run_episode, start_online, sync_cards
from train.train import Data, load_data, load_run_data

ESTIMATE_FACTOR = 3.0


class SpendCapReached(RuntimeError):
    pass


@dataclass
class Spend:
    cap: float
    spent: float = 0.0
    calls: int = 0

    def preflight(self, estimate: float) -> None:
        if self.spent + estimate > self.cap:
            raise SpendCapReached(f"spent ${self.spent:.4f}; next call (est. ${estimate:.4f}) would pass ${self.cap:.2f}")

    def charge(self, usd: float) -> None:
        self.spent += usd
        self.calls += 1


class LiveDepthEnv:
    """StepEnvironment over real questions: each round is a real provider call scored by exact match."""

    ep_type = "live"
    verifier_flip_p = 0.0  # exact match: no known flip, nothing to de-bias

    def __init__(self, skeleton: Skeleton, questions: pd.DataFrame, card_costs: Callable[[], dict[str, float]],
                 caller: Callable, spend: Spend, step_types: dict[str, int], recorded: pd.DataFrame | None = None,
                 arms: dict | None = None):
        self.skeleton, self.q, self.card_costs, self.caller, self.spend = skeleton, questions, card_costs, caller, spend
        self.arms = arms or {a.name: a for a in ARMS}
        self.step_types, self.recorded = step_types, recorded
        self.k, self.budget_left, self.agree = 0, float(skeleton.budget_usd), []

    def start(self):
        first = self.skeleton.prompt_ids[0]
        return first, self.q.loc[first, "text"], self.skeleton.budget_usd, len(self.skeleton.prompt_ids)

    def current_step(self) -> StepSpec | None:
        if self.k >= len(self.skeleton.prompt_ids) or self.budget_left <= 0:
            return None
        pid = self.skeleton.prompt_ids[self.k]
        return StepSpec(step_id=f"{self.skeleton.episode_id}-{self.k}", step_key=pid, step_text=self.q.loc[pid, "text"],
                        step_type=self.step_types[self.q.loc[pid, "source"]])

    def play(self, model_id: str, round: int) -> RoundOutcome:
        pid = self.skeleton.prompt_ids[self.k]
        self.spend.preflight(ESTIMATE_FACTOR * self.card_costs()[model_id])
        arm = self.arms[model_id]
        text, tin, tout, treason = self.caller(arm, self.q.loc[pid, "text"])
        cost = call_cost(arm, tin, tout, treason)
        self.spend.charge(cost)
        self.budget_left -= cost
        _, ok = score(text, self.q.loc[pid, "source"], str(self.q.loc[pid, "gold"]))
        if self.recorded is not None and (pid, model_id) in self.recorded.index:
            self.agree.append(bool(self.recorded.loc[(pid, model_id)]) == ok)
        item = TraceItem(model_id=model_id, round=round, verifier_ok=ok, cost_usd=cost, latency_ms=0.0, tokens_out=tout,
                         escalated=round == 2, output_emb_key=None, step_index=self.k, step_emb_key=pid)
        return RoundOutcome(trace_item=item, quality=float(ok), cost_usd=cost, audit_quality=float(ok))

    def end_step(self) -> None:
        self.k += 1


def fake_caller(questions: pd.DataFrame) -> Callable:
    """Dry run: answers every question correctly, reports no tokens (so no spend); no network."""
    by_text = {t: g for t, g in zip(questions["text"], questions["gold"])}

    def call(arm, text):
        return f"Answer: {by_text[text]}", 0, 0, 0

    return call


def run_live(init_dir: Path, data: Data, start_ids: Sequence[str], joiners: list, skeletons: Sequence[Skeleton],
             questions: pd.DataFrame, caller: Callable, spend: Spend, out: Path, cfg: OnlineConfig,
             explore_share: float = 0.2, seed: int = 0, recorded: pd.DataFrame | None = None,
             arms: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    learner, session, pool = start_online(init_dir, out, data, cfg, start_ids=start_ids, seed=seed)
    if joiners:
        pool.add_models(joiners, sim_known=True)
        sync_cards(pool, session, learner)
    step_types = {s: i for i, s in enumerate(("mmlu", "hellaswag", "arc-challenge", "winogrande", "gsm8k"))}
    rng = np.random.default_rng(seed)
    log_f = (out / "log.jsonl").open("a")
    agree, rows, stopped = [], [], None
    try:
        for i, sk in enumerate(skeletons):
            lam_idx = i % len(LAMS)
            mode = "sample" if rng.random() < explore_share else "serve"
            state, version = learner.published()
            env = LiveDepthEnv(sk, questions, lambda: dict(zip(pool.model_ids(), pool.costs())), caller, spend,
                               step_types, recorded, arms)
            try:
                log = run_episode(session, env, lam_idx, mode, pool.bonus, state, version)
            except SpendCapReached as e:
                stopped = str(e)
                break
            agree += env.agree
            pool.record(log)
            sync_cards(pool, session, learner)
            learner.observe(log)  # writes its own "update" rows to the same log.jsonl
            row = {"type": "episode", "episode": i, "lam": LAMS[lam_idx], "mode": mode, "quality": log.quality,
                   "cost_usd": log.cost_usd, "reward": sum(d.reward for d in log.decisions),
                   "policy_version": log.policy_version, "used": [u[0] for u in log.used],
                   "alarms": learner.monitor(log), "spent_total": spend.spent, "ts": time.time()}
            rows.append(row)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()  # the learner appends its update rows to the same file
    finally:
        log_f.close()
        (out / "pool.json").write_text(json.dumps(pool.to_json(), indent=2))  # learner state saved first
    summary = {"episodes": len(rows), "spent_usd": spend.spent, "calls": spend.calls, "cap_usd": spend.cap,
               "stopped": stopped, "updates": learner.updates, "policy_version": learner.version,
               "mean_reward": float(np.mean([r["reward"] for r in rows])) if rows else None,
               "mean_quality": float(np.mean([r["quality"] for r in rows])) if rows else None,
               "live_vs_recorded_agreement": float(np.mean(agree)) if agree else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="depth-pool simulator-RL checkpoint dir")
    ap.add_argument("--join", action="append", default=[], help="option joining live at episode 0 (repeatable)")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--spend-cap-usd", type=float)
    ap.add_argument("--explore-share", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=OnlineConfig.lr)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/online_live"))
    a = ap.parse_args()
    if not a.dry_run and a.spend_cap_usd is None:
        raise SystemExit("a live run needs --spend-cap-usd")
    from eval.online_sim import cards_from_data, probe_card

    summary = json.loads((a.init / "metrics.json").read_text())
    start = load_run_data(summary)
    data = load_data(Path(summary["data_dir"]))  # full pool: joiners' probe cards and world
    start_ids = list(start.world.model_ids)
    known = cards_from_data(data)
    joiners = [probe_card(m, data, [c for c in known if c.model_id in start_ids]) for m in a.join]
    table = pd.read_parquet(Path(summary["data_dir"]).parent / "depth" / "table.parquet").set_index("prompt_id")
    texts = pd.read_parquet(Path(summary["data_dir"]) / "prompts.parquet").set_index("prompt_id")
    test = [sk for sk in load_skeletons("test", data.data_dir) if all(p in table.index for p in sk.prompt_ids)]
    questions = texts.loc[sorted({p for sk in test for p in sk.prompt_ids}), ["text", "source"]].join(table["gold"])
    recorded = pd.read_parquet(Path(summary["data_dir"]).parent / "depth" / "outcomes.parquet").set_index(
        ["prompt_id", "arm"])["correct"]
    caller = fake_caller(questions) if a.dry_run else (lambda arm, text, env=load_env(): call_arm(arm, text, env))
    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-dry" if a.dry_run else "")
    out = a.out / run_id
    result = run_live(a.init, data, start_ids, joiners, test[:a.episodes], questions, caller,
                      Spend(cap=a.spend_cap_usd if a.spend_cap_usd is not None else float("inf")), out,
                      OnlineConfig(lr=a.lr), a.explore_share, a.seed, recorded)
    print(json.dumps(result, indent=2), "\n", out)


if __name__ == "__main__":
    main()
