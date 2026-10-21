"""
Purpose:  Drive a ControllerSession through a StepEnvironment one episode at a time, and replay logged
          episodes through the current weights to get differentiable log-probs for online RL
          (plan Part 2, Task 4).
Owner:    C2 Learning (R R H J Rathnayaka)
Contract: v1
Inputs:   ControllerSession, DecisionLog (controller/session.py); StepEnvironment (measurement/online_env.py);
          episode_log_rewards (measurement/rl_reward.py).
Outputs:  run_episode -> EpisodeLog (decisions with rewards filled in).
          replay -> (RLTrace [n, T], behaviour_logp [n, T]) for train/rl.py's loss.
          OnlineLearner (Task 5): <out_dir>/v{n}.pt per published version, <out_dir>/log.jsonl per update.
Notes:    Decisions alternate CHOOSE (before round 1) and HALT (after round 1); round 2 has no decision.
          Decision t "triggers" one call: CHOOSE triggers round 1, HALT triggers round 2 on escalate
          (cost 0 on stop). Each step's quality goes on its last decision (episode_log_rewards), and
          steps the episode never reached count with quality 0, so the rewards sum to the episode reward
          over the planned n_steps. Replay re-runs the logged views in order with fresh memory, episodes
          batched in lockstep by decision index (shorter episodes drop out), and scores the LOGGED
          action with the logged mask and bonus, so with unchanged weights it reproduces behaviour_logp.
          Live replay has no labels: band comes from the current complexity, band_target is that
          complexity detached (no auxiliary pull), and sup_model / sup_halt stay zero (anchor=False).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from controller.contracts import ModelCard
from controller.features import CHOOSE, HALT, collate
from controller.policy import score_choose, score_halt
from controller.session import ControllerSession, DecisionLog
from measurement.online_env import StepEnvironment
from data.routerbench import OutcomeTable
from measurement.rl_reward import LAMS, episode_log_rewards
from train.rollout import RLTrace, World
from train.train import Data


@dataclass
class EpisodeLog:
    episode_id: str
    ep_type: str
    lam_idx: int
    mode: str
    decisions: list[DecisionLog]
    quality: float  # mean step quality over the planned n_steps (verifier signal live)
    cost_usd: float
    n_steps: int
    policy_version: int
    # (model_id, step_type, round quality, round cost): PoolState.record updates cards and costs from these
    used: list[tuple[str, int, float, float]] = field(default_factory=list)


def _episode_id(env: StepEnvironment) -> str:
    episode = getattr(env, "episode", None)
    if episode is not None:
        return episode.skeleton.episode_id
    return getattr(env, "episode_id", "live")


def _audit(out) -> float:
    return out.quality if out.audit_quality is None else out.audit_quality


def _seed(episode_id: str, policy_version: int) -> int:
    key = f"{episode_id}|{policy_version}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")


def run_episode(session: ControllerSession, env: StepEnvironment, lam_idx: int, mode: str,
                bonus_fn: Callable[[], np.ndarray], state_dict: dict | None, policy_version: int,
                cost_ref: float | None = None) -> EpisodeLog:
    episode_id = _episode_id(env)
    task_key, task_text, budget_usd, expected_steps = env.start()
    session.reset(episode_id, task_key, task_text, budget_usd, expected_steps, lam_idx, mode,
                  state_dict=state_dict, policy_version=policy_version, seed=_seed(episode_id, policy_version))
    round_costs: list[float] = []  # one per decision: the call it triggered
    decision_step: list[int] = []
    step_quality: dict[int, float] = {}  # the verdict the controller saw (monitoring, EpisodeLog.quality)
    step_reward_q: dict[int, float] = {}  # independent audit verdict of the step's final answer (rewards)
    audited: set[int] = set()  # steps whose final round was played (a real, noisy audit verdict)
    used: list[tuple[str, int, float, float]] = []
    step = 0
    while (spec := env.current_step()) is not None:
        first = session.decide(spec.step_id, spec.step_key, spec.step_text, spec.step_type, bonus_fn())
        out = env.play(first.model_id, 1)
        round_costs.append(out.cost_usd)
        decision_step.append(step)
        if out.trace_item is None:  # unaffordable round 1: the episode ends, this step scores 0
            step_quality[step] = step_reward_q[step] = 0.0
            break
        used.append((first.model_id, spec.step_type, out.quality, out.cost_usd))
        quality, audit = out.quality, _audit(out)
        second = session.after_round(out.trace_item, bonus_fn())  # HALT decision
        decision_step.append(step)
        if second is None:
            round_costs.append(0.0)
        else:
            out2 = env.play(second.model_id, 2)
            round_costs.append(out2.cost_usd)
            if out2.trace_item is None:  # unaffordable escalation: the episode ends, this step scores 0
                step_quality[step] = step_reward_q[step] = 0.0
                break
            used.append((second.model_id, spec.step_type, out2.quality, out2.cost_usd))
            session.after_round(out2.trace_item, bonus_fn())  # round 2: recorded, no decision
            quality, audit = out2.quality, _audit(out2)
        step_quality[step], step_reward_q[step] = quality, audit
        audited.add(step)
        session.end_step()
        env.end_step()
        step += 1

    decisions = session.finish()
    if len(decisions) != len(round_costs):
        raise RuntimeError(f"{len(decisions)} decisions but {len(round_costs)} triggered calls")
    n_steps = int(expected_steps)
    qualities = [step_quality.get(k, 0.0) for k in range(n_steps)]
    if cost_ref is None:
        cost_ref = session.cost_ref
    # A verifier that flips the outcome with known probability f gives E[observed] = f + (1 - 2f) * true, which
    # shrinks quality differences while costs stay exact: the learner would optimise as if lambda were
    # lambda / (1 - 2f) (43% higher at f = 0.15) and drift to cheap models. (q - f) / (1 - 2f) is unbiased.
    # Rewards use the audit verdict (independent of the escalation decision), then de-bias it.
    # Only steps whose final round was played and audited carry a noisy verdict to de-bias; a step never
    # reached, or cut short by the budget, scored an exact 0 (no answer), and de-biasing that 0 would charge
    # it -f / (1 - 2f) (-0.21 at f = 0.15).
    flip = float(getattr(env, "verifier_flip_p", 0.0))
    reward_q = [((step_reward_q[k] - flip) / (1 - 2 * flip) if k in audited else 0.0) for k in range(n_steps)]
    rewards = episode_log_rewards(round_costs, reward_q, decision_step, LAMS[lam_idx], cost_ref)
    for d, r in zip(decisions, rewards):
        d.reward = float(r)
    return EpisodeLog(episode_id=episode_id, ep_type=env.ep_type, lam_idx=int(lam_idx), mode=mode,
                      decisions=decisions, quality=float(np.mean(qualities)) if n_steps else 0.0,
                      cost_usd=float(sum(round_costs)), n_steps=n_steps, policy_version=int(policy_version),
                      used=used)


def _pad(rows: list[list[torch.Tensor | None]], n: int, T: int, device) -> torch.Tensor:
    """[n][T] scalars (None past an episode's end) -> [n, T], zeros as padding, graph kept."""
    zero = torch.zeros((), device=device)
    if T == 0:
        return torch.zeros(n, 0, device=device)
    return torch.stack([torch.stack([v if v is not None else zero for v in r]) for r in rows])


def replay(model, logs: Sequence[EpisodeLog], log_price_fn: Callable[[int], torch.Tensor],
           bp_steps: int = 5) -> tuple[RLTrace, torch.Tensor]:
    """Recompute log-probs of the logged actions under the model's current weights (with grad).

    Returns the RLTrace (logp, entropy, complexity keep the graph) and behaviour_logp [n, T]."""
    n = len(logs)
    T = max((len(log.decisions) for log in logs), default=0)
    device = next(model.parameters()).device
    logp = [[None] * T for _ in range(n)]
    entropy = [[None] * T for _ in range(n)]
    complexity = [[None] * T for _ in range(n)]
    memory = model.init_memory(n)
    full_cards = model.cards  # the current pool; older decisions saw only its first pool_size rows
    lam_all = torch.tensor([LAMS[log.lam_idx] for log in logs], dtype=torch.float32, device=device)

    for t in range(T):
        active = [i for i in range(n) if t < len(logs[i].decisions)]
        by_pool: dict[int, list[int]] = {}
        for i in active:
            by_pool.setdefault(logs[i].decisions[t].pool_size, []).append(i)
        for pool_size, rows in by_pool.items():  # one forward per pool size (the pool can grow mid-run)
            idx = torch.tensor(rows, device=device)
            ds = [logs[i].decisions[t] for i in rows]
            try:
                if pool_size != full_cards.shape[0]:
                    model.set_cards(full_cards[:pool_size])
                out = model(collate([d.view for d in ds], device), memory.select(idx) if memory is not None else None,
                            bp_steps=bp_steps, lam=lam_all[idx], log_price=log_price_fn(pool_size).to(device))
            finally:
                model.set_cards(full_cards)
            if memory is not None:
                memory = memory.merge(idx, out.memory)
            bonus = torch.tensor([d.bonus for d in ds], dtype=torch.float32, device=device)
            logits = out.policy_logits + model.heads.log_sharpness.exp() * bonus  # same units as the session
            mask = torch.tensor([d.mask for d in ds], dtype=torch.bool, device=device)
            action = torch.tensor([d.action for d in ds], dtype=torch.long, device=device)
            lp = torch.zeros(len(rows), device=device)
            ent = torch.zeros(len(rows), device=device)
            choose = torch.tensor([d.phase == CHOOSE for d in ds], device=device)
            if choose.any():
                c = choose.nonzero().squeeze(-1)
                lp_c, ent_c = score_choose(logits[c], mask[c], action[c])
                lp = lp.index_put((c,), lp_c)
                ent = ent.index_put((c,), ent_c)
            if (~choose).any():
                h = (~choose).nonzero().squeeze(-1)
                lp_h, ent_h, _ = score_halt(out.halt_policy_logit[h], logits[h], mask[h], action[h])
                lp = lp.index_put((h,), lp_h)
                ent = ent.index_put((h,), ent_h)
            cx = torch.sigmoid(out.complexity_logit)
            for j, i in enumerate(rows):
                logp[i][t], entropy[i][t], complexity[i][t] = lp[j], ent[j], cx[j]

    logp_t, entropy_t, complexity_t = (_pad(x, n, T, device) for x in (logp, entropy, complexity))
    valid = torch.zeros(n, T, dtype=torch.bool)
    choice = torch.zeros(n, T, dtype=torch.bool)
    halt_slot = torch.zeros(n, T, dtype=torch.bool)
    reward = torch.zeros(n, T, dtype=torch.float32)
    behaviour = torch.zeros(n, T, dtype=torch.float32)
    for i, log in enumerate(logs):
        t = len(log.decisions)
        valid[i, :t] = True
        choice[i, :t] = torch.tensor([d.has_choice for d in log.decisions], dtype=torch.bool)
        halt_slot[i, :t] = torch.tensor([d.phase == HALT for d in log.decisions], dtype=torch.bool)
        reward[i, :t] = torch.tensor([d.reward for d in log.decisions], dtype=torch.float32)
        behaviour[i, :t] = torch.tensor([d.behaviour_logp for d in log.decisions], dtype=torch.float32)
    logp_t = logp_t * choice.to(device)  # forced stops carry no log-prob
    band = torch.clamp(torch.round(complexity_t.detach() * 4), 0, 4).long().cpu() * valid
    trace = RLTrace(logp=logp_t, entropy=entropy_t * choice.to(device), complexity=complexity_t, reward=reward,
                    band=band, band_target=complexity_t.detach().cpu() * valid, lam_idx=torch.tensor(
                        [log.lam_idx for log in logs], dtype=torch.long), choice=choice, valid=valid,
                    halt_slot=halt_slot)
    return trace, behaviour


# ---------- pools, wiring ----------

def subset_world(world: World, ids: Sequence[str]) -> World:
    """The simulator world restricted to (and ordered by) model ids."""
    idx = [world.model_ids.index(m) for m in ids]
    t = world.table
    table = OutcomeTable(prompt_ids=t.prompt_ids, step_type=t.step_type, correct=t.correct[:, idx],
                         cost=t.cost[:, idx], band=t.band)
    return World(table, list(ids), np.asarray(world.card_costs)[idx], world.emb_index,
                 {m: i for i, m in enumerate(ids)})


def subset_data(data: Data, ids: Sequence[str]) -> Data:
    idx = [data.world.model_ids.index(m) for m in ids]
    return Data(subset_world(data.world, ids), data.train, data.val, data.embeddings, data.cards[idx], data.n_types,
                data.data_dir, data.exclude_models)


def cards_from_data(data: Data) -> list[ModelCard]:
    return [ModelCard(model_id=m, provider="simulator", tier=i, mean_cost_usd=float(c), price_source="pool data",
                      price_date="", capability=[float(x) for x in data.cards[i]])
            for i, (m, c) in enumerate(zip(data.world.model_ids, data.world.card_costs))]


def start_online(init_dir: Path, out_dir: Path, data, cfg: OnlineConfig, *, start_ids: Sequence[str] | None = None,
                 world=None, canary_skeletons=None, seed: int = 0, flip_p: float = 0.0):
    """Learner, serving session and pool from a simulator-RL checkpoint dir (model.pt, baseline.pt).

    start_ids: the models the checkpoint was trained on (default: the whole pool); world: the simulator world in
    pool order (start models, then any joiners); canary_skeletons default to data.val[:cfg.canary_episodes];
    flip_p: the verifier's known flip rate, for de-biased card updates (0 live). The learner trains its own copy;
    the session serves a separate one, loading published weights at reset; the teacher is the frozen start."""
    from controller.embedding_cache import EmbeddingCache
    from controller.policy import log_price
    from measurement.profile_updater import PRIOR_WEIGHT, ProfileUpdater
    from measurement.rl_reward import cost_ref_for
    from train.baseline import BucketedBaseline
    from train.pool_state import PoolState
    from train.train import load_model

    init_dir = Path(init_dir)
    start_ids = list(start_ids or data.world.model_ids)
    world = world or data.world
    model = load_model(init_dir, subset_data(data, start_ids), strict=False)
    baseline = BucketedBaseline()
    if (init_dir / "baseline.pt").exists():
        baseline.load_state_dict(torch.load(init_dir / "baseline.pt", weights_only=False))
    teacher, serving = copy.deepcopy(model), copy.deepcopy(model)
    known = {c.model_id: c for c in cards_from_data(data)}
    pool = PoolState([known[m] for m in start_ids], ProfileUpdater(prior_weight=PRIOR_WEIGHT), flip_p=flip_p)
    cost_ref = cost_ref_for(data.data_dir)
    if canary_skeletons is None and data.val:
        canary_skeletons = data.val[:cfg.canary_episodes]
    canary = None
    if canary_skeletons:
        canary = make_canary(list(canary_skeletons), lambda: subset_world(world, pool.sim_model_ids()), cost_ref,
                             cards_fn=lambda: pool.capability_tensor()[pool.sim_indices()])
    learner = OnlineLearner(model, teacher, baseline, cfg, canary, Path(out_dir), np.random.default_rng(seed),
                            log_price_fn=lambda m: log_price(pool.costs()[:m], cost_ref))
    session = ControllerSession(serving, pool.costs, cost_ref, EmbeddingCache(serving, dict(world.emb_index), None),
                                pool.model_ids, pool.available, lams=LAMS)
    sync_cards(pool, session, learner)
    return learner, session, pool


def sync_cards(pool, session: ControllerSession, learner=None) -> None:
    """Push the pool's current cards to every model that serves or learns (after add_models / record)."""
    cards = pool.capability_tensor()
    session.model.set_cards(cards)
    if learner is not None:
        learner.model.set_cards(cards)
        learner.teacher.set_cards(cards)


# ---------- Task 5: OnlineLearner (C2-11 replay with importance weights, C2-07 online safety) ----------

@dataclass
class OnlineConfig:
    update_every: int = 16  # new sample-mode episodes per update
    replay_size: int = 48  # extra sample-mode episodes drawn uniformly from the buffer
    buffer_capacity: int = 2048  # FIFO
    lr: float = 1e-5
    beta: float = 0.001  # entropy bonus (fixed)
    clip_grad: float = 1.0
    is_clip: float = 2.0  # importance ratio cap
    distill: float = 0.2  # complexity distillation toward the frozen teacher
    # Optional label-free anchor on decisions: 0.5 * (logp_now - logp_teacher)^2 on the logged actions (k2 KL
    # estimate). Off by default: the shift pilot's drift to cheaper choices came from a biased reward (fixed in
    # run_episode), and the anchor also blocked genuine adaptation (the easy-drift test stayed at 0%).
    kl: float = 0.0
    canary_every: int = 10  # updates
    canary_episodes: int = 500
    canary_tol: float = 0.05
    monitor_window: int = 200
    # published checkpoints kept on disk (oldest deleted); None keeps all. Keeping every v{n}.pt (~3 MB each,
    # one per update) filled the disk during the simulated experiments.
    keep_versions: int | None = 3


def online_loss(trace: RLTrace, behaviour: torch.Tensor, c_teacher: torch.Tensor, baseline, cfg: OnlineConfig,
                logp_teacher: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """-mean(rho * A * logp) - beta * mean(H) + distill * mean_valid((c - c_teacher)^2)
    + kl * mean_choice(0.5 * (logp - logp_teacher)^2), rho capped at is_clip.

    A = G - b[band, lam], normalised over choice slots (population std), as in Part 1."""
    from train.rl import returns_to_go

    device = trace.logp.device
    choice, valid = trace.choice.to(device), trace.valid.to(device)
    reward, band = trace.reward.to(device), trace.band.to(device)
    returns = returns_to_go(reward)
    advantage = returns - baseline.value(band, trace.lam_idx.to(device)[:, None].expand_as(band)).to(device)
    zero = torch.zeros((), device=device)
    if choice.any():
        a = advantage[choice]
        norm = ((advantage - a.mean()) / (a.std(unbiased=False) + 1e-8)).detach()
        raw_rho = torch.exp(trace.logp.detach() - behaviour.to(device))
        rho = torch.clamp(raw_rho, max=cfg.is_clip)
        pg = -(rho * norm * trace.logp)[choice].mean()
        ent = trace.entropy[choice].mean()
        stats = {"mean_rho": float(rho[choice].mean()), "max_rho": float(rho[choice].max()),
                 "clipped_frac": float((raw_rho[choice] > cfg.is_clip).float().mean())}
    else:
        pg, ent = zero, zero
        stats = {"mean_rho": 1.0, "max_rho": 1.0, "clipped_frac": 0.0}
    kl = zero
    if logp_teacher is not None and choice.any():
        kl = (0.5 * (trace.logp - logp_teacher.to(device)) ** 2)[choice].mean()
    distill = ((trace.complexity - c_teacher.to(device)) ** 2)[valid].mean() if valid.any() else zero
    loss = pg - cfg.beta * ent + cfg.distill * distill + cfg.kl * kl
    return loss, stats | {"pg_loss": float(pg.detach()), "distill": float(distill.detach()), "kl": float(kl.detach()),
                          "entropy": float(ent.detach()),
                          "episode_return": float(returns[:, 0].mean()) if returns.shape[1] else 0.0}


class OnlineLearner:
    """Buffers EpisodeLogs and turns them into small policy-gradient updates between episodes.

    Only sample-mode logs feed the loss; serve-mode logs are kept (card updates, monitoring) but never
    trigger or join an update. The learner trains its own model; serving sessions load published()
    weights at reset, so a publish never changes a running episode. log_price_fn(pool_size) gives the
    price vector replay needs for decisions made with that pool size.

    Canary: every canary_every updates the learner and the frozen starting weights (the teacher) are both
    scored by canary() under the same current cards; a drop of more than canary_tol below the teacher at any
    lambda rolls back. Comparing with a reference measured once at the start instead made card updates alone
    trigger rollbacks (30 in 5,000 episodes even at lr 1e-5), which no weight rollback can undo."""

    def __init__(self, model, teacher, baseline, cfg: OnlineConfig, canary, out_dir: Path, rng: np.random.Generator,
                 log_price_fn: Callable[[int], torch.Tensor]):
        self.model, self.teacher, self.baseline, self.cfg = model, teacher, baseline, cfg
        self.canary, self.out_dir, self.rng, self.log_price_fn = canary, Path(out_dir), rng, log_price_fn
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.lr = cfg.lr
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.buffer: deque[EpisodeLog] = deque(maxlen=cfg.buffer_capacity)
        self.new: list[EpisodeLog] = []
        self.updates, self.version = 0, 0
        self.model.train()
        self._published = self._snapshot()
        self.last_good = {"model": self._published, "baseline": copy.deepcopy(baseline.state_dict())}
        self._rewards: deque[float] = deque(maxlen=cfg.monitor_window)
        self._reward_ref: float | None = None

    def _snapshot(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

    def published(self) -> tuple[dict, int]:
        return {k: v.clone() for k, v in self._published.items()}, self.version

    def observe(self, log: EpisodeLog) -> dict | None:
        self.buffer.append(log)
        if log.mode != "sample":
            return None
        self.new.append(log)
        if len(self.new) < self.cfg.update_every:
            return None
        return self._update()

    def _batch(self) -> list[EpisodeLog]:
        new, self.new = self.new, []
        fresh = {id(x) for x in new}
        older = [x for x in self.buffer if x.mode == "sample" and id(x) not in fresh]
        k = min(self.cfg.replay_size, len(older))
        extra = [older[i] for i in self.rng.choice(len(older), size=k, replace=False)] if k else []
        return new + extra

    def _update(self) -> dict:
        from train.rl import returns_to_go

        batch = self._batch()
        self.model.train()
        trace, behaviour = replay(self.model, batch, self.log_price_fn)
        with torch.no_grad():
            teacher_trace = replay(self.teacher, batch, self.log_price_fn)[0]
        loss, stats = online_loss(trace, behaviour, teacher_trace.complexity, self.baseline, self.cfg,
                                  logp_teacher=teacher_trace.logp)
        self.opt.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad))
        self.opt.step()
        if trace.choice.any():
            returns = returns_to_go(trace.reward)
            lam_bt = trace.lam_idx[:, None].expand_as(trace.band)
            self.baseline.update(trace.band[trace.choice], lam_bt[trace.choice], returns[trace.choice])
        self.updates += 1
        alarms = []
        if self.canary is not None and self.updates % self.cfg.canary_every == 0:
            score, reference = self.canary(self.model), self.canary(self.teacher)
            self.model.train()
            stats |= {"canary": score, "canary_reference": reference}
            if any(v < reference[i] - self.cfg.canary_tol for i, v in score.items()):
                self._rollback()
                alarms.append("rollback")
            else:
                self.last_good = {"model": self._snapshot(), "baseline": copy.deepcopy(self.baseline.state_dict())}
        if "rollback" not in alarms:
            self._published = self._snapshot()
        self.version += 1
        torch.save(self._published, self.out_dir / f"v{self.version}.pt")
        if self.cfg.keep_versions is not None:
            stale = self.out_dir / f"v{self.version - self.cfg.keep_versions}.pt"
            if stale.exists():
                stale.unlink()
        row = {"type": "update", "update": self.updates, "loss": float(loss.detach()), "grad_norm": grad_norm,
               "lr": self.lr,
               "policy_version": self.version, "batch": len(batch), "alarms": alarms} | stats
        with (self.out_dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def _rollback(self) -> None:
        self.model.load_state_dict(self.last_good["model"])
        self.baseline.load_state_dict(copy.deepcopy(self.last_good["baseline"]))
        self.lr /= 2
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)  # drop stale moments
        self._published = {k: v.clone() for k, v in self.last_good["model"].items()}

    def monitor(self, log: EpisodeLog) -> list[str]:
        """Rolling live reward; alarm (no rollback: the live signal is noisy) on a drop > canary_tol."""
        self._rewards.append(sum(d.reward for d in log.decisions))
        if len(self._rewards) < self.cfg.monitor_window:
            return []
        mean = float(np.mean(self._rewards))
        if self._reward_ref is None:
            self._reward_ref = mean
            return []
        return ["live_reward_drop"] if mean < self._reward_ref - self.cfg.canary_tol else []


def make_canary(skeletons, world_fn: Callable[[], object], cost_ref: float,
                cards_fn: Callable[[], torch.Tensor] | None = None) -> Callable:
    """Serve-mode val reward at every lambda index on a fixed set of simulator skeletons (reward, not quality:
    high-lambda policies trade quality for cost by design, as in Part 1's guard; review finding 9).

    world_fn rebuilds the simulator world from the non-retired models that have simulator outcomes
    (PoolState.sim_model_ids()); cards_fn gives the matching card rows (pool.capability_tensor()[
    pool.sim_indices()]), swapped in for the canary and restored after. A model with no simulator
    outcomes is judged by live reward, not by the canary."""
    from train.rl import evaluate_rl

    def canary(model) -> dict[int, float]:
        world = world_fn()
        full = model.cards
        if cards_fn is not None:
            model.set_cards(cards_fn())
        try:
            return {i: evaluate_rl(model, skeletons, world, i, cost_ref)["reward"] for i in range(len(LAMS))}
        finally:
            model.set_cards(full)

    return canary
