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

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from controller.features import CHOOSE, HALT, collate
from controller.policy import score_choose, score_halt
from controller.session import ControllerSession, DecisionLog
from measurement.online_env import StepEnvironment
from measurement.rl_reward import LAMS, episode_log_rewards
from train.rollout import RLTrace


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
    used: list[tuple[str, int, float]] = field(default_factory=list)  # (model_id, step_type, round quality)


def _episode_id(env: StepEnvironment) -> str:
    episode = getattr(env, "episode", None)
    if episode is not None:
        return episode.skeleton.episode_id
    return getattr(env, "episode_id", "live")


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
    step_quality: dict[int, float] = {}
    used: list[tuple[str, int, float]] = []
    step = 0
    while (spec := env.current_step()) is not None:
        first = session.decide(spec.step_id, spec.step_key, spec.step_text, spec.step_type, bonus_fn())
        out = env.play(first.model_id, 1)
        round_costs.append(out.cost_usd)
        decision_step.append(step)
        if out.trace_item is None:  # unaffordable round 1: the episode ends, this step scores 0
            step_quality[step] = 0.0
            break
        used.append((first.model_id, spec.step_type, out.quality))
        quality = out.quality
        second = session.after_round(out.trace_item, bonus_fn())  # HALT decision
        decision_step.append(step)
        if second is None:
            round_costs.append(0.0)
        else:
            out2 = env.play(second.model_id, 2)
            round_costs.append(out2.cost_usd)
            if out2.trace_item is None:  # unaffordable escalation: the episode ends, this step scores 0
                step_quality[step] = 0.0
                break
            used.append((second.model_id, spec.step_type, out2.quality))
            session.after_round(out2.trace_item, bonus_fn())  # round 2: recorded, no decision
            quality = out2.quality
        step_quality[step] = quality
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
    rewards = episode_log_rewards(round_costs, qualities, decision_step, LAMS[lam_idx], cost_ref)
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
    lam_all = torch.tensor([LAMS[log.lam_idx] for log in logs], dtype=torch.float32, device=device)

    for t in range(T):
        active = [i for i in range(n) if t < len(logs[i].decisions)]
        by_pool: dict[int, list[int]] = {}
        for i in active:
            by_pool.setdefault(logs[i].decisions[t].pool_size, []).append(i)
        for pool_size, rows in by_pool.items():  # one forward per pool size (the pool can grow mid-run)
            idx = torch.tensor(rows, device=device)
            ds = [logs[i].decisions[t] for i in rows]
            out = model(collate([d.view for d in ds], device), memory.select(idx) if memory is not None else None,
                        bp_steps=bp_steps, lam=lam_all[idx], log_price=log_price_fn(pool_size).to(device))
            if memory is not None:
                memory = memory.merge(idx, out.memory)
            bonus = torch.tensor([d.bonus for d in ds], dtype=torch.float32, device=device)
            logits = out.policy_logits + bonus
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
