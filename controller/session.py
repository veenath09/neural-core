"""Stateful per-episode controller session for live serving (plan Part 2 Task 2).

One decision at a time, mirroring train.rollout's sample/serve branch decision for
decision: a CHOOSE pass before round 1, a HALT pass after round 1, nothing after
round 2. Every pass is logged with what is needed to recompute its log-probability
later (view, mask, bonus, action), for importance-weighted replay.

controller/ must not import measurement/ or train/, so the lambda grid comes in as
`lams` (callers pass measurement.rl_reward.LAMS)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from controller.contracts import MAX_ROUNDS, Strategy, TraceItem
from controller.embedding_cache import EmbeddingCache
from controller.features import CHOOSE, HALT, View, build_view, collate
from controller.model import Controller
from controller.policy import choose_step, halt_step, log_price, score_choose

MODES = ("sample", "serve")


@dataclass
class DecisionLog:
    phase: int
    view: View
    lam_idx: int
    mask: list[bool]  # allowed models at CHOOSE, escalation options at HALT
    bonus: list[float]
    action: int  # model index; -1 = stop
    behaviour_logp: float
    has_choice: bool
    complexity: float
    policy_version: int
    pool_size: int
    reward: float = 0.0  # filled in by run_episode


class ControllerSession:
    def __init__(self, model: Controller, pool_costs: Callable[[], np.ndarray], cost_ref: float,
                 emb: EmbeddingCache, model_ids: Callable[[], list[str]],
                 available: Callable[[], np.ndarray], lams: Sequence[float], bp_steps: int = 5):
        self.model = model
        self.pool_costs = pool_costs
        self.cost_ref = cost_ref
        self.emb = emb
        self.model_ids = model_ids
        self.available = available
        self.lams = tuple(lams)
        self.bp_steps = bp_steps
        self._active = False

    # ---- episode lifecycle -------------------------------------------------

    def reset(self, episode_id: str, task_key: str, task_text: str | None, budget_usd: float,
              expected_steps: int, lam_idx: int, mode: str, state_dict: dict | None = None,
              policy_version: int = 0, seed: int = 0) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if not 0 <= lam_idx < len(self.lams):
            raise ValueError(f"lam_idx {lam_idx} outside 0..{len(self.lams) - 1}")
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.episode_id = episode_id
        self.task_row = self.emb.row(task_key, task_text)
        self.budget_total = float(budget_usd)
        self.budget_left = float(budget_usd)
        self.spent = 0.0
        self.n_steps = int(expected_steps)
        self.lam_idx = int(lam_idx)
        self.mode = mode
        self.policy_version = int(policy_version)
        self.gen = torch.Generator().manual_seed(seed) if mode == "sample" else None
        self.memory = self.model.init_memory(1)
        self.trace: list[TraceItem] = []
        self.n_written = 0
        self.step_index = 0
        self.rounds = 0
        self.step = None  # (step_id, step_key, step_row, step_type) of the open step
        self.cur_model: int | None = None
        self.logs: list[DecisionLog] = []
        self._active = True

    def decide(self, step_id: str, step_key: str, step_text: str | None, step_type: int,
               bonus: np.ndarray) -> Strategy:
        """CHOOSE pass: the round-1 model for a new step."""
        self._check_active()
        if self.step is not None:
            raise RuntimeError("decide() called before end_step() of the previous step")
        self.step = (step_id, step_key, self.emb.row(step_key, step_text), int(step_type))
        self.rounds = 0
        costs, avail = self._pool()
        allowed = avail & (costs <= self.budget_left)
        if not allowed.any():  # nothing affordable: fall back to every available model
            allowed = avail.copy()
        out, view, logits = self._forward(CHOOSE, bonus, costs)
        allowed_t = torch.as_tensor(allowed, device=self.device)[None]
        with torch.no_grad():
            m, logp, _ = choose_step(logits, allowed_t, self.mode, self.gen)
        m = int(m[0])
        c = self._log(CHOOSE, view, allowed, bonus, m, float(logp[0]), True, out)
        self.cur_model = m
        self.issued = self.model_ids()[m]
        return Strategy(step_id=step_id, round=1, model_id=self.model_ids()[m], halt="stop",
                        complexity=c, confidence=float(torch.exp(logp[0])))

    def after_round(self, item: TraceItem, bonus: np.ndarray) -> Strategy | None:
        """Record a finished round. After round 1, run the HALT pass: None to stop, or the
        round-2 Strategy to escalate. After round 2, only record and return None."""
        self._check_active()
        if self.step is None:
            raise RuntimeError("after_round() called with no open step")
        if self.rounds >= MAX_ROUNDS:
            raise RuntimeError(f"no rounds left in this step (max {MAX_ROUNDS})")
        step_id, step_key, _, _ = self.step
        if item.model_id != getattr(self, "issued", item.model_id):
            raise ValueError(f"round played {item.model_id!r}, but the session issued {self.issued!r}")
        item = replace(item,
                       step_index=self.step_index if item.step_index is None else item.step_index,
                       step_emb_key=step_key if item.step_emb_key is None else item.step_emb_key)
        self.trace.append(item)
        self.spent += item.cost_usd
        self.budget_left -= item.cost_usd
        self.rounds += 1
        self.cur_model = self.model_ids().index(item.model_id)
        if self.rounds >= MAX_ROUNDS:
            return None

        costs, avail = self._pool()
        # escalation is to a strictly pricier card, not a higher index: joiners are appended whatever their price
        options = avail & (costs <= self.budget_left) & (costs > costs[self.cur_model])
        out, view, logits = self._forward(HALT, bonus, costs)
        options_t = torch.as_tensor(options, device=self.device)[None]
        with torch.no_grad():
            m2, logp, _, has_choice = halt_step(out.halt_policy_logit, logits, options_t, self.mode, self.gen)
        m2 = int(m2[0])
        c = self._log(HALT, view, options, bonus, m2, float(logp[0]), bool(has_choice[0]), out)
        if m2 < 0:
            return None
        with torch.no_grad():
            logp_m2, _ = score_choose(logits, options_t, torch.tensor([m2], device=self.device))
        self.issued = self.model_ids()[m2]
        return Strategy(step_id=step_id, round=2, model_id=self.model_ids()[m2], halt="stop",
                        complexity=c, confidence=float(torch.exp(logp_m2[0])))

    def end_step(self) -> None:
        self._check_active()
        if self.step is None or self.rounds == 0:
            raise RuntimeError("end_step() called before any round of the step was recorded")
        self.step_index += 1
        self.rounds = 0
        self.step = None
        self.cur_model = None

    def finish(self) -> list[DecisionLog]:
        self._check_active()
        self._active = False
        return list(self.logs)

    # ---- internals ---------------------------------------------------------

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("no active episode; call reset() first")

    def _pool(self) -> tuple[np.ndarray, np.ndarray]:
        costs = np.asarray(self.pool_costs(), dtype=float)
        avail = np.asarray(self.available(), dtype=bool)
        if costs.shape != avail.shape:
            raise ValueError(f"pool_costs {costs.shape} and available {avail.shape} disagree")
        return costs, avail

    def _forward(self, phase: int, bonus: np.ndarray, costs: np.ndarray):
        _, _, step_row, step_type = self.step
        ids = self.model_ids()
        view = build_view(task_row=self.task_row, step_row=step_row, step_type=step_type, phase=phase,
                          rounds=self.rounds, budget_total=self.budget_total, budget_left=self.budget_left,
                          spent=self.spent, step_index=self.step_index, n_steps=self.n_steps,
                          trace=self.trace, n_written=self.n_written, emb_index=self.emb.emb_index,
                          model_index={m: i for i, m in enumerate(ids)})
        lam = torch.tensor([self.lams[self.lam_idx]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(collate([view], self.device), self.memory, bp_steps=self.bp_steps,
                             lam=lam, log_price=log_price(costs, self.cost_ref).to(self.device))
            bonus_t = torch.as_tensor(np.asarray(bonus, dtype=np.float32), device=self.device)
            if bonus_t.shape != (out.policy_logits.shape[1],):
                raise ValueError(f"bonus shape {tuple(bonus_t.shape)} != pool size {out.policy_logits.shape[1]}")
            # bonus in pre-sharpness logit units: policy = k * (logits + bonus) (review finding 8)
            logits = out.policy_logits + self.model.heads.log_sharpness.exp() * bonus_t[None]
        self.memory = out.memory
        if view.write:
            self.n_written = len(self.trace)
        return out, view, logits

    def _log(self, phase, view, mask, bonus, action, logp, has_choice, out) -> float:
        c = float(torch.sigmoid(out.complexity_logit[0]))
        self.logs.append(DecisionLog(phase=phase, view=view, lam_idx=self.lam_idx,
                                     mask=[bool(x) for x in mask],
                                     bonus=[float(x) for x in np.asarray(bonus)], action=action,
                                     behaviour_logp=logp, has_choice=has_choice, complexity=c,
                                     policy_version=self.policy_version, pool_size=len(mask)))
        return c
