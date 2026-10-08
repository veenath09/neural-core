"""Controller inputs and training targets from simulator states (plan D2, D4, D10, D11).

Inputs use only what a deployed controller could see: the prompts, the trace with
its noisy verifier signal, and the budget. Ground truth (`cur_correct`, `n_correct`,
the outcome table's correctness) is read only by `target`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch

from controller.contracts import K_TRACE
from data.env import Episode, EnvState
from data.oracle import halt_label

CHOOSE, HALT = 0, 1  # pass before a round (pick a model) / after round 1 (escalate or stop)
BUDGET_FEATS = 6
TRACE_FEATS = 7


def _log_cost(usd: float) -> float:
    return math.log10(usd + 1e-9) / 4 + 1


@dataclass
class View:
    task_row: int
    step_row: int
    step_type: int
    round: int
    phase: int
    budget: list[float]
    trace_row: list[int]
    trace_model: list[int]
    trace_feats: list[list[float]]
    trace_present: list[bool]
    trace_earlier: list[bool]  # from an earlier step: masked in context_free
    write: bool  # a round outcome not yet stored in memory
    write_dt: float


def view(ep: Episode, s: EnvState, phase: int, n_written: int, emb_index: dict[str, int],
         model_index: dict[str, int]) -> View:
    if s.done:
        raise ValueError("cannot build a view of a finished episode")
    sk, n = ep.skeleton, ep.n_steps
    total = sk.budget_usd
    budget = [s.budget_left / total, s.spent / total, s.step_index / n, 1.0 / n, _log_cost(s.budget_left), 0.0]
    rows, models, feats, present, earlier = [], [], [], [], []
    for item in reversed(s.trace[-K_TRACE:]):  # newest first
        same_step = item.step_index == s.step_index
        rows.append(emb_index[item.step_emb_key])
        models.append(model_index[item.model_id])
        feats.append([1.0 if item.verifier_ok else -1.0, float(item.escalated), float(item.round - 1),
                      _log_cost(item.cost_usd), float(same_step), (s.step_index - item.step_index) / n, 1.0])
        present.append(True)
        earlier.append(not same_step)
    while len(rows) < K_TRACE:
        rows.append(0)
        models.append(0)
        feats.append([0.0] * TRACE_FEATS)
        present.append(False)
        earlier.append(False)
    write = len(s.trace) > n_written
    return View(task_row=emb_index[sk.prompt_ids[0]], step_row=emb_index[sk.prompt_ids[s.slot]],
                step_type=int(ep.table.step_type[ep.row(s)]),
                round=s.rounds + 1 if phase == CHOOSE else s.rounds, phase=phase, budget=budget,
                trace_row=rows, trace_model=models, trace_feats=feats, trace_present=present,
                trace_earlier=earlier, write=write, write_dt=1.0 / n if write else 0.0)


@dataclass
class Batch:
    task_row: torch.Tensor
    step_row: torch.Tensor
    step_type: torch.Tensor
    round: torch.Tensor
    phase: torch.Tensor
    budget: torch.Tensor
    trace_row: torch.Tensor
    trace_model: torch.Tensor
    trace_feats: torch.Tensor
    trace_present: torch.Tensor
    trace_earlier: torch.Tensor
    write: torch.Tensor
    write_dt: torch.Tensor

    def __len__(self) -> int:
        return self.task_row.shape[0]


_DTYPES = {"budget": torch.float32, "trace_feats": torch.float32, "write_dt": torch.float32,
           "trace_present": torch.bool, "trace_earlier": torch.bool, "write": torch.bool}


def collate(views: Sequence[View], device: str | torch.device = "cpu") -> Batch:
    return Batch(**{f.name: torch.tensor([getattr(v, f.name) for v in views],
                                         dtype=_DTYPES.get(f.name, torch.long), device=device)
                    for f in fields(Batch)})


@dataclass
class Target:
    succeeds: list[float]
    halt: float
    band: float


def target(ep: Episode, s: EnvState, phase: int) -> Target:
    ok = ep.succeeds(s)
    return Target(succeeds=[float(x) for x in ok], halt=float(halt_label(ep, s)) if phase == HALT else 0.0,
                  band=float((~ok).sum()) / len(ok))


def collate_targets(targets: Sequence[Target], device: str | torch.device = "cpu"):
    return (torch.tensor([t.succeeds for t in targets], dtype=torch.float32, device=device),
            torch.tensor([t.halt for t in targets], dtype=torch.float32, device=device),
            torch.tensor([t.band for t in targets], dtype=torch.float32, device=device))
