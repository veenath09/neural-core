"""Controller = encoder -> episode memory -> H/L core -> heads (plan §5, D2, D10, D11, D12).
arch="attn" keeps the six role tokens as a sequence for the gated-attention core."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from controller.contracts import STATE_DIM
from controller.core import HLCore, rms_norm
from controller.encoder import StepEncoder
from controller.features import Batch
from controller.heads import Heads
from controller.memory import EpisodeMemory, MemoryState


@dataclass(frozen=True)
class ControllerConfig:
    d: int = STATE_DIM
    h_cycles: int = 2
    l_cycles: int = 3
    memory: bool = True
    gate: str = "cfc"  # cfc | stateless | const
    gate_units: int = 16
    context_free: bool = False
    arch: str = "mlp"  # mlp (D11) | attn (D12: gated attention over role tokens)
    n_heads: int = 4


CONFIGS = {
    "full": ControllerConfig(),
    "gate_off": ControllerConfig(gate="const"),
    "gate_stateless": ControllerConfig(gate="stateless"),
    "memory_off": ControllerConfig(memory=False),
    # Masking the memory readout is the same as having no memory, so none is built.
    "context_free": ControllerConfig(memory=False, context_free=True),
    "full_attn": ControllerConfig(arch="attn"),
    "memory_off_attn": ControllerConfig(arch="attn", memory=False),
    "context_free_attn": ControllerConfig(arch="attn", memory=False, context_free=True),
}

SLOTS = 6  # task, budget, step, trace-1, trace-2, memory (D12)
STEP_SLOT = 2


@dataclass
class Output:
    model_logits: torch.Tensor  # [B, M]
    halt_logit: torch.Tensor  # [B]
    complexity_logit: torch.Tensor  # [B]
    z_h: torch.Tensor  # [B, d], the DecisionRecord decision_state
    memory: MemoryState | None
    stats: dict[str, torch.Tensor] = field(default_factory=dict)
    residuals: list[tuple[str, float]] = field(default_factory=list)


class Controller(nn.Module):
    def __init__(self, cfg: ControllerConfig, embeddings: torch.Tensor, cards: torch.Tensor, n_types: int):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("emb", embeddings.float(), persistent=False)  # frozen bge-small vectors
        self.register_buffer("cards", cards.float(), persistent=False)  # capability vectors of the pool
        self.encoder = StepEncoder(cfg.d, embeddings.shape[1], n_types, cards.shape[1])
        self.memory = EpisodeMemory(cfg.d, cfg.gate, cfg.gate_units) if cfg.memory else None
        self.core = HLCore(cfg.d, cfg.h_cycles, cfg.l_cycles, arch=cfg.arch, n_heads=cfg.n_heads)
        self.heads = Heads(cfg.d)
        if cfg.arch == "attn":
            self.slot = nn.Parameter(torch.empty(SLOTS, cfg.d))
            nn.init.normal_(self.slot, std=0.02)

    def init_memory(self, batch: int) -> MemoryState | None:
        return self.memory.init_state(batch) if self.memory is not None else None

    def forward(self, b: Batch, memory: MemoryState | None = None, bp_steps: int = 5,
                record: bool = False) -> Output:
        e = self.encoder(b, self.emb, self.cards)
        task, budget, trace, present = e.task, e.budget, e.trace, b.trace_present
        if self.cfg.context_free:  # D10/D11: keep only the current step's own rounds
            present = present & ~b.trace_earlier
            trace = trace * present[..., None]
            task, budget = torch.zeros_like(task), torch.zeros_like(budget)
        h = rms_norm((task + budget) / 2)
        if self.cfg.arch == "attn":
            return self._forward_tokens(b, e, task, budget, trace, present, h, memory, bp_steps, record)
        x = (e.step + trace.sum(1)) / (1 + present.sum(1, keepdim=True).float())
        stats: dict[str, torch.Tensor] = {}
        if self.memory is not None:
            memory, stats = self.memory.write(memory, e.trace[:, 0], b.write, b.write_dt)
            x = x + self.memory.read(memory, h)
        z_h, residuals = self.core(h, x, bp_steps, record)
        model_logits, halt_logit, complexity_logit = self.heads(z_h, e.card)
        return Output(model_logits, halt_logit, complexity_logit, z_h, memory, stats, residuals)

    def _forward_tokens(self, b, e, task, budget, trace, present, h, memory, bp_steps, record) -> Output:
        n = task.shape[0]
        on = torch.ones(n, dtype=torch.bool, device=task.device)
        stats: dict[str, torch.Tensor] = {}
        mem_tok, mem_ok = torch.zeros_like(task), ~on
        if self.memory is not None:
            memory, stats = self.memory.write(memory, e.trace[:, 0], b.write, b.write_dt)
            mem_tok, mem_ok = self.memory.read(memory, h), on
        ctx = ~on if self.cfg.context_free else on
        tokens = torch.stack([task, budget, e.step, trace[:, 0], trace[:, 1], mem_tok], dim=1) + self.slot
        valid = torch.stack([ctx, ctx, on, present[:, 0], present[:, 1], mem_ok], dim=1)
        tokens = tokens * valid[..., None]
        z_seq, residuals = self.core(tokens, tokens, bp_steps, record, key_mask=valid)
        z_h = z_seq[:, STEP_SLOT]
        model_logits, halt_logit, complexity_logit = self.heads(z_h, e.card)
        return Output(model_logits, halt_logit, complexity_logit, z_h, memory, stats, residuals)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
