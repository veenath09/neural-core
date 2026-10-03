"""Batched rollout of the controller through the simulator (plan D1, D5, D10, D11).

Training and evaluation share this loop. Episodes advance in lockstep: each active
episode makes one forward pass per iteration, either before a round (CHOOSE a model)
or after round 1 (HALT: escalate or stop). Memory rows follow their episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from controller.contracts import MAX_ROUNDS
from controller.features import CHOOSE, HALT, collate, collate_targets, target, view
from data.env import Episode
from data.oracle import halt_label, step_labels
from data.routerbench import OutcomeTable
from eval.policies import EpisodeResult

SOURCES = ("oracle", "random", "model")
SHARES = (0.5, 0.25, 0.25)  # behaviour mixture per step (D1, D11)
RANDOM_ESCALATE_P = 0.3
LOSS_WEIGHTS = {"model": 1.0, "halt": 0.5, "complexity": 0.2}  # model / halt / complexity (spec, Section 4)


@dataclass
class World:
    table: OutcomeTable
    model_ids: list[str]
    card_costs: np.ndarray  # mean cost per call from the model cards
    emb_index: dict[str, int]
    model_index: dict[str, int]


def tau_choice(p: np.ndarray, candidates: Sequence[int], tau: float) -> int:
    """Cheapest candidate with p >= tau, else the most likely to succeed (plan §5)."""
    above = [m for m in candidates if p[m] >= tau]
    return min(above) if above else max(candidates, key=lambda m: p[m])


@dataclass
class RolloutOutput:
    loss: torch.Tensor | None
    parts: dict[str, float]
    results: list[EpisodeResult]
    log: list[dict] = field(default_factory=list)
    residuals: list[list[tuple[str, float]]] = field(default_factory=list)
    decision_states: list[list[torch.Tensor]] = field(default_factory=list)
    decision_actions: list[list[dict[str, int | str | None]]] = field(default_factory=list)


def rollout(
    model,
    episodes: Sequence[Episode],
    world: World,
    *,
    policy: str = "greedy",
    tau: float = 0.5,
    halt_tau: float = 0.5,  # escalate after round 1 when p(escalate) exceeds this
    model_share: bool = True,
    rng: np.random.Generator | None = None,
    bp_steps: int = 5,
    compute_loss: bool = False,
    record: bool = False,
) -> RolloutOutput:
    rng = rng or np.random.default_rng(0)
    device = next(model.parameters()).device
    n = len(episodes)

    states = [ep.initial_state() for ep in episodes]
    written = [0] * n
    source = ["model"] * n
    memory = model.init_memory(n)

    terms: dict[str, list[torch.Tensor]] = {
        "model": [],
        "halt": [],
        "complexity": [],
    }

    log: list[dict] = []
    residuals: list[list[tuple[str, float]]] = []

    # One list per episode.
    # Each controller decision stores its 128-d z_h decision state here.
    decision_states: list[list[torch.Tensor]] = [
        [] for _ in range(n)
    ]
    decision_actions: list[list[dict[str, int | str | None]]] = [
    [] for _ in range(n)
    ]

    while True:
        for i, ep in enumerate(episodes):
            if not states[i].done and states[i].rounds == MAX_ROUNDS:
                states[i] = ep.end_step(states[i])

        active = [
            i for i in range(n)
            if not states[i].done
        ]

        if not active:
            break

        phases = [
            CHOOSE if states[i].rounds == 0 else HALT
            for i in active
        ]

        views = [
            view(
                episodes[i],
                states[i],
                ph,
                written[i],
                world.emb_index,
                world.model_index,
            )
            for i, ph in zip(active, phases)
        ]

        idx = torch.tensor(active, device=device)

        out = model(
            collate(views, device),
            memory.select(idx) if memory is not None else None,
            bp_steps=bp_steps,
            record=record,
        )

        # Save the real controller decision state z_h for each active episode.
        # detach() prevents keeping the current autograd graph.
        # cpu() makes the stored state easy to replay/use later.
        for j, i in enumerate(active):
            decision_states[i].append(
                out.z_h[j].detach().cpu()
            )

        if memory is not None:
            memory = memory.merge(idx, out.memory)

        if compute_loss:
            succ, halt_t, band = collate_targets(
                [
                    target(
                        episodes[i],
                        states[i],
                        ph,
                    )
                    for i, ph in zip(active, phases)
                ],
                device,
            )

            terms["model"].append(
                F.binary_cross_entropy_with_logits(
                    out.model_logits,
                    succ,
                    reduction="none",
                ).mean(1)
            )

            terms["complexity"].append(
                (torch.sigmoid(out.complexity_logit) - band) ** 2
            )

            is_halt = torch.tensor(
                [ph == HALT for ph in phases],
                device=device,
            )

            if is_halt.any():
                terms["halt"].append(
                    F.binary_cross_entropy_with_logits(
                        out.halt_logit[is_halt],
                        halt_t[is_halt],
                        reduction="none",
                    )
                )

        if record:
            residuals.append(out.residuals)

            for j, (i, v) in enumerate(zip(active, views)):
                if v.write and out.stats:
                    newest = states[i].trace[-1]

                    log.append(
                        {
                            "episode_id": episodes[i].skeleton.episode_id,
                            "ep_type": episodes[i].skeleton.ep_type,
                            "step_index": newest.step_index,
                            "round": newest.round,
                            "model_id": newest.model_id,
                            "verifier_ok": newest.verifier_ok,
                            "surprise": out.stats["surprise"][j].item(),
                            "rate": out.stats["rate"][j].item(),
                            "alpha": out.stats["alpha"][j].item(),
                        }
                    )

        p_model = (
            torch.sigmoid(out.model_logits)
            .detach()
            .cpu()
            .numpy()
        )

        p_halt = (
            torch.sigmoid(out.halt_logit)
            .detach()
            .cpu()
            .numpy()
        )

        for j, (i, ph, v) in enumerate(
            zip(active, phases, views)
        ):
            ep = episodes[i]
            s = states[i]

            if v.write:
                written[i] = len(s.trace)

            if ph == CHOOSE:
                if policy == "mixture":
                    src = str(
                        rng.choice(
                            SOURCES,
                            p=SHARES,
                        )
                    )

                    source[i] = (
                        "oracle"
                        if src == "model" and not model_share
                        else src
                    )

                if source[i] == "oracle":
                    m = step_labels(ep, s).oracle_model

                elif source[i] == "random":
                    m = int(
                        rng.integers(ep.n_models)
                    )

                else:
                    affordable = [
                        k
                        for k in range(ep.n_models)
                        if world.card_costs[k]
                        <= s.budget_left
                    ]

                    m = tau_choice(
                        p_model[j],
                        affordable
                        or list(range(ep.n_models)),
                        tau,
                    )
                decision_actions[i].append({
                    "phase": "choose",
                    "model": int(m),
                    "halt": None,})

                states[i] = ep.play(s, m)
                continue

            options = [
                k
                for k in range(
                    s.cur_model + 1,
                    ep.n_models,
                )
                if world.card_costs[k]
                <= s.budget_left
            ]

            m2 = None

            if source[i] == "oracle":
                if halt_label(ep, s):
                    ok = ep.succeeds(s)

                    m2 = next(
                        k
                        for k in ep.escalation_options(s)
                        if ok[k]
                    )

            elif source[i] == "random":
                if (
                    options
                    and rng.random()
                    < RANDOM_ESCALATE_P
                ):
                    m2 = int(
                        rng.choice(options)
                    )

            elif options and p_halt[j] > halt_tau:
                m2 = tau_choice(
                    p_model[j],
                    options,
                    tau,
                )

            decision_actions[i].append({
                "phase": "halt",
                "model": int(m2) if m2 is not None else None,
                "halt": "escalate" if m2 is not None else "stop",})

            states[i] = (
                ep.play(s, m2)
                if m2 is not None
                else ep.end_step(s)
            )

    results = [
        EpisodeResult(
            ep.skeleton.episode_id,
            ep.skeleton.ep_type,
            ep.quality(s),
            s.spent,
        )
        for ep, s in zip(episodes, states)
    ]

    loss = None
    parts = {}

    if compute_loss:
        means = {
            k: (
                torch.cat(v).mean()
                if v
                else torch.zeros(
                    (),
                    device=device,
                )
            )
            for k, v in terms.items()
        }

        loss = sum(LOSS_WEIGHTS[k] * means[k] for k in LOSS_WEIGHTS)

        parts = {
            k: v.item()
            for k, v in means.items()
        }

    return RolloutOutput(
        loss=loss,
        parts=parts,
        results=results,
        log=log,
        residuals=residuals,
        decision_states=decision_states,
        decision_actions=decision_actions,
    )