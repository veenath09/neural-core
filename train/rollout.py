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
from controller.policy import choose_step, halt_step, log_price as log_price_fn
from data.env import Episode
from data.oracle import halt_label, step_labels
from data.routerbench import OutcomeTable
from eval.policies import EpisodeResult
from measurement.rl_reward import LAMS, step_reward

SOURCES = ("oracle", "random", "model")
SHARES = (0.5, 0.25, 0.25)  # behaviour mixture per step (D1, D11)
RANDOM_ESCALATE_P = 0.3
LOSS_WEIGHTS = {"model": 1.0, "halt": 0.5, "complexity": 0.2}  # model / halt / complexity (spec, Section 4)
RL_POLICIES = ("sample", "serve")


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
class RLTrace:
    """Per-decision Monte Carlo RL trace (plan C2), padded to T = the longest episode.

    logp, entropy, complexity [n,T] keep the autograd graph. reward [n,T] is the
    per-decision reward (measurement.rl_reward.step_reward); band [n,T] is the
    complexity bucket round(sigmoid(complexity_logit)*4) clamped to 0..4; band_target
    [n,T] is features.target(...).band at the decision's pre-action state. lam_idx [n]
    indexes LAMS. choice [n,T] marks slots that counted as a policy choice (CHOOSE,
    and HALT escalate/stop when an option existed); valid [n,T] marks real (non-pad)
    decision slots. Padding is 0 / False everywhere.

    sup_model [n,T] keeps the autograd graph (plan D13 deviation, Task 9): per decision,
    F.binary_cross_entropy_with_logits(model_logits, succeeds, reduction="none").mean(-1)
    where succeeds = features.target(...).succeeds at the decision's pre-action state --
    the same targets supervised training uses. sup_halt [n,T] keeps the autograd graph:
    at HALT decisions, F.binary_cross_entropy_with_logits(halt_logit, halt_label) with
    halt_label = features.target(ep, s, HALT).halt; 0 at CHOOSE. halt_slot [n,T] bool
    marks valid HALT decisions (halt_slot is a subset of valid). These anchor the
    lambda-independent success/halt heads to their supervised labels during RL
    (train/rl.py:rl_loss, keyword anchor=True); traces built without labels (Part 2's
    online replay) may leave sup_model/sup_halt all zero and pass anchor=False.
    """

    logp: torch.Tensor
    entropy: torch.Tensor
    complexity: torch.Tensor
    reward: torch.Tensor
    band: torch.Tensor
    band_target: torch.Tensor
    lam_idx: torch.Tensor
    choice: torch.Tensor
    valid: torch.Tensor
    sup_model: torch.Tensor | None = None
    sup_halt: torch.Tensor | None = None
    halt_slot: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # Traces built without labels (hand-made test traces predating Task 9,
        # or Part 2's online replay) may omit sup_model/sup_halt/halt_slot;
        # default them to all-zero/False so rl_loss(..., anchor=False) -- or an
        # anchor computed over an empty mask -- works unmodified.
        if self.sup_model is None:
            self.sup_model = torch.zeros_like(self.valid, dtype=torch.float32)
        if self.sup_halt is None:
            self.sup_halt = torch.zeros_like(self.valid, dtype=torch.float32)
        if self.halt_slot is None:
            self.halt_slot = torch.zeros_like(self.valid, dtype=torch.bool)


@dataclass
class RolloutOutput:
    loss: torch.Tensor | None
    parts: dict[str, float]
    results: list[EpisodeResult]
    log: list[dict] = field(default_factory=list)
    residuals: list[list[tuple[str, float]]] = field(default_factory=list)
    decision_states: list[list[torch.Tensor]] = field(default_factory=list)
    decision_actions: list[list[dict[str, int | str | None]]] = field(default_factory=list)
    rl: RLTrace | None = None


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
    lam_idx: np.ndarray | None = None,
    cost_ref: float | None = None,
) -> RolloutOutput:
    rng = rng or np.random.default_rng(0)
    device = next(model.parameters()).device
    n = len(episodes)

    rl_mode = policy in RL_POLICIES

    if rl_mode and (lam_idx is None or cost_ref is None):
        raise ValueError(
            f"rollout(policy={policy!r}) requires both lam_idx and cost_ref"
        )

    gen = None
    lam_tensor = None
    log_price_vec = None

    if rl_mode:
        lam_tensor = torch.tensor(LAMS, dtype=torch.float32, device=device)[
            torch.as_tensor(lam_idx, dtype=torch.long, device=device)
        ]
        log_price_vec = log_price_fn(world.card_costs, cost_ref).to(device)

        if policy == "sample":
            gen = torch.Generator()
            gen.manual_seed(int(rng.integers(2**31)))

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

    # Per-episode RL trace lists (sample/serve only), one entry per decision.
    rl_logp: list[list[torch.Tensor]] = [[] for _ in range(n)]
    rl_entropy: list[list[torch.Tensor]] = [[] for _ in range(n)]
    rl_complexity: list[list[torch.Tensor]] = [[] for _ in range(n)]
    rl_reward: list[list[float]] = [[] for _ in range(n)]
    rl_band: list[list[int]] = [[] for _ in range(n)]
    rl_band_target: list[list[float]] = [[] for _ in range(n)]
    rl_choice: list[list[bool]] = [[] for _ in range(n)]
    rl_sup_model: list[list[torch.Tensor]] = [[] for _ in range(n)]
    rl_sup_halt: list[list[torch.Tensor]] = [[] for _ in range(n)]
    rl_halt_slot: list[list[bool]] = [[] for _ in range(n)]

    while True:
        for i, ep in enumerate(episodes):
            if not states[i].done and states[i].rounds == MAX_ROUNDS:
                before = states[i]
                states[i] = ep.end_step(before)

                if rl_mode and rl_reward[i]:
                    d_quality = states[i].n_correct - before.n_correct
                    rl_reward[i][-1] += step_reward(
                        d_quality=d_quality,
                        d_spent=0.0,
                        n_steps=ep.n_steps,
                        lam=LAMS[int(lam_idx[i])],
                        cost_ref=cost_ref,
                    )

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

        pre_states = [states[i] for i in active]

        out = model(
            collate(views, device),
            memory.select(idx) if memory is not None else None,
            bp_steps=bp_steps,
            record=record,
            lam=lam_tensor[idx] if rl_mode else None,
            log_price=log_price_vec if rl_mode else None,
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

        if rl_mode:
            n_models = out.model_logits.shape[1]

            allowed_mask = torch.zeros(
                len(active), n_models, dtype=torch.bool, device=device
            )
            options_mask = torch.zeros(
                len(active), n_models, dtype=torch.bool, device=device
            )

            for j, i in enumerate(active):
                s = pre_states[j]

                row_allowed = [
                    k
                    for k in range(episodes[i].n_models)
                    if world.card_costs[k] <= s.budget_left
                ]

                if not row_allowed:
                    row_allowed = list(range(episodes[i].n_models))

                allowed_mask[j, row_allowed] = True

                # escalation targets: strictly pricier cards (same as range(cur + 1, M) on a cost-ordered pool)
                row_options = [
                    k
                    for k in range(episodes[i].n_models)
                    if (s.cur_model is None or world.card_costs[k] > world.card_costs[s.cur_model])
                    and world.card_costs[k] <= s.budget_left
                ]

                if row_options:
                    options_mask[j, row_options] = True

            rl_model_choice, rl_choose_logp, rl_choose_entropy = choose_step(
                out.policy_logits, allowed_mask, policy, gen
            )
            (
                rl_halt_model,
                rl_halt_logp,
                rl_halt_entropy,
                rl_halt_has_choice,
            ) = halt_step(
                out.halt_policy_logit,
                out.policy_logits,
                options_mask,
                policy,
                gen,
            )

            complexity_vals = torch.sigmoid(out.complexity_logit)
            band_vals = (
                torch.round(complexity_vals * 4)
                .clamp(0, 4)
                .to(torch.long)
            )

            rl_targets = [
                target(episodes[i], pre_states[j], ph)
                for j, (i, ph) in enumerate(zip(active, phases))
            ]

            band_targets = [t.band for t in rl_targets]

            succ_t = torch.tensor(
                [t.succeeds for t in rl_targets],
                dtype=torch.float32,
                device=device,
            )

            sup_model_vals = F.binary_cross_entropy_with_logits(
                out.model_logits,
                succ_t,
                reduction="none",
            ).mean(-1)

            is_halt_slot = torch.tensor(
                [ph == HALT for ph in phases],
                device=device,
            )

            halt_t = torch.tensor(
                [t.halt for t in rl_targets],
                dtype=torch.float32,
                device=device,
            )

            sup_halt_vals = torch.where(
                is_halt_slot,
                F.binary_cross_entropy_with_logits(
                    out.halt_logit,
                    halt_t,
                    reduction="none",
                ),
                torch.zeros(len(active), device=device),
            )

        for j, (i, ph, v) in enumerate(
            zip(active, phases, views)
        ):
            ep = episodes[i]
            s = states[i]

            if v.write:
                written[i] = len(s.trace)

            if rl_mode:
                rl_complexity[i].append(complexity_vals[j])
                rl_band[i].append(int(band_vals[j]))
                rl_band_target[i].append(float(band_targets[j]))
                rl_sup_model[i].append(sup_model_vals[j])
                rl_sup_halt[i].append(sup_halt_vals[j])
                rl_halt_slot[i].append(ph == HALT)

            if ph == CHOOSE:
                if rl_mode:
                    m = int(rl_model_choice[j])

                    decision_actions[i].append({
                        "phase": "choose",
                        "model": int(m),
                        "halt": None,
                    })

                    before = s
                    states[i] = ep.play(s, m)
                    after = states[i]

                    rl_logp[i].append(rl_choose_logp[j])
                    rl_entropy[i].append(rl_choose_entropy[j])
                    rl_choice[i].append(True)
                    rl_reward[i].append(
                        step_reward(
                            d_quality=after.n_correct - before.n_correct,
                            d_spent=after.spent - before.spent,
                            n_steps=ep.n_steps,
                            lam=LAMS[int(lam_idx[i])],
                            cost_ref=cost_ref,
                        )
                    )
                    continue
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
                for k in range(ep.n_models)
                if world.card_costs[k] > world.card_costs[s.cur_model]
                and world.card_costs[k] <= s.budget_left
            ]

            if rl_mode:
                m2 = (
                    int(rl_halt_model[j])
                    if bool(rl_halt_has_choice[j]) and int(rl_halt_model[j]) != -1
                    else None
                )
                has_choice = bool(rl_halt_has_choice[j])

                decision_actions[i].append({
                    "phase": "halt",
                    "model": int(m2) if m2 is not None else None,
                    "halt": "escalate" if m2 is not None else "stop",
                })

                before = s
                states[i] = (
                    ep.play(s, m2)
                    if m2 is not None
                    else ep.end_step(s)
                )
                after = states[i]

                rl_logp[i].append(rl_halt_logp[j])
                rl_entropy[i].append(rl_halt_entropy[j])
                rl_choice[i].append(has_choice)
                rl_reward[i].append(
                    step_reward(
                        d_quality=after.n_correct - before.n_correct,
                        d_spent=after.spent - before.spent,
                        n_steps=ep.n_steps,
                        lam=LAMS[int(lam_idx[i])],
                        cost_ref=cost_ref,
                    )
                )
                continue

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

    rl_trace = None

    if rl_mode:
        T = max((len(x) for x in rl_logp), default=0)

        logp = torch.zeros(n, T, device=device)
        entropy = torch.zeros(n, T, device=device)
        complexity = torch.zeros(n, T, device=device)
        reward = torch.zeros(n, T, dtype=torch.float32)
        band = torch.zeros(n, T, dtype=torch.long)
        band_target = torch.zeros(n, T, dtype=torch.float32)
        choice = torch.zeros(n, T, dtype=torch.bool)
        valid = torch.zeros(n, T, dtype=torch.bool)
        sup_model = torch.zeros(n, T, device=device)
        sup_halt = torch.zeros(n, T, device=device)
        halt_slot = torch.zeros(n, T, dtype=torch.bool)

        for i in range(n):
            t = len(rl_logp[i])

            if t == 0:
                continue

            logp[i, :t] = torch.stack(rl_logp[i])
            entropy[i, :t] = torch.stack(rl_entropy[i])
            complexity[i, :t] = torch.stack(rl_complexity[i])
            reward[i, :t] = torch.tensor(rl_reward[i], dtype=torch.float32)
            band[i, :t] = torch.tensor(rl_band[i], dtype=torch.long)
            band_target[i, :t] = torch.tensor(rl_band_target[i], dtype=torch.float32)
            choice[i, :t] = torch.tensor(rl_choice[i], dtype=torch.bool)
            valid[i, :t] = True
            sup_model[i, :t] = torch.stack(rl_sup_model[i])
            sup_halt[i, :t] = torch.stack(rl_sup_halt[i])
            halt_slot[i, :t] = torch.tensor(rl_halt_slot[i], dtype=torch.bool)

        rl_trace = RLTrace(
            logp=logp,
            entropy=entropy,
            complexity=complexity,
            reward=reward,
            band=band,
            band_target=band_target,
            lam_idx=torch.as_tensor(lam_idx, dtype=torch.long),
            choice=choice,
            valid=valid,
            sup_model=sup_model,
            sup_halt=sup_halt,
            halt_slot=halt_slot,
        )

    return RolloutOutput(
        loss=loss,
        parts=parts,
        results=results,
        log=log,
        residuals=residuals,
        decision_states=decision_states,
        decision_actions=decision_actions,
        rl=rl_trace,
    )