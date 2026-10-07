"""Policy constants and distributions for Monte Carlo RL (plan D13). Owned by controller/:
no import from measurement/, so only PRICE_INIT, log_price, choose_step and halt_step live
here (LAMS stays in measurement.rl_reward)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PRICE_INIT = 0.1


def log_price(card_costs: np.ndarray, cost_ref: float) -> torch.Tensor:
    """ln(card_cost / cost_ref) per model card, float32 [M]."""
    return torch.log(torch.as_tensor(card_costs, dtype=torch.float32) / cost_ref)


def choose_step(
    policy_logits: torch.Tensor,
    allowed: torch.Tensor,
    mode: str,
    gen: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked categorical over models. policy_logits [B,M], allowed [B,M] bool.

    mode="sample" draws with torch.multinomial(generator=gen); mode="serve" is argmax.
    Disallowed entries are set to -inf before the softmax. Returns (model [B] long,
    logp [B], entropy [B]); logp/entropy keep the autograd graph back to policy_logits.
    Raises ValueError if any row has nothing allowed.
    """
    if not torch.any(allowed, dim=-1).all():
        raise ValueError("choose_step: a row has no allowed model")

    masked = policy_logits.masked_fill(~allowed, float("-inf"))
    logp_all = F.log_softmax(masked, dim=-1)

    if mode == "sample":
        probs = logp_all.exp()
        model = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
    elif mode == "serve":
        model = torch.argmax(masked, dim=-1)
    else:
        raise ValueError(f"choose_step: unknown mode {mode!r}")

    logp = logp_all.gather(1, model[:, None]).squeeze(1)

    # log_softmax of an all-but-one-allowed row is exactly -inf at the disallowed
    # entries. Multiplying that by p=0 is 0*-inf=nan in the backward pass (the
    # forward value is masked back to 0, but torch.where still differentiates both
    # branches, so the nan would otherwise leak into the surviving entries' grad).
    # Zeroing logp_all at disallowed entries *before* multiplying avoids ever
    # forming that product.
    safe_logp_all = torch.where(allowed, logp_all, torch.zeros_like(logp_all))
    p_all = logp_all.exp()
    terms = torch.where(allowed, -(p_all * safe_logp_all), torch.zeros_like(p_all))
    entropy = terms.sum(dim=-1)

    return model.to(torch.long), logp, entropy


def halt_step(
    halt_policy_logit: torch.Tensor,
    policy_logits: torch.Tensor,
    options: torch.Tensor,
    mode: str,
    gen: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bernoulli halt decision, escalating into a masked categorical over options.

    halt_policy_logit [B], policy_logits [B,M], options [B,M] bool. A row with no option
    is a forced stop (model=-1, logp=0, entropy=0, has_choice=False), independent of mode
    or the halt logit. Otherwise p = sigmoid(halt_policy_logit); sample mode escalates iff
    u < p (u drawn once for the batch), serve mode escalates iff p > 0.5. On escalation,
    m2 comes from choose_step restricted to options and logp = logsigmoid(x) + log pi(m2);
    on stop, logp = logsigmoid(-x). entropy = H_Bernoulli(p) + H_Categorical(options).
    Returns (model [B] long, logp [B], entropy [B], has_choice [B] bool).
    """
    has_option = torch.any(options, dim=-1)
    x = halt_policy_logit

    if mode == "sample":
        u = torch.rand(x.shape[0], generator=gen)
        escalate = (u < torch.sigmoid(x)) & has_option
    elif mode == "serve":
        escalate = (x > 0.0) & has_option
    else:
        raise ValueError(f"halt_step: unknown mode {mode!r}")

    # Rows with no option would make choose_step raise; give those rows a safe dummy
    # mask (column 0 allowed) so choose_step never raises, and discard its result for
    # them below via torch.where on has_option.
    dummy = torch.zeros_like(options)
    dummy[:, 0] = True
    safe_options = torch.where(has_option[:, None], options, dummy)
    m2, logp_m2, cat_entropy = choose_step(policy_logits, safe_options, mode=mode, gen=gen)

    log_sigma_x = F.logsigmoid(x)
    log_sigma_neg_x = F.logsigmoid(-x)
    bern_entropy = -(torch.sigmoid(x) * log_sigma_x + torch.sigmoid(-x) * log_sigma_neg_x)

    escalate_logp = log_sigma_x + logp_m2
    stop_logp = log_sigma_neg_x
    logp_choice = torch.where(escalate, escalate_logp, stop_logp)
    entropy_choice = bern_entropy + cat_entropy

    model_choice = torch.where(escalate, m2, torch.full_like(m2, -1))

    zeros = torch.zeros_like(logp_choice)
    logp = torch.where(has_option, logp_choice, zeros)
    entropy = torch.where(has_option, entropy_choice, zeros)
    model = torch.where(has_option, model_choice, torch.full_like(model_choice, -1))

    return model.to(torch.long), logp, entropy, has_option
