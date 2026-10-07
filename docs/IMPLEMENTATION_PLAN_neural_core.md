# Controller Prototype: Implementation Plan

**Project:** J26-DS-359, C1 Adaptive Orchestration Controller
**Owner:** V M Wickramaarachchi (IT23230460)
**Timebox:** 15 working days (3 weeks). A 2-week compression is given at the end.
**Goal:** Show, on data, that a controller which reads workflow state beats one that only reads the query on the quality–cost trade-off, and measure what the liquid-gated memory contributes.

---

## 1. Scope

| In | Out (next phase) |
|---|---|
| Encoder with a frozen `bge-small-en-v1.5` | Calibration under workload shift |
| H/L core with MLP blocks (2 H cycles × 3 L steps) | Attention blocks |
| L = TTT-Linear memory + CfC gate | TTT-MLP, gate tuning |
| Model head over a 4-model pool | Tools, memory, depth, verification and topology heads (fixed defaults) |
| Simple halting: stop or escalate | Full cost-aware halting values |
| Supervised warm start, then RL fine-tune in the simulator (see `RL_DESIGN.md`) | RL on live episodes, offline RL from logs |
| Synthetic episodes chained from RouterBench | Team episode generator, live API runs |
| 4 baselines, 2 ablations, 3 seeds | Specialisation probe, TwinRouterBench training |

**Claim at the end:** a working prototype evaluated in a simulated workflow environment built from RouterBench, reporting whether conditioning on workflow state improves the quality–cost trade-off over a context-free router, and the contribution of the liquid-gated memory.

---

## 1a. Design decisions v1.3 (supersede later sections where they conflict)

Settled after code review, before Week 1 coding. Where §2–§8 disagree with this section, this section wins.

### D1. Episodes are skeletons; outcomes are rolled out live

Parquet stores **policy-independent skeletons** only: episode id, type, ordered prompt ids, starting budget. Outcomes depend on earlier choices, so they are computed at train time by `data/env.py` from table lookups.

**Behaviour policy for supervised training** (per step, sampled):

| Source | Share |
|---|---|
| Episode oracle (D6) | 50% |
| Uniform random over pool | 25% |
| Current model (from epoch 2; oracle in epoch 1) | 25% |

At every visited step, `succeeds[m]` is computed **for every pool model given the realised history**, so labels are always defined, and the current-model share limits exposure bias. Test episodes are rolled out by the policy under evaluation.

### D2. What the memory remembers

Two loops, two kinds of state:

| State | Lives for | Updated | What it is |
|---|---|---|---|
| `W` (TTT-Linear fast weights, 128×128) | **One episode** (reset at episode start) | Once per decision, on the newest trace token | Compressed history of this task |
| `z_L`, `z_H` (HRM reasoning state) | **One decision** | 2 cycles × 3 L-steps | Working state for this choice |

Per decision:

1. **Memory write (at most once).** Store the newest round outcome in `W`. Exact equations, gate and timing in **D11** (superseded text: `surprise = obs − W·key(obs)`, "TTT-Linear closed form"). Log `‖surprise‖`, `rate` and `α`.
2. **Reasoning (2 × 3).** Every L-step sees the same input: pooled L-tokens + memory readout `W·query(z_H)`. `z_L ← L_block(z_L + z_H + input)`. After 3 L-steps, `z_H ← H_block(z_H + z_L)`. `z_L` restarts from a fixed initial state each decision (D11).
3. Heads read the final `z_H`.

**Trace length k = 2** (not 4). With episodes of 3–6 steps, k = 4 shows almost the whole history explicitly and the memory has nothing to add; k = 2 forces older history through `W`.

Ablations then mean:

| Config | Change |
|---|---|
| `gate_off` | `rate` = one learned constant |
| `gate_stateless` | `rate` = σ(MLP(gate input)): same input as the CfC gate, no state, no time (D11) |
| `memory_off` | No `W`; memory readout = 0 |
| `context_free` | Earlier steps' trace, task token, budget token **and** memory readout masked (D10, D11) |

### D3. Outcome rules

- **Retry:** the failed item is the next step. `succeeds[m]` = table value.
- **Error propagation:** after a failure by a model of tier *j* at step *k*, at step *k+1* `succeeds[m] = table[m] AND tier(m) > j`.
- **Budget:** a round whose cost exceeds `budget_left` is not run; that step scores 0 and the episode ends.
- **Sources:** keep only RouterBench sources with 0/1 scores. Drop MT-Bench and any source whose scores are not binary (check in Day 2). `n_step_types` = number of kept sources.

### D4. The controller never sees ground truth

At runtime nobody knows whether an answer is correct. The trace therefore carries a **noisy verifier signal**: `verifier_ok` = true correctness, flipped with probability 0.15 (seeded). Quality and reward use true correctness; the controller only ever sees `verifier_ok`. State this as a limitation in the write-up.

### D5. Halting = one escalation per step

- **Escalate** = re-run the **same step** with a strictly more expensive pool model, charged again. At most 2 rounds per step.
- **Halt head target (BCE):** 1 if the round's answer is wrong **and** some strictly more expensive, affordable model succeeds; else 0.
- **Inference:** model choice by the τ rule; after the round, escalate iff halt head p > 0.5 and budget allows. The escalation model is chosen by the τ rule over the more expensive models.

### D6. Oracles

- **Per-step oracle** (supervised label): cheapest model with `succeeds = 1`, else cheapest.
- **Episode oracle** (ceiling on every plot, and 50% of the behaviour policy): exact search over model choices including escalations, maximising episode quality, ties broken by lower cost; for RL plots, maximising quality − λ·cost. ≤ 6 steps × 4 models × 2 rounds is small enough to brute-force with memoisation on (step, budget left, last failing tier).
- Test asserts: episode oracle ≥ every policy on every episode.

### D7. Heads and losses

| Head | Target | Loss |
|---|---|---|
| Model (per-model success) | `succeeds[m]` | BCE |
| Halting | D5 | BCE |
| Complexity | difficulty band ÷ 4 (band = pool models failing this step given history) | MSE |
| Confidence | — | **No separate head.** Confidence = temperature-calibrated success probability of the chosen model |

### D8. Contracts v1

```python
# controller/contracts.py
@dataclass
class TraceItem:
    model_id: str; round: int
    verifier_ok: bool            # noisy, D4
    cost_usd: float; latency_ms: float; tokens_out: int
    escalated: bool
    output_emb_key: str | None   # cache key; None in prototype

@dataclass
class TaskRep:
    episode_id: str; step_id: str; round: int
    task_emb_key: str
    step_type: int; step_emb_key: str
    trace: list[TraceItem]       # last k = 2
    budget_total_usd: float; budget_left_usd: float; spent_usd: float
    step_index: int; expected_steps: int
    lam: float | None = None     # RL only
    version: str = "v1"

@dataclass
class Strategy:
    step_id: str; round: int
    model_id: str
    halt: Literal["stop", "escalate"]
    tools: list[str] = field(default_factory=list)
    memory: Literal["none", "session", "retrieval"] = "none"
    depth: Literal["low", "medium", "high"] = "medium"
    verification: Literal["none", "self_check", "cross_model"] = "none"
    topology: Literal["single", "parallel", "sub_agents"] = "single"
    complexity: float = 0.0; confidence: float = 0.0
    version: str = "v1"

@dataclass
class DecisionRecord:
    step_id: str; round: int
    strategy: Strategy
    log_probs: dict[str, float]; entropy: dict[str, float]
    complexity_band: int         # 0..4
    lam: float | None
    decision_state: list[float]  # len 128
    state_version: str = "v1"
```

`ModelCard` as in `model_onboarding.md`. Rewards are **not** in `DecisionRecord`; they are joined later on `(step_id, round)`.

### D10. Review fixes (v1.1 → v1.2)

- **Control-row fairness.** `context_free` masks earlier steps' trace items, the budget token and the memory readout, but still sees the **current step's own round-1 result**. Escalating within a step is not workflow context; masking it would hand `full` a gain on plain random chains.
- **Halt timing.** A `Strategy` is emitted per round. `model_id` comes from the forward pass before the round; `halt` is filled from a second forward pass on the post-round `TaskRep` (trace includes that round). `halt = "escalate"` is invalid in round `MAX_ROUNDS = 2`.
- **Retry episodes.** The skeleton lists `n` prompts; the episode always has `n` steps. In a retry episode, if the step's final round is flagged by the verifier (`verifier_ok` false) and the item is not already a retry, the next step is the same item (marked retried); otherwise the next listed prompt. Items are retried at most once. Deciding on the verifier, not on true correctness, keeps a returning prompt from revealing ground truth (D4).
- **Step outcome.** A step's outcome is its last round's answer. Propagation uses the tier of the model that gave that final wrong answer.
- **TTT surprise.** `surprise = value(obs) − W·key(obs)`. The gate rate is one scalar per decision. At an episode's first decision there is no outcome to store, so the memory update is skipped. Gradients flow through `W` across the whole episode.
- **Sources (checked on the data).** Kept: MMLU (grouped), HellaSwag, ARC-Challenge, Winogrande, MBPP → 5 step types. Dropped: grade-school-math (0.25-step scores), MT-Bench (graded), small miscellaneous evals. Duplicate prompt texts (16) are removed before splitting.
- **Pool.** mistral-7b-chat, mixtral-8x7b-chat, gpt-3.5-turbo-1106, gpt-4-1106-preview (mean cost per call rises with tier). Costs are RouterBench's per-prompt `total_cost` (API prices for proprietary models, Together AI for open models; the paper gives no pricing date).
- **ModelCard v1** (until `model_onboarding.md` lands): `model_id, provider, tier, mean_cost_usd, price_source, price_date, capability` where `capability` = train accuracy per step type + log10(mean cost).
- **Sequencing.** `env.py` and the episode oracle move to Day 5 (D1 needs them for the behaviour policy).

### D11. Reference-grounded design (v1.2 → v1.3)

Checked against the code and papers of HRM-Text (`models/baselines/hrm_nocarry_bp_warmup.py`, arXiv 2605.20613), HRM (`models/hrm/hrm_act_v1.py`, arXiv 2506.21734), TRM (`models/recursive_reasoning/trm.py`, arXiv 2510.04871), ttt-lm-pytorch (`ttt.py`, arXiv 2407.04620), titans-pytorch (`neural_memory.py`, arXiv 2501.00663), ncps (`torch/cfc_cell.py`) and CfC (arXiv 2106.13898), and RouterBench (arXiv 2403.12031). Where D2 or §5 disagree, this section wins.

**Core (`controller/core.py`)**

- Each module = 2 pre-norm **SwiGLU MLP** blocks, γ-free RMSNorm before each block and at the module exit (MagicNorm, HRM-Text §2.1.1). No attention: each module holds one vector, and TRM finds MLPs better on small fixed contexts.
- `z_L` starts from a **fixed buffer** (truncated normal, std 1), as in HRM, HRM-Text and TRM. `z_H` starts from `h = RMSNorm(mean(H tokens))`.
- L-step: `z_L ← L(z_L + z_H + x)` with `x` injected every step (HRM/TRM; HRM-Text injects only through `z_H⁰`). H-step: `z_H ← H(z_H + z_L + h)`. Re-injecting `h` is our deviation: under truncated backprop it is the only path by which the task and budget encoders receive gradient.
- Gradient horizon = HRM-Text staircase: `bp = 2 + ⌊min(1, step / (0.2·total)) · 3⌋` module calls, split `H_bp = min(2, bp − 1)`, `L_bp = bp − H_bp`; earlier calls run without grad. TRM shows a 1-step gradient is weak, so a run may start at `bp_min = 3`.
- **EMA** of weights with warm-up, decay = min(0.999, (1+t)/(10+t)) (0.999 as in TRM; the warm-up because our runs are ~1.6k steps). Validation and evaluation use the EMA weights.

**Memory (`controller/memory.py`)**, TTT-Linear inner model with Titans-style forgetting, no momentum:

```
k = ‖W_k o‖⁻¹ W_k o        v = W_v o        q = ‖W_q h‖⁻¹ W_q h
s = v − W k                                   (surprise)
u = [s/‖s‖, log‖s‖, o]
r = gate(u, Δt)            ∈ (0, 1)
α = σ(w_α·[log‖s‖, o] + b_α),  b_α = −4        (forgetting, starts ≈ 0)
W ← W₀ + (1 − α)(W − W₀) + r · s kᵀ           (W₀ learned, init 0)
readout m = RMSNorm(W q)
```

With unit-norm `k`, one write moves `W k` exactly a fraction `r` of the way to `v` (`s' = (1 − r) s` when `α = 0`), so no clipping is needed. This is one explicit delta-rule step, not TTT's dual form.

- `o` = the newest round's trace token. A pass writes when a round outcome has not been stored yet: the post-round pass stores that round; a round-2 outcome is stored at the next step's first pass; the episode's first pass stores nothing.
- `Δt = 1 / n_steps` per write: time is measured as the fraction of the episode, so episodes of different lengths move the CfC clock at different speeds.
- Neither TTT nor Titans feeds the error into the rate (both use the input); gating on `s` is our addition. `‖∇_W‖ = ‖s‖` here, so the gate sees both loss and gradient size.

**Gate.** `full` uses a **CfC cell** (Eq. 10, ncps default mode, re-implemented): 16 units, backbone 32 (LeCun-tanh), state reset at episode start and carried across writes, time input `Δt`; `r = σ(w_r·c + b_r)`. `gate_stateless` gets the same `u` through σ(MLP) with no state or time. `gate_off` uses `r = σ(c)`. The thesis may call the memory "liquid-gated" only if `full` beats `gate_stateless`.

**Heads.** Model head = additive match, `logit_m = w·SiLU(A z_H + card_m) + b`, so the pool is a list. Halt head = BCE on the D5 target (TRM-style; no Q-learning, no extra forward pass).

**Inputs**

- `TraceItem` gains optional `step_index` and `step_emb_key` (the prompt of that round's step), so the controller can see that an item is a retry.
- Task token = embedding of the episode's first prompt. `context_free` also masks it, because it carries information about another step.
- Model choice and escalation consider only models whose card mean cost ≤ `budget_left`; the controller cannot know a call's exact cost in advance.

**Behaviour policy (revises D1).** The oracle share uses the **per-step** oracle (cheapest succeeding model; escalate iff the halt label is 1), not the episode oracle: re-planning after every deviation costs ~15 ms per step. The episode oracle stays the evaluation ceiling.

**Configs** are Python presets in `controller/model.py`: `full`, `gate_off`, `gate_stateless`, `memory_off`, `context_free` (no YAML).

**Data honesty (RouterBench paper)**

- We use the 0-shot split; the paper's main MMLU/HellaSwag/ARC/Winogrande results are 5-shot. Our numbers are not comparable to its tables.
- RouterBench judges MBPP with GPT-4, not unit tests: treat MBPP labels as noisier.
- The paper gives no pricing date: `ModelCard.price_date = "not stated by RouterBench"`.
- GSM8K's 0.25-step scores are an observed data issue (the paper says exact match).
- Also report the paper's AIQ and Zero-router (non-decreasing convex hull of single models) on our 4-model pool.

**Licences.** HRM, HRM-Text, ncps, CfC: Apache-2.0. ttt-lm-pytorch, titans-pytorch, TRM: MIT. Our modules re-implement from the equations; each file names its sources in a header, and `LICENSE-HRM-Text` is kept because `core.py` follows HRM-Text's schedule.

### D12. Gated-attention core variant (v1.3 → v1.4)

A second core architecture, `arch = "attn"`, alongside the MLP core of D11. The six role tokens `[task, budget, step, trace−1, trace−2, memory readout]` keep their identity (learned slot embedding each, no RoPE because they are roles, not a sequence) instead of being averaged, so the controller can compare the current step with earlier rounds directly (retry, propagation). Each H/L module is 2 pre-norm blocks of HRM-Text gated self-attention (fused bias-free gate/q/k/v projection, `o_proj(σ(gate) ⊙ attn)`, 4 heads × 32) followed by SwiGLU, capped by a weightless RMSNorm. The H/L schedule, fixed z_L init, input re-injection and staircase gradient horizon are those of D11, applied per token. An attention key mask hides absent trace slots and everything `context_free` must not see; the heads read the current-step token. Configs: `full_attn`, `memory_off_attn`, `context_free_attn`. The MLP configs are unchanged. In attention mode the H and L modules both receive the full masked token sequence (it is passed as both h and x), so they differ in schedule, not in which tokens they see; residual diagnostics average over valid tokens only.

### D13. Monte Carlo RL (Stage B, simulator; Stage C, live online)

Stage B starts from the supervised EMA checkpoint. Reward per episode `R = quality − λ·(spent/n_steps)/cost_ref` (`CompositeScorer` with `cost_usd = spent/n_steps`; `cost_ref` from `conf/score.yaml` is a per-step statistic), paid out per decision so that `Σ r_t = R`. λ ∈ {0.1, 0.3, 0.6, 1.0, 2.0}, sampled per episode. λ enters only the heads: `policy_logits = success_logits − λ·softplus(g(z_H, ln λ, c))·ln(card_cost/cost_ref)`, `halt_policy = halt_logit + h(z_H, ln λ, c)`, `c = σ(complexity_logit)`. It never enters the tokens, because `context_free` masks the budget token and must still see λ. The complexity head stays trained (auxiliary 0.2·MSE to the exact band), keys the bucketed baseline `b[band, λ]`, and conditions the policy. Loss `−mean(A·log π) − β·mean(H) + 0.2·MSE_complexity`, β 0.01 → 0.001, lr 1e-4, clip 1.0. Safety guard every 250 updates: val quality more than 5 points below the supervised reference at any λ → restore the last good weights, halve the lr. Frontier: serve mode per λ against the λ-oracle `weighted_episode_oracle(ep, λ/(n_steps·cost_ref))`. Configs `full` and `context_free` × 3 seeds. Stage C (live online RL) continues from the Stage B checkpoint and baseline; see the online RL plan.

### D9. Engineering

- Layout: packages at repo root (`controller/`, `data/`, `train/`, `eval/`). Delete the empty `models/` to avoid confusion with LLM "models".
- Python **3.12**. Do not spend time on 3.14.
- `.gitignore`: venv, `*.npy`, `*.parquet`, `data/raw/` (downloads: RouterBench pickle, bge-small weights), `data/processed/` (built dataset), `checkpoints/`, `runs/`. Nothing is read from the Hugging Face cache.
- Parameter count: print the measured count in `test_shapes.py`; the write-up quotes the measured number only.
- Work on `main`; one commit per task.

---

## 2. Repository layout

```
controller/
  contracts.py        # TaskRep, Strategy, DecisionRecord, ModelCard (v1)
  stub.py             # random valid Strategy, published day 1
  encoder.py          # StepEncoder: tokens for H and L
  core.py             # H/L schedule, adapted from HRM-Text (Apache-2.0)
  memory.py           # TTTLinear + CfCGate (the L module)
  heads.py            # model-scoring head, halting, complexity, confidence
  model.py            # Controller: encoder -> core -> heads
data/
  routerbench.py      # load, pick pool, split prompts
  embed.py            # bge-small embeddings, cached to .npy
  episodes.py         # synthetic episode generator
  oracle.py           # per-step labels, episode oracle
  env.py              # simulator wrapper for RL rollouts
train/
  train.py            # supervised training loop (stage A)
  rl.py               # Monte Carlo PG, bucketed baseline (stage B)
  configs/            # full.yaml, gate_off.yaml, memory_off.yaml, context_free.yaml
eval/
  policies.py         # always-strong, always-cheap, random, oracle
  evaluate.py         # frontier sweep, per-episode-type table
  diagnostics.py      # residual plot, surprise/gate trace, latency
tests/
  test_contracts.py
  test_shapes.py
  test_episodes.py
LICENSE-HRM-Text      # upstream licence, kept verbatim
README.md
```

`core.py` carries a header noting it is adapted from `sapientinc/HRM-Text` (`models/baselines/hrm_nocarry_bp_warmup.py`) and listing the changes.

---

## 3. Fixed settings (`TaskRep v1`)

| Setting | Value |
|---|---|
| Width `d` | 128 |
| Text encoder | `bge-small-en-v1.5`, 384-d, frozen |
| H tokens | 2: task, budget/state |
| L tokens | 1 current step + `k = 2` trace (D2) |
| Step types | one per kept 0/1-scored source (D3) |
| H cycles × L steps | 2 × 3 |
| Gradient horizon | last 2 steps, widening to 5 over first 20% of training |
| Pool | 4 models spanning the price range |
| Trainable parameters | measured in `test_shapes.py` (estimate 0.3–1M) |

Lock these on day 1. The other three components build against them.

---

## 4. Data

### 4.1 RouterBench preparation

1. Load the 0-shot split of `withmartian/routerbench`.
2. Choose **4 models** spanning the price range (cheapest to most expensive) as the pool. Record each model's price and the date in a `ModelCard`.
3. Drop prompts where the 4 models' results are missing.
4. **Split prompts 80/20 into train and test pools before building any episode.** Hold out 10% of the train pool for validation. No prompt appears in more than one pool.
5. Embed every prompt once with `bge-small` and cache to `.npy`.

### 4.2 Episode generator

Each episode is 3–6 steps with a starting budget. At each step the policy picks a model; the outcome (correctness, cost) is looked up from the table.

| Type | Share | How it is built | Why context matters |
|---|---|---|---|
| Correlated difficulty | 40% | All steps from one source and one difficulty band | Earlier outcomes reveal how hard the task is |
| Retry | 25% | A failed step's item returns as the next step | The failed model is now known to fail |
| Error propagation | 20% | Failure at step k lowers step k+1's quality unless a stronger model is used | Past failure changes what the next step needs |
| Plain random chain | 15% | Unrelated prompts, mixed sources | **Control:** context should not help here |

Budget pressure is applied across all types: the starting budget is drawn so that roughly a third of episodes cannot afford the strongest model on every step. Overspending zeroes quality for remaining steps.

Difficulty band for a prompt = number of pool models that fail it (0–4).

### 4.3 Sizes

| Split | Episodes | Source pool |
|---|---|---|
| Train | 20,000 (scale to 50,000 if the learning curve is still rising) | train prompts |
| Validation | 2,000 | validation prompts |
| Test | 5,000 | test prompts |

Generate with a fixed seed. Save as Parquet, one row per step.

### 4.4 Labels

For every step, `succeeds[m]` = whether pool model `m` gets it right, given the episode's state (retry and propagation rules applied). The oracle choice is the **cheapest model with `succeeds = 1`**; if none succeed, the cheapest model.

---

## 5. Model

```
StepEncoder -> H tokens: task, budget/state           (2 × 128)
               L tokens: current step, trace −1, −2   (3 × 128)
h = RMSNorm(mean(H tokens));  x = mean(L tokens)
Memory (once, if a new outcome): s = v − W k; r = CfC(u, Δt); W ← W₀ + (1−α)(W − W₀) + r s kᵀ
x = x + RMSNorm(W q)
Core: z_H = h; z_L = fixed init
  for cycle in 1..2:
      for step in 1..3:  z_L = L(z_L + z_H + x)
      z_H = H(z_H + z_L + h)
Heads (read z_H):
  p_success[m] = σ(w·SiLU(A z_H + card_m) + b)   # per pool model
  p_escalate   = σ(halt(z_H))
  complexity   = σ(cplx(z_H))                      # target band / 4
```

Full equations and their sources in D11. The capability vector = the model's train-split accuracy per step type (5 numbers) plus log₁₀ mean cost.

### Policy at inference

Among models whose card mean cost ≤ budget left, pick the **cheapest with `p_success ≥ τ`**; if none clears τ, the one with the highest `p_success`. Sweeping τ from 0 to 1 traces the quality–cost frontier.

Halting (D5, D10): after round 1, a second forward pass reads `p_escalate`; escalate iff it is > 0.5 and a stronger affordable model exists, choosing that model by the τ rule over the stronger ones.

---

## 6. Training

| Item | Setting |
|---|---|
| Loss | BCE on `succeeds[m]` for every pool model at every step, + BCE for halting, + MSE for complexity |
| Optimiser | AdamW, lr 3e-4, cosine decay, weight decay 0.01 |
| Batch | 256 episodes |
| Epochs | Up to 20, early stop on validation loss (patience 3) |
| Seeds | 0, 1, 2 |
| Hardware | Laptop GPU; CPU works |
| Time per run | ~10–30 min |

Keep an EMA of the weights (decay 0.999) and validate with it (D11). After training, fit one temperature on the validation set for the success probabilities. Freeze it.

### Configurations

| Config | What changes |
|---|---|
| `full` | Everything on |
| `gate_off` | CfC gate replaced by a fixed learned rate |
| `gate_stateless` | CfC gate replaced by σ(MLP) on the same input |
| `memory_off` | No memory `W`; readout = 0 |
| `context_free` | Earlier trace, task, budget and memory readout masked; identical otherwise |

5 configs × 3 seeds = 15 runs.

`context_free` is the decisive comparison: same architecture, data and training, with the only difference being whether the controller can see the workflow.

### Stage B: RL fine-tune

After supervised training, `full` and `context_free` are fine-tuned with Monte Carlo policy gradient in the simulator: reward = quality − λ·cost, λ sampled per episode and given to the policy, bucketed moving-average baseline by complexity band. Full specification in `RL_DESIGN.md`. This adds 2 configs × 3 seeds = 6 runs (~30–60 min each).

---

## 7. Evaluation

### Baselines

Always-strongest, always-cheapest, random, oracle, and `context_free`.

### Metrics

| Metric | How |
|---|---|
| Quality | Mean episode quality (fraction of steps correct) |
| Cost | Mean episode spend in USD (dated prices) |
| Frontier | Quality vs cost as τ sweeps 0→1; area under it |
| Gap recovered | (method − cheapest) / (oracle − cheapest) at matched cost |
| Calibration | ECE of success probabilities on test |
| Latency | Controller forward time, with and without cached task embedding |

All numbers: mean ± 95% confidence interval over 3 seeds.

### Required outputs

1. **Frontier plot:** all policies and configs on one quality–cost chart, oracle as the ceiling.
2. **Per-episode-type table:** one row per type, `full` vs `context_free`. Expected: a clear gain on correlated, retry and propagation; no meaningful gain on plain random chains.
3. **Ablation table:** `full`, `gate_off`, `memory_off` at a matched cost point.
4. **Residual plot:** H and L step-to-step change across the 6 inner steps, showing the sawtooth.
5. **Surprise trace:** one test episode showing a failure, the surprise spike, the gate opening and the escalation that follows.
6. **Learning curve:** test score at 5k, 20k, 50k training episodes (if time).

---

## 8. Day-by-day plan

### Week 1: skeleton and data

| Day | Work | Done when |
|---|---|---|
| 1 | `contracts.py` (v1) and `stub.py`; send to team | Stub returns a schema-valid `Strategy`; `test_contracts` passes |
| 2 | Load RouterBench, choose the 4-model pool, split prompts | Split files saved; no prompt overlap across splits (asserted) |
| 3 | Embed all prompts; cache | `.npy` cached; loads in under a second |
| 4 | Episode generator, all four types, budget pressure | 100 sample episodes look right by eye; type shares match the table |
| 5 | Oracle labels; generate train/val/test; always-strong, always-cheap, random and oracle policies | Baseline numbers on test exist; oracle beats everything |

### Week 2: the mechanism

| Day | Work | Done when |
|---|---|---|
| 6 | Contract/env additions, feature builder, `core.py` (MagicNorm blocks, staircase horizon) | Shape and gradient-horizon tests pass |
| 7 | Memory and gates, encoder, heads, model, rollout engine, training loop; `memory_off` end to end | First trained model beats always-cheapest on validation |
| 8 | Residual diagnostic | Plot saved; L shows within-cycle convergence and reset |
| 9 | `gate_off` (memory with fixed rate) | Trains stably; residual plot re-run |
| 10 | `full` (CfC gate) and `gate_stateless` | Trains stably; surprise, rate and α logged per write |

### Week 3: RL and evidence

| Day | Work | Done when |
|---|---|---|
| 11 | `context_free` config; temperature calibration; start all 15 supervised runs | Masking verified (trace tokens have zero effect when masked) |
| 12 | `rl.py` (env exists since Day 5); λ already in the budget token | One RL run on `full` improves validation return over its supervised start |
| 13 | RL runs: `full` and `context_free` × 3 seeds | Checkpoints and test predictions saved |
| 14 | Frontier sweep (τ for supervised, λ for RL), AIQ and Zero-router hull, per-type table, ablation table, ECE, halting behaviour | Tables and plots generated from saved predictions |
| 15 | Surprise trace, latency, write-up | Draft section ready for supervisor |

---

## 9. Two-week compression

- Days 1–5 unchanged.
- Days 6–9: encoder, core, heads, then TTT and gate together (skip the separate `gate_off` checkpoint day; still train it as a config).
- Day 10: all supervised runs with **1 seed**, plus `context_free` on 3 seeds (it carries the main claim).
- Day 11: RL fine-tune of `full` only, 1 seed.
- Day 12: frontier, per-type table, residual plot, short write-up. Drop the learning curve and latency breakdown.

---

## 10. Cut-lines, in order

If a day slips, cut from the top:

1. Learning curve.
2. Seeds for the ablations (keep 3 for `full` vs `context_free`, and for `full` vs `gate_stateless` if claiming "liquid").
3. RL on `context_free` (keep RL on `full`, report supervised `context_free` as the comparison).
4. Halting head (use the τ rule only).
5. RL stage entirely (supervised results stand on their own).
6. CfC gate: report the memory with a fixed rate as the full model, drop the "liquid" claim, and move the gate to the next phase.

Never cut: the prompt split, the `context_free` comparison, the plain-random control row, the oracle line.

---

## 11. Risks

| Risk | Sign | Response |
|---|---|---|
| Context gives no gain | `full` ≈ `context_free` on all types | Check that dependencies in the generator actually change oracle labels; inspect retry episodes first |
| Gain only from generator design | Gain on random chains too | Bug: context leaking into `context_free`; re-check masking |
| TTT unstable | Loss spikes, NaNs | Check `‖k‖ = 1` and `r ∈ (0,1)` (D11 makes each write a contraction); raise forgetting bias; shorten gradient horizon |
| Gate does nothing | `full` ≈ `gate_off` or ≈ `gate_stateless` | Report honestly; drop the "liquid" claim; check gate rate varies with surprise in the trace plot |
| Leakage | Test far above validation | Re-assert the prompt split |
| Panel says data is built to favour the method | — | Random-chain control row; TwinRouterBench static set if time allows |

---

## 12. Stretch (only if everything above is done)

- **Leave-one-model-out:** train on 3 pool models, add the 4th as a new `ModelCard` at test time, no retraining. Directly tests the no-retraining requirement.
- **TwinRouterBench static set:** evaluate tier prediction on the 970 real prefixes.

---

## 13. Reference repositories

All links checked to exist. "Use" says what to take from each; nothing is forked whole.

### Architecture

| Repo | Licence note | Use |
|---|---|---|
| [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) | Apache-2.0 | H/L forward loop, MagicNorm placement, gradient-horizon warmup (`models/baselines/hrm_nocarry_bp_warmup.py`). Source of `core.py`; keep `LICENSE-HRM-Text`. |
| [sapientinc/HRM](https://github.com/sapientinc/HRM) | Apache-2.0 | Original HRM: hierarchical convergence, ACT halting, residual diagnostics |
| [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | MIT | Small recursive-reasoning reference; useful for compact training loops |
| [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) | MIT | TTT-Linear update rule and mini-batch inner step, for `memory.py` |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) | MIT; unofficial | Surprise-gated memory updates; closest prior work to the gated L memory |
| [mlech26l/ncps](https://github.com/mlech26l/ncps) | Apache-2.0 | CfC and LTC cells in PyTorch; reference for `CfCGate` |
| [raminmh/CfC](https://github.com/raminmh/CfC) | Apache-2.0 | Original closed-form continuous-time network code |

### Data and benchmarks

| Repo | Licence note | Use |
|---|---|---|
| [withmartian/routerbench](https://github.com/withmartian/routerbench) | check repo | RouterBench loaders and evaluation; data also on Hugging Face as `withmartian/routerbench` |
| [ynulihao/LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) | check repo | Modern 33-model, 21-dataset routing data and 10 baseline routers; next-phase data source |
| [CommonstackAI/TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench) | Apache-2.0 | Step-level agent routing data with verified tier labels; stretch evaluation (§12) |

### Routing baselines and related systems

| Repo | Licence note | Use |
|---|---|---|
| [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | check repo | Context-free router implementations; reference for the external baseline |
| [ZhangYiqun018/MTRouter](https://github.com/ZhangYiqun018/MTRouter) | check repo | Multi-turn router; related work for the step-level routing subsection |

### Embeddings

| Repo | Licence note | Use |
|---|---|---|
| [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | check repo | Library for running the frozen text encoder |
| [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | check repo | Source of `bge-small-en-v1.5` |

### RL

| Repo | Licence note | Use |
|---|---|---|
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | check repo | Single-file policy-gradient implementations; reference for `rl.py` structure, advantage normalisation, entropy bonus |
| [ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning) | check repo | Offline RL from logged episodes (next phase, §10 of `RL_DESIGN.md`) |

**Rule for copied code:** any file adapted from a repo carries a header naming the source repo, the file, the licence and what was changed. Copy the upstream licence file into the repo when the licence requires it. Check each "check repo" licence before copying code (reading for reference needs no check).