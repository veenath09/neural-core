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

## 1a. Design decisions v1.2 (supersede later sections where they conflict)

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

1. **Memory update (once).** `obs` = encoding of the previous round's outcome. `surprise = obs − W·key(obs)`; `rate = CfCGate(surprise)`; `W ← W − rate · ∇‖W·key(obs) − value(obs)‖²` (TTT-Linear closed form). Log `‖surprise‖` and `rate`.
2. **Reasoning (2 × 3).** Every L-step sees the same input: pooled L-tokens + memory readout `W·query(z_H)`. `z_L ← L_block(z_L + z_H + input)`. After 3 L-steps, `z_H ← H_block(z_H + z_L)`. `z_L` restarts from its learned init each decision.
3. Heads read the final `z_H`.

**Trace length k = 2** (not 4). With episodes of 3–6 steps, k = 4 shows almost the whole history explicitly and the memory has nothing to add; k = 2 forces older history through `W`.

Ablations then mean:

| Config | Change |
|---|---|
| `gate_off` | `rate` = one learned constant |
| `memory_off` | No `W`; memory readout = 0 |
| `context_free` | Trace tokens, budget token **and** memory readout masked |

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
- **Retry episodes.** The skeleton lists `n` prompts; the episode always has `n` steps. In a retry episode, if a step's final answer is wrong and the item is not already a retry, the next step is the same item (marked retried); otherwise the next listed prompt. Items are retried at most once.
- **Step outcome.** A step's outcome is its last round's answer. Propagation uses the tier of the model that gave that final wrong answer.
- **TTT surprise.** `surprise = value(obs) − W·key(obs)`. The gate rate is one scalar per decision. At an episode's first decision there is no outcome to store, so the memory update is skipped. Gradients flow through `W` across the whole episode.
- **Sources (checked on the data).** Kept: MMLU (grouped), HellaSwag, ARC-Challenge, Winogrande, MBPP → 5 step types. Dropped: grade-school-math (0.25-step scores), MT-Bench (graded), small miscellaneous evals. Duplicate prompt texts (16) are removed before splitting.
- **Pool.** mistral-7b-chat, mixtral-8x7b-chat, gpt-3.5-turbo-1106, gpt-4-1106-preview (mean cost per call rises with tier). Costs are RouterBench's per-prompt `total_cost`, prices as of the RouterBench release (2024-03).
- **ModelCard v1** (until `model_onboarding.md` lands): `model_id, provider, tier, mean_cost_usd, price_source, price_date, capability` where `capability` = train accuracy per step type + log10(mean cost).
- **Sequencing.** `env.py` and the episode oracle move to Day 5 (D1 needs them for the behaviour policy).

### D9. Engineering

- Layout: packages at repo root (`controller/`, `data/`, `train/`, `eval/`). Delete the empty `models/` to avoid confusion with LLM "models".
- Python **3.12**. Do not spend time on 3.14.
- `.gitignore`: venv, `*.npy`, `*.parquet`, `data/cache/`, `checkpoints/`, `runs/`, HF cache.
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
StepEncoder -> (h_tokens: 2×128, l_tokens: 5×128)
Core:
  z_H = pooled(h_tokens); z_L = learned init
  for cycle in 1..2:
      for step in 1..3:
          surprise = l_obs - predict(z_L)
          rate     = CfCGate(surprise)
          z_L      = TTTLinear update scaled by rate, under z_H
      z_H = H_block(z_H, z_L)
Heads (read z_H):
  success_prob[m] = sigmoid(match(need(z_H), capability[m]))   # per pool model
  halt_logit      = halting head
  complexity, confidence
```

- H and L use 2 MLP blocks each, pre-norm, with a final norm at each module's exit (MagicNorm pattern).
- The model head scores each `ModelCard` by matching a requirement vector against its capability vector, so the pool is a list, not fixed slots.
- Capability vector for the prototype = the model's train-split accuracy per step type (8 numbers) plus log price.

### Policy at inference

Pick the **cheapest model whose `success_prob ≥ τ`**; if none clear τ, pick the model with the highest `success_prob`. Sweeping τ from 0 to 1 traces the quality–cost frontier.

Halting: after a step, if the chosen model's predicted success is below τ and budget allows, escalate to the next model up; otherwise stop.

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

After training, fit one temperature on the validation set for the success probabilities. Freeze it.

### Configurations

| Config | What changes |
|---|---|
| `full` | Everything on |
| `gate_off` | CfC gate replaced by a fixed learned rate |
| `memory_off` | L is a plain MLP block, no TTT, no gate |
| `context_free` | Trace and budget tokens masked; identical otherwise |

4 configs × 3 seeds = 12 runs.

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
| 6 | `encoder.py`, `core.py` with plain L | Shape tests pass; forward runs on a batch |
| 7 | Heads, training loop, `memory_off` config end to end | First trained model beats always-cheapest on validation |
| 8 | Residual diagnostic | Plot saved; L shows within-cycle convergence and reset |
| 9 | TTT-Linear in L, fixed rate (`gate_off`) | Trains stably; residual plot re-run |
| 10 | CfC gate (`full`) | Trains stably; surprise and gate rate logged per step |

### Week 3: RL and evidence

| Day | Work | Done when |
|---|---|---|
| 11 | `context_free` config; temperature calibration; start all 12 supervised runs | Masking verified (trace tokens have zero effect when masked) |
| 12 | `env.py` and `rl.py`; add λ to the budget token | One RL run on `full` improves validation return over its supervised start |
| 13 | RL runs: `full` and `context_free` × 3 seeds | Checkpoints and test predictions saved |
| 14 | Frontier sweep (τ for supervised, λ for RL), per-type table, ablation table, ECE, halting behaviour | Tables and plots generated from saved predictions |
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
2. Seeds for the ablations (keep 3 for `full` vs `context_free`).
3. RL on `context_free` (keep RL on `full`, report supervised `context_free` as the comparison).
4. Halting head (use the τ rule only).
5. RL stage entirely (supervised results stand on their own).
6. CfC gate: report TTT with fixed rate as the full model and the gate as next phase.

Never cut: the prompt split, the `context_free` comparison, the plain-random control row, the oracle line.

---

## 11. Risks

| Risk | Sign | Response |
|---|---|---|
| Context gives no gain | `full` ≈ `context_free` on all types | Check that dependencies in the generator actually change oracle labels; inspect retry episodes first |
| Gain only from generator design | Gain on random chains too | Bug: context leaking into `context_free`; re-check masking |
| TTT unstable | Loss spikes, NaNs | Clip inner update; lower inner learning rate; shorten gradient horizon |
| Gate does nothing | `full` ≈ `gate_off` | Report honestly; check gate rate varies with surprise in the trace plot |
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
| [sapientinc/HRM](https://github.com/sapientinc/HRM) | check repo | Original HRM: hierarchical convergence, ACT halting, residual diagnostics |
| [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | check repo | Small recursive-reasoning reference; useful for compact training loops |
| [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) | check repo | TTT-Linear update rule and mini-batch inner step, for `memory.py` |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) | check repo; unofficial | Surprise-gated memory updates; closest prior work to the gated L memory |
| [mlech26l/ncps](https://github.com/mlech26l/ncps) | check repo | CfC and LTC cells in PyTorch; reference for `CfCGate` |
| [raminmh/CfC](https://github.com/raminmh/CfC) | check repo | Original closed-form continuous-time network code |

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