# Neuro-inspired Framework for Adaptive AI Agent Orchestration


---

## 1. What this is

AI agents solve a task as a **workflow**: a chain of steps, each one an LLM call. Every step forces an orchestration decision:

- **which model** should do this step: a cheap, weak one, or an expensive, strong one?
- whether to **check the answer and escalate** to a stronger model if it looks wrong;
- how to stay **within budget** across the whole task.

Most routers decide this from the current question alone. They forget what happened earlier in the workflow, so they never learn that "this task is hard and cheap models keep failing".

This framework orchestrates agents with a small controller **inspired by how the brain controls behaviour**:

- it reasons at **two speeds**: slow task-level planning and fast step-level processing;
- it keeps a **short-term memory that learns while the task runs** and writes harder when something surprising happens;
- the strength of those writes is set by a **liquid neural cell** with its own internal dynamics and sense of time.

The controller watches the workflow unfold and **adapts its routing to the task in front of it**, with no retraining.

**Research question.** Does a neuro-inspired controller that reads workflow state beat a router that reads only the current query on the quality–cost trade-off? And does its liquid-gated memory contribute?

---

## 2. Where the "neuro-inspired" comes from

| Brain idea | What it means in neuroscience | How the framework uses it | Source of the engineering design |
|---|---|---|---|
| **Hierarchical, multi-timescale processing** | Higher cortical areas integrate slowly (context, goals); lower areas react fast (details) | **H/L core:** a slow H module holds the task-level picture, and a fast L module iterates on the current step; L runs 3 steps for each H update | HRM / HRM-Text (Sapient, 2025) |
| **Synaptic plasticity / fast weights** | Connections change within seconds, giving short-term memory without permanent learning | **Episode memory W:** a weight matrix rewritten after every round of the task, then discarded when the task ends | TTT-Linear (2024), fast-weight programmers |
| **Prediction error ("surprise") drives learning** | Neurons signal the gap between expected and actual outcome (predictive coding, dopamine) | The memory compares what it expected with what happened. Surprise `s = v − W·k` decides what gets written | Titans (2025) |
| **Forgetting / homeostasis** | Old traces fade so memory stays useful | A learned forgetting rate pulls memory back toward its resting state | Titans (2025) |
| **Liquid neurons** | Neurons with input-dependent time constants (from C. elegans studies) | The **CfC gate**, a liquid-time-constant cell with its own state and a time input, sets how strongly each experience is stored | Liquid / CfC networks (Hasani et al., 2021–22) |
| **Metacognition: knowing when you're wrong** | Monitoring confidence and seeking help | A **halt head** decides, after seeing the verifier's verdict, whether to escalate to a stronger model | D5 of the spec; TRM-style halting |

These are **inspirations**, not claims of biological realism. Each one is implemented as a standard, testable neural-network building block.

---

## 3. The whole solution at a glance

```
                 ┌──────────────────────────────────────────────────────────┐
  a task  ──────▶│  AGENT WORKFLOW  (steps 1..n, each step = an LLM call)   │
                 └───────────────┬──────────────────────────▲───────────────┘
                                 │ step + history            │ Strategy
                                 ▼                           │ (model, escalate?)
        ┌────────────────────────────────────────────────────┴───────────────┐
        │               ADAPTIVE ORCHESTRATION CONTROLLER                    │
        │                                                                    │
        │  perceive ──▶ remember ──▶ reason ──▶ decide ──▶ monitor itself    │
        │  (encoder)    (episode     (H/L core:  (model     (halt head:      │
        │               memory W,    slow H +    head over  escalate?)       │
        │               CfC gate)    fast L)     model cards)                │
        └────────────────────────────┬───────────────────────────────────────┘
                                     │ calls the chosen model
                                     ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │  MODEL POOL   mistral-7b · mixtral-8x7b · gpt-3.5-turbo · gpt-4    │
        │               (cheap ◀──────────────────────────────▶ strong)      │
        └────────────────────────────┬───────────────────────────────────────┘
                                     │ answer, cost
                                     ▼
                         verifier (noisy "looks right / wrong")
                                     │
                                     └──▶ outcome written into the trace and into memory
```

The controller talks to the rest of the system through four typed **contracts** (`controller/contracts.py`):

| Contract | Direction | What it carries |
|---|---|---|
| `TaskRep` | → controller | The step to decide on: prompt, step type, round, the last 2 outcomes (`TraceItem`s), budget, position in the workflow |
| `Strategy` | controller → | The decision: `model_id`, `halt` (stop / escalate), complexity and confidence. Also `tools`, `memory`, `depth`, `verification`, `topology`, which are fixed defaults in this prototype and are the hooks for later orchestration decisions |
| `DecisionRecord` | controller → logs | Why it decided: log-probs, entropy, predicted difficulty, and its internal state (for monitoring and RL) |
| `ModelCard` | → controller | Describes a model in the pool: price tier, cost, accuracy per task type. The pool is a list, so models can be added without retraining |

`controller/stub.py` returns a random but valid `Strategy`, for testing integrations before a trained model exists.

---

## 4. Inside the controller

### 4.1 One decision, step by step

```
 INPUTS (TaskRep)                ENCODE                     TOKENS (d = 128)
 ───────────────                 ──────                     ────────────────
 task prompt ─────────┐
 current step prompt ─┼──▶ bge-small (frozen) ──▶ 384→128 ──▶ H: [ task ]
 step type, round ────┘                                       H: [ budget/state ]
 last 2 rounds ─────────▶ feature MLP + model-card embed ───▶ L: [ step ][ trace−1 ][ trace−2 ]
 budget, step i/n ──────▶ feature MLP ─────────────────────▶ (into H: budget/state)

 REMEMBER   (once per new outcome)
 ──────────
   o = newest outcome token
   surprise  s = v − W·k
   rate      r = CfC_gate(s, Δt)           ◀── liquid cell, keeps its own state
   forget    α
   W ← W₀ + (1−α)(W − W₀) + r · s kᵀ
   readout = RMSNorm(W · q)  ──────────────▶ added to the L input

 REASON   (H/L core, 2 cycles × 3 fast steps)
 ──────
   z_H = h (task + budget)          z_L = fixed start
   cycle 1:  z_L ← L(z_L + z_H + x)  ×3   then   z_H ← H(z_H + z_L + h)
   cycle 2:  z_L ← L(z_L + z_H + x)  ×3   then   z_H ← H(z_H + z_L + h)

 DECIDE   (heads read z_H)
 ──────
   p_success[m] for every model card m   ──▶  τ rule: cheapest model with p ≥ τ
   p_escalate (read after round 1)       ──▶  escalate to a stronger affordable model?
   complexity (how hard is this step?)
```

### 4.2 The parts

| Part | File | What it does |
|---|---|---|
| Feature builder | `controller/features.py` | Turns the workflow state into numbers using **only what a real system could see**: prompts, the noisy verifier and the budget. A test checks that no ground truth leaks in |
| Step encoder | `controller/encoder.py` | Frozen **bge-small** sentence embeddings (384-d) → 128-d tokens, plus embeddings for step type, round and model cards |
| Episode memory | `controller/memory.py` | Fast-weight matrix `W` (128×128), learned at test time, reset each task. Delta-rule write with surprise and forgetting |
| Liquid gate | `controller/memory.py` (`CfCGate`) | Closed-form continuous-time cell: 16 units, state carried across the task, time input Δt |
| H/L core | `controller/core.py` | Two reasoning modules, each 2 SwiGLU blocks with MagicNorm. Truncated backprop whose horizon widens from 2 to 5 steps during training |
| Heads | `controller/heads.py` | Model head (matches a requirement vector against each model card), halt head, complexity head |
| Controller | `controller/model.py` | Wires everything together; holds the 5 experimental configs |

Size: **737k trainable parameters**. Latency: **~1.6 ms per decision** on a laptop CPU, plus about 52 ms to embed a new prompt. Both are negligible next to an LLM call.

### 4.3 Experimental configurations (ablations)

| Config | What's removed | Question it answers |
|---|---|---|
| `full` | nothing | The proposed neuro-inspired controller |
| `gate_stateless` | liquid cell → plain MLP gate | Does the **liquid** part (internal state and time) help? |
| `gate_off` | gate → a single constant rate | Does **surprise-driven** writing help? |
| `memory_off` | no episode memory | Does the **fast-weight memory** help? |
| `context_free` | no history, no budget, no memory | **Main comparison:** does reading the workflow help at all? (= a conventional query-only router) |

---

## 5. How it is trained and tested

### 5.1 Data: real model outcomes, simulated workflows

- **RouterBench** (0-shot) records, for 27,241 prompts, whether each of 4 models answered correctly and what the call cost. The prompts come from 5 task types: MMLU, HellaSwag, ARC-Challenge, Winogrande and MBPP.
- The prompts are split into train, validation and test **before** anything else, so no prompt is ever in two splits.
- Prompts are chained into **27,000 simulated workflows** (3–6 steps, with a budget):

| Workflow type | Share | Rule | What the controller should learn |
|---|---|---|---|
| Correlated difficulty | 40% | All steps from one task type and difficulty level | "This task is hard; stop trying cheap models" |
| Retry | 25% | A step the verifier flags as failed comes back once | "That model just failed this; use another" |
| Error propagation | 20% | A failure makes the next step need a stronger model | "After a failure, step up" |
| Random chain | 15% | Unrelated steps | **Control:** history should *not* help here |

- **Simulator** (`data/env.py`):
  - escalation to a stronger model costs an extra call;
  - running out of budget ends the task;
  - the controller only sees a **noisy verifier**, wrong 15% of the time, never the true answer.
- **Oracle** (`data/oracle.py`): an exact search that knows every outcome and finds the best possible routing. It is the ceiling on every result.

### 5.2 Training

- **Live rollouts.** Workflows are played in the simulator. At each step the action comes from the oracle (50%), a random model (25%), or the controller itself (25%), so it learns from its own mistakes too.
- **Loss.** For every model at every visited step, predict whether it would succeed. Also predict whether escalating would help, and how hard the step is.
- **Optimiser.** AdamW with cosine decay and early stopping. An EMA of the weights with warm-up is used for validation.

### 5.3 Results so far (validation set, one seed: indicative only)

| Policy | Quality (steps correct) | Cost per task | Correlated | Retry | Propagation | Random (control) |
|---|---|---|---|---|---|---|
| Always cheapest (mistral-7b) | 0.228 | $0.0001 | | | | |
| Random model | 0.553 | $0.0023 | | | | |
| Always strongest (gpt-4) | 0.689 | $0.0070 | | | | |
| `memory_off` | 0.682 | $0.0028 | 0.745 | 0.673 | 0.550 | 0.701 |
| `gate_off` | 0.693 | $0.0030 | 0.764 | 0.682 | 0.567 | 0.692 |
| `gate_stateless` | 0.699 | $0.0031 | 0.768 | 0.693 | 0.563 | 0.705 |
| **`full` (neuro-inspired)** | **0.699** | **$0.0029** | **0.774** | 0.683 | 0.563 | 0.704 |
| Oracle (perfect knowledge) | 0.924 | $0.0013 | | | | |

What this shows so far:
- **The controller reaches gpt-4-level quality at about 40% of gpt-4's cost.**
- **The memory adapts within the task.** The gain is largest on correlated workflows (0.774 against 0.745 without memory) and almost zero on random chains, which is the pattern an adaptive controller should produce.
- **The remaining gap to the oracle is caution.** The controller sends almost nothing to the cheapest model (about 1% of steps, against the oracle's 25%).

What it does **not** show yet:
- The main comparison against `context_free` is still to come.
- There are no repeat seeds yet.
- Calibration and the full quality–cost curve are still to come.
- With one seed, the liquid gate is not yet distinguishable from a plain gate.

---

## 6. Roadmap to prove the thesis

| Claim | Experiment | Status |
|---|---|---|
| Reading the workflow beats a query-only router | `full` against `context_free`, 3 seeds, per workflow type | next |
| The controller adapts **within a task** at test time | Quality by step index (should rise on correlated tasks) | planned |
| It adapts to an **unseen task type** | Train on 4 task types, test on the 5th | planned |
| It adapts to a **new model** without retraining | Train with 3 models, add the 4th as a new model card | planned |
| It beats a TRINITY-style coordinator (Sakana, 2025) | A small query-only head on frozen embeddings as a baseline | planned |
| The **liquid** gate matters | `full` against `gate_stateless` and against `gate_notime` | planned |
| Better quality–cost trade-off overall | Calibrated τ sweep → frontier, AIQ, RouterBench Zero-router | planned |
| Learns from its own outcomes | RL fine-tune in the simulator | later |

---

## 7. Repository layout

```
controller/   contracts.py  stub.py  features.py  encoder.py  core.py  memory.py  heads.py  model.py
data/         routerbench.py  embed.py  episodes.py  env.py  oracle.py
              raw/        downloads: RouterBench, bge-small weights   (git-ignored)
              processed/  built dataset: prompts, model cards, embeddings, workflows (git-ignored)
train/        rollout.py  train.py
eval/         policies.py  baselines.py  diagnostics.py
tests/        one test file per module (118 tests)
docs/         IMPLEMENTATION_PLAN_neural_core.md   full design spec, decisions D1–D11
              superpowers/plans/                   task-by-task implementation plans
runs/         training outputs (git-ignored)
LICENSE-HRM-Text
```

## 8. How to run

```bash
# setup (Python 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# build the data (downloads into data/raw/ once)
.venv/bin/python -m data.routerbench     # prompt table, split, model cards
.venv/bin/python -m data.embed           # bge-small embeddings (a few minutes on CPU)
.venv/bin/python -m data.episodes        # simulated workflows

# reference numbers
.venv/bin/python -m eval.baselines val

# train a configuration: full | gate_off | gate_stateless | memory_off | context_free
.venv/bin/python -m train.train --config full --seed 0

# diagnostics: residuals.png, memory_trace.csv in runs/full-s0/
.venv/bin/python -m eval.diagnostics runs/full-s0

# tests
.venv/bin/python -m pytest
```

## 9. References and licences

| Work | Used for | Licence |
|---|---|---|
| HRM, HRM-Text (Sapient) | Hierarchical slow/fast reasoning core, MagicNorm, gradient-horizon warm-up | Apache-2.0 |
| TRM (Samsung SAIL Montreal) | MLP blocks, weight EMA, halting loss | MIT |
| TTT-Linear | Fast-weight memory learned at test time | MIT |
| Titans | Surprise-driven writing and forgetting | MIT |
| Liquid / CfC networks (ncps) | Liquid gate | Apache-2.0 |
| TRINITY, Conductor (Sakana AI) | Related work and planned baseline | — |
| RouterBench (Martian) | Model outcomes and costs, AIQ metric | see repo |
| bge-small-en-v1.5 (BAAI) | Frozen text encoder | MIT |

All controller modules are re-implemented from the published equations, and each source file names its sources in its header. The full design rationale is in [`docs/IMPLEMENTATION_PLAN_neural_core.md`](docs/IMPLEMENTATION_PLAN_neural_core.md).
