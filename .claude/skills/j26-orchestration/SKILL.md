---
name: j26-orchestration
description: Master build guide for the J26-DS-359 adaptive agent orchestration project (SLIIT). Use for ANY coding, design, review, planning or research work in this repo, covering the controller (encoder, H/L core, TTT memory, CfC gate, heads, halting), learning (supervised, Monte Carlo RL, checkpoints), measurement (simulator, episodes, oracle, scorer, baselines), execution (gateway, providers, executor, catalogue), database and file structures, the research dashboard UI, experiment monitoring and alarms, research questions, hypotheses, expected results, statistics, and paper or GitHub references. Also use it when someone asks what to build next, how components connect, what a dataclass, table or tensor shape should be, how to run or monitor experiments, or what a result means. Read it before writing code even for small tasks, because the shared contracts and fixed settings keep four people's code compatible.
---

# J26-DS-359 Orchestration Layer: Master Build Guide

This skill is the single source of truth for building the project. Four students build four components that must fit together. Most bugs in a project like this come from two people guessing differently about a shape, a field name or who owns a file, so this guide fixes those things up front.

## 1. What we are building

Middleware between agent frameworks (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK) and LLM providers. For each step of an agent workflow it decides which model to call and how to run the step, conditioned on the state of the workflow so far, and trades quality against cost.

Research claim the code must be able to test:

> A controller that reads the workflow's state (trace, budget, adapting memory) achieves a better quality-per-cost trade-off than a context-free router that reads only the current query, and the liquid-gated memory contributes to that gain.

Everything in the repo exists to produce evidence for or against that claim. When a design choice is unclear, choose the option that makes the `full` vs `context_free` comparison cleaner.

## 2. Team and ownership

| Component | Package | Owner | Builds |
|---|---|---|---|
| Shared | `common/` | All (changes need team agreement) | Contracts, protocols, ids |
| C1 Controller | `controller/` | V M Wickramaarachchi | Encoder, H/L core, TTT memory, gate, heads, calibration |
| C2 Learning | `learning/` | R R H J Rathnayaka | Rollouts, supervised trainer, RL, baseline, replay, exploration, safety, checkpoints |
| C3 Measurement | `measurement/` | K V A Bethmi | Simulator, episode generator, oracle, verifier, scorer, profiler, baselines, evaluation harness |
| C4 Execution | `execution/` | S A Gunerathne | Gateway, framework adapters, strategy executor, provider adapter, catalogue, trace recorder, live demo console |
| Research dashboard | `ui/` | C3 leads, all contribute metrics | Streamlit + DuckDB views over runs and results |

Rule: a package imports only from `common/` and from its own files. Components talk to each other only through the protocols in `common/protocols.py`. If you need something from another component, add it to the protocol (with agreement), never import their internals.

## 3. Scope decision (read before Phase 3)

The build is **simulation-first**. The research claim is tested in a simulator built from RouterBench tables, where every model's outcome is already recorded and runs cost nothing. A live middleware is built afterwards as a thin demonstration on a small sample.

Consequences:

- Phases 0 to 2 need no database server, no Redis and no API keys. Storage is Parquet files and an in-process session dictionary, behind the storage interfaces in `references/data-and-storage.md`.
- PostgreSQL and Redis are Phase 3 backends for the same interfaces. Do not build them early.
- If the supervisor requires a production-grade live middleware, this section changes first, then the phase plan.

## 4. Fixed settings (contract v1)

These are part of the contract. Changing any of them changes tensor shapes or data meaning for everyone, so they change only with a version bump (Section 9).

| Setting | Value |
|---|---|
| Python | 3.12 |
| Controller width `d` | 128 |
| Text encoder | `BAAI/bge-small-en-v1.5`, 384-d, frozen |
| Trace items shown explicitly `k` | 2 |
| H cycles x L steps | 2 x 3 |
| Blocks per module | 2 MLP blocks, RMSNorm, exit norm |
| Gradient horizon | last 2 steps, widening to 5 over first 20% of training |
| Model pool (prototype) | 4 RouterBench models spanning the price range |
| Max rounds per step | 2 (one escalation) |
| Simulated verifier flip probability | 0.15 |
| Episode length | 3 to 6 steps |
| Episode mix | correlated 40%, retry 25%, propagation 20%, random 15% |
| Prompt split | train 72%, val 8%, test 20%, split before building episodes |
| Seeds | 0, 1, 2 |

## 5. Architecture in one page

```text
Framework call
  -> C4 builds TaskRep (task, current step, last k trace items, budget, lambda)
  -> C1 decide():
       observe last TraceItem: surprise -> gate rate -> TTT update of W (once per decision)
       encode tokens; H/L core 2 cycles x 3 L-steps with memory readout
       heads on z_H: model (scores a variable pool), halting, complexity (+ later: tools, memory, depth, verification, topology)
       -> Strategy (to C4) + DecisionRecord (to C2)
  -> C4 runs Strategy against provider (or C3 simulator), returns TraceItem
  -> C3 scores the round -> Score (reward)
  -> C2 trains on DecisionRecords + Scores -> checkpoint -> C1
```

Two kinds of controller state, and mixing them up is the most common design error:

| State | Lives for | Updated | Holds |
|---|---|---|---|
| `W` TTT fast weights (d x d) | one episode, reset at start | once per decision | compressed history of this task |
| `z_H`, `z_L` | one decision | 2 x 3 inner loop | working state for this choice |

Full diagrams, module tables and the forward pass are in `references/architecture.md`.

## 6. Where to read next

Read the file for the task in front of you. Do not load everything.

| You are... | Read |
|---|---|
| Anyone, before first commit | `references/conventions.md` |
| Touching any dataclass or protocol | `references/contracts.md`, then `assets/common/contracts.py` |
| Needing the big picture or a diagram | `references/architecture.md` |
| Building C1 | `references/c1-controller.md` |
| Building C2 | `references/c2-learning.md` |
| Building C3 | `references/c3-measurement.md` |
| Building C4 | `references/c4-execution.md` |
| Handling files, caches, splits, DB tables, tracking files | `references/data-and-storage.md` |
| Running experiments or making plots | `references/evaluation.md` |
| Research questions, hypotheses, expected results, statistics, threats | `references/research-plan.md` |
| Dashboard, what to log, alarms, live demo console | `references/ui-and-monitoring.md` |
| Citing or reading related papers | `references/papers.md` |
| Finding reference code or checking a licence | `references/repos.md` |
| Setting up the repo on day 1 | Section 8 below, then copy `assets/` |

## 7. Phase plan

| Phase | Target | Goal | Done when |
|---|---|---|---|
| 0 Scaffold | Week 1 | Repo, contracts, stubs, data prep | Every package imports; stub controller returns a valid `Strategy`; RouterBench cleaned and split |
| 1 Prototype | Weeks 2 to 3 | Core claim in simulation | `full` vs `context_free` per-episode-type table, 3 seeds |
| 2 Full components | Weeks 4 to 12 | Every head, RL, baselines, profiling | All configs and baselines run from one command each |
| 3 Live demo | Weeks 13 to 16 | Thin live path on real providers | N real episodes served, scored, and compared with simulation |
| 4 Experiments | Weeks 17 to 24 | Every output in `references/evaluation.md` | Results regenerated from saved predictions by one script |
| 5 Write-up | to May 2027 | Reports and viva | Code tagged; numbers in reports match `results/` |

Week numbers count from repo creation. Replace them with real dates once the team confirms deadlines.

Each component guide breaks its phases into numbered tasks. Every task has: files, what to implement, inputs and outputs, a test that proves it is done, and a commit message.

## 8. Day-1 setup (one person does this, everyone pulls)

1. Create the repo layout from `references/conventions.md`.
2. Copy `assets/common/contracts.py` and `assets/common/protocols.py` into `common/`.
3. Copy `assets/conf/` into `conf/`, `assets/common/tracking.py` into `common/`, and `assets/ui/` into `ui/`.
4. Create a Python 3.12 venv; install `torch`, `sentence-transformers`, `pyarrow`, `pandas`, `numpy`, `hydra-core`, `pytest`, `matplotlib`, `streamlit`, `duckdb`, `plotly`. Phase 3 adds `fastapi`, `uvicorn`, `httpx`, `litellm`, `psycopg[binary]`, `redis`.
5. Write `.gitignore`: venv, `*.npy`, `*.parquet`, `data/raw/`, `data/cache/`, `checkpoints/`, `results/`, `runs/`, Hugging Face cache.
6. Add `tests/contracts/test_roundtrip.py`: every dataclass survives `to_json` then `from_json` unchanged. This test protects everyone.
7. Commit: `chore: scaffold repo, contracts v1, configs`.

## 9. Changing shared things

- Any change to `common/` or to Section 4 values: open a pull request titled `contract: ...`, tag all three teammates, bump the `version` field (`v1` to `v2`), update this skill.
- Checkpoints record the contract version; the controller refuses to load a mismatched one.
- Scorer changes bump `scorer_version`; never mix scorer versions in one results table.

## 10. Rules that protect the research claim

These are the things an examiner will probe. Each one exists because getting it wrong silently inflates results.

1. **Split prompts before building episodes.** A prompt in both train and test episodes is leakage.
2. **The controller never sees ground truth.** Trace items carry `verifier_ok` (noisy), never true correctness. Rewards use true correctness.
3. **`context_free` differs from `full` only by masking** trace tokens, budget token and memory readout. Same architecture, data, training. A test asserts its output is invariant to those inputs.
4. **Keep the plain random-chain episodes.** On them `full` should not beat `context_free`. If it does, something leaks.
5. **Report the oracle line** on every quality-cost plot, using the exact episode oracle.
6. **Three seeds, mean and 95% CI** for every headline number.
7. **Report measured numbers only** (parameter counts, latency, costs with price dates).
8. **Credit adapted code.** Any file adapted from another repo has the header in `assets/templates/adapted_header.py`.

## 11. Code comment standard

Every module starts with a docstring in this form, so anyone can see shapes and ownership without reading the body:

```python
"""
Purpose:  One sentence on what this module does and why it exists.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   name: shape/type - meaning
Outputs:  name: shape/type - meaning
Notes:    Non-obvious decisions, with the reason.
Source:   (only if adapted) repo, file, licence, what changed
"""
```

Functions with tensor arguments state shapes in the docstring using the symbols in `references/architecture.md` (B, d, e, k, P, U, F). Comments explain why, not what.

## 12. Research monitoring

Every training and evaluation script logs through `common/tracking.py` `RunLogger` with the fixed metric names in `references/ui-and-monitoring.md`. The dashboard (`ROOT=. streamlit run ui/dashboard.py`) reads those files plus `results/`, and shows runs, curves, alarms, the main comparison, calibration and single episodes.

The research questions, hypotheses, success criteria and statistics are fixed in `references/research-plan.md`. Run seed 0 as a pilot, record it in `docs/experiment_log.md`, freeze the analysis (tag `analysis-freeze`), then run seeds 1 and 2. Do not change the analysis after seeing them.

## 13. When you are stuck

- Shape or field question: the answer is in `assets/common/contracts.py` or `references/architecture.md`. If not, it is a contract gap: raise it, do not guess.
- "Who builds this?": Section 2 table, then the component guide's task list.
- A result looks too good: check rules 1 to 4 in Section 10 first.
- A result looks like no gain: check that the simulator's dependency rules actually change oracle labels (inspect retry episodes), and that masking is not applied to `full` by mistake.
