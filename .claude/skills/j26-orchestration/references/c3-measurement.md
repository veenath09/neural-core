# C3 Measurement: Step-by-Step Build Guide

Owner: K V A Bethmi. Package: `measurement/`.

C3 builds the ground everyone stands on: the data, the simulator, the oracle and the scores. The prototype cannot start without C3-01 to C3-06, so these come first. If the controller owner writes any of them to save time in Phase 1, the code still lives in `measurement/` and C3 owns it afterwards.

## Phase 0: data

### C3-01 RouterBench loader and cleaning
- Files: `measurement/routerbench.py`, `scripts/prepare_data.py`
- Build:
  1. Download the 0-shot split of `withmartian/routerbench` into `data/raw/`.
  2. Inspect each source's score column. Keep only sources whose scores are exactly 0/1. Drop MT-Bench and any non-binary source. Record the kept list; it defines the step types.
  3. Choose 4 pool models spanning the price range (cheapest, two middle, most expensive) that have results on every kept prompt.
  4. Write `data/clean/routerbench_binary.parquet`: `prompt_id, source, step_type, prompt_text, {model}_correct, {model}_cost` for the 4 models.
  5. Compute `difficulty = number of pool models that fail` (0 to 4).
  6. Create `ModelCard`s for the 4 models with prices and the price date used by RouterBench.
- Test: no nulls; every correctness value in {0,1}; difficulty in 0..4.
- Commit: `feat(measurement): RouterBench loader, binary filter, 4-model pool`

### C3-02 Leakage-safe split
- Build: split prompt ids 72/8/20 (train/val/test), stratified by source and difficulty, seed 1234. Save `data/clean/splits.parquet`.
- Test: the three sets are disjoint and cover all prompts.
- Commit: `feat(measurement): leakage-safe prompt split`

### C3-03 Capability vectors for the pool
- Build: for each pool model, train-split accuracy per step type plus log price and log mean latency (latency 0 if unavailable). Store in `ModelCard.capability` with `probe_version = "rb_train_v1"`.
- Why: in the prototype the train split doubles as the probe set. Never compute capabilities from val or test.
- Commit: `feat(measurement): capability vectors from train split`

## Phase 1: simulator

### C3-04 Episode skeleton generator
- Files: `measurement/episodes.py`
- Build: `generate(split, n, mix, seed) -> list[EpisodeSkeleton]`.

| Type | Share | Construction |
|---|---|---|
| correlated | 40% | all prompts from one source and one difficulty band |
| retry | 25% | mixed prompts; retries are inserted live by the simulator |
| propagation | 20% | mixed prompts; propagation rule applied live |
| random | 15% | unrelated prompts, mixed sources and difficulty (control) |

- Length uniform 3 to 6. Budget: draw so that about one third of episodes cannot afford the most expensive model on every step (`budget = factor * length * mean_cost_strong`, factor from [0.4, 1.5]).
- Only prompts from the matching split. Save `data/episodes/skeletons_{split}.parquet`.
- Test: shares within 2% of target; no cross-split prompt ids.
- Commit: `feat(measurement): episode skeleton generator`

### C3-05 Simulator (EnvironmentAPI)
- Files: `measurement/simulator.py`

```python
class Simulator:
    """
    Purpose:  RouterBench-backed environment with workflow dependencies.
    Inputs:   EpisodeSkeleton, lam, Strategy per round
    Outputs:  TaskRep, TraceItem (noisy verifier), Score, done
    Notes:    Outcomes depend on earlier choices, so they are computed here, live,
              never stored in skeletons. counterfactual() exposes, for every pool
              model, success given the realised history: this is how labels are made.
    """
```

Outcome rules, applied in this order:

1. **Base:** `success(m) = table[prompt, m]`.
2. **Retry (retry episodes):** if a step's final round fails, the same prompt is inserted as the next step (once per original step).
3. **Propagation (propagation episodes):** if step i's final round failed with a model of price tier j, then at step i+1 `success(m) = table[prompt, m] and tier(m) > j`.
4. **Budget:** a round whose cost exceeds `budget_left` is not run; the step scores 0 and the episode ends with status `budget_exhausted`.
5. **Escalation:** `halt="escalate"` on round 0 runs round 1 on the same step with the next chosen model; both rounds are charged. Round 1 always stops.
6. **Verifier:** `verifier_ok = success XOR Bernoulli(0.15)` with a per-episode seeded RNG.

- Quality per step = success of its final round. Cost per round from the table.
- Test: one hand-written episode per rule with known expected outputs; verifier flip rate within 0.15 plus or minus 0.01 over 10,000 rounds.
- Commit: `feat(measurement): simulator with retry, propagation, budget and noisy verifier`

### C3-06 Oracles
- Files: `measurement/oracle.py`
- Build:
  - Per-step oracle: cheapest model with `counterfactual = True`, else cheapest.
  - Exact episode oracle: search over model choices for every step and optional escalation, maximising episode quality then minimising cost (and, for RL plots, maximising quality minus lam times cost). Memoise on `(step_index, budget_left rounded, last failing tier, pending retry)`. At most 6 steps x 4 models x 2 rounds, plus retries: small.
- Test: on 1,000 test episodes the episode oracle is at least as good as always-strong, always-cheap, random and the per-step oracle on every episode.
- Commit: `feat(measurement): per-step and exact episode oracles`

### C3-07 Composite scorer
- Files: `measurement/scorer.py`
- Build: `CompositeScorer(cfg).score(...)` producing `Score` with `reward = quality - lam*cost/cost_ref - mu*latency/latency_ref - penalty`. `cost_ref` = mean per-step cost of the most expensive pool model on the train split, frozen in `conf/score.yaml` once computed.
- Commit: `feat(measurement): composite scorer`

### C3-08 Simple baselines
- Files: `measurement/baselines.py`
- Build: always-strongest, always-cheapest, random, episode oracle, each as a `ControllerAPI`.
- Commit: `feat(measurement): simple baseline policies`

### C3-09 Evaluation harness
- Files: `measurement/evaluate.py`, `scripts/evaluate.py`, `scripts/make_figures.py`
- Build: run a `ControllerAPI` over all test skeletons; save per-step predictions to `results/<eval_id>/predictions.parquet`; compute everything in `evaluation.md` from saved predictions only.
- Commit: `feat(measurement): evaluation harness and figures`

## Phase 2: full measurement

### C3-10 Context-free router baseline from the literature
- Build: RouteLLM-style matrix-factorisation or BERT router trained on the train split (single prompts, no context). Complements the in-house `context_free` ablation.
- Commit: `feat(measurement): RouteLLM-style router baseline`

### C3-11 LLM-as-router baseline
- Build: prompt a frontier model with the task, trace and model cards; parse its choice. Report its own routing cost added to episode cost. Run on a fixed 500-episode test subset to bound API spend.
- Why: answers the objection that the controller could be replaced by an LLM, on both cost and accuracy.
- Commit: `feat(measurement): LLM-as-router baseline`

### C3-12 Probe profiler and profile updater
- Build: `ProbeProfiler.run(card, probe_set)` returns a capability vector; `ProfileUpdater` applies `new = (w*prior + n*observed)/(w + n)` per step type, `w = 50`.
- Test: leave-one-model-out: profile a held-out model, add its card, evaluate without retraining.
- Commit: `feat(measurement): probe profiler and profile updater`

### C3-13 Workload-shift test set
- Build: a test skeleton set with different source proportions from training, for calibration-under-shift.
- Commit: `feat(measurement): shifted test set`

## Phase 3: live

### C3-14 Live verifiers
- Build: exact match for QA, unit tests for code, judge model for open text. Same `Score` output.
- Commit: `feat(measurement): live verifiers`

### C3-15 Simulation-vs-live check
- Build: run the same 100 to 300 episodes in the simulator and live; report agreement of success and cost per step type.
- Commit: `feat(measurement): sim-vs-live validation`

## Hand-offs

| You provide | To | When |
|---|---|---|
| Clean data, splits, pool `ModelCard`s | everyone | C3-03 |
| `Simulator` with `counterfactual` | C1 (testing), C2 (rollouts, labels) | C3-05 |
| Episode oracle | C2 (behaviour policy), evaluation | C3-06 |
| `Score` | C2 (rewards) | C3-07 |
| Evaluation harness | everyone | C3-09 |
