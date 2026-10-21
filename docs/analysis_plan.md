# Frozen analysis plan

Fixed on 2026-10-22 after the seed-0 pilots and before seeds 1–2 (git tag `analysis-freeze`). Nothing below
changes after seeing seed 1–2 or test results. Everything not listed here is exploratory and reported as such.

## Data and runs

| Pool | Simulator | Configs | Seeds |
|---|---|---|---|
| RouterBench, 4 models (`data/processed`) | standard | `full`, `context_free` | 0, 1, 2 |
| RouterBench, 4 models | hidden episode difficulty, `hard_p = 0.3` | `full`, `context_free` | 0, 1, 2 |
| Depth pool + synthetic train questions (`data/processed/depth_pool_aug`) | standard | `full`, `context_free` | 0, 1, 2 |

`memory_off` and `gate_off` stay seed-0 ablations (indicative only). Simulator RL (v5) and the online
experiments are seed 0 and reported as indicative.

## Procedure

1. Train (supervised, spec loss weights 1.0 / 0.5 / 0.2, early stop on val loss).
2. Choose thresholds on **val** only (`eval/tau_sweep.py`): best val quality among settings whose val cost is at most
   the random-option baseline's, ties within 0.002 to the cheaper setting.
3. Evaluate **once** on the **test** split at those thresholds (`eval/final_test.py`), with always-cheapest,
   always-strongest, random and the exact episode oracle on the same test episodes.

## Hypotheses and tests

| | Claim | Test | Supported if |
|---|---|---|---|
| H1 | The learned controller routes better than fixed policies | test quality and cost of `full` vs always-strongest | higher quality at lower cost, seed-mean 95% t-interval on quality excluding always-strongest's value |
| **H2 (primary)** | Workflow context helps where history is informative | paired test difference `full − context_free` on the **hidden-difficulty** simulator, per test episode, averaged per seed; 95% t-interval over the 3 seed means | interval above 0 |
| H2b | Same on the standard simulator | as H2 | reported either way; a tie is expected (history changes outcomes on 0.4% of steps) |
| H3 | The controller learns when to think deeper | depth pool: thinking tokens per step by difficulty band; AUC of predicted success for the cheapest option vs the task-average baseline | tokens increase with band; AUC above baseline (seed-mean interval) |
| H4 (indicative) | Online learning helps when there is something to learn | drift experiments: held-out paired reward of learn / frozen (cards adapt) / static arms per λ | reported per λ with paired 95% CIs; no claim beyond seed 0 |

Episode-type breakdowns (correlated, retry, propagation, random) are reported for H2/H2b. The random-chain
episodes are the control: `full` must not beat `context_free` there; if it does, check for leakage first.

## Statistics

- Seed aggregate: mean ± t(0.975, n − 1) · sd / √n over the per-seed values (n = 3).
- Paired contrasts: per test episode within a seed, averaged per seed, then the seed t-interval.
- Costs are list prices with their price dates; the depth pool uses its own cost reference (`cost_ref.json`).
- No multiple-comparison correction: one primary hypothesis (H2); everything else is secondary and labelled.

## What would count as a negative result

H2 is not supported if the interval includes or is below 0. That is reported as a finding, not tuned away:
the controller then routes as well as a per-query router but does not exploit workflow context under this
simulator.
