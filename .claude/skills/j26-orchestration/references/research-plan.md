# Research Plan: Questions, Hypotheses and Expected Results

This file says what the experiments are for, what we expect to see, what would count as support or refutation, and how results are analysed. Fix it before the final runs (seeds 1 and 2); after that, changes are recorded in the experiment log with a reason.

## Contents

1. Research questions and hypotheses
2. Success criteria
3. Statistical analysis
4. Analysis freeze (pilot then final)
5. Threats to validity
6. Experiment log and decision log

## 1. Research questions and hypotheses

Expected direction is stated; magnitudes are not predicted in advance because no prior work uses this exact setting. The seed-0 run is the pilot that sets expectations for seeds 1 and 2.

| RQ | Question | Hypothesis | Evidence (outputs in `evaluation.md`) |
|---|---|---|---|
| RQ1 | Does conditioning on workflow state improve the quality-cost trade-off over context-free routing? | H1: `full` has larger frontier area than `context_free` on correlated, retry and propagation episodes, and no difference on random chains | O1, O2 |
| RQ2 | Does the episode memory contribute beyond the explicit trace? | H2: `full` beats `memory_off`, with a larger gap on longer episodes (5 to 6 steps), where history exceeds the k = 2 trace | O3 split by length |
| RQ3a | Does an adaptive update rate help? | H3a: `full` and `gate_stateless` both beat `gate_off` | O3 |
| RQ3b | Does the liquid (CfC) form matter beyond surprise-gating? | Exploratory, no directional prediction. `full` vs `gate_stateless` is reported either way; equality means the gain comes from surprise-gating (as in Titans) and the CfC form is a defensible, not decisive, choice. The memory may be called "liquid-gated" only if `full` > `gate_stateless` | O3 |
| RQ4 | Does episode-level RL improve over myopic supervised labels? | H4: RL beats supervised on budget-tight episodes; small or no difference elsewhere | O5 |
| RQ5 | Does halting allocate effort sensibly? | H5: escalation rate rises with difficulty band and falls as budget fraction drops | O7 |
| RQ6 | Is confidence calibrated, including under workload shift? | H6: ECE after temperature scaling is lower than uncalibrated; ECE on the shifted set rises but stays below the uncalibrated value | O6 |
| RQ7 | Can new models be used without retraining? | H7: in leave-one-model-out, adding the held-out card at test time recovers most of the quality of a controller trained with it | O11 |
| RQ8 | Do the H and L modules specialise when they read different inputs? | Exploratory: H state predicts task-level variables better than L; L predicts execution-level variables better than H; both above an untrained control | specialisation probe |
| RQ9 | How does the controller compare with existing routers, including an LLM used as a router? | H9: the controller matches or beats the RouteLLM-style router at equal cost; the LLM-as-router may reach similar quality but at a much higher total cost once its routing calls are counted | O4 |
| RQ10 | Does the core architecture matter? | Exploratory architecture ablation: `mlp` vs `attn` (gated attention over role tokens, D12). The context effect is tested within each family (`full` vs `context_free`; `full_attn` vs `context_free_attn`). Any cross-family statement must state the confound: the attn family has about 45% more parameters and about 2.5x the latency | O3, O8, O10 |

## 2. Success criteria

| Hypothesis | Supported if | Refuted if |
|---|---|---|
| H1 | Paired difference in frontier area (`full` minus `context_free`) on the pooled dependent types has a 95% CI above 0, and the random-chain CI includes 0 | Dependent-type CI includes or is below 0 |
| H1 validity | Random-chain CI includes 0 | Random-chain CI above 0: treat as leakage until shown otherwise |
| H2 | `full` minus `memory_off` CI above 0 on 5 to 6 step episodes | CI includes 0 on all lengths |
| H3a | `gate_off` below both adaptive gates, CI excludes 0 | otherwise |
| H4 | RL minus supervised CI above 0 on budget-tight episodes | otherwise |
| H5 | Spearman correlation of escalation rate with difficulty band positive, with budget fraction positive, both p < 0.05 | otherwise |
| H6 | Calibrated ECE below uncalibrated ECE on both test and shifted test | otherwise |
| H7 | Held-out-card controller within 2 quality points of the retrained one at matched cost | gap larger |

A refuted hypothesis is still a result. Report it with the same care as a supported one.

## 3. Statistical analysis

- **Paired design.** All policies are evaluated on the same test skeletons with the same simulator seeds, so differences are paired by episode.
- **Within a seed:** paired bootstrap over test episodes (10,000 resamples) for the difference in the metric. Gives a CI per seed.
- **Across seeds:** mean of the three per-seed differences with a t-based 95% CI (df = 2). Headline numbers use this.
- **Multiple comparisons:** the ablation family (H2, H3a, and the RQ3b comparison) uses Holm correction.
- **Architecture families:** headline comparisons stay within a family (`mlp` or `attn`); `mlp` vs `attn` is an exploratory ablation reported with its parameter and latency confound.
- **Frontier area:** trapezoidal area under quality vs cost between the always-cheapest and always-strongest costs, after interpolating each policy's curve onto a common grid of 20 cost points.
- **Matched cost:** quality interpolated at the 25th, 50th and 75th percentiles of the cheapest-to-strongest cost range.
- Implement all of this once, in `measurement/stats.py`, used by both the dashboard and `make_figures.py`.

## 4. Analysis freeze (pilot then final)

1. Run seed 0 for every config. Inspect results in the dashboard. Fix bugs only.
2. Record in the experiment log: metrics, operating points, any changes to this file and why.
3. Freeze this file (tag the commit `analysis-freeze`).
4. Run seeds 1 and 2. Do not change the analysis after seeing them. Any extra analysis is labelled exploratory.

This guards against choosing the analysis that flatters the result after seeing it.

## 5. Threats to validity

| Threat | Why it matters | Mitigation |
|---|---|---|
| Synthetic dependencies | Retry and propagation rules are designed by us; context might help only because we built episodes where it does | Random-chain control; TwinRouterBench check on real agent trajectories; report results per episode type |
| Old model pool | RouterBench models date from 2024 | Next-phase replication on LLMRouterBench (33 recent models) |
| Noisy-verifier assumption | 15% flip rate is a modelling choice | Sensitivity run at 0.05 and 0.30 for `full` and `context_free` |
| Price dating | Prices change; cost conclusions depend on them | Report price dates; sensitivity with current prices for the pool |
| Single benchmark family | Results may not transfer | TwinRouterBench and sim-vs-live sample |
| Train split as probe set | Capability vectors fit on the same prompts as training | Leave-one-model-out uses a separate probe split |
| Implementation bugs flattering the method | Masking or split errors inflate gains | Masking invariance test; split disjointness test; random-chain row |

## 6. Experiment log and decision log

`docs/experiment_log.md`, one entry per batch of runs:

```markdown
## 2026-10-20 seed-0 pilot, supervised, all configs
Runs: full-s0, gate_off-s0, gate_stateless-s0, memory_off-s0, context_free-s0, full_attn-s0, memory_off_attn-s0, context_free_attn-s0
Commit: <hash>  Data version: rb_v1  Scorer: s1
Observed: <key numbers from the dashboard>
Surprises: <anything unexpected>
Changes made: <none / what and why>
Next: <what runs next>
```

`docs/decision_log.md` records design decisions with the evidence and alternatives considered, matching the proposal's decision-log format (date, claim, evidence, alternatives, decision, supervisor discussion).
