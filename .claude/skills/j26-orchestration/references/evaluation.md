# Evaluation

Every number in the reports comes from `results/` via `scripts/make_figures.py`. Nothing is computed by hand.

What each output is evidence for (research questions, hypotheses, success criteria, statistics) is in `research-plan.md`. How results are watched during the project is in `ui-and-monitoring.md`.

## Policies compared

| Policy | Kind | Built in |
|---|---|---|
| `full` | proposed controller | C1, trained by C2 |
| `context_free` | same model, context masked (decisive comparison) | C1, C2 |
| `gate_off`, `gate_stateless`, `memory_off` | ablations | C1, C2 |
| `full_attn`, `memory_off_attn`, `context_free_attn` | architecture ablation (gated-attention core, D12); compared within family | C1, C2 |
| Always-strongest, always-cheapest, random | simple baselines | C3-08 |
| RouteLLM-style router | literature context-free router | C3-10 |
| LLM-as-router | prompted frontier model, its routing cost included | C3-11 |
| Episode oracle | ceiling | C3-06 |

Each learned policy: supervised stage, and RL stage for `full` and `context_free`. Seeds 0, 1, 2.

## Metrics

| Metric | Definition |
|---|---|
| Quality | mean over episodes of the fraction of steps whose final round succeeded |
| Cost | mean episode spend in USD (price date stated) |
| Frontier | quality vs cost as tau (supervised) or lam (RL) is swept; area under it between the cheapest and strongest cost |
| Gap recovered | (method - cheapest) / (oracle - cheapest) quality at matched cost |
| ECE | expected calibration error of the chosen model's confidence, 15 bins, test and shifted test |
| Escalation rate | share of steps with a round 1, by difficulty band and by budget fraction |
| Latency overhead | controller time / end-to-end time, p50 and p95 |

All headline numbers: mean and 95% confidence interval over 3 seeds.

## Required outputs

| ID | Output | Shows |
|---|---|---|
| O1 | Frontier plot, all policies, oracle line | the main trade-off |
| O2 | Per-episode-type table, `full` vs `context_free` | where context helps; random-chain row is the control |
| O3 | Ablation table at matched cost | contributions of memory, adaptive rate, CfC form |
| O4 | Baselines table | standing against simple, literature and LLM routers |
| O5 | Supervised vs RL table | what episode-level optimisation adds, especially on budget-pressure episodes |
| O6 | ECE on test and shifted test | calibration and its robustness |
| O7 | Escalation vs difficulty and vs budget | halting behaves sensibly |
| O8 | Residual plot | two-timescale core is computing, not stalled (attn: residuals averaged over valid tokens only) |
| O9 | Surprise trace for one episode | failure, surprise spike, gate opening, escalation |
| O10 | Latency table | overhead requirement (under 5%) |
| O11 | Leave-one-model-out | new model added with no retraining |
| O12 | Sim vs live agreement (Phase 3) | simulator validity |
| O13 | Learning curve at 5k, 20k, 50k episodes (optional) | data sufficiency |

## Expected pattern and what to do if it is absent

| Observation | Meaning | Action |
|---|---|---|
| `full` beats `context_free` on correlated, retry, propagation; ties on random | claim supported | report |
| `full` beats `context_free` on random chains too | leakage | check masking test and split |
| No gain anywhere | dependencies not changing labels, or memory not trained | inspect retry episodes' oracle labels; confirm episodes are unrolled in order during training |
| `full` equals `memory_off` | memory unused | check `W` actually changes across steps; check k=2 not larger |
| `full` equals `gate_stateless` | surprise-gating helps, CfC form not decisive | report honestly; it is still a result; do not call the memory "liquid-gated" |
| `full_attn` differs from `full` | architecture effect, confounded by about 45% more parameters and about 2.5x latency | report within-family context gaps as the headline; state the confound for any cross-family claim |
| Test far above val | leakage | re-check split disjointness |

## Matched-cost comparison

To compare two policies at "matched cost", interpolate each policy's frontier at the same cost points (the 25th, 50th and 75th percentile of the cheapest-to-strongest cost range) and compare quality there.
