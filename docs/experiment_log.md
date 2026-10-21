# Experiment log

One entry per batch of runs. Numbers are validation unless marked test. Single-seed numbers are indicative only.

## 2026-10-08 seed-0 pilot, supervised, RouterBench pool, loss weights 1/1/1
Runs: full-s0, gate_off-s0, gate_stateless-s0, memory_off-s0, context_free-s0, full_attn-s0, memory_off_attn-s0, context_free_attn-s0 (runs/)
Commit: before b67621d  Data version: rb_v1  Scorer: simulator
Observed: val quality 0.682-0.699 at the default thresholds (tau 0.5, escalate above 0.5); always-strongest 0.689 at $0.0070; oracle 0.924 at $0.0013.
Surprises: none recorded.
Changes made: superseded by the 2026-10-21 batch (loss weights changed to the spec's 1.0/0.5/0.2, thresholds now tuned on val).
Next: re-run with spec loss weights.

## 2026-10-21 seed-0 pilot, supervised, three pools, spec loss weights
Runs:
- runs/routerbench/{full,context_free,memory_off,gate_off}-s0 (4 RouterBench models, 19,615 train questions)
- runs/depth/{...}-s0 (19 model x reasoning-depth options from our collected dataset, 633 real train questions)
- runs/depth_aug/{...}-s0 (same 19 options, 633 real + 6,330 synthetic train questions; val/test real and unchanged)
Commit: 93081f6  Data version: rb_v1; depth dataset 2026-10-20 (manifest in data/processed/depth); depth_pool_aug (156cb8a)  Scorer: simulator, verifier flip 0.15
Settings: d=128, 2x3 H/L, loss 1.0/0.5/0.2, AdamW 3e-4, batch 256, <=20 epochs, early stop on val loss (patience 3), EMA.
Thresholds: tau and halt_tau swept on val (tau 0.5-0.99, halt_tau 0.1-0.7); rule (decided 2026-10-21, before any test number):
best val quality with cost <= the random-option baseline's cost, ties within 0.002 -> cheaper.

Observed (val, 2,000 episodes per pool):

| Pool | Run | Epochs | Best val loss | Selected tau / halt | Quality | Cost/episode | Best quality at cost <= $0.004 |
|---|---|---|---|---|---|---|---|
| RouterBench | full | 18 | 0.655 | 0.5 / 0.7 * | 0.647 | $0.00234 | 0.745 |
| RouterBench | context_free | 20 | 0.666 | 0.5 / 0.7 | 0.609 | $0.00181 | 0.749 |
| RouterBench | memory_off | 16 | 0.687 | 0.5 / 0.7 * | 0.631 | $0.00230 | 0.747 |
| RouterBench | gate_off | 15 | 0.658 | 0.5 / 0.7 | 0.645 | $0.00228 | 0.744 |
| depth | full | 8 | 0.187 | 0.90 / 0.1 | 0.951 | $0.00424 | 0.951 |
| depth | context_free | 6 | 0.194 | 0.95 / 0.3 | 0.946 | $0.00395 | 0.946 |
| depth | memory_off | 12 | 0.187 | 0.85 / 0.1 | 0.952 | $0.00301 | 0.952 |
| depth | gate_off | 7 | 0.186 | 0.90 / 0.1 | 0.951 | $0.00419 | 0.951 |
| depth_aug | full | 16 | 0.176 | 0.90 / 0.3 | 0.954 | $0.00333 | 0.955 |
| depth_aug | context_free | 9 | 0.187 | 0.95 / 0.3 | 0.955 | $0.00322 | 0.955 |
| depth_aug | memory_off | 12 | 0.181 | 0.95 / 0.1 | 0.956 | $0.00503 | 0.953 |
| depth_aug | gate_off | 16 | 0.176 | 0.90 / 0.2 | 0.954 | $0.00346 | 0.955 |

\* no setting was as cheap as the random baseline ($0.00228); the rule fell back to the cheapest setting.

Reference points (val): RouterBench always-cheapest 0.228, random 0.553 @ $0.0023, always-strongest 0.689 @ $0.0070, oracle 0.924 @ $0.0013.
Depth pool always-cheapest 0.929 @ $0.00006, random 0.949 @ $0.0054, always-strongest 0.854 @ $0.0217 (breaks the budget), oracle 0.980 @ $0.00006.

- Learned routing works on the RouterBench pool: at <= $0.004 every config reaches 0.744-0.749, above always-strongest (0.689) at about half its cost, on every episode type.
- Context does not help yet. At matched cost, full and context_free tie on every pool. Paired val difference at the earlier max-quality setting, RouterBench: full - context_free = -0.0035 +- 0.0075 (95% CI); no episode type (correlated, retry, propagation, random) favours full.
- Depth pool, real data only: the controllers predicted per-question success no better than a per-task average (AUC on the cheapest option 0.62-0.70 vs 0.72) and context_free overfit after 2 epochs (633 train questions).
- Synthetic training questions fixed that: AUC on the cheapest option 0.77-0.87, lower val loss for every config, no early overfitting, and every config beats random at lower cost.

Surprises:
- With max-quality selection the best escalation threshold sat on the grid edge (0.3, then 0.1); grid widened to 0.1-0.7 before any test run.
- Depth pool val and test are too small to measure episode-type effects: with 80 val questions the only (task, band) groups large enough for correlated episodes are all-correct questions, so every run scores 1.000 on correlated episodes. The same holds for test (179 questions).
- The cost cap is tight on the RouterBench pool, so two configs fell back to their cheapest setting; matched-cost columns are the fair comparison there.

Changes made: loss weights 1/1/1 -> 1.0/0.5/0.2 (spec); thresholds tuned on val; selection rule switched to the cost-capped rule (decision log 2026-10-21).
Scope: this is the Phase 1 prototype (supervised, three heads, one escalation per step). The RL stage, the other heads (tools, memory, depth, verification, topology), calibration and the real verifier are not in these runs.
Next: sanity checks for the null context result (masking invariance, full actually reads its context, retry/propagation change oracle labels); then the analysis freeze and seeds 1-2. Test set not used yet.

## 2026-10-21 sanity checks for the null context result (val only)
Runs: runs/routerbench/full-s0, runs/depth_aug/full-s0 (no new training)
Commit: d416274
1. Masking: tests/test_model.py passes (context_free ignores task, budget and earlier steps; full reads them).
2. Does the trained full model use its context? Removing inputs at inference (RouterBench, tau 0.5, escalate above 0.2;
   paired over 2,000 val episodes): task -0.013 +- 0.008, budget -0.014 +- 0.009 (cost +72%), earlier steps -0.011 +- 0.006,
   memory -0.029 +- 0.010 (cost +47%), all context -0.024 +- 0.010 (cost +81%). depth_aug: memory -0.012 +- 0.004,
   budget -0.008 +- 0.004 (cost x4.6), all context -0.022 +- 0.005 (cost x6.6). full relies on its context; no masking bug.
3. Can history help at all in this simulator? Per-step oracle that sees history vs one that ignores it (val, RouterBench):
   correlated 0.933 vs 0.933, retry 0.891 vs 0.891, propagation 0.903 vs 0.881, random 0.909 vs 0.909. History changes
   which models succeed on 0.4% of steps; retry steps are 3.5% of steps.
Observed: the most any controller could gain from history is about +0.5 points overall (+2.3 on propagation episodes,
which are 20% of episodes), below the +-0.75 point 95% CI of a single-seed paired comparison.
Conclusion: the null full-vs-context_free result is a property of the simulator's dependency rules, not of the controller:
full uses its context, but context_free can reach the same quality from the current question alone. This is the
"synthetic dependencies" threat in the research plan.
Changes made: none.
Next: propose stronger dependencies to C3 (a retried question needs a different model than the one that failed; a failure
affects every later step; budget carried across steps matters), as a contract change agreed by the team, before seeds 1-2.

## 2026-10-21/22 simulator RL pilots v1-v5, seed 0 (RouterBench pool, val)
Runs: runs/routerbench/{full,context_free}-s0/{rl, rl_v3, rl_v4, rl_v5}; diagnostics in this session's notes.
Commits: rl-online 628abcc (v1) ... 1b77e8b (v3) ... e532371 (v4) ... 20d4e56 (v5 calibration)
Observed:
- v1 (default start, quality guard): RL never beat the supervised tau rule (0.735 vs 0.744 at matched cost); the guard
  rolled back every lambda-sensitive policy.
- Root cause 1: the default D13 start sits at the expensive corner (always-strongest, no escalation). A two-number search
  on the untrained policy beat both supervised and 1,000 RL updates at every lambda (lambda 1: 0.534 vs 0.057 reward).
- Root cause 2: the quality guard rejected high-lambda policies (fixed: reward guard, v2).
- v3 (calibrated start): beats supervised at matched cost (0.753 vs 0.745 full; 0.768 vs 0.749 context_free), but every
  RL check rolled back: updates made lambda 0.3-1.0 worse. Lower lr and no anchor did not help.
- Root cause 3: the sampling policy was near-uniform (lambda 1: sample 0.33 vs serve 0.54 reward). v4 adds a learnable
  sharpness k (serve decisions unchanged): no rollbacks, reward up at every lambda (lambda 2: -0.058 -> 0.470).
- v5 (guard reference from the supervised start, zero-tolerance refit, sharpness grid to 50): context_free 0.772 vs
  supervised 0.749 at <= $0.004, best checkpoint update 750, 0 rollbacks, lambda 2 reward 0.455. full/kimi/luna finishing.
Changes made: calibrated start, sharpness, reward guard, guard reference before calibration (D13 deviations).
Next: v5 frontier for all four runs; online experiments from v5.

## 2026-10-22 online RL experiments, seed 0 (rounds v1-v5; paired held-out evaluation from v3)
Runs: results/online_sim{,_v2,_v3,_v4,_v5}; scripts eval/online_sim.py; final checkpoints only (intermediate v*.pt removed,
15.9 GB, after the disk filled; learner now keeps 3).
Observed, round by round (each round fixed a real bug the previous one exposed):
- v1 shift: 30 rollbacks even at lr 1e-5 -> cards drifted toward 0.5 (verifier noise not de-biased) and the canary compared
  with a reference taken once at the start. Fixed: de-biased cards, teacher-relative canary.
- v2 shift: learn vs frozen differences of +-0.03 were sampling noise (~+-0.04 CI per lambda cell). Fixed: paired greedy
  evaluation of final weights on 2,000 held-out shifted episodes.
- v2/v3 paired: online learning drifted to cheaper choices at every lambda (lambda 0.1: $0.0019 vs $0.0038, quality 0.760
  vs 0.783). Cause: the 15%-flip verifier shrinks quality differences by 30% while costs are exact (learner optimises
  lambda/0.7). Fixed: rewards de-biased with the known flip rate.
- v3 depth new-model runs: mean rewards above 1 (impossible). Cause: reward hacking - escalation is triggered by the
  verifier and re-rolls it, so the decision verdict reads high; the learner escalated to re-roll the checker (Kimi: 3.2x
  cost, lower true quality). Fixed: rewards from an independent audit verdict of the final answer.
- Review fix wave (two sessions): per-pool cost_ref everywhere (depth results before it are invalid), escalation by card
  cost (joiners are appended), bonus in pre-sharpness units, start_online wiring, canary on reward.
- v4 paired (after audit fix; depth runs invalid, wrong cost_ref): shift and new models mostly small losses (-0.002 to
  -0.09 at some lambdas), new-model shares near 0 (Kimi and Luna are not better than existing options, so ignoring them
  is correct).
- v5 paired, context_free shift (2 of 8 done): lr 1e-5: +0.022* at lambda 0.6, -0.063* at 0.1, others ns; lr 1e-3: 6
  rollbacks, mixed. Full/new-model v5 still running.
Surprises: every online round until v5 was dominated by a measurement/reward bug, not by learning.
Changes made: see D13 deviations (docs/IMPLEMENTATION_PLAN_neural_core.md).
Next: drift experiments with a static arm (results/online_drift) to separate card adaptation from weight learning.

## 2026-10-22 context claim on a simulator with hidden episode difficulty, seed 0 (val)
Runs: runs/routerbench_hard/{full,context_free}-s0 (hard_p 0.3: a hard episode fails the cheaper half of the models on
every step; only earlier results reveal it). Commit aad7259.
Observed: full val loss 0.627 vs context_free 0.654; at the selected thresholds full 0.665 quality ($0.0042) vs
context_free 0.616 ($0.0038); best quality at <= $0.006: 0.752 vs 0.744. Oracle 0.915, random 0.492.
Interpretation: first sign that workflow context helps when history is informative. Not a result yet: H2 is decided by
the paired test contrast over 3 seeds (docs/analysis_plan.md).
Next: seeds 1-2 (running), then eval.final_test.
