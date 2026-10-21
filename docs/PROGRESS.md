# Progress and next steps

Snapshot: 2026-10-22, ~10:00. Branch `rl-online` (worktree `neural-core-worktrees/rl-online`); `main` is 17 commits
ahead of `origin/main` and does not yet contain `rl-online`. Nothing is pushed. Owner of this file: C1.

## Where things stand

| Area | State |
|---|---|
| Controller (H/L core, liquid-gated episode memory, card-matching heads, λ-conditioned policy, learnable sharpness) | built, tested |
| Reasoning-depth dataset: 892 real questions × 19 model/thinking options (2026 models), $26.05 spend, manifest of every parameter | done (`data/processed/depth`) |
| Synthetic depth training questions (hot-deck, 6,330; real val/test untouched) | done |
| Supervised training, τ sweep with the cost-capped rule, 12 seed-0 runs on 3 pools | done |
| Simulator RL: calibrated start, sharpness, reward guard (v5) | done; v5 runs finishing |
| Online learner (Part 2 Tasks 1–7): session, replay, importance-weighted updates, canary, new models joining, drift experiments | built, tested |
| Review fix wave (findings 1–10 + minors, two sessions merged) | done |
| Live pilot script (Task 8) with spend cap and dry run | built, dry-run tested; **not run live** |
| Simulator with hidden episode difficulty (opt-in) for the context claim | built; seed 0 trained |
| Frozen analysis plan, tag `analysis-freeze` | done (`docs/analysis_plan.md`) |
| Monitoring: MLflow + Grafana | done (`monitoring/README.md`, on `main`) |

## Headline results so far (seed 0, val unless stated; indicative only)

1. **Routing works.** RouterBench pool: ~0.75 quality at ~$0.004/episode vs always-GPT-4 0.69 at $0.007.
2. **Thinking adapts to difficulty.** Depth pool: ~56 thinking tokens per step on easy questions vs ~4,700 on hard
   ones; with synthetic training data the success-prediction AUC for the cheapest option is 0.85 (task-average
   baseline 0.72); beats a random option at ~60% of its cost.
3. **Simulator RL now improves on supervised.** v5 `context_free`: 0.772 vs 0.749 at ≤ $0.004/episode, no
   rollbacks, best checkpoint at update 750; λ = 2 reward −0.19 (v3) → +0.455.
4. **Context, standard simulator: tie.** `full` ≈ `context_free`; history changes outcomes on 0.4% of steps there.
5. **Context, hidden-difficulty simulator (first sign):** `full` 0.665 vs `context_free` 0.616 val quality, val loss
   0.627 vs 0.654. To be confirmed on test across 3 seeds (H2, the primary hypothesis).
6. **Online learning: safe, small or no gain so far.** Paired held-out differences are within a few points either
   way; several real bugs were found and fixed on the way (below). Drift experiments, where a static router must
   lose, are queued to answer this properly.

## Findings worth a paragraph in the thesis

- **Reward hacking of a noisy verifier.** When escalation is triggered by the verifier and the same verifier
  scores the reward, flagged rounds get re-rolled and the reward reads high; the online learner learned to
  escalate to re-roll the checker (3.2× cost for lower true quality). Fix: reward from an independent audit verdict.
- **Verifier noise skews the cost trade-off.** A 15%-flip verifier shrinks quality differences by 30% while costs
  stay exact, so a learner optimises λ/0.7 and drifts to cheap models. Fix: de-bias with the known flip rate.
- **Standard routing simulators cannot test workflow context** (history matters on 0.4% of steps).
- **RL needs a calibrated start and a sharp enough sampling policy**, otherwise updates make a good policy worse.

## Running right now (CPU only, no API spend)

| Job | Output | Check |
|---|---|---|
| RL v5 ×4 → online shift / new-model experiments + paired evals | `runs/*/rl_v5`, `results/online_sim_v5/` | `ls results/online_sim_v5/*/*/paired.json` |
| Drift experiments (price shock, quality drop, synthetic cheaper twin; learn/frozen/static) | `results/online_drift/` | `ls results/online_drift/*/drift_*/summary.json` |
| Seeds 1–2: `full`, `context_free` × {RouterBench, hidden-difficulty, depth_aug} | `runs/{routerbench,routerbench_hard,depth_aug}/*-s{1,2}` | `ls runs/*/*-s[12]/tau.json` (12 when done) |

All launched from the worktree (`runs/launch_v5.sh`, `runs/launch_drift.sh`, `runs/seed_jobs.txt` via
`runs/seed_job.sh`). Stop everything with `pkill -f "train.train|train.rl|eval.online_sim|eval.tau_sweep"`.

## To do, in order

1. **When the seed jobs finish:** run the frozen test evaluation per pool
   `python -m eval.final_test --pool-dir <runs/pool> --configs full context_free --seeds 0 1 2 --out results/final/<pool>`
   for `routerbench`, `routerbench_hard` (H2, primary) and `depth_aug`. Do not change the analysis.
2. **Read the online results** (`results/online_sim_v5`, `results/online_drift`) and log them; the drift arms say
   whether weight learning adds anything beyond card updates.
3. **Write the results** into `docs/experiment_log.md` (with CIs) and the build guide's status table.
4. **Merge `rl-online` into `main`** (resolve: `train/train.py` monitoring hooks on `main` vs exclusions/hard_p on
   `rl-online`; `eval/tau_sweep.py`), then push / open a PR when you decide.
5. **Live pilot (Task 8)** — needs your approval and a dollar cap. Dry run first:
   `python scripts/online_live.py --init runs/depth_x_kimi/full-s0/rl_v5 --join kimi-k2.6@on --episodes 50 --dry-run`.
6. **Team items:** propose the hidden-difficulty simulator (or real agent traces) to C3 as the benchmark for the
   context claim; Sonnet's 4 depth levels need the Anthropic usage limit raised (~$40–100 to collect).
7. Optional: depth-pool hard questions (~$25–35 per 1,000) to firm up the depth results.

## Decisions waiting for you

- Live pilot: go / no-go, provider list, dollar cap.
- Push to GitHub: `main` directly or a PR.
- Whether to keep `memory_off` / `gate_off` as seed-0-only ablations (current plan) or run them on 3 seeds.
