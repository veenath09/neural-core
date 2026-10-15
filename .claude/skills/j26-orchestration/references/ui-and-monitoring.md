# UI and Monitoring

## Contents

1. What the UI is for
2. Stack
3. Dashboard pages
4. What every run must log
5. Alarms
6. Live demo console (Phase 3)
7. Build tasks

## 1. What the UI is for

The project has two UIs, both read-only views of saved data:

| UI | Phase | Users | Purpose |
|---|---|---|---|
| Research dashboard | 0 onward | the team, supervisor | Watch training, check results against hypotheses, inspect episodes, see the model pool |
| Live demo console | 3 | panel, supervisor | Send a real multi-step workflow through the middleware and watch each decision |

Neither UI computes results of its own. Numbers come from `ui/data.py` and `measurement/stats.py`, the same code that produces the report's figures, so the dashboard and the report cannot disagree.

## 2. Stack

| Concern | Choice | Reason |
|---|---|---|
| App | Streamlit | Python only; one file; no front-end build |
| Query layer | DuckDB over Parquet and JSON | No database server in Phases 0 to 2; SQL views; can attach PostgreSQL in Phase 3 with the same view names |
| Charts | Plotly | Interactive zoom and hover |
| Training curves (optional) | TensorBoard via `RunLogger(tensorboard=True)` | Live curves during long runs |

Starter code: `assets/ui/data.py` and `assets/ui/dashboard.py` (tested headless with Streamlit's `AppTest` on synthetic runs and on an empty repo). Copy to `ui/`. Run with:

```bash
ROOT=. streamlit run ui/dashboard.py
```

## 3. Dashboard pages

| Tab | Shows | Reads | Answers |
|---|---|---|---|
| Overview | Run table and status counts; alarm and rollback events; validity check (random-chain gap) | `runs`, `events`, `by_type` | Is anything broken or leaking right now? |
| Training | Metric curves per run, train and val, selectable | `metrics` | Is each run learning; did anything collapse? |
| Results | O1 frontier, O2 per-type table with seed CIs, O3 ablations | `frontier`, `by_type` | Do the hypotheses hold? |
| Calibration | O6 reliability diagram and ECE per evaluation set | `predictions` | Can confidence be trusted? |
| Episodes | O9 step-by-step view: model chosen vs oracle, verifier signal, surprise, gate rate, budget | `predictions` | Does the mechanism behave as described? |
| Model pool | Cards, prices with dates, capability heatmap | `catalogue` | What can each model do? |

Planned additions (Phase 2): Halting tab (O7 escalation vs difficulty and budget), Baselines tab (O4), RL tab (return per lambda, entropy, baseline buckets), Residuals tab (O8).

## 4. What every run must log

All training and evaluation scripts use `common/tracking.py` `RunLogger`. Metric names below are fixed so the dashboard finds them.

### Supervised (C2)

| Metric | Split | Every | Meaning |
|---|---|---|---|
| `loss`, `loss_model`, `loss_halt`, `loss_complexity` | train, val | 50 steps / epoch | Total and per-head loss |
| `val_frontier_area` | val | epoch | Primary selection metric |
| `val_quality_at_mid`, `val_cost_at_mid` | val | epoch | Matched-cost point |
| `ece` | val | epoch | Calibration |
| `gate_rate_mean`, `gate_rate_std` | train | 50 steps | Gate activity; std near 0 means the gate collapsed |
| `surprise_norm_mean` | train | 50 steps | Should be higher on failed rounds |
| `W_norm_mean` | train | 50 steps | Memory size; steady growth means instability |
| `bp_horizon` | train | epoch | Truncated backprop horizon in use |
| `lr` | train | 50 steps | Learning rate |

### RL (C2)

| Metric | Meaning |
|---|---|
| `return_mean_lam{0.1,0.3,0.6,1.0,2.0}` | Mean episode return per lambda |
| `entropy_model`, `entropy_halt` | Policy entropy per head |
| `adv_std` | Advantage spread before normalisation |
| `baseline_b{band}_l{lam}` | Bucket values |
| `action_change_vs_sup` | Share of greedy actions that differ from the supervised checkpoint |
| `val_frontier_area` | As above, every 250 updates |

Events: `rollback`, `early_stop`, `checkpoint`, `alarm`.

### Evaluation (C3)

Writes `results/<eval_id>/predictions.parquet` (schema in `data-and-storage.md`) and logs `frontier_area`, `ece` per policy into a run with `stage = "eval"`.

### Controller diagnostics (C1)

`DecisionRecord.diagnostics` carries `surprise_norm`, `gate_rate`, `latency_ms`; the evaluation harness copies them into predictions.

## 5. Alarms

`learning/monitor.py` checks these after each logging interval and writes an `alarm` event. The Overview tab shows them.

| Alarm | Condition | Likely cause |
|---|---|---|
| Loss not finite | NaN or inf in any loss | TTT update exploding: clip rate times update, lower lr |
| Gate collapsed | `gate_rate_std < 0.01` for 1,000 steps (CfC or stateless configs) | Gate saturated; check surprise scale and init |
| Memory exploding | `W_norm_mean` up more than 10x from start | Missing clip or normalisation in the TTT step |
| Entropy collapse | `entropy_model < 0.05` in RL before update 1,000 | Beta too low or advantages too large |
| Val regression | `val_frontier_area` below the supervised checkpoint by more than 5% | RL instability; safety guard should roll back |
| Leakage suspicion | random-chain gap (full minus context_free) above 0.02 | Masking or split bug |
| Val-test gap | test quality more than 0.05 above val | Split leakage |

## 6. Live demo console (Phase 3)

Owner: C4. A Streamlit page added as a seventh tab, enabled when `LIVE=1`.

| Element | Behaviour |
|---|---|
| Workflow picker | Choose a scripted multi-step task (e.g. a 4-step research-and-compute workflow) |
| Budget and lambda inputs | Sent as `x-budget-usd`, `x-cost-weight` |
| Run button | Sends each step through the gateway with one `x-episode-id` |
| Step timeline | Per step: model chosen, rounds, verifier result, cost, controller latency, surprise and gate rate |
| Comparison toggle | Re-runs the same workflow with `context_free` to show the difference live |
| Totals | Episode cost, quality (if references exist), controller overhead percentage |

Phase 3 data: the console reads the same view names, created by DuckDB over PostgreSQL (`ATTACH 'dbname=... ' AS pg (TYPE postgres)`) instead of Parquet.

## 7. Build tasks

| ID | Owner | Task | Done when |
|---|---|---|---|
| UI-01 | all (C2 first) | Use `RunLogger` in every training and eval script with the metric names above | Runs appear in Overview and Training |
| UI-02 | C3 | Copy `assets/ui/` to `ui/`; serve on the team machine | Dashboard opens with real runs |
| UI-03 | C3 | `measurement/stats.py` (bootstrap, seed CI, Holm, frontier area); dashboard uses it | Results tab matches `make_figures.py` exactly |
| UI-04 | C2 | `learning/monitor.py` alarms | A forced NaN produces an alarm event |
| UI-05 | C3 | Halting, Baselines, RL, Residuals tabs | O4, O5, O7, O8 visible |
| UI-06 | C4 | Live demo console | A 4-step workflow runs end to end from the UI |
