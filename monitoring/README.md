# Training monitoring: MLflow + Grafana

Everything is local: MLflow writes to `monitoring/mlflow/mlflow.db` (git-ignored), and Grafana reads that
file read-only. Nothing leaves the machine.

## Log a run

```bash
python -m train.train --config full --seed 0 --data-dir data/processed --out-dir runs/routerbench --track
python -m eval.tau_sweep runs/routerbench/full-s0 --track      # adds the chosen thresholds to the same run
python -m monitoring.backfill                                  # import finished runs from their logs (idempotent)
```

`--track` logs parameters, every per-epoch metric (losses per head, val quality and cost, gap closed,
choice spread, escalation, grad norm, lr) and alarms. Without `--track`, or without mlflow installed,
nothing is written.

## Look at it

| Tool | Start | Open |
|---|---|---|
| MLflow UI (compare runs, params, artifacts) | `mlflow ui --backend-store-uri sqlite:///monitoring/mlflow/mlflow.db --host 127.0.0.1 --port 5000` | http://127.0.0.1:5000 |
| Grafana (live dashboard, refresh 30 s) | `docker compose -f monitoring/docker-compose.yml up -d` | http://127.0.0.1:3000 (anonymous view; admin / admin to edit) |

Stop Grafana with `docker compose -f monitoring/docker-compose.yml down`.

## Dashboard

`neural-core: training monitor` (rebuild with `python monitoring/grafana/build_dashboard.py`):

- **Supervised (x = epoch):** val quality, val cost, train vs val loss, loss per head, oracle gap closed,
  choice entropy (collapse alarm), escalation rate, grad norm, lr, alarms per epoch, a table of the
  thresholds each run selected on val, and the alarm log.
- **Simulator RL (x = update):** val reward and quality at a chosen lambda, mean val reward, sample-mode
  train return, policy entropy, and the last val check per lambda.

Selectors at the top: pool, supervised runs, RL runs, lambda index.
