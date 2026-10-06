"""
Purpose:  Generate the Grafana training dashboard (monitoring/grafana/dashboards/training.json) from
          readable panel definitions, querying the MLflow SQLite store through frser-sqlite-datasource.
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   metric names written by train.tracking (train.train, eval.tau_sweep, train.rl, monitoring.backfill).
Outputs:  monitoring/grafana/dashboards/training.json (provisioned; Grafana reloads it within 30 s).
Notes:    Curves are plotted against the metric step (epoch for supervised runs, update for RL), not
          wall-clock time, so backfilled and live runs line up. Each query returns long rows
          (step, run, value) and a partitionByValues transformation splits them into one series per run.

    python monitoring/grafana/build_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "frser-sqlite-datasource", "uid": "mlflow"}
LAMS = (0.1, 0.3, 0.6, 1.0, 2.0)
OUT = Path(__file__).resolve().parent / "dashboards" / "training.json"


def _target(sql: str, ref: str = "A") -> dict:
    return {"refId": ref, "datasource": DS, "queryType": "table", "rawQueryText": sql, "queryText": sql, "timeColumns": []}


def series_sql(metrics: list[str], run_var: str, label_with_key: bool = False) -> str:
    keys = ", ".join(f"'{k}'" for k in metrics)
    label = "r.name || '  ' || m.key" if label_with_key else "r.name"
    return (f"SELECT m.step AS step, {label} AS run, m.value AS value FROM metrics m "
            f"JOIN runs r ON r.run_uuid = m.run_uuid WHERE m.key IN ({keys}) "
            f"AND r.name IN (${{{run_var}:sqlstring}}) AND r.lifecycle_stage = 'active' ORDER BY run, step")


def trend(title: str, metrics: list[str], run_var: str, x: int, y: int, w: int = 12, h: int = 8, unit: str = "",
          label_with_key: bool = False, description: str = "", xlabel: str = "epoch") -> dict:
    return {
        "type": "trend", "title": title, "description": description, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [_target(series_sql(metrics, run_var, label_with_key))],
        # the trend panel takes one frame: split by run, then outer-join the series back on step
        "transformations": [{"id": "partitionByValues",
                             "options": {"fields": ["run"], "keepFields": False, "naming": {"asLabels": True}}},
                            {"id": "joinByField", "options": {"byField": "step", "mode": "outer"}}],
        "fieldConfig": {"defaults": {"unit": unit, "displayName": "${__field.labels.run}",
                                     "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "auto",
                                                "axisLabel": ""}},
                        "overrides": [{"matcher": {"id": "byName", "options": "step"},
                                       "properties": [{"id": "custom.axisLabel", "value": xlabel},
                                                      {"id": "unit", "value": "none"}, {"id": "decimals", "value": 0}]}]},
        "options": {"xField": "step", "legend": {"displayMode": "table", "placement": "right", "showLegend": True},
                    "tooltip": {"mode": "multi"}},
    }


def last(key: str) -> str:
    return (f"(SELECT value FROM metrics x WHERE x.run_uuid = r.run_uuid AND x.key = '{key}' "
            f"ORDER BY x.step DESC, x.timestamp DESC LIMIT 1)")


def tag(key: str) -> str:
    return f"(SELECT value FROM tags t WHERE t.run_uuid = r.run_uuid AND t.key = '{key}')"


def table(title: str, sql: str, x: int, y: int, w: int = 24, h: int = 8, description: str = "") -> dict:
    return {"type": "table", "title": title, "description": description, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [_target(sql)],
            "fieldConfig": {"defaults": {"decimals": 4}, "overrides": []},
            "options": {"showHeader": True, "cellHeight": "sm"}}


def row(title: str, y: int) -> dict:
    return {"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


def variable(name: str, label: str, sql: str, multi: bool = True) -> dict:
    # Default to All: with an empty selection Grafana renders ${var:sqlstring} as '' and every panel is empty.
    current = {"selected": True, "text": ["All"], "value": ["$__all"]} if multi else {}
    return {"name": name, "label": label, "type": "query", "datasource": DS, "query": sql,
            "definition": sql, "refresh": 2, "multi": multi, "includeAll": multi, "current": current,
            "sort": 1, "options": []}


def runs_of(stage: str) -> str:
    return ("SELECT r.name FROM runs r JOIN tags s ON s.run_uuid = r.run_uuid AND s.key = 'stage' "
            "JOIN tags p ON p.run_uuid = r.run_uuid AND p.key = 'pool' "
            f"WHERE s.value = '{stage}' AND p.value IN (${{pool:sqlstring}}) AND r.lifecycle_stage = 'active' "
            "ORDER BY r.name")


def build() -> dict:
    panels, y = [], 0
    panels.append(row("Supervised training (per epoch)", y)); y += 1
    panels += [
        trend("Val quality", ["val_quality"], "run", 0, y, description="share of steps answered correctly, val episodes"),
        trend("Val cost per episode", ["val_cost"], "run", 12, y, unit="currencyUSD")]
    y += 8
    panels += [
        trend("Train vs val loss", ["train_loss", "val_loss"], "run", 0, y, label_with_key=True),
        trend("Loss per head (train)", ["train_model", "train_halt", "train_complexity"], "run", 12, y,
              label_with_key=True, description="model 1.0 / halt 0.5 / complexity 0.2 in the total")]
    y += 8
    panels += [
        trend("Oracle gap closed", ["val_gap_closed"], "run", 0, y, w=8, unit="percentunit",
              description="(quality - always cheapest) / (oracle - always cheapest)"),
        trend("Choice entropy", ["val_choice_entropy"], "run", 8, y, w=8,
              description="0 = one option every time (collapse), 1 = uniform"),
        trend("Escalation rate", ["val_escalation_rate"], "run", 16, y, w=8, unit="percentunit")]
    y += 8
    panels += [
        trend("Gradient norm", ["grad_norm"], "run", 0, y, w=8),
        trend("Learning rate", ["lr"], "run", 8, y, w=8),
        trend("Alarms per epoch", ["alarm_count"], "run", 16, y, w=8,
              description="collapse, non-finite values, train/val gap")]
    y += 8
    summary = ("SELECT r.name AS run, " + tag("pool") + " AS pool, " + tag("config") + " AS config, "
               + last("epochs_run") + " AS epochs, " + last("best_val_loss") + " AS best_val_loss, "
               + last("tau.tau") + " AS tau, " + last("tau.halt_tau") + " AS halt_tau, "
               + last("tau.val_quality") + " AS val_quality, " + last("tau.val_cost_usd") + " AS val_cost_usd, "
               + last("tau.random_quality") + " AS random_quality, " + last("tau.oracle.oracle_quality")
               + " AS oracle_quality, r.status AS status "
               "FROM runs r WHERE r.name IN (${run:sqlstring}) AND r.lifecycle_stage = 'active' ORDER BY r.name")
    panels.append(table("Runs: selected thresholds on val (tau sweep)", summary, 0, y,
                        description="selection rule: best val quality with cost <= random baseline"))
    y += 8
    alarms = ("SELECT r.name AS run, t.value AS alarms FROM runs r JOIN tags t ON t.run_uuid = r.run_uuid "
              "AND t.key = 'alarms' WHERE r.name IN (${run:sqlstring}) ORDER BY r.name")
    panels.append(table("Alarm log", alarms, 0, y, h=6)); y += 6

    panels.append(row("Simulator RL (per update)", y)); y += 1
    panels += [
        trend("Val reward at lambda index ${lam}", ["val.${lam}.reward"], "rl_run", 0, y, xlabel="update",
              description="lambda index 0..4 = 0.1, 0.3, 0.6, 1.0, 2.0 (choose it in the 'lambda index' selector)"),
        trend("Val quality at lambda index ${lam}", ["val.${lam}.quality"], "rl_run", 12, y, xlabel="update",
              description="lambda index 0..4 = 0.1, 0.3, 0.6, 1.0, 2.0 (choose it in the 'lambda index' selector)")]
    y += 8
    panels += [
        trend("Mean val reward (all lambdas)", ["mean_val_reward"], "rl_run", 0, y, w=8, xlabel="update"),
        trend("Sample-mode train return", ["train_return"], "rl_run", 8, y, w=8, xlabel="update"),
        trend("Policy entropy", ["entropy"], "rl_run", 16, y, w=8, xlabel="update",
              description="nats over the pool; ln 4 = 1.39 is uniform")]
    y += 8
    final = " UNION ALL ".join(
        f"SELECT r.name AS run, {lam} AS lambda, {last(f'val.{i}.quality')} AS quality, "
        f"{last(f'val.{i}.cost_usd')} AS cost_usd, {last(f'val.{i}.reward')} AS reward, "
        f"{last(f'val.{i}.escalation_rate')} AS escalation_rate FROM runs r "
        f"WHERE r.name IN (${{rl_run:sqlstring}}) AND r.lifecycle_stage = 'active'"
        for i, lam in enumerate(LAMS)) + " ORDER BY run, lambda"
    panels.append(table("RL: last val check per lambda", final, 0, y, h=10,
                        description="the saved model is the best check that passed the safety guard"))

    lam_var = {"name": "lam", "label": "lambda index", "type": "custom", "multi": False, "includeAll": False,
               "query": "0,1,2,3,4", "current": {"text": "3", "value": "3"},
               "options": [{"text": str(i), "value": str(i), "selected": i == 3} for i in range(5)]}
    return {
        "uid": "neural-core-training", "title": "neural-core: training monitor", "tags": ["neural-core", "mlflow"],
        "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "30s", "editable": True,
        "time": {"from": "now-30d", "to": "now"},
        "templating": {"list": [
            variable("pool", "pool", "SELECT DISTINCT value FROM tags WHERE key = 'pool' ORDER BY value"),
            variable("run", "supervised runs", runs_of("supervised")),
            variable("rl_run", "RL runs", runs_of("rl_sim")),
            lam_var]},
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2))
    print(f"wrote {OUT}")
