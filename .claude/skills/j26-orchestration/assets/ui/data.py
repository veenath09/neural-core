"""
Purpose:  Read-only data layer for the research dashboard. Builds DuckDB views over the
          Parquet/JSON files written by RunLogger, the evaluation harness and the catalogue.
Owner:    C3 Measurement (dashboard), shared use
Contract: v1
Inputs:   root directory containing runs/, results/, data/clean/catalogue.json
Outputs:  duckdb connection with views: runs, metrics, events, predictions, step_final,
          episode_summary, frontier, by_type, catalogue
Notes:    No server needed in Phases 0-2. In Phase 3 the same view names can be created
          over PostgreSQL with DuckDB's postgres extension (ATTACH ... (TYPE postgres)),
          so the dashboard code does not change.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import duckdb
import pandas as pd

# Student t critical values (two-sided 95%) for small seed counts; df = n - 1.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _has(pattern: str) -> bool:
    return len(glob.glob(pattern)) > 0


def connect(root: str | os.PathLike = ".") -> duckdb.DuckDBPyConnection:
    root = Path(root)
    con = duckdb.connect()
    runs = str(root / "runs" / "*" / "run.json")
    metrics = str(root / "runs" / "*" / "metrics-*.parquet")
    events = str(root / "runs" / "*" / "events-*.parquet")
    preds = str(root / "results" / "*" / "predictions.parquet")
    cat = root / "data" / "clean" / "catalogue.json"

    if _has(runs):
        con.execute(f"CREATE VIEW runs AS SELECT * FROM read_json_auto('{runs}', union_by_name=true)")
    if _has(metrics):
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{metrics}', union_by_name=true)")
    if _has(events):
        con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{events}', union_by_name=true)")
    if cat.exists():
        con.execute(f"CREATE VIEW catalogue AS SELECT * FROM read_json_auto('{cat}')")
    if _has(preds):
        con.execute(f"CREATE VIEW predictions AS SELECT * FROM read_parquet('{preds}', union_by_name=true)")
        # final round of each step decides its quality; cost counts every round
        con.execute("""
            CREATE VIEW step_final AS
            SELECT p.* FROM predictions p
            JOIN (SELECT eval_id, policy, config, stage, seed, operating_pt, step_id, max(round) AS r
                  FROM predictions GROUP BY ALL) m
              ON p.eval_id = m.eval_id AND p.policy = m.policy AND p.config = m.config
             AND p.stage = m.stage AND p.seed = m.seed AND p.operating_pt = m.operating_pt
             AND p.step_id = m.step_id AND p.round = m.r
        """)
        con.execute("""
            CREATE VIEW episode_summary AS
            SELECT q.eval_id, q.policy, q.config, q.stage, q.seed, q.operating_pt,
                   q.episode_id, q.episode_type, q.quality, c.cost_usd, c.escalations
            FROM (SELECT eval_id, policy, config, stage, seed, operating_pt, episode_id, episode_type,
                         avg(success_true::DOUBLE) AS quality
                  FROM step_final GROUP BY ALL) q
            JOIN (SELECT eval_id, policy, config, stage, seed, operating_pt, episode_id,
                         sum(cost_usd) AS cost_usd, sum(CASE WHEN round > 0 THEN 1 ELSE 0 END) AS escalations
                  FROM predictions GROUP BY ALL) c
            USING (eval_id, policy, config, stage, seed, operating_pt, episode_id)
        """)
        con.execute("""
            CREATE VIEW frontier AS
            SELECT policy, config, stage, seed, operating_pt,
                   avg(quality) AS quality, avg(cost_usd) AS cost_usd, count(*) AS n_episodes
            FROM episode_summary GROUP BY ALL
        """)
        con.execute("""
            CREATE VIEW by_type AS
            SELECT policy, config, stage, seed, operating_pt, episode_type,
                   avg(quality) AS quality, avg(cost_usd) AS cost_usd, count(*) AS n_episodes
            FROM episode_summary GROUP BY ALL
        """)
    return con


def has_view(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(con.execute(
        "SELECT count(*) FROM duckdb_views() WHERE view_name = ?", [name]).fetchone()[0])


def seed_ci(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Mean and 95% CI half-width across seeds (t distribution), grouped by keys."""
    g = df.groupby(keys)[value].agg(["mean", "std", "count"]).reset_index()
    g["ci95"] = [
        (T95.get(int(n) - 1, 1.96) * s / (n ** 0.5)) if n > 1 else float("nan")
        for s, n in zip(g["std"].fillna(0), g["count"])
    ]
    return g.rename(columns={"mean": value, "count": "n_seeds"}).drop(columns=["std"])


def reliability(con: duckdb.DuckDBPyConnection, where: str = "TRUE", bins: int = 15) -> tuple[pd.DataFrame, float]:
    """Reliability-diagram bins and expected calibration error over round-level predictions."""
    df = con.execute(f"""
        SELECT confidence, success_true::DOUBLE AS y FROM predictions
        WHERE confidence IS NOT NULL AND {where}
    """).df()
    if df.empty:
        return df, float("nan")
    df["bin"] = (df["confidence"].clip(0, 0.999999) * bins).astype(int)
    r = df.groupby("bin").agg(conf=("confidence", "mean"), acc=("y", "mean"), n=("y", "size")).reset_index()
    ece = float((r["n"] / r["n"].sum() * (r["conf"] - r["acc"]).abs()).sum())
    return r, ece
