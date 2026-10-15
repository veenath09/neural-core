"""
Purpose:  Research dashboard. One place to watch training runs, check evaluation results
          against the hypotheses, inspect single episodes and see the model pool.
Owner:    C3 Measurement (dashboard), shared use
Contract: v1
Inputs:   ROOT env var (default ".") with runs/, results/, data/clean/catalogue.json
Outputs:  Streamlit app:  streamlit run ui/dashboard.py
Notes:    Read-only. Every number shown is computed from saved files through ui/data.py,
          the same files scripts/make_figures.py uses, so the dashboard and the report
          cannot disagree.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from data import connect, has_view, reliability, seed_ci

ROOT = os.environ.get("ROOT", ".")
st.set_page_config(page_title="J26-DS-359 Research Dashboard", layout="wide")


@st.cache_resource
def get_con(root: str):
    return connect(root)


con = get_con(ROOT)
st.title("J26-DS-359 Research Dashboard")

tabs = st.tabs(["Overview", "Training", "Results", "Calibration", "Episodes", "Model pool"])

# --------------------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("Runs")
    if has_view(con, "runs"):
        runs = con.execute("""
            SELECT run_id, component, config_name, stage, seed, status, started_at, ended_at
            FROM runs ORDER BY started_at DESC
        """).df()
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs", len(runs))
        c2.metric("Running", int((runs["status"] == "running").sum()))
        c3.metric("Failed", int((runs["status"] == "failed").sum()))
        st.dataframe(runs, width="stretch", hide_index=True)
    else:
        st.info("No runs yet. Training scripts write runs/<run_id>/ through RunLogger.")

    st.subheader("Alarms and events")
    if has_view(con, "events"):
        ev = con.execute("SELECT * FROM events ORDER BY time DESC LIMIT 200").df()
        alarms = ev[ev["kind"].isin(["alarm", "rollback"])]
        if len(alarms):
            st.error(f"{len(alarms)} alarm/rollback events")
        st.dataframe(ev, width="stretch", hide_index=True)
    else:
        st.caption("No events logged.")

    st.subheader("Validity checks")
    if has_view(con, "by_type"):
        bt = con.execute("""
            SELECT config, seed, operating_pt, quality FROM by_type
            WHERE episode_type = 'random' AND config IN ('full', 'context_free')
        """).df()
        if not bt.empty and {"full", "context_free"} <= set(bt["config"]):
            piv = bt.pivot_table(index=["seed", "operating_pt"], columns="config", values="quality").dropna()
            gap = float((piv["full"] - piv["context_free"]).mean()) if len(piv) else float("nan")
            msg = f"Random-chain gap (full - context_free): {gap:+.3f}. Expected about 0."
            (st.warning if abs(gap) > 0.02 else st.success)(msg)
        else:
            st.caption("Needs evaluated full and context_free runs.")
    else:
        st.caption("No predictions yet.")

# --------------------------------------------------------------------------- Training
with tabs[1]:
    if has_view(con, "metrics"):
        run_ids = con.execute("SELECT DISTINCT run_id FROM metrics ORDER BY run_id").df()["run_id"].tolist()
        chosen = st.multiselect("Runs", run_ids, default=run_ids[: min(4, len(run_ids))])
        names = con.execute("SELECT DISTINCT metric FROM metrics ORDER BY metric").df()["metric"].tolist()
        default = [m for m in ["loss", "val_frontier_area", "gate_rate_mean", "entropy"] if m in names] or names[:2]
        metrics_sel = st.multiselect("Metrics", names, default=default)
        if chosen and metrics_sel:
            df = con.execute(
                "SELECT * FROM metrics WHERE run_id IN (SELECT unnest(?)) AND metric IN (SELECT unnest(?))",
                [chosen, metrics_sel],
            ).df()
            for m in metrics_sel:
                sub = df[df["metric"] == m]
                fig = px.line(sub, x="step", y="value", color="run_id", line_dash="split", title=m)
                st.plotly_chart(fig, width="stretch")
    else:
        st.info("No metrics logged yet.")

# --------------------------------------------------------------------------- Results
with tabs[2]:
    if has_view(con, "frontier"):
        stage = st.radio("Stage", ["supervised", "rl", "eval"], horizontal=True)
        fr = con.execute("SELECT * FROM frontier WHERE stage = ? OR policy <> 'controller'", [stage]).df()
        agg = (fr.groupby(["policy", "config", "operating_pt"])[["quality", "cost_usd"]].mean().reset_index())
        agg["label"] = agg["policy"] + ":" + agg["config"]
        fig = px.line(agg.sort_values("cost_usd"), x="cost_usd", y="quality", color="label", markers=True,
                      title="O1 Quality vs cost (mean over seeds)")
        st.plotly_chart(fig, width="stretch")

        st.subheader("O2 Per-episode-type: full vs context_free")
        bt = con.execute("""
            SELECT * FROM by_type WHERE config IN ('full', 'context_free') AND stage = ?
        """, [stage]).df()
        if not bt.empty:
            op = st.selectbox("Operating point", sorted(bt["operating_pt"].unique()))
            t = seed_ci(bt[bt["operating_pt"] == op], ["episode_type", "config"], "quality")
            st.dataframe(t.pivot(index="episode_type", columns="config", values=["quality", "ci95"]),
                         width="stretch")

        st.subheader("O3 Ablations")
        ab = fr[fr["config"].isin(["full", "gate_off", "gate_sigmoid", "memory_off", "context_free"])]
        if not ab.empty:
            st.dataframe(seed_ci(ab, ["config", "operating_pt"], "quality"), width="stretch",
                         hide_index=True)
    else:
        st.info("No evaluation results yet. scripts/evaluate.py writes results/<eval_id>/predictions.parquet.")

# --------------------------------------------------------------------------- Calibration
with tabs[3]:
    if has_view(con, "predictions"):
        evals = con.execute("SELECT DISTINCT eval_id FROM predictions ORDER BY eval_id").df()["eval_id"].tolist()
        e = st.selectbox("Evaluation set", evals)
        r, ece = reliability(con, where=f"eval_id = '{e}' AND policy = 'controller'")
        st.metric("ECE", f"{ece:.3f}" if ece == ece else "n/a")
        if not r.empty:
            fig = px.scatter(r, x="conf", y="acc", size="n", title="O6 Reliability (ideal: diagonal)")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No predictions yet.")

# --------------------------------------------------------------------------- Episodes
with tabs[4]:
    if has_view(con, "predictions"):
        cfgs = con.execute("SELECT DISTINCT config FROM predictions ORDER BY config").df()["config"].tolist()
        cfg = st.selectbox("Config", cfgs)
        eps = con.execute("""
            SELECT DISTINCT episode_id, episode_type FROM predictions WHERE config = ? ORDER BY episode_id LIMIT 2000
        """, [cfg]).df()
        etype = st.selectbox("Episode type", sorted(eps["episode_type"].unique()))
        ep = st.selectbox("Episode", eps[eps["episode_type"] == etype]["episode_id"].tolist())
        d = con.execute("""
            SELECT * FROM predictions WHERE config = ? AND episode_id = ?
            ORDER BY seed, operating_pt, step_index, round
        """, [cfg, ep]).df()
        if not d.empty:
            first = d[(d["seed"] == d["seed"].min()) & (d["operating_pt"] == d["operating_pt"].iloc[0])]
            st.caption("O9 One episode, step by step: model chosen, verifier signal, surprise, gate rate.")
            cols = [c for c in ["step_index", "round", "model_id", "oracle_model_id", "halt", "verifier_ok",
                                "success_true", "cost_usd", "confidence", "surprise_norm", "gate_rate",
                                "budget_left_before"] if c in first.columns]
            st.dataframe(first[cols], width="stretch", hide_index=True)
            if "surprise_norm" in first and first["surprise_norm"].notna().any():
                long = first.melt(id_vars=["step_index", "round"], value_vars=["surprise_norm", "gate_rate"])
                long["t"] = long["step_index"] + 0.5 * long["round"]
                st.plotly_chart(px.line(long, x="t", y="value", color="variable", markers=True,
                                        title="Surprise and gate rate across the episode"),
                                width="stretch")
    else:
        st.info("No predictions yet.")

# --------------------------------------------------------------------------- Model pool
with tabs[5]:
    if has_view(con, "catalogue"):
        cat = con.execute("SELECT * FROM catalogue").df()
        st.dataframe(cat[[c for c in ["model_id", "provider", "price_in_per_m", "price_out_per_m",
                                      "price_date", "status", "probe_version"] if c in cat.columns]],
                     width="stretch", hide_index=True)
        if "capability" in cat.columns:
            rows = []
            for _, r in cat.iterrows():
                cap = r["capability"]
                cap = json.loads(cap) if isinstance(cap, str) else dict(cap)
                rows += [{"model_id": r["model_id"], "step_type": k, "score": v} for k, v in cap.items()]
            heat = pd.DataFrame(rows).pivot(index="model_id", columns="step_type", values="score")
            st.plotly_chart(px.imshow(heat, text_auto=".2f", title="Capability by step type"),
                            width="stretch")
    else:
        st.info("No catalogue yet (data/clean/catalogue.json).")
