# app_memory_bus.py
# Streamlit demo: Memory Bus (event log) + Shared Data UI with mock data
# Run: streamlit run app_memory_bus.py

import os
import json
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------------------------------
# CONFIG / PATHS
# ----------------------------------------------------
st.set_page_config(page_title="Memory Bus + Shared Data (Demo)", layout="wide")
st.title("🧠 Memory Bus & Shared Data — Demo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_demo")
os.makedirs(DATA_DIR, exist_ok=True)

BUS_JSONL = os.path.join(DATA_DIR, "bus_events.jsonl")
FORECAST_CSV = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_CSV = os.path.join(DATA_DIR, "inventory_plan.csv")
SIMULATION_CSV = os.path.join(DATA_DIR, "simulation_output.csv")
TASKS_CSV = os.path.join(DATA_DIR, "collab_tasks.csv")

alt.data_transformers.disable_max_rows()

def strong_palette(k=10):
    return ["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e",
            "#17becf","#e377c2","#7f7f7f","#bcbd22","#8c564b"][:k]

# ----------------------------------------------------
# LIGHTWEIGHT MEMORY BUS
# ----------------------------------------------------
if "events" not in st.session_state:
    st.session_state.events = []  # list of dicts

def bus_emit(evt_type: str, payload: dict):
    evt = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "type": evt_type,
        "payload": payload or {},
    }
    st.session_state.events.append(evt)

def bus_reset():
    st.session_state.events = []

def bus_save_jsonl(path=BUS_JSONL):
    with open(path, "w", encoding="utf-8") as f:
        for e in st.session_state.events:
            f.write(json.dumps(e) + "\n")

def bus_load_jsonl(path=BUS_JSONL):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.session_state.events = [json.loads(line) for line in f if line.strip()]

# ----------------------------------------------------
# SEED MOCK DATA
# ----------------------------------------------------
def seed_mock_tables():
    rng = np.random.default_rng(42)

    cities = ["Buffalo", "Austin", "Seattle"]
    brands = ["Brand X", "Brand Y", "Brand Z"]
    start = pd.Timestamp.today().to_period("M").to_timestamp()  # month start
    months = pd.period_range(start=start, periods=3, freq="M").to_timestamp()

    # Forecast (month, city, brand, forecast_sales)
    rows = []
    for m in months:
        for c in cities:
            for b in brands:
                base = 200 + rng.integers(-40, 60)
                lift = 60 if (c == "Buffalo" and b == "Brand X" and m.month == months[0].month) else 0
                rows.append({"month": m.strftime("%Y-%m"),
                             "city": c, "brand": b,
                             "forecast_sales": max(50, base + lift)})
    fc = pd.DataFrame(rows)
    fc.to_csv(FORECAST_CSV, index=False)

    # Inventory snapshot/plan (simple: on-hand & pipeline per city/brand/month)
    inv_rows = []
    for m in months:
        for c in cities:
            for b in brands:
                on_hand = rng.integers(80, 260)
                pipeline = rng.integers(0, 120)
                inv_rows.append({"month": m.strftime("%Y-%m"), "city": c, "brand": b,
                                 "on_hand": int(on_hand), "pipeline": int(pipeline)})
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(INVENTORY_CSV, index=False)

    # Simulation (derive simple required_units & pick strategy by rules/cost)
    sim = fc.merge(inv, on=["month","city","brand"], how="left")
    sim["required_units"] = (sim["forecast_sales"]*1.10 - sim["on_hand"]).clip(lower=0).round(0)
    # cheap rule: if small need → Do Nothing; else choose between Reorder/Transfer/Hybrid by thresholds
    def choose_strategy(r):
        R = r["required_units"]
        if R <= 20: return "Do Nothing"
        if R <= 120: return "Reorder"
        if R <= 250: return "Hybrid"
        return "Transfer"
    sim["best_strategy"] = sim.apply(choose_strategy, axis=1)

    # fake costs
    reorder_uc, transfer_uc, penalty_uc = 60, 30, 80
    sim["cost_do_nothing"] = penalty_uc*sim["required_units"]
    sim["cost_reorder"] = reorder_uc*sim["required_units"]
    sim["cost_transfer"] = transfer_uc*sim["required_units"]
    # hybrid = 50/50 blend for demo
    sim["cost_hybrid"] = 0.5*sim["cost_reorder"] + 0.5*sim["cost_transfer"]

    sim.to_csv(SIMULATION_CSV, index=False)

    # Tasks (seed from simulation)
    tdf = sim[["month","city","brand","best_strategy","required_units"]].copy()
    tdf["task_id"] = range(1001, 1001+len(tdf))
    tdf["title"] = "Execute: " + tdf["best_strategy"]
    tdf["status"] = np.where(tdf["best_strategy"].eq("Transfer"), "In-Progress", "Pending")
    tdf["priority"] = np.where(tdf["best_strategy"].eq("Transfer"), "High",
                        np.where(tdf["best_strategy"].eq("Hybrid"), "Medium", "Low"))
    tdf["owner"] = np.random.choice(["Sam","Priya","Lee","Taylor"], size=len(tdf))
    # due date = month end
    tdf["due_date"] = pd.to_datetime(tdf["month"], format="%Y-%m") + pd.offsets.MonthEnd(0)
    tdf["due_date"] = tdf["due_date"].dt.strftime("%Y-%m-%d")
    tdf.rename(columns={"best_strategy":"decision"}, inplace=True)
    tdf.to_csv(TASKS_CSV, index=False)

def seed_mock_bus():
    now = datetime.utcnow()
    samples = [
        ("refresh", {}),
        ("pipeline_run", {"sensing": True, "forecast": True, "inventory": True, "simulation": True}),
        ("params_changed", {"agent": "Sensing", "key": "z_thresh", "old": 1.0, "new": 1.3}),
        ("forecast_saved", {"rows": 27, "path": "data_demo/forecast_output.csv"}),
        ("decision_viewed", {"city": "Buffalo", "brand": "Brand X", "strategy": "Transfer", "required": 180}),
        ("tasks_seeded", {"rows": 27, "path": "data_demo/collab_tasks.csv"}),
    ]
    st.session_state.events = []
    for i, (t, p) in enumerate(samples):
        evt_time = (now - timedelta(hours=6-i)).isoformat(timespec="seconds")
        st.session_state.events.append({"ts": evt_time, "type": t, "payload": p})

# ----------------------------------------------------
# READ HELPERS
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path, parse_month=False):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    if parse_month and "month" in df.columns:
        m1 = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        m2 = pd.to_datetime(df["month"], errors="coerce")
        df["month_dt"] = m1.fillna(m2).dt.to_period("M").dt.to_timestamp()
    return df

# ----------------------------------------------------
# FIRST-RUN SEED
# ----------------------------------------------------
if st.sidebar.button("🔁 Reset demo data (seed)"):
    seed_mock_tables()
    seed_mock_bus()
    bus_save_jsonl()
    st.cache_data.clear()
    st.success("Seeded mock tables & bus.")
    st.experimental_rerun()

if not (os.path.exists(FORECAST_CSV) and os.path.exists(INVENTORY_CSV) and os.path.exists(SIMULATION_CSV) and os.path.exists(TASKS_CSV)):
    seed_mock_tables()
if not st.session_state.events:
    if os.path.exists(BUS_JSONL):
        bus_load_jsonl()
    else:
        seed_mock_bus()
        bus_save_jsonl()

# ----------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------
st.sidebar.header("🔎 Global Filters")
fc_df  = read_csv(FORECAST_CSV, parse_month=True)
inv_df = read_csv(INVENTORY_CSV, parse_month=True)
sim_df = read_csv(SIMULATION_CSV, parse_month=True)
task_df= read_csv(TASKS_CSV, parse_month=False)

cities = sorted(set(fc_df.get("city", [])).union(inv_df.get("city", []), sim_df.get("city", [])))
brands = sorted(set(fc_df.get("brand", [])).union(inv_df.get("brand", []), sim_df.get("brand", [])))
months = []
for df in (fc_df, inv_df, sim_df):
    if "month_dt" in df.columns:
        months.extend(df["month_dt"].dropna().unique().tolist())
months = sorted(pd.to_datetime(pd.Series(months)).unique()) if months else []

sel_city  = st.sidebar.selectbox("City", ["All"] + cities, index=0)
sel_brand = st.sidebar.selectbox("Brand", ["All"] + brands, index=0)
sel_month = st.sidebar.selectbox("Month", ["All"] + [pd.to_datetime(m).strftime("%b-%Y") for m in months], index=0)

def apply_filters(df):
    if df.empty: return df
    x = df.copy()
    if sel_city != "All" and "city" in x.columns:
        x = x[x["city"] == sel_city]
    if sel_brand != "All" and "brand" in x.columns:
        x = x[x["brand"] == sel_brand]
    if sel_month != "All":
        if "month_dt" in x.columns:
            x = x[pd.to_datetime(x["month_dt"]).dt.strftime("%b-%Y") == sel_month]
        elif "month" in x.columns:
            x = x[x["month"].astype(str).str[:7] == pd.to_datetime(sel_month).strftime("%Y-%m")]
    return x

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab_bus, tab_shared = st.tabs(["🧾 Memory Bus", "🔗 Shared Data"])

# ================== MEMORY BUS ==================
with tab_bus:
    st.subheader("🧾 Event Log (Memory Bus)")

    with st.expander("➕ Emit Event"):
        colA, colB = st.columns(2)
        with colA:
            evt_type = st.selectbox(
                "Type",
                ["refresh","pipeline_run","params_changed","forecast_saved","decision_viewed","tasks_seeded","custom"]
            )
            payload_text = st.text_area(
                "Payload (JSON)", 
                value='{"example":"value"}' if evt_type=="custom" else "{}",
                height=100
            )
        with colB:
            if st.button("Emit"):
                try:
                    payload = json.loads(payload_text) if payload_text.strip() else {}
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    payload = {}
                bus_emit(evt_type, payload)
                st.success(f"Emitted {evt_type}")
                bus_save_jsonl()

    # Bus table
    if not st.session_state.events:
        st.info("No events yet.")
    else:
        df_bus = pd.DataFrame(st.session_state.events).copy()
        # flat payload for display
        df_bus["payload_str"] = df_bus["payload"].apply(lambda d: json.dumps(d, ensure_ascii=False))
        st.dataframe(df_bus[["ts","type","payload_str"]].sort_values("ts", ascending=False), use_container_width=True, height=280)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            cnt = df_bus["type"].value_counts().reset_index()
            cnt.columns = ["type","count"]
            ch = (
                alt.Chart(cnt)
                .mark_bar()
                .encode(x=alt.X("type:N", title="Type"),
                        y=alt.Y("count:Q", title="Events"),
                        color=alt.Color("type:N", legend=None, scale=alt.Scale(range=strong_palette(8))),
                        tooltip=["type","count"])
                .properties(height=260, title="Events by Type")
            )
            st.altair_chart(ch, use_container_width=True)
        with c2:
            df_bus["ts_dt"] = pd.to_datetime(df_bus["ts"], errors="coerce")
            timeline = df_bus.groupby(pd.Grouper(key="ts_dt", freq="H"))["type"].count().reset_index().rename(columns={"type":"events"})
            tl = (
                alt.Chart(timeline)
                .mark_line(point=True)
                .encode(x=alt.X("ts_dt:T", title="Time (Hourly)"),
                        y=alt.Y("events:Q", title="Count"),
                        tooltip=["ts_dt:T","events:Q"])
                .properties(height=260, title="Event Timeline")
            )
            st.altair_chart(tl, use_container_width=True)

        # Export
        bus_json = "\n".join(json.dumps(e) for e in st.session_state.events)
        st.download_button("⬇️ Download Bus (.jsonl)", data=bus_json.encode("utf-8"),
                           file_name="bus_events.jsonl", mime="application/jsonl")

# ================== SHARED DATA ==================
with tab_shared:
    st.subheader("🔗 Shared Data (Mock Artifacts)")

    # Forecast
    st.markdown("### 📈 Forecast")
    fc_view = apply_filters(fc_df)
    st.dataframe(fc_view, use_container_width=True, height=220)
    if {"forecast_sales","month_dt"}.issubset(fc_view.columns) and not fc_view.empty:
        viz = fc_view.copy()
        viz["series"] = "Forecast"
        ch = (
            alt.Chart(viz)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("month_dt:T", title="Month"),
                y=alt.Y("forecast_sales:Q", title="Forecast"),
                color=alt.Color("series:N", legend=None, scale=alt.Scale(range=["#e8590c"])),
                tooltip=[alt.Tooltip("month_dt:T", title="Month"), alt.Tooltip("forecast_sales:Q", title="Forecast", format=".0f")]
            ).properties(height=280)
        )
        st.altair_chart(ch, use_container_width=True)

    # Inventory
    st.markdown("### 📦 Inventory")
    inv_view = apply_filters(inv_df)
    st.dataframe(inv_view, use_container_width=True, height=220)
    if {"month_dt","on_hand","pipeline"}.issubset(inv_view.columns) and not inv_view.empty:
        melt = inv_view.groupby("month_dt", as_index=False)[["on_hand","pipeline"]].sum().melt("month_dt", var_name="series", value_name="units")
        ch2 = (
            alt.Chart(melt)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("month_dt:T", title="Month"),
                y=alt.Y("units:Q", title="Units"),
                color=alt.Color("series:N", legend=None, scale=alt.Scale(range=["#1f77b4","#2ca02c"])),
                tooltip=["month_dt:T","series:N", alt.Tooltip("units:Q", format=".0f")],
            ).properties(height=280, title="On-Hand + Pipeline")
        )
        st.altair_chart(ch2, use_container_width=True)

    # Simulation
    st.markdown("### 🔁 Simulation")
    sim_view = apply_filters(sim_df)
    st.dataframe(sim_view, use_container_width=True, height=220)
    if "best_strategy" in sim_view.columns and not sim_view.empty:
        cnt = sim_view["best_strategy"].value_counts().reset_index()
        cnt.columns = ["strategy","count"]
        ch3 = (
            alt.Chart(cnt)
            .mark_bar()
            .encode(
                x=alt.X("strategy:N", title="Strategy"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("strategy:N", legend=None, scale=alt.Scale(range=["#2f9e44","#2c7be5","#e8590c","#7048e8"])),
                tooltip=["strategy","count"]
            ).properties(height=240, title="Strategy Mix")
        )
        st.altair_chart(ch3, use_container_width=True)

        # Coverage vs Demand
        if {"month_dt","forecast_sales","on_hand"}.issubset(sim_view.columns):
            tmp = sim_view.copy()
            if "month_dt" not in tmp.columns:
                tmp["month_dt"] = pd.to_datetime(tmp["month"], errors="coerce")
            cov = tmp.groupby("month_dt", as_index=False)[["forecast_sales","on_hand"]].sum()
            cov = cov.melt("month_dt", var_name="series", value_name="units")
            lc = (
                alt.Chart(cov)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("month_dt:T", title="Month"),
                    y=alt.Y("units:Q", title="Units"),
                    color=alt.Color("series:N", legend=None, scale=alt.Scale(range=["#1f77b4","#d62728"])),
                    tooltip=["month_dt:T","series:N", alt.Tooltip("units:Q", format=".0f")],
                ).properties(height=280, title="Coverage vs Demand")
            )
            st.altair_chart(lc, use_container_width=True)

    # Tasks
    st.markdown("### 🤝 Tasks")
    task_view = task_df.copy()
    if sel_city != "All" and "city" in task_view.columns:
        task_view = task_view[task_view["city"] == sel_city]
    if sel_brand != "All" and "brand" in task_view.columns:
        task_view = task_view[task_view["brand"] == sel_brand]
    st.dataframe(task_view, use_container_width=True, height=240)

    # Quick KPI tiles
    open_tasks = (task_view["status"].isin(["Pending","In-Progress"])).sum() if "status" in task_view.columns else 0
    high_risk  = (task_view["priority"].eq("High")).sum() if "priority" in task_view.columns else 0
    c1, c2 = st.columns(2)
    c1.metric("Open Tasks", int(open_tasks))
    c2.metric("High Priority", int(high_risk))

st.caption("Tip: Use the sidebar to filter the shared tables. Use the Memory Bus tab to emit and export events.")