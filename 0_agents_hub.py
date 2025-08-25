# pages/0_agent_hub.py
import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ----------------------------------------------------
# CONFIG / PATHS
# ----------------------------------------------------
st.set_page_config(page_title="Agentic AI — Hub", layout="wide")
st.title("🧠 Agentic AI — Control Hub")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FORECAST_PATH   = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH  = os.path.join(DATA_DIR, "inventory_plan.csv")
SIMULATION_PATH = os.path.join(DATA_DIR, "simulation_output.csv")
COST_PATH       = os.path.join(DATA_DIR, "cost_parameters.csv")
PAST_SPIKES     = os.path.join(DATA_DIR, "Detected_Spikes_Past.csv")
FUTURE_SPIKES   = os.path.join(DATA_DIR, "Detected_Spikes_Projected.csv")

os.makedirs(DATA_DIR, exist_ok=True)
alt.data_transformers.disable_max_rows()

# ----------------------------------------------------
# LIGHTWEIGHT "BUS" IN SESSION STATE
# ----------------------------------------------------
if "events" not in st.session_state:
    st.session_state.events = []  # list of dicts: {"ts":..., "type":..., "payload":...}

def bus_emit(evt_type: str, payload: dict):
    st.session_state.events.append({
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "type": evt_type,
        "payload": payload or {},
    })

def bus_table():
    if not st.session_state.events:
        st.info("No events yet. Run an action to see events.")
        return
    df = pd.DataFrame(st.session_state.events)
    st.dataframe(df, use_container_width=True)

# ----------------------------------------------------
# IO HELPERS
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def safe_read_csv(path: str, parse_month: bool = False):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if parse_month and "month" in df.columns:
        # accept either YYYY-MM or MMM-YYYY
        try:
            # try both common formats
            m = pd.to_datetime(df["month"], errors="coerce")
            # if many NaT, try alt format
            if m.isna().mean() > 0.5:
                m = pd.to_datetime(df["month"], format="%b-%Y", errors="coerce")
            df["month_dt"] = m.dt.to_period("M").dt.to_timestamp()
        except Exception:
            df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
    return df

def strong_palette(k=10):
    # high-contrast palette
    return ["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e",
            "#17becf","#e377c2","#7f7f7f","#bcbd22","#8c564b"][:k]

# ----------------------------------------------------
# SIDEBAR GLOBAL FILTERS
# ----------------------------------------------------
st.sidebar.header("🔎 Global View Filters")
# Load what we have to populate filters
fc_df  = safe_read_csv(FORECAST_PATH, parse_month=True)
inv_df = safe_read_csv(INVENTORY_PATH, parse_month=True)
sim_df = safe_read_csv(SIMULATION_PATH, parse_month=True)
past_df = safe_read_csv(PAST_SPIKES, parse_month=True)
fut_df  = safe_read_csv(FUTURE_SPIKES, parse_month=True)

# Build filter domain from whatever exists
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

def apply_filters(df: pd.DataFrame):
    if df.empty:
        return df
    out = df.copy()
    if sel_city != "All" and "city" in out.columns:
        out = out[out["city"] == sel_city]
    if sel_brand != "All" and "brand" in out.columns:
        out = out[out["brand"] == sel_brand]
    if sel_month != "All":
        # compare on formatted string
        if "month_dt" in out.columns:
            out = out[pd.to_datetime(out["month_dt"]).dt.strftime("%b-%Y") == sel_month]
        elif "month" in out.columns:
            out = out[out["month"].astype(str).str[:7] == pd.to_datetime(sel_month).strftime("%Y-%m")]
    return out

# ----------------------------------------------------
# ACTIONS (best-effort local pipeline triggers)
# ----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Actions")
colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        bus_emit("refresh", {})
        st.experimental_rerun()
with colB:
    if st.button("Run Pipeline"):
        # This button is a UI-level “agentic” pipeline. It doesn’t retrain models here;
        # it just checks files and logs steps. Your per-agent pages do the heavy lifting.
        steps = {
            "sensing": os.path.exists(PAST_SPIKES) or os.path.exists(FUTURE_SPIKES),
            "forecast": os.path.exists(FORECAST_PATH),
            "inventory": os.path.exists(INVENTORY_PATH),
            "simulation": os.path.exists(SIMULATION_PATH),
        }
        bus_emit("pipeline_run", steps)
        st.success("Queued a pipeline check. Use the tabs below. (This hub reads the CSV outputs.)")

# ----------------------------------------------------
# TABS (one per agent)
# ----------------------------------------------------
tab_sense, tab_fc, tab_inv, tab_sim, tab_collab, tab_bus = st.tabs(
    ["📡 Sensing", "📈 Forecast", "📦 Inventory", "🔁 Simulation", "🤝 Collaboration", "🧾 Event Log"]
)

# ===================== SENSING =====================
with tab_sense:
    st.subheader("📡 Sensing Agent")
    if past_df.empty and fut_df.empty:
        st.info("No sensing outputs found. Run your Sensing page to generate `Detected_Spikes_Past.csv` and `Detected_Spikes_Projected.csv`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Detected Past Spikes**")
            st.dataframe(apply_filters(past_df), use_container_width=True, height=260)
        with c2:
            st.markdown("**Projected Future Spikes**")
            st.dataframe(apply_filters(fut_df), use_container_width=True, height=260)

        # City distribution
        for name, dfX, scheme in [
            ("Past Spikes by City", past_df, "reds"),
            ("Future Spikes by City", fut_df, "blues"),
        ]:
            if not dfX.empty and {"city"}.issubset(dfX.columns):
                agg = apply_filters(dfX).groupby("city", as_index=False).size()
                ch = (
                    alt.Chart(agg)
                    .mark_bar()
                    .encode(
                        x=alt.X("city:N", title="City"),
                        y=alt.Y("size:Q", title="Spikes"),
                        color=alt.value("#d62728" if scheme=="reds" else "#1f77b4"),
                        tooltip=["city","size"]
                    ).properties(height=260, title=name)
                )
                st.altair_chart(ch, use_container_width=True)

# ===================== FORECAST =====================
with tab_fc:
    st.subheader("📈 Forecast Agent")
    if fc_df.empty:
        st.info("No `forecast_output.csv` found. Run the Forecast page.")
    else:
        view = apply_filters(fc_df)
        st.markdown("**Forecast Table**")
        st.dataframe(view, use_container_width=True, height=260)

        # Historical+Forecast line (if month_dt exists)
        if "forecast_sales" in fc_df.columns and "month_dt" in fc_df.columns:
            viz = view.copy()
            viz["series"] = "Forecast"
            # If your historical sales exist elsewhere, you could merge and draw both series here.
            line = (
                alt.Chart(viz)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("month_dt:T", title="Month"),
                    y=alt.Y("forecast_sales:Q", title="Forecast"),
                    color=alt.Color("series:N", legend=None, scale=alt.Scale(range=["#e8590c"])),
                    tooltip=[
                        alt.Tooltip("month_dt:T", title="Month"),
                        alt.Tooltip("forecast_sales:Q", title="Forecast", format=".2f"),
                    ],
                )
                .properties(height=320, title="Forecast (by filters)")
            )
            st.altair_chart(line, use_container_width=True)

# ===================== INVENTORY =====================
with tab_inv:
    st.subheader("📦 Inventory Agent")
    if inv_df.empty:
        st.info("No `onhand_inventory_snapshot.csv` or `inventory_plan.csv` found yet.")
    else:
        view = apply_filters(inv_df)
        st.markdown("**Inventory Snapshot / Plan**")
        st.dataframe(view, use_container_width=True, height=260)

        # If plan columns exist, chart planned coverage
        if {"month_dt","on_hand","pipeline"}.issubset(view.columns):
            melt = view.groupby("month_dt", as_index=False)[["on_hand","pipeline"]].sum().melt(
                "month_dt", var_name="series", value_name="units"
            )
            line = (
                alt.Chart(melt)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("month_dt:T", title="Month"),
                    y=alt.Y("units:Q", title="Units"),
                    color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#1f77b4","#2ca02c"])),
                    tooltip=[
                        alt.Tooltip("month_dt:T", title="Month"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("units:Q", title="Units", format=".2f"),
                    ],
                )
               .properties(height=320, title="On-Hand + Pipeline")
            )
            st.altair_chart(line, use_container_width=True)

# ===================== SIMULATION =====================
with tab_sim:
    st.subheader("🔁 Simulation Agent")
    if sim_df.empty:
        st.info("No `simulation_output.csv` found. Run the Simulation page.")
    else:
        view = apply_filters(sim_df)
        st.markdown("**Simulation Output (filtered)**")
        st.dataframe(view, use_container_width=True, height=260)

        # Strategy mix
        if "best_strategy" in view.columns and not view.empty:
            cnt = view["best_strategy"].value_counts().reset_index()
            cnt.columns = ["strategy","count"]
            bar = (
                alt.Chart(cnt)
                .mark_bar()
                .encode(
                    x=alt.X("strategy:N", title="Strategy"),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("strategy:N", legend=None,
                                    scale=alt.Scale(range=["#2f9e44","#2c7be5","#e8590c","#7048e8"])),
                    tooltip=["strategy","count"]
                ).properties(height=260, title="Strategy Mix")
            )
            st.altair_chart(bar, use_container_width=True)

        # Coverage vs Demand (if columns exist)
        if {"month","forecast_units","on_hand_units"}.issubset(view.columns):
            tmp = view.copy()
            if "month_dt" not in tmp.columns:
                # try parse
                tmp["month_dt"] = pd.to_datetime(tmp["month"], errors="coerce")
            cov = tmp.groupby("month_dt", as_index=False)[["forecast_units","on_hand_units"]].sum()
            if not cov.empty:
                cov = cov.melt("month_dt", var_name="series", value_name="units")
                lc = (
                    alt.Chart(cov)
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("month_dt:T", title="Month"),
                        y=alt.Y("units:Q", title="Units"),
                        color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#1f77b4","#d62728"])),
                        tooltip=[
                            alt.Tooltip("month_dt:T", title="Month"),
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("units:Q", title="Units", format=".2f"),
                        ],
                    )
                    .properties(height=320, title="Coverage vs Demand")
                )
                st.altair_chart(lc, use_container_width=True)
            else:
                st.info("Coverage chart has no non‑zero rows with current filters.")

        st.markdown("---")
        st.markdown("### 🤖 Decision Assistant")
        # Local, business‑friendly drop‑downs
        dv = sim_df.copy()
        # Domains (from sim only to avoid empty selections)
        dd_city  = ["— choose —"] + sorted(dv.get("city", pd.Series(dtype=str)).dropna().unique().tolist())
        dd_brand = ["— choose —"] + sorted(dv.get("brand", pd.Series(dtype=str)).dropna().unique().tolist())
        colx, coly = st.columns(2)
        with colx:
            d_city = st.selectbox("City", dd_city, index=0, key="sim_city")
        with coly:
            d_brand = st.selectbox("Brand", dd_brand, index=0, key="sim_brand")

        def badge(text, color):
            st.markdown(
                f"""
                <div style="border-radius:10px;padding:10px 12px;margin:8px 0;
                            background:{color}15;border:1px solid {color}44;color:{color};font-weight:700;">
                    {text}
                </div>
                """,
                unsafe_allow_html=True
            )

        if d_city != "— choose —" and d_brand != "— choose —":
            sub = dv[(dv.get("city","")==d_city) & (dv.get("brand","")==d_brand)].copy()
            if sub.empty:
                st.info("No matching rows. Try different filters.")
            else:
                sub = sub.sort_values("month_dt" if "month_dt" in sub.columns else "month").tail(1)
                r = sub.iloc[0]
                strat = str(r.get("best_strategy",""))
                req   = float(pd.to_numeric(r.get("required_units", 0), errors="coerce") or 0)
                fcast = float(pd.to_numeric(r.get("forecast_units", 0), errors="coerce") or 0)
                onh   = float(pd.to_numeric(r.get("on_hand_units", 0), errors="coerce") or 0)

                st.write(f"**Latest month:** {r.get('month', r.get('month_dt',''))}")
                st.write(f"**Forecast:** {fcast:,.0f} · **On‑hand:** {onh:,.0f} · **Required:** {req:,.0f}")

                if strat == "Do Nothing":
                    badge("✅ Decision: Do Nothing — coverage is adequate.", "#2f9e44")
                elif strat == "Reorder":
                    badge("🛠️ Decision: Reorder — place a supplier order.", "#2c7be5")
                elif strat == "Transfer":
                    badge("🔁 Decision: Transfer — pull stock from another warehouse.", "#e8590c")
                else:
                    badge("🧩 Decision: Hybrid — mix reorder & transfer.", "#7048e8")

                bus_emit("decision_viewed", {"city": d_city, "brand": d_brand, "strategy": strat, "required": req})
        else:
            st.info("Pick City and Brand to see a tailored recommendation.")

# ===================== COLLABORATION =====================
with tab_collab:
    st.subheader("🤝 Collaboration Agent (View)")
    # If Collaboration page produced task CSVs, you can mirror summaries here.
    TASKS_CSV = os.path.join(DATA_DIR, "collab_tasks.csv")
    if os.path.exists(TASKS_CSV):
        tasks = safe_read_csv(TASKS_CSV)
        st.dataframe(apply_filters(tasks), use_container_width=True, height=280)
        if not tasks.empty and {"status","owner"}.issubset(tasks.columns):
            kpi_open = (tasks["status"].isin(["Pending","In-Progress"])).sum()
            owners = tasks[tasks["status"].isin(["Pending","In-Progress"])].groupby("owner", as_index=False)["status"].count()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Open Tasks", int(kpi_open))
            with col2:
                ch = (
                    alt.Chart(owners)
                    .mark_bar()
                    .encode(
                        x=alt.X("owner:N", title="Owner"),
                        y=alt.Y("status:Q", title="Open"),
                        color=alt.Color("owner:N", legend=None, scale=alt.Scale(range=strong_palette(6))),
                        tooltip=["owner","status"]
                    ).properties(height=240, title="Open Tasks by Owner")
                )
                st.altair_chart(ch, use_container_width=True)
    else:
        st.info("No collaboration tasks CSV found yet.")

# ===================== EVENT LOG =====================
with tab_bus:
    st.subheader("🧾 Event Log (Local)")
    bus_table()