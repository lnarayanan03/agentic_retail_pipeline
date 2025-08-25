# pages/1_info.py
import os
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Optional: wordcloud (guarded so UI never crashes if lib/font missing)
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Agentic AI — Info & EDA", layout="wide")
st.title("📊 Agentic AI Retail — Information & EDA")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Simulated_Agentic_AI_Dataset_RandomizedDates.csv")
alt.data_transformers.disable_max_rows()

# -----------------------------
# LOAD
# -----------------------------
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize columns we use repeatedly
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["month_str"] = df["month"].dt.strftime("%b-%Y")
    for c in ["city", "brand", "keyword", "season"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # Numeric coercions (won’t crash if columns aren’t there)
    for c in ["units_sold", "tweet_count", "tweet_volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find dataset at `{DATA_PATH}`")
    st.stop()

df = load_df(DATA_PATH)
st.caption(f"Loaded **{len(df)} rows × {df.shape[1]} columns** from `{os.path.basename(DATA_PATH)}`")

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("Dataset Overview")
st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# SIDEBAR FILTERS (for EDA below)
# -----------------------------
st.sidebar.header("🔎 EDA Filters")
city_opts  = ["All"] + (sorted(df["city"].dropna().unique()) if "city" in df.columns else [])
brand_opts = ["All"] + (sorted(df["brand"].dropna().unique()) if "brand" in df.columns else [])
month_opts = ["All"] + (sorted(df["month_str"].dropna().unique()) if "month_str" in df.columns else [])

sel_city  = st.sidebar.selectbox("City", city_opts, index=0)
sel_brand = st.sidebar.selectbox("Brand", brand_opts, index=0)
sel_month = st.sidebar.selectbox("Month", month_opts, index=0)

def apply_filters(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    if sel_city != "All" and "city" in out.columns:
        out = out[out["city"] == sel_city]
    if sel_brand != "All" and "brand" in out.columns:
        out = out[out["brand"] == sel_brand]
    if sel_month != "All" and "month_str" in out.columns:
        out = out[out["month_str"] == sel_month]
    return out

df_view = apply_filters(df)

# -----------------------------
# EXPLORATORY VISUALIZATIONS
# -----------------------------
st.subheader("Exploratory Visualizations")

cols = st.columns(2)

with cols[0]:
    st.write("**Sales Over Time by City**")
    if {"month", "units_sold", "city"}.issubset(df.columns):
        ch = (
            alt.Chart(df_view if not df_view.empty else df)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("units_sold:Q", title="Units Sold"),
                color=alt.Color("city:N", title="City", scale=alt.Scale(scheme="tableau10")),
                tooltip=[alt.Tooltip("month:T"), "city:N", alt.Tooltip("units_sold:Q", format=".0f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("Missing columns for this chart: month, units_sold, city")

with cols[1]:
    st.write("**Tweet Count Over Time by City**")
    if {"month", "tweet_count", "city"}.issubset(df.columns):
        ch2 = (
            alt.Chart(df_view if not df_view.empty else df)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("tweet_count:Q", title="Tweet Count"),
                color=alt.Color("city:N", title="City", scale=alt.Scale(scheme="category10")),
                tooltip=[alt.Tooltip("month:T"), "city:N", alt.Tooltip("tweet_count:Q", format=".0f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(ch2, use_container_width=True)
    else:
        st.info("Missing columns for this chart: month, tweet_count, city")

st.write("**Tweet Count Over Time by Brand**")
if {"month", "tweet_count", "brand"}.issubset(df.columns):
    brand_chart = (
        alt.Chart(df_view if not df_view.empty else df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("tweet_count:Q", title="Tweet Count"),
            color=alt.Color("brand:N", title="Brand", scale=alt.Scale(scheme="dark2")),
            tooltip=[alt.Tooltip("month:T"), "brand:N", alt.Tooltip("tweet_count:Q", format=".0f")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(brand_chart, use_container_width=True)
else:
    st.info("Missing columns for this chart: month, tweet_count, brand")

# Word Cloud (guarded)
st.write("**Top Fashion Keywords Word Cloud**")
if WORDCLOUD_OK and "keyword" in df.columns and df["keyword"].notna().any():
    try:
        wc_text = " ".join(df["keyword"].dropna().astype(str))
        wc = WordCloud(width=900, height=300, background_color="white").generate(wc_text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"WordCloud skipped: {e}")
else:
    st.info("No keywords or wordcloud package unavailable.")

# Heatmap
st.write("**Heatmap: Tweet Volume by City and Season**")
if {"city", "season", "tweet_volume"}.issubset(df.columns):
    heatmap_data = df_view if not df_view.empty else df
    hm = heatmap_data.groupby(["city", "season"])["tweet_volume"].sum().unstack().fillna(0)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(hm, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax2)
    st.pyplot(fig2, use_container_width=True)
else:
    st.info("Missing columns for this heatmap: city, season, tweet_volume")

st.markdown("---")

# -----------------------------
# AGENT MODULES — RICH ABOUT CARDS
# -----------------------------
st.subheader("🧠 Agent Modules Overview")

AGENTS = [
    {
        "name": "Demand Sensing Agent",
        "icon": "📡",
        "status": "🟢 Active",
        "summary": "Listens to social signals (volume, velocity, and sentiment) to flag incipient demand spikes.",
        "sources": [
            "Social streams (Tweet volume & count, hashtags, keywords)",
            "Seasonal calendar & local events",
            "(Optional) Weather & influencer posts"
        ],
        "does": [
            "Cleans + filters brand/category mentions",
            "Computes moving-average baselines & z-scores",
            "Raises spike flags with reasoning"
        ],
        "outputs": [
            "`Detected_Spikes_Past.csv`",
            "`Detected_Spikes_Projected.csv`",
            "EDA charts: volume trend, heatmaps"
        ],
        "kpis": [
            "Detection precision/recall (vs. realized demand)",
            "Lead time (how early spikes are flagged)"
        ],
    },
    {
        "name": "Forecast Refinement Agent",
        "icon": "📈",
        "status": "🟢 Active",
        "summary": "Combines historical sales with sensing features to produce adaptive forecasts.",
        "sources": [
            "Historical sales (by month/city/brand)",
            "Sensing outputs (spike flags, volumes)",
            "Holiday flags, promo calendars"
        ],
        "does": [
            "Feature engineering (lags, rolling means, seasonality)",
            "Model training (ElasticNet + GBDT; Prophet optional)",
            "Selects best model on time-split CV"
        ],
        "outputs": [
            "`forecast_output.csv` (3-month horizon)",
            "Performance KPIs (RMSE, R²)",
            "Actual vs Pred chart"
        ],
        "kpis": [
            "RMSE / MAPE on holdout",
            "Bias & stability across cities/brands"
        ],
    },
    {
        "name": "Inventory Agent",
        "icon": "📦",
        "status": "🟢 Active",
        "summary": "Plans coverage & safety stock by city/brand using a 3‑month horizon.",
        "sources": [
            "`forecast_output.csv`",
            "`onhand_inventory_snapshot.csv`"
        ],
        "does": [
            "Builds planning grid (months × city × brand)",
            "Computes safety stock & shortfall",
            "Writes `inventory_plan.csv`"
        ],
        "outputs": [
            "`inventory_plan.csv`",
            "Coverage vs Demand charts"
        ],
        "kpis": [
            "Fill rate, stockout %",
            "Days of supply"
        ],
    },
    {
        "name": "Simulation Agent",
        "icon": "🔁",
        "status": "🟢 Active",
        "summary": "Scores strategies (Do Nothing / Reorder / Transfer / Hybrid) under weighted costs.",
        "sources": [
            "`forecast_output.csv` + `inventory_plan.csv`",
            "`cost_parameters.csv` (auto-created if missing)"
        ],
        "does": [
            "Computes required units with safety %",
            "Applies cost weights (raw, prod, logistics, transfer, SLA)",
            "Chooses best strategy per row"
        ],
        "outputs": [
            "`simulation_output.csv`",
            "Strategy mix & monthly cost charts",
            "Decision Assistant (city/brand)"
        ],
        "kpis": [
            "Total cost vs service achieved",
            "Strategy diversity & SLA risk"
        ],
    },
    {
        "name": "Collaboration Agent",
        "icon": "🤝",
        "status": "🟢 Active",
        "summary": "Turns decisions into shareable tasks/alerts; tracks owners, status, and approvals.",
        "sources": [
            "`simulation_output.csv`",
            "User entries (owners, notes, comms)"
        ],
        "does": [
            "Seeds tasks from simulation outputs",
            "Tracks status, owner, and SLA risk",
            "Captures comms + approvals"
        ],
        "outputs": [
            "`collab_tasks.csv`, `collab_comms.csv`, `collab_approvals.csv`",
            "Workload & risk dashboards"
        ],
        "kpis": [
            "Task cycle time",
            "Overdue & high‑risk counts"
        ],
    },
    {
        "name": "Learning Agent",
        "icon": "🧠",
        "status": "🟠 WIP",
        "summary": "Closes the loop: learns from outcomes to tune thresholds & model configs.",
        "sources": [
            "All agent outputs (KPIs, logs, final results)"
        ],
        "does": [
            "Collects decision/outcome pairs",
            "Finds patterns → recommends new thresholds",
            "(Optional) RL-style policy updates"
        ],
        "outputs": [
            "`learning_suggestions.json` (future)",
            "Dashboard of what improved/declined"
        ],
        "kpis": [
            "Error reduction across sprints",
            "Fewer escalations / stockouts"
        ],
    },
]

# Render agent cards
grid = st.columns(3)
for i, meta in enumerate(AGENTS):
    col = grid[i % 3]
    with col:
        st.markdown(
            f"""
            <div style="
                border:1px solid #e2e8f0; border-radius:14px; padding:14px 16px; margin-bottom:14px;">
              <div style="font-weight:800; font-size:18px; margin-bottom:4px;">
                {meta['icon']} {meta['name']}
              </div>
              <div style="margin-bottom:6px;">Status: {meta['status']}</div>
              <div style="font-size:14px; opacity:0.9; margin-bottom:8px;">
                {meta['summary']}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📚 Sources", expanded=False):
            for s in meta["sources"]:
                st.markdown(f"- {s}")
        with st.expander("⚙️ What this agent does", expanded=False):
            for s in meta["does"]:
                st.markdown(f"- {s}")
        with st.expander("📤 Outputs", expanded=False):
            for s in meta["outputs"]:
                st.markdown(f"- {s}")
        with st.expander("📏 KPIs / How to judge", expanded=False):
            for s in meta["kpis"]:
                st.markdown(f"- {s}")

st.markdown("---")
