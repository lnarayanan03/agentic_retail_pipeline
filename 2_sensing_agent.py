# pages/2_sensing_agent.py
import os
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Prophet is optional — we fall back to a moving-average forecast if unavailable
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Sensing Agent", layout="wide")
st.title("📡 Sensing Agent Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "Simulated_Agentic_AI_Dataset_RandomizedDates.csv")
OUT_PAST  = os.path.join(DATA_DIR, "Detected_Spikes_Past.csv")
OUT_FUT   = os.path.join(DATA_DIR, "Detected_Spikes_Projected.csv")
os.makedirs(DATA_DIR, exist_ok=True)
alt.data_transformers.disable_max_rows()

# -----------------------------
# LOAD
# -----------------------------
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # robust month parse
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"]).copy()
    # tidy strings
    for c in ["city", "brand", "keyword", "season"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # numeric coerce
    for c in ["tweet_count", "tweet_volume", "units_sold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # helpers
    df["month_str"] = df["month"].dt.strftime("%b-%Y")
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Data not found: `{DATA_PATH}`")
    st.stop()

df = load_df(DATA_PATH)
st.caption(f"Loaded **{len(df)} rows × {df.shape[1]} cols** from `{os.path.basename(DATA_PATH)}`")

# =====================================================
# 🔽 Business-friendly explainers (TOP)
# =====================================================
with st.expander("📥 What information does this listen to?", expanded=True):
    st.markdown("""
- **Social buzz** (Twitter/X): posts/mentions, hashtag volumes, overall engagement.
- **Context**: City, Brand, Season, Topic/Keyword (e.g., “puffer jacket”).
- **Grounding**: When available, recent **sell-through** to see whether attention converts to buying.
""")

with st.expander("🧠 What it does (plain language)", expanded=True):
    st.markdown("""
It **listens** to buzz for each **city × brand**, tracks the **recent baseline**, and **flags unusual, high-quality surges**:
1) Summarizes chatter & engagement each month per location and brand.  
2) Checks **quality** (engagement per post) and **unusualness** (jump vs. recent months).  
3) Looks **1–3 months ahead** to see if momentum is likely to persist.  
4) Publishes **simple spike lists** (past & projected) that other pages use automatically.
""")

with st.expander("🔗 What it hands off to other parts", expanded=True):
    st.markdown(f"""
- **Past spike list** → where meaningful buzz already happened.  
- **Projected next-month watchlist** → where to prepare stock or promos.

Files created:
- `{os.path.basename(OUT_PAST)}` — historical spikes  
- `{os.path.basename(OUT_FUT)}` — projected spikes
""")

with st.expander("✨ Why this adds practical value", expanded=True):
    st.markdown("""
- **Action-first**: Produces **clear flags** per location & brand—easy to plan against.  
- **Adaptive by locality**: Uses **recent, local baselines** so Buffalo and Austin aren’t treated the same.  
- **Controllable**: Sensitivity knobs are in the UI (no retraining needed).  
- **Resilient**: If advanced forecasting isn’t available, it **falls back** and keeps signals flowing.  
- **Explainable**: You can see the **quality** and **unusualness** logic behind alerts.
""")

# NEW — subtle, deeper “agentic” explainer
with st.expander("🧭 Design notes: how this behaves like a proactive sensing assistant", expanded=True):
    st.markdown("""
Rather than only charting data, this module is designed to **move information toward decisions**:

**1) Decisions, not just predictions**  
It turns social signals into **operational triggers** (“spike” for a city/brand), which downstream pages can act on immediately.

**2) Policy you can steer at runtime**  
The thresholds for “quality” and “unusualness” are **business-tunable** in the sidebar. You can tighten/loosen alerting **without code changes**.

**3) Automatic hand-offs**  
Outputs are saved as **clean CSVs** that other pages read. This keeps the flow from sensing → forecast → inventory/simulation **frictionless**.

**4) Local context baked in**  
It measures each city/brand **against its own recent history**, capturing neighborhood-level momentum instead of one global rule.

**5) Robust by default**  
If a forecasting library isn’t present, it **degrades gracefully** to a moving-average approach, ensuring continuity of signals.

**6) Transparent & auditable**  
Charts show what happened and **why it was flagged**. That traceability helps planners trust and adopt the recommendations.

In short, this sensing module complements classic analytics by **operationalizing the “what now?” step**—it’s built to trigger planning actions reliably, not just display numbers.
""")

# -----------------------------
# SIDEBAR: FILTERS + SPIKE PARAMS
# -----------------------------
st.sidebar.header("🔎 Filters")
city_opts  = sorted(df["city"].dropna().unique()) if "city" in df.columns else []
brand_opts = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
kw_opts    = sorted(df["keyword"].dropna().unique()) if "keyword" in df.columns else []

sel_city    = st.sidebar.selectbox("City", city_opts, index=0) if city_opts else ""
sel_brand   = st.sidebar.selectbox("Brand", brand_opts, index=0) if brand_opts else ""
sel_keyword = st.sidebar.selectbox("Topic/Keyword", kw_opts, index=0) if kw_opts else ""

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Spike Sensitivity")
ratio_thresh = st.sidebar.slider("Min engagement per post (ratio)", 0.5, 3.0, 1.30, 0.05)
z_thresh     = st.sidebar.slider("Min jump vs recent months (z-score)", 0.0, 5.0, 1.0, 0.1)
min_history  = st.sidebar.slider("Min months of history", 4, 24, 6, 1)
months_ahead = st.sidebar.slider("Forecast months ahead", 1, 6, 3, 1)

def filter_view(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    if sel_city and "city" in out.columns:
        out = out[out["city"] == sel_city]
    if sel_brand and "brand" in out.columns:
        out = out[out["brand"] == sel_brand]
    if sel_keyword and "keyword" in out.columns:
        out = out[out["keyword"] == sel_keyword]
    return out

df_view = filter_view(df)

# -----------------------------
# KPIs
# -----------------------------
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.metric("Cities tracked", len(df["city"].dropna().unique()) if "city" in df.columns else 0)
with k2:
    st.metric("Brands tracked", len(df["brand"].dropna().unique()) if "brand" in df.columns else 0)
with k3:
    try:
        latest = df["month"].max()
        st.metric("Data freshness", latest.strftime("%b %Y") if pd.notna(latest) else "—")
    except Exception:
        st.metric("Data freshness", "—")
with k4:
    pairs = df.dropna(subset=["city","brand"]).drop_duplicates(["city","brand"])
    st.metric("City × Brand pairs", int(len(pairs)))

st.markdown("---")

# -----------------------------
# EDA
# -----------------------------
st.subheader("📊 Exploratory Visualizations")

c1, c2 = st.columns(2)
with c1:
    st.write("**Tweet Count Over Time by City**")
    if {"month","tweet_count","city"}.issubset(df.columns):
        ch = (
            alt.Chart(df_view if not df_view.empty else df)
            .mark_line(point=True, strokeDash=[5,3])
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("tweet_count:Q", title="Tweet Count"),
                color=alt.Color("city:N", title="City"),
                tooltip=[alt.Tooltip("month:T"), "city:N", alt.Tooltip("tweet_count:Q", format=".0f")],
            )
            .properties(height=340)
            .interactive()
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("Need columns: month, tweet_count, city")

with c2:
    st.write("**Units Sold Distribution by Brand**")
    if {"brand","units_sold"}.issubset(df.columns):
        bar = (
            alt.Chart(df_view if not df_view.empty else df)
            .mark_bar()
            .encode(
                x=alt.X("brand:N", title="Brand"),
                y=alt.Y("units_sold:Q", title="Units Sold"),
                color=alt.Color("brand:N", legend=None),
                tooltip=["brand:N", alt.Tooltip("units_sold:Q", format=".0f")],
            )
            .properties(height=340)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Need columns: brand, units_sold")

st.write("**Tweet Count Over Time by Brand**")
if {"month","tweet_count","brand"}.issubset(df.columns):
    ch3 = (
        alt.Chart(df_view if not df_view.empty else df)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("tweet_count:Q", title="Tweet Count"),
            color=alt.Color("brand:N", title="Brand"),
            tooltip=[alt.Tooltip("month:T"), "brand:N", alt.Tooltip("tweet_count:Q", format=".0f")],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(ch3, use_container_width=True)
else:
    st.info("Need columns: month, tweet_count, brand")

st.write("**Heatmap: Tweet Volume by City and Season**")
if {"city","season","tweet_volume"}.issubset(df.columns):
    hm = (df_view if not df_view.empty else df).groupby(["city","season"])["tweet_volume"].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(hm, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Need columns: city, season, tweet_volume")

# Waterfall-like month-over-month change (choose brand)
if "brand" in df.columns and "tweet_volume" in df.columns:
    sample_brand = st.selectbox("Brand for MoM Tweet Volume Change", sorted(df["brand"].dropna().unique()))
    bdf = df[df["brand"] == sample_brand].sort_values("month").copy()
    bdf["volume_change"] = bdf["tweet_volume"].diff().fillna(0)
    bdf["month_str"] = bdf["month"].dt.strftime("%b-%Y")
    bdf["change_type"] = np.where(bdf["volume_change"] >= 0, "Increase", "Decrease")
    ch4 = (
        alt.Chart(bdf)
        .mark_bar()
        .encode(
            x=alt.X("month_str:N", title="Month"),
            y=alt.Y("volume_change:Q", title="Δ Tweet Volume"),
            color=alt.Color("change_type:N",
                            scale=alt.Scale(domain=["Increase","Decrease"], range=["#2f9e44","#e03131"])),
            tooltip=["month_str:N", "change_type:N", alt.Tooltip("volume_change:Q", format=".0f")],
        )
        .properties(height=360)
    )
    st.altair_chart(ch4, use_container_width=True)

st.markdown("---")

# -----------------------------
# SPIKE DETECTION
# -----------------------------
def _zscore(x: pd.Series, win=3):
    m = x.rolling(win, min_periods=1).mean()
    s = x.rolling(win, min_periods=1).std(ddof=0).replace(0, 1e-6)
    return (x - m) / s

def detect_past_spikes(data: pd.DataFrame, ratio_thr: float, z_thr: float) -> pd.DataFrame:
    need = {"month","city","brand","tweet_volume","tweet_count"}
    if not need.issubset(data.columns):
        return pd.DataFrame(columns=["month","city","brand","spike_detected"])
    g = (data
         .groupby(["city","brand","month"], as_index=False)
         .agg(tweet_volume=("tweet_volume","sum"),
              tweet_count=("tweet_count","sum")))
    g["ratio"] = g["tweet_volume"] / g["tweet_count"].replace(0, np.nan)
    g["z"] = g.groupby(["city","brand"])["tweet_volume"].transform(_zscore)
    g["spike_detected"] = (g["ratio"] > ratio_thr) & (g["z"] > z_thr)
    spikes = g[g["spike_detected"]].copy()
    if spikes.empty:
        return pd.DataFrame(columns=["month","city","brand","spike_detected"])
    spikes["spike_detected"] = "Yes"
    spikes["month"] = spikes["month"].dt.to_period("M").astype(str)
    return spikes[["month","city","brand","spike_detected"]]

def forecast_and_detect_future_spikes(data: pd.DataFrame,
                                      months_ahead: int,
                                      ratio_thr: float,
                                      z_thr: float,
                                      min_hist: int) -> pd.DataFrame:
    need_cols = {"month","city","brand","tweet_volume","tweet_count"}
    if not need_cols.issubset(data.columns):
        return pd.DataFrame(columns=["month","city","brand","spike_detected"])

    out = []
    for (ct, br), sub in data.groupby(["city","brand"]):
        s = sub[["month","tweet_volume","tweet_count"]].dropna().sort_values("month").copy()
        if len(s) < min_hist:
            continue

        # Forecast tweet_volume next few months
        if PROPHET_OK:
            tmp = s.rename(columns={"month":"ds", "tweet_volume":"y"})
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            try:
                m.fit(tmp[["ds","y"]])
                future = m.make_future_dataframe(periods=months_ahead, freq="MS")
                fc = m.predict(future)[["ds","yhat"]].tail(months_ahead).rename(columns={"ds":"month","yhat":"yhat"})
                yhat = fc["yhat"].values
                idx  = fc["month"].values
            except Exception:
                base = pd.Series(s["tweet_volume"]).rolling(3, min_periods=1).mean().iloc[-1]
                yhat = np.repeat(base, months_ahead)
                idx  = pd.date_range(s["month"].max() + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS")
        else:
            base = pd.Series(s["tweet_volume"]).rolling(3, min_periods=1).mean().iloc[-1]
            yhat = np.repeat(base, months_ahead)
            idx  = pd.date_range(s["month"].max() + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS")

        ratio = yhat / max(s["tweet_count"].mean(), 1e-6)
        recent = s["tweet_volume"].tail(3)
        r_mu, r_sd = recent.mean(), max(recent.std(ddof=0), 1e-6)
        z = (yhat - r_mu) / r_sd

        flag = (ratio > ratio_thr) & (z > z_thr)
        if flag.any():
            det = pd.DataFrame({
                "month": pd.to_datetime(idx).astype("datetime64[ns]"),
                "city": ct,
                "brand": br,
                "spike_detected": np.where(flag, "Yes", None)
            })
            det = det[det["spike_detected"].notna()].copy()
            det["month"] = det["month"].dt.to_period("M").astype(str)
            out.append(det[["month","city","brand","spike_detected"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["month","city","brand","spike_detected"])

st.subheader("🔍 Spike Detection Summary")

# Run detection
past_spikes = detect_past_spikes(df, ratio_thresh, z_thresh)
future_spikes = forecast_and_detect_future_spikes(df, months_ahead, ratio_thresh, z_thresh, min_history)

# Save outputs
past_spikes.to_csv(OUT_PAST, index=False)
future_spikes.to_csv(OUT_FUT, index=False)

# Results tables (business labels)
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 🕓 Past Spikes")
    st.dataframe(
        past_spikes.rename(columns={"month":"Period","city":"City","brand":"Brand","spike_detected":"Spike"}),
        use_container_width=True, height=260
    )
with c2:
    st.markdown("### 🔮 Projected Spikes")
    st.dataframe(
        future_spikes.rename(columns={"month":"Period","city":"City","brand":"Brand","spike_detected":"Spike"}),
        use_container_width=True, height=260
    )

# -----------------------------
# POST-SENSING EDA
# -----------------------------
st.subheader("📈 Post-Sensing EDA")

df_eda = df.copy()
df_eda["month_dt"] = df_eda["month"].dt.to_period("M").dt.to_timestamp()
for d in (past_spikes, future_spikes):
    if "month" in d.columns:
        d["month_dt"] = pd.to_datetime(d["month"], format="%Y-%m", errors="coerce")
        m2 = pd.to_datetime(d["month"], format="%b-%Y", errors="coerce")
        d["month_dt"] = d["month_dt"].fillna(m2)

# Past spikes: brand counts
if not past_spikes.empty:
    brand_counts = past_spikes.groupby("brand", as_index=False).size()
    ch_b = (
        alt.Chart(brand_counts)
        .mark_bar()
        .encode(
            x=alt.X("brand:N", title="Brand"),
            y=alt.Y("size:Q", title="Spikes"),
            color=alt.Color("brand:N", legend=None),
            tooltip=["brand:N","size:Q"],
        )
        .properties(height=320, title="Past Spikes by Brand")
    )
    st.altair_chart(ch_b, use_container_width=True)

# Past spikes: tweet volume line (merged)
try:
    merged = df_eda.merge(past_spikes, left_on=["month_str","city","brand"],
                          right_on=["month","city","brand"], how="inner")
    if not merged.empty and {"month_dt","tweet_volume","brand"}.issubset(merged.columns):
        ch_tv = (
            alt.Chart(merged)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_dt:T", title="Month"),
                y=alt.Y("tweet_volume:Q", title="Tweet Volume"),
                color=alt.Color("brand:N", title="Brand"),
                tooltip=[alt.Tooltip("month_dt:T"), "brand:N", alt.Tooltip("tweet_volume:Q", format=".0f")],
            )
            .properties(height=320, title="Tweet Volume — Spiked Brands")
        )
        st.altair_chart(ch_tv, use_container_width=True)
except Exception as e:
    st.warning(f"Merge for volume chart skipped: {e}")

# Past spikes: heatmap city×brand
if not past_spikes.empty:
    hm = past_spikes.pivot_table(index="city", columns="brand", aggfunc="size", fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(hm, annot=True, cmap="Reds", fmt="d", ax=ax2)
    st.pyplot(fig2, use_container_width=True)

# Future spikes: city distribution
if not future_spikes.empty:
    city_counts = future_spikes.groupby("city", as_index=False).size()
    ch_fc = (
        alt.Chart(city_counts)
        .mark_bar()
        .encode(
            x=alt.X("city:N", title="City"),
            y=alt.Y("size:Q", title="Spikes"),
            color=alt.value("#1f77b4"),
            tooltip=["city:N","size:Q"],
        )
        .properties(height=320, title="Projected Spikes by City")
    )
    st.altair_chart(ch_fc, use_container_width=True)

# Future spikes: heatmap city×brand
if not future_spikes.empty:
    hm2 = future_spikes.pivot_table(index="city", columns="brand", aggfunc="size", fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(hm2, annot=True, cmap="Blues", fmt="d", ax=ax3)
    st.pyplot(fig3, use_container_width=True)

# -----------------------------
# COMPLETION MESSAGE (high-contrast)
# -----------------------------
bg = "#ffffff"
fg = "#111111"
st.markdown(
    f"""
<div style="
  border-radius:12px;
  padding:14px 16px;
  margin-top:8px;
  border:1px solid rgba(0,0,0,.15);
  background:{bg};
  color:{fg};
">
  <div style="font-weight:800; font-size:16px; margin-bottom:6px;">Sensing Completed</div>
  Saved:
  <ul style="margin:6px 0 0 18px;">
    <li><code>{os.path.basename(OUT_PAST)}</code> — historical spikes</li>
    <li><code>{os.path.basename(OUT_FUT)}</code> — projected spikes ({months_ahead}M ahead)</li>
  </ul>
  Prophet available: <b>{'Yes' if PROPHET_OK else 'No (using moving-average fallback)'}</b>
</div>
""",
    unsafe_allow_html=True
)