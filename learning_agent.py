# pages/7_learning_agent.py
import os
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Learning Agent", layout="wide")
st.title("🧠 Learning Agent — Threshold & Policy Tuning")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")

# Core artifacts used by this page
SENSE_SRC  = os.path.join(DATA_DIR, "Simulated_Agentic_AI_Dataset_RandomizedDates.csv")
PAST_SPIKES= os.path.join(DATA_DIR, "Detected_Spikes_Past.csv")
FUT_SPIKES = os.path.join(DATA_DIR, "Detected_Spikes_Projected.csv")
FC_PANEL   = os.path.join(DATA_DIR, "ForecastingDataset_Final_Cleaned.csv")
FC_OUT     = os.path.join(DATA_DIR, "forecast_output.csv")
INV_SNAP   = os.path.join(DATA_DIR, "onhand_inventory_snapshot.csv")
INV_PLAN   = os.path.join(DATA_DIR, "inventory_plan.csv")
COSTS      = os.path.join(DATA_DIR, "cost_parameters.csv")

SUGGEST_JSON = os.path.join(DATA_DIR, "learning_suggestions.json")

os.makedirs(DATA_DIR, exist_ok=True)
alt.data_transformers.disable_max_rows()

# ==========================================
# HELPERS
# ==========================================
def safe_read_csv(path, parse_dt_cols=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    parse_dt_cols = parse_dt_cols or []
    for c in parse_dt_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def zscore_rolling(x, win=3):
    x = pd.Series(x, dtype="float")
    m = x.rolling(win, min_periods=1).mean()
    s = x.rolling(win, min_periods=1).std(ddof=0).replace(0, 1e-6)
    return (x - m) / s

def sales_uplift_label(sales, win=3, uplift_pct=0.15):
    """Label months where sales are > (1+uplift_pct) × rolling mean of prior 'win' months."""
    s = pd.Series(sales, dtype="float")
    base = s.shift(1).rolling(win).mean()
    return (s > (1.0 + uplift_pct) * base).astype(int)

def f1_from_pr(prec, rec):
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true==1) & (y_pred==1)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    return prec, rec, f1_from_pr(prec, rec), tp, fp, fn

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ==========================================
# LOAD DATA
# ==========================================
sense_df = safe_read_csv(SENSE_SRC)
fc_panel = safe_read_csv(FC_PANEL)
fc_out   = safe_read_csv(FC_OUT)
inv_snap = safe_read_csv(INV_SNAP)
inv_plan = safe_read_csv(INV_PLAN)
costs_df = safe_read_csv(COSTS)

# Normalize key columns
for df in [sense_df, fc_panel, fc_out, inv_snap, inv_plan]:
    if df.empty: continue
    if "month" in df.columns:
        m1 = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
        m2 = pd.to_datetime(df["month"], errors="coerce")
        df["month_dt"] = m1.fillna(m2)
        df["month_str"] = df["month_dt"].dt.to_period("M").astype(str)
    for c in ["city","brand","keyword","fashion_trend"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Learning Controls")

# ---- Sensing threshold tuning controls
uplift_pct = st.sidebar.slider("Sales uplift threshold (for 'ground truth')", 0.05, 0.50, 0.15, 0.05)
win_z      = st.sidebar.slider("Rolling window for z-score (months)", 3, 6, 3, 1)
q_grid     = np.round(np.arange(1.0, 2.1, 0.1), 2)   # tau_q candidates
z_grid     = np.round(np.arange(0.5, 2.6, 0.1), 2)   # tau_z candidates

# ---- Safety % heuristic controls
service_target = st.sidebar.selectbox("Service target proxy", ["90% (z≈1.28)","95% (z≈1.65)","97.5% (z≈1.96)"], index=1)
z_map = {"90% (z≈1.28)":1.28, "95% (z≈1.65)":1.65, "97.5% (z≈1.96)":1.96}
z_service = z_map[service_target]
max_safety_cap = st.sidebar.slider("Cap safety %", 0.10, 0.60, 0.35, 0.05)

# ==========================================
# SECTION A — TUNE SENSING (τq, τz)
# ==========================================
st.subheader("A) Tune Sensing Thresholds (quality + unusualness)")

if sense_df.empty or not {"month","city","brand","tweet_volume","tweet_count","units_sold"}.issubset(sense_df.columns):
    st.info("Need Sensing source with columns: month, city, brand, tweet_volume, tweet_count, units_sold.")
else:
    g = (sense_df[["month_dt","month_str","city","brand","tweet_volume","tweet_count","units_sold"]]
         .dropna(subset=["month_dt"]).copy())
    g.sort_values(["city","brand","month_dt"], inplace=True)

    # Build per (city,brand) rolling z and ratio; and ground-truth 'uplift' from units_sold
    parts = []
    for (ct, br), sub in g.groupby(["city","brand"]):
        sub = sub.copy().sort_values("month_dt")
        sub["ratio"] = sub["tweet_volume"] / sub["tweet_count"].replace(0, np.nan)
        sub["z_tv"]  = zscore_rolling(sub["tweet_volume"], win=win_z)
        sub["y_uplift"] = sales_uplift_label(sub["units_sold"], win=win_z, uplift_pct=uplift_pct)
        parts.append(sub)
    grid = pd.concat(parts, ignore_index=True).dropna(subset=["ratio","z_tv","y_uplift"])

    # Grid search
    results = []
    for tq in q_grid:
        for tz in z_grid:
            pred = ((grid["ratio"] > tq) & (grid["z_tv"] > tz)).astype(int)
            prec, rec, f1, tp, fp, fn = precision_recall_f1(grid["y_uplift"].values, pred.values)
            results.append({"tau_q": float(tq), "tau_z": float(tz), "precision":prec, "recall":rec, "f1":f1, "tp":tp, "fp":fp, "fn":fn})
    res_df = pd.DataFrame(results).sort_values(["f1","recall"], ascending=[False, False])

    best = res_df.iloc[0].to_dict() if not res_df.empty else None
    if best:
        c1,c2,c3 = st.columns(3)
        c1.metric("Best τq", f"{best['tau_q']:.2f}")
        c2.metric("Best τz", f"{best['tau_z']:.2f}")
        c3.metric("F1 (micro)", f"{best['f1']:.3f}")
        st.caption(f"Precision={best['precision']:.3f} · Recall={best['recall']:.3f} · TP={int(best['tp'])} · FP={int(best['fp'])} · FN={int(best['fn'])}")

        # Heatmap of F1 landscape
        heat = res_df.pivot_table(index="tau_q", columns="tau_z", values="f1")
        hm = heat.reset_index().melt("tau_q", var_name="tau_z", value_name="f1")
        ch = (
            alt.Chart(hm)
            .mark_rect()
            .encode(
                x=alt.X("tau_z:Q", title="τz"),
                y=alt.Y("tau_q:Q", title="τq"),
                color=alt.Color("f1:Q", title="F1", scale=alt.Scale(scheme="turbo")),
                tooltip=["tau_q","tau_z",alt.Tooltip("f1:Q", format=".3f")]
            ).properties(height=320, title="F1 grid — Sensing thresholds")
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.warning("Could not compute threshold grid (insufficient valid rows).")

# ==========================================
# SECTION B — SUGGEST SAFETY % (s)
# ==========================================
st.subheader("B) Suggest Safety Stock Percentage (s)")

# Heuristic: safety % ≈ clamp( z_service × mean(CV by city-brand) × 0.5, 5%..cap )
# where CV = std(sales) / mean(sales) from FC panel (historical target variable)
if fc_panel.empty or "sales" not in fc_panel.columns:
    st.info("Need Forecasting panel with 'sales' to compute volatility-based safety %.")
    safety_suggest = None
else:
    panel = fc_panel.copy()
    if "month" in panel.columns:
        m1 = pd.to_datetime(panel["month"], format="%Y-%m", errors="coerce")
        panel["month_dt"] = m1.fillna(pd.to_datetime(panel["month"], errors="coerce"))
    # group by city-brand if present
    keys = [c for c in ["city","brand"] if c in panel.columns]
    if keys:
        agg = (panel.dropna(subset=["sales"])
               .groupby(keys)["sales"].agg(["mean","std"])
               .reset_index())
        agg["cv"] = (agg["std"] / agg["mean"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cv_mean = float(agg["cv"].mean())
    else:
        s = panel["sales"].dropna()
        cv_mean = float(s.std(ddof=0) / (s.mean() if s.mean()!=0 else 1.0))

    # A gentle mapping from volatility to s (kept conservative for demo)
    s_raw = z_service * cv_mean * 0.5
    safety_suggest = clamp(s_raw, 0.05, max_safety_cap)

    c1,c2,c3 = st.columns(3)
    c1.metric("Mean CV (sales)", f"{cv_mean:.3f}")
    c2.metric("Service z", f"{z_service:.2f}")
    c3.metric("Suggested s", f"{safety_suggest:.2%}")

    # Show distribution of CV by brand/city if available
    if "cv" in locals() and "agg" in locals() and not agg.empty:
        top = agg.sort_values("cv", ascending=False).head(15)
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("cv:Q", title="Coefficient of Variation"),
                y=alt.Y((("brand:N" if "brand" in agg.columns else "index:N")),
                        sort="-x",
                        title=("Brand" if "brand" in agg.columns else "")),
                color=alt.Color(("city:N" if "city" in agg.columns else "cv:Q"), legend=("bottom" if "city" in agg.columns else None)),
                tooltip=[c for c in ["city","brand","mean","std","cv"] if c in agg.columns]
            ).properties(height=320, title="Top volatility (higher CV)")
        )
        st.altair_chart(chart, use_container_width=True)

# ==========================================
# SECTION C — DRIFT CHECKS (feature→sales)
# ==========================================
st.subheader("C) Simple Drift Checks")

if fc_panel.empty:
    st.info("Need Forecasting panel to run drift checks.")
else:
    df = fc_panel.copy()
    # try month parse if needed
    m1 = pd.to_datetime(df.get("month"), format="%Y-%m", errors="coerce")
    df["month_dt"] = m1.fillna(pd.to_datetime(df.get("month"), errors="coerce"))
    df = df.dropna(subset=["month_dt","sales"])
    df.sort_values("month_dt", inplace=True)

    # split into early vs recent halves
    cut = int(len(df) * 0.5)
    a, b = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    drift_rows = []
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["sales"]]
    for c in num_cols:
        if a[c].nunique() < 2 or b[c].nunique() < 2:
            continue
        corr_a = a[[c,"sales"]].corr().iloc[0,1]
        corr_b = b[[c,"sales"]].corr().iloc[0,1]
        if pd.isna(corr_a) or pd.isna(corr_b): 
            continue
        delta = float(corr_b - corr_a)
        drift_rows.append({"feature": c, "corr_early": float(corr_a), "corr_recent": float(corr_b), "delta": delta})
    drift = pd.DataFrame(drift_rows).sort_values("delta", ascending=True)  # most negative first (possible flip)

    if drift.empty:
        st.info("No numeric features with enough variation to compute correlation drift.")
    else:
        st.markdown("**Largest correlation shifts (early → recent)**")
        st.dataframe(drift.head(20), use_container_width=True, height=260)
        ch = (
            alt.Chart(drift.head(20))
            .mark_bar()
            .encode(
                x=alt.X("delta:Q", title="Δ correlation (recent - early)"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                color=alt.Color("delta:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=[alt.Tooltip("feature:N"),
                         alt.Tooltip("corr_early:Q", format=".3f"),
                         alt.Tooltip("corr_recent:Q", format=".3f"),
                         alt.Tooltip("delta:Q", format=".3f")]
            ).properties(height=320, title="Top correlation drifts")
        )
        st.altair_chart(ch, use_container_width=True)

# ==========================================
# SECTION D — WRITE SUGGESTIONS
# ==========================================
st.subheader("D) Write Learning Suggestions")

suggestions = {}
# sensing thresholds
if 'best' in locals() and best:
    suggestions["sensing_thresholds"] = {
        "tau_q": round(float(best["tau_q"]), 2),
        "tau_z": round(float(best["tau_z"]), 2),
        "metrics": {
            "precision": round(float(best["precision"]), 4),
            "recall":    round(float(best["recall"]), 4),
            "f1":        round(float(best["f1"]), 4),
            "tp": int(best["tp"]), "fp": int(best["fp"]), "fn": int(best["fn"])
        },
        "params": {
            "uplift_pct": uplift_pct,
            "z_window":   win_z
        }
    }

# safety suggestion
if 'safety_suggest' in locals() and safety_suggest is not None:
    suggestions["inventory_policy"] = {
        "suggested_safety_pct": round(float(safety_suggest), 4),
        "service_target": service_target
    }

# drift summary
if 'drift' in locals() and not drift.empty:
    suggestions["drift_alerts"] = {
        "top_features": drift.head(10).to_dict(orient="records")
    }

# Save JSON + show
if suggestions:
    with open(SUGGEST_JSON, "w") as f:
        json.dump(suggestions, f, indent=2)
    st.success(f"✅ Suggestions written to `{SUGGEST_JSON}`")
    st.code(json.dumps(suggestions, indent=2), language="json")
else:
    st.info("No suggestions to write yet — check data inputs above.")

st.caption("Notes: Sensing thresholds are tuned against a sales-uplift proxy; safety % uses a volatility heuristic; drift uses early vs recent correlation shifts. All are local, CSV-driven, and reproducible.")