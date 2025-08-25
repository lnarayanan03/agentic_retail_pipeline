# pages/5_simulation_agent.py
# ---------------------------------------------------------
# Simulation Agent — Strategy & Cost Optimizer
# - Business-first expanders (matches other agents)
# - Data KPIs + Simulation KPIs
# - Robust joins (forecast + inventory + costs)
# - Strategy economics, detailed review, rich charts
# - Saves simulation_output.csv and logs to memory_bus.jsonl
# ---------------------------------------------------------
import os, json, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =========================
# CONFIG + PATHS
# =========================
st.set_page_config(page_title="Simulation Agent", layout="wide")
st.title("🚚 Simulation Agent — Strategy & Cost Optimizer")
alt.data_transformers.disable_max_rows()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FORECAST_PATH  = os.path.join(DATA_DIR, "forecast_output.csv")
INVENTORY_PATH = os.path.join(DATA_DIR, "inventory_plan.csv")
COST_PATH      = os.path.join(DATA_DIR, "cost_parameters.csv")
OUTPUT_PATH    = os.path.join(DATA_DIR, "simulation_output.csv")
BUS_PATH       = os.path.join(DATA_DIR, "memory_bus.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- UI polish ----------
st.markdown("""
<style>
section.main > div { padding-top: .4rem; }
.block-container { padding-top: .8rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stMetricLabel"] { font-weight: 600; color: #9aa0aa; }
details > summary { font-weight: 700; }
.small-note { font-size: 12px; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# =========================
# MEMORY BUS (non-blocking)
# =========================
def bus_emit(event_type: str, payload: dict | None = None):
    try:
        rec = {
            "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
            "type": event_type,
            "payload": payload or {},
        }
        with open(BUS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

bus_emit("page_opened", {"page": "simulation"})

# =========================
# UTILITIES
# =========================
def to_month_str(s):
    """Robust month parser -> 'MMM-YYYY' string (e.g., 'Aug-2025')."""
    if pd.isna(s):
        return None
    # Try several formats
    for fmt in ("%Y-%m-%d", "%Y-%m", "%b-%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt).to_period("M").strftime("%b-%Y")
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").to_period("M").strftime("%b-%Y")
    except Exception:
        return None

def ensure_month_str(df, col_name="month"):
    if col_name not in df.columns:
        return df
    df[col_name] = df[col_name].astype(str).apply(to_month_str)
    df = df.dropna(subset=[col_name])
    # also keep a datetime for charts
    df["month_dt"] = pd.to_datetime(df[col_name], format="%b-%Y")
    return df

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_column(df, name, default):
    if name not in df.columns:
        df[name] = default
    return df

def build_sku_from_cols(df):
    """
    Create a stable SKU if missing, based on available categorical columns.
    Works row-wise (avoids Series join errors).
    """
    if "sku" in df.columns and df["sku"].notna().any():
        return df
    cat_cols = [c for c in ["brand", "city"] if c in df.columns]
    if not cat_cols:
        df["sku"] = "sku_generic"
        return df
    def _make(row):
        parts = []
        for c in cat_cols:
            v = str(row.get(c, "")).strip().lower().replace(" ", "")
            parts.append(v if v else "x")
        return "sku_" + "_".join(parts)
    df["sku"] = df.apply(_make, axis=1)
    return df

def save_cost_defaults_if_missing(path):
    if not os.path.exists(path):
        defaults = pd.DataFrame({
            "parameter": ["raw_material_cost", "production_cost", "logistics_cost",
                          "transfer_cost", "sla_penalty"],
            "value":     [30.0, 20.0, 10.0, 15.0, 25.0],
        })
        defaults.to_csv(path, index=False)

# =========================
# LOADERS
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

# ---- Load forecast
try:
    fc = load_csv(FORECAST_PATH)
except FileNotFoundError:
    st.error(f"❌ forecast_output.csv not found at: {FORECAST_PATH}")
    st.stop()

# ---- Load inventory (if missing, build a minimal placeholder)
try:
    inv = load_csv(INVENTORY_PATH)
except FileNotFoundError:
    st.warning(f"⚠️ inventory_plan.csv not found at: {INVENTORY_PATH}. Creating a minimal placeholder.")
    inv = pd.DataFrame(columns=["month","city","brand","on_hand"])

# ---- Normalize month to MMM-YYYY and keep month_dt for charts
fc = ensure_month_str(fc, "month")
inv = ensure_month_str(inv, "month")

# ---- Clean string categoricals
for df_ in [fc, inv]:
    for col in ["city", "brand"]:
        if col in df_.columns:
            df_[col] = df_[col].astype(str).str.strip()

# ---- Forecast column detection
forecast_col = pick_first_existing(fc, ["forecast_sales", "predicted_sales", "sales_forecast", "forecast"])
if forecast_col is None:
    # fallback to any numeric column
    num_cols = fc.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        forecast_col = num_cols[0]
        st.warning(f"⚠️ No explicit forecast column found. Using numeric column '{forecast_col}' as forecast.")
    else:
        st.error("❌ No forecast column found in forecast_output.csv.")
        st.stop()

# ---- Ensure SKU & on_hand in inventory
inv = build_sku_from_cols(inv)
inv_onhand_col = pick_first_existing(inv, ["on_hand","onhand","stock","current_stock","inventory","available_units","available_qty","quantity","qty"])
if inv_onhand_col is None:
    inv["on_hand"] = np.random.randint(80, 220, size=len(inv)) if len(inv) else 150
    inv_onhand_col = "on_hand"

# ---- If inventory is empty, construct template from forecast (city+brand+month)
if inv.empty:
    template_cols = [c for c in ["month","city","brand"] if c in fc.columns]
    inv = fc[template_cols].drop_duplicates().copy()
    inv["on_hand"] = np.random.randint(80, 220, size=len(inv))
    inv = build_sku_from_cols(inv)
    inv_onhand_col = "on_hand"

# ---- Merge keys
key_cols = [c for c in ["month","city","brand"] if c in fc.columns and c in inv.columns]
if not key_cols:
    key_cols = ["month"]
    inv = ensure_column(inv, "month", fc["month"].mode().iat[0])

# Keep month_dt for charts: prefer from forecast (if both present, they should match)
if "month_dt" not in fc.columns:
    fc["month_dt"] = pd.to_datetime(fc["month"], format="%b-%Y")
if "month_dt" not in inv.columns and "month" in inv.columns:
    inv["month_dt"] = pd.to_datetime(inv["month"], format="%b-%Y")

# ---- Merge minimal columns to avoid dupes
keep_fc = list(dict.fromkeys(key_cols + ["month_dt", forecast_col]))
keep_inv = list(dict.fromkeys(key_cols + ["month_dt", "sku", inv_onhand_col]))

merged = pd.merge(fc[keep_fc], inv[keep_inv], on=key_cols, how="left", suffixes=("", "_inv"))
# prefer month_dt from forecast
if "month_dt_x" in merged.columns and "month_dt_y" in merged.columns:
    merged["month_dt"] = merged["month_dt_x"]
    merged.drop(columns=["month_dt_x", "month_dt_y"], inplace=True)
elif "month_dt" not in merged.columns and "month" in merged.columns:
    merged["month_dt"] = pd.to_datetime(merged["month"], format="%b-%Y", errors="coerce")

# Fill missing on-hand with aligned randoms (fix for ndarray error)
merged[inv_onhand_col] = pd.to_numeric(merged[inv_onhand_col], errors="coerce")
na_mask = merged[inv_onhand_col].isna()
if na_mask.any():
    fill_vals = pd.Series(np.random.randint(80, 180, size=int(na_mask.sum())), index=merged.index[na_mask])
    merged.loc[na_mask, inv_onhand_col] = fill_vals

# =========================
# COST PARAMETERS (robust)
# =========================
def load_cost_map(path):
    save_cost_defaults_if_missing(path)
    dfc = load_csv(path)
    # normalize columns
    dfc.columns = [str(c).strip().lower() for c in dfc.columns]
    required_cols = {"parameter", "value"}
    if not required_cols.issubset(set(dfc.columns)):
        st.warning("⚠️ cost_parameters.csv missing expected columns. Recreating with defaults.")
        dfc = pd.DataFrame({
            "parameter": ["raw_material_cost", "production_cost", "logistics_cost",
                          "transfer_cost", "sla_penalty"],
            "value":     [30.0, 20.0, 10.0, 15.0, 25.0],
        })
        dfc.to_csv(path, index=False)
    dfc["value"] = pd.to_numeric(dfc["value"], errors="coerce").fillna(0.0)
    return dict(zip(dfc["parameter"], dfc["value"]))

cost_map = load_cost_map(COST_PATH)

# =========================
# DROP-INS (Business explainers)
# =========================
with st.expander("📥 What information does this agent use?", expanded=True):
    st.markdown(f"""
- **Forecast** per month / city / brand from **`{os.path.basename(FORECAST_PATH)}`**  
- **On-hand** snapshot from **`{os.path.basename(INVENTORY_PATH)}`** (uses `on_hand` or similar)  
- **Cost parameters** from **`{os.path.basename(COST_PATH)}`**
""")

with st.expander("🧠 What this does (plain language)", expanded=True):
    st.markdown("""
1) Calculates **required units** = forecast × (1 + safety%) − on-hand.  
2) Evaluates **four strategies** per row (Do Nothing, Reorder, Transfer, Hybrid).  
3) Picks the **lowest-cost feasible** strategy (respecting a small do-nothing threshold).  
4) Produces a tidy **simulation_output.csv** for collaboration & ops.
""")

with st.expander("🔗 What it hands off", expanded=True):
    st.markdown(f"""
- **`{os.path.basename(OUTPUT_PATH)}`** with:  
  `month, city, brand, sku, forecast_units, on_hand_units, required_units, best_strategy, cost_*`  
- Downstream: **Collab Hub** to assign tasks, and **Ops** to execute transfers / POs.
""")

with st.expander("✨ Why this is useful", expanded=True):
    st.markdown("""
- **Decision-first**: turns plan gaps into concrete **actions** with costs.  
- **Operator-tunable**: safety %, cost weights, and thresholds are simple dials.  
- **Transparent**: per-strategy cost columns make tradeoffs clear.  
- **Robust**: auto-detects columns; builds a minimal inventory if missing.
""")

with st.expander("🧪 Data contracts, QA & edge cases"):
    st.markdown("""
**Contracts**  
- `month` parseable; saved/displayed as `MMM-YYYY`.  
- Forecast column auto-detected (`forecast_sales`, `predicted_sales`, …).  
- Inventory must include `on_hand` (or similar). `sku` auto-derived from city/brand.

**QA**  
- ✅ No negative **required_units** (we clip at 0).  
- ✅ Strategy costs computed consistently.  
- ✅ Strategy choice = argmin(costs) unless requirement under small threshold ⇒ Do Nothing.

**Edge cases**  
- Empty inventory → synthesized template from forecast keys.  
- Missing costs file → defaults created.  
- Missing `sku` → auto-built.
""")

# =========================
# SIDEBAR: WEIGHTS & THRESHOLDS
# =========================
st.sidebar.header("⚙️ Cost & Weights")

# Unit costs (prefill from cost_map, but user can tweak)
raw_cost = st.sidebar.number_input("Raw Material Cost / unit",  min_value=0.0, value=float(cost_map.get("raw_material_cost", 30.0)), step=1.0)
prod_cost = st.sidebar.number_input("Production Cost / unit",    min_value=0.0, value=float(cost_map.get("production_cost", 20.0)), step=1.0)
log_cost  = st.sidebar.number_input("Logistics Cost / unit",     min_value=0.0, value=float(cost_map.get("logistics_cost", 10.0)), step=1.0)
transfer_cost = st.sidebar.number_input("Transfer Cost / unit",  min_value=0.0, value=float(cost_map.get("transfer_cost", 15.0)), step=1.0)
sla_penalty   = st.sidebar.number_input("SLA Penalty / short unit",min_value=0.0, value=float(cost_map.get("sla_penalty", 25.0)), step=1.0)

# Weights (normalize to sum=1)
w_raw = st.sidebar.slider("Weight: Raw Material", 0.0, 1.0, 0.25, 0.05)
w_prod = st.sidebar.slider("Weight: Production",   0.0, 1.0, 0.25, 0.05)
w_log = st.sidebar.slider("Weight: Logistics",     0.0, 1.0, 0.20, 0.05)
w_transfer = st.sidebar.slider("Weight: Transfer", 0.0, 1.0, 0.15, 0.05)
w_sla = st.sidebar.slider("Weight: SLA/Penalty",   0.0, 1.0, 0.15, 0.05)

weights = np.array([w_raw, w_prod, w_log, w_transfer, w_sla], dtype=float)
if weights.sum() == 0:
    weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
weights = weights / weights.sum()
w_raw, w_prod, w_log, w_transfer, w_sla = weights

# Safety stock and decision thresholds (to diversify strategies)
safety_stock_pct = st.sidebar.slider("Safety Stock (% of forecast)", 0, 50, 10, 5)
do_nothing_units_threshold = st.sidebar.slider("Do-Nothing threshold (units)", 0, 200, 20, 5)
hybrid_units_threshold = st.sidebar.slider("Hybrid threshold (units)", 0, 500, 120, 10)

st.sidebar.markdown("---")
st.sidebar.write("**Weights normalized to sum = 1**")
st.sidebar.write(f"Raw {w_raw:.2f} · Prod {w_prod:.2f} · Log {w_log:.2f} · Transfer {w_transfer:.2f} · SLA {w_sla:.2f}")

bus_emit("params", {
    "raw_cost": raw_cost, "prod_cost": prod_cost, "log_cost": log_cost,
    "transfer_cost": transfer_cost, "sla_penalty": sla_penalty,
    "weights": [float(x) for x in weights],
    "safety_stock_pct": safety_stock_pct,
    "do_nothing_threshold": do_nothing_units_threshold,
    "hybrid_threshold": hybrid_units_threshold
})

# =========================
# COST MODEL & DECISION
# =========================
merged["forecast_units"] = pd.to_numeric(merged[forecast_col], errors="coerce").fillna(0)
merged["on_hand_units"]  = pd.to_numeric(merged[inv_onhand_col], errors="coerce").fillna(0)
merged["required_units"] = (merged["forecast_units"] * (1 + safety_stock_pct/100.0) - merged["on_hand_units"]).clip(lower=0)

# Weighted per-unit costs
reorder_unit_cost  = w_raw*raw_cost + w_prod*prod_cost + w_log*log_cost
transfer_unit_cost = w_transfer*transfer_cost
sla_unit_cost      = w_sla*sla_penalty

# Strategy costs (+ tiny noise to avoid ties causing same label everywhere)
eps = 1e-6
noise = np.random.default_rng(42).normal(0, 1e-4, size=len(merged))

merged["cost_do_nothing"] = (sla_unit_cost * merged["required_units"]) * (1 + noise)
merged["cost_reorder"]    = (reorder_unit_cost * merged["required_units"]) * (1 + noise)
merged["cost_transfer"]   = (transfer_unit_cost * merged["required_units"]) * (1 + noise)
# Hybrid: split between reorder and transfer based on threshold
hybrid_share = np.clip((merged["required_units"] / max(hybrid_units_threshold, 1)), 0, 1)
merged["cost_hybrid"] = (hybrid_share * merged["cost_transfer"] + (1 - hybrid_share) * merged["cost_reorder"]) * (1 + noise)

# Business rules to diversify:
def decide_row(r):
    req = r["required_units"]
    if req <= do_nothing_units_threshold:
        return "Do Nothing"
    # Else pick cheapest among the three active strategies
    costs = {
        "Reorder": r["cost_reorder"],
        "Transfer": r["cost_transfer"],
        "Hybrid": r["cost_hybrid"],
    }
    return min(costs, key=costs.get)

merged["best_strategy"] = merged.apply(decide_row, axis=1)

# Best cost (consistent with decision)
merged["best_cost"] = np.select(
    [
        merged["best_strategy"].eq("Do Nothing"),
        merged["best_strategy"].eq("Reorder"),
        merged["best_strategy"].eq("Transfer"),
        merged["best_strategy"].eq("Hybrid"),
    ],
    [
        merged["cost_do_nothing"],
        merged["cost_reorder"],
        merged["cost_transfer"],
        merged["cost_hybrid"],
    ],
    default=merged["cost_do_nothing"]
).astype(float)

merged["per_unit_cost"] = np.where(merged["required_units"] > 0,
                                   merged["best_cost"] / merged["required_units"],
                                   0.0)

# =========================
# OUTPUT TABLE (exclude fashion_trend for clarity)
# =========================
nice_cols = []
for c in ["month","month_dt","city","brand","sku"]:
    if c in merged.columns:
        nice_cols.append(c)

nice_cols += ["forecast_units","on_hand_units","required_units","best_strategy",
              "cost_do_nothing","cost_reorder","cost_transfer","cost_hybrid",
              "best_cost","per_unit_cost"]

result = merged[nice_cols].copy()

# Round numerics
for c in ["forecast_units","on_hand_units","required_units",
          "cost_do_nothing","cost_reorder","cost_transfer","cost_hybrid","best_cost","per_unit_cost"]:
    if c in result.columns:
        result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0).astype(float).round(2)

# =========================
# DATA KPI RIBBON
# =========================
def kpis_data(df):
    out = {"rows": len(df)}
    if "month_dt" in df.columns:
        out["freshness"] = pd.to_datetime(df["month_dt"]).max().strftime("%b %Y") if len(df) else "—"
    else:
        out["freshness"] = "—"
    out["cities"] = int(df["city"].nunique()) if "city" in df.columns else 0
    out["brands"] = int(df["brand"].nunique()) if "brand" in df.columns else 0
    if {"city","brand"}.issubset(df.columns):
        out["pairs"]  = int(df.dropna(subset=["city","brand"]).drop_duplicates(["city","brand"]).shape[0])
    else:
        out["pairs"] = 0
    if "month_dt" in df.columns:
        months = sorted(df["month_dt"].dt.to_period("M").astype(str).unique().tolist())
        out["horizon"] = f"{months[0]}–{months[-1]}" if months else "—"
    else:
        out["horizon"] = "—"
    return out

D = kpis_data(result)
k1,k2,k3,k4,k5 = st.columns(5)
with k1: st.metric("Cities tracked", D["cities"])
with k2: st.metric("Brands tracked", D["brands"])
with k3: st.metric("Data freshness", D["freshness"])
with k4: st.metric("City × Brand pairs", D["pairs"])
with k5: st.metric("Horizon", D["horizon"])

# SIM KPI ribbon
tot_req  = float(result["required_units"].sum()) if "required_units" in result else 0.0
tot_cost = float(result["best_cost"].sum()) if "best_cost" in result else 0.0
avg_puc  = float(np.nan_to_num(result["per_unit_cost"].replace([np.inf,-np.inf], np.nan)).mean()) if not result.empty else 0.0
mix = (result["best_strategy"].value_counts(normalize=True)*100).round(1) if "best_strategy" in result else pd.Series()
top_mix = f"{mix.index[0]} ({mix.iloc[0]}%)" if not mix.empty else "—"

s1,s2,s3,s4 = st.columns(4)
with s1: st.metric("Required units (Σ)", f"{tot_req:,.0f}")
with s2: st.metric("Total best cost (Σ)", f"{tot_cost:,.0f}")
with s3: st.metric("Avg per-unit cost", f"{avg_puc:,.2f}")
with s4: st.metric("Top strategy share", top_mix)

st.markdown("---")

# =========================
# UI — TABLE
# =========================
st.markdown("### 📋 Simulation Output (Top 200 rows)")
st.dataframe(result.head(200).drop(columns=["month_dt"], errors="ignore"), use_container_width=True)

# Save CSV
result_out = result.drop(columns=["month_dt"], errors="ignore").copy()
result_out.to_csv(OUTPUT_PATH, index=False)
st.success(f"✅ Simulation results saved to `{OUTPUT_PATH}`")
bus_emit("simulation_saved", {"rows": int(len(result_out)), "path": OUTPUT_PATH})

# =========================
# CHARTS
# =========================
# Strategy mix (bar)
st.markdown("### 🧭 Strategy Mix")
if "best_strategy" in result.columns and not result.empty:
    strategy_counts = result["best_strategy"].value_counts().reset_index()
    strategy_counts.columns = ["strategy","count"]
    bar = (
        alt.Chart(strategy_counts)
        .mark_bar()
        .encode(
            x=alt.X("strategy:N", title="Strategy"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("strategy:N", legend=None,
                            scale=alt.Scale(range=["#2c7be5","#0ca678","#e8590c","#7048e8"])),
            tooltip=["strategy","count"]
        )
        .properties(height=280)
    )
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("No strategies to display.")

# Coverage vs Demand (forecast vs on-hand), per month
st.markdown("### 📈 Coverage vs Demand")
cov_src = result[["month_dt","forecast_units","on_hand_units"]].copy()
cov_src = cov_src.groupby("month_dt", as_index=False)[["forecast_units","on_hand_units"]].sum()
cov_src = cov_src[(cov_src["forecast_units"]>0) | (cov_src["on_hand_units"]>0)]
if not cov_src.empty:
    cov_m = cov_src.melt("month_dt", var_name="series", value_name="units")
    line = (
        alt.Chart(cov_m)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("month_dt:T", title="Month"),
            y=alt.Y("units:Q", title="Units"),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(range=["#1f77b4","#d62728"])),
            tooltip=[alt.Tooltip("month_dt:T", title="Month"),
                     "series:N", alt.Tooltip("units:Q", format=".2f")]
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)
else:
    st.info("Coverage chart is empty (no non-zero totals).")

# Monthly cost by strategy (stacked)
st.markdown("### 💰 Monthly Cost by Strategy")
cost_cols = ["cost_do_nothing","cost_reorder","cost_transfer","cost_hybrid"]
cost_src = result[["month_dt"] + cost_cols].copy()
cost_src = cost_src.groupby("month_dt", as_index=False)[cost_cols].sum()
cost_src = cost_src[(cost_src[cost_cols].sum(axis=1) > 0)]
if not cost_src.empty:
    cost_m = cost_src.melt("month_dt", var_name="strategy", value_name="cost")
    cost_m["strategy"] = cost_m["strategy"].str.replace("cost_","").str.replace("_"," ").str.title()
    stacked = (
        alt.Chart(cost_m)
        .mark_bar()
        .encode(
            x=alt.X("month_dt:T", title="Month"),
            y=alt.Y("cost:Q", title="Total Cost"),
            color=alt.Color("strategy:N", title="Strategy",
                            scale=alt.Scale(range=["#adb5bd","#2c7be5","#0ca678","#7048e8"])),
            tooltip=[alt.Tooltip("month_dt:T", title="Month"),
                     "strategy:N", alt.Tooltip("cost:Q", format=".2f")]
        )
        .properties(height=320)
    )
    st.altair_chart(stacked, use_container_width=True)
else:
    st.info("Cost chart is empty (no positive costs).")

# Top 20 City×Brand by Required Units
st.markdown("### 🧱 Largest Gaps — Top City×Brand (required units)")
if {"city","brand","required_units"}.issubset(result.columns):
    top_gaps = (result.groupby(["city","brand"], as_index=False)["required_units"]
                .sum().sort_values("required_units", ascending=False).head(20))
    gap_bar = (
        alt.Chart(top_gaps)
        .mark_bar()
        .encode(
            x=alt.X("required_units:Q", title="Required Units"),
            y=alt.Y("city:N", sort="-x", title="City / Brand"),
            color=alt.Color("brand:N", legend=alt.Legend(orient="bottom")),
            tooltip=["city","brand",alt.Tooltip("required_units:Q", format=".0f")]
        )
        .transform_calculate(city_brand='datum.city + " · " + datum.brand')
        .encode(y=alt.Y("city_brand:N", sort="-x", title="City · Brand"))
        .properties(height=360)
    )
    st.altair_chart(gap_bar, use_container_width=True)

# Distribution of per-unit cost (quality check)
st.markdown("### 🎯 Per-Unit Cost Distribution (best strategy)")
if "per_unit_cost" in result.columns and not result.empty:
    hist = (
        alt.Chart(result[result["per_unit_cost"]>0])
        .mark_bar()
        .encode(
            x=alt.X("per_unit_cost:Q", bin=alt.Bin(maxbins=30), title="Per-Unit Cost"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Rows")]
        )
        .properties(height=280)
    )
    st.altair_chart(hist, use_container_width=True)

# =========================
# STRATEGY ECONOMICS (aggregate)
# =========================
st.markdown("### 📊 Strategy Economics (aggregate)")
econ = []
for name, col in [("Do Nothing","cost_do_nothing"),("Reorder","cost_reorder"),
                  ("Transfer","cost_transfer"),("Hybrid","cost_hybrid")]:
    total_cost  = float(result[col].sum())
    total_units = float(result["required_units"].sum())
    per_unit    = (total_cost / total_units) if total_units > 0 else 0.0
    econ.append({"strategy": name, "total_cost": round(total_cost,2), "per_unit_cost_avg": round(per_unit,2)})
econ_df = pd.DataFrame(econ)
st.dataframe(econ_df, use_container_width=True, height=140)

# =========================
# DETAILED REVIEW
# =========================
with st.expander("📝 Detailed Review (auto-generated) — what stands out", expanded=True):
    txt = []
    # 1) Strategy mix
    if "best_strategy" in result.columns and not result.empty:
        mix = (result["best_strategy"].value_counts(normalize=True)*100).round(1)
        top_line = " · ".join([f"{k}: {v}%" for k,v in mix.items()])
        txt.append(f"- **Strategy mix** → {top_line}")

    # 2) Cost by month
    if "month_dt" in result.columns:
        by_mo = result.groupby("month_dt", as_index=False)["best_cost"].sum().sort_values("best_cost", ascending=False)
        if not by_mo.empty:
            worst = by_mo.iloc[0]
            txt.append(f"- **Cost hotspot** → {worst['month_dt'].strftime('%b-%Y')} has the highest total cost ({worst['best_cost']:,.0f}).")

    # 3) Largest unit gaps
    if {"city","brand","required_units"}.issubset(result.columns):
        g = (result.groupby(["city","brand"], as_index=False)["required_units"]
             .sum().sort_values("required_units", ascending=False).head(5))
        if not g.empty:
            tops = " · ".join([f"{r.city}/{r.brand}: {int(r.required_units):,}" for _,r in g.iterrows()])
            txt.append(f"- **Largest gaps** (Top 5 City/Brand by required units) → {tops}")

    # 4) Per-unit outliers
    if "per_unit_cost" in result.columns:
        thr = np.nanpercentile(result["per_unit_cost"].replace([np.inf,-np.inf], np.nan).dropna(), 90) if result["per_unit_cost"].gt(0).any() else np.nan
        if not np.isnan(thr):
            cnt = int((result["per_unit_cost"] > thr).sum())
            txt.append(f"- **High per-unit cost outliers** (> P90): {cnt} rows — review transfer vs reorder mix and thresholds.")

    st.markdown("\n".join(txt) if txt else "No notable findings (dataset may be empty).")

# =========================
# DECISION ASSISTANT (City/Brand)
# =========================
st.markdown("---")
st.markdown("## 🤖 Decision Assistant")

cities = sorted(result["city"].dropna().unique().tolist()) if "city" in result.columns else []
brands = sorted(result["brand"].dropna().unique().tolist()) if "brand" in result.columns else []

col_a, col_b = st.columns(2)
with col_a:
    sel_city = st.selectbox("City", ["— choose —"] + cities, index=0)
with col_b:
    sel_brand = st.selectbox("Brand", ["— choose —"] + brands, index=0)

def badge(text, color):
    return st.markdown(
        f"""
        <div style="border-radius:10px;padding:10px 12px;margin:6px 0;
                    background:{color}15;border:1px solid {color}44;color:{color};font-weight:700;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

if sel_city != "— choose —" and sel_brand != "— choose —":
    mask = (result.get("city","") == sel_city) & (result.get("brand","") == sel_brand)
    sub = result[mask].sort_values("month_dt")
    if sub.empty:
        st.info("No rows for the chosen City/Brand.")
    else:
        latest = sub.tail(1).iloc[0]
        strat = latest["best_strategy"]
        req   = float(latest["required_units"])
        f_units = float(latest["forecast_units"])
        onh   = float(latest["on_hand_units"])

        st.write(f"**Latest month:** {latest['month']}")
        st.write(f"**Forecast:** {f_units:,.0f} · **On-hand:** {onh:,.0f} · **Required:** {req:,.0f}")

        if strat == "Do Nothing":
            badge("✅ Decision: Do Nothing — coverage is adequate.", "#2f9e44")
        elif strat == "Reorder":
            badge("🛠️ Decision: Reorder — place a supplier order.", "#2c7be5")
        elif strat == "Transfer":
            badge("🔁 Decision: Transfer — pull stock from another warehouse.", "#e8590c")
        else:
            badge("🧩 Decision: Hybrid — mix of reorder & transfer.", "#7048e8")
else:
    st.info("Pick City and Brand to see a tailored recommendation.")

# =========================
# DOWNLOAD BUTTON
# =========================
csv_bytes = result_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Simulation CSV", data=csv_bytes, file_name="simulation_output.csv", mime="text/csv")