# pages/4_inventory_agent.py
import os
import warnings
warnings.filterwarnings("ignore")

# ✅ imports must come before any st.* calls
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import datetime as _dt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Inventory Agent", layout="wide")
st.title("📦 Inventory Planning Agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FC_PATH   = os.path.join(DATA_DIR, "forecast_output.csv")              # input (from forecasting agent)
SNAP_PATH = os.path.join(DATA_DIR, "onhand_inventory_snapshot.csv")    # input/output (seed + reuse)
PLAN_PATH = os.path.join(DATA_DIR, "inventory_plan.csv")               # output (for sim & hub)
BUS_PATH  = os.path.join(DATA_DIR, "memory_bus.jsonl")                 # event log (memory bus)

os.makedirs(DATA_DIR, exist_ok=True)
alt.data_transformers.disable_max_rows()

# ---------- UI polish (lightweight CSS) ----------
st.markdown("""
<style>
section.main > div { padding-top: 0.4rem; }
.block-container { padding-top: 0.8rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stMetricLabel"] { font-weight: 600; color: #bbb; }
details > summary { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------- Reusable: Inventory drop-ins ----------
def render_inventory_dropins(fc_path, snap_path, plan_path, bus_path):
    with st.expander("📬 What information does this use?", expanded=True):
        st.markdown(f"""
**Inputs**
- **Forecast** by month / city / brand → **`{os.path.basename(fc_path)}`**  
- **Inventory snapshot** (on-hand & inbound pipeline) → **`{os.path.basename(snap_path)}`**

**Join keys**
- `month` (monthly), `city`, `brand` → optional derived `sku`  
- If `forecast_sales` is named differently, we auto-detect (`predicted_sales`, `sales_forecast`, `forecast`).

**Assumptions**
- `pipeline` arrives *now* (no delay). Lead-time UI reserved for future shift logic.
- Missing numerics are coerced to 0. Missing texts are stripped/normalized.
""")

    with st.expander("🧠 What this page does (plain language)"):
        st.markdown("""
It converts demand forecasts into **coverage** and **reorder quantities** using a simple, controllable policy:
1) Compute **safety stock** = *service level % × forecast*.  
2) Compare **available** (*on_hand + pipeline*) vs **target** (*forecast + safety*).  
3) If available < target ⇒ **reorder_qty** = shortfall to reach target.  
4) Save a clean **inventory_plan.csv** for Simulation & Ops.
""")

    with st.expander("🔗 What it hands off to other parts"):
        st.markdown(f"""
**Artifact:** **`{os.path.basename(plan_path)}`** with these columns:
- `month, city, brand, sku`  
- `forecast_sales, on_hand_start, on_hand, pipeline`  
- `safety_stock, demand_plan, alloc_to_demand, shortfall, reorder_qty, on_hand_end`  
- `service_level_pct`

**Memory bus:** appends JSON lines to **`{os.path.basename(bus_path)}`** for page open, parameter & filter changes, and saves.
""")

    with st.expander("✨ Why this adds practical value"):
        st.markdown("""
- **Action-ready:** gives **reorder_qty** per city × brand × month.
- **Operator-tunable:** service level is a simple dial (no retraining).
- **Traceable:** settings & saves are logged to a **memory bus**.
- **Robust:** auto-detects the forecast column name; creates SKUs if missing.
""")

    with st.expander("🧭 Design notes: how this behaves like a proactive planning assistant"):
        st.markdown("""
- **Decision-first**: turns model outputs into **purchase/transfer signals**.  
- **Composable**: clean CSV contracts for Simulation → Collaboration.  
- **Transparent**: minimal, readable policy; numbers add up row-by-row.  
- **Local context**: operates at `city × brand` granularity with roll-ups for monitoring.
""")

    with st.expander("🧪 Data contracts, QA & edge cases"):
        st.markdown("""
**Contracts**
- `month`: parseable to monthly period; saved as `YYYY-MM`.
- `forecast_sales`: numeric (auto-rename supported).
- `on_hand`, `pipeline`: numeric (missing ⇒ 0).

**QA checklist**
- ✅ Latest month in forecast shows in **Data freshness**.  
- ✅ No negative values in `reorder_qty`, `shortfall`, `on_hand_end`.  
- ✅ Sum(`alloc_to_demand`) ≤ Sum(`forecast_sales`).  
- ✅ Sum across groups equals overview totals.

**Edge cases handled**
- Empty forecast → graceful stop with message.  
- Unknown forecast column name → auto-detection (or explicit error).  
- Snapshot missing → deterministic synthetic snapshot seeded from forecast keys.
""")

# ---------- Reusable: KPI ribbon ----------
def render_inventory_kpis(plan: pd.DataFrame, fc: pd.DataFrame, inv: pd.DataFrame):
    def _u(series):
        return int(series.nunique()) if series is not None else 0

    fresh = "—"
    try:
        m = pd.to_datetime(fc["month"], errors="coerce").max()
        fresh = m.strftime("%b %Y") if pd.notna(m) else "—"
    except Exception:
        pass

    cities  = _u(plan["city"])  if "city"  in plan.columns else 0
    brands  = _u(plan["brand"]) if "brand" in plan.columns else 0
    pairs   = _u(plan[["city","brand"]].dropna().astype(str).agg("|".join, axis=1)) if {"city","brand"}.issubset(plan.columns) else 0
    skus    = _u(plan["sku"])   if "sku"   in plan.columns else 0
    months  = sorted(pd.to_datetime(plan["month"], errors="coerce").dt.to_period("M").astype(str).dropna().unique().tolist())
    mspan   = f"{months[0]}–{months[-1]}" if months else "—"

    fsum = float(plan.get("forecast_sales", pd.Series(dtype=float)).sum())
    asg  = float(plan.get("alloc_to_demand", pd.Series(dtype=float)).sum())
    short= float(plan.get("shortfall", pd.Series(dtype=float)).sum())
    req  = float(plan.get("reorder_qty", pd.Series(dtype=float)).sum())
    slvl = float(plan.get("service_level_pct", pd.Series(dtype=float)).mean()) if "service_level_pct" in plan.columns else np.nan
    fill = (asg / fsum) if fsum > 0 else np.nan

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Cities tracked", cities)
    with c2: st.metric("Brands tracked", brands)
    with c3: st.metric("Data freshness", fresh)
    with c4: st.metric("City × Brand pairs", pairs)
    with c5: st.metric("SKUs planned", skus)

    c6,c7,c8,c9,c10 = st.columns(5)
    with c6: st.metric("Months planned", mspan)
    with c7: st.metric("Total forecast (units)", f"{fsum:,.0f}")
    with c8: st.metric("Fill rate", "—" if np.isnan(fill) else f"{fill*100:,.1f}%")
    with c9: st.metric("Shortfall (units)", f"{short:,.0f}")
    with c10: st.metric("Reorder qty (units)", f"{req:,.0f}")

    if not np.isnan(slvl):
        st.caption(f"Average service level across rows: **{slvl*100:.1f}%**")

# =============================
# MEMORY BUS (JSONL)
# =============================
def bus_emit(event_type: str, payload: dict | None = None):
    """Append one JSON line to the local memory bus."""
    try:
        rec = {
            "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": event_type,
            "payload": payload or {},
        }
        with open(BUS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        st.toast(f"Bus write failed: {e}", icon="⚠️")

def _track_param(name: str, value):
    """Log param changes once per change using session_state."""
    key = f"_last_{name}"
    old = st.session_state.get(key, None)
    if old != value and old is not None:
        bus_emit("params_changed", {"page":"inventory","param":name,"old":old,"new":value})
    st.session_state[key] = value

# =============================
# LOADERS & UTILITIES
# =============================
def build_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Stable SKU from available categoricals (brand, city)."""
    if "sku" in df.columns and df["sku"].notna().any():
        return df
    cat_cols = [c for c in ["brand", "city"] if c in df.columns]
    if not cat_cols:
        df["sku"] = "sku_generic"
        return df

    def _mk(row):
        parts = []
        for c in cat_cols:
            v = str(row.get(c, "")).strip().lower().replace(" ", "")
            parts.append(v if v else "x")
        return "sku_" + "_".join(parts)

    df["sku"] = df.apply(_mk, axis=1)
    return df

@st.cache_data(show_spinner=False)
def load_forecast(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # month -> month-begin Timestamp; keep only valid
    m1 = pd.to_datetime(df.get("month"), format="%Y-%m", errors="ignore")
    m2 = pd.to_datetime(df.get("month"), errors="coerce")
    df["month"] = (m1 if isinstance(m1, pd.Series) else m2).fillna(m2).dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["month"]).copy()

    # tidy categoricals
    for c in ["city", "brand"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # detect forecast column
    candidates = ["forecast_sales", "predicted_sales", "sales_forecast", "forecast"]
    fcol = next((c for c in candidates if c in df.columns), None)
    if fcol is None:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["month"]]
        if len(num_cols) == 1:
            fcol = num_cols[0]
        else:
            raise ValueError("No forecast column found in forecast_output.csv")

    if fcol != "forecast_sales":
        df = df.rename(columns={fcol: "forecast_sales"})
    df["forecast_sales"] = pd.to_numeric(df["forecast_sales"], errors="coerce").fillna(0.0)
    return df

def _build_snapshot_from_forecast(fc: pd.DataFrame) -> pd.DataFrame:
    """If no snapshot exists, synthesize a deterministic one from fc keys."""
    if fc.empty:
        return pd.DataFrame(columns=["city","brand","on_hand","pipeline"])
    keys = [k for k in ["city","brand"] if k in fc.columns]
    latest_keys = fc.drop_duplicates(keys)[keys].copy() if keys else pd.DataFrame(index=[0])
    rng = np.random.default_rng(42)
    n = len(latest_keys) if not latest_keys.empty else 1
    snap = latest_keys.copy() if not latest_keys.empty else pd.DataFrame(index=range(n))
    snap["on_hand"]  = rng.integers(50, 300, size=len(snap))
    snap["pipeline"] = rng.integers(0, 120, size=len(snap))
    return snap

@st.cache_data(show_spinner=False)
def load_onhand_snapshot(path: str, fc: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(path):
        inv = pd.read_csv(path)
        for c in ["city","brand"]:
            if c in inv.columns:
                inv[c] = inv[c].astype(str).str.strip()
        for c in ["on_hand","pipeline"]:
            if c in inv.columns:
                inv[c] = pd.to_numeric(inv[c], errors="coerce").fillna(0.0)
        if "on_hand" not in inv.columns:  inv["on_hand"] = 0.0
        if "pipeline" not in inv.columns: inv["pipeline"] = 0.0
        return inv
    inv = _build_snapshot_from_forecast(fc)
    inv = build_sku(inv)
    inv.to_csv(path, index=False)
    return inv

# =============================
# LOAD DATA + PAGE-OPEN BUS EVENT
# =============================
try:
    fc = load_forecast(FC_PATH)
except Exception as e:
    st.error(f"❌ Problem reading `{FC_PATH}`: {e}")
    bus_emit("error", {"page":"inventory","where":"load_forecast","msg":str(e)})
    st.stop()

inv = load_onhand_snapshot(SNAP_PATH, fc)

bus_emit("page_open", {
    "page":"inventory",
    "fc_rows": int(len(fc)),
    "inv_rows": int(len(inv)),
})

# =============================
# FILTERS + SETTINGS (DROP-INS STYLE)
# =============================
st.sidebar.header("🔎 Filters")
city_opts  = sorted({*fc.get("city", pd.Series(dtype=str)).dropna().unique(),
                     *inv.get("city", pd.Series(dtype=str)).dropna().unique()})
brand_opts = sorted({*fc.get("brand", pd.Series(dtype=str)).dropna().unique(),
                     *inv.get("brand", pd.Series(dtype=str)).dropna().unique()})

sel_city  = st.sidebar.multiselect("City", city_opts, default=city_opts[:1] if city_opts else [])
sel_brand = st.sidebar.multiselect("Brand", brand_opts, default=brand_opts[:1] if brand_opts else [])

st.sidebar.header("⚙️ Planning Settings")
service_level = st.sidebar.slider("Service level (safety % of forecast)", 0.0, 0.5, 0.15, 0.05)
lead_time_months = st.sidebar.slider("Lead time (months)", 0, 3, 1, 1)  # reserved for future logic
safety_pct = service_level

# track changes
def _track_and_emit():
    _track_param("service_level", service_level)
    _track_param("lead_time_months", lead_time_months)
    _track_param("filter_city", tuple(sel_city))
    _track_param("filter_brand", tuple(sel_brand))
_track_and_emit()

# render the “drop-ins” help blocks
render_inventory_dropins(FC_PATH, SNAP_PATH, PLAN_PATH, BUS_PATH)

# =============================
# CORE ALLOCATION LOGIC
# =============================
def allocate_inventory_fixed(group_df, snapshot_row, safety_pct: float):
    """Row-wise simple policy with safety stock; pipeline arrives immediately."""
    g = group_df.copy().sort_values("month")
    g["forecast_sales"] = pd.to_numeric(g["forecast_sales"], errors="coerce").fillna(0.0)

    on_hand  = float(snapshot_row.get("on_hand", 0.0)) if snapshot_row else 0.0
    pipeline = float(snapshot_row.get("pipeline", 0.0)) if snapshot_row else 0.0

    available = on_hand + pipeline
    records = []

    for _, row in g.iterrows():
        demand = float(row["forecast_sales"])
        safety = demand * safety_pct
        target = demand + safety

        reorder_needed   = max(0.0, target - available)
        alloc_to_demand  = min(available, demand)
        shortfall        = max(0.0, demand - available)
        end_balance      = max(0.0, available - demand)   # safety is not “consumed”

        rec = row.to_dict()
        rec.update({
            "on_hand_start": available,
            "on_hand": available,
            "pipeline": pipeline,
            "safety_stock": safety,
            "demand_plan": demand,
            "alloc_to_demand": alloc_to_demand,
            "shortfall": shortfall,
            "reorder_qty": reorder_needed,
            "on_hand_end": end_balance,
            "service_level_pct": safety_pct,
        })
        records.append(rec)

        # advance one step (no new pipeline modeled here)
        available = end_balance

    return pd.DataFrame(records)

# filter the forecast by user selection (view only; not altering saved plan)
fc_view = fc.copy()
if sel_city:
    fc_view = fc_view[fc_view["city"].isin(sel_city)]
if sel_brand:
    fc_view = fc_view[fc_view["brand"].isin(sel_brand)]

# build full grid over months × dims present in fc_view/inv
dims = [c for c in ["city","brand"] if c in set(fc.columns).union(inv.columns)]
dim_values = {}
for d in dims:
    vals_fc  = fc_view[d].dropna().unique().tolist() if d in fc_view.columns else []
    vals_inv = inv[d].dropna().unique().tolist() if d in inv.columns else []
    dim_values[d] = sorted(list({*vals_fc, *vals_inv}))

fc_months = pd.to_datetime(fc_view["month"]).sort_values().unique().tolist()
if not fc_months:
    st.warning("No rows match current filters. Try widening your selection.")
    st.stop()

grid_frames = [pd.Series(fc_months, name="month")]
for d in dims:
    grid_frames.append(pd.Series(dim_values[d], name=d))
grid = grid_frames[0].to_frame()
for s in grid_frames[1:]:
    grid["_k"] = 1
    s = s.to_frame(); s["_k"] = 1
    grid = grid.merge(s, on="_k").drop(columns="_k")

merge_keys = ["month"] + dims
fc_small = fc_view[merge_keys + ["forecast_sales"]].copy() if set(merge_keys).issubset(fc_view.columns) else pd.DataFrame(columns=merge_keys + ["forecast_sales"])
fc_complete = grid.merge(fc_small, on=merge_keys, how="left")
fc_complete["forecast_sales"] = pd.to_numeric(fc_complete["forecast_sales"], errors="coerce").fillna(0.0)
fc_complete = fc_complete.sort_values(merge_keys).reset_index(drop=True)

# allocate per group
inv_idx = inv.set_index(dims) if dims else None
plans = []
if dims:
    for key, g in fc_complete.groupby(dims, dropna=False):
        snap = None
        if isinstance(key, tuple):
            if key in inv_idx.index:
                snap = inv_idx.loc[key].to_dict()
        else:
            if (key,) in inv_idx.index:
                snap = inv_idx.loc[(key,)].to_dict()
        plan_g = allocate_inventory_fixed(g, snap, safety_pct)
        # reattach group keys explicitly (for safety)
        if isinstance(key, tuple):
            for i, d in enumerate(dims):
                plan_g[d] = key[i]
        else:
            plan_g[dims[0]] = key
        plans.append(plan_g)
else:
    snap = inv.iloc[0].to_dict() if not inv.empty else {"on_hand":0.0,"pipeline":0.0}
    plans.append(allocate_inventory_fixed(fc_complete.copy(), snap, safety_pct))

plan = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()
plan = build_sku(plan)

# =============================
# KPI RIBBON + TABLE + CHARTS
# =============================
render_inventory_kpis(plan, fc, inv)

st.markdown("### 📋 Consolidated Plan (first 50 rows)")
show_cols = (["month"]
             + [c for c in ["city","brand","sku"] if c in plan.columns]
             + ["forecast_sales","on_hand_start","on_hand","pipeline",
                "safety_stock","demand_plan","alloc_to_demand",
                "shortfall","reorder_qty","on_hand_end"])
st.dataframe(plan[show_cols].head(50), use_container_width=True)

# Coverage vs Demand (summed)
sum_plan = plan.groupby("month", as_index=False)[["forecast_sales","alloc_to_demand","shortfall"]].sum()
sum_plan["month_str"] = pd.to_datetime(sum_plan["month"]).dt.strftime("%b-%Y")

st.markdown("### 📊 Coverage vs Demand (All Locations / Brands)")
bar_df = sum_plan.melt(id_vars=["month","month_str"],
                       value_vars=["forecast_sales","alloc_to_demand","shortfall"],
                       var_name="metric", value_name="value")
palette = ["#1f77b4", "#2ca02c", "#d62728"]
bar = (
    alt.Chart(bar_df)
    .mark_bar()
    .encode(
        x=alt.X("month_str:N", title="Month", sort=list(sum_plan["month_str"].unique())),
        y=alt.Y("value:Q", title="Units"),
        color=alt.Color("metric:N",
                        legend=alt.Legend(orient="bottom", title=None, labelFontSize=12),
                        scale=alt.Scale(range=palette)),
        tooltip=["month_str","metric",alt.Tooltip("value:Q", format=".2f")]
    )
    .properties(height=320)
)
st.altair_chart(bar, use_container_width=True)

if dims:
    st.markdown("### 🧭 Demand by Group (stacked)")
    stack_dim = dims[0]
    stacked = (
        alt.Chart(plan.assign(month_str=pd.to_datetime(plan["month"]).dt.strftime("%b-%Y")))
        .mark_bar()
        .encode(
            x=alt.X("month_str:N", title="Month"),
            y=alt.Y("forecast_sales:Q", title="Forecast"),
            color=alt.Color(f"{stack_dim}:N", legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(scheme="tableau10")),
            tooltip=[f"{stack_dim}:N", alt.Tooltip("forecast_sales:Q", format=".2f"), "month_str:N"]
        )
        .properties(height=300)
    )
    st.altair_chart(stacked, use_container_width=True)

# =============================
# SAVE ARTIFACTS + BUS EVENTS
# =============================
plan_to_save = plan.copy()
plan_to_save["month"] = pd.to_datetime(plan_to_save["month"]).dt.strftime("%Y-%m")
plan_to_save.to_csv(PLAN_PATH, index=False)

snap_clean = inv.copy()
for c in ["on_hand","pipeline"]:
    if c in snap_clean.columns:
        snap_clean[c] = pd.to_numeric(snap_clean[c], errors="coerce").fillna(0.0)
snap_clean = build_sku(snap_clean)
snap_clean.to_csv(SNAP_PATH, index=False)

st.success(f"💾 Plan saved to: `{PLAN_PATH}`")
st.download_button(
    "⬇️ Download inventory_plan.csv",
    data=plan_to_save.to_csv(index=False).encode("utf-8"),
    file_name="inventory_plan.csv",
    mime="text/csv"
)

bus_emit("plan_saved", {
    "page":"inventory",
    "rows": int(len(plan_to_save)),
    "path": PLAN_PATH,
    "service_level": service_level,
    "lead_time_months": lead_time_months,
})

st.caption(f"On-hand snapshot present at `{SNAP_PATH}` (used to seed starting balance).")