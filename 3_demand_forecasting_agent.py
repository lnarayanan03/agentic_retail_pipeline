# pages/3_demand_forecasting_agent.py
# ---------------------------------------------------------
# Demand Forecasting Agent — minimalist UI + detailed drop-ins
# - Business-first expanders (mirror your sensing page)
# - Dataset KPI ribbon (Cities, Brands, Freshness, Pairs)
# - Anti-overfit controls, model choice & selection metric
# - Models: ElasticNet, GBDT, (optional) HistGBDT, and a Blend
# - Diagnostics, EDA, Feature importance, and Forecast tabs
# - Saves forecast_output.csv and a tiny model card JSON
# - Logs to data/memory_bus.jsonl (non-blocking)
# ---------------------------------------------------------
import os, json, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional (falls back gracefully if missing)
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HGBR_OK = True
except Exception:
    HGBR_OK = False

# ===================== Page / Paths =====================
st.set_page_config(page_title="Demand Forecasting Agent", layout="wide")
st.title("📈 Demand Forecasting Agent")
alt.data_transformers.disable_max_rows()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
DATA_PATH  = os.path.join(DATA_DIR, "ForecastingDataset_Final_Cleaned.csv")
OUTPUT_FC  = os.path.join(DATA_DIR, "forecast_output.csv")
MODEL_CARD = os.path.join(DATA_DIR, "model_card_forecasting.json")
BUS_PATH   = os.path.join(DATA_DIR, "memory_bus.jsonl")
os.makedirs(DATA_DIR, exist_ok=True)

# ===================== UI polish (CSS) =====================
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

# ===================== Memory bus =====================
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
        pass  # never block UI

bus_emit("page_opened", {"page": "forecasting"})

# ===================== Helpers =====================
def safe_ohe(drop="first"):
    """OneHotEncoder with backward-compatible kwargs."""
    try:
        return OneHotEncoder(drop=drop, handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop=drop, handle_unknown="ignore", sparse=False)

def yesno_to01(series: pd.Series | None, n: int) -> pd.Series:
    if series is None:
        return pd.Series([0]*n, dtype=int)
    return pd.to_numeric(series.map({"yes":1,"no":0,"Yes":1,"No":0}), errors="coerce").fillna(0).astype(int)

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred))/2.0, 1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def coverage_kpis(df: pd.DataFrame) -> dict:
    out = {"rows": len(df)}
    m = pd.to_datetime(df["month"], errors="coerce")
    out["freshness"] = m.max().strftime("%b %Y") if m.notna().any() else "—"
    out["cities"]    = int(df["city"].nunique()) if "city" in df.columns else 0
    out["brands"]    = int(df["brand"].nunique()) if "brand" in df.columns else 0
    if {"city","brand"}.issubset(df.columns):
        out["pairs"] = int(df.dropna(subset=["city","brand"]).drop_duplicates(["city","brand"]).shape[0])
    else:
        out["pairs"] = 0
    return out

def get_feature_names(ct: ColumnTransformer) -> list[str]:
    """Best-effort to recover transformed feature names."""
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                if isinstance(trans, OneHotEncoder):
                    arr = trans.get_feature_names_out(cols)
                else:
                    arr = trans.get_feature_names_out()
                names += list(arr)
            except Exception:
                names += list(cols)
        else:
            names += list(cols)
    return names

# ===================== Sidebar Controls =====================
st.sidebar.header("⚙️ Training Controls")
anti_overfit  = st.sidebar.toggle("Anti-overfit mode", value=True)
metric_choice = st.sidebar.selectbox("Model selection metric", ["MAE (robust)", "MSE (sensitive)"], index=0)
scoring       = "neg_mean_absolute_error" if "MAE" in metric_choice else "neg_mean_squared_error"
model_choice  = st.sidebar.selectbox(
    "Use model",
    ["Auto (best)", "ElasticNet", "GBDT"] + (["HistGBDT"] if HGBR_OK else []) + ["Blend (EN+GBDT)"],
    index=0
)
horizon       = st.sidebar.slider("Forecast horizon (months)", 1, 6, 3, 1)
bus_emit("params", {"anti_overfit": anti_overfit, "metric": metric_choice, "model_choice": model_choice, "horizon": horizon})

# ===================== Load data =====================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # month parsing (YYYY-MM preferred, but robust)
    m1 = pd.to_datetime(df.get("month"), format="%Y-%m", errors="coerce")
    m2 = pd.to_datetime(df.get("month"), errors="coerce")
    df["month"] = m1.fillna(m2)
    df = df.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

    # tidy categoricals
    for c in ["city","brand","fashion_trend"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # binary flags
    df["projected_spike"] = yesno_to01(df.get("projected_spike"), len(df))
    df["holiday_flag"]    = yesno_to01(df.get("holiday_flag"), len(df))

    # target
    df["sales"] = pd.to_numeric(df.get("sales"), errors="coerce")
    df = df.dropna(subset=["sales"])
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Could not find dataset at: `{DATA_PATH}`")
    st.stop()

df_raw = load_data(DATA_PATH)
st.caption(f"Loaded **{len(df_raw)} rows × {df_raw.shape[1]} cols** from `{os.path.basename(DATA_PATH)}`")

# ===================== Drop-ins (Top) =====================
with st.expander("📥 What information does this listen to?", expanded=True):
    st.markdown("""
- **Sales history** (month, city, brand)  
- **Engineered momentum**: lags (t-1,t-2), rolling(3), Δ vs last month  
- **Context flags**: **Projected Spike** (from sensing) & **Holiday**  
- Optional **trend tags** (e.g., `fashion_trend`) are one-hot encoded
""")

with st.expander("🧠 What it does (plain language)", expanded=True):
    st.markdown(f"""
1) Cleans & engineers features.  
2) Trains **ElasticNet**, **GBDT**{", **HistGBDT**" if HGBR_OK else ""} with time-aware CV.  
3) Picks the **best** model by **{metric_choice.split()[0]}** (or use your manual choice).  
4) Publishes a **{horizon}-month** forecast per city×brand to a tidy CSV.
""")

with st.expander("🔗 What it hands off to other parts", expanded=True):
    st.markdown(f"""
- **`{os.path.basename(OUTPUT_FC)}`** → columns: `month, city, brand, forecast_sales`  
- Used by **Inventory**, **Simulation**, and **Collab Hub**.
""")

with st.expander("✨ Why this adds practical value", expanded=True):
    st.markdown("""
- **Action-first**: produces the exact table downstream agents need.  
- **Policy dials**: anti-overfit, metric, model, horizon — no code changes required.  
- **Explainable**: diagnostics, correlations, and feature importance for trust.  
- **Resilient**: works even if some optional fields are missing.
""")

with st.expander("🧪 Data contracts, QA & edge cases", expanded=False):
    st.markdown("""
**Contracts**
- `month` parseable to month; output saved as `YYYY-MM`.  
- `sales` numeric target.  
- Flags (`projected_spike`, `holiday_flag`) → 0/1 automatically.  

**QA checklist**
- ✅ Freshness reflects last row month.  
- ✅ No NaNs in output; city/brand text is trimmed/lowercased.  
- ✅ Forecast horizon ≥ 1; template rows exist for the last month.  

**Edge cases handled**
- Missing flags → treated as 0.  
- No brand/city → still forecasts aggregate series.  
- HistGBDT unavailable → automatically skipped.
""")

# ===================== Dataset KPI ribbon =====================
cov = coverage_kpis(df_raw)
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Cities tracked", cov["cities"])
with k2: st.metric("Brands tracked", cov["brands"])
with k3: st.metric("Data freshness", cov["freshness"])
with k4: st.metric("City × Brand pairs", cov["pairs"])
st.markdown("---")

# ===================== Feature engineering =====================
df = df_raw.copy()
df["month_num"] = df["month"].dt.month
df["year_num"]  = df["month"].dt.year

def add_lags_rolling(g):
    g = g.sort_values("month").copy()
    med = g["sales"].median()
    g["sales_lag_1"]     = g["sales"].shift(1).fillna(med)
    g["sales_lag_2"]     = g["sales"].shift(2).fillna(med)
    g["rolling_sales_3"] = g["sales"].shift(1).rolling(3).mean().fillna(med)
    g["sales_change_1"]  = g["sales"] - g["sales_lag_1"]
    return g

if {"city","brand"}.issubset(df.columns):
    df = df.groupby(["city","brand"], group_keys=False).apply(add_lags_rolling)
else:
    df = add_lags_rolling(df)

# numeric NaNs → medians
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# Feature selection (keep weak>0.05 + required drivers)
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique()>1]
corr_s = df[numeric_cols].corr()["sales"].drop(labels=["sales"], errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
selected_numeric = [c for c,v in corr_s.items() if abs(v) > 0.05]
must_keep = [c for c in ["projected_spike","holiday_flag","month_num","year_num",
                         "sales_lag_1","sales_lag_2","rolling_sales_3","sales_change_1"]
             if c in df.columns]
numeric_features = sorted(list(set(selected_numeric + must_keep)))
categorical_features = [c for c in ["city","brand","fashion_trend"] if c in df.columns]

st.markdown(
    "**Numeric features:** " + (", ".join(numeric_features) if numeric_features else "—") +
    " &nbsp;·&nbsp; **Categoricals:** " + (", ".join(categorical_features) if categorical_features else "—")
)

# ===================== Split & Preprocess =====================
cut = int(len(df) * 0.8)
train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
if train_df.empty or test_df.empty:
    st.error("Split produced empty train/test — please check your dataset.")
    st.stop()

y_train, y_test = train_df["sales"], test_df["sales"]
X_train = pd.concat([train_df[numeric_features], train_df[categorical_features]], axis=1) if categorical_features else train_df[numeric_features].copy()
X_test  = pd.concat([test_df[numeric_features],  test_df[categorical_features]],  axis=1) if categorical_features else test_df[numeric_features].copy()

transformers = [("num", StandardScaler(), numeric_features)]
if categorical_features:
    transformers.append(("cat", safe_ohe(drop="first"), categorical_features))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# ===================== Models & grids =====================
pipe_en = Pipeline([("prep", preprocessor), ("reg", ElasticNet(max_iter=20000, random_state=42))])
param_en = {
    "reg__alpha": ([0.1, 0.5, 1, 2, 5, 10] if anti_overfit else [0.05, 0.1, 0.5, 1, 5]),
    "reg__l1_ratio": ([0.2, 0.5, 0.8] if anti_overfit else [0.1, 0.5, 0.9]),
}

pipe_g = Pipeline([("prep", preprocessor), ("reg", GradientBoostingRegressor(random_state=42))])
param_g = {
    "reg__n_estimators": ([400, 800, 1200] if anti_overfit else [150, 300]),
    "reg__learning_rate": ([0.03, 0.05] if anti_overfit else [0.05, 0.1]),
    "reg__max_depth": [2, 3],
    "reg__min_samples_leaf": ([20, 50] if anti_overfit else [1, 5]),
    "reg__subsample": ([0.5, 0.7, 0.9] if anti_overfit else [0.9, 1.0]),
    "reg__max_features": (["sqrt","log2",0.5] if anti_overfit else ["sqrt", None]),
    "reg__loss": (["huber","squared_error"] if anti_overfit else ["squared_error"]),
    "reg__validation_fraction": [0.2],
    "reg__n_iter_no_change": ([20] if anti_overfit else [None]),
    "reg__tol": [1e-4],
}

if HGBR_OK:
    pipe_h = Pipeline([("prep", preprocessor), ("reg", HistGradientBoostingRegressor(random_state=42, early_stopping=True))])
    param_h = {
        "reg__learning_rate": ([0.03, 0.05] if anti_overfit else [0.05, 0.1]),
        "reg__max_depth": [None, 6],
        "reg__max_leaf_nodes": ([15, 31, 63] if anti_overfit else [31, 63]),
        "reg__min_samples_leaf": ([20, 50] if anti_overfit else [10]),
        "reg__l2_regularization": ([0.0, 0.1, 0.5] if anti_overfit else [0.0, 0.1]),
        "reg__loss": (["least_squares", "poisson"] if anti_overfit else ["least_squares"]),
    }

# time-aware CV
try:
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
except TypeError:
    tscv = TimeSeriesSplit(n_splits=5)

# ===================== Train =====================
with st.spinner("Training models…"):
    grid_en = GridSearchCV(pipe_en, param_en, cv=tscv, scoring=scoring, n_jobs=-1)
    grid_en.fit(X_train, y_train)

    grid_g  = GridSearchCV(pipe_g, param_g, cv=tscv, scoring=scoring, n_jobs=-1)
    grid_g.fit(X_train, y_train)

    if HGBR_OK:
        grid_h  = GridSearchCV(pipe_h, param_h, cv=tscv, scoring=scoring, n_jobs=-1)
        grid_h.fit(X_train, y_train)

# ===================== Evaluate =====================
def evaluate(name, est, X, y):
    yp = est.predict(X)
    return {
        "name": name,
        "rmse": float(np.sqrt(mean_squared_error(y, yp))),
        "mae":  float(mean_absolute_error(y, yp)),
        "mape": mape(y, yp),
        "smape": smape(y, yp),
        "r2":   float(r2_score(y, yp)),
        "yhat": yp,
    }

ev_en = evaluate("ElasticNet", grid_en.best_estimator_, X_test, y_test)
ev_g  = evaluate("GBDT",       grid_g.best_estimator_,  X_test, y_test)
evs   = [ev_en, ev_g]

if HGBR_OK:
    ev_h = evaluate("HistGBDT", grid_h.best_estimator_, X_test, y_test)
    evs.append(ev_h)

# simple blend EN+GBDT
blend_w_en, blend_w_g = 0.3, 0.7
yhat_blend = blend_w_en*ev_en["yhat"] + blend_w_g*ev_g["yhat"]
ev_b = {
    "name":"Blend(EN+GBDT)",
    "rmse": float(np.sqrt(mean_squared_error(y_test, yhat_blend))),
    "mae":  float(mean_absolute_error(y_test, yhat_blend)),
    "mape": mape(y_test, yhat_blend),
    "smape": smape(y_test, yhat_blend),
    "r2":   float(r2_score(y_test, yhat_blend)),
}
evs.append(ev_b)

def pick_best(metric):
    return min(evs, key=lambda d: d["mae" if "MAE" in metric else "rmse"])

best_auto = pick_best(metric_choice)

if model_choice == "Auto (best)":
    chosen = best_auto["name"]
else:
    chosen = model_choice

# ===================== KPI ribbon (model) =====================
perf_map = {e["name"]: e for e in evs}
perf = perf_map[chosen]
c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.metric("Selected model", chosen)
with c2: st.metric("RMSE (test)", f"{perf['rmse']:,.2f}")
with c3: st.metric("MAE (test)", f"{perf['mae']:,.2f}")
with c4: st.metric("MAPE (test)", f"{perf['mape']:.1f}%")
with c5: st.metric("R² (test)", f"{perf['r2']:.3f}")

st.markdown("---")

# ===================== Tabs =====================
tab_diag, tab_eda, tab_imp, tab_fc = st.tabs(["📊 Diagnostics", "🔎 EDA", "🌿 Feature Importance", "🔮 Forecast"])

with tab_diag:
    st.markdown("#### 📉 Actual vs Predicted (Chosen model)")
    yhat = perf["yhat"]
    viz = pd.DataFrame({"month": test_df["month"], "Actual": y_test.values, "Pred": yhat}).melt("month", var_name="series", value_name="sales")
    chart = (
        alt.Chart(viz)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("sales:Q", title="Sales"),
            color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#1f77b4","#d62728"])),
            tooltip=[alt.Tooltip("month:T", title="Month"), "series:N", alt.Tooltip("sales:Q", format=".2f")],
        ).properties(height=320).interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### 📑 Model comparison (test metrics)")
    comp = pd.DataFrame([{k:v for k,v in d.items() if k not in ["yhat"]} for d in evs]).rename(columns={
        "name":"Model","rmse":"RMSE","mae":"MAE","mape":"MAPE","smape":"sMAPE","r2":"R²"
    })
    st.dataframe(comp, use_container_width=True, height=240)

with tab_eda:
    st.subheader("Top correlations with sales")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "sales" in num_cols:
        corr_all = df[num_cols].corr()["sales"].drop(labels=["sales"], errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
        corr_df = corr_all.reset_index()
        corr_df.columns = ["feature","corr_with_sales"]
        corr_df["abs_corr_with_sales"] = corr_df["corr_with_sales"].abs()
        bar = (
            alt.Chart(corr_df.head(20))
            .mark_bar()
            .encode(
                x=alt.X("abs_corr_with_sales:Q", title="|Correlation| with Sales"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                color=alt.Color("corr_with_sales:Q", title="Sign", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("corr_with_sales:Q", format=".3f")],
            ).properties(height=340)
        )
        st.altair_chart(bar, use_container_width=True)

    if "brand" in df.columns:
        st.subheader("Sales over time by brand")
        trend = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("sales:Q", title="Sales"),
                color=alt.Color("brand:N", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("month:T", title="Month"), "brand:N", alt.Tooltip("sales:Q", format=".2f")],
            ).properties(height=320).interactive()
        )
        st.altair_chart(trend, use_container_width=True)

with tab_imp:
    st.subheader("Most important features (chosen model)")
    # try to extract feature importances or coefficients
    chosen_est = (
        grid_en.best_estimator_ if chosen=="ElasticNet" else
        grid_g.best_estimator_  if chosen=="GBDT" else
        (grid_h.best_estimator_ if HGBR_OK and chosen=="HistGBDT" else None)
    )
    if chosen_est is None and chosen.startswith("Blend"):
        chosen_est = grid_g.best_estimator_  # show tree importances for blend

    try:
        prep = chosen_est.named_steps["prep"]
        feat_names = get_feature_names(prep)
        reg = chosen_est.named_steps["reg"]
        if hasattr(reg, "feature_importances_"):  # trees
            vals = reg.feature_importances_
        elif hasattr(reg, "coef_"):  # linear
            vals = np.abs(reg.coef_)
        else:
            vals = None

        if vals is not None and len(vals) == len(feat_names):
            imp = pd.DataFrame({"feature": feat_names, "importance": vals}).sort_values("importance", ascending=False).head(25)
            bar = (
                alt.Chart(imp)
                .mark_bar()
                .encode(
                    x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    tooltip=["feature:N", alt.Tooltip("importance:Q", format=".4f")],
                ).properties(height=360)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Could not compute feature importances for the selected model.")
    except Exception as e:
        st.info(f"Feature importance unavailable ({e}).")

with tab_fc:
    st.subheader(f"Forecast next {horizon} month(s)")

    last_month = df["month"].max()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    # templates: last row per (city,brand) or last overall
    if {"city","brand"}.issubset(df.columns):
        templates = df.sort_values("month").groupby(["city","brand"], as_index=False).tail(1).copy()
    else:
        templates = df[df["month"]==last_month].copy()
        if templates.empty:
            templates = df.tail(1).copy()

    def predict_model(Xframe: pd.DataFrame):
        if chosen == "ElasticNet":
            return grid_en.best_estimator_.predict(Xframe)
        elif chosen == "GBDT":
            return grid_g.best_estimator_.predict(Xframe)
        elif chosen == "HistGBDT" and HGBR_OK:
            return grid_h.best_estimator_.predict(Xframe)
        else:  # Blend
            return 0.3*grid_en.best_estimator_.predict(Xframe) + 0.7*grid_g.best_estimator_.predict(Xframe)

    future_rows, state = [], templates.copy()
    for m in future_months:
        step = state.copy()
        step["month"] = m
        step["month_num"] = m.month
        step["year_num"]  = m.year

        # recursive lags
        lag1 = step["pred_sales"].values if "pred_sales" in step.columns else step["sales"].values
        lag2 = step.get("prev_lag1", pd.Series(lag1)).values

        step["sales_lag_1"]     = lag1
        step["sales_lag_2"]     = lag2
        step["rolling_sales_3"] = pd.Series(lag1).rolling(3).mean().fillna(np.median(lag1)).values
        step["sales_change_1"]  = step["sales_lag_1"] - step["sales_lag_2"]
        if "projected_spike" not in step.columns:
            step["projected_spike"] = 0

        # Build model input with training columns
        model_in = pd.DataFrame(columns=numeric_features + categorical_features)
        for c in numeric_features:     model_in[c] = step[c].values if c in step.columns else 0
        for c in categorical_features: model_in[c] = step[c].values if c in step.columns else ""

        preds = predict_model(model_in)
        step["pred_sales"] = preds
        step["prev_lag1"]  = step["sales_lag_1"]

        keep = ["month","pred_sales"] + [c for c in ["city","brand","fashion_trend"] if c in step.columns]
        future_rows.append(step[keep])
        state = step.copy()

    future_out = pd.concat(future_rows, ignore_index=True) if future_rows else pd.DataFrame()
    if not future_out.empty:
        future_out = future_out.rename(columns={"pred_sales":"forecast_sales"})
        future_out["month"] = pd.to_datetime(future_out["month"]).dt.strftime("%Y-%m")
        future_out.to_csv(OUTPUT_FC, index=False)
        bus_emit("forecast_saved", {"rows": int(len(future_out)), "path": OUTPUT_FC, "model": chosen})

        st.success(f"✅ Forecast saved to `{OUTPUT_FC}`")
        st.dataframe(future_out.rename(columns={
            "month":"Period","city":"City","brand":"Brand","forecast_sales":"Forecasted Sales"
        }), use_container_width=True, height=360)

        with open(OUTPUT_FC, "rb") as f:
            st.download_button("⬇️ Download forecast_output.csv", f, file_name="forecast_output.csv", mime="text/csv")

        # continuation plot
        hist = df[["month","sales"]].copy(); hist["series"] = "Historical"
        fut  = future_out[["month","forecast_sales"]].copy()
        fut["month"] = pd.to_datetime(fut["month"]); fut = fut.rename(columns={"forecast_sales":"sales"}); fut["series"] = "Forecast"
        viz = pd.concat([hist, fut], ignore_index=True)

        chart = (
            alt.Chart(viz)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("sales:Q", title="Sales"),
                color=alt.Color("series:N", legend=alt.Legend(orient="bottom", title=None),
                                scale=alt.Scale(range=["#1f77b4","#d62728"])),
                tooltip=[alt.Tooltip("month:T", title="Month"), "series:N", alt.Tooltip("sales:Q", format=".2f")],
            ).properties(height=320, title=f"Historical + {chosen} Forecast").interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No future forecasts produced. Check that template rows exist for the last month.")

# ===================== Model card =====================
card = {
    "selected_model": chosen,
    "selection_metric": metric_choice,
    "anti_overfit": anti_overfit,
    "metrics_test": {d["name"]: {k:v for k,v in d.items() if k!="yhat"} for d in evs},
    "best_params": {
        "ElasticNet": grid_en.best_params_,
        "GBDT":       grid_g.best_params_,
        **({"HistGBDT": (grid_h.best_params_ if HGBR_OK else None)} if HGBR_OK else {})
    },
    "data_file": os.path.basename(DATA_PATH),
    "forecast_file": os.path.basename(OUTPUT_FC),
}
with open(MODEL_CARD, "w") as f:
    json.dump(card, f, indent=2)