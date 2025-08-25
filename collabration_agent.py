# pages/6_collaboration_agent.py
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------------
# CONFIG & PATHS
# -------------------------------
st.set_page_config(page_title="Collaboration Agent", layout="wide")
st.title("🤝 Collaboration Agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TASKS_CSV = os.path.join(DATA_DIR, "collab_tasks.csv")
COMMS_CSV = os.path.join(DATA_DIR, "collab_comms.csv")
APPR_CSV = os.path.join(DATA_DIR, "collab_approvals.csv")
SIM_CSV = os.path.join(DATA_DIR, "simulation_output.csv")  # to seed tasks

os.makedirs(DATA_DIR, exist_ok=True)
alt.data_transformers.disable_max_rows()

# -------------------------------
# HELPERS
# -------------------------------
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _parse_dt(s):
    if pd.isna(s):
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def _week_start(dti: pd.Series) -> pd.Series:
    """Return Monday-start week timestamps (for grouping)."""
    dti = pd.to_datetime(dti, errors="coerce")
    return (dti - pd.to_timedelta(dti.dt.weekday, unit="D")).dt.normalize()

def ensure_csv(path, cols, seed_df=None):
    if not os.path.exists(path):
        df = seed_df[cols].copy() if seed_df is not None and not seed_df.empty else pd.DataFrame(columns=cols)
        df.to_csv(path, index=False)

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    df.to_csv(path, index=False)

# -------------------------------
# BUSINESS EXPLAINERS (like your other pages)
# -------------------------------
with st.expander("📬 What information does this page use?", expanded=True):
    st.markdown(f"""
- **Tasks**: `{os.path.basename(TASKS_CSV)}` — id, city, brand, title, status, owner, due date, risk.  
- **Comms**: `{os.path.basename(COMMS_CSV)}` — timestamped notes (email/slack/call).  
- **Approvals**: `{os.path.basename(APPR_CSV)}` — who approved or rejected what.  
- **Optional seed**: `{os.path.basename(SIM_CSV)}` to create tasks from simulation decisions.
""")

with st.expander("🧠 What this page does (plain language)"):
    st.markdown("""
1) Maintains a **task board** for planners and city/brand owners.  
2) Tracks **flow**: tasks **created** vs **completed** per week, and shows **backlog**.  
3) Surfaces **risks** (due soon, SLA risk) and logs **communication** & **approvals**.  
4) Writes everything back to CSV so other pages / people can consume it.
""")

with st.expander("🔗 Hand-offs"):
    st.markdown("""
- Upstream: **Simulation** → seeds tasks with strategy & required units.  
- Downstream: **Ops/Leadership** → reads status, backlog trend, risks, approvals.  
- CSV contracts let you plug this page into external tools without code changes.
""")

with st.expander("✨ Why this is practically useful"):
    st.markdown("""
- **Accountability**: each decision becomes a dated, owned task.  
- **Throughput awareness**: if **created > done**, backlog grows — easy to spot.  
- **Risk-first** views: due-soon & high-risk filters help you triage.  
- **Traceability**: comms and approvals are attached to task IDs.
""")

with st.expander("🧪 Data contracts & QA"):
    st.markdown("""
- `created_at` and `completed_at` must be parseable timestamps (we auto-add them).  
- Moving a task to **Done** stamps `completed_at`; moving out of Done clears it.  
- Weekly charts use **Monday** as the week start.  
- If files are missing, we create them with minimal schemas and (optionally) seed from simulation.
""")

# -------------------------------
# SEED FROM SIMULATION (optional, deterministic)
# -------------------------------
rng = np.random.default_rng(42)
sim_df = None
if os.path.exists(SIM_CSV):
    try:
        sim_df = pd.read_csv(SIM_CSV)
    except Exception:
        sim_df = None

seed_tasks = pd.DataFrame()
if sim_df is not None and not sim_df.empty:
    keep = [c for c in ["month", "city", "brand", "best_strategy", "required_units"] if c in sim_df.columns]
    tmp = sim_df[keep].copy()
    tmp["title"] = "Execute: " + tmp.get("best_strategy", "Action").astype(str)
    tmp["status"] = "Pending"
    tmp["priority"] = np.where(tmp.get("best_strategy", "").eq("Transfer"), "High", "Medium")
    tmp["owner"] = "unassigned"
    # Due date = end of target month or +14 days fallback
    if "month" in tmp.columns:
        mdt = pd.to_datetime(tmp["month"], format="%b-%Y", errors="coerce")
        tmp["due_date"] = np.where(
            mdt.notna(), (mdt + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
        )
    else:
        tmp["due_date"] = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

    # Create some **historical** created_at dates (last 6–30 days) so the weekly chart has signal
    base = datetime.now()
    offsets = rng.integers(6, 31, size=len(tmp))
    tmp["created_at"] = [(base - timedelta(days=int(d))).strftime("%Y-%m-%d %H:%M:%S") for d in offsets]
    tmp["updated_at"] = tmp["created_at"]
    tmp["completed_at"] = ""  # empty until done

    tmp["source_agent"] = "Simulation"
    tmp["decision"] = tmp.get("best_strategy", "")
    tmp["notes"] = ""
    # SLA risk heuristic
    req = pd.to_numeric(tmp.get("required_units", 0), errors="coerce").fillna(0)
    tmp["sla_risk"] = pd.cut(req, bins=[-1, 50, 150, 1e9], labels=["Low", "Medium", "High"]).astype(str)

    # Final ordering
    seed_tasks = tmp.rename(columns={"required_units": "required_units"})[
        ["month", "city", "brand", "title", "status", "priority", "owner", "due_date",
         "source_agent", "decision", "required_units", "sla_risk", "notes",
         "created_at", "updated_at", "completed_at"]
    ].copy()

# Ensure files exist with schema
TASK_COLS = [
    "task_id", "month", "city", "brand", "title", "status", "priority", "owner",
    "due_date", "source_agent", "required_units", "decision", "notes",
    "created_at", "updated_at", "completed_at", "sla_risk"
]
if seed_tasks.empty:
    seed_tasks = pd.DataFrame(columns=[c for c in TASK_COLS if c != "task_id"])
seed_tasks = seed_tasks.reset_index(drop=True)
seed_tasks.insert(0, "task_id", range(1001, 1001 + len(seed_tasks)))

ensure_csv(TASKS_CSV, TASK_COLS, seed_df=seed_tasks)
ensure_csv(COMMS_CSV, ["ts", "channel", "from", "to", "message", "related_task_id"])
ensure_csv(APPR_CSV, ["approval_id", "task_id", "approver", "status", "decided_at", "comments"])

# -------------------------------
# LOAD DATA
# -------------------------------
tasks = load_csv(TASKS_CSV)
comms = load_csv(COMMS_CSV)
apprs = load_csv(APPR_CSV)

# Ensure columns exist even if older file versions are present
for col in TASK_COLS:
    if col not in tasks.columns:
        tasks[col] = ""

# Coerce types
for c in ["required_units"]:
    tasks[c] = pd.to_numeric(tasks[c], errors="coerce").fillna(0.0).astype(float)

tasks["created_at_dt"] = _parse_dt(tasks.get("created_at"))
tasks["updated_at_dt"] = _parse_dt(tasks.get("updated_at"))
tasks["completed_at_dt"] = _parse_dt(tasks.get("completed_at"))
tasks["due_dt"] = pd.to_datetime(tasks.get("due_date"), errors="coerce")

# -------------------------------
# OVERVIEW (KPIs + Flow)
# -------------------------------
st.subheader("📊 Overview")

open_mask = tasks["status"].isin(["Pending", "In-Progress", "Blocked"])
total_open = int(open_mask.sum())

high_risk = int((tasks.get("sla_risk", "").astype(str) == "High").sum())

if "due_dt" in tasks.columns:
    due_in_days = (tasks["due_dt"] - pd.Timestamp.today()).dt.days
    due_7days = int(((due_in_days <= 7) & (due_in_days >= 0) & open_mask).sum())
else:
    due_7days = 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Open Tasks", total_open)
k2.metric("High Risk", high_risk)
k3.metric("Due ≤ 7 Days", due_7days)

# — Status mix (quick glance)
if {"status"}.issubset(tasks.columns) and not tasks.empty:
    mix = tasks["status"].value_counts().reset_index()
    mix.columns = ["status", "count"]
    mix_bar = (
        alt.Chart(mix)
        .mark_bar()
        .encode(
            x=alt.X("status:N", title="Status"),
            y=alt.Y("count:Q", title="Tasks"),
            color=alt.Color("status:N", legend=None,
                            scale=alt.Scale(range=["#2c7be5", "#e8590c", "#fab005", "#37b24d"])),
            tooltip=["status", "count"],
        )
        .properties(height=220)
    )
    k4.altair_chart(mix_bar, use_container_width=True)

st.markdown("### 📈 Flow — Created vs Done (weekly)")

# Build weekly created/done series
created = tasks.dropna(subset=["created_at_dt"]).copy()
created["week"] = _week_start(created["created_at_dt"])
created_w = created.groupby("week", as_index=False)["task_id"].count().rename(columns={"task_id": "created"})

done = tasks.dropna(subset=["completed_at_dt"]).copy()
done["week"] = _week_start(done["completed_at_dt"])
done_w = done.groupby("week", as_index=False)["task_id"].count().rename(columns={"task_id": "done"})

flow = pd.merge(created_w, done_w, on="week", how="outer").fillna(0)
flow = flow.sort_values("week")

# Keep last 12 weeks (but show something if fewer)
if not flow.empty:
    last_12_cut = flow["week"].max() - pd.Timedelta(days=7 * 12)
    flow = flow[flow["week"] >= last_12_cut]

    flow_m = flow.melt("week", var_name="series", value_name="tasks")
    flow_chart = (
        alt.Chart(flow_m)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("week:T", title="Week (Mon start)"),
            y=alt.Y("tasks:Q", title="Tasks"),
            color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["created", "done"],
                                                                    range=["#1f77b4", "#d62728"])),
            tooltip=[alt.Tooltip("week:T", title="Week"), "series:N", alt.Tooltip("tasks:Q", format=".0f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(flow_chart, use_container_width=True)
else:
    st.info("No `created_at` / `completed_at` timestamps yet — create or complete some tasks and the chart will populate.")

# Backlog = cumulative created - cumulative done
if not flow.empty:
    flow["cum_created"] = flow["created"].cumsum()
    flow["cum_done"] = flow["done"].cumsum()
    flow["backlog"] = (flow["cum_created"] - flow["cum_done"]).clip(lower=0)
    backlog = flow[["week", "backlog"]]
    backlog_chart = (
        alt.Chart(backlog)
        .mark_area(opacity=0.35)
        .encode(
            x=alt.X("week:T", title="Week (Mon start)"),
            y=alt.Y("backlog:Q", title="Open Tasks (cumulative)"),
            tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("backlog:Q", format=".0f")],
        )
        .properties(height=200, title="🧳 Backlog (end of week)")
    )
    st.altair_chart(backlog_chart, use_container_width=True)

st.markdown("---")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗂 Task Board", "🚨 Alerts / Risks", "💬 Comms Log", "✅ Approvals"]
)

# ========== TAB 1: Task Board ==========
with tab1:
    st.subheader("🗂 Task Board")

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    cities = ["All"] + sorted(tasks["city"].dropna().unique().tolist()) if "city" in tasks.columns else ["All"]
    brands = ["All"] + sorted(tasks["brand"].dropna().unique().tolist()) if "brand" in tasks.columns else ["All"]
    statuses = ["All"] + sorted(tasks["status"].dropna().unique().tolist()) if "status" in tasks.columns else ["All"]
    risks = ["All", "Low", "Medium", "High"]

    with f1:
        sel_city = st.selectbox("City", cities, index=0, key="flt_city")
    with f2:
        sel_brand = st.selectbox("Brand", brands, index=0, key="flt_brand")
    with f3:
        sel_status = st.selectbox("Status", statuses, index=0, key="flt_status")
    with f4:
        sel_risk = st.selectbox("SLA Risk", risks, index=0, key="flt_risk")

    view = tasks.copy()
    if sel_city != "All":
        view = view[view["city"] == sel_city]
    if sel_brand != "All":
        view = view[view["brand"] == sel_brand]
    if sel_status != "All":
        view = view[view["status"] == sel_status]
    if sel_risk != "All":
        view = view[view["sla_risk"] == sel_risk]

    st.dataframe(view, use_container_width=True, height=350)

    st.markdown("### ✏️ Quick Update")
    if not view.empty:
        row = st.selectbox(
            "Select Task to Update",
            view["task_id"].tolist(),
            key="upd_task_select",
        )
        new_status = st.selectbox(
            "New Status", ["Pending", "In-Progress", "Blocked", "Done"], key="upd_status"
        )
        new_owner = st.text_input("Owner", value="unassigned", key="upd_owner")
        new_notes = st.text_area(
            "Notes", value="", placeholder="Add a brief update…", key="upd_notes"
        )

        if st.button("Update Task", key="btn_update_task"):
            idx = tasks.index[tasks["task_id"] == row]
            if len(idx):
                i = idx[0]
                old_status = str(tasks.at[i, "status"])

                tasks.at[i, "status"] = new_status
                tasks.at[i, "owner"] = new_owner
                if new_notes:
                    prev = str(tasks.at[i, "notes"]) if "notes" in tasks.columns else ""
                    sep = "\n" if prev else ""
                    tasks.at[i, "notes"] = prev + sep + f"[{_now_str()}] {new_notes}"
                tasks.at[i, "updated_at"] = _now_str()

                # Stamp/clear completed_at appropriately
                if new_status == "Done" and not pd.notna(_parse_dt(tasks.at[i, "completed_at"])):
                    tasks.at[i, "completed_at"] = _now_str()
                if new_status != "Done" and pd.notna(_parse_dt(tasks.at[i, "completed_at"])):
                    tasks.at[i, "completed_at"] = ""

                save_csv(tasks.drop(columns=[c for c in ["created_at_dt", "updated_at_dt", "completed_at_dt"] if c in tasks.columns]), TASKS_CSV)
                st.success("Task updated.")
                st.experimental_rerun()

# ========== TAB 2: Alerts / Risks ==========
with tab2:
    st.subheader("🚨 Alerts & Risks")
    alerts = tasks.copy()
    alerts["due_days"] = (alerts.get("due_dt", pd.Timestamp.today()) - pd.Timestamp.today()).dt.days
    overdue = alerts[(alerts["status"].isin(["Pending", "In-Progress", "Blocked"])) & (alerts["due_days"] < 0)]
    highrisk = alerts[alerts["sla_risk"].eq("High")] if "sla_risk" in alerts.columns else pd.DataFrame()

    st.markdown("**Overdue**")
    st.dataframe(overdue, use_container_width=True, height=220)

    st.markdown("**High Risk**")
    st.dataframe(highrisk, use_container_width=True, height=220)

# ========== TAB 3: Comms Log ==========
with tab3:
    st.subheader("💬 Communication Log")
    st.dataframe(comms.sort_values("ts", ascending=False), use_container_width=True, height=280)

    st.markdown("### ➕ Add Note")
    c1, c2, c3 = st.columns(3)
    with c1:
        channel = st.selectbox("Channel", ["Email", "Slack", "Call", "Note"], key="comms_channel")
    with c2:
        sender = st.text_input("From", value="planner@team", key="comms_from")
    with c3:
        target = st.text_input("To", value="ops@team", key="comms_to")

    rel_id = st.text_input("Related Task ID (optional)", value="", key="comms_related_task_id")
    msg = st.text_area("Message", placeholder="Short note…", key="comms_message")

    if st.button("Add Message", key="btn_add_message"):
        new = {
            "ts": _now_str(),
            "channel": channel,
            "from": sender,
            "to": target,
            "message": msg,
            "related_task_id": rel_id,
        }
        comms = pd.concat([comms, pd.DataFrame([new])], ignore_index=True)
        save_csv(comms, COMMS_CSV)
        st.success("Message saved.")
        st.experimental_rerun()

# ========== TAB 4: Approvals ==========
with tab4:
    st.subheader("✅ Approvals")
    st.dataframe(apprs.sort_values("decided_at", ascending=False), use_container_width=True, height=280)

    st.markdown("### ➕ New Approval")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        task_id_in = st.text_input("Task ID", key="appr_task_id")
    with a2:
        approver = st.text_input("Approver", value="city_manager", key="appr_approver")
    with a3:
        status = st.selectbox("Status", ["Pending", "Approved", "Rejected"], index=0, key="appr_status")
    with a4:
        decided_at = st.text_input("Decided At", value=_now_str(), key="appr_decided_at")
    comments = st.text_area("Comments", value="", key="appr_comments")

    if st.button("Save Approval", key="btn_save_approval"):
        next_id = (apprs["approval_id"].max() + 1) if not apprs.empty else 10001
        row = {
            "approval_id": next_id,
            "task_id": task_id_in,
            "approver": approver,
            "status": status,
            "decided_at": decided_at,
            "comments": comments,
        }
        apprs = pd.concat([apprs, pd.DataFrame([row])], ignore_index=True)
        save_csv(apprs, APPR_CSV)
        st.success("Approval recorded.")
        st.experimental_rerun()