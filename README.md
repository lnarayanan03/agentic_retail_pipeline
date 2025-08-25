Agentic AI Retail – End-to-End Demo (Streamlit)

A multi-agent, CSV-driven Streamlit app that senses demand, refines forecasts, plans inventory, simulates costed strategies, and turns decisions into tasks. Built to be explainable, operator-tunable, and file-based so you can run it locally or in CI without external services.

🧭 What’s inside

/pages
  0_agent_hub.py                 # Read-only hub: shows outputs from all agents
  1_info.py                      # Dataset info & EDA
  2_sensing_agent.py             # Social spike detection → Detected_Spikes_*.csv
  3_demand_forecasting_agent.py  # Train + forecast → forecast_output.csv
  4_inventory_agent.py           # Coverage + reorder → inventory_plan.csv
  5_simulation_agent.py          # Strategy & costs → simulation_output.csv
  6_collaboration_agent.py       # Tasks/comms/approvals → collab_*.csv
  7_learning_agent.py            # Threshold & policy tuning → learning_suggestions.json
app_memory_bus.py                # (optional) demo of memory bus & shared data
/data
  Simulated_Agentic_AI_Dataset_RandomizedDates.csv   # sensing base dataset
  ForecastingDataset_Final_Cleaned.csv               # forecasting panel
  cost_parameters.csv                                # unit/penalty costs
  onhand_inventory_snapshot.csv                      # seed snapshot (can be auto-built)
  # The app writes these as you run pages (small samples included):
  Detected_Spikes_Past.csv, Detected_Spikes_Projected.csv
  forecast_output.csv
  inventory_plan.csv
  simulation_output.csv
  collab_tasks.csv, collab_comms.csv, collab_approvals.csv
  memory_bus.jsonl                                   # lightweight “event bus”

  🛠️ Setup

Requirements
	•	macOS / Linux / Windows
	•	Python 3.10 – 3.12
	•	(Optional) prophet for the sensing forecast fallback; app degrades gracefully if not present

Install

# 1) Clone
git clone <your-repo-url>
cd your-repo

# 2) Create & activate a virtual env (macOS)
python3 -m venv .venv
source .venv/bin/activate

# 3) Install deps
pip install -U pip
pip install -r requirements.txt
    If prophet fails to install, ignore it; the app falls back to a moving-average in Sensing.

▶️ Run

Pick one of these entry points:

# Full read-only hub over all CSVs
streamlit run pages/0_agent_hub.py

# Or launch any individual agent page
streamlit run pages/2_sensing_agent.py
# streamlit run pages/3_demand_forecasting_agent.py
# streamlit run pages/4_inventory_agent.py
# streamlit run pages/5_simulation_agent.py
# streamlit run pages/6_collaboration_agent.py
# streamlit run pages/7_learning_agent.py

# (Optional) memory bus demo
streamlit run app_memory_bus.py

🔄 Recommended run order
	1.	Info & EDA (1_info.py) – sanity-check the base dataset.
	2.	Sensing (2_sensing_agent.py) – writes Detected_Spikes_Past.csv and Detected_Spikes_Projected.csv.
	3.	Forecasting (3_demand_forecasting_agent.py) – trains & writes forecast_output.csv.
	4.	Inventory (4_inventory_agent.py) – computes coverage/reorder → inventory_plan.csv.
	5.	Simulation (5_simulation_agent.py) – scores strategies/costs → simulation_output.csv.
	6.	Collaboration (6_collaboration_agent.py) – turns rows into tasks/comms → collab_*.csv.
	7.	Learning (7_learning_agent.py) – tunes thresholds & policy → learning_suggestions.json.

Finally, open Agent Hub (0_agent_hub.py) to view everything in one place (filters by city/brand/month).

🗂️ Datasets (what they contain)
	•	Simulated_Agentic_AI_Dataset_RandomizedDates.csv
Monthly social signals and context by city, brand, keyword, season + optional units_sold.
Used by Sensing for spike detection and EDA.
	•	ForecastingDataset_Final_Cleaned.csv
Historical sales panel with context (city, brand, fashion_trend, flags).
Used by Forecasting to engineer features (lags/rolling/season) and train models.
	•	cost_parameters.csv
Simple table of per-unit costs: raw, production, logistics, transfer, SLA penalty.
Used by Simulation to cost strategies.
	•	onhand_inventory_snapshot.csv
Starter on-hand/pipeline per city×brand. If missing, Inventory will synthesize a deterministic snapshot.
	•	Outputs written by agents (CSV/JSON):
	•	Sensing → Detected_Spikes_Past.csv, Detected_Spikes_Projected.csv
	•	Forecasting → forecast_output.csv
	•	Inventory → inventory_plan.csv
	•	Simulation → simulation_output.csv
	•	Collaboration → collab_tasks.csv, collab_comms.csv, collab_approvals.csv
	•	Learning → learning_suggestions.json
	•	Shared event log → memory_bus.jsonl

🧠 How the “agents” chain together

Sensing → Forecasting → Inventory → Simulation → Collaboration
         (features)     (coverage)    (strategy)     (tasks)
        ▼               ▼             ▼              ▼
Detected_Spikes_*  forecast_output  inventory_plan  simulation_output → collab_*

	•	Memory Bus (memory_bus.jsonl) is a simple JSON-lines event log each page appends to (page opened, params changed, file saved), so you can audit runs and keep lightweight state across pages.

⚙️ Key controls you can tune (no code changes)
	•	Sensing: engagement ratio threshold, z-score jump threshold, min history, months ahead
	•	Forecasting: model choice (ElasticNet / GBDT / HistGBDT / Blend), selection metric (MAE/MSE), horizon, anti-overfit grid
	•	Inventory: service level (% safety stock), (reserved) lead-time, per-group grid
	•	Simulation: per-unit costs & weights (normalize to 1), decision thresholds (do-nothing & hybrid)
	•	Learning: grid-search sensing thresholds vs sales-uplift proxy, safety% heuristic from volatility, correlation drift checks

🧪 Troubleshooting
	•	“Data not found” → check files in /data and exact filenames; CSVs are case-sensitive.
	•	Duplicate Streamlit widget keys → when you duplicate a widget label, add a unique key="...".
	•	TypeError: truth value of Series is ambiguous → use vectorized checks like .any()/.all(); the code already guards the common cases.
	•	Prophet install issues → ignore; sensing will auto-fallback to a moving-average forecast.
	•	macOS permission denied when cd → ensure you’re not trying to execute a folder. Use cd /path/to/folder (no ./folder as a command).

📦 Requirements

See requirements.txt:

streamlit
pandas
numpy
altair
matplotlib
seaborn
scikit-learn
wordcloud
prophet    # optional; app falls back if missing

🤝 Contributing
	•	PRs welcome for: new features (lead-time aware inventory, policy A/Bs), UX polish, test fixtures.
	•	Keep artifacts file-based & small so the demo remains portable.

Tip: Use the included .gitignore to keep large/private data out of git, while whitelisting the tiny sample CSVs needed for the demo to run.