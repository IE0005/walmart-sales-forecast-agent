# Walmart Store Sales Forecast

Interactive Streamlit app for the Walmart Recruiting Store Sales Forecasting
dataset. Pick a store and department to see historical weekly sales plus a
forecasted trend for upcoming weeks.

Also includes an agentic AI assistant (`agent_app.py`) that answers plain-English
restocking questions ("Will Store 20 need more inventory for Department 5 next
month?") by calling tools backed by the same pipeline and trained model.

## Project layout

- `pipeline.py` — shared data loading + feature engineering (used by training and both apps)
- `train_model.py` — trains the Random Forest model and saves it to `models/sales_model.pkl`
- `app.py` — the Streamlit sales-forecast dashboard
- `agent_tools.py` — the agent's tools (query sales history, run forecast, detect anomalies)
- `agent_core.py` — the Claude tool-use agentic loop
- `agent_app.py` — the Streamlit chat UI for the agent, showing tool calls per turn
- `train.csv`, `test.csv`, `features.csv`, `stores.csv` — raw competition data

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 train_model.py   # only needed once, or after changing the data/model
streamlit run app.py
```

### Running the inventory assistant

The agent needs an Anthropic API key. Either export it before launching Streamlit:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run agent_app.py
```

or add it to `.streamlit/secrets.toml` (not committed to git):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

By default the agent uses `claude-opus-5`; override with the `AGENT_MODEL` env var
(e.g. `AGENT_MODEL=claude-sonnet-5`) for a cheaper/faster model.

The agent has three tools it calls as needed, and the chat UI shows each call's
inputs/outputs in an expander under the assistant's reply:

- **`query_sales_history`** — recent weekly sales for a store/department, with summary stats
- **`forecast_demand`** — runs the trained Random Forest model to predict upcoming weeks
- **`detect_anomalies`** — flags statistically unusual recent weeks (z-score based), separating holiday swings from real anomalies

Claude decides whether a restock is needed by reasoning over the combined tool
outputs — that reasoning isn't a separate tool call, it's Claude synthesizing the
trend, forecast, and anomaly results into a plain-English answer.

## Model

A Random Forest (`max_depth=18`, `min_samples_leaf=4`, 50 trees) trained on
store metadata, calendar features, and economic indicators (CPI,
unemployment, fuel price, temperature, markdowns). This configuration was
chosen over the notebook's unconstrained Extra Trees model (R²=0.984)
because that model pickles to ~5.7GB — this one gets R²=0.977 at ~58MB,
which is deployable on Streamlit Community Cloud and GitHub.

Forecasted weeks use `test.csv`'s real future dates and known exogenous
features (that file covers 2012-11 through 2013-07) rather than assuming
those inputs stay constant.

## Deploying to Streamlit Community Cloud

1. Push this directory to a GitHub repo (see below).
2. Go to https://share.streamlit.io, sign in, and click "New app".
3. Point it at the repo, branch, and `app.py` as the main file.
4. Deploy. Community Cloud installs from `requirements.txt` automatically.
