import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import FEATURE_COLS, DATA_DIR, build_datasets

MODEL_PATH = DATA_DIR / "models" / "sales_model.pkl"
METRICS_PATH = DATA_DIR / "models" / "metrics.json"
VALIDATION_PATH = DATA_DIR / "models" / "validation_predictions.csv"
COMPARISON_PATH = DATA_DIR / "models" / "model_comparison.csv"

st.set_page_config(page_title="Walmart Sales Forecast", page_icon="📈", layout="centered")


@st.cache_data
def get_data():
    return build_datasets()


@st.cache_data
def get_validation():
    return pd.read_csv(VALIDATION_PATH, parse_dates=["Date"])


@st.cache_data
def get_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_comparison():
    return pd.read_csv(COMPARISON_PATH)


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


data_train, data_test = get_data()
validation_df = get_validation()
metrics = get_metrics()
comparison_df = get_comparison()
model = get_model()

st.title("📈 Walmart Store Sales Forecast")
st.caption("Historical weekly sales with a forecasted trend for upcoming weeks, by store and department.")

with st.expander("Model performance (validation set)", expanded=False):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:,.0f}")
    m2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
    m3.metric("R²", f"{metrics['R2']:.3f}")
    m4.metric("WMAE", f"{metrics['WMAE']:,.0f}")
    st.caption(
        f"Computed on a held-out 10% sample ({metrics['n_validation_rows']:,} weeks) "
        "the model never trained on. WMAE weights holiday weeks 5x, matching the "
        "competition's scoring metric."
    )

with st.expander("Compare candidate models", expanded=False):
    show_df = comparison_df.copy()
    show_df["MAE"] = show_df["MAE"].map(lambda v: f"{v:,.0f}")
    show_df["RMSE"] = show_df["RMSE"].map(lambda v: f"{v:,.0f}")
    show_df["R2"] = show_df["R2"].map(lambda v: f"{v:.3f}")
    show_df["WMAE"] = show_df["WMAE"].map(lambda v: f"{v:,.0f}")
    show_df["Pickled_Size_MB"] = show_df["Pickled_Size_MB"].map(lambda v: f"{v:,.0f} MB")
    show_df = show_df.rename(columns={"R2": "R²", "Pickled_Size_MB": "Model size"})
    st.dataframe(show_df, hide_index=True, use_container_width=True)
    st.caption(
        "All five were trained on the identical 90/10 split. Extra Trees scores "
        "best but pickles to over 1GB (uncompressed, 5.7GB) because Store/Dept "
        "let it memorize almost every row -- too large for GitHub or Streamlit "
        "Community Cloud. The deployed Random Forest trades a small amount of "
        "accuracy for a ~54MB file that actually fits within those limits."
    )

stores = sorted(data_train["Store"].unique())

with st.sidebar:
    st.header("Selection")
    store = st.selectbox("Store", stores)

    depts = sorted(data_train.loc[data_train["Store"] == store, "Dept"].unique())
    dept = st.selectbox("Department (product category)", depts)

    horizon = st.slider("Weeks to forecast", min_value=4, max_value=39, value=12)

hist = data_train[(data_train["Store"] == store) & (data_train["Dept"] == dept)].sort_values("Date")
future = data_test[(data_test["Store"] == store) & (data_test["Dept"] == dept)].sort_values("Date").head(horizon)
holdout = validation_df[(validation_df["Store"] == store) & (validation_df["Dept"] == dept)].sort_values("Date")

if hist.empty:
    st.warning("No historical data for this store/department combination.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Last recorded week", hist["Date"].max().strftime("%Y-%m-%d"))
col2.metric("Last week's sales", f"${hist['Weekly_Sales'].iloc[-1]:,.0f}")

if not future.empty:
    forecast = model.predict(future[FEATURE_COLS])
    col3.metric(f"Forecasted total (next {len(future)}w)", f"${forecast.sum():,.0f}")
else:
    forecast = None
    col3.metric("Forecast", "No data available")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist["Date"], y=hist["Weekly_Sales"],
    mode="lines", name="Historical sales", line=dict(color="#1f77b4"),
))

if not holdout.empty:
    fig.add_trace(go.Scatter(
        x=holdout["Date"], y=holdout["Predicted_Sales"],
        mode="markers", name="Model prediction (held-out weeks)",
        marker=dict(color="#2ca02c", size=7, symbol="diamond"),
        customdata=holdout["Actual_Sales"],
        hovertemplate="Predicted: $%{y:,.0f}<br>Actual: $%{customdata:,.0f}<extra></extra>",
    ))

if forecast is not None:
    bridge_x = pd.concat([hist["Date"].iloc[-1:], future["Date"]])
    bridge_y = [hist["Weekly_Sales"].iloc[-1], *forecast]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines", name="Forecast", line=dict(color="#ff7f0e", dash="dash"),
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Weekly Sales ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
    height=450,
)

st.plotly_chart(fig, use_container_width=True)

if not holdout.empty:
    local_mae = np.abs(holdout["Actual_Sales"] - holdout["Predicted_Sales"]).mean()
    st.caption(
        f"Green diamonds are weeks held out during training for Store {store} / "
        f"Dept {dept} — the model never saw these when learning, so they show real "
        f"accuracy for this exact store/department. Average error here (MAE): {local_mae:,.0f}."
    )

with st.expander("About this forecast"):
    st.write(
        "The trend line is produced by a Random Forest regression model trained on "
        "historical sales together with store metadata, calendar features (week, "
        "month, holidays), and economic indicators (CPI, unemployment, fuel price, "
        "temperature, markdowns). Forecasted weeks reuse the known future values of "
        "those exogenous features rather than assuming they stay constant."
    )
