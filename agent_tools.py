"""Tool implementations for the inventory-forecasting agent.

Each tool reuses the existing pipeline (pipeline.py) and the pretrained
Random Forest model (models/sales_model.pkl) rather than duplicating any
forecasting logic. These functions are plain Python so they can be called
directly (for testing) or dispatched by the Claude tool-use loop in
agent_core.py.
"""
from __future__ import annotations

from functools import lru_cache

import joblib

from pipeline import DATA_DIR, FEATURE_COLS, build_datasets

MODEL_PATH = DATA_DIR / "models" / "sales_model.pkl"

TYPE_LABELS = {1: "A", 2: "B", 3: "C"}


@lru_cache(maxsize=1)
def _get_data():
    return build_datasets()


@lru_cache(maxsize=1)
def _get_model():
    return joblib.load(MODEL_PATH)


def _validate_store_dept(store: int, dept: int) -> str | None:
    data_train, _ = _get_data()
    if store not in data_train["Store"].unique():
        return f"Store {store} does not exist in the dataset."
    depts = data_train.loc[data_train["Store"] == store, "Dept"]
    if dept not in depts.unique():
        return f"Department {dept} was not found at Store {store}."
    return None


def query_sales_history(store: int, dept: int, weeks: int = 26) -> dict:
    """Return recent historical weekly sales for a store/department."""
    error = _validate_store_dept(store, dept)
    if error:
        return {"error": error}

    data_train, _ = _get_data()
    hist = (
        data_train[(data_train["Store"] == store) & (data_train["Dept"] == dept)]
        .sort_values("Date")
        .tail(weeks)
    )
    if hist.empty:
        return {"error": f"No historical sales rows for Store {store} / Dept {dept}."}

    weekly = [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "weekly_sales": round(float(row.Weekly_Sales), 2),
            "is_holiday": bool(row.IsHoliday),
        }
        for row in hist.itertuples()
    ]
    sales = hist["Weekly_Sales"].to_numpy()
    store_row = hist.iloc[-1]
    return {
        "store": store,
        "dept": dept,
        "store_type": TYPE_LABELS.get(int(store_row["Type"]), "unknown"),
        "store_size": int(store_row["Size"]),
        "weeks_returned": len(weekly),
        "weekly_sales": weekly,
        "summary": {
            "mean": round(float(sales.mean()), 2),
            "std": round(float(sales.std()), 2),
            "min": round(float(sales.min()), 2),
            "max": round(float(sales.max()), 2),
            "last_week": round(float(sales[-1]), 2),
            "trailing_4wk_avg": round(float(sales[-4:].mean()), 2) if len(sales) >= 4 else None,
        },
    }


def forecast_demand(store: int, dept: int, horizon_weeks: int = 8) -> dict:
    """Forecast upcoming weekly sales using the trained Random Forest model."""
    error = _validate_store_dept(store, dept)
    if error:
        return {"error": error}

    _, data_test = _get_data()
    model = _get_model()

    future = (
        data_test[(data_test["Store"] == store) & (data_test["Dept"] == dept)]
        .sort_values("Date")
        .head(horizon_weeks)
    )
    if future.empty:
        return {
            "error": (
                f"No future/test-period data available for Store {store} / Dept {dept} "
                "to forecast against."
            )
        }

    predictions = model.predict(future[FEATURE_COLS])
    forecast = [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "forecast_sales": round(float(pred), 2),
            "is_holiday": bool(row.IsHoliday),
        }
        for row, pred in zip(future.itertuples(), predictions)
    ]
    return {
        "store": store,
        "dept": dept,
        "horizon_weeks": len(forecast),
        "forecast": forecast,
        "total_forecast_sales": round(float(predictions.sum()), 2),
        "avg_weekly_forecast": round(float(predictions.mean()), 2),
    }


def detect_anomalies(
    store: int, dept: int, lookback_weeks: int = 16, z_threshold: float = 2.0
) -> dict:
    """Flag statistically unusual recent weeks in a store/department's sales history."""
    error = _validate_store_dept(store, dept)
    if error:
        return {"error": error}

    data_train, _ = _get_data()
    hist = (
        data_train[(data_train["Store"] == store) & (data_train["Dept"] == dept)]
        .sort_values("Date")
        .tail(lookback_weeks)
    )
    if len(hist) < 4:
        return {"error": "Not enough history to evaluate anomalies (need at least 4 weeks)."}

    non_holiday = hist.loc[~hist["IsHoliday"].astype(bool), "Weekly_Sales"]
    baseline = non_holiday if len(non_holiday) >= 4 else hist["Weekly_Sales"]
    mean, std = float(baseline.mean()), float(baseline.std())

    anomalies = []
    for row in hist.itertuples():
        z = 0.0 if std == 0 else (row.Weekly_Sales - mean) / std
        if abs(z) >= z_threshold:
            anomalies.append(
                {
                    "date": row.Date.strftime("%Y-%m-%d"),
                    "weekly_sales": round(float(row.Weekly_Sales), 2),
                    "z_score": round(float(z), 2),
                    "direction": "spike" if z > 0 else "drop",
                    "is_holiday": bool(row.IsHoliday),
                }
            )

    return {
        "store": store,
        "dept": dept,
        "weeks_evaluated": len(hist),
        "baseline_mean": round(mean, 2),
        "baseline_std": round(std, 2),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


TOOL_FUNCTIONS = {
    "query_sales_history": query_sales_history,
    "forecast_demand": forecast_demand,
    "detect_anomalies": detect_anomalies,
}

TOOLS = [
    {
        "name": "query_sales_history",
        "description": (
            "Get recent historical weekly sales for a specific store and department, "
            "including store metadata and summary statistics. Call this first to "
            "understand the recent sales trend before forecasting or checking anomalies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer", "description": "Store number, e.g. 20"},
                "dept": {"type": "integer", "description": "Department number, e.g. 5"},
                "weeks": {
                    "type": "integer",
                    "description": "How many recent weeks of history to return (default 26).",
                },
            },
            "required": ["store", "dept"],
        },
    },
    {
        "name": "forecast_demand",
        "description": (
            "Run the trained demand forecasting model to predict upcoming weekly sales "
            "for a store and department over a given horizon. Uses known future calendar "
            "and economic features rather than assuming they stay constant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer", "description": "Store number, e.g. 20"},
                "dept": {"type": "integer", "description": "Department number, e.g. 5"},
                "horizon_weeks": {
                    "type": "integer",
                    "description": "How many upcoming weeks to forecast (default 8).",
                },
            },
            "required": ["store", "dept"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Check recent weekly sales for a store and department for statistically "
            "unusual spikes or drops (z-score based), separating holiday-driven swings "
            "from unexplained anomalies. Call this to sanity-check whether the recent "
            "trend is reliable before recommending a restock."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer", "description": "Store number, e.g. 20"},
                "dept": {"type": "integer", "description": "Department number, e.g. 5"},
                "lookback_weeks": {
                    "type": "integer",
                    "description": "How many recent weeks to evaluate (default 16).",
                },
                "z_threshold": {
                    "type": "number",
                    "description": "Z-score magnitude to flag as anomalous (default 2.0).",
                },
            },
            "required": ["store", "dept"],
        },
    },
]
