"""Shared data loading / feature engineering for the Walmart sales forecasting app.

This mirrors the feature engineering performed in the original exploration
notebook so that the offline training script and the Streamlit app stay in
sync.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# Feature columns used to train the model (matches the notebook's best run).
FEATURE_COLS = [
    "Store", "Dept", "IsHoliday", "Size", "Week", "Type", "Year", "Day",
    "Month", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "SuperBowlWeek", "LaborDay", "Thanksgiving", "Christmas",
    "MarkdownsSum",
]

TYPE_MAP = {"A": 1, "B": 2, "C": 3}


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SuperBowlWeek"] = (df["Week"] == 6).astype(int)
    df["LaborDay"] = (df["Week"] == 36).astype(int)
    df["Thanksgiving"] = (df["Week"] == 47).astype(int)
    df["Christmas"] = (df["Week"] == 52).astype(int)
    df["MarkdownsSum"] = df[MARKDOWN_COLS].sum(axis=1)
    df["Type"] = df["Type"].map(TYPE_MAP)
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


def load_raw():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    features = pd.read_csv(DATA_DIR / "features.csv")
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    return train, test, features, stores


def build_datasets():
    """Load raw CSVs and return fully feature-engineered train/test frames.

    Returns
    -------
    data_train : historical rows with known Weekly_Sales (for charting + training)
    data_test  : future rows (test.csv date range) with engineered features,
                 used to produce the forward-looking forecast in the app.
    """
    train, test, features, stores = load_raw()

    features = features.copy()
    features[MARKDOWN_COLS] = features[MARKDOWN_COLS].fillna(0)
    features["CPI"] = features["CPI"].fillna(features["CPI"].median())
    features["Unemployment"] = features["Unemployment"].fillna(features["Unemployment"].median())

    feature_store = features.merge(stores, on="Store")
    feature_store["Date"] = pd.to_datetime(feature_store["Date"])
    feature_store["Year"] = feature_store["Date"].dt.year
    feature_store["Month"] = feature_store["Date"].dt.month
    feature_store["Week"] = feature_store["Date"].dt.isocalendar().week.astype(int)
    feature_store["Day"] = feature_store["Date"].dt.day

    train = train.copy()
    test = test.copy()
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train_df = train.merge(feature_store, how="inner", on=["Store", "Date", "IsHoliday"])
    test_df = test.merge(feature_store, how="inner", on=["Store", "Date", "IsHoliday"])

    data_train = _engineer(train_df)
    data_test = _engineer(test_df)

    return data_train, data_test
