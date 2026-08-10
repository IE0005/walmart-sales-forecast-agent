"""Compare candidate models on the same train/valid split used by train_model.py.

This is an offline/exploratory script -- xgboost, lightgbm, and catboost are
NOT in requirements.txt because the deployed app only ever loads the single
pickled Random Forest. Run this locally (once, or whenever you want to
re-check the comparison) and it writes models/model_comparison.csv, which the
Streamlit app reads and displays as a static reference table.

    python compare_models.py
"""
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from pipeline import FEATURE_COLS, build_datasets, DATA_DIR
from train_model import MODEL_PARAMS as DEPLOYED_RF_PARAMS, weighted_mae

OUT_PATH = DATA_DIR / "models" / "model_comparison.csv"


def pickled_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        joblib.dump(model, path, compress=3)
        return os.path.getsize(path) / 1e6
    finally:
        os.remove(path)


def main():
    print("Loading data and engineering features...")
    data_train, _ = build_datasets()
    X = data_train[FEATURE_COLS].copy()
    y = data_train["Weekly_Sales"].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    candidates = {
        "Random Forest (deployed)": RandomForestRegressor(**DEPLOYED_RF_PARAMS),
        "Extra Trees (notebook best, unconstrained)": ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=8, learning_rate=0.05,
            loss_function="RMSE", random_seed=42, verbose=False,
        ),
    }

    results = []
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_valid, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_valid, y_pred)),
            "R2": r2_score(y_valid, y_pred),
            "WMAE": weighted_mae(X_valid["IsHoliday"].values, y_valid.values, y_pred),
            "Pickled_Size_MB": pickled_size_mb(model),
        })

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved comparison to {OUT_PATH}\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
