"""Train the forecasting model used by the Streamlit app and pickle it to disk.

Run this once (and again any time train.csv/features.csv change):

    python train_model.py
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import FEATURE_COLS, build_datasets, DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "sales_model.pkl"
METRICS_PATH = DATA_DIR / "models" / "metrics.json"
VALIDATION_PATH = DATA_DIR / "models" / "validation_predictions.csv"


def weighted_mae(is_holiday, actual, predicted):
    weights = np.where(is_holiday == 1, 5, 1)
    return float(np.sum(weights * np.abs(actual - predicted)) / np.sum(weights))

# NOTE: the notebook's best-performing model (Extra Trees, unbounded depth,
# n_estimators=100) pickles to ~5.7GB because Store/Dept give it enough
# splits to nearly memorize all 421k rows -- far too large for GitHub or
# Streamlit Community Cloud. Random Forest with bounded depth/leaf size
# gets within ~0.007 R2 of that model at a ~50MB file size, which is what
# we ship instead.
MODEL_PARAMS = dict(
    n_estimators=50,
    max_depth=18,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)


def main():
    print("Loading data and engineering features...")
    data_train, _ = build_datasets()

    X = data_train[FEATURE_COLS].copy()
    y = data_train["Weekly_Sales"].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print("Training Random Forest Regressor (size-constrained for deployment)...")
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    wmae = weighted_mae(X_valid["IsHoliday"].values, y_valid.values, y_pred)
    print(f"Validation MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}  WMAE: {wmae:.2f}")

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "WMAE": wmae, "n_validation_rows": len(y_valid)}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    # Keep the held-out predictions (with Store/Dept/Date) so the app can show
    # actual-vs-predicted for whichever store/dept the user picks, not just
    # the aggregate numbers above.
    validation_df = data_train.loc[X_valid.index, ["Store", "Dept", "Date"]].copy()
    validation_df["Actual_Sales"] = y_valid.values
    validation_df["Predicted_Sales"] = y_pred
    validation_df.to_csv(VALIDATION_PATH, index=False)
    print(f"Saved validation predictions to {VALIDATION_PATH}")

    print("Refitting on the full training set...")
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Saved model to {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
