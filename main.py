import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATA_PATH = "Data-Keff-Num.csv"
RESULTS_PATH = "results.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_model(name: str):
    """Return an estimator by name. Scale-sensitive models get a StandardScaler."""
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=1e-6))
    if name == "svr":
        # Standardising both X and y is standard practice for SVR (Hsu, Chang &
        # Lin, "A Practical Guide to Support Vector Classification"): epsilon and
        # C are defined on the scale of y, and with y ~1e-3 the default tube
        # would swallow the whole target range. The scaler is fitted on training
        # data only and predictions are inverse-transformed, so there is no
        # leakage and metrics stay in original units.
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.01)),
            transformer=StandardScaler(),
        )
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "xgb":
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "mlp":
        # Scaling X and y is standard practice for neural networks (LeCun et
        # al., "Efficient BackProp"): weight initialisation and the default
        # learning rate assume O(1) outputs, so a ~1e-3 target trains poorly.
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
            transformer=StandardScaler(),
        )
    raise ValueError(f"Unknown model: {name}")


def compute_metrics(y_true, y_pred) -> dict:
    """Regression metrics in original units; relative errors as fractions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "max_rel_err": np.max(np.abs(y_pred - y_true) / np.abs(y_true)),
    }


def save_results(path: str, new_rows: pd.DataFrame, key_cols: list):
    """Upsert: drop existing rows matching the new key, then append the new rows."""
    if os.path.exists(path):
        results = pd.read_csv(path)
        key = [new_rows.iloc[0][col] for col in key_cols]
        mask = (results[key_cols] == key).all(axis=1)
        results = pd.concat([results[~mask], new_rows], ignore_index=True)
    else:
        results = new_rows
    results.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Thermal conductivity prediction")
    parser.add_argument(
        "--model",
        default="linear",
        choices=["linear", "ridge", "lasso", "svr", "rf", "xgb", "mlp"],
        help="Which model to train (default: linear)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not append metrics to results.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Fixed seed + split: every model sees the exact same train/test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model(args.model)
    model.fit(X_train, y_train)

    train_metrics = compute_metrics(y_train, model.predict(X_train))
    test_metrics = compute_metrics(y_test, model.predict(X_test))
    metrics = {
        "model": args.model,
        "train_r2": train_metrics["r2"],
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }

    print(f"Model: {metrics['model']}")
    print(f"Train R^2 : {metrics['train_r2']:.6f}")
    print(f"Test  R^2 : {metrics['test_r2']:.6f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.6g}")
    print(f"Test  MAE : {metrics['test_mae']:.6g}")
    print(f"Test  MAPE: {metrics['test_mape']:.4%}")
    print(f"Test  max rel err: {metrics['test_max_rel_err']:.4%}")

    if not args.no_save:
        # Keep one row per model: rerunning a model overwrites its previous result.
        save_results(RESULTS_PATH, pd.DataFrame([metrics]), ["model"])
        print(f"Metrics saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
