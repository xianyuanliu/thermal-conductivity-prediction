"""Effective thermal conductivity prediction from two-phase composite data.

Trains one regression model per run on (K1, K2, V2) -> Keff and reports
R^2, RMSE, MAE, MAPE and max relative error. Three independent experiments
are selected via --cv:

  python main.py --model svr              # single fixed 80/20 split
  python main.py --model svr --cv random  # shuffled 10-fold CV
  python main.py --model svr --cv v2      # leave-one-V2-out 10-fold CV

Each experiment writes to its own results file, so runs never interfere:
results.csv (single split), results_cv.csv and results_cv_folds.csv (CV).
Rerunning a configuration overwrites its previous rows.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATA_PATH = "Data-Keff-Num.csv"
RESULTS_PATH = "results.csv"
CV_RESULTS_PATH = "results_cv.csv"
CV_FOLDS_PATH = "results_cv_folds.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"
MODEL_NAMES = ["linear", "ridge", "lasso", "svr", "rf", "xgb", "mlp"]
METRIC_NAMES = ["r2", "rmse", "mae", "mape", "max_rel_err"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 10


def build_model(name: str):
    """Returns a fresh estimator by name.

    Scale-sensitive models get a StandardScaler on the features and, where
    needed, on the target.
    """
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=1e-6))
    if name == "svr":
        # Standardising both X and y is standard practice for SVR (Hsu, Chang &
        # Lin, "A Practical Guide to Support Vector Classification"): epsilon
        # and C are defined on the scale of y, and with y ~1e-3 the default
        # tube would swallow the whole target range. The scaler is fitted on
        # training data only and predictions are inverse-transformed, so there
        # is no leakage and metrics stay in original units.
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.01)
            ),
            transformer=StandardScaler(),
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )
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
    """Returns regression metrics keyed by METRIC_NAMES.

    Values are in original units; relative errors are fractions, not percent.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "max_rel_err": np.max(np.abs(y_pred - y_true) / np.abs(y_true)),
    }


def save_results(path: str, new_rows: pd.DataFrame, key_cols: list) -> None:
    """Upserts rows into a CSV: drops rows matching the new key, then appends.

    Args:
        path: CSV file to update (created if missing).
        new_rows: Rows to write; all share the same key values.
        key_cols: Columns identifying which existing rows to replace.
    """
    if os.path.exists(path):
        results = pd.read_csv(path)
        key = [new_rows.iloc[0][col] for col in key_cols]
        keep = ~(results[key_cols] == key).all(axis=1)
        results = pd.concat([results[keep], new_rows], ignore_index=True)
    else:
        results = new_rows
    results.to_csv(path, index=False)


def iter_folds(X: pd.DataFrame, cv: str):
    """Yields (fold, v2_value, train_idx, test_idx) as positional indices.

    "random" is a shuffled 10-fold split (v2_value is NaN). "v2" is
    leave-one-V2-out: the data holds exactly ten V2 values, so each fold holds
    out every sample of one volume fraction. Folds are ordered by V2; the
    first and last folds are true extrapolation beyond the training range.
    """
    if cv == "random":
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            yield fold, np.nan, train_idx, test_idx
    else:
        v2 = X["V2"].to_numpy()
        for fold, value in enumerate(np.sort(np.unique(v2))):
            test_mask = v2 == value
            yield fold, value, np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def run_single_split(model_name: str, X: pd.DataFrame, y: pd.Series, save: bool) -> None:
    """Trains on a fixed 80/20 split and reports train/test metrics."""
    # Fixed seed + split: every model sees the exact same train/test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model(model_name)
    model.fit(X_train, y_train)

    test_metrics = compute_metrics(y_test, model.predict(X_test))
    metrics = {
        "model": model_name,
        "train_r2": r2_score(y_train, model.predict(X_train)),
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }

    print(f"Model: {model_name}")
    print(f"Train R^2 : {metrics['train_r2']:.6f}")
    print(f"Test  R^2 : {metrics['test_r2']:.6f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.6g}")
    print(f"Test  MAE : {metrics['test_mae']:.6g}")
    print(f"Test  MAPE: {metrics['test_mape']:.4%}")
    print(f"Test  max rel err: {metrics['test_max_rel_err']:.4%}")

    if save:
        save_results(RESULTS_PATH, pd.DataFrame([metrics]), ["model"])
        print(f"Metrics saved to {RESULTS_PATH}")


def run_cv(model_name: str, cv: str, X: pd.DataFrame, y: pd.Series, save: bool) -> None:
    """Runs 10-fold CV and reports per-fold metrics plus their mean and std."""
    rows = []
    for fold, v2_value, train_idx, test_idx in iter_folds(X, cv):
        model = build_model(model_name)  # Fresh model per fold.
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_metrics = compute_metrics(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        rows.append(
            {"model": model_name, "cv": cv, "fold": fold, "v2_value": v2_value}
            | fold_metrics
        )
    folds = pd.DataFrame(rows)

    print(f"Model: {model_name} | CV: {cv} ({N_FOLDS} folds)")
    shown = (["fold", "v2_value"] if cv == "v2" else ["fold"]) + METRIC_NAMES
    print(folds[shown].to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    summary = {"model": model_name, "cv": cv}
    print("Summary (mean +/- std):")
    for name in METRIC_NAMES:
        summary[f"{name}_mean"] = folds[name].mean()
        summary[f"{name}_std"] = folds[name].std()
        print(f"  {name:12s}: {summary[f'{name}_mean']:.6g} +/- {summary[f'{name}_std']:.6g}")

    if save:
        save_results(CV_FOLDS_PATH, folds, ["model", "cv"])
        save_results(CV_RESULTS_PATH, pd.DataFrame([summary]), ["model", "cv"])
        print(f"Fold metrics saved to {CV_FOLDS_PATH}, summary to {CV_RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal conductivity prediction")
    parser.add_argument(
        "--model",
        default="linear",
        choices=MODEL_NAMES,
        help="Which model to train (default: linear)",
    )
    parser.add_argument(
        "--cv",
        choices=["random", "v2"],
        help="Run 10-fold CV instead of the single split: "
        "'random' is shuffled, 'v2' is leave-one-V2-out",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save metrics to the results files"
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    if args.cv:
        run_cv(args.model, args.cv, X, y, save=not args.no_save)
    else:
        run_single_split(args.model, X, y, save=not args.no_save)


if __name__ == "__main__":
    main()
