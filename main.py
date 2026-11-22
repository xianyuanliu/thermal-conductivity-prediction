"""Effective thermal conductivity prediction from two-phase composite data.

Trains regression models on (K1, K2, V2) -> Keff and reports R^2, RMSE, MAE,
MAPE and max relative error. Run one model from the command line:

  python main.py --model svr              # single fixed 80/20 split
  python main.py --model svr --cv random  # shuffled 10-fold CV
  python main.py --model svr --cv v2      # leave-one-V2-out 10-fold CV

or run several at once from a YAML config (see configs/baseline.yaml):

  python main.py --config configs/baseline.yaml

Defaults (paths, constants, per-model hyperparameters) live in config.py; a
config file only needs to list what it overrides. Pass --results PATH (or set
`results:` in the config) to save metrics, upserted one row per configuration;
omit it to run without saving. For CV, per-fold metrics are written next to the
summary with a _folds suffix (results.csv -> results_folds.csv).
"""

import argparse
import importlib
import os

import numpy as np
import pandas as pd
import yaml
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

from config import (
    DATA_PATH,
    DEFAULT_PARAMS,
    EVALUATIONS,
    FEATURES,
    METRIC_NAMES,
    MODEL_NAMES,
    N_FOLDS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def resolve_params(name: str, params: dict | None = None) -> dict:
    """Returns the hyperparameters a model will actually be built with.

    config.DEFAULT_PARAMS[name] with params merged on top (overrides win), e.g.
    {"C": 100} for svr or {"n_estimators": 500} for rf. This is also what the
    results files record, so a row stays identifiable after the defaults change.
    """
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown model: {name}")
    return {**DEFAULT_PARAMS[name], **(params or {})}


def require(package: str, model: str):
    """Imports an optional foundation-model backend, or explains how to install it.

    tabpfn, tabfm and tabicl pull in PyTorch, so they are imported only when
    their model is built; the other models never need them. Only a missing
    top-level package gets the install hint; any other import failure inside an
    installed backend (a missing dependency, a broken binary) is re-raised as is.
    """
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as e:
        if e.name != package.partition(".")[0]:
            raise
        raise ModuleNotFoundError(
            f"Model {model!r} needs the {package!r} package; install it with pip install -r requirements-fm.txt"
        ) from e


def build_model(name: str, params: dict | None = None):
    """Returns a fresh estimator by name, with optional hyperparameter overrides.

    Scale-sensitive models get a StandardScaler on the features and, where the
    target's ~1e-3 scale matters (svr, mlp), on the target too.
    """
    p = resolve_params(name, params)
    if name == "linear":
        return LinearRegression(**p)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(**p))
    if name == "svr":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(**p)),
            transformer=StandardScaler(),
        )
    if name == "rf":
        return RandomForestRegressor(**p)
    if name == "xgb":
        return XGBRegressor(**p)
    if name == "mlp":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(**p)),
            transformer=StandardScaler(),
        )
    if name == "tabpfn":
        return require("tabpfn", name).TabPFNRegressor(**p)
    if name == "tabfm":
        tabfm = require("tabfm", name)
        model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")
        return tabfm.TabFMRegressor(model=model, **p)
    if name == "tabicl":
        return require("tabicl", name).TabICLRegressor(**p)
    raise ValueError(f"Unknown model: {name}")


def compute_metrics(y_true, y_pred):
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


def save_results(path: str, new_rows: pd.DataFrame, key_cols: list):
    """Upserts rows into a CSV: drops rows matching the new key, then appends.

    Args:
        path: CSV file to update (created if missing).
        new_rows: Rows to write; all share the same key values.
        key_cols: Columns identifying which existing rows to replace.
    """
    if os.path.exists(path):
        results = pd.read_csv(path)
        for col in key_cols:  # A file written before this column existed.
            if col not in results.columns:
                results[col] = pd.NA
        key = [new_rows.iloc[0][col] for col in key_cols]
        keep = ~(results[key_cols] == key).all(axis=1)
        results = pd.concat([results[keep], new_rows], ignore_index=True)
    else:
        results = new_rows
    results.to_csv(path, index=False)


def folds_path(results_path: str):
    """Derives the per-fold CSV path from the summary path: foo.csv -> foo_folds.csv."""
    root, ext = os.path.splitext(results_path)
    return f"{root}_folds{ext}"


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


def run_single_split(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    results_path: str | None = None,
    params: dict | None = None,
):
    """Trains on the fixed 80/20 split shared by every model and reports metrics.

    Saves to results_path when given (one row per model, upserted).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model(model_name, params)
    model.fit(X_train, y_train)

    test_metrics = compute_metrics(y_test, model.predict(X_test))
    metrics = {
        "model": model_name,
        "params": str(resolve_params(model_name, params)),
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

    if results_path:
        save_results(results_path, pd.DataFrame([metrics]), ["model", "params"])
        print(f"Metrics saved to {results_path}")


def run_cv(
    model_name: str,
    cv: str,
    X: pd.DataFrame,
    y: pd.Series,
    results_path: str | None = None,
    params: dict | None = None,
):
    """Runs 10-fold CV and reports per-fold metrics plus their mean and std.

    Saves the summary to results_path and per-fold metrics to its _folds
    variant when results_path is given.
    """
    params_repr = str(resolve_params(model_name, params))
    rows = []
    for fold, v2_value, train_idx, test_idx in iter_folds(X, cv):
        model = build_model(model_name, params)  # Fresh model per fold.
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_metrics = compute_metrics(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        rows.append(
            {"model": model_name, "params": params_repr, "cv": cv, "fold": fold, "v2_value": v2_value}
            | fold_metrics
        )
    folds = pd.DataFrame(rows)

    print(f"Model: {model_name} | CV: {cv} ({N_FOLDS} folds)")
    shown = (["fold", "v2_value"] if cv == "v2" else ["fold"]) + METRIC_NAMES
    print(folds[shown].to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    summary = {"model": model_name, "params": params_repr, "cv": cv}
    print("Summary (mean +/- std):")
    for name in METRIC_NAMES:
        summary[f"{name}_mean"] = folds[name].mean()
        summary[f"{name}_std"] = folds[name].std()
        print(f"  {name:12s}: {summary[f'{name}_mean']:.6g} +/- {summary[f'{name}_std']:.6g}")

    if results_path:
        folds_file = folds_path(results_path)
        save_results(folds_file, folds, ["model", "params", "cv"])
        save_results(results_path, pd.DataFrame([summary]), ["model", "params", "cv"])
        print(f"Fold metrics saved to {folds_file}, summary to {results_path}")


def run_config(config_path: str, X: pd.DataFrame, y: pd.Series):
    """Runs every model listed in a YAML config under one evaluation mode.

    The config keys are: 'evaluation' (single/random/v2, default single),
    'results' (optional CSV path), and 'models' (a map of model name to a dict
    of hyperparameter overrides; {} keeps the defaults).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    evaluation = config.get("evaluation", "single")
    if evaluation not in EVALUATIONS:
        raise ValueError(f"evaluation must be one of {EVALUATIONS}, got {evaluation!r}")

    results_path = config.get("results")
    models = config.get("models") or {}
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s) in config: {', '.join(unknown)}")
    if not models:
        raise ValueError("config lists no models to run")

    for name, params in models.items():
        if evaluation == "single":
            run_single_split(name, X, y, results_path=results_path, params=params)
        else:
            run_cv(name, evaluation, X, y, results_path=results_path, params=params)
        print()


def main():
    parser = argparse.ArgumentParser(description="Thermal conductivity prediction")
    parser.add_argument(
        "--config",
        help="YAML config to run several models at once; overrides --model/--cv/--results",
    )
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
        "--results",
        help="CSV path to save metrics to; if omitted, results are not saved",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    if args.config:
        run_config(args.config, X, y)
    elif args.cv:
        run_cv(args.model, args.cv, X, y, results_path=args.results)
    else:
        run_single_split(args.model, X, y, results_path=args.results)


if __name__ == "__main__":
    main()
