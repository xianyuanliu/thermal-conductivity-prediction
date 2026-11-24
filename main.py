"""Effective conductivity prediction from two-phase composite data.

Trains regression models on (K1, K2, V2) to predict Keff and reports R^2, RMSE,
MAE, MAPE and max relative error. Run one model from the command line:

  python main.py --model svr              # single fixed 80/20 training and test split
  python main.py --model svr --cv random  # shuffled 10-fold cross-validation
  python main.py --model svr --cv v2      # leave-one-V2-out cross-validation (extrapolation test)

or run several models at once using a YAML config file:

  python main.py --config configs/concrete_thermal.yaml
"""

import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
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
    SEARCH_ONLY_PARAMS,
    SEARCH_SPACES,
    TARGET,
    TEST_SIZE,
)


def resolve_params(name: str, params: dict | None = None) -> dict:
    """Returns the model hyperparameters."""
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown model: {name}")
    return {**DEFAULT_PARAMS[name], **(params or {})}


def build_model(name: str, params: dict | None = None):
    """Returns a fresh estimator by name, with optional hyperparameter overrides.

    Scale-sensitive models get a StandardScaler on the features and, for svr and
    mlp, on the target.
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
    raise ValueError(f"Unknown model: {name}")


def compute_metrics(y_true, y_pred):
    """Returns regression metrics keyed by METRIC_NAMES."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "max_rel_err": np.max(np.abs(y_pred - y_true) / np.abs(y_true)),
    }


def load_config(path: str | None) -> dict:
    """Reads a YAML task config; None gives an empty one."""
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_data(args, config: dict):
    """Returns (data path, target column): the command line wins, then the config, then config.py."""
    return args.data or config.get("data", DATA_PATH), args.target or config.get("target", TARGET)


def model_params(parser: argparse.ArgumentParser, args, config: dict) -> dict | None:
    """Returns args.model's hyperparameter overrides from the config, or None for config.py defaults."""
    if not args.config:
        return None
    models = config.get("models") or {}
    if args.model not in models:
        parser.error(f"{args.model} is not listed under models in {args.config}")
    return models[args.model]


def task_output_dir(data_path: str) -> str:
    """Maps data/<task>/<file>.csv to outputs/<task>, so each dataset writes to its own folder."""
    return os.path.join("outputs", os.path.basename(os.path.dirname(os.path.abspath(data_path))))


def add_data_args(parser: argparse.ArgumentParser):
    """Adds --data and --target, which override the config and config.py defaults."""
    parser.add_argument("--data", help=f"Training data CSV (default: from --config, else {DATA_PATH})")
    parser.add_argument("--target", help=f"Target column (default: from --config, else {TARGET})")


def save_results(path: str, new_rows: pd.DataFrame, key_cols: list):
    """Saves rows into a CSV: drops rows matching the new key, then appends."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        results = pd.read_csv(path)
        key = [new_rows.iloc[0][col] for col in key_cols]
        keep = ~(results[key_cols] == key).all(axis=1)
        results = pd.concat([results[keep], new_rows], ignore_index=True)
    else:
        results = new_rows
    results.to_csv(path, index=False)


def add_suffix(path: str, suffix: str) -> str:
    """Inserts a suffix before the extension: foo.csv -> foo_<suffix>.csv."""
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext}"


def iter_folds(X: pd.DataFrame, cv: str):
    """Yields (fold, v2_value, train_idx, test_idx) as positional indices.

    "random" is a shuffled 10-fold split (v2_value is NaN). "v2" is
    leave-one-V2-out: the data holds exactly ten V2 values, so each fold holds
    out every sample of one volume fraction. Folds are ordered by V2.
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


def candidates(space: dict):
    """Yields every combination in a search space as a params dict."""
    keys = list(space)
    for values in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, values))


def val_score(name: str, params: dict, X_fit, y_fit, X_val, y_val) -> float:
    """Returns the validation R^2 of one candidate fitted on (X_fit, y_fit)."""
    # Silence MLP max_iter warnings; set here because workers don't inherit filters.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    model = build_model(name, {**params, **SEARCH_ONLY_PARAMS.get(name, {})})
    model.fit(X_fit, y_fit)
    return r2_score(y_val, model.predict(X_val))


def select_params(name: str, X: pd.DataFrame, y: pd.Series, cv: str, params: dict | None = None):
    """Returns the SEARCH_SPACES[name] candidate with the best validation R^2, chosen from (X, y) alone.

    The validation set is carved out of (X, y): one randomly chosen V2 value for
    "v2", a random TEST_SIZE share otherwise. Callers pass training data only, so
    the test data never influences the choice. Models without a search space keep params.
    """
    if name not in SEARCH_SPACES:
        return params
    if cv == "v2":
        v2 = X["V2"].to_numpy()
        val_mask = v2 == np.random.default_rng(RANDOM_STATE).choice(np.unique(v2))
    else:
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[train_test_split(np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE)[1]] = True
    trials = [{**(params or {}), **candidate} for candidate in candidates(SEARCH_SPACES[name])]
    scores = Parallel(n_jobs=-1)(
        delayed(val_score)(name, trial, X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]) for trial in trials
    )
    return trials[int(np.argmax(scores))]


def run_single_split(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str | None = None,
    params: dict | None = None,
    tune: bool = False,
):
    """Trains on the fixed 80/20 split; tune selects hyperparameters on the training part only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if tune:
        params = select_params(model_name, X_train, y_train, "random", params)

    model = build_model(model_name, params)
    model.fit(X_train, y_train)

    test_metrics = compute_metrics(y_test, model.predict(X_test))
    metrics = {
        "model": model_name,
        "params": str(resolve_params(model_name, params)),
        "train_r2": r2_score(y_train, model.predict(X_train)),
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }

    print(f"Model: {model_name}" + (f" | tuned: {params}" if tune else ""))
    print(f"Train R^2 : {metrics['train_r2']:.6f}")
    print(f"Test  R^2 : {metrics['test_r2']:.6f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.6g}")
    print(f"Test  MAE : {metrics['test_mae']:.6g}")
    print(f"Test  MAPE: {metrics['test_mape']:.4%}")
    print(f"Test  max rel err: {metrics['test_max_rel_err']:.4%}")

    if output_path:
        # Tuned runs go to their own file so they never replace untuned rows.
        output_path = add_suffix(output_path, "tuned") if tune else output_path
        save_results(output_path, pd.DataFrame([metrics]), ["model"] if tune else ["model", "params"])
        print(f"Metrics saved to {output_path}")


def run_cv(
    model_name: str,
    cv: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str | None = None,
    params: dict | None = None,
    tune: bool = False,
):
    """Runs CV and reports per-fold metrics plus their mean and std.

    tune selects hyperparameters inside each fold from its training part only.
    """
    params_repr = "tuned" if tune else str(resolve_params(model_name, params))
    rows = []
    for fold, v2_value, train_idx, test_idx in iter_folds(X, cv):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        fold_params = select_params(model_name, X_train, y_train, cv, params) if tune else params
        model = build_model(model_name, fold_params)
        model.fit(X_train, y_train)
        fold_metrics = compute_metrics(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        fold_repr = str(resolve_params(model_name, fold_params)) if tune else params_repr
        rows.append(
            {"model": model_name, "params": fold_repr, "cv": cv, "fold": fold, "v2_value": v2_value}
            | fold_metrics
        )
    folds = pd.DataFrame(rows)

    print(f"Model: {model_name} | CV: {cv} ({N_FOLDS} folds)" + (" | tuned per fold" if tune else ""))
    shown = (["fold", "v2_value"] if cv == "v2" else ["fold"]) + METRIC_NAMES + (["params"] if tune else [])
    print(folds[shown].to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    summary = {"model": model_name, "params": params_repr, "cv": cv}
    print("Summary (mean +/- std):")
    for name in METRIC_NAMES:
        summary[f"{name}_mean"] = folds[name].mean()
        summary[f"{name}_std"] = folds[name].std()
        print(f"  {name:12s}: {summary[f'{name}_mean']:.6g} +/- {summary[f'{name}_std']:.6g}")

    if output_path:
        output_path = add_suffix(output_path, "tuned") if tune else output_path
        folds_file = add_suffix(output_path, "folds")
        key_cols = ["model", "cv"] if tune else ["model", "params", "cv"]
        save_results(folds_file, folds, key_cols)
        save_results(output_path, pd.DataFrame([summary]), key_cols)
        print(f"Fold metrics saved to {folds_file}, summary to {output_path}")


def run_config(config: dict, X: pd.DataFrame, y: pd.Series, tune: bool = False):
    """Runs every model listed in a task config under one evaluation mode.

    The config keys are: 'data' and 'target' (read by main), 'evaluation'
    (single/random/v2, default single), 'output' (optional CSV path), and
    'models' (a map of model name to a dict of hyperparameter overrides. {}
    keeps the defaults).
    """
    evaluation = config.get("evaluation", "single")
    if evaluation not in EVALUATIONS:
        raise ValueError(f"evaluation must be one of {EVALUATIONS}, got {evaluation!r}")

    output_path = config.get("output")
    models = config.get("models") or {}
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s) in config: {', '.join(unknown)}")
    if not models:
        raise ValueError("config lists no models to run")

    for name, params in models.items():
        if evaluation == "single":
            run_single_split(name, X, y, output_path=output_path, params=params, tune=tune)
        else:
            run_cv(name, evaluation, X, y, output_path=output_path, params=params, tune=tune)
        print()


def main():
    parser = argparse.ArgumentParser(description="Effective conductivity prediction")
    parser.add_argument(
        "--config",
        help="YAML config file listing models to run; if omitted, a single model is run",
    )
    parser.add_argument(
        "--model",
        default="linear",
        choices=MODEL_NAMES,
        help="Model to run (default: linear)",
    )
    parser.add_argument(
        "--cv",
        choices=["random", "v2"],
        help="Run cross-validation: 'random' is shuffled 10-fold, 'v2' is leave-one-V2-out",
    )
    parser.add_argument(
        "--output", help="CSV path to save results. If omitted, results are not saved"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Select svr/mlp/rf hyperparameters inside each fold from training data only; saves to *_tuned.csv",
    )
    add_data_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    data_path, target = resolve_data(args, config)
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[target]

    if args.config:
        run_config(config, X, y, args.tune)
    elif args.cv:
        run_cv(args.model, args.cv, X, y, output_path=args.output, tune=args.tune)
    else:
        run_single_split(args.model, X, y, output_path=args.output, tune=args.tune)


if __name__ == "__main__":
    main()
