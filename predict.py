"""Predicts effective conductivity with a model trained on the full dataset.

The first run trains the model and caches it under outputs/<task>/models/;
later runs load the cached model.

Usage:
  python predict.py --k1 0.0018 --k2 0.0012 --v2 0.3
  python predict.py --input data/concrete_thermal/blind_validation_data.csv
  python predict.py --config configs/soil_electrical.yaml --k1 0.1 --k2 1e-6 --v2 0.6
  python predict.py --model rf --k1 ... --k2 ... --v2 ...   # non-default model
"""

import argparse
import hashlib
import os

import joblib
import pandas as pd

from config import FEATURES, MODEL_NAMES
from main import add_data_args, build_model, load_config, model_params, resolve_data, resolve_params, task_output_dir


def cache_key(name: str, data_path: str, params: dict | None = None) -> dict:
    """Get the key for a cached model, including hyperparameters and the data used."""
    with open(data_path, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()
    return {"params": resolve_params(name, params), "data_sha256": data_hash}


def load_or_train(
    name: str,
    data_path: str,
    target: str,
    params: dict | None = None,
    retrain: bool = False,
    models_dir: str | None = None,
):
    """Returns the fitted model and each feature's (min, max) training range.

    Loads the cached model unless retrain is set, no cache exists, or the
    hyperparameters or data have changed since it was cached. models_dir
    defaults to outputs/<task>/models.
    """
    models_dir = models_dir or os.path.join(task_output_dir(data_path), "models")
    path = os.path.join(models_dir, f"{name}.joblib")
    key = cache_key(name, data_path, params)
    if not retrain and os.path.exists(path):
        payload = joblib.load(path)
        if payload.get("key") == key:
            return payload["model"], payload["feature_ranges"]
        print(f"Cached {name} is out of date (hyperparameters or data changed); retraining")

    df = pd.read_csv(data_path)
    model = build_model(name, params)
    model.fit(df[FEATURES], df[target])
    feature_ranges = {feature: (df[feature].min(), df[feature].max()) for feature in FEATURES}

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump({"key": key, "model": model, "feature_ranges": feature_ranges}, path)
    print(f"Trained {name} on the full dataset and cached it to {path}")
    return model, feature_ranges


def warn_if_out_of_range(X: pd.DataFrame, feature_ranges: dict):
    """Warns when inputs fall outside the training range."""
    for feature, (low, high) in feature_ranges.items():
        outside = (X[feature] < low) | (X[feature] > high)
        if outside.any():
            print(
                f"Warning: {int(outside.sum())} {feature} value(s) outside the "
                f"training range [{low:g}, {high:g}]; predictions there may be "
                "unreliable."
            )


def main():
    parser = argparse.ArgumentParser(description="Predict effective conductivity")
    parser.add_argument(
        "--model",
        default="svr",
        choices=MODEL_NAMES,
        help="Model to use (default: svr, the best in evaluation)",
    )
    parser.add_argument("--k1", type=float, help="Conductivity of phase 1")
    parser.add_argument("--k2", type=float, help="Conductivity of phase 2")
    parser.add_argument("--v2", type=float, help="Volume fraction of phase 2")
    parser.add_argument("--input", help="CSV with K1, K2, V2 columns")
    parser.add_argument(
        "--output", help="CSV path to save predictions (default: outputs/<task>/predictions.csv)"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain even if a cached model exists"
    )
    parser.add_argument("--config", help="Task YAML config to read data, target and hyperparameters from")
    add_data_args(parser)
    args = parser.parse_args()

    point = [args.k1, args.k2, args.v2]
    if args.input and any(value is not None for value in point):
        parser.error("Use either --input or --k1/--k2/--v2, not both")
    if not args.input and any(value is None for value in point):
        parser.error("Provide --k1, --k2 and --v2 together, or --input for batch mode")

    config = load_config(args.config)
    data_path, target = resolve_data(args, config)
    params = model_params(parser, args, config)
    model, feature_ranges = load_or_train(args.model, data_path, target, params, args.retrain)

    if args.input:
        data = pd.read_csv(args.input)
        missing = [feature for feature in FEATURES if feature not in data.columns]
        if missing:
            parser.error(f"{args.input} is missing columns: {', '.join(missing)}")
        warn_if_out_of_range(data, feature_ranges)
        data["Keff-Pred"] = model.predict(data[FEATURES])
        output = args.output or os.path.join(task_output_dir(data_path), "predictions.csv")
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        data.to_csv(output, index=False)
        print(f"{len(data)} predictions written to {output}")
    else:
        X = pd.DataFrame([point], columns=FEATURES)
        warn_if_out_of_range(X, feature_ranges)
        print(f"Predicted Keff: {model.predict(X)[0]:.6g}")


if __name__ == "__main__":
    main()
