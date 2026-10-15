"""Predicts effective thermal conductivity with a model trained on the full dataset.

The first run trains the model and caches it under models/; later runs load the cache.

Usage:
  python predict.py --k1 0.0018 --k2 0.0012 --v2 0.3
  python predict.py --input blind_validation_data.csv --output predictions.csv
  python predict.py --model rf --k1 ... --k2 ... --v2 ...   # non-default model
"""

import argparse
import hashlib
import os

import joblib
import pandas as pd

from config import DATA_PATH, FEATURES, MODEL_NAMES, TARGET
from main import build_model, resolve_params

MODELS_DIR = "models"
PREDICTION_COLUMN = "Keff-Pred"


def cache_key(name: str) -> dict:
    """Identifies what a cached model was trained with: its hyperparameters and the data."""
    with open(DATA_PATH, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()
    return {"params": resolve_params(name), "data_sha256": data_hash}


def load_or_train(name: str, retrain: bool):
    """Returns the fitted model and each feature's (min, max) training range.

    Loads the cached model unless retrain is set, no cache exists, or the
    hyperparameters or data have changed since it was cached.
    """
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    key = cache_key(name)
    if not retrain and os.path.exists(path):
        payload = joblib.load(path)
        if payload.get("key") == key:
            return payload["model"], payload["feature_ranges"]
        print(f"Cached {name} is out of date (hyperparameters or data changed); retraining")

    df = pd.read_csv(DATA_PATH)
    model = build_model(name)
    model.fit(df[FEATURES], df[TARGET])
    feature_ranges = {feature: (df[feature].min(), df[feature].max()) for feature in FEATURES}

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"key": key, "model": model, "feature_ranges": feature_ranges}, path)
    print(f"Trained {name} on the full dataset and cached it to {path}")
    return model, feature_ranges


def warn_if_out_of_range(X: pd.DataFrame, feature_ranges: dict):
    """Warns when inputs fall outside the training range (extrapolation)."""
    for feature, (low, high) in feature_ranges.items():
        outside = (X[feature] < low) | (X[feature] > high)
        if outside.any():
            print(
                f"Warning: {int(outside.sum())} {feature} value(s) outside the "
                f"training range [{low:g}, {high:g}]; predictions there may be "
                "unreliable."
            )


def main():
    parser = argparse.ArgumentParser(description="Predict effective thermal conductivity")
    parser.add_argument(
        "--model",
        default="svr",
        choices=MODEL_NAMES,
        help="Which model to use (default: svr, the best in evaluation)",
    )
    parser.add_argument("--k1", type=float, help="Thermal conductivity of phase 1")
    parser.add_argument("--k2", type=float, help="Thermal conductivity of phase 2")
    parser.add_argument("--v2", type=float, help="Volume fraction of phase 2")
    parser.add_argument("--input", help="CSV with K1, K2, V2 columns for batch prediction")
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Where to write batch predictions (default: predictions.csv)",
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain even if a cached model exists"
    )
    args = parser.parse_args()

    point = [args.k1, args.k2, args.v2]
    if args.input and any(value is not None for value in point):
        parser.error("Use either --input or --k1/--k2/--v2, not both")
    if not args.input and any(value is None for value in point):
        parser.error("Provide --k1, --k2 and --v2 together, or --input for batch mode")

    model, feature_ranges = load_or_train(args.model, args.retrain)

    if args.input:
        data = pd.read_csv(args.input)
        missing = [feature for feature in FEATURES if feature not in data.columns]
        if missing:
            parser.error(f"{args.input} is missing columns: {', '.join(missing)}")
        warn_if_out_of_range(data, feature_ranges)
        data[PREDICTION_COLUMN] = model.predict(data[FEATURES])
        data.to_csv(args.output, index=False)
        print(f"{len(data)} predictions written to {args.output}")
    else:
        X = pd.DataFrame([point], columns=FEATURES)
        warn_if_out_of_range(X, feature_ranges)
        print(f"Predicted Keff: {model.predict(X)[0]:.6g}")


if __name__ == "__main__":
    main()
