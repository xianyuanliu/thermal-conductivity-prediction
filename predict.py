"""User-facing prediction of effective thermal conductivity.

Predicts Keff for given (K1, K2, V2) inputs using a model trained on the full
dataset (evaluation belongs to main.py; the deployed model has no reason to
hold data back). The first call trains and caches the model under models/;
later calls load the cache.

Usage:
  python predict.py --k1 0.0018 --k2 0.0012 --v2 0.3
  python predict.py --input new_points.csv --output predictions.csv
  python predict.py --model rf --k1 ... --k2 ... --v2 ...   # non-default model
"""

import argparse
import os

import joblib
import pandas as pd

from main import DATA_PATH, FEATURES, MODEL_NAMES, TARGET, build_model

MODELS_DIR = "models"
PREDICTION_COLUMN = "Keff-Pred"


def load_or_train(name: str, retrain: bool):
    """Returns (model, feature_ranges), training and caching on a cache miss.

    Args:
        name: Model name understood by build_model.
        retrain: Ignore any cached model and train from scratch.

    Returns:
        The fitted model and a dict mapping each feature to its (min, max)
        range in the training data.
    """
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not retrain and os.path.exists(path):
        payload = joblib.load(path)
        return payload["model"], payload["feature_ranges"]

    df = pd.read_csv(DATA_PATH)
    model = build_model(name)
    model.fit(df[FEATURES], df[TARGET])
    feature_ranges = {feature: (df[feature].min(), df[feature].max()) for feature in FEATURES}

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": model, "feature_ranges": feature_ranges}, path)
    print(f"Trained {name} on the full dataset and cached it to {path}")
    return model, feature_ranges


def warn_if_out_of_range(X: pd.DataFrame, feature_ranges: dict) -> None:
    """Warns when inputs fall outside the training range (extrapolation)."""
    for feature, (low, high) in feature_ranges.items():
        outside = (X[feature] < low) | (X[feature] > high)
        if outside.any():
            print(
                f"Warning: {int(outside.sum())} {feature} value(s) outside the "
                f"training range [{low:g}, {high:g}]; predictions there may be "
                "unreliable."
            )


def main() -> None:
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
