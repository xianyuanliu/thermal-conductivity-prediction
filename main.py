import argparse
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
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
        # Target is ~1e-3 in magnitude, so scale y as well (scale-sensitive model).
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

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model": args.model,
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
    }

    print(f"Model: {metrics['model']}")
    print(f"Train R^2 : {metrics['train_r2']:.6f}")
    print(f"Test  R^2 : {metrics['test_r2']:.6f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.6g}")
    print(f"Test  MAE : {metrics['test_mae']:.6g}")

    if not args.no_save:
        # Keep one row per model: rerunning a model overwrites its previous result.
        row = pd.DataFrame([metrics])
        if os.path.exists(RESULTS_PATH):
            results = pd.read_csv(RESULTS_PATH)
            results = results[results["model"] != args.model]
            results = pd.concat([results, row], ignore_index=True)
        else:
            results = row
        results.to_csv(RESULTS_PATH, index=False)
        print(f"Metrics saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
