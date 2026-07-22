import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATA_PATH = "Data-Keff-Num.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL = "linear"


def build_model(name: str):
    """Return an estimator by name. Scale-sensitive models get a StandardScaler."""
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=1e-6, random_state=RANDOM_STATE))
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
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Fixed seed + split: every model sees the exact same train/test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model(MODEL)
    model.fit(X_train, y_train)

    print(f"Model: {MODEL}")
    print(f"Train R^2: {r2_score(y_train, model.predict(X_train)):.6f}")
    print(f"Test  R^2: {r2_score(y_test, model.predict(X_test)):.6f}")


if __name__ == "__main__":
    main()
