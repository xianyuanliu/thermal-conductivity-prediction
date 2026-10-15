"""Default configuration: data paths, constants and per-model hyperparameters."""

DATA_PATH = "thermal_conductivity_data.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"

MODEL_NAMES = ["linear", "ridge", "lasso", "svr", "rf", "xgb", "mlp"]
METRIC_NAMES = ["r2", "rmse", "mae", "mape", "max_rel_err"]
EVALUATIONS = ["single", "random", "v2"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 10

# Default hyperparameters per model. svr, mlp and rf values are extracted from tune.py.
DEFAULT_PARAMS = {
    "linear": {},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 1e-6},
    "svr": {"kernel": "rbf", "C": 100.0, "epsilon": 0.001, "gamma": "scale"},
    "rf": {
        "n_estimators": 600,
        "max_depth": 20,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "xgb": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "mlp": {
        "hidden_layer_sizes": (128, 128),
        "learning_rate_init": 0.003,
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
    },
}
