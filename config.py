"""Default configuration: data paths, constants and per-model hyperparameters."""

DATA_PATH = "data/concrete_thermal/thermal_conductivity_data.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"

MODEL_NAMES = ["linear", "ridge", "lasso", "svr", "rf", "xgb", "mlp"]
METRIC_NAMES = ["r2", "rmse", "mae", "mape", "max_rel_err"]
EVALUATIONS = ["single", "random", "v2"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 10

# Default hyperparameters per model. svr, mlp and rf values are selected by tune.py.
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

# Candidate values per hyperparameter for tune.py and main.py --tune; the search is the full cross product.
# svr's C and epsilon act on the standardized target (see main.build_model).
SEARCH_SPACES = {
    "svr": {
        "C": [1.0, 10.0, 100.0, 1000.0],
        "epsilon": [0.001, 0.005, 0.01, 0.05],
        "gamma": ["scale", 0.1, 0.3, 1.0, 3.0],
    },
    "mlp": {
        "hidden_layer_sizes": [(64, 64), (128, 128), (64, 64, 64), (256,)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-3, 3e-3],
    },
    "rf": {
        "n_estimators": [300, 600],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt"],
    },
}

GREATER_IS_BETTER = {"r2"}

# rf parallelizes internally (n_jobs=-1 above); during a search the
# outer loop over candidates owns the cores. Applied to the fit only, so it is
# not part of the candidate params.
SEARCH_ONLY_PARAMS = {"rf": {"n_jobs": 1}}
