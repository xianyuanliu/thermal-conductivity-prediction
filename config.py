"""Default configuration for the thermal conductivity experiments.

Everything tunable lives here: data location, feature/target names, the
evaluation constants, and the default hyperparameters per model. A YAML file
passed to ``main.py --config`` overrides any model hyperparameters per run;
these values are the fallback when the YAML (or the CLI) leaves them out.
"""

DATA_PATH = "Data-Keff-Num.csv"
FEATURES = ["K1", "K2", "V2"]
TARGET = "Keff-Num"

MODEL_NAMES = ["linear", "ridge", "lasso", "svr", "rf", "xgb", "mlp", "tabpfn", "tabfm", "tabicl"]
METRIC_NAMES = ["r2", "rmse", "mae", "mape", "max_rel_err"]
EVALUATIONS = ["single", "random", "v2"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 10

# Default hyperparameters per model, passed straight to the underlying
# estimator. build_model merges a model's dict here with any overrides
# (overrides win). Notes on the less obvious choices:
#   svr: epsilon and C live on the scale of y (Hsu, Chang & Lin, "A Practical
#        Guide to Support Vector Classification"); with y ~1e-3 the target is
#        standardized before the SVR sees it, so these act on unit-variance y.
#        Values are from tune.py (leave-one-V2-out); C=1000 ranked marginally
#        higher but fits 2.5x slower with a worse MAPE.
#   mlp: weight init and learning rate assume O(1) outputs (LeCun et al.,
#        "Efficient BackProp"), so the target is standardized here too.
#   mlp, rf: also tune.py winners; neither beats svr.
#   tabpfn, tabfm, tabicl: pretrained tabular foundation models that predict by
#        in-context learning on the training set, so there is nothing to tune;
#        they normalize features and target internally.
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
    "tabpfn": {},
    "tabfm": {},
    "tabicl": {"random_state": RANDOM_STATE},
}
