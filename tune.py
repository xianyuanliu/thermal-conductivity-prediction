"""Hyperparameter search under the leave-one-V2-out protocol.

Ranks every candidate in a model's search space by its mean metric across the
ten leave-one-V2-out folds -- the same folds main.py --cv v2 reports on, so a
winner here is a winner at extrapolation, not just at interpolation.

  python tune.py                          # search svr, mlp and rf
  python tune.py --models svr             # one model only
  python tune.py --metric max_rel_err     # rank by worst-case error instead

The full ranking is written to tuning.csv; the best row per model is printed
ready to paste into config.DEFAULT_PARAMS.

Note on the score: tuning and reporting share the same folds, so the winning
score is optimistic (the search saw every fold it is graded on). Treat it as a
ranking, not as an unbiased estimate of error on new points.
"""

import argparse
import itertools
import math
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from config import DATA_PATH, FEATURES, METRIC_NAMES, TARGET
from main import build_model, compute_metrics, iter_folds

# Candidate values per hyperparameter; the search is the full cross product.
# Chosen around the current config.py defaults, one order of magnitude either
# way. For svr, C and epsilon act on a standardized target (see config.py), so
# epsilon is a fraction of a standard deviation rather than of ~1e-3.
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

# rf parallelizes internally (n_jobs=-1 in config.py); during the search the
# outer loop over candidates owns the cores. Applied to the fit only, so it is
# not part of the candidate params.
SEARCH_ONLY_PARAMS = {"rf": {"n_jobs": 1}}

RESULTS_PATH = "tuning.csv"


def candidates(space: dict):
    """Yields every combination in a search space as a params dict."""
    keys = list(space)
    for values in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, values))


def score_candidate(name: str, params: dict, X: pd.DataFrame, y: pd.Series, folds: list):
    """Returns mean metrics for one candidate across the given folds.

    A fresh model is fitted per fold, exactly as main.run_cv does, so the
    numbers here are comparable to the ones in results_v2.csv.
    """
    # MLPRegressor warns per fit when it hits max_iter; the search is about
    # relative ranking, so that is noise. Set here because workers do not
    # inherit the parent's filters.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    fit_params = {**params, **SEARCH_ONLY_PARAMS.get(name, {})}
    rows = []
    for _, _, train_idx, test_idx in folds:
        model = build_model(name, fit_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        rows.append(compute_metrics(y.iloc[test_idx], model.predict(X.iloc[test_idx])))
    return {"model": name, "params": str(params), **pd.DataFrame(rows).mean().to_dict()}


def search(name: str, X: pd.DataFrame, y: pd.Series, metric: str, n_jobs: int):
    """Scores every candidate for one model in parallel, ranked best first."""
    folds = list(iter_folds(X, "v2"))
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(score_candidate)(name, params, X, y, folds)
        for params in candidates(SEARCH_SPACES[name])
    )
    return pd.DataFrame(rows).sort_values(
        metric, ascending=metric not in GREATER_IS_BETTER, ignore_index=True
    )


def main():
    parser = argparse.ArgumentParser(description="Leave-one-V2-out hyperparameter search")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SEARCH_SPACES),
        choices=list(SEARCH_SPACES),
        help="Which models to search (default: all with a search space)",
    )
    parser.add_argument(
        "--metric",
        default="r2",
        choices=METRIC_NAMES,
        help="Metric to rank candidates by (default: r2)",
    )
    parser.add_argument(
        "--results", default=RESULTS_PATH, help=f"Where to write the ranking (default: {RESULTS_PATH})"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Candidates to score in parallel (default: all cores)"
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]

    rankings = []
    for name in args.models:
        total = math.prod(len(values) for values in SEARCH_SPACES[name].values())
        print(f"Searching {name} ({total} candidates)", flush=True)
        rankings.append(search(name, X, y, args.metric, args.n_jobs))
        best = rankings[-1].iloc[0]
        print(f"  best {args.metric}={best[args.metric]:.6g}  {best['params']}\n", flush=True)

    ranked = pd.concat(rankings, ignore_index=True)
    ranked.to_csv(args.results, index=False)

    best = pd.DataFrame([ranking.iloc[0] for ranking in rankings]).sort_values(
        args.metric, ascending=args.metric not in GREATER_IS_BETTER
    )
    print(f"Best per model (leave-one-V2-out, ranked by {args.metric}):")
    print(best[["model", *METRIC_NAMES, "params"]].to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print()
    print(f"Winner: {best.iloc[0]['model']} with {best.iloc[0]['params']}")
    print(f"Full ranking written to {args.results}")


if __name__ == "__main__":
    main()
