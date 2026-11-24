"""Hyperparameter search under the leave-one-V2-out protocol.

Ranks every candidate in a model's search space by its mean metric across the
ten leave-one-V2-out folds -- the same folds main.py --cv v2 reports on, so a
winner here is a winner at extrapolation, not just at interpolation.

  python tune.py                          # search svr, mlp and rf
  python tune.py --models svr             # one model only
  python tune.py --metric max_rel_err     # rank by worst-case error instead

The full ranking is written to outputs/<task>/tuning.csv; the best row per model is printed
ready to paste into config.DEFAULT_PARAMS.

Note on the score: tuning and reporting share the same folds, so the winning
score is optimistic (the search saw every fold it is graded on). Treat it as a
ranking, not as an unbiased estimate of error on new points; main.py --tune
gives that estimate by tuning inside each fold on training data only.
"""

import argparse
import math
import os
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from config import FEATURES, GREATER_IS_BETTER, METRIC_NAMES, SEARCH_ONLY_PARAMS, SEARCH_SPACES
from main import add_data_args, build_model, candidates, compute_metrics, iter_folds, resolve_data, task_output_dir


def score_candidate(name: str, params: dict, X: pd.DataFrame, y: pd.Series, folds: list):
    """Returns mean metrics for one candidate, fitted per fold as in main.run_cv."""
    # Silence MLP max_iter warnings; set here because workers don't inherit filters.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    fit_params = {**params, **SEARCH_ONLY_PARAMS.get(name, {})}
    rows = []
    for _, _, train_idx, test_idx in folds:
        model = build_model(name, fit_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        rows.append(compute_metrics(y.iloc[test_idx], model.predict(X.iloc[test_idx])))
    return {"model": name, "params": str(params), **pd.DataFrame(rows).mean().to_dict()}


def search(
    name: str, X: pd.DataFrame, y: pd.Series, metric: str, n_jobs: int, space: dict | None = None
):
    """Scores every candidate for one model in parallel, ranked best first.

    space overrides SEARCH_SPACES[name].
    """
    space = SEARCH_SPACES[name] if space is None else space
    folds = list(iter_folds(X, "v2"))
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(score_candidate)(name, params, X, y, folds)
        for params in candidates(space)
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
        "--output", help="CSV path to save the ranking (default: outputs/<task>/tuning.csv)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Candidates to score in parallel (default: all cores)"
    )
    add_data_args(parser)
    args = parser.parse_args()

    data_path, target = resolve_data(args, {})
    output = args.output or os.path.join(task_output_dir(data_path), "tuning.csv")
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[target]

    rankings = []
    for name in args.models:
        total = math.prod(len(values) for values in SEARCH_SPACES[name].values())
        print(f"Searching {name} ({total} candidates)", flush=True)
        rankings.append(search(name, X, y, args.metric, args.n_jobs))
        best = rankings[-1].iloc[0]
        print(f"  best {args.metric}={best[args.metric]:.6g}  {best['params']}\n", flush=True)

    ranked = pd.concat(rankings, ignore_index=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    ranked.to_csv(output, index=False)

    best = pd.DataFrame([ranking.iloc[0] for ranking in rankings]).sort_values(
        args.metric, ascending=args.metric not in GREATER_IS_BETTER
    )
    print(f"Best per model (leave-one-V2-out, ranked by {args.metric}):")
    print(best[["model", *METRIC_NAMES, "params"]].to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print()
    print(f"Winner: {best.iloc[0]['model']} with {best.iloc[0]['params']}")
    print(f"Full ranking written to {output}")


if __name__ == "__main__":
    main()
