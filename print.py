"""Bar charts of cross-validation results: one figure per metric.

Reads a CV summary CSV written by main.py (e.g. results_random.csv) and draws
one bar chart per metric, with each model's mean as the bar and its std as the
error bar. Figures are written to outputs/concrete_thermal/figures/ unless --output is given.

Usage:
  python print.py                              # plots outputs/concrete_thermal/results_v2.csv
  python print.py --input outputs/concrete_thermal/results_random.csv
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import METRIC_NAMES, N_FOLDS, OUTPUT_DIR

METRIC_LABELS = {
    "r2": "$R^2$",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "max_rel_err": "Max relative error",
}
CV_LABELS = {
    "random": f"shuffled {N_FOLDS}-fold CV",
    "v2": "leave-one-V2-out CV",
}


def plot_cv_results(
    df: pd.DataFrame,
    stem: str,
    output_dir: str = os.path.join(OUTPUT_DIR, "figures"),
    metric_labels: dict = METRIC_LABELS,
    cv_labels: dict = CV_LABELS,
):
    """Draws one bar chart per metric from a CV summary and saves them as {stem}_{metric}.png."""
    os.makedirs(output_dir, exist_ok=True)
    cv_label = cv_labels.get(df["cv"].iloc[0], "cross-validation")

    for metric in METRIC_NAMES:
        label = metric_labels.get(metric, metric)
        fig, ax = plt.subplots()
        ax.bar(df["model"], df[f"{metric}_mean"], yerr=df[f"{metric}_std"], capsize=4)
        ax.set_xlabel("Model")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by model ({cv_label}, mean ± std)")
        fig.tight_layout()

        path = os.path.join(output_dir, f"{stem}_{metric}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot CV results, one figure per metric")
    parser.add_argument(
        "--input",
        default=os.path.join(OUTPUT_DIR, "results_v2.csv"),
        help="CV summary CSV to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "figures"),
        help="Directory to save figures (default: %(default)s)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    plot_cv_results(df, stem, args.output)


if __name__ == "__main__":
    main()
