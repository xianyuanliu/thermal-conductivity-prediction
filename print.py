"""Bar charts of cross-validation results: one figure per metric.

Reads a CV summary CSV written by main.py (e.g. results_random.csv) and draws
one bar chart per metric, with each model's mean as the bar and its std as the
error bar. Figures are written to figures/ unless --output is given.

Usage:
  python print.py                              # plots results_v2.csv
  python print.py --input results_random.csv
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import METRIC_NAMES, N_FOLDS

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


def main():
    parser = argparse.ArgumentParser(description="Plot CV results, one figure per metric")
    parser.add_argument("--input", default="results_v2.csv", help="CV summary CSV to plot")
    parser.add_argument("--output", default="figures", help="Directory to save figures (default: figures)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    # Label the plots with the CV mode named in the filename (results_random -> random).
    cv_key = next((key for key in CV_LABELS if key in stem.lower()), None)
    cv_label = CV_LABELS[cv_key] if cv_key else "cross-validation"

    for metric in METRIC_NAMES:
        label = METRIC_LABELS.get(metric, metric)
        fig, ax = plt.subplots()
        ax.bar(df["model"], df[f"{metric}_mean"], yerr=df[f"{metric}_std"], capsize=4)
        ax.set_xlabel("Model")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by model ({cv_label}, mean ± std)")
        fig.tight_layout()

        path = os.path.join(args.output, f"{stem}_{metric}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
