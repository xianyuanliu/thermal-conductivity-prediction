"""SHAP analysis of a fitted model (default SVR): how (K1, K2, V2) drive predicted Keff.

With three features, Kernel SHAP enumerates all 2^3 coalitions, so the
Shapley values are exact. Prints the importance ranking and writes summary
and dependence plots to figures/ unless --output is given.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from config import DATA_PATH, FEATURES, MODEL_NAMES, RANDOM_STATE, TARGET, TEST_SIZE
from main import build_model

N_BACKGROUND = 50  # K-means summary of the training set used as SHAP baseline.


def run_shap(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "svr",
    n_background: int = N_BACKGROUND,
    output_dir: str = "figures",
):
    """Explains one model with Kernel SHAP, saves the plots and returns mean |SHAP| per feature."""
    # Same fixed split as main.py, so the explained model is the reported one.
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model(model_name)
    model.fit(X_train, y_train)

    def predict(data: np.ndarray):
        # SHAP passes bare arrays; restore column names to match the fitted pipeline.
        return model.predict(pd.DataFrame(data, columns=FEATURES))

    background = shap.kmeans(X_train, n_background)
    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(X_test)

    # Sanity check: SHAP values must reconstruct the predictions.
    reconstruction = explainer.expected_value + shap_values.sum(axis=1)
    max_gap = np.max(np.abs(reconstruction - model.predict(X_test)))
    print(f"Additivity check, max |gap|: {max_gap:.3g}")

    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
    importance = importance.sort_values(ascending=False)
    print("\nMean |SHAP| (feature importance, in Keff units):")
    for feature, value in importance.items():
        print(f"  {feature}: {value:.6g} ({value / importance.sum():.1%})")

    os.makedirs(output_dir, exist_ok=True)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP summary ({model_name.upper()})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"), dpi=200)
    plt.close()

    for feature in FEATURES:
        shap.dependence_plot(feature, shap_values, X_test, show=False)
        plt.title(f"SHAP dependence: {feature} ({model_name.upper()})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_shap_dependence_{feature}.png"), dpi=200)
        plt.close()

    print(f"\nPlots saved to {output_dir}/")
    return importance


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis of a fitted model")
    parser.add_argument(
        "--model", default="svr", choices=MODEL_NAMES, help="Model to explain (default: svr)"
    )
    parser.add_argument("--output", default="figures", help="Directory to save figures (default: figures)")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    run_shap(df[FEATURES], df[TARGET], args.model, output_dir=args.output)


if __name__ == "__main__":
    main()
