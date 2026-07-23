"""SHAP analysis of the SVR surrogate model.

Explains how (K1, K2, V2) drive the predicted effective thermal conductivity.
The SVR pipeline scales features internally, so SHAP values are computed in
the original physical units. With three features, Kernel SHAP enumerates all
2^3 coalitions, giving exact Shapley values rather than a sampled
approximation.

Usage:
  python shap_analysis.py

Writes the feature importance ranking to stdout and the plots to figures/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from main import DATA_PATH, FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, build_model

FIGURES_DIR = "figures"
N_BACKGROUND = 50  # K-means summary of the training set used as SHAP baseline.


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Same fixed split as main.py, so the explained model is the reported one.
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = build_model("svr")
    model.fit(X_train, y_train)

    def predict(data: np.ndarray) -> np.ndarray:
        # SHAP passes bare arrays; restore column names to match the fitted pipeline.
        return model.predict(pd.DataFrame(data, columns=FEATURES))

    background = shap.kmeans(X_train, N_BACKGROUND)
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

    os.makedirs(FIGURES_DIR, exist_ok=True)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP summary (SVR)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=200)
    plt.close()

    for feature in FEATURES:
        shap.dependence_plot(feature, shap_values, X_test, show=False)
        plt.title(f"SHAP dependence: {feature} (SVR)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"shap_dependence_{feature}.png"), dpi=200)
        plt.close()

    print(f"\nPlots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
