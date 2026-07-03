"""
flight_risk.plots
=================
Evaluation plots, kept separate from the model logic so the training pipeline
stays free of matplotlib concerns.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> Path:
    """
    Plot and save the confusion matrix.

    Parameters
    ----------
    y_test : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    output_dir : Path
        Directory where ``confusion_matrix.png`` is saved.

    Returns
    -------
    Path
        Path of the saved figure.
    """
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On-time", "Delayed"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — LightGBM Binary")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")
    return cm_path


def plot_feature_importance(model, output_dir: Path, top_n: int = 25) -> pd.DataFrame:
    """
    Plot and save the top-N feature importances; also return them as a DataFrame.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Fitted model exposing ``feature_name_`` and ``feature_importances_``.
    output_dir : Path
        Directory where ``feature_importance.png`` is saved.
    top_n : int, optional
        Number of top features to plot (default: 25).

    Returns
    -------
    pd.DataFrame
        The top-N features with their importance, sorted descending.
    """
    feat_imp = (
        pd.DataFrame({
            "feature":    model.feature_name_,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(data=feat_imp, y="feature", x="importance", ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Features — LightGBM Binary")
    ax.set_xlabel("Importance (split)")
    plt.tight_layout()
    fi_path = output_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Feature importance saved: {fi_path}")
    return feat_imp
