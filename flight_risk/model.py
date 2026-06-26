"""
flight_risk.model
=================
Model lifecycle: temporal split, training with early stopping, and evaluation
on the held-out test set.
"""

from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from .plots import plot_confusion_matrix, plot_feature_importance


class Splits(NamedTuple):
    """Container for the six train/val/test arrays returned by :func:`temporal_split`."""
    X_train: pd.DataFrame
    X_val:   pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_val:   pd.Series
    y_test:  pd.Series


# ---------------------------------------------------------------------------
# Temporal split by year
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_years: tuple = (2022, 2023, 2024),
    val_cutoff: str = "2025-07-01",
) -> Splits:
    """
    Split dataset by calendar year to preserve temporal order.

    Using full years avoids data leakage and ensures all seasons are
    represented in both training and test sets.

    Split:
        Train:      years in ``train_years``          (~74 %)
        Validation: 2025-jan to ``val_cutoff``        (~13 %)  — used for early stopping
        Test:       ``val_cutoff`` to end of dataset  (~13 %)  — held-out final evaluation

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset containing ``dep_scheduled``.
    X : pd.DataFrame
        Feature matrix aligned with ``df``.
    y : pd.Series
        Target series aligned with ``df``.
    train_years : tuple, optional
        Calendar years assigned to the training set.
    val_cutoff : str, optional
        ISO date string separating validation from test within 2025.

    Returns
    -------
    Splits
        Named tuple with ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    dates = df["dep_scheduled"]

    train_mask = dates.dt.year.isin(train_years)
    val_mask   = (~train_mask) & (dates < val_cutoff)
    test_mask  = (~train_mask) & (dates >= val_cutoff)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    total = len(X)
    print(f"\nTemporal split:")
    print(f"  Train      (2022–2024):  {len(X_train):,}  ({100*len(X_train)/total:.1f}%)"
          f"  |  delay rate: {y_train.mean()*100:.1f}%")
    print(f"  Validation (jan–jun 25): {len(X_val):,}   ({100*len(X_val)/total:.1f}%)"
          f"  |  delay rate: {y_val.mean()*100:.1f}%")
    print(f"  Test       (jul–dez 25): {len(X_test):,}   ({100*len(X_test)/total:.1f}%)"
          f"  |  delay rate: {y_test.mean()*100:.1f}%")

    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    """
    Train a binary LightGBM classifier with early stopping.

    ``class_weight='balanced'`` compensates for the class imbalance
    (~85 % on-time / ~15 % delayed) without generating synthetic samples.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame
        Validation feature matrix used for early stopping.
    y_val : pd.Series
        Validation target.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted model.
    """
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=10000,
        learning_rate=0.01,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )

    print(f"\nBest iteration: {model.best_iteration_}")
    return model


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, output_dir: Path) -> dict:
    """
    Evaluate the trained model on the held-out test set and save plots.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Fitted model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target.
    output_dir : Path
        Directory where plots will be saved.

    Returns
    -------
    dict
        Dictionary with ``roc_auc`` and ``f1`` scores.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT — LightGBM Binary")
    print("=" * 55)
    print(classification_report(
        y_test, y_pred,
        target_names=["On-time", "Delayed"],
        digits=3,
    ))

    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"F1 (delayed): {f1:.4f}")

    # Plots (delegated to flight_risk.plots)
    plot_confusion_matrix(y_test, y_pred, output_dir)
    feat_imp = plot_feature_importance(model, output_dir, top_n=25)

    print("\nTop 25 features:")
    print(feat_imp.to_string(index=False))

    return {"roc_auc": auc, "f1": f1}
