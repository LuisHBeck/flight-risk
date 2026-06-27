"""
flight-risk/train.py
====================
Trains a binary LightGBM classifier to predict flight departure delays.

Expects the pre-processed dataset produced by ``feature_engineering.py``.
All feature engineering, lookup table computation, and outlier removal are
handled upstream — this script focuses exclusively on training and evaluation.

Prerequisites
-------------
Run ``feature_engineering.py`` first to generate:
    .data/flights_features.parquet
    .data/route_stats.pkl
    .data/airline_hour_stats.pkl

Usage
-----
    python train.py
    python train.py --data .data/flights_features.parquet --output .models/

Outputs saved to --output:
    lgbm_binary.pkl        trained LightGBM model
    encoders.pkl           LabelEncoders for categorical columns
    confusion_matrix.png
    feature_importance.png
"""

import argparse
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH  = Path(__file__).parent.parent / ".data" / "flights_features.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / ".models"

TARGET   = "is_delayed"
METADATA = ["dep_scheduled"]   # kept in parquet for splitting only — not a feature

CAT_COLS = [
    "airline_icao", "origin_icao", "destination_icao",
    "origin_region", "destination_region",
    "route", "region_pair", "flight_range",
    "dep_time_block",
    "origin_wx_condition", "destination_wx_condition",
]


# ---------------------------------------------------------------------------
# 2. LOAD
# ---------------------------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    """
    Load the pre-processed feature dataset produced by feature_engineering.py.

    Parameters
    ----------
    path : Path
        Path to ``flights_features.parquet``.

    Returns
    -------
    pd.DataFrame
        Feature dataset with all engineered columns, binary target
        ``is_delayed``, and metadata column ``dep_scheduled``.
    """
    print(f"Loading features: {path}")
    df = pd.read_parquet(path)
    df["dep_scheduled"] = pd.to_datetime(df["dep_scheduled"])
    print(f"  Shape: {df.shape}")
    print(f"  Period: {df['dep_scheduled'].min().date()} → {df['dep_scheduled'].max().date()}")

    dist = df[TARGET].value_counts(normalize=True).mul(100).round(1)
    print(f"  Target — on-time: {dist.get(0, 0):.1f}%  |  delayed: {dist.get(1, 0):.1f}%")
    return df


# ---------------------------------------------------------------------------
# 3. PREPARE X / y
# ---------------------------------------------------------------------------

def prepare_xy(df: pd.DataFrame):
    """
    Separate features and target, encode categorical columns.

    ``dep_scheduled`` is excluded from X — it is a metadata column used only
    for temporal splitting and must never be seen by the model.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset as loaded by :func:`load_features`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with categorical columns label-encoded.
    y : pd.Series
        Binary target series (``is_delayed``).
    encoders : dict
        Mapping of column name → fitted ``LabelEncoder``. Must be saved and
        reused at inference time to apply identical transformations.
    """
    X = df.drop(columns=[TARGET] + METADATA)
    y = df[TARGET].copy()

    encoders = {}
    for col in CAT_COLS:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    print(f"\nFeatures: {X.shape[1]}  |  Samples: {X.shape[0]:,}")
    return X, y, encoders


# ---------------------------------------------------------------------------
# 4. TEMPORAL SPLIT BY YEAR
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_years: tuple = (2022, 2023, 2024),
    val_cutoff: str = "2025-07-01",
):
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
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
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

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------------------------------

def train(X_train, y_train, X_val, y_val):
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
# 6. EVALUATE
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

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On-time", "Delayed"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — LightGBM Binary")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved: {cm_path}")
    plt.close()

    # Feature importance
    feat_imp = (
        pd.DataFrame({
            "feature":    model.feature_name_,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(25)
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(data=feat_imp, y="feature", x="importance", ax=ax, palette="viridis")
    ax.set_title("Top 25 Features — LightGBM Binary")
    ax.set_xlabel("Importance (split)")
    plt.tight_layout()
    fi_path = output_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    print(f"Feature importance saved: {fi_path}")
    plt.close()

    print("\nTop 25 features:")
    print(feat_imp.to_string(index=False))

    return {"roc_auc": auc, "f1": f1}


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flight delay binary LightGBM trainer.")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH,
        help="Path to flights_features.parquet (output of feature_engineering.py).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where model artefacts will be saved.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-processed features
    df = load_features(args.data)

    # 2. Prepare X / y
    print("\n--- Preparing X / y ---")
    X, y, encoders = prepare_xy(df)

    # 3. Temporal split by year
    print("\n--- Temporal split ---")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(df, X, y)

    # 4. Train
    print("\n--- Training LightGBM ---")
    model = train(X_train, y_train, X_val, y_val)

    # 5. Evaluate
    print("\n--- Evaluation on test set ---")
    metrics = evaluate(model, X_test, y_test, args.output)

    # 6. Save artefacts
    model_path    = args.output / "lgbm_binary.pkl"
    encoders_path = args.output / "encoders.pkl"

    joblib.dump(model,    model_path)
    joblib.dump(encoders, encoders_path)

    print(f"\nModel saved:    {model_path}")
    print(f"Encoders saved: {encoders_path}")
    print(f"\nROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1:      {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
