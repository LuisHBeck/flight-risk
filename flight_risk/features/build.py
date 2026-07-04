"""
flight_risk.features.build
==========================
Orchestrates the full feature-engineering pipeline and prepares the
model-ready dataset.

Pipeline order:
    parse datetimes → delay cols → temporal → route → congestion → weather

This module also owns:

* :func:`select_model_columns` — builds the binary target and drops every
  column that must not reach the model.
* :func:`prepare_xy`           — splits features/target and label-encodes the
  categorical columns (train-time concern, used by the training pipeline).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .. import config
from .temporal import build_temporal_features
from .route import build_route_features
from .congestion import build_congestion_features
from .weather import build_weather_features


# ---------------------------------------------------------------------------
# Preprocessing sub-steps
# ---------------------------------------------------------------------------

def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all datetime columns to ``datetime64[ns]`` truncated to the second.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with datetime columns correctly typed.
    """
    cols = ["dep_scheduled", "dep_actual", "arr_scheduled", "arr_actual"]
    df[cols] = df[cols].apply(
        lambda col: pd.to_datetime(col, format="ISO8601")
        .dt.floor("s")
        .astype("datetime64[ns]")
    )
    return df


def build_delay_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw delay columns and remove obvious data errors from ANAC records.

    Flights with departure delay outside [-60, 500] minutes are treated as
    data entry errors and dropped (~0.08 % of records).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with parsed datetime columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with delay columns added and outlier rows removed.
    """
    df["dep_delay_min"]     = (df["dep_actual"] - df["dep_scheduled"]).dt.total_seconds() / 60
    df["arr_delay_min"]     = (df["arr_actual"] - df["arr_scheduled"]).dt.total_seconds() / 60
    df["dep_is_delayed"]    = (df["dep_delay_min"] > 15).astype(int)
    df["arr_is_delayed"]    = (df["arr_delay_min"] > 15).astype(int)
    df["delay_propagation"] = df["arr_delay_min"] - df["dep_delay_min"]

    before = len(df)
    df = df[
        (df["dep_delay_min"] >= -60) &
        (df["dep_delay_min"] <= 500)
    ].copy()
    print(f"  Outliers removed: {before - len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to the raw dataframe.

    Orchestrates all sub-steps in the correct order:
    datetime parsing → delay columns → temporal → route → congestion → weather.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded from ``flights_with_weather.parquet``.

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with all engineered features. Still contains raw
        and intermediate columns; call :func:`select_model_columns` to obtain
        the final model-ready dataset.
    """
    df = df.copy()
    df = parse_datetimes(df)
    df = build_delay_cols(df)
    df = build_temporal_features(df)
    df = build_route_features(df)
    df = build_congestion_features(df)
    df = build_weather_features(df)
    return df


# ---------------------------------------------------------------------------
# Column selection + binary target
# ---------------------------------------------------------------------------

def select_model_columns(df: pd.DataFrame, delay_threshold: int = config.DELAY_THRESHOLD) -> pd.DataFrame:
    """
    Build the binary target column and drop all columns not used for training.

    Removes leakage columns (only available after the flight), redundant
    encodings (raw integers superseded by cyclic sin/cos), raw coordinates
    (superseded by route/region features), and remaining datetime columns
    except ``dep_scheduled``, which is kept as a metadata column for
    temporal splitting (train/val/test by year).

    Parameters
    ----------
    df : pd.DataFrame
        Fully enriched dataframe (features + lookup statistics).
    delay_threshold : int, optional
        Departure delay in minutes above which a flight is labelled as delayed.
        Default is ``config.DELAY_THRESHOLD`` (15, ANAC official threshold).

    Returns
    -------
    pd.DataFrame
        Model-ready dataframe containing feature columns, the binary target
        ``is_delayed``, and ``dep_scheduled`` as a metadata column for
        temporal splitting. ``dep_scheduled`` must be excluded from X when
        training the model.
    """
    df = df.copy()
    df["is_delayed"] = (df["dep_delay_min"] > delay_threshold).astype(int)

    keep_cols = [
        c for c in df.columns
        if c not in config.DROP_COLS
        and (
            c == "dep_scheduled"                              # keep as metadata
            or not pd.api.types.is_datetime64_any_dtype(df[c])
        )
    ]

    df = df[keep_cols]

    delayed_rate = df["is_delayed"].mean() * 100
    print(f"  Target distribution — on-time: {100 - delayed_rate:.1f}%  |  delayed: {delayed_rate:.1f}%")
    print(f"  Final columns ({len(df.columns)}): {df.columns.tolist()}")

    return df


# ---------------------------------------------------------------------------
# X / y preparation (train-time)
# ---------------------------------------------------------------------------

def prepare_xy(df: pd.DataFrame):
    """
    Separate features and target, encode categorical columns.

    ``dep_scheduled`` is excluded from X — it is a metadata column used only
    for temporal splitting and must never be seen by the model.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset as loaded by :func:`flight_risk.data.load_features`.

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
    X = df.drop(columns=[config.TARGET] + config.METADATA)
    y = df[config.TARGET].copy()

    encoders = {}
    for col in config.CAT_COLS:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    print(f"\nFeatures: {X.shape[1]}  |  Samples: {X.shape[0]:,}")
    return X, y, encoders
