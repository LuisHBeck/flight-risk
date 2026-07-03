"""
flight_risk.data
================
I/O helpers: load the raw collection-pipeline output and the model-ready
feature dataset.
"""

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: Path) -> pd.DataFrame:
    """
    Load the raw parquet file produced by the data collection pipeline.

    Parameters
    ----------
    path : Path
        Path to ``flights_with_weather.parquet``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with 31 columns as delivered by the collection pipeline.
    """
    print(f"Loading raw data: {path}")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    return df


def load_features(path: Path) -> pd.DataFrame:
    """
    Load the pre-processed feature dataset produced by the feature pipeline.

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

    dist = df[config.TARGET].value_counts(normalize=True).mul(100).round(1)
    print(f"  Target — on-time: {dist.get(0, 0):.1f}%  |  delayed: {dist.get(1, 0):.1f}%")
    return df
