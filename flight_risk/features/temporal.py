"""
flight_risk.features.temporal
=============================
Temporal features derived from the scheduled departure timestamp.
"""

import numpy as np
import pandas as pd
import holidays


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the scheduled departure timestamp.

    Raw integers (hour, day-of-week, month) are encoded as cyclic sin/cos
    pairs so the model understands that e.g. 23 h and 0 h are adjacent.
    The raw integers are retained temporarily for groupby operations and
    dropped later via ``config.DROP_COLS``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with parsed datetime columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with temporal feature columns added.
    """
    df["dep_hour"]        = df["dep_scheduled"].dt.hour
    df["dep_minute"]      = df["dep_scheduled"].dt.minute
    df["dep_day_of_week"] = df["dep_scheduled"].dt.dayofweek
    df["dep_month"]       = df["dep_scheduled"].dt.month
    df["dep_day_of_year"] = df["dep_scheduled"].dt.dayofyear
    df["dep_is_weekend"]  = (df["dep_day_of_week"] >= 5).astype(int)

    df["dep_hour_sin"]  = np.sin(2 * np.pi * df["dep_hour"] / 24)
    df["dep_hour_cos"]  = np.cos(2 * np.pi * df["dep_hour"] / 24)
    df["dep_dow_sin"]   = np.sin(2 * np.pi * df["dep_day_of_week"] / 7)
    df["dep_dow_cos"]   = np.cos(2 * np.pi * df["dep_day_of_week"] / 7)
    df["dep_month_sin"] = np.sin(2 * np.pi * df["dep_month"] / 12)
    df["dep_month_cos"] = np.cos(2 * np.pi * df["dep_month"] / 12)

    def _time_block(hour: int) -> str:
        if hour < 6:  return "early_morning"
        if hour < 12: return "morning"
        if hour < 18: return "afternoon"
        return "evening"

    df["dep_time_block"]   = df["dep_hour"].apply(_time_block)
    df["dep_is_peak_hour"] = df["dep_hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

    br_holidays = holidays.Brazil(years=range(2022, 2027))
    df["dep_is_holiday"] = (
        df["dep_scheduled"].dt.date
        .astype("datetime64[ns]")
        .isin(pd.to_datetime(list(br_holidays.keys())))
        .astype(int)
    )
    return df
