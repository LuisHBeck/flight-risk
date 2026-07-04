"""
flight_risk.features.congestion
===============================
Hourly airport congestion proxies computed from the dataset itself
(no external data required).
"""

import pandas as pd


def build_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hourly airport congestion proxies from the dataset itself.

    Counts how many flights depart from the same origin airport in the same
    hour and how many arrive at the same destination airport in the same hour.
    These are proxies for apron/gate pressure without requiring external data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with parsed datetime columns and route columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with three congestion columns added.
    """
    dep_date = df["dep_scheduled"].dt.date
    dep_hour = df["dep_scheduled"].dt.hour

    df["origin_hourly_flights"] = df.groupby(
        [df["origin_icao"], dep_date, dep_hour]
    )["origin_icao"].transform("count")

    arr_date = df["arr_scheduled"].dt.date
    arr_hour = df["arr_scheduled"].dt.hour

    df["destination_hourly_arrivals"] = df.groupby(
        [df["destination_icao"], arr_date, arr_hour]
    )["destination_icao"].transform("count")

    df["total_hourly_congestion"] = (
        df["origin_hourly_flights"] + df["destination_hourly_arrivals"]
    )
    return df
