"""
flight_risk.features.route
==========================
Route, geography and airport features, including Haversine great-circle
distance and the trunk-route flag.
"""

import numpy as np
import pandas as pd

from .. import config


def build_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive route, geography and airport features.

    Uses the Haversine formula to compute great-circle distance between origin
    and destination coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with origin/destination lat-lon columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with route feature columns added.
    """
    df["route"]       = df["origin_icao"] + "_" + df["destination_icao"]
    df["region_pair"] = df["origin_region"] + "_" + df["destination_region"]
    df["same_region"] = (df["origin_region"] == df["destination_region"]).astype(int)

    def _haversine(lat1, lon1, lat2, lon2):
        R = 6_371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return R * 2 * np.arcsin(np.sqrt(a))

    df["distance_km"] = _haversine(
        df["origin_lat"], df["origin_lon"],
        df["destination_lat"], df["destination_lon"],
    )
    df["flight_range"] = pd.cut(
        df["distance_km"],
        bins=[0, 500, 1_500, np.inf],
        labels=["short", "medium", "long"],
    )

    df["elevation_diff_ft"]        = df["destination_elevation_ft"] - df["origin_elevation_ft"]
    df["origin_airport_size"]      = df["origin_type"].map(config.AIRPORT_SIZE)
    df["destination_airport_size"] = df["destination_type"].map(config.AIRPORT_SIZE)
    df["both_large_airports"]      = (
        (df["origin_airport_size"] == 3) & (df["destination_airport_size"] == 3)
    ).astype(int)
    df["is_trunk_route"] = df["route"].isin(config.TRUNK_ROUTES).astype(int)

    # Duração programada do voo em minutos — voos mais longos têm mais
    # oportunidade de recuperar atraso no ar, mas também acumulam mais variação
    df["scheduled_duration_min"] = (
        df["arr_scheduled"] - df["dep_scheduled"]
    ).dt.total_seconds() / 60

    return df
