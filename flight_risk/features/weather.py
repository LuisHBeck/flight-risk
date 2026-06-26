"""
flight_risk.features.weather
============================
Categorical and binary weather features derived from Open-Meteo weather codes.
"""

import pandas as pd


def build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive categorical and binary weather features from Open-Meteo weather codes.

    Maps the numeric ``weathercode`` column to a human-readable condition
    category and then creates binary indicator flags for operationally relevant
    conditions (fog, rain, storm).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with ``origin_wx_weathercode`` and
        ``destination_wx_weathercode`` columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with weather condition columns added.
    """
    def _wx_condition(code) -> str:
        if pd.isna(code):           return "unknown"
        code = int(code)
        if code == 0:               return "clear"
        if code in [1, 2, 3]:      return "cloudy"
        if code in [45, 48]:       return "fog"
        if code in range(51, 68):  return "rain"
        if code in range(71, 78):  return "snow"
        if code in range(80, 83):  return "showers"
        if code in range(95, 100): return "storm"
        return "other"

    df["origin_wx_condition"]      = df["origin_wx_weathercode"].apply(_wx_condition)
    df["destination_wx_condition"] = df["destination_wx_weathercode"].apply(_wx_condition)

    for prefix in ("origin", "destination"):
        col = f"{prefix}_wx_condition"
        df[f"{prefix}_wx_is_fog"]   = (df[col] == "fog").astype(int)
        df[f"{prefix}_wx_is_rain"]  = (df[col].isin(["rain", "showers"])).astype(int)
        df[f"{prefix}_wx_is_storm"] = (df[col] == "storm").astype(int)

    return df
