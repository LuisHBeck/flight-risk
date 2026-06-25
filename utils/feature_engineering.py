"""
flight-risk/feature_engineering.py
===================================
Standalone feature engineering pipeline for the flight delay prediction project.

Reads the raw ``flights_with_weather.parquet`` produced by the data collection
pipeline, applies all feature engineering transformations, and writes:

* ``flights_features.parquet``  — model-ready dataset containing only the
  columns that will be used for training (features + binary target).
  Datetime columns, raw coordinates, redundant encodings, and leakage columns
  are excluded.

* ``route_stats.pkl``           — lookup table (DataFrame indexed by ``route``)
  with historical delay statistics per route. Used at inference time to enrich
  new flights without re-computing on the full dataset.

* ``airline_hour_stats.pkl``    — lookup table (DataFrame indexed by
  ``(airline_icao, dep_hour)``) with historical delay statistics per airline
  and departure hour. Also used at inference time.

Usage
-----
    python feature_engineering.py
    python feature_engineering.py --data .data/flights_with_weather.parquet
    python feature_engineering.py --data .data/flights_with_weather.parquet \\
        --output .data/ --delay-threshold 15

Parameters
----------
--data : str
    Path to the raw ``flights_with_weather.parquet`` file.
    Default: ``.data/flights_with_weather.parquet`` relative to this script.
--output : str
    Directory where all output files will be written.
    Default: ``.data/`` relative to this script.
--delay-threshold : int
    Minutes of departure delay above which a flight is labelled as delayed
    (binary target = 1). Default: 15 (ANAC official threshold).
"""

import argparse
import warnings
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUNK_ROUTES: set[str] = {
    "SBRJ_SBGR", "SBGR_SBRJ",  # Air bridge RJ-SP
    "SBGR_SBBR", "SBBR_SBGR",  # SP-Brasília
    "SBGR_SBSV", "SBSV_SBGR",  # SP-Salvador
    "SBGR_SBRF", "SBRF_SBGR",  # SP-Recife
    "SBGR_SBFZ", "SBFZ_SBGR",  # SP-Fortaleza
    "SBGR_SBPA", "SBPA_SBGR",  # SP-Porto Alegre
    "SBGR_SBBH", "SBBH_SBGR",  # SP-BH
}

AIRPORT_SIZE: dict[str, int] = {
    "small_airport": 1,
    "medium_airport": 2,
    "large_airport": 3,
}

# Columns excluded from the final feature set.
# Grouped by reason so the intent is explicit.
_LEAKAGE_COLS: list[str] = [
    # Only available after the flight has departed / arrived
    # Note: dep_scheduled is intentionally kept — it is used as a metadata
    # column for temporal splitting and is NOT used as a model feature.
    "dep_actual", "arr_actual", "arr_scheduled",
    "dep_delay_min", "arr_delay_min",
    "dep_is_delayed", "arr_is_delayed",
    "delay_propagation",
]

_REDUNDANT_COLS: list[str] = [
    # Raw integers already captured by cyclic sin/cos encodings
    "dep_hour", "dep_minute", "dep_day_of_week", "dep_month",
    # Raw weathercode already captured by wx_condition + wx_is_* flags
    "origin_wx_weathercode", "destination_wx_weathercode",
    # Raw coordinates already captured by route and region_pair
    "origin_lat", "origin_lon", "destination_lat", "destination_lon",
    # Airport type string already captured by airport_size ordinal
    "origin_type", "destination_type",
    # Redundant with region_pair
    "same_region",
    # Redundant with origin_airport_size + destination_airport_size
    "both_large_airports",
]

DROP_COLS: list[str] = _LEAKAGE_COLS + _REDUNDANT_COLS


# ---------------------------------------------------------------------------
# Step 1 — Load raw data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 2 — Core feature engineering
# ---------------------------------------------------------------------------

def _parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_delay_cols(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the scheduled departure timestamp.

    Raw integers (hour, day-of-week, month) are encoded as cyclic sin/cos
    pairs so the model understands that e.g. 23 h and 0 h are adjacent.
    The raw integers are retained temporarily for groupby operations and
    dropped later via ``DROP_COLS``.

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


def _build_route_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["origin_airport_size"]      = df["origin_type"].map(AIRPORT_SIZE)
    df["destination_airport_size"] = df["destination_type"].map(AIRPORT_SIZE)
    df["both_large_airports"]      = (
        (df["origin_airport_size"] == 3) & (df["destination_airport_size"] == 3)
    ).astype(int)
    df["is_trunk_route"] = df["route"].isin(TRUNK_ROUTES).astype(int)

    # Duração programada do voo em minutos — voos mais longos têm mais
    # oportunidade de recuperar atraso no ar, mas também acumulam mais variação
    df["scheduled_duration_min"] = (
        df["arr_scheduled"] - df["dep_scheduled"]
    ).dt.total_seconds() / 60

    return df


def _build_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df = _parse_datetimes(df)
    df = _build_delay_cols(df)
    df = _build_temporal_features(df)
    df = _build_route_features(df)
    df = _build_congestion_features(df)
    df = _build_weather_features(df)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Auxiliary lookup tables
# ---------------------------------------------------------------------------

def build_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical delay statistics aggregated by route.

    Used both to enrich the training dataset and as a lookup table at
    inference time (saved as ``route_stats.pkl``). For unseen routes, callers
    should fall back to the global mean of each statistic.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe containing ``route``, ``dep_delay_min``, and
        ``dep_is_delayed`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``route`` with columns:

        * ``route_hist_delay_mean`` — mean departure delay in minutes.
        * ``route_hist_delay_std``  — standard deviation of departure delay.
        * ``route_hist_delay_rate`` — fraction of flights delayed > 15 min.
    """
    stats = df.groupby("route")["dep_delay_min"].agg(
        route_hist_delay_mean="mean",
        route_hist_delay_std="std",
    )
    stats["route_hist_delay_rate"] = df.groupby("route")["dep_is_delayed"].mean()
    return stats


def build_airline_hour_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical delay statistics aggregated by airline and departure hour.

    Captures the punctuality profile of each airline at each hour of the day.
    Used both to enrich the training dataset and as a lookup table at inference
    time (saved as ``airline_hour_stats.pkl``).

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe containing ``airline_icao``, ``dep_hour``,
        ``dep_is_delayed``, and ``dep_delay_min`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``(airline_icao, dep_hour)`` with columns:

        * ``airline_hour_delay_rate`` — fraction of flights delayed > 15 min.
        * ``airline_hour_delay_mean`` — mean departure delay in minutes.
    """
    stats = df.groupby(["airline_icao", "dep_hour"]).agg(
        airline_hour_delay_rate=("dep_is_delayed", "mean"),
        airline_hour_delay_mean=("dep_delay_min",  "mean"),
    ).round(4)
    return stats


def join_lookup_tables(
    df: pd.DataFrame,
    route_stats: pd.DataFrame,
    airline_hour_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join both lookup tables onto the main dataframe.

    Missing values that arise from unseen keys are filled with the global mean
    of each statistic, so the output contains no NaN in these columns.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe with ``route``, ``airline_icao``, and ``dep_hour``
        columns.
    route_stats : pd.DataFrame
        Output of :func:`build_route_stats`.
    airline_hour_stats : pd.DataFrame
        Output of :func:`build_airline_hour_stats`.

    Returns
    -------
    pd.DataFrame
        Dataframe with six additional historical statistics columns.
    """
    df = df.join(route_stats, on="route")
    df["route_hist_delay_mean"].fillna(route_stats["route_hist_delay_mean"].mean(), inplace=True)
    df["route_hist_delay_std"].fillna(route_stats["route_hist_delay_std"].mean(),   inplace=True)
    df["route_hist_delay_rate"].fillna(route_stats["route_hist_delay_rate"].mean(), inplace=True)

    df = df.join(airline_hour_stats, on=["airline_icao", "dep_hour"])
    df["airline_hour_delay_rate"].fillna(airline_hour_stats["airline_hour_delay_rate"].mean(), inplace=True)
    df["airline_hour_delay_mean"].fillna(airline_hour_stats["airline_hour_delay_mean"].mean(), inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 4 — Select model columns and build binary target
# ---------------------------------------------------------------------------

def select_model_columns(df: pd.DataFrame, delay_threshold: int = 15) -> pd.DataFrame:
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
        Default is 15 (ANAC official threshold).

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
        if c not in DROP_COLS
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full feature engineering pipeline and persist all outputs.

    Outputs
    -------
    <output>/flights_features.parquet
        Model-ready dataset with all engineered features and binary target.
    <output>/route_stats.pkl
        Historical delay statistics per route (lookup table for inference).
    <output>/airline_hour_stats.pkl
        Historical delay statistics per airline × hour (lookup table for inference).
    """
    parser = argparse.ArgumentParser(
        description="Flight delay feature engineering pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / ".data" / "flights_with_weather.parquet",
        help="Path to the raw flights_with_weather.parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / ".data",
        help="Directory where output files will be saved.",
    )
    parser.add_argument(
        "--delay-threshold",
        type=int,
        default=15,
        help="Minutes above which a flight is labelled as delayed (default: 15).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_raw(args.data)

    # 2. Feature engineering
    print("\n[1/4] Building features...")
    df = build_features(df)
    print(f"  Shape after engineering: {df.shape}")

    # 3. Lookup tables — computed on the full dataset before any split
    print("\n[2/4] Building lookup tables...")
    route_stats        = build_route_stats(df)
    airline_hour_stats = build_airline_hour_stats(df)
    print(f"  Unique routes:          {len(route_stats):,}")
    print(f"  Unique airline×hour:    {len(airline_hour_stats):,}")

    df = join_lookup_tables(df, route_stats, airline_hour_stats)

    # 4. Select model columns + target
    print("\n[3/4] Selecting model columns...")
    df_model = select_model_columns(df, delay_threshold=args.delay_threshold)

    # 5. Persist
    print("\n[4/4] Saving outputs...")

    features_path          = args.output / "flights_features.parquet"
    route_stats_path       = args.output / "route_stats.pkl"
    airline_hour_stats_path = args.output / "airline_hour_stats.pkl"

    df_model.to_parquet(features_path, index=False)
    joblib.dump(route_stats,        route_stats_path)
    joblib.dump(airline_hour_stats, airline_hour_stats_path)

    print(f"  flights_features.parquet  → {features_path}  ({df_model.shape[0]:,} rows × {df_model.shape[1]} cols)")
    print(f"  route_stats.pkl           → {route_stats_path}")
    print(f"  airline_hour_stats.pkl    → {airline_hour_stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
