"""
flight_risk.config
==================
Single source of truth for paths, constants and column lists.

Both the feature-engineering and the training pipelines import from here, so
they can never disagree about which columns exist, which are dropped, or where
artefacts live on disk.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# ROOT is the project root (the folder that contains this package).

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / ".data"
MODELS_DIR = ROOT / ".models"

# Inputs / outputs of the feature-engineering pipeline
RAW_PATH                 = DATA_DIR / "flights_with_weather.parquet"
FEATURES_PATH            = DATA_DIR / "flights_features.parquet"
ROUTE_STATS_PATH         = DATA_DIR / "route_stats.pkl"
AIRLINE_HOUR_STATS_PATH  = DATA_DIR / "airline_hour_stats.pkl"

# Outputs of the training pipeline
MODEL_PATH               = MODELS_DIR / "lgbm_binary.pkl"
ENCODERS_PATH            = MODELS_DIR / "encoders.pkl"


# ---------------------------------------------------------------------------
# Target / metadata
# ---------------------------------------------------------------------------

TARGET   = "is_delayed"
METADATA = ["dep_scheduled"]   # kept in parquet for splitting only — not a feature

# Minutes of departure delay above which a flight is labelled delayed
# (ANAC official threshold).
DELAY_THRESHOLD = 15


# ---------------------------------------------------------------------------
# Categorical columns (label-encoded at train time)
# ---------------------------------------------------------------------------

CAT_COLS = [
    "airline_icao", "origin_icao", "destination_icao",
    "origin_region", "destination_region",
    "route", "region_pair", "flight_range",
    "dep_time_block",
    "origin_wx_condition", "destination_wx_condition",
]


# ---------------------------------------------------------------------------
# Feature-engineering constants
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


# ---------------------------------------------------------------------------
# Columns excluded from the final feature set.
# Grouped by reason so the intent is explicit.
# ---------------------------------------------------------------------------

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
