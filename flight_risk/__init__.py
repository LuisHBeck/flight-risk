"""
flight-risk
===========
Binary flight departure-delay prediction (LightGBM).

Convenience re-exports so common entrypoints are importable directly:

    from flight_risk import build_features, load_features, train
"""

from . import config
from .data import load_raw, load_features
from .features.build import build_features, select_model_columns, prepare_xy
from .features.lookup import (
    build_route_stats,
    build_airline_hour_stats,
    join_lookup_tables,
)
from .model import temporal_split, train, evaluate, Splits

__version__ = "0.1.0"

__all__ = [
    "config",
    "load_raw",
    "load_features",
    "build_features",
    "select_model_columns",
    "prepare_xy",
    "build_route_stats",
    "build_airline_hour_stats",
    "join_lookup_tables",
    "temporal_split",
    "train",
    "evaluate",
    "Splits",
]
