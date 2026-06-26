"""
scripts/run_features.py
=======================
Thin CLI entrypoint for the feature-engineering pipeline.

Reads the raw ``flights_with_weather.parquet``, applies all feature
engineering, and writes:

    <output>/flights_features.parquet   model-ready dataset (features + target)
    <output>/route_stats.pkl            per-route delay lookup table
    <output>/airline_hour_stats.pkl     per-airline×hour delay lookup table

Usage
-----
    python -m scripts.run_features
    python -m scripts.run_features --data .data/flights_with_weather.parquet
    python -m scripts.run_features --output .data/ --delay-threshold 15

Or, after `pip install -e .`:
    flight-risk-features
"""

import argparse
from pathlib import Path

import joblib

from flight_risk import config
from flight_risk.data import load_raw
from flight_risk.features.build import build_features, select_model_columns
from flight_risk.features.lookup import (
    build_route_stats,
    build_airline_hour_stats,
    join_lookup_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flight delay feature engineering pipeline."
    )
    parser.add_argument(
        "--data", type=Path, default=config.RAW_PATH,
        help="Path to the raw flights_with_weather.parquet file.",
    )
    parser.add_argument(
        "--output", type=Path, default=config.DATA_DIR,
        help="Directory where output files will be saved.",
    )
    parser.add_argument(
        "--delay-threshold", type=int, default=config.DELAY_THRESHOLD,
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

    features_path           = args.output / "flights_features.parquet"
    route_stats_path        = args.output / "route_stats.pkl"
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
