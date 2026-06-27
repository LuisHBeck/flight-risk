"""
scripts/run_predict.py
======================
Thin CLI entrypoint for batch inference.

Reads a file of flights to score (parquet or csv), runs the predictor, and
writes the same rows back with ``delay_proba`` and ``predicted_delayed``
columns appended.

The input file must contain the columns listed in
``flight_risk.predict.REQUIRED_INPUT_COLS``. Pass the full schedule for the
time window so the congestion features are computed correctly (see note in
``FlightDelayPredictor.build_features``).

Usage
-----
    python -m scripts.run_predict --input flights_to_score.parquet
    python -m scripts.run_predict --input day.csv --output scored.csv --threshold 0.4

Or, after `pip install -e .`:
    flight-risk-predict --input flights_to_score.parquet
"""

import argparse
from pathlib import Path

from flight_risk import config
from flight_risk.predict import FlightDelayPredictor


def _read(path: Path):
    import pandas as pd
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _write(df, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flight delay batch inference.")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Flights to score (.parquet or .csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write scored flights. Default: <input stem>_scored<suffix>.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold on P(delayed). Default: 0.5.",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=config.MODELS_DIR,
        help="Directory with lgbm_binary.pkl and encoders.pkl.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=config.DATA_DIR,
        help="Directory with route_stats.pkl and airline_hour_stats.pkl.",
    )
    args = parser.parse_args()

    output = args.output or args.input.with_name(
        f"{args.input.stem}_scored{args.input.suffix}"
    )

    print(f"Loading predictor — models: {args.models_dir} | data: {args.data_dir}")
    predictor = FlightDelayPredictor.from_dir(args.models_dir, args.data_dir)

    print(f"Reading flights: {args.input}")
    flights = _read(args.input)
    print(f"  {len(flights):,} flights to score")

    scored = predictor.predict_df(flights, threshold=args.threshold)

    n_delayed = int(scored["predicted_delayed"].sum())
    print(f"  Predicted delayed: {n_delayed:,} ({100 * n_delayed / len(scored):.1f}%) "
          f"at threshold {args.threshold}")

    _write(scored, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
