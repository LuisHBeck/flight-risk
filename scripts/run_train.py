"""
scripts/run_train.py
====================
Thin CLI entrypoint for the training pipeline.

Loads the model-ready dataset produced by ``run_features.py``, performs the
temporal split, trains a LightGBM classifier with early stopping, evaluates on
the held-out test set, and saves all artefacts.

Outputs saved to --output:
    lgbm_binary.pkl        trained LightGBM model
    encoders.pkl           LabelEncoders for categorical columns
    confusion_matrix.png
    feature_importance.png

Usage
-----
    python -m scripts.run_train
    python -m scripts.run_train --data .data/flights_features.parquet --output .models/

Or, after `pip install -e .`:
    flight-risk-train
"""

import argparse
from pathlib import Path

import joblib

from flight_risk import config
from flight_risk.data import load_features
from flight_risk.features.build import prepare_xy
from flight_risk.model import temporal_split, train, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Flight delay binary LightGBM trainer.")
    parser.add_argument(
        "--data", type=Path, default=config.FEATURES_PATH,
        help="Path to flights_features.parquet (output of run_features.py).",
    )
    parser.add_argument(
        "--output", type=Path, default=config.MODELS_DIR,
        help="Directory where model artefacts will be saved.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-processed features
    df = load_features(args.data)

    # 2. Prepare X / y
    print("\n--- Preparing X / y ---")
    X, y, encoders = prepare_xy(df)

    # 3. Temporal split by year
    print("\n--- Temporal split ---")
    splits = temporal_split(df, X, y)

    # 4. Train
    print("\n--- Training LightGBM ---")
    model = train(splits.X_train, splits.y_train, splits.X_val, splits.y_val)

    # 5. Evaluate
    print("\n--- Evaluation on test set ---")
    metrics = evaluate(model, splits.X_test, splits.y_test, args.output)

    # 6. Save artefacts
    model_path    = args.output / "lgbm_binary.pkl"
    encoders_path = args.output / "encoders.pkl"

    joblib.dump(model,    model_path)
    joblib.dump(encoders, encoders_path)

    print(f"\nModel saved:    {model_path}")
    print(f"Encoders saved: {encoders_path}")
    print(f"\nROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1:      {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
