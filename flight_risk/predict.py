"""
flight_risk.predict
====================
Inference: load the trained model and all reference artefacts, reproduce the
exact feature engineering applied at training time, and predict the probability
that a flight departs delayed (> threshold minutes).

What this module reuses (so train and inference can never drift apart):
    flight_risk.features.temporal.build_temporal_features
    flight_risk.features.route.build_route_features
    flight_risk.features.congestion.build_congestion_features
    flight_risk.features.weather.build_weather_features
    flight_risk.features.lookup.join_lookup_tables

What it deliberately does NOT run (post-flight / target-only):
    parse_datetimes / build_delay_cols   — these need dep_actual & arr_actual,
    which do not exist for a flight that has not departed yet.

Required input columns per flight
---------------------------------
Operational identifiers:
    airline_icao, origin_icao, destination_icao,
    origin_region, destination_region
Airport attributes (can be filled from your airport_reference table):
    origin_lat, origin_lon, destination_lat, destination_lon,
    origin_elevation_ft, destination_elevation_ft,
    origin_type, destination_type
Schedule (datetime-like):
    dep_scheduled, arr_scheduled
Weather forecast (Open-Meteo codes):
    origin_wx_weathercode, destination_wx_weathercode

See :func:`enrich_airports` for filling the airport-attribute block from an
airport reference table keyed by ICAO.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features.temporal import build_temporal_features
from .features.route import build_route_features
from .features.congestion import build_congestion_features
from .features.weather import build_weather_features
from .features.lookup import join_lookup_tables


# Columns the caller must provide for each flight (before airport enrichment).
# The airport-attribute block (lat, lon, elevation_ft, type, region) may be
# filled automatically by calling enrich_airports() first.
REQUIRED_INPUT_COLS = [
    "airline_icao", "origin_icao", "destination_icao",
    "origin_region", "destination_region",
    "origin_lat", "origin_lon", "destination_lat", "destination_lon",
    "origin_elevation_ft", "destination_elevation_ft",
    "origin_type", "destination_type",
    "dep_scheduled", "arr_scheduled",
    # WMO weather code (used to derive condition category + binary flags)
    "origin_wx_weathercode", "destination_wx_weathercode",
    # Numeric weather variables from Open-Meteo (passed through to model as-is)
    "origin_wx_temperature_2m", "origin_wx_precipitation",
    "origin_wx_windspeed_10m",  "origin_wx_windgusts_10m",
    "origin_wx_cloudcover",     "origin_wx_surface_pressure",
    "destination_wx_temperature_2m", "destination_wx_precipitation",
    "destination_wx_windspeed_10m",  "destination_wx_windgusts_10m",
    "destination_wx_cloudcover",     "destination_wx_surface_pressure",
]

# Columns satisfied by enrich_airports — they do not need to be in the input.
AIRPORT_ENRICHED_COLS = [
    "origin_region", "destination_region",
    "origin_lat", "origin_lon", "destination_lat", "destination_lon",
    "origin_elevation_ft", "destination_elevation_ft",
    "origin_type", "destination_type",
]

# Minimum columns required when using enrich_airports
REQUIRED_INPUT_COLS_WITH_ENRICHMENT = [
    c for c in REQUIRED_INPUT_COLS if c not in AIRPORT_ENRICHED_COLS
]


# ---------------------------------------------------------------------------
# Airport enrichment from the project's airport_reference table
# ---------------------------------------------------------------------------

def _build_airport_lookup(airport_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise ``airport_reference`` into a lookup table indexed by ICAO.

    Schema expected (matches the project's saved airport_reference file):

        ident           — ICAO code (e.g. "SBGR")
        type            — airport type string (e.g. "large_airport")
        name            — human-readable name (not used, ignored)
        latitude_deg    — decimal latitude
        longitude_deg   — decimal longitude
        elevation_ft    — field elevation in feet
        municipality    — city name (not used, ignored)
        iso_region      — ISO 3166-2 region code (e.g. "BR-SP")

    ``iso_region`` is trimmed to the subdivision code only ("BR-SP" → "SP")
    so it matches the ``origin_region`` / ``destination_region`` values the
    feature pipeline was trained on.

    Parameters
    ----------
    airport_reference : pd.DataFrame
        Raw airport reference table as loaded from disk.

    Returns
    -------
    pd.DataFrame
        Lookup indexed by ``ident`` with columns:
        ``lat``, ``lon``, ``elevation_ft``, ``type``, ``region``.
    """
    ref = airport_reference.copy()

    # Derive the short region code from "BR-SP" → "SP"
    ref["region"] = ref["iso_region"].str.split("-").str[-1]

    lookup = ref.rename(columns={
        "latitude_deg":  "lat",
        "longitude_deg": "lon",
    })[["ident", "lat", "lon", "elevation_ft", "type", "region"]].set_index("ident")

    return lookup


def enrich_airports(
    df: pd.DataFrame,
    airport_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill the airport-attribute columns from the project's airport_reference table.

    Expects the schema produced by the data collection pipeline::

        ident, type, name, latitude_deg, longitude_deg,
        elevation_ft, municipality, iso_region

    For each flight, looks up ``origin_icao`` and ``destination_icao`` and
    populates the six attribute columns the feature pipeline needs:

        origin_lat,  origin_lon,  origin_elevation_ft,  origin_type,  origin_region
        destination_lat, destination_lon, destination_elevation_ft,
        destination_type, destination_region

    ``iso_region`` ("BR-SP") is trimmed to the subdivision code ("SP") to
    match the values the model was trained on.

    After calling this function, the flights DataFrame only needs the three
    ICAO identifiers, the two scheduled timestamps, and the weather codes — the
    rest is filled in automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Flights with at least ``origin_icao`` and ``destination_icao``.
        Existing airport columns are overwritten.
    airport_reference : pd.DataFrame
        Airport reference table with the schema described above.

    Returns
    -------
    pd.DataFrame
        ``df`` with the six origin/destination attribute columns populated.

    Raises
    ------
    ValueError
        If any ICAO code in ``df`` is not found in ``airport_reference``.

    Examples
    --------
    >>> import joblib, pandas as pd
    >>> from flight_risk.predict import FlightDelayPredictor, enrich_airports
    >>>
    >>> airport_ref = pd.read_parquet(".data/airport_reference.parquet")
    >>> predictor   = FlightDelayPredictor.from_dir()
    >>>
    >>> flights = pd.DataFrame([{
    ...     "airline_icao":             "GLO",
    ...     "origin_icao":              "SBGR",
    ...     "destination_icao":         "SBRJ",
    ...     "dep_scheduled":            "2025-08-10 06:00",
    ...     "arr_scheduled":            "2025-08-10 07:05",
    ...     "origin_wx_weathercode":    0,
    ...     "destination_wx_weathercode": 61,
    ... }])
    >>>
    >>> flights = enrich_airports(flights, airport_ref)
    >>> predictor.predict_df(flights)
    """
    df = df.copy()
    lookup = _build_airport_lookup(airport_reference)

    # Validate — better to fail loudly than predict silently with NaNs
    for side in ("origin", "destination"):
        icao_col = f"{side}_icao"
        unknown = set(df[icao_col].unique()) - set(lookup.index)
        if unknown:
            raise ValueError(
                f"ICAO codes not found in airport_reference for {side}: {sorted(unknown)}. "
                "Add them to the reference table or remove those flights."
            )

    for side in ("origin", "destination"):
        keys = df[f"{side}_icao"]
        df[f"{side}_lat"]          = keys.map(lookup["lat"])
        df[f"{side}_lon"]          = keys.map(lookup["lon"])
        df[f"{side}_elevation_ft"] = keys.map(lookup["elevation_ft"])
        df[f"{side}_type"]         = keys.map(lookup["type"])
        df[f"{side}_region"]       = keys.map(lookup["region"])

    return df


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class FlightDelayPredictor:
    """
    Load model + artefacts once, then score many flights.

    Construct via :meth:`from_dir` to load everything from the standard project
    locations, or pass already-loaded objects to ``__init__`` directly (useful
    for tests or serving where artefacts are cached in memory).

    Examples
    --------
    >>> predictor = FlightDelayPredictor.from_dir()
    >>> predictor.predict_proba(flights_df)          # array of P(delayed)
    >>> predictor.predict_df(flights_df)             # input + columns added
    """

    def __init__(
        self,
        model,
        encoders: dict,
        route_stats: pd.DataFrame,
        airline_hour_stats: pd.DataFrame,
    ):
        self.model = model
        self.encoders = encoders
        self.route_stats = route_stats
        self.airline_hour_stats = airline_hour_stats
        # Exact feature set & order the model was trained on — the contract
        # every prediction must satisfy.
        self.feature_names = list(model.feature_name_)

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        models_dir: Path = config.MODELS_DIR,
        data_dir: Path = config.DATA_DIR,
    ) -> "FlightDelayPredictor":
        """
        Load all artefacts from disk.

        Parameters
        ----------
        models_dir : Path, optional
            Directory holding ``lgbm_binary.pkl`` and ``encoders.pkl``.
        data_dir : Path, optional
            Directory holding ``route_stats.pkl`` and ``airline_hour_stats.pkl``.

        Returns
        -------
        FlightDelayPredictor
        """
        model    = joblib.load(models_dir / "lgbm_binary.pkl")
        encoders = joblib.load(models_dir / "encoders.pkl")
        route_stats        = joblib.load(data_dir / "route_stats.pkl")
        airline_hour_stats = joblib.load(data_dir / "airline_hour_stats.pkl")
        return cls(model, encoders, route_stats, airline_hour_stats)

    # -- feature building --------------------------------------------------

    @staticmethod
    def _parse_scheduled(df: pd.DataFrame) -> pd.DataFrame:
        """Parse the two scheduled timestamps (actuals do not exist yet)."""
        df = df.copy()
        for col in ("dep_scheduled", "arr_scheduled"):
            df[col] = pd.to_datetime(df[col]).dt.floor("s").astype("datetime64[ns]")
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reproduce the training-time feature engineering, minus the target path.

        Runs: parse scheduled datetimes → temporal → route → congestion →
        weather → join lookup tables. Returns the enriched (un-encoded) frame.

        Note on congestion
        -------------------
        ``origin_hourly_flights`` / ``destination_hourly_arrivals`` are counted
        *within the rows you pass in*. To reproduce them faithfully, pass the
        full schedule for the time window (e.g. all of a day's flights), not a
        single isolated flight — a lone flight will count as 1.
        """
        missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required input columns: {missing}. "
                f"Fill airport attributes with enrich_airports() if needed."
            )

        df = self._parse_scheduled(df)
        df = build_temporal_features(df)     # also creates dep_hour (needed below)
        df = build_route_features(df)
        df = build_congestion_features(df)
        df = build_weather_features(df)
        df = join_lookup_tables(df, self.route_stats, self.airline_hour_stats)
        return df

    # -- encoding & alignment ---------------------------------------------

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the saved LabelEncoders, mapping unseen categories to -1.

        The training pipeline fed label-encoded integers to LightGBM as plain
        numeric features, so an out-of-vocabulary sentinel of -1 is safe: the
        model simply sees a value outside the range it was trained on rather
        than raising, which is what a strict ``le.transform`` would do.
        """
        X = X.copy()
        for col, le in self.encoders.items():
            if col not in X.columns:
                continue
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)
        return X

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns to exactly match the trained model."""
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Engineered frame is missing features: {missing}")
        return X[self.feature_names]

    # -- prediction --------------------------------------------------------

    def _features_to_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._align(self._encode(self.build_features(df)))

    def _enriched_to_X(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """Encode + align a frame that already went through build_features."""
        return self._align(self._encode(enriched))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return P(delayed) for each flight as a 1-D array in ``df`` row order.
        """
        X = self._features_to_X(df)
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_from_enriched(self, enriched: pd.DataFrame) -> np.ndarray:
        """
        Return P(delayed) from a frame already produced by :meth:`build_features`.

        Used by the API router to avoid running build_features twice — once for
        the human-readable feature audit and again for the model score.
        """
        X = self._enriched_to_X(enriched)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions (1 = delayed) using a probability threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Flights to score.
        threshold : float, optional
            Decision threshold on P(delayed). Default 0.5. Lower it to favour
            recall (catch more delays at the cost of more false alarms).
        """
        return (self.predict_proba(df) >= threshold).astype(int)

    def predict_df(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        proba_col: str = "delay_proba",
        label_col: str = "predicted_delayed",
    ) -> pd.DataFrame:
        """
        Return a copy of ``df`` with probability and label columns appended.

        Parameters
        ----------
        df : pd.DataFrame
            Flights to score.
        threshold : float, optional
            Decision threshold on P(delayed). Default 0.5.
        proba_col, label_col : str, optional
            Names of the appended columns.

        Returns
        -------
        pd.DataFrame
            ``df`` plus ``proba_col`` (float) and ``label_col`` (0/1).
        """
        proba = self.predict_proba(df)
        out = df.copy()
        out[proba_col] = proba
        out[label_col] = (proba >= threshold).astype(int)
        return out
