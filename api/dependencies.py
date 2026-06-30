"""
api.dependencies
================
Carrega todos os artefatos do modelo e a tabela de referência de aeroportos
uma única vez na inicialização da API e os disponibiliza via injeção de
dependência do FastAPI.

Uso nos routers::

    from api.dependencies import get_predictor, get_airport_reference

    @router.post("/predict")
    def predict(
        flight: FlightInput,
        predictor: FlightDelayPredictor = Depends(get_predictor),
        airport_ref: pd.DataFrame       = Depends(get_airport_reference),
    ): ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flight_risk import config
from flight_risk.predict import FlightDelayPredictor

# ---------------------------------------------------------------------------
# Artefatos — preenchidos pelo lifespan em main.py
# ---------------------------------------------------------------------------

_predictor:       FlightDelayPredictor | None = None
_airport_reference: pd.DataFrame | None       = None


def init_resources(
    models_dir: Path = config.MODELS_DIR,
    data_dir:   Path = config.DATA_DIR,
    airport_reference_path: Path = config.DATA_DIR / "airports_reference.csv",
) -> None:
    """
    Carrega modelo, encoders, lookup tables e airport_reference em memória.

    Chamado pelo lifespan da aplicação FastAPI — roda uma vez na inicialização
    e disponibiliza os objetos para todas as requisições.

    Parameters
    ----------
    models_dir : Path
        Pasta com lgbm_binary.pkl e encoders.pkl.
    data_dir : Path
        Pasta com route_stats.pkl e airline_hour_stats.pkl.
    airport_reference_path : Path
        Arquivo parquet (ou csv) com a tabela de referência de aeroportos.
        Schema esperado: ident, type, name, latitude_deg, longitude_deg,
        elevation_ft, municipality, iso_region.
    """
    global _predictor, _airport_reference

    import numpy as np

    print(f"[startup] Carregando modelo de {models_dir}...")
    _predictor = FlightDelayPredictor.from_dir(models_dir, data_dir)
    print(f"[startup] Modelo carregado. Features: {len(_predictor.feature_names)}")

    print("[startup] Pre-warming SHAP (numba JIT compilation)...")
    _dummy = pd.DataFrame(
        [np.zeros(len(_predictor.feature_names))],
        columns=_predictor.feature_names,
    )
    _predictor._explainer.shap_values(_dummy)
    print("[startup] SHAP pronto.")

    print(f"[startup] Carregando airport_reference de {airport_reference_path}...")
    suffix = airport_reference_path.suffix.lower()
    if suffix == ".csv":
        _airport_reference = pd.read_csv(airport_reference_path)
    else:
        _airport_reference = pd.read_parquet(airport_reference_path)
    print(f"[startup] {len(_airport_reference):,} aeroportos carregados.")


def get_predictor() -> FlightDelayPredictor:
    """Dependency: retorna o predictor já inicializado."""
    if _predictor is None:
        raise RuntimeError("Predictor não inicializado. Verifique o lifespan da API.")
    return _predictor


def get_airport_reference() -> pd.DataFrame:
    """Dependency: retorna a tabela de referência de aeroportos."""
    if _airport_reference is None:
        raise RuntimeError("airport_reference não inicializado. Verifique o lifespan da API.")
    return _airport_reference
