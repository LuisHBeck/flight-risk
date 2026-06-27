"""
api.routers.predict
===================
Router com o endpoint POST /predict.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from flight_risk.predict import FlightDelayPredictor, enrich_airports

from api.dependencies import get_predictor, get_airport_reference
from api.schemas import (
    FlightInput,
    PredictionResponse,
    ComputedFeatures,
    TemporalFeatures,
    RouteFeatures,
    CongestionFeatures,
    WeatherFeatures,
    LookupFeatures,
)

router = APIRouter(prefix="/predict", tags=["prediction"])


# ---------------------------------------------------------------------------
# Colunas de cada grupo de features (para montar o ComputedFeatures)
# ---------------------------------------------------------------------------

_TEMPORAL_COLS = [
    "dep_hour_sin", "dep_hour_cos",
    "dep_dow_sin", "dep_dow_cos",
    "dep_month_sin", "dep_month_cos",
    "dep_time_block", "dep_is_peak_hour",
    "dep_is_weekend", "dep_is_holiday", "dep_day_of_year",
]

_ROUTE_COLS = [
    "route", "region_pair", "distance_km", "flight_range",
    "elevation_diff_ft", "origin_airport_size", "destination_airport_size",
    "is_trunk_route", "scheduled_duration_min",
]

_CONGESTION_COLS = [
    "origin_hourly_flights", "destination_hourly_arrivals", "total_hourly_congestion",
]

_WEATHER_COLS = [
    # Numéricas
    "origin_wx_temperature_2m", "origin_wx_precipitation",
    "origin_wx_windspeed_10m",  "origin_wx_windgusts_10m",
    "origin_wx_cloudcover",     "origin_wx_surface_pressure",
    "destination_wx_temperature_2m", "destination_wx_precipitation",
    "destination_wx_windspeed_10m",  "destination_wx_windgusts_10m",
    "destination_wx_cloudcover",     "destination_wx_surface_pressure",
    # Derivadas do weathercode
    "origin_wx_condition", "destination_wx_condition",
    "origin_wx_is_fog", "origin_wx_is_rain", "origin_wx_is_storm",
    "destination_wx_is_fog", "destination_wx_is_rain", "destination_wx_is_storm",
]

_LOOKUP_COLS = [
    "route_hist_delay_mean", "route_hist_delay_std", "route_hist_delay_rate",
    "airline_hour_delay_rate", "airline_hour_delay_mean",
]


def _row_to_computed_features(row: pd.Series) -> ComputedFeatures:
    """Extrai uma linha do DataFrame enriquecido e monta o ComputedFeatures."""
    def _get(col):
        val = row[col]
        # converte numpy scalars para tipos Python nativos
        if hasattr(val, "item"):
            return val.item()
        return val

    return ComputedFeatures(
        temporal=TemporalFeatures(**{c: _get(c) for c in _TEMPORAL_COLS}),
        route=RouteFeatures(**{c: _get(c) for c in _ROUTE_COLS}),
        congestion=CongestionFeatures(**{c: _get(c) for c in _CONGESTION_COLS}),
        weather=WeatherFeatures(**{c: _get(c) for c in _WEATHER_COLS}),
        lookup=LookupFeatures(**{c: _get(c) for c in _LOOKUP_COLS}),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Prever probabilidade de atraso de um voo",
    description="""
Recebe os dados operacionais de um voo *antes* da partida e retorna:

- **delay_proba**: probabilidade de atraso na partida (P > threshold = atrasado)
- **predicted_delayed**: rótulo binário (1 = atrasado, 0 = pontual)
- **features**: todas as features calculadas pela pipeline (temporal, rota,
  congestionamento, clima, estatísticas históricas) — útil para debug e
  explicabilidade

As coordenadas, tipo e região dos aeroportos são preenchidos automaticamente
a partir da tabela de referência carregada na inicialização da API.
""",
)
def predict_flight(
    flight: FlightInput,
    predictor:   FlightDelayPredictor = Depends(get_predictor),
    airport_ref: pd.DataFrame         = Depends(get_airport_reference),
) -> PredictionResponse:

    # 1. Montar DataFrame com uma linha
    row = pd.DataFrame([{
        "airline_icao":               flight.airline_icao,
        "origin_icao":                flight.origin_icao,
        "destination_icao":           flight.destination_icao,
        "dep_scheduled":              flight.dep_scheduled,
        "arr_scheduled":              flight.arr_scheduled,
        # WMO code (usado para derivar condição + flags)
        "origin_wx_weathercode":               flight.origin_wx_weathercode,
        "destination_wx_weathercode":          flight.destination_wx_weathercode,
        # Variáveis numéricas (passadas diretamente ao modelo)
        "origin_wx_temperature_2m":            flight.origin_wx_temperature_2m,
        "origin_wx_precipitation":             flight.origin_wx_precipitation,
        "origin_wx_windspeed_10m":             flight.origin_wx_windspeed_10m,
        "origin_wx_windgusts_10m":             flight.origin_wx_windgusts_10m,
        "origin_wx_cloudcover":                flight.origin_wx_cloudcover,
        "origin_wx_surface_pressure":          flight.origin_wx_surface_pressure,
        "destination_wx_temperature_2m":       flight.destination_wx_temperature_2m,
        "destination_wx_precipitation":        flight.destination_wx_precipitation,
        "destination_wx_windspeed_10m":        flight.destination_wx_windspeed_10m,
        "destination_wx_windgusts_10m":        flight.destination_wx_windgusts_10m,
        "destination_wx_cloudcover":           flight.destination_wx_cloudcover,
        "destination_wx_surface_pressure":     flight.destination_wx_surface_pressure,
    }])

    # 2. Enriquecer com dados dos aeroportos
    try:
        row = enrich_airports(row, airport_ref)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # 3. Construir features uma única vez — reutilizadas para auditoria e para o score
    try:
        enriched = predictor.build_features(row)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na engenharia de features: {exc}",
        )

    # 4. Probabilidade e rótulo — direto do enriched, sem reconstruir features
    try:
        proba = float(predictor.predict_proba_from_enriched(enriched)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na predição: {exc}",
        )

    label = int(proba >= flight.threshold)

    # 5. Montar resposta
    return PredictionResponse(
        airline_icao=flight.airline_icao,
        origin_icao=flight.origin_icao,
        destination_icao=flight.destination_icao,
        dep_scheduled=flight.dep_scheduled,
        arr_scheduled=flight.arr_scheduled,
        delay_proba=round(proba, 6),
        predicted_delayed=label,
        threshold=flight.threshold,
        features=_row_to_computed_features(enriched.iloc[0]),
    )
