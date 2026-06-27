"""
api.schemas
===========
Pydantic models for the /predict endpoint.

FlightInput   — o que o cliente envia (apenas os campos que ele conhece)
PredictionResponse — o que a API devolve (predição + features calculadas)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class FlightInput(BaseModel):
    """
    Campos necessários para prever o atraso de um voo.

    O cliente só precisa enviar os dados operacionais que ele tem *antes* da
    partida — coordenadas, tipo e região dos aeroportos são preenchidos
    automaticamente a partir da tabela de referência carregada na API.

    Exemplo de payload::

        {
            "airline_icao": "GLO",
            "origin_icao": "SBGR",
            "destination_icao": "SBRJ",
            "dep_scheduled": "2025-08-10T06:00:00",
            "arr_scheduled": "2025-08-10T07:05:00",
            "origin_wx_weathercode": 0,
            "destination_wx_weathercode": 61
        }
    """

    # Identificadores operacionais
    airline_icao:      str = Field(..., examples=["GLO"], description="ICAO da companhia aérea.")
    origin_icao:       str = Field(..., examples=["SBGR"], description="ICAO do aeroporto de origem.")
    destination_icao:  str = Field(..., examples=["SBRJ"], description="ICAO do aeroporto de destino.")

    # Horários programados
    dep_scheduled: datetime = Field(..., description="Horário programado de partida (ISO 8601).")
    arr_scheduled: datetime = Field(..., description="Horário programado de chegada (ISO 8601).")

    # Código WMO de tempo (Open-Meteo)
    # https://open-meteo.com/en/docs#weathervariables
    origin_wx_weathercode:      int = Field(..., ge=0, le=99, description="Código WMO de tempo na origem.")
    destination_wx_weathercode: int = Field(..., ge=0, le=99, description="Código WMO de tempo no destino.")

    # Variáveis numéricas de clima — origem
    origin_wx_temperature_2m:   float = Field(..., description="Temperatura a 2 m na origem (°C).")
    origin_wx_precipitation:    float = Field(..., ge=0, description="Precipitação na origem (mm).")
    origin_wx_windspeed_10m:    float = Field(..., ge=0, description="Velocidade do vento a 10 m na origem (km/h).")
    origin_wx_windgusts_10m:    float = Field(..., ge=0, description="Rajadas de vento a 10 m na origem (km/h).")
    origin_wx_cloudcover:       float = Field(..., ge=0, le=100, description="Cobertura de nuvens na origem (%).")
    origin_wx_surface_pressure: float = Field(..., ge=0, description="Pressão atmosférica na origem (hPa).")

    # Variáveis numéricas de clima — destino
    destination_wx_temperature_2m:   float = Field(..., description="Temperatura a 2 m no destino (°C).")
    destination_wx_precipitation:    float = Field(..., ge=0, description="Precipitação no destino (mm).")
    destination_wx_windspeed_10m:    float = Field(..., ge=0, description="Velocidade do vento a 10 m no destino (km/h).")
    destination_wx_windgusts_10m:    float = Field(..., ge=0, description="Rajadas de vento a 10 m no destino (km/h).")
    destination_wx_cloudcover:       float = Field(..., ge=0, le=100, description="Cobertura de nuvens no destino (%).")
    destination_wx_surface_pressure: float = Field(..., ge=0, description="Pressão atmosférica no destino (hPa).")

    # Limiar de decisão (opcional — default 0.5)
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Limiar de P(atraso) para classificar como atrasado. Default: 0.5.",
    )

    @model_validator(mode="after")
    def arr_after_dep(self) -> "FlightInput":
        if self.arr_scheduled <= self.dep_scheduled:
            raise ValueError("arr_scheduled deve ser posterior a dep_scheduled.")
        return self


class TemporalFeatures(BaseModel):
    """Features temporais calculadas a partir do horário de partida."""
    dep_hour_sin:    float
    dep_hour_cos:    float
    dep_dow_sin:     float
    dep_dow_cos:     float
    dep_month_sin:   float
    dep_month_cos:   float
    dep_time_block:  str
    dep_is_peak_hour: int
    dep_is_weekend:  int
    dep_is_holiday:  int
    dep_day_of_year: int


class RouteFeatures(BaseModel):
    """Features de rota e aeroporto."""
    route:                    str
    region_pair:              str
    distance_km:              float
    flight_range:             str
    elevation_diff_ft:        float
    origin_airport_size:      int
    destination_airport_size: int
    is_trunk_route:           int
    scheduled_duration_min:   float


class CongestionFeatures(BaseModel):
    """Proxies de congestionamento horário."""
    origin_hourly_flights:       int
    destination_hourly_arrivals: int
    total_hourly_congestion:     int


class WeatherFeatures(BaseModel):
    """Features de tempo: variáveis numéricas + categorias derivadas do código WMO."""
    # Numéricas (passadas diretamente ao modelo)
    origin_wx_temperature_2m:        float
    origin_wx_precipitation:         float
    origin_wx_windspeed_10m:         float
    origin_wx_windgusts_10m:         float
    origin_wx_cloudcover:            float
    origin_wx_surface_pressure:      float
    destination_wx_temperature_2m:   float
    destination_wx_precipitation:    float
    destination_wx_windspeed_10m:    float
    destination_wx_windgusts_10m:    float
    destination_wx_cloudcover:       float
    destination_wx_surface_pressure: float
    # Derivadas do weathercode
    origin_wx_condition:      str
    destination_wx_condition: str
    origin_wx_is_fog:         int
    origin_wx_is_rain:        int
    origin_wx_is_storm:       int
    destination_wx_is_fog:    int
    destination_wx_is_rain:   int
    destination_wx_is_storm:  int


class LookupFeatures(BaseModel):
    """Estatísticas históricas de rota e companhia×hora."""
    route_hist_delay_mean:   float
    route_hist_delay_std:    float
    route_hist_delay_rate:   float
    airline_hour_delay_rate: float
    airline_hour_delay_mean: float


class ComputedFeatures(BaseModel):
    """
    Todas as features calculadas durante a inferência.
    Útil para debug, auditoria e explicabilidade.
    """
    temporal:   TemporalFeatures
    route:      RouteFeatures
    congestion: CongestionFeatures
    weather:    WeatherFeatures
    lookup:     LookupFeatures


class PredictionResponse(BaseModel):
    """Resposta do endpoint POST /predict."""

    # Eco dos campos de identificação do input
    airline_icao:     str
    origin_icao:      str
    destination_icao: str
    dep_scheduled:    datetime
    arr_scheduled:    datetime

    # Resultado da predição
    delay_proba:       float = Field(..., description="P(atraso > threshold) em [0, 1].")
    predicted_delayed: int   = Field(..., description="1 = atrasado, 0 = pontual.")
    threshold:         float = Field(..., description="Limiar utilizado para o rótulo binário.")

    # Features calculadas (auditoria)
    features: ComputedFeatures
