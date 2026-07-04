"""
api.routers.weather
====================
Endpoint GET /weather — busca a previsão meteorológica do Open-Meteo para os
aeroportos de origem e destino e devolve exatamente as variáveis que o
endpoint POST /predict/ espera no payload.

Fluxo:
    1. Recebe origin_icao, destination_icao e dep_scheduled (query params)
    2. Busca as coordenadas de cada aeroporto na airport_reference em memória
    3. Chama o Open-Meteo Forecast API uma única vez com as duas localizações
       (a API suporta múltiplos pares lat/lon no mesmo request)
    4. Extrai os valores horários mais próximos do horário de partida
    5. Devolve um dict pronto para ser colado no payload do /predict/

Open-Meteo Forecast API (gratuita, sem autenticação):
    https://open-meteo.com/en/docs
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.dependencies import get_airport_reference

router = APIRouter(prefix="/weather", tags=["weather"])

# ---------------------------------------------------------------------------
# Variáveis solicitadas ao Open-Meteo (mesmas usadas no treino)
# ---------------------------------------------------------------------------

_HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "windgusts_10m",
    "cloudcover",
    "surface_pressure",
    "weathercode",
]

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Schema de resposta
# ---------------------------------------------------------------------------

class AirportWeather(BaseModel):
    """Variáveis meteorológicas para um aeroporto no horário de partida."""
    weathercode:      int
    temperature_2m:   float
    precipitation:    float
    windspeed_10m:    float
    windgusts_10m:    float
    cloudcover:       float
    surface_pressure: float


class WeatherResponse(BaseModel):
    """
    Resposta do GET /weather.

    Os campos ``origin_wx_*`` e ``destination_wx_*`` estão prontos para
    serem incluídos diretamente no payload do POST /predict/.
    """
    origin_icao:      str
    destination_icao: str
    dep_scheduled:    datetime
    valid_time:       datetime   # horário real usado (hora cheia mais próxima)

    # Campos no formato exato esperado pelo /predict/
    origin_wx_weathercode:           int
    origin_wx_temperature_2m:        float
    origin_wx_precipitation:         float
    origin_wx_windspeed_10m:         float
    origin_wx_windgusts_10m:         float
    origin_wx_cloudcover:            float
    origin_wx_surface_pressure:      float

    destination_wx_weathercode:           int
    destination_wx_temperature_2m:        float
    destination_wx_precipitation:         float
    destination_wx_windspeed_10m:         float
    destination_wx_windgusts_10m:         float
    destination_wx_cloudcover:            float
    destination_wx_surface_pressure:      float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coords(airport_ref: pd.DataFrame, icao: str) -> tuple[float, float]:
    """Retorna (lat, lon) de um ICAO a partir da airport_reference."""
    row = airport_ref[airport_ref["ident"] == icao]
    if row.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ICAO '{icao}' não encontrado na airport_reference.",
        )
    return float(row.iloc[0]["latitude_deg"]), float(row.iloc[0]["longitude_deg"])


def _nearest_hour_index(times: list[str], target: datetime) -> int:
    """Retorna o índice da hora mais próxima de `target` na lista ISO de horários."""
    parsed = [datetime.fromisoformat(t) for t in times]
    # garante que target é naive para comparação
    if target.tzinfo is not None:
        target = target.replace(tzinfo=None)
    deltas = [abs((t - target).total_seconds()) for t in parsed]
    return deltas.index(min(deltas))


def _extract(hourly: dict, var: str, idx: int) -> float | int:
    """Extrai o valor de uma variável horária pelo índice."""
    val = hourly[var][idx]
    if val is None:
        return 0.0
    return val


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=WeatherResponse,
    summary="Buscar previsão meteorológica para dois aeroportos",
    description="""
Chama o **Open-Meteo Forecast API** (gratuito, sem autenticação) e retorna as
variáveis de clima para os aeroportos de origem e destino no horário de
partida informado.

A resposta está no formato exato esperado pelo **POST /predict/**, então basta
copiá-la diretamente no payload de predição.

**Observação:** o Open-Meteo fornece previsões horárias para os próximos 7–16
dias. Para horários históricos use o endpoint `/weather/historical` (não
implementado aqui — requereria a Historical Weather API do Open-Meteo).
""",
)
async def get_weather(
    origin_icao:      str      = Query(..., description="ICAO do aeroporto de origem."),
    destination_icao: str      = Query(..., description="ICAO do aeroporto de destino."),
    dep_scheduled:    datetime = Query(..., description="Horário programado de partida (ISO 8601)."),
    airport_ref: pd.DataFrame  = Depends(get_airport_reference),
) -> WeatherResponse:

    # 1. Coordenadas
    orig_lat, orig_lon = _get_coords(airport_ref, origin_icao)
    dest_lat, dest_lon = _get_coords(airport_ref, destination_icao)

    # 2. Chamar Open-Meteo com os dois aeroportos num único request
    params = {
        "latitude":         f"{orig_lat},{dest_lat}",
        "longitude":        f"{orig_lon},{dest_lon}",
        "hourly":           ",".join(_HOURLY_VARS),
        "forecast_days":    16,
        "timezone":         "America/Sao_Paulo",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Open-Meteo não respondeu no tempo esperado. Tente novamente.",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erro ao chamar Open-Meteo: {exc.response.status_code}",
        )

    # 3. A API retorna lista quando há múltiplos locais
    #    Se um único local, retorna dict; garantir que seja lista.
    if isinstance(data, dict):
        data = [data]

    if len(data) < 2:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Open-Meteo retornou dados incompletos para os dois aeroportos.",
        )

    orig_hourly = data[0]["hourly"]
    dest_hourly = data[1]["hourly"]

    # 4. Índice da hora mais próxima ao horário de partida
    times = orig_hourly["time"]   # ambos têm o mesmo grid de horas
    idx = _nearest_hour_index(times, dep_scheduled)
    valid_time = datetime.fromisoformat(times[idx])

    # 5. Montar resposta
    return WeatherResponse(
        origin_icao=origin_icao,
        destination_icao=destination_icao,
        dep_scheduled=dep_scheduled,
        valid_time=valid_time,

        origin_wx_weathercode=          int(_extract(orig_hourly, "weathercode",      idx)),
        origin_wx_temperature_2m=       float(_extract(orig_hourly, "temperature_2m",   idx)),
        origin_wx_precipitation=        float(_extract(orig_hourly, "precipitation",    idx)),
        origin_wx_windspeed_10m=        float(_extract(orig_hourly, "windspeed_10m",    idx)),
        origin_wx_windgusts_10m=        float(_extract(orig_hourly, "windgusts_10m",    idx)),
        origin_wx_cloudcover=           float(_extract(orig_hourly, "cloudcover",       idx)),
        origin_wx_surface_pressure=     float(_extract(orig_hourly, "surface_pressure", idx)),

        destination_wx_weathercode=     int(_extract(dest_hourly, "weathercode",      idx)),
        destination_wx_temperature_2m=  float(_extract(dest_hourly, "temperature_2m",   idx)),
        destination_wx_precipitation=   float(_extract(dest_hourly, "precipitation",    idx)),
        destination_wx_windspeed_10m=   float(_extract(dest_hourly, "windspeed_10m",    idx)),
        destination_wx_windgusts_10m=   float(_extract(dest_hourly, "windgusts_10m",    idx)),
        destination_wx_cloudcover=      float(_extract(dest_hourly, "cloudcover",       idx)),
        destination_wx_surface_pressure=float(_extract(dest_hourly, "surface_pressure", idx)),
    )
