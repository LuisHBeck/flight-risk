"""
api.main
========
Aplicação FastAPI para o serviço de previsão de atrasos.

Inicialização
-------------
Todos os artefatos (modelo, encoders, lookup tables, airport_reference) são
carregados uma única vez no lifespan da aplicação — não há I/O por requisição.

Execução local
--------------
    uvicorn api.main:app --reload

Em produção
-----------
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints
---------
    GET  /health    — liveness / readiness check
    POST /predict/  — previsão de atraso para um voo
    GET  /docs      — Swagger UI (automático)
    GET  /redoc     — ReDoc (automático)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

import api.dependencies as _deps
from api.dependencies import init_resources
from api.routers import explain as explain_router
from api.routers import predict as predict_router
from api.routers import weather as weather_router


# ---------------------------------------------------------------------------
# Lifespan — carrega artefatos antes de aceitar requisições
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carrega modelo e tabelas na inicialização; libera recursos no shutdown.

    Em testes, os recursos podem ser injetados diretamente em
    ``api.dependencies`` antes de criar o TestClient — nesse caso o lifespan
    detecta que já estão preenchidos e não tenta ler os artefatos do disco.
    """
    if _deps._predictor is None or _deps._airport_reference is None:
        init_resources()
    yield
    # nada para liberar explicitamente (joblib/pandas geridos pelo GC)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Flight Risk API",
    description=(
        "Previsão binária de atraso na partida de voos domésticos brasileiros. "
        "Modelo: LightGBM treinado com dados ANAC 2022-2025 e meteorologia Open-Meteo."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict_router.router)
app.include_router(weather_router.router)
app.include_router(explain_router.router)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"], summary="Liveness / readiness check")
def health():
    """
    Retorna 200 quando a API está pronta para receber requisições.
    O modelo e os artefatos são carregados no startup — se este endpoint
    responder, tudo foi inicializado com sucesso.
    """
    return {"status": "ok"}
