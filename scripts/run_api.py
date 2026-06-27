"""
scripts/run_api.py
==================
Inicia o servidor uvicorn com a aplicação FastAPI.

Uso
---
    python -m scripts.run_api
    python -m scripts.run_api --host 0.0.0.0 --port 8000 --workers 4

Ou, após `pip install -e .`:
    flight-risk-api
"""

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Flight Risk API server.")
    parser.add_argument("--host",    default="127.0.0.1",  help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port",    default=8000, type=int, help="Bind port. Default: 8000")
    parser.add_argument("--workers", default=1,    type=int, help="Número de workers. Default: 1")
    parser.add_argument("--reload",  action="store_true",   help="Ativar hot-reload (dev only).")
    args = parser.parse_args()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
