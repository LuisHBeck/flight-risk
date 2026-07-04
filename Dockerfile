FROM python:3.12-slim

WORKDIR /app

# Install build deps needed by LightGBM / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY flight_risk/ flight_risk/
COPY api/ api/
COPY scripts/ scripts/

RUN pip install --no-cache-dir .

COPY .models/ .models/

COPY .data/airports_reference.csv .data/airports_reference.csv
COPY .data/route_stats.pkl        .data/route_stats.pkl
COPY .data/airline_hour_stats.pkl .data/airline_hour_stats.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
