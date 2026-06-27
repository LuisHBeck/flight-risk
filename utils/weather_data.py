"""
Pipeline: ANAC VRAA + Open-Meteo
==================================
Steps:
  1. Load ANAC dataset (already with airport coordinates)
  2. Check and remove rows without coordinates (if any)
  3. Download historical weather data (Open-Meteo) per airport
  4. Join weather data with flights (origin + destination)
  5. Export final dataset as parquet and CSV

Requirements:
  pip install pandas openmeteo-requests requests-cache retry-requests pyarrow
"""

import time
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS — adjust if needed
# ─────────────────────────────────────────────
DATA_DIR         = Path.cwd() / ".data"

ANAC_PATH        = DATA_DIR / "vra_2022_to_2025_merged.csv"

CACHE_DIR_CLIMA  = DATA_DIR / "cache" / "cache_weather"       # Per-airport parquet cache
CACHE_OPENMETEO  = DATA_DIR / "cache" / "cache_openmeteo"   # HTTP request cache

WEATHER_DIR      = DATA_DIR / "raw" / "weather"
OUTPUT_WEATHER   = WEATHER_DIR / "weather_airports.parquet"  # Consolidated weather cache

OUTPUT_FINAL_PQ  = DATA_DIR / "flights_with_weather.parquet"
OUTPUT_FINAL_CSV = DATA_DIR / "flights_with_weather.csv"

WEATHER_VARIABLES = [
    "temperature_2m",   # Temperature (°C)
    "precipitation",    # Precipitation (mm)
    "windspeed_10m",    # Wind speed (km/h)
    "windgusts_10m",    # Wind gusts (km/h)
    "visibility",       # Visibility (m)
    "cloudcover",       # Cloud cover (%)
    "weathercode",      # WMO code (0=clear, 95+=thunderstorm)
    "surface_pressure", # Atmospheric pressure (hPa)
]


# ─────────────────────────────────────────────
# 1. LOAD ANAC DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading ANAC dataset...")
print("=" * 60)

data = pd.read_csv(ANAC_PATH, sep=";", low_memory=False)
print(f"  Original shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")

# Parse datetime columns
data["dep_scheduled"] = pd.to_datetime(data["dep_scheduled"])
data["arr_scheduled"] = pd.to_datetime(data["arr_scheduled"])

# Dynamic date range based on the dataset
WEATHER_START = data["dep_scheduled"].min().strftime("%Y-%m-%d")
WEATHER_END   = data["dep_scheduled"].max().strftime("%Y-%m-%d")
print(f"  Detected period: {WEATHER_START} → {WEATHER_END}")


# ─────────────────────────────────────────────
# 2. VALIDATE COORDINATES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Validating coordinates in dataset...")
print("=" * 60)

null_origin = data["origin_lat"].isnull().sum()
null_dest   = data["destination_lat"].isnull().sum()
print(f"  Null origin_lat:      {null_origin}")
print(f"  Null destination_lat: {null_dest}")

if null_origin > 0 or null_dest > 0:
    before = len(data)
    data = data.dropna(subset=["origin_lat", "origin_lon", "destination_lat", "destination_lon"])
    print(f"  Dropped: {before - len(data)} rows without coordinates")

print(f"  Shape after validation: {data.shape}")


# ─────────────────────────────────────────────
# 3. DOWNLOAD WEATHER FROM OPEN-METEO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Downloading weather data...")
print("=" * 60)

# Create directories
CACHE_DIR_CLIMA.mkdir(parents=True, exist_ok=True)
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

# Setup Open-Meteo client with cache + retry
cache_session  = requests_cache.CachedSession(str(CACHE_OPENMETEO), expire_after=-1)
retry_session  = retry(cache_session, retries=3, backoff_factor=1.0)
openmeteo      = openmeteo_requests.Client(session=retry_session)


def fetch_weather(icao: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": WEATHER_START,
        "end_date":   WEATHER_END,
        "hourly":     WEATHER_VARIABLES,
        "timezone":   "America/Sao_Paulo",
    }
    responses = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    r      = responses[0]
    hourly = r.Hourly()

    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True)
                    .tz_convert("America/Sao_Paulo"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
                  .tz_convert("America/Sao_Paulo"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        **{var: hourly.Variables(i).ValuesAsNumpy()
           for i, var in enumerate(WEATHER_VARIABLES)},
    })
    df["icao"]     = icao
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df


# Unique airports (origin + destination)
orig = data[["origin_icao", "origin_lat", "origin_lon"]].drop_duplicates()
orig.columns = ["icao", "lat", "lon"]
dest = data[["destination_icao", "destination_lat", "destination_lon"]].drop_duplicates()
dest.columns = ["icao", "lat", "lon"]
unique_airports = pd.concat([orig, dest]).drop_duplicates(subset="icao").reset_index(drop=True)

total  = len(unique_airports)
errors = []
print(f"  Total unique airports: {total}")

for i, row in unique_airports.iterrows():
    icao       = row["icao"]
    cache_file = CACHE_DIR_CLIMA / f"{icao}.parquet"

    # Already downloaded — skip
    if cache_file.exists():
        print(f"  [{i+1}/{total}] {icao} — cached, skipping.")
        continue

    print(f"  [{i+1}/{total}] Fetching {icao}...", end=" ", flush=True)
    try:
        df = fetch_weather(icao, row["lat"], row["lon"])
        df.to_parquet(cache_file, index=False)
        print(f"OK ({len(df)} records)")
        time.sleep(0.5)

    except Exception as e:
        errors.append(icao)
        msg = str(e)

        if "Hourly API request limit exceeded" in msg:
            # First retry: wait 5 minutes
            print("RATE LIMIT — waiting 5 minutes before retrying...")
            time.sleep(300)
            try:
                df = fetch_weather(icao, row["lat"], row["lon"])
                df.to_parquet(cache_file, index=False)
                errors.remove(icao)
                print(f"  [{i+1}/{total}] {icao} — OK after 5 min ({len(df)} records)")
                time.sleep(0.5)
            except Exception:
                # Second retry: wait 1 hour
                print("  Still rate limited — waiting 60 minutes...")
                for remaining in range(60, 0, -10):
                    print(f"    Resuming in {remaining} minutes...")
                    time.sleep(600)
                try:
                    df = fetch_weather(icao, row["lat"], row["lon"])
                    df.to_parquet(cache_file, index=False)
                    errors.remove(icao)
                    print(f"  [{i+1}/{total}] {icao} — OK after 1h ({len(df)} records)")
                    time.sleep(0.5)
                except Exception as e3:
                    print(f"  [{i+1}/{total}] {icao} — permanent error: {e3}")
        else:
            print(f"ERROR: {e}")

# Consolidate all per-airport parquets into one file
parquets = list(CACHE_DIR_CLIMA.glob("*.parquet"))
print(f"\n  Consolidating {len(parquets)} cached files...")
weather = pd.concat([pd.read_parquet(f) for f in parquets]).reset_index(drop=True)

if errors:
    print(f"  ⚠️  Airports with errors (no data): {errors}")

weather.to_parquet(OUTPUT_WEATHER, index=False)
print(f"  Weather saved to '{OUTPUT_WEATHER}' — shape: {weather.shape}")


# ─────────────────────────────────────────────
# 4. JOIN WEATHER + FLIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Joining weather data with flights...")
print("=" * 60)

# Truncate to hour (Open-Meteo granularity)
data["dep_hour"] = data["dep_scheduled"].dt.floor("h")
data["arr_hour"] = data["arr_scheduled"].dt.floor("h")

weather["datetime"] = pd.to_datetime(weather["datetime"]).dt.floor("h")

# ── Join weather at ORIGIN (departure time) ──
weather_origin = weather.rename(columns={
    "icao":     "origin_icao",
    "datetime": "dep_hour",
    **{v: f"origin_wx_{v}" for v in WEATHER_VARIABLES},
})
data = data.merge(weather_origin, on=["origin_icao", "dep_hour"], how="left")

# ── Join weather at DESTINATION (scheduled arrival time) ──
weather_dest = weather.rename(columns={
    "icao":     "destination_icao",
    "datetime": "arr_hour",
    **{v: f"destination_wx_{v}" for v in WEATHER_VARIABLES},
})
data = data.merge(weather_dest, on=["destination_icao", "arr_hour"], how="left")

print(f"  Final shape: {data.shape}")

wx_cols = (
    [f"origin_wx_{v}"      for v in WEATHER_VARIABLES] +
    [f"destination_wx_{v}" for v in WEATHER_VARIABLES]
)
print("\n  Nulls in weather features:")
print(data[wx_cols].isnull().sum().to_string())


# ─────────────────────────────────────────────
# 5. EXPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Exporting final dataset...")
print("=" * 60)

# Drop auxiliary hour columns
data = data.drop(columns=["dep_hour", "arr_hour", "destination_wx_visibility", "origin_wx_visibility"])

data.to_parquet(OUTPUT_FINAL_PQ, index=False)
print(f"  Parquet saved: '{OUTPUT_FINAL_PQ}'")

# data.to_csv(OUTPUT_FINAL_CSV, index=False, sep=";")
# print(f"  CSV saved:     '{OUTPUT_FINAL_CSV}'")

print("\n✅ Pipeline completed successfully!")
print(f"   Rows: {len(data):,}  |  Columns: {data.shape[1]}")
print(f"   Weather features added: {len(wx_cols)}")
