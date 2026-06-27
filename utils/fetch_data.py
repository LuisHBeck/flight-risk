import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# --- Settings ---
BASE_URL              = os.environ.get("ANAC_BASE_URL")
START_YEAR            = int(os.environ.get("START_YEAR"))
END_YEAR              = int(os.environ.get("END_YEAR"))
OUR_AIRPORTS_DATA_URL = os.environ.get("OUR_AIRPORTS_DATA_URL")

RAW_DIR      = Path.cwd() / ".data" / "raw"
ANAC_DIR     = RAW_DIR / "anac"
AIRPORTS_DIR = RAW_DIR / "airports"

CONSOLIDATED_ANAC = RAW_DIR.parent / f"vra_{START_YEAR}_to_{END_YEAR}.csv"
CLEANED_ANAC      = RAW_DIR.parent / f"vra_{START_YEAR}_to_{END_YEAR}_clean.csv"
AIRPORTS_FILE     = AIRPORTS_DIR / "airports.csv"
AIRPORTS_REF_FILE = RAW_DIR.parent / "airports_reference.csv"
MERGED_FILE       = RAW_DIR.parent / f"vra_{START_YEAR}_to_{END_YEAR}_merged.csv"

MAX_WORKERS = 10

# Selected columns from ANAC data and their standardized names
COLUMNS_MAPPING = {
    "ICAO Empresa Aérea":     "airline_icao",
    "Código Tipo Linha":      "flight_type_code",
    "ICAO Aeródromo Origem":  "origin_icao",
    "ICAO Aeródromo Destino": "destination_icao",
    "Partida Prevista":       "dep_scheduled",
    "Partida Real":           "dep_actual",
    "Chegada Prevista":       "arr_scheduled",
    "Chegada Real":           "arr_actual",
    "Situação Voo":           "flight_status",
}

DATETIME_COLS = ["dep_scheduled", "dep_actual", "arr_scheduled", "arr_actual"]

# Columns used in the merged model dataset (excludes human-readable labels)
AIRPORTS_MODEL_COLS = [
    "ident",
    "type",
    "latitude_deg",
    "longitude_deg",
    "elevation_ft",
    "iso_region",
]

# All airport columns — used for the reference file
AIRPORTS_REF_COLS = [
    "ident",
    "type",
    "name",
    "latitude_deg",
    "longitude_deg",
    "elevation_ft",
    "municipality",
    "iso_region",
]


# ---------------------------------------------------------------------------
# ANAC — download and consolidation
# ---------------------------------------------------------------------------

def list_files(url):
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    return [
        a["href"] for a in soup.find_all("a") if a.get("href") and a["href"] != "../"
    ]


def normalize_filename(filename):
    match = re.match(r"(VRA_)(\d{4})(\d{1,2})(\.csv)", filename, re.IGNORECASE)
    if match:
        prefix, year, month, ext = match.groups()
        return f"{prefix}{year}{month.zfill(2)}{ext}"
    return filename


def download_file(url_file, destination):
    r = requests.get(url_file, timeout=60)
    if r.status_code == 200:
        destination.write_bytes(r.content)
        return True
    return False


def collect_tasks():
    """Walk the directory tree and return list of (url, destination) tuples."""
    tasks = []
    years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    all_folders = list_files(BASE_URL)
    year_folders = [f for f in all_folders if f.strip("/") in years]

    for year_folder in year_folders:
        url_year = BASE_URL + year_folder
        for month_folder in list_files(url_year):
            url_month = url_year + month_folder
            for filename in list_files(url_month):
                if not filename.lower().endswith(".csv"):
                    continue
                normalized = normalize_filename(filename)
                destination = ANAC_DIR / normalized
                if destination.exists():
                    continue
                tasks.append((url_month + filename, destination))
    return tasks


def download_all(tasks):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, url, dest): dest for url, dest in tasks
        }
        for future in as_completed(futures):
            dest = futures[future]
            try:
                success = future.result()
                status = "✓" if success else "HTTP error"
            except Exception as e:
                status = f"error: {e}"
            print(f"[{'OK' if status == '✓' else 'FAIL'}] {dest.name} {status}")


def consolidate():
    csv_files = sorted(
        f for f in ANAC_DIR.glob("VRA_??????.csv") if f != CONSOLIDATED_ANAC
    )
    if not csv_files:
        print("No CSV files found to consolidate.")
        return

    print(f"\nConsolidating {len(csv_files)} files...")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, sep=";", skiprows=1, dtype=str, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")

    if not dfs:
        print("No files could be read.")
        return

    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(CONSOLIDATED_ANAC, index=False, sep=";", encoding="utf-8")
    print(f"✓ Consolidated file saved to {CONSOLIDATED_ANAC} ({len(result):,} rows)")


# ---------------------------------------------------------------------------
# ANAC — cleaning
# ---------------------------------------------------------------------------

def clean():
    print("\nLoading consolidated ANAC data...")
    data = pd.read_csv(CONSOLIDATED_ANAC, sep=";", dtype=str, low_memory=False)
    rows_raw = len(data)
    print(f"  {rows_raw:,} rows loaded.")

    # Select and rename only the required columns
    data = data[list(COLUMNS_MAPPING.keys())].rename(columns=COLUMNS_MAPPING)

    # Keep only completed domestic flights
    data = data[data["flight_status"] == "REALIZADO"]
    data = data[data["flight_type_code"] == "N"]
    data = data.drop(columns=["flight_status", "flight_type_code"])

    # Remove duplicates
    data = data.drop_duplicates()

    # Drop rows with missing datetime values
    data = data.dropna(subset=DATETIME_COLS)

    # Strip whitespace from ICAO code columns
    for col in ["airline_icao", "origin_icao", "destination_icao"]:
        data[col] = data[col].str.strip()

    # Parse datetime columns
    data[DATETIME_COLS] = data[DATETIME_COLS].apply(
        lambda col: pd.to_datetime(col, format="ISO8601")
        .dt.floor("s")
        .astype("datetime64[ns]")
    )

    rows_clean = len(data)
    print(f"  {rows_raw - rows_clean:,} rows removed during cleaning ({rows_clean:,} remaining).")

    data.to_csv(CLEANED_ANAC, index=False, sep=";", encoding="utf-8")
    print(f"✓ Cleaned file saved to {CLEANED_ANAC}")


# ---------------------------------------------------------------------------
# OurAirports — download
# ---------------------------------------------------------------------------

def download_airports():
    """Download the OurAirports CSV if it does not already exist."""
    if AIRPORTS_FILE.exists():
        print("Airports file already exists, skipping download.")
        return

    print(f"Downloading airports data from {OUR_AIRPORTS_DATA_URL} ...")
    r = requests.get(OUR_AIRPORTS_DATA_URL, timeout=60)
    if r.status_code == 200:
        AIRPORTS_FILE.write_bytes(r.content)
        print(f"✓ Airports file saved to {AIRPORTS_FILE}")
    else:
        raise RuntimeError(
            f"Failed to download airports data (HTTP {r.status_code})"
        )


# ---------------------------------------------------------------------------
# Merge ANAC + OurAirports
# ---------------------------------------------------------------------------

def merge():
    print("\nLoading cleaned ANAC data...")
    data = pd.read_csv(CLEANED_ANAC, sep=";", low_memory=False)
    print(f"  {len(data):,} rows loaded.")

    print("Loading airports data...")
    airport = pd.read_csv(AIRPORTS_FILE, dtype=str, low_memory=False)

    # Model merge uses only the columns relevant for training (no name/city)
    airport_model = airport[AIRPORTS_MODEL_COLS].copy()

    # --- Merge 1: origin airport ---
    print("Merging origin airport data...")
    data = data.merge(
        airport_model.rename(
            columns={
                "ident":         "origin_icao",
                "type":          "origin_type",
                "latitude_deg":  "origin_lat",
                "longitude_deg": "origin_lon",
                "elevation_ft":  "origin_elevation_ft",
                "iso_region":    "origin_region",
            }
        ),
        on="origin_icao",
        how="left",
    )

    # --- Merge 2: destination airport ---
    print("Merging destination airport data...")
    data = data.merge(
        airport_model.rename(
            columns={
                "ident":         "destination_icao",
                "type":          "destination_type",
                "latitude_deg":  "destination_lat",
                "longitude_deg": "destination_lon",
                "elevation_ft":  "destination_elevation_ft",
                "iso_region":    "destination_region",
            }
        ),
        on="destination_icao",
        how="left",
    )

    data.to_csv(MERGED_FILE, index=False, sep=";", encoding="utf-8")
    print(f"✓ Merged file saved to {MERGED_FILE} ({len(data):,} rows)")

    # --- Airport reference file: only airports that matched after merge ---
    matched_origin = data.loc[data["origin_type"].notna(), "origin_icao"].unique()
    matched_dest   = data.loc[data["destination_type"].notna(), "destination_icao"].unique()
    matched_icaos  = pd.Index(matched_origin).union(matched_dest)
    airport_ref    = airport[airport["ident"].isin(matched_icaos)][AIRPORTS_REF_COLS].drop_duplicates()
    airport_ref.to_csv(AIRPORTS_REF_FILE, index=False, encoding="utf-8")
    print(f"✓ Airport reference file saved to {AIRPORTS_REF_FILE} ({len(airport_ref):,} airports)")

    # Coverage report
    origin_matched = data["origin_type"].notna().sum()
    dest_matched   = data["destination_type"].notna().sum()
    total          = len(data)
    print(
        f"\nCoverage report:"
        f"\n  Origin airports matched:      {origin_matched:,} / {total:,} ({origin_matched / total:.1%})"
        f"\n  Destination airports matched: {dest_matched:,} / {total:,} ({dest_matched / total:.1%})"
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create directory structure
    ANAC_DIR.mkdir(parents=True, exist_ok=True)
    AIRPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ANAC — download and consolidation
    print("=== ANAC: Scanning directories... ===")
    tasks = collect_tasks()
    if tasks:
        print(f"Downloading {len(tasks)} files with {MAX_WORKERS} workers...\n")
        download_all(tasks)
    else:
        print("All ANAC files already downloaded.")
    consolidate()

    # 2. ANAC — cleaning
    print("\n=== Cleaning ===")
    clean()

    # 3. OurAirports
    print("\n=== OurAirports ===")
    download_airports()

    # 4. Merge
    print("\n=== Merge ===")
    merge()
