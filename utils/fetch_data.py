import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("ANAC_BASE_URL")
START_YEAR = int(os.environ.get("START_YEAR"))
END_YEAR = int(os.environ.get("END_YEAR"))
OUTPUT_DIR = Path.cwd() / ".data" / "raw"
CONSOLIDATED_FILE = OUTPUT_DIR.parent / f"vra_{START_YEAR}_to_{END_YEAR}.csv"
MAX_WORKERS = 10


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
                destination = OUTPUT_DIR / normalized
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
        f for f in OUTPUT_DIR.glob("VRA_??????.csv") if f != CONSOLIDATED_FILE
    )
    if not csv_files:
        print("No CSV files found to consolidate.")
        return

    print(f"\nConsolidating {len(csv_files)} files...")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(
                f, sep=";", skiprows=1, dtype=str, low_memory=False
            )
            dfs.append(df)
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")

    if not dfs:
        print("No files could be read.")
        return

    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(CONSOLIDATED_FILE, index=False, sep=";", encoding="utf-8")
    print(f"✓ Consolidated file saved to {CONSOLIDATED_FILE} ({len(result):,} rows)")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning directories...")
    tasks = collect_tasks()

    if tasks:
        print(f"Downloading {len(tasks)} files with {MAX_WORKERS} workers...\n")
        download_all(tasks)
    else:
        print("All files already downloaded.")

    consolidate()
