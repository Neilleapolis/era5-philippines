# scripts/01_download_era5.py
# =============================================================================
# STAGE 2 - Download ERA5 data for all city-year-month combinations
#
# Usage:
#   python scripts/01_download_era5.py                # full batch (144 requests)
#   python scripts/01_download_era5.py --test         # Muntinlupa 2023 Jan only
#   python scripts/01_download_era5.py --city cebu    # one city only, all years
#
# Why monthly? CDS cost limit is 120,000 items per request.
# A full year x 15 vars x 4 grid cells = 525,600 items -- rejected.
# One month x 15 vars x 4 grid cells = ~44,640 items -- accepted.
#
# Output: data/raw/{city}/{year}_{month:02d}.nc  (144 files total)
# =============================================================================

import sys
import time
import zipfile
import shutil
import tempfile
import platform
from pathlib import Path
from datetime import datetime

import cdsapi
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.config import (
    CITIES,
    YEARS,
    ERA5_DATASET,
    ERA5_VARIABLES_LONG,
    ERA5_VARIABLES_SHORT,
    RAW_DATA_DIR,
    DOCS_DIR,
)

MAX_RETRIES  = 3
RETRY_DELAYS = [60, 300, 900]

MONTHS = [f"{m:02d}" for m in range(1, 13)]

LOG_PATH      = DOCS_DIR / "download_log.txt"
DECISIONS_LOG = DOCS_DIR / "decisions_log.md"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_package_version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "not installed"


def write_decisions_entry(mode, results):
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_hdr = datetime.now().strftime("%Y-%m-%d")

    counts = {}
    for status in results.values():
        counts[status] = counts.get(status, 0) + 1
    summary_line = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    env_lines = [
        f"- **Timestamp:** {now}",
        f"- **Mode:** {mode}",
        f"- **CDS API (cdsapi):** {get_package_version('cdsapi')}",
        f"- **Python:** {platform.python_version()}",
        f"- **OS:** {platform.system()} {platform.release()} ({platform.version()})",
        f"- **xarray:** {get_package_version('xarray')}",
        f"- **netcdf4:** {get_package_version('netcdf4')}",
        f"- **dask:** {get_package_version('dask')}",
        f"- **pandas:** {get_package_version('pandas')}",
        f"- **numpy:** {get_package_version('numpy')}",
    ]

    # Only include non-SUCCESS results in the details, to keep the log readable
    non_success = {k: v for k, v in results.items() if v != "SUCCESS"}
    if non_success:
        result_lines = [f"  - `{k}`: {v}" for k, v in sorted(non_success.items())]
        detail_block = "**Non-SUCCESS results:**\n" + "\n".join(result_lines)
    else:
        detail_block = "**All requests succeeded.**"

    entry = f"""
---

## {date_hdr} — ERA5 download ({mode})

**Environment**
{chr(10).join(env_lines)}

**Summary:** {summary_line} (total {len(results)} requests)

{detail_block}

**Raw log:** `docs/download_log.txt`

**Notes:** _TODO: any anomalies, retries, license issues, deviations_

"""

    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(entry)

    log(f"Decisions log updated: {DECISIONS_LOG}")


# -----------------------------------------------------------------------------
# Request building
# -----------------------------------------------------------------------------
def build_request(city_name, year, month):
    """Construct CDS API request for one city, one year, one month."""
    city = CITIES[city_name]
    return {
        "product_type": "reanalysis",
        "format":       "netcdf",
        "variable":     ERA5_VARIABLES_LONG,
        "year":         str(year),
        "month":        [month],
        "day":          [f"{d:02d}" for d in range(1, 32)],
        "time":         [f"{h:02d}:00" for h in range(24)],
        "area": [
            city["north"],
            city["west"],
            city["south"],
            city["east"],
        ],
    }


# -----------------------------------------------------------------------------
# ZIP handling
# -----------------------------------------------------------------------------
def handle_zip_if_needed(nc_path):
    """If file is a ZIP, extract and merge to a single .nc in place.
    Renames `valid_time` -> `time` (new CDS API convention)."""
    with open(nc_path, "rb") as f:
        header = f.read(2)
    if header != b"PK":
        return

    log(f"    ZIP detected, extracting and merging...")
    tmpdir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(nc_path, "r") as z:
            z.extractall(tmpdir)

        nc_files = sorted(tmpdir.rglob("*.nc"))
        if not nc_files:
            raise ValueError("No .nc files inside ZIP")

        if len(nc_files) == 1:
            datasets = [xr.open_dataset(str(nc_files[0]))]
        else:
            datasets = [xr.open_dataset(str(p)) for p in nc_files]

        try:
            merged = xr.merge(datasets, compat="override")

            if "valid_time" in merged.dims and "time" not in merged.dims:
                merged = merged.rename({"valid_time": "time"})

            for c in ("number", "expver"):
                if c in merged.coords and merged[c].ndim == 0:
                    merged = merged.drop_vars(c)

            merged.to_netcdf(nc_path, engine="netcdf4")
        finally:
            for ds in datasets:
                ds.close()
            if "merged" in locals():
                merged.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
def expected_hours(year, month):
    """Hours in a given year-month. Leap-year aware."""
    days_per_month = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days_per_month[2] = 29
    return days_per_month[int(month)] * 24


def verify_nc(nc_path, year, month):
    try:
        ds = xr.open_dataset(nc_path)

        actual = set(ds.data_vars)
        missing_long  = set(ERA5_VARIABLES_LONG)  - actual
        missing_short = set(ERA5_VARIABLES_SHORT) - actual
        if missing_long and missing_short:
            ds.close()
            return False, "missing variables"

        n_time = ds.sizes.get("time", 0) or ds.sizes.get("valid_time", 0)
        ds.close()

        exp = expected_hours(year, month)
        if n_time != exp:
            return False, f"time dimension is {n_time}, expected {exp}"
        return True, f"{n_time} hours, all vars present"
    except Exception as e:
        return False, f"open failed: {e}"


# -----------------------------------------------------------------------------
# Download one city-year-month
# -----------------------------------------------------------------------------
def download_one(client, city_name, year, month, output_path=None):
    if output_path is None:
        output_path = RAW_DATA_DIR / city_name / f"{year}_{month}.nc"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label = f"{city_name} {year}-{month}"

    if output_path.exists():
        ok, msg = verify_nc(output_path, year, month)
        if ok:
            log(f"[{label}] SKIPPED (exists, valid: {msg})")
            return "SKIPPED"
        else:
            log(f"[{label}] existing file invalid ({msg}), re-downloading")
            output_path.unlink()

    request = build_request(city_name, year, month)
    log(f"[{label}] requesting from CDS...")

    for attempt in range(MAX_RETRIES):
        try:
            client.retrieve(ERA5_DATASET, request, str(output_path))
            handle_zip_if_needed(output_path)

            ok, msg = verify_nc(output_path, year, month)
            if ok:
                log(f"[{label}] SUCCESS ({msg})")
                return "SUCCESS"
            else:
                log(f"[{label}] VERIFY FAILED: {msg}")
                return "FAILED"
        except Exception as e:
            wait = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
            log(f"[{label}] attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                log(f"    retrying in {wait}s...")
                time.sleep(wait)
            else:
                log(f"[{label}] FAILED after {MAX_RETRIES} attempts")
                return "FAILED"
    return "FAILED"


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------
def run_test(client):
    """Muntinlupa 2023 January only."""
    log("=" * 70)
    log("TEST MODE: Muntinlupa 2023, January only")
    log("=" * 70)
    out = RAW_DATA_DIR / "_test" / "muntinlupa_2023_01.nc"
    status = download_one(client, "muntinlupa", 2023, "01", output_path=out)
    log(f"TEST RESULT: {status}")
    write_decisions_entry("test", {"muntinlupa_2023_01": status})
    return status


def run_full(client, cities=None):
    """Full batch: all cities x years x months."""
    cities = cities or list(CITIES.keys())
    total  = len(cities) * len(YEARS) * len(MONTHS)
    log("=" * 70)
    log(f"FULL BATCH: {len(cities)} cities x {len(YEARS)} years x {len(MONTHS)} months = {total} requests")
    log("=" * 70)

    results = {}
    for city_name in cities:
        for year in YEARS:
            for month in MONTHS:
                key = f"{city_name}_{year}_{month}"
                results[key] = download_one(client, city_name, year, month)

    # Summary
    log("-" * 70)
    log("SUMMARY")
    log("-" * 70)
    counts = {}
    for s in results.values():
        counts[s] = counts.get(s, 0) + 1
    for status in sorted(counts):
        log(f"  {status}: {counts[status]}")
    log(f"  TOTAL: {len(results)}")
    log("Re-run the script to retry any FAILED entries.")

    write_decisions_entry("full", results)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    client = cdsapi.Client()

    if "--test" in sys.argv:
        run_test(client)
        return

    cities = None
    if "--city" in sys.argv:
        idx = sys.argv.index("--city")
        if idx + 1 < len(sys.argv):
            cities = [sys.argv[idx + 1]]

    run_full(client, cities=cities)


if __name__ == "__main__":
    main()