"""
Phase 3 — Extract & Merge
=========================
Flatten 144 monthly ERA5 NetCDF files (4 cities × 36 months) into
4 per-city CSVs, one row per hour (UTC), 15 raw variables, no
transforms. Select the single grid cell nearest each city center.

Inputs:
    data/raw/{city}/{year}_{month:02d}.nc   (144 files)
    config/config.py
    config/paper1_config.py

Outputs:
    data/extracted/{city}.csv               (4 files)

Naming convention (locked from this phase forward):
    - Technical layer  -> SHORT codenames  (ssrd, fdir, t2m, ...)
      Used in: CSV columns, DataFrame columns, feature names,
      model inputs, split files, metric keys, error `key` fields.
    - Human-readable layer -> LONG CDS names (surface_solar_radiation_downwards,
      2m_temperature, ...)
      Used in: console verification printouts, error messages,
      docstrings, and (later) figure/table labels.
    - Single source of truth for the short->long map:
      config.config.ERA5_VARIABLES

Rules honoured:
    - No unit conversion (Phase 4)
    - No local_time / timezone shift (Phase 4)
    - No solar geometry (Phase 5)
    - No feature engineering (Phase 6)
    - No train/val/test split (Phase 7)
    - Raw files untouched
    - Variable names unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

# --- Make project root importable when run as `python scripts/02_extract_to_csv.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    CITIES,
    ERA5_VARIABLES,           # short -> long mapping (single source of truth)
    ERA5_VARIABLES_SHORT,     # list of short codenames
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

# ---------------------------------------------------------------------------
# City centres (lat, lon) — must match the raw bounding boxes from Phase 2.
# Using .sel(..., method="nearest") on each monthly file picks the grid cell
# whose centre is closest to these coordinates. Phase 2 kept a 2×2 box per
# city because boxes land on ERA5 grid lines, so nearest-cell selection here
# is the deterministic tie-breaker.
# ---------------------------------------------------------------------------
CITY_CENTERS: dict[str, tuple[float, float]] = {
    "muntinlupa": (14.25, 121.00),
    "baguio":     (16.25, 120.50),
    "cebu":       (10.25, 123.75),
    "davao":      (7.00, 125.50),
}

# ---------------------------------------------------------------------------
# Phase 3 canonical column order for data/extracted/*.csv
# ---------------------------------------------------------------------------
# Technical layer uses SHORT codenames (ssrd, fdir, ...) — these are what
# every downstream script, model, and metric key will see.
# Human-readable layer uses LONG names (surface_solar_radiation_downwards,
# ...) — these appear only in verification printouts and sanity displays.
# The short->long mapping itself lives in config.config.ERA5_VARIABLES.
PHASE3_VARIABLE_ORDER = [
    "ssrd", "fdir", "tp", "t2m", "d2m", "u10", "v10",
    "tcc", "lcc", "mcc", "hcc", "tclw", "tciw", "tcwv", "blh",
]

# Sanity: config and this script must agree on which variables exist,
# even if their order differs.
assert set(PHASE3_VARIABLE_ORDER) == set(ERA5_VARIABLES_SHORT), (
    "Phase 3 variable list does not match config.ERA5_VARIABLES_SHORT. "
    f"Phase 3: {sorted(PHASE3_VARIABLE_ORDER)}  "
    f"Config:  {sorted(ERA5_VARIABLES_SHORT)}"
)
assert set(PHASE3_VARIABLE_ORDER) == set(ERA5_VARIABLES), (
    "Phase 3 variable list does not match config.ERA5_VARIABLES (short->long). "
    f"Missing: {sorted(set(PHASE3_VARIABLE_ORDER) - set(ERA5_VARIABLES))}"
)

# Convenience aliases used throughout this script.
VARIABLES = PHASE3_VARIABLE_ORDER
EXPECTED_COLUMNS = ["time"] + VARIABLES


def human_name(short: str) -> str:
    """Return the long CDS name for a short codename (falls back to short)."""
    return ERA5_VARIABLES.get(short, short)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _monthly_files(city: str) -> list[Path]:
    """Return the 36 monthly NetCDF files for a city, sorted by (year, month)."""
    city_dir = RAW_DIR / city
    if not city_dir.is_dir():
        raise FileNotFoundError(f"Raw directory missing for city: {city_dir}")

    files = sorted(city_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {city_dir}")

    # Sort explicitly by the {year}_{month}.nc stem so ordering is stable
    # regardless of filesystem enumeration order.
    def _key(p: Path) -> tuple[int, int]:
        year, month = p.stem.split("_")
        return int(year), int(month)

    return sorted(files, key=_key)


def _select_nearest_cell(ds: xr.Dataset, city: str) -> xr.Dataset:
    """
    Select the grid cell nearest the city centre and squeeze lat/lon
    down to scalar coordinates so the result is a pure time series.
    """
    lat_c, lon_c = CITY_CENTERS[city]

    # ERA5 uses latitude descending and longitude ascending on 0.25° grid.
    # method="nearest" guarantees a single cell is chosen even when the
    # city centre sits exactly on a grid line (Phase 2 caveat #2).
    ds = ds.sel(latitude=lat_c, longitude=lon_c, method="nearest")

    # Drop the now-scalar lat/lon coordinates to avoid them appearing as
    # extra columns when we convert to DataFrame.
    ds = ds.squeeze(drop=True)
    ds = ds.drop_vars(["latitude", "longitude"], errors="ignore")
    return ds


def _check_variables(ds: xr.Dataset, city: str, path: Path) -> None:
    """Fail loudly if any expected variable is missing from a monthly file."""
    present = set(ds.data_vars)
    expected = set(VARIABLES)
    missing = expected - present
    if missing:
        # Technical context: short codenames (what the pipeline uses).
        # Human context: long CDS names (what a person reading the error
        # would type into the CDS web UI to fix the download).
        missing_long = [human_name(v) for v in sorted(missing)]
        present_long = [human_name(v) for v in sorted(present)]
        raise KeyError(
            f"[{city}] {path.name} is missing expected variables.\n"
            f"  Missing (short): {sorted(missing)}\n"
            f"  Missing (long):  {missing_long}\n"
            f"  Present (short): {sorted(present)}\n"
            f"  Present (long):  {present_long}"
        )


def _to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert the squeezed Dataset to a pandas DataFrame indexed by UTC time,
    with the 15 variables as columns in canonical order.
    """
    # to_dataframe() returns a long frame; since the dataset is now a pure
    # time series (one row per timestamp), we can just reset_index().
    df = ds[VARIABLES].to_dataframe().reset_index()
    return df


def _coerce_time_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the 'time' column is tz-naive UTC pandas datetime."""
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    return df


# ---------------------------------------------------------------------------
# Per-city pipeline
# ---------------------------------------------------------------------------
def extract_city(city: str) -> pd.DataFrame:
    """Read, select, merge, and return the full 3-year hourly DataFrame."""
    files = _monthly_files(city)
    print(f"\n[{city}] Found {len(files)} monthly files.")

    monthly_frames: list[pd.DataFrame] = []
    rows_accumulated = 0

    for i, path in enumerate(files, start=1):
        with xr.open_dataset(path) as ds:
            # Phase 2 renamed valid_time -> time. Guard against regression.
            if "time" not in ds.dims and "time" not in ds.coords:
                raise KeyError(
                    f"[{city}] {path.name} has no 'time' dimension/coordinate. "
                    f"Dims: {list(ds.dims)}, coords: {list(ds.coords)}"
                )

            _check_variables(ds, city, path)

            ds_cell = _select_nearest_cell(ds, city)

            # Load into memory before the file handle closes.
            frame = _to_dataframe(ds_cell.load())

        monthly_frames.append(frame)
        rows_accumulated += len(frame)

        # Progress every 6 files (≈ half a year) and on the last file.
        if i % 6 == 0 or i == len(files):
            print(
                f"[{city}]   read {i:2d}/{len(files)} files  |  "
                f"rows so far: {rows_accumulated}"
            )

    df = pd.concat(monthly_frames, ignore_index=True)
    df = _coerce_time_to_utc(df)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    df = df[EXPECTED_COLUMNS]

    print(f"[{city}] Merged rows: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_city(city: str, df: pd.DataFrame, csv_path: Path) -> dict:
    """Run the 6 verification checks and return a summary dict."""
    checks: dict[str, object] = {}

    # 1. File exists
    checks["file_exists"] = csv_path.is_file()

    # 2. Row count ~26,304 (±2 for leap-hour edges)
    #    2023 (non-leap) = 8,760 h; 2024 (leap) = 8,784 h; 2025 = 8,760 h
    #    Total = 26,304 h.
    n = len(df)
    checks["rows"] = n
    checks["rows_ok"] = abs(n - 26304) <= 2

    # 3. Exactly 16 columns
    checks["columns"] = list(df.columns)
    checks["columns_ok"] = list(df.columns) == EXPECTED_COLUMNS

    # 4. No NaN in any column
    nan_counts = df.isna().sum()
    checks["nan_total"] = int(nan_counts.sum())
    checks["no_nan_ok"] = bool(nan_counts.sum() == 0)

    # 5. No duplicate timestamps
    dup = int(df["time"].duplicated().sum())
    checks["duplicate_timestamps"] = dup
    checks["no_duplicates_ok"] = dup == 0

    # 6. Timestamps monotonically increasing and full UTC range
    checks["monotonic_ok"] = bool(df["time"].is_monotonic_increasing)
    expected_start = pd.Timestamp("2023-01-01 00:00:00")
    expected_end = pd.Timestamp("2025-12-31 23:00:00")
    checks["time_start"] = df["time"].iloc[0]
    checks["time_end"] = df["time"].iloc[-1]
    checks["time_range_ok"] = (
        df["time"].iloc[0] == expected_start
        and df["time"].iloc[-1] == expected_end
    )

    return checks


def print_verification(city: str, checks: dict, df: pd.DataFrame) -> None:
    print(f"\n  Verification — {city}")
    print(f"    file exists ............... {checks['file_exists']}")
    print(
        f"    rows ...................... {checks['rows']} "
        f"(target 26304 ±2) -> {checks['rows_ok']}"
    )
    print(
        f"    columns ................... {len(checks['columns'])} "
        f"-> {checks['columns_ok']}"
    )
    print(
        f"    NaNs total ................ {checks['nan_total']} "
        f"-> {checks['no_nan_ok']}"
    )
    print(
        f"    duplicate timestamps ...... {checks['duplicate_timestamps']} "
        f"-> {checks['no_duplicates_ok']}"
    )
    print(
        f"    monotonic increasing ...... {checks['monotonic_ok']}"
    )
    print(
        f"    time range ................ {checks['time_start']} -> "
        f"{checks['time_end']} -> {checks['time_range_ok']}"
    )

    # Human-readable variable inventory (short -> long) so a person can
    # eyeball the schema without leaving the terminal.
    print("    columns present (short -> long):")
    for short in VARIABLES:
        print(f"      {short:5s}  ->  {human_name(short)}")

    # Sanity rows: first and last
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print("\n    Sanity row — first:")
        print(df.head(1).to_string(index=False))
        print("\n    Sanity row — last:")
        print(df.tail(1).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {EXTRACTED_DIR}")

    all_checks: dict[str, dict] = {}

    for city in CITY_CENTERS:
        if city not in CITIES:
            raise KeyError(
                f"City '{city}' not in config.CITIES. "
                f"Config has: {list(CITIES)}"
            )

        df = extract_city(city)

        csv_path = EXTRACTED_DIR / f"{city}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{city}] Wrote {csv_path}  ({csv_path.stat().st_size / 1e3:.1f} KB)")

        checks = verify_city(city, df, csv_path)
        all_checks[city] = checks
        print_verification(city, checks, df)

    # ---- Overall pass/fail gate ------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL VERIFICATION  (technical view: short codenames)")
    print("=" * 60)
    all_ok = True
    for city, c in all_checks.items():
        city_ok = all([
            c["file_exists"],
            c["rows_ok"],
            c["columns_ok"],
            c["no_nan_ok"],
            c["no_duplicates_ok"],
            c["monotonic_ok"],
            c["time_range_ok"],
        ])
        all_ok &= city_ok
        print(f"  {city:12s} -> {'PASS' if city_ok else 'FAIL'}")

    if not all_ok:
        print("\nVerification FAILED. Do not proceed to Phase 4.")
        return 1

    print("\nAll 6 checks passed for all 4 cities. Ready for Phase 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())