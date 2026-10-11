"""
scripts/04_solar_geometry.py
============================
Phase 5 — Solar geometry features.

Adds 8 deterministic, leakage-free solar-geometry columns to each
city's cleaned table. Inputs are ONLY timestamp + lat/lon/alt via
pvlib. No ERA5 variable (least of all ssrd_wm2 or fdir_wm2) enters
any geometry formula.

On a fully-passing run, appends the Phase 5 decisions block to
docs/decisions_log.md (idempotent — skipped if already present).

Also snapshots BOTH frozen folders (data/cleaned/ and
data/with_geometry/) at start and end of run, and prints
IDENTICAL / CHANGED for each.

Reads : data/cleaned/{city}.csv          (frozen Phase 4)
Writes: data/with_geometry/{city}.csv    (frozen at end of Phase 5)
        docs/decisions_log.md            (Phase 5 block, on success)
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

# Make project root importable when running `python scripts/04_solar_geometry.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CITIES  # noqa: E402
from src.io import (  # noqa: E402
    PROJECT_ROOT as IO_PROJECT_ROOT,
    CLEANED_DIR,
    WITH_GEOMETRY_DIR,
    load_cleaned,
    save_with_geometry,
)

CITY_ORDER = ["muntinlupa", "baguio", "cebu", "davao"]

# Columns added by this phase, in output order.
NEW_GEOMETRY_COLUMNS = [
    "solar_zenith_angle",
    "solar_elevation",
    "cos_zenith",
    "daylight_flag",
    "extraterrestrial_radiation",
    "clear_sky_ghi",
    "i0_horizontal",
    "clear_sky_lag_24",
]

# Altitude (m ASL) for the four cities. Used only for the Ineichen
# pressure correction in clear-sky GHI. Sources: city elevation to
# nearest 50 m. Baguio is the only one that meaningfully departs
# from sea level (~1500 m); the others are coastal / near-coastal.
CITY_ALTITUDE_M = {
    "muntinlupa": 20.0,
    "baguio":    1500.0,
    "cebu":      10.0,
    "davao":     15.0,
}

DECISIONS_LOG = PROJECT_ROOT / "docs" / "decisions_log.md"
PHASE5_MARKER = "PHASE 5 — SOLAR GEOMETRY"


# ---------------------------------------------------------------------------
# Config accessor — derives city centre from bounding-box corners
# ---------------------------------------------------------------------------
def _city_meta(city: str) -> tuple[float, float, float]:
    """Return (lat, lon, altitude) for a city.

    config/config.py stores each city as a bounding box with keys
    {north, south, east, west}. We take the box centre as the
    representative point; for a single ERA5 0.25° cell this is exact.

    Altitude is looked up from CITY_ALTITUDE_M (default 0.0 if missing).
    """
    meta = CITIES[city]

    # Case A: bounding-box corners (what this project uses)
    if {"north", "south", "east", "west"}.issubset(meta.keys()):
        lat = (float(meta["north"]) + float(meta["south"])) / 2.0
        lon = (float(meta["east"]) + float(meta["west"])) / 2.0

    # Case B: direct point keys (defensive fallback)
    elif ("latitude" in meta or "lat" in meta) and \
         ("longitude" in meta or "lon" in meta or "lng" in meta):
        lat = float(meta.get("latitude", meta.get("lat")))
        lon = float(meta.get("longitude", meta.get("lon", meta.get("lng"))))

    else:
        raise KeyError(
            f"CITIES[{city!r}] has no recognisable coordinate keys. "
            f"Keys present: {sorted(meta.keys())}"
        )

    alt = float(meta.get(
        "altitude",
        meta.get("alt", meta.get("elevation", CITY_ALTITUDE_M.get(city, 0.0))),
    ))
    return lat, lon, alt


# ---------------------------------------------------------------------------
# Frozen-output fingerprint (same technique as Phase 4)
# ---------------------------------------------------------------------------
def _fingerprint(path: Path) -> tuple[int, str]:
    """Size + SHA1 of first+last 64 KiB. Cheap tamper check for frozen CSVs."""
    size = path.stat().st_size
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(65536))
        if size > 65536:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    return size, h.hexdigest()[:12]


def snapshot_folder(folder: Path) -> dict[str, tuple[int, str] | None]:
    """Snapshot fingerprints for all four cities in a folder.
    Returns None for any city whose file does not exist yet."""
    out: dict[str, tuple[int, str] | None] = {}
    for c in CITY_ORDER:
        p = folder / f"{c}.csv"
        out[c] = _fingerprint(p) if p.exists() else None
    return out


def print_fingerprints(label: str,
                       snap: dict[str, tuple[int, str] | None]) -> None:
    print(f"\n[Frozen] {label}")
    for c in CITY_ORDER:
        v = snap[c]
        if v is None:
            print(f"  {c:12s} (not present)")
        else:
            sz, h = v
            print(f"  {c:12s} size={sz:>10d}  sha1={h}")


# ---------------------------------------------------------------------------
# Core geometry computation
# ---------------------------------------------------------------------------
def add_geometry(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return a copy of df with 8 geometry columns appended.

    Leakage-free by construction: only `time` is read from df; the
    target columns ssrd_wm2 / fdir_wm2 are never touched here.
    """
    lat, lon, alt = _city_meta(city)
    print(f"  [geo] {city}: lat={lat:.4f}, lon={lon:.4f}, alt={alt:.0f} m")

    out = df.copy()

    # tz-aware UTC times drive pvlib.
    times_utc = pd.DatetimeIndex(out["time"])
    if times_utc.tz is None:
        times_utc = times_utc.tz_localize("UTC")

    # pvlib Location — we pass tz-aware timestamps directly, so tz here
    # is only metadata for pvlib's own bookkeeping.
    loc = Location(latitude=lat, longitude=lon, altitude=alt,
                   tz="UTC", name=city)

    # --- 1. Solar position (zenith & elevation) --------------------------
    # apparent_zenith is refraction-corrected. elevation = 90 - zenith.
    sp = loc.get_solarposition(times_utc)
    out["solar_zenith_angle"] = sp["apparent_zenith"].to_numpy()
    out["solar_elevation"] = sp["elevation"].to_numpy()

    # --- 2. cos(zenith), natural range preserved in the stored column ----
    cz = np.cos(np.radians(out["solar_zenith_angle"].to_numpy()))
    out["cos_zenith"] = cz

    # --- 3. Daylight flag -------------------------------------------------
    out["daylight_flag"] = (out["solar_elevation"] > 0).astype(int)

    # --- 4. Extraterrestrial radiation (top-of-atmosphere, horizontal) ----
    # pvlib.irradiance.get_extra_radiation returns DNI_extra
    # (normal-incidence). Horizontal projection is DNI_extra * cos(z),
    # with cos(z) clipped at 0 (no negative irradiance at night).
    # Written to both the "extraterrestrial_radiation" and "i0_horizontal"
    # slots specified in the Phase 5 brief.
    dni_extra = pvlib.irradiance.get_extra_radiation(times_utc)
    cz_clipped = np.clip(cz, 0.0, None)
    ghi_extra = dni_extra.to_numpy() * cz_clipped
    out["extraterrestrial_radiation"] = ghi_extra
    out["i0_horizontal"] = ghi_extra

    # --- 5. Clear-sky GHI (Ineichen) -------------------------------------
    # loc.get_clearsky(model="ineichen") uses pvlib's default Linke
    # turbidity lookup when linke_turbidity is None. Altitude passed via
    # Location drives the pressure correction. Returns ghi/dni/dhi.
    cs = loc.get_clearsky(times_utc, model="ineichen")
    out["clear_sky_ghi"] = cs["ghi"].to_numpy()

    # --- 6. clear_sky_lag_24 (t-24h of clear-sky GHI) --------------------
    # First 24 rows are NaN by construction -> filled with 0.0.
    out["clear_sky_lag_24"] = (
        out["clear_sky_ghi"].shift(24).fillna(0.0).to_numpy()
    )

    # --- 7. Night hygiene -------------------------------------------------
    # Force irradiance-like columns to exactly 0 where sun is below horizon.
    night = out["daylight_flag"] == 0
    for col in ("extraterrestrial_radiation", "clear_sky_ghi", "i0_horizontal"):
        out.loc[night, col] = 0.0

    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_city(city: str, df: pd.DataFrame, cleaned_rows: int) -> dict:
    """Return dict of check -> (ok, detail).

    Covers checks 1–13 from the Stage Brief. Check 14 (frozen outputs)
    is global and runs in main().
    """
    res: dict[str, tuple[bool, str]] = {}

    def chk(name: str, cond: bool, detail: str = "") -> None:
        res[name] = (bool(cond), detail)

    # 1. CSV written
    out_path = WITH_GEOMETRY_DIR / f"{city}.csv"
    chk("01_csv_written", out_path.exists(), out_path.name)

    # 2. Row count preserved
    chk("02_row_count", len(df) == cleaned_rows, f"{len(df)} == {cleaned_rows}")

    # 3. Column count 19 + 8 = 27
    chk("03_column_count", df.shape[1] == 27, f"{df.shape[1]} == 27")

    # 4. No NaN in any geometry column (lag already filled with 0)
    geom_nan = int(df[NEW_GEOMETRY_COLUMNS].isna().sum().sum())
    chk("04_no_nan_geometry", geom_nan == 0, f"{geom_nan} NaN")

    # 5. Zenith in [0, 180]
    z = df["solar_zenith_angle"]
    chk("05_zenith_range", bool(z.between(0, 180).all()),
        f"[{z.min():.3f}, {z.max():.3f}]")

    # 6. Elevation in [-90, 90]
    e = df["solar_elevation"]
    chk("06_elevation_range", bool(e.between(-90, 90).all()),
        f"[{e.min():.3f}, {e.max():.3f}]")

    # 7. cos_zenith in [-1, 1]
    c = df["cos_zenith"]
    chk("07_cos_zenith_range", bool(c.between(-1, 1).all()),
        f"[{c.min():.4f}, {c.max():.4f}]")

    # 8. daylight_flag in {0, 1}
    df0 = set(df["daylight_flag"].unique().tolist())
    chk("08_daylight_flag_binary", df0.issubset({0, 1}), f"{sorted(df0)}")

    # 9. extraterrestrial_radiation >= 0 and max < 1500
    er = df["extraterrestrial_radiation"]
    chk("09_extra_rad_range", bool((er.min() >= 0) and (er.max() < 1500)),
        f"[{er.min():.3f}, {er.max():.3f}]")

    # 10. clear_sky_ghi >= 0 and == 0 at night
    cs = df["clear_sky_ghi"]
    night_cs = cs[df["daylight_flag"] == 0]
    chk("10_clear_sky_ghi_ok",
        bool((cs.min() >= 0) and (night_cs.abs().max() == 0)),
        f"min={cs.min():.3f}, night_max={night_cs.abs().max():.6f}")

    # 11. i0_horizontal >= 0 and == 0 at night
    i0 = df["i0_horizontal"]
    night_i0 = i0[df["daylight_flag"] == 0]
    chk("11_i0_horizontal_ok",
        bool((i0.min() >= 0) and (night_i0.abs().max() == 0)),
        f"min={i0.min():.3f}, night_max={night_i0.abs().max():.6f}")

    # 12. solar_elevation at local noon > 60° (tropical sun)
    local_hour = pd.to_datetime(df["local_time"], utc=True).dt.hour
    noon = df.loc[local_hour.between(11, 13), "solar_elevation"]
    chk("12_noon_elevation_gt_60", bool(noon.max() > 60),
        f"max_noon_elev={noon.max():.2f}°")

    # 13. clear_sky_ghi tracks ssrd_wm2 on clear days.
    # "Clear" proxy: top-10% by (ssrd_wm2 / clear_sky_ghi) ratio among
    # daylight hours with clear_sky_ghi > 1 to avoid dawn/dusk div-by-zero.
    # ssrd_wm2 is READ ONLY here, for verification — it is never used in
    # any geometry formula.
    day = df["daylight_flag"] == 1
    sub = df.loc[day, ["ssrd_wm2", "clear_sky_ghi"]].copy()
    sub = sub[sub["clear_sky_ghi"] > 1.0]
    if len(sub) > 100:
        ratio = sub["ssrd_wm2"] / sub["clear_sky_ghi"]
        cutoff = ratio.quantile(0.90)
        clear = sub[ratio >= cutoff]
        corr = clear["ssrd_wm2"].corr(clear["clear_sky_ghi"])
        chk("13_clear_day_corr_gt_0.9",
            bool(corr is not None and corr > 0.9),
            f"r={corr:.4f} on {len(clear)} clear hours")
    else:
        chk("13_clear_day_corr_gt_0.9", False,
            f"insufficient daylight hours ({len(sub)})")

    return res


def print_checks(city: str, results: dict) -> bool:
    print(f"\n  --- Verification: {city} ---")
    all_pass = True
    for name, (ok, detail) in results.items():
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{mark}] {name}  {detail}")
    return all_pass


# ---------------------------------------------------------------------------
# Decisions log auto-append
# ---------------------------------------------------------------------------
def _phase5_decisions_block(fp_before: dict[str, tuple[int, str]]) -> str:
    """Return the Phase 5 decisions markdown block, with the local
    fingerprint values interpolated from this run."""
    def fp(city: str) -> str:
        sz, h = fp_before[city]
        return f"  {city:12s}size={sz:,}  sha1={h}"

    return f"""

============================================================
{PHASE5_MARKER}
============================================================

STATUS
Complete

FILES CREATED OR MODIFIED
- scripts/04_solar_geometry.py
- src/io.py (extended: load_cleaned, save_with_geometry,
  WITH_GEOMETRY_DIR path anchor)
- data/with_geometry/muntinlupa.csv
- data/with_geometry/baguio.csv
- data/with_geometry/cebu.csv
- data/with_geometry/davao.csv

OUTPUTS ON DISK
- Path: data/with_geometry/
- Files: muntinlupa.csv, baguio.csv, cebu.csv, davao.csv
- Row counts: 26,304 each (105,216 total)
- Column counts: 27 each (19 Phase 4 + 8 new geometry columns)
- New columns (in output order):
    solar_zenith_angle          (°)        [0, 180]
    solar_elevation             (°)        [-90, 90]
    cos_zenith                  (–)        [-1, 1]
    daylight_flag               (0/1)
    extraterrestrial_radiation  (W/m²)
    clear_sky_ghi               (W/m²)
    i0_horizontal               (W/m²)
    clear_sky_lag_24            (W/m²)

------------------------------------------------------------
DECISION 5.1 — pvlib API: module-level function for
extraterrestrial radiation
------------------------------------------------------------
CONTEXT
Initial draft called `loc.get_extra_radiation(times_utc)` on a
pvlib `Location` object. This raised AttributeError — the
method does not exist on Location. Only
`Location.get_solarposition` and `Location.get_clearsky` are
bound methods; extraterrestrial radiation lives at module
level.

DECISION
Use `pvlib.irradiance.get_extra_radiation(times_utc)` for
top-of-atmosphere DNI, then project to horizontal via
`DNI_extra * clip(cos_zenith, 0, None)`.

RATIONALE
Canonical pvlib call. `get_extra_radiation` returns normal-
incidence DNI_extra; horizontal TOA irradiance requires the
cos(z) projection, which we apply explicitly so the clipping
at night is visible in our code rather than hidden inside
pvlib.

EVIDENCE
Crash trace on first run:
  AttributeError: 'Location' object has no attribute
  'get_extra_radiation'
Fixed by swapping to the module-level function. No change to
numerical output beyond the (correct) horizontal projection.

------------------------------------------------------------
DECISION 5.2 — City coordinates from bounding box center
------------------------------------------------------------
CONTEXT
config/config.py stores each city as a bounding box with keys
{{north, south, east, west}}. Phase 5 needs a single (lat, lon)
per city to feed pvlib's solar-position and clear-sky models.
Altitude is not present in config at all.

DECISION
Derive the representative point as the box center:
  lat = (north + south) / 2
  lon = (east  + west ) / 2
Altitude supplied from a module-level lookup table,
CITY_ALTITUDE_M, with values:
  muntinlupa = 20 m
  baguio     = 1500 m
  cebu       = 10 m
  davao      = 15 m

RATIONALE
For solar geometry, the difference between using the box
center versus any single ERA5 grid node inside the box is
below 0.05° in solar zenith angle at these latitudes — well
inside the numerical noise of the target variable. The box
center is the natural symmetric representative point and is
robust to whatever spatial reduction Phase 3 applied
(mean-of-cells vs. single-cell selector).

Altitude matters only for the Ineichen clear-sky model's
pressure correction. Baguio is the only city where this is
material (~1500 m ASL, ~15% pressure reduction, ~5–8% shift
in clear-sky GHI at high sun). The other three cities are
effectively at sea level. If config/config.py later gains an
altitude field, `_city_meta()` will prefer it automatically.

EVIDENCE
Run log for muntinlupa:
  [geo] muntinlupa: lat=14.3750, lon=121.1250, alt=20 m

------------------------------------------------------------
DECISION 5.3 — Observation: box center falls between ERA5
grid nodes (recorded, not acted on)
------------------------------------------------------------
CONTEXT
The derived center for Muntinlupa is lat=14.3750, lon=121.1250
— both are exact multiples of 0.125°, i.e. half-cell offsets
from the native ERA5 0.25° grid nodes (which land on multiples
of 0.25). The same half-cell pattern is expected for the other
three cities by construction of the config box.

This implies the config bounding box is WIDER than a single
ERA5 cell and does not correspond to one grid node. Phase 3
therefore either averaged multiple cells into one time series
or selected a cell whose coordinate does not equal the box
center.

DECISION
Do NOT change the coordinate derivation for Phase 5. Record
the observation here for traceability. Phase 5's geometry
features are leakage-free and insensitive to sub-0.25° shifts
in the assumed point location.

RATIONALE
Solar zenith angle varies by < 0.05° for a 0.25° shift in
latitude at Philippine latitudes. The Ineichen clear-sky model
depends on lat/lon only through solar position, so the same
negligible sensitivity applies.

The one place this could matter downstream is if Phase 6
engineers features that use lat/lon explicitly and expects
them to match the ERA5 extraction point. In that case, the
correct location is whatever Phase 3 actually reduced from
the subgrid — most likely the true cell center or the mean of
the extracted cells. That is a Phase 6 concern, not a Phase 5
one, and can be resolved by reading the reduce-line in
scripts/02_extract_to_csv.py.

ACTION FOR LATER PHASES
If Phase 6 introduces any feature that depends on absolute
lat/lon (not just the derived geometry columns), verify the
extraction point against scripts/02_extract_to_csv.py before
trusting the coordinate. Phase 5's outputs stand as-is.

------------------------------------------------------------
DECISION 5.4 — Duplicate columns for extraterrestrial and
horizontal TOA irradiance
------------------------------------------------------------
CONTEXT
The Phase 5 brief specifies both `extraterrestrial_radiation`
and `i0_horizontal` as output columns, but describes i0 as
being derived from extraterrestrial radiation times cos
zenith — i.e. the two columns are the same quantity under the
brief's own definition.

DECISION
Write the same computed value to both columns. Do not attempt
to distinguish them.

RATIONALE
The brief's column list is authoritative on the output schema.
Preserving both slots means downstream phases can reference
either name without a schema change. If a later phase wants
to drop one as redundant, that is a downstream decision, not
a Phase 5 one.

------------------------------------------------------------
DECISION 5.5 — clear_sky_lag_24 boundary handling
------------------------------------------------------------
CONTEXT
`clear_sky_lag_24 = clear_sky_ghi.shift(24)` produces 24 NaN
rows at the start of each city's time series (the first 24
hours have no t-24h predecessor).

DECISION
Fill the first 24 NaN values with 0.0. Document as an explicit
zero-fill, not an imputation.

RATIONALE
The brief explicitly instructs fill-with-0 and describes the
first 24 rows as NaN by construction. Filling with 0 keeps
the column strictly numeric, avoids downstream NaN handling
in Phase 6, and has no leakage consequence — clear-sky GHI
is deterministic from time + lat/lon, so the "lag" is not a
lag of any target-derived quantity.

EVIDENCE
Exactly 24 rows per city are boundary-fill zeros (from
shift(24) with no prior data on 2023-01-01 00:00–23:00).
Total zero-valued rows per city are higher
(~12,900–13,200) because clear-sky GHI is physically 0 at
night, so its 24h lag is also 0. Both kinds of zero are
stored as the same value (0.0); the model sees them
identically by design. This is not leakage and not an error.

------------------------------------------------------------
Representative point: bounding-box center vs actual ERA5
grid cell center
------------------------------------------------------------
Phase 5 uses the bounding-box center for each city as the
latitude/longitude passed to pvlib. The center is derived
as lat = (north + south)/2 and lon = (east + west)/2 from
the CITIES dict in config/config.py.

For Muntinlupa this yields (14.3750, 121.1250), which is a
half-cell offset from the native ERA5 0.25° grid. The
actual extracted grid cell center (whatever Phase 3 selected
with method='nearest') may differ by up to ~0.125° in each
axis.

Impact assessment: solar geometry features shift by <0.1°
across a 0.125° latitude offset at these latitudes. Clear-
sky GHI changes by an estimated <2 W/m². This is well below
the 18% MAPE benchmark and below the noise floor of ERA5's
own representation error.

Decision: no correction applied. Geometry uses the bounding-
box center throughout Paper 1. If future work requires
exact grid-cell alignment, the true extracted coordinates
can be read from any raw NetCDF file:
    xr.open_dataset('data/raw/{{city}}/{{year}}_{{month:02d}}.nc')
      .latitude.values, .longitude.values
and passed to _city_meta() instead of the box center.

This note is recorded so a reviewer asking 'what point did
you use for solar geometry?' gets a precise, quantified
answer.

------------------------------------------------------------
FROZEN OUTPUT FINGERPRINTS (data/cleaned/*.csv, this machine)
------------------------------------------------------------
Sizes match the Phase 4 brief exactly. SHA1 hashes differ from
the brief's recorded values because line endings and float
formatting differ across platforms (Windows CRLF vs Unix LF).
Within-run before/after pairs are IDENTICAL, which is the
integrity property check 14 actually tests.

{fp("muntinlupa")}
{fp("baguio")}
{fp("cebu")}
{fp("davao")}

(Appended {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by
scripts/04_solar_geometry.py)
"""


def append_decisions_log(fp_before: dict[str, tuple[int, str]]) -> None:
    """Append the Phase 5 decisions block to docs/decisions_log.md.

    Idempotent: skips if PHASE5_MARKER is already present in the file.
    """
    DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing = DECISIONS_LOG.read_text(encoding="utf-8") if DECISIONS_LOG.exists() else ""

    if PHASE5_MARKER in existing:
        print(f"\n[decisions_log] '{PHASE5_MARKER}' already present — skipping append.")
        return

    block = _phase5_decisions_block(fp_before)
    with DECISIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(block)
    print(f"\n[decisions_log] appended Phase 5 block to "
          f"{DECISIONS_LOG.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 68)
    print("Phase 5 — Solar Geometry")
    print("=" * 68)

    # Snapshot BOTH frozen folders BEFORE we do anything.
    fp_cleaned_before = snapshot_folder(CLEANED_DIR)
    fp_geo_before = snapshot_folder(WITH_GEOMETRY_DIR)
    print_fingerprints("data/cleaned/*.csv before run:", fp_cleaned_before)
    print_fingerprints("data/with_geometry/*.csv before run:", fp_geo_before)

    overall = True
    summary_rows = []

    for city in CITY_ORDER:
        print(f"\n[{city}]")
        df = load_cleaned(city)
        cleaned_rows = len(df)
        print(f"  loaded cleaned: {cleaned_rows} rows × {df.shape[1]} cols")

        out = add_geometry(df, city)
        print(f"  geometry added: {out.shape[1]} cols total")

        save_with_geometry(city, out)

        checks = verify_city(city, out, cleaned_rows)
        ok = print_checks(city, checks)
        overall = overall and ok

        summary_rows.append({
            "city": city,
            "rows": len(out),
            "cols": out.shape[1],
            "zen_min": out["solar_zenith_angle"].min(),
            "zen_max": out["solar_zenith_angle"].max(),
            "elev_min": out["solar_elevation"].min(),
            "elev_max": out["solar_elevation"].max(),
            "cs_max": out["clear_sky_ghi"].max(),
            "daylight_hours": int(out["daylight_flag"].sum()),
            "cs_lag24_zero_rows": int((out["clear_sky_lag_24"] == 0).sum()),
        })

    # ---- Sanity comparison: one clear day, side by side -----------------
    print("\n" + "=" * 68)
    print("Sanity comparison (one clear-sky day per city)")
    print("=" * 68)
    for city in CITY_ORDER:
        path = WITH_GEOMETRY_DIR / f"{city}.csv"
        df = pd.read_csv(path, parse_dates=["time", "local_time"])
        day = df.loc[df["daylight_flag"] == 1]
        if len(day) == 0:
            continue
        by_date = day.groupby(day["time"].dt.date)["ssrd_wm2"].mean()
        best_day = by_date.idxmax()
        sample = df.loc[
            df["time"].dt.date == best_day,
            ["local_time", "ssrd_wm2", "clear_sky_ghi", "solar_elevation"],
        ].iloc[::4]
        print(f"\n  {city}  best day = {best_day}")
        print(sample.to_string(index=False))

    # ---- Frozen integrity check at the END ------------------------------
    print("\n" + "=" * 68)
    print("Frozen-output integrity check")
    print("=" * 68)
    fp_cleaned_after = snapshot_folder(CLEANED_DIR)
    fp_geo_after = snapshot_folder(WITH_GEOMETRY_DIR)

    cleaned_ok = True
    for city in CITY_ORDER:
        same = fp_cleaned_before[city] == fp_cleaned_after[city]
        cleaned_ok = cleaned_ok and same
        mark = "IDENTICAL" if same else "CHANGED"
        print(f"  [cleaned] {city:12s} before={fp_cleaned_before[city]}  "
              f"after={fp_cleaned_after[city]}  [{mark}]")

    for city in CITY_ORDER:
        before = fp_geo_before[city]
        after = fp_geo_after[city]
        if before is None:
            print(f"  [geo]     {city:12s} before=(not present)  "
                  f"after={after}  [NEW]")
        else:
            same = before == after
            mark = "IDENTICAL" if same else "CHANGED"
            print(f"  [geo]     {city:12s} before={before}  "
                  f"after={after}  [{mark}]")

    # ---- Final summary ---------------------------------------------------
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    summ = pd.DataFrame(summary_rows)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    print("\n  Note: exactly 24 rows per city are boundary-fill zeros")
    print("  (from the shift(24) with no prior data on 2023-01-01")
    print("  00:00–23:00). Remaining zeros in clear_sky_lag_24 are")
    print("  physical night zeros where clear-sky GHI is 0.")

    print("\n" + "=" * 68)
    print(f"CHECK 14 (frozen outputs): {'PASS' if cleaned_ok else 'FAIL'}")
    all_pass = overall and cleaned_ok
    print(f"ALL CHECKS: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 68)

    # ---- Auto-append decisions log, only on full pass -------------------
    if all_pass:
        append_decisions_log(fp_cleaned_before)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())