"""
scripts/05_feature_engineering.py
=================================
Phase 6 — Feature Engineering.

Reads:  data/with_geometry/{city}.csv   (FROZEN, 27 cols)
Writes: data/features/{city}.csv        (NEW, 80 cols)

Does not touch any frozen folder. Does not create target-derived
features. Leakage audit runs on train rows only, with a test-set
stability row appended.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make repo root importable when run as `python scripts/05_...py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import CITIES
from src.io import (
    load_with_geometry,
    save_features,
    sha1_head_tail,
    WITH_GEOMETRY_DIR,
    FEATURES_DIR,
)
from src.features import (
    compute_all_features,
    leakage_audit,
    NEW_FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
)

# tz-aware so comparison against the parsed `time` column works.
TRAIN_END  = pd.Timestamp("2024-07-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# Locked Phase 5 column order (exact, per Phase 5 Stage Summary).
EXPECTED_WG_COLUMNS = [
    "time", "local_time", "ssrd_wm2", "fdir_wm2", "tp_mm",
    "t2m_c", "d2m_c", "u10", "v10", "wind_speed", "wind_direction",
    "tcc", "lcc", "mcc", "hcc", "tclw", "tciw", "tcwv", "blh",
    "solar_zenith_angle", "solar_elevation", "cos_zenith",
    "daylight_flag", "extraterrestrial_radiation",
    "i0_horizontal", "clear_sky_ghi", "clear_sky_lag_24",
]

# Locked output column order: 27 with_geometry + 53 new.
EXPECTED_OUT_COLUMNS = EXPECTED_WG_COLUMNS + NEW_FEATURE_COLUMNS
EXPECTED_OUT_NCOLS = len(EXPECTED_OUT_COLUMNS)


def fingerprint_frozen_inputs() -> dict:
    fp = {}
    for city in CITIES:
        p = WITH_GEOMETRY_DIR / f"{city}.csv"
        fp[city] = (p.stat().st_size, sha1_head_tail(p))
    return fp


def main():
    t0 = time.time()
    print("=" * 78)
    print("Phase 6 — Feature Engineering")
    print("=" * 78)

    fp_before = fingerprint_frozen_inputs()
    print("\n[FROZEN INPUT FINGERPRINTS — BEFORE]")
    for c, (sz, h) in fp_before.items():
        print(f"  {c:12s}  {sz:>10,d} bytes  sha1_headtail={h[:12]}")

    per_city = {}

    for city in CITIES:
        print("\n" + "-" * 78)
        print(f"[CITY] {city}")
        print("-" * 78)

        df = load_with_geometry(city)
        n_in_rows, n_in_cols = df.shape
        print(f"  loaded {n_in_rows:,} rows × {n_in_cols} cols")

        if list(df.columns) != EXPECTED_WG_COLUMNS:
            raise RuntimeError(
                f"{city}: with_geometry columns do not match locked "
                f"Phase 5 order.\n"
                f"  got:      {list(df.columns)}\n"
                f"  expected: {EXPECTED_WG_COLUMNS}"
            )
        print("  column order matches locked Phase 5 spec ✓")

        df = compute_all_features(df)

        # Enforce exact output shape and order.
        if df.shape[1] != EXPECTED_OUT_NCOLS:
            raise RuntimeError(
                f"{city}: expected {EXPECTED_OUT_NCOLS} cols after "
                f"feature engineering, got {df.shape[1]}."
            )
        df = df[EXPECTED_OUT_COLUMNS]

        save_features(city, df)
        per_city[city] = df.shape

        # Range / membership checks per brief §7–13.
        assert df["hour_local"].between(0, 23).all()
        assert df["month"].between(1, 12).all()
        assert df["weekday"].between(1, 7).all()
        assert df["season"].isin([0, 1]).all()
        assert df["daylight_flag"].isin([0, 1]).all()
        assert df["rain_binary"].isin([0, 1]).all()
        assert df["is_overcast"].isin([0, 1]).all()
        assert df["convective_flag"].isin([0, 1]).all()
        assert df["rh"].between(0, 100).all()
        assert (df["vapor_pressure_deficit"] >= 0).all()
        assert df["hours_since_rain"].between(0, 72).all()
        assert df["cloud_regime"].isin([0, 1, 2, 3]).all()
        assert df["temp_cat"].isin([0, 1, 2, 3]).all()
        assert df["humidity_cat"].isin([0, 1, 2, 3]).all()
        assert df["wind_cat"].isin([0, 1, 2, 3]).all()
        assert (df["rolling_std_6"]  >= 0).all()
        assert (df["rolling_std_24"] >= 0).all()

        assert not df["time"].duplicated().any()
        assert not df.isna().any().any(), f"{city}: NaN remains after fill"

        for f in FORBIDDEN_COLUMNS:
            assert f not in df.columns, (
                f"{city}: forbidden column present: {f}"
            )

    # ---- Per-category count table ----------------------------------
    category_map = [
        ("A. Time",                 10),
        ("B. Solar geometry",        0),   # skipped — already in wg
        ("C. Direct radiation",      0),   # skipped — already in wg
        ("D. Core weather (rh)",     1),
        ("E. Thermodynamic",         3),
        ("F. Cloud engineered",      4),
        ("G. Cloud dynamics",        8),
        ("H. Rain",                  5),
        ("I. Lags",                  6),
        ("J. Rolling stats",         5),
        ("K. Interactions",          3),
        ("L. Seasonal lag splits",   2),
        ("M. Regime categories",     5),
        ("N. Season",                1),
    ]
    print("\n" + "=" * 78)
    print("FEATURE COUNT BY CATEGORY (new Phase 6 columns)")
    print("=" * 78)
    total_new = 0
    for cat, n in category_map:
        print(f"  {cat:24s}  {n:>3d}")
        total_new += n
    print(f"  {'TOTAL NEW':24s}  {total_new:>3d}")
    print(f"  {'INHERITED (with_geometry)':24s}  {len(EXPECTED_WG_COLUMNS):>3d}")
    print(f"  {'TOTAL OUTPUT':24s}  {EXPECTED_OUT_NCOLS:>3d}")

    # ---- Leakage audit --------------------------------------------
    print("\n" + "=" * 78)
    print("LEAKAGE AUDIT — Pearson |r| vs ssrd_wm2, train rows only")
    print(f"  train: time < {TRAIN_END}   test: time >= {TEST_START}")
    print("=" * 78)

    # Aggregate |r| across cities for the top-N ranking.
    per_city_audit = {}
    for city in CITIES:
        df = pd.read_csv(FEATURES_DIR / f"{city}.csv",
                         parse_dates=["time", "local_time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        a = leakage_audit(df, TRAIN_END, TEST_START, top_n=len(df.columns))
        per_city_audit[city] = a.set_index("feature")

    agg = pd.DataFrame({
        c: per_city_audit[c]["abs_r_train"] for c in CITIES
    }).mean(axis=1).sort_values(ascending=False)

    print(f"\nTop 10 features by mean |r| across 4 cities (train):")
    print(f"  {'rank':>4s}  {'feature':32s}  {'|r| train':>10s}  "
          f"{'|r| test (mean)':>14s}")
    for i, feat in enumerate(agg.head(10).index, 1):
        r_tr = agg[feat]
        r_te = float(np.mean([
            abs(per_city_audit[c].loc[feat, "r_test"]) for c in CITIES
        ]))
        flag = "  <-- FLAG >0.99" if r_tr > 0.99 else ""
        print(f"  {i:>4d}  {feat:32s}  {r_tr:10.4f}  {r_te:14.4f}{flag}")

    if (agg > 0.99).any():
        print("\n*** LEAKAGE FLAG — features exceeding |r|=0.99 ***")
        print(agg[agg > 0.99])
        raise SystemExit("Leakage audit failed.")

    # ---- Frozen-input integrity check -----------------------------
    fp_after = fingerprint_frozen_inputs()
    print("\n" + "=" * 78)
    print("FROZEN-INPUT INTEGRITY CHECK")
    print("=" * 78)
    ok = True
    for c in CITIES:
        same = fp_before[c] == fp_after[c]
        print(f"  {c:12s}  {'IDENTICAL' if same else 'CHANGED!!!'}")
        ok &= same
    if not ok:
        raise SystemExit("Frozen inputs changed during run — aborting.")

    # ---- Summary ---------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for c in CITIES:
        r, k = per_city[c]
        print(f"  {c:12s}  {r:,} rows × {k} cols")
    print(f"  runtime: {time.time() - t0:.1f}s")
    print(f"  frozen inputs UNCHANGED: {ok}")


if __name__ == "__main__":
    main()