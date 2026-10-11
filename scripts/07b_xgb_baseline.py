# scripts/07b_xgb_baseline.py
"""
Phase 10: Model 2 — XGBoost Baseline (6 variables, untuned)

Trains an untuned XGBRegressor on 13 features (the pilot study's
Model 2 feature set) for each of 4 Philippine cities. Reads from
the FROZEN data/splits/paper1/ folder and writes ONLY to
results/paper1/. Establishes the baseline that Phase 11's tuned
80-feature XGBoost must beat.

Rules honoured:
  - Only the 13 specified features are used (asserted).
  - No current-hour ssrd_wm2 / fdir_wm2 (excluded by list).
  - sha1_head_tail imported from src.io (not reimplemented).
  - all_metrics, mape_daytime imported from src.metrics.
  - No sklearn train_test_split. Chronological split already on disk.
  - Predictions clipped to >= 0 before metrics.
  - Fingerprints snapshotted at start AND end; fail loudly on change.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Make repo root importable regardless of CWD --------------------------
# scripts/07b_xgb_baseline.py -> parents[1] is the repo root.
# This lets `python scripts\07b_xgb_baseline.py` find the `src` package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

# --- Single source of truth for fingerprints, dirs, metrics ---------------
from src.io import sha1_head_tail, SPLITS_DIR, RESULTS_DIR
from src.metrics import all_metrics, mape_daytime

try:
    import xgboost as xgb
except ImportError:
    sys.exit(
        "ERROR: xgboost is not installed.\n"
        "Install with:  pip install \"xgboost>=1.6,<3\"\n"
        "(Script will not install it for you.)"
    )

# --- Constants ------------------------------------------------------------
CITIES = ["muntinlupa", "baguio", "cebu", "davao"]

FEATURES = [
    "u10", "v10", "t2m_c", "d2m_c", "tp_mm",
    "hour_local", "month", "hour_sin", "hour_cos",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24",
]
TARGET = "ssrd_wm2"

# Early-stopping round count. Placement differs by xgboost version:
#   - xgboost 3.x    : passed to the CONSTRUCTOR (XGBRegressor(...))
#   - xgboost 1.6-2.x: also accepts it in the constructor (deprecated
#                      there) OR in .fit(early_stopping_rounds=...)
# This run hit the 3.x API (TypeError: fit() got unexpected kwarg
# 'early_stopping_rounds'), so we pass it in XGB_PARAMS below.
EARLY_STOPPING_ROUNDS = 50

# Untuned hyperparameters (thesis Section 3.7)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)

# Phase 7 split files (frozen). We hash each of the 12 split CSVs.
SPLIT_FILES = [
    SPLITS_DIR / f"{city}_{split}.csv"
    for city in CITIES
    for split in ("train", "val", "test")
]


# --- Fingerprint helpers --------------------------------------------------
def snapshot_fingerprints() -> dict[str, str]:
    """Return {relative_path_str: sha1} for all frozen split files."""
    fp = {}
    for p in SPLIT_FILES:
        if not p.exists():
            sys.exit(f"ERROR: frozen split file missing: {p}")
        fp[str(p.relative_to(SPLITS_DIR.parent.parent))] = sha1_head_tail(p)
    return fp


def diff_fingerprints(before: dict, after: dict) -> None:
    """Fail loudly if any input fingerprint changed during the run."""
    if before == after:
        print("[fingerprint] start == end: OK (no input mutation)")
        return
    changed = [k for k in before if before.get(k) != after.get(k)]
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    sys.exit(
        "FATAL: input fingerprints changed during run.\n"
        f"  changed: {changed}\n  added:   {added}\n  removed: {removed}\n"
        "Frozen-folder policy violated."
    )


# --- Data loading ---------------------------------------------------------
def load_city(city: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLITS_DIR / f"{city}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{city}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{city}_test.csv")
    return train, val, test


def check_feature_columns(df: pd.DataFrame, city: str) -> None:
    """Ensure the 13 features + target are present; fail otherwise."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR [{city}]: missing columns in split: {missing}")


# --- Training -------------------------------------------------------------
def train_one_city(city: str) -> tuple[dict, dict]:
    """Train, predict, clip, and score one city.

    Returns (metrics_row, summary_row) as dicts.
    """
    print(f"\n=== {city} ===")
    train, val, test = load_city(city)
    for name, df in (("train", train), ("val", val), ("test", test)):
        check_feature_columns(df, f"{city}/{name}")
        print(f"  {name}: {len(df):,} rows, {df.shape[1]} cols")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # --- Early-stopping API note -----------------------------------------
    # xgboost 3.x: early_stopping_rounds lives in the CONSTRUCTOR
    #              (already set in XGB_PARAMS above).
    # xgboost 2.x: also accepted in the constructor (with a warning).
    # Do NOT pass early_stopping_rounds to .fit() — it raises TypeError
    # on 3.x. eval_set + verbose go to .fit() only.
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_it = getattr(model, "best_iteration", None)
    print(f"  trained: best_iteration={best_it}")

    # Predict on test ONLY here (after training is complete).
    y_pred = np.clip(model.predict(X_test), 0.0, None)

    # --- Overall metrics from the single source of truth ------------------
    m = all_metrics(y_test.to_numpy(), y_pred)
    mape_overall = m.get("mape", float("nan"))
    mae = m.get("mae", float("nan"))
    rmse = m.get("rmse", float("nan"))
    r2 = m.get("r2", float("nan"))
    n_total = int(len(y_test))

    # --- Daytime MAPE via the same single source of truth -----------------
    daytime_mape = mape_daytime(y_test.to_numpy(), y_pred)
    n_daytime = int(np.sum(y_test.to_numpy() >= 10))  # DAYTIME_THRESHOLD_WM2

    # --- Dry / wet seasonal breakdown -------------------------------------
    if "season" not in test.columns:
        sys.exit(f"ERROR [{city}]: 'season' column missing in test split.")
    season = test["season"].to_numpy()
    dry_mask = season == 0
    wet_mask = season == 1
    mape_dry = mape_daytime(y_test.to_numpy()[dry_mask], y_pred[dry_mask])
    mape_wet = mape_daytime(y_test.to_numpy()[wet_mask], y_pred[wet_mask])
    wet_minus_dry_pp = mape_wet - mape_dry

    print(
        f"  MAPE={mape_overall:.2f}%  daytime_mape={daytime_mape:.2f}%  "
        f"MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.3f}  "
        f"n_daytime={n_daytime:,}/{n_total:,}"
    )
    print(
        f"  dry_mape={mape_dry:.2f}%  wet_mape={mape_wet:.2f}%  "
        f"wet-dry={wet_minus_dry_pp:+.2f} pp"
    )

    metrics_row = dict(
        city=city,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape_overall,
        n_daytime=n_daytime,
        n_total=n_total,
    )
    summary_row = dict(
        city=city,
        daytime_mape=daytime_mape,
        dry_mape=mape_dry,
        wet_mape=mape_wet,
        wet_minus_dry_pp=wet_minus_dry_pp,
        mae=mae,
        rmse=rmse,
        r2=r2,
        n_daytime=n_daytime,
    )
    return metrics_row, summary_row


# --- Main -----------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 10: XGBoost Baseline (untuned, 13 features)")
    print("=" * 72)
    print(f"xgboost version: {xgb.__version__}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"Target:  {TARGET}")
    print(f"SPLITS_DIR:  {SPLITS_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")

    # --- Start fingerprint snapshot ---------------------------------------
    before = snapshot_fingerprints()
    print("\n[fingerprint] start snapshot:")
    for k, v in before.items():
        print(f"  {v}  {k}")
    print(
        "\n[fingerprint] compare these against docs/frozen_fingerprints.md "
        "(Phase 7 split SHA1s). Mismatch = STOP."
    )

    # --- Train all cities -------------------------------------------------
    metrics_rows, summary_rows = [], []
    for city in CITIES:
        mr, sr = train_one_city(city)
        metrics_rows.append(mr)
        summary_rows.append(sr)

    metrics_df = pd.DataFrame(metrics_rows).reindex(
        columns=["city", "mae", "rmse", "r2", "mape", "n_daytime", "n_total"]
    )
    summary_df = pd.DataFrame(summary_rows).reindex(
        columns=[
            "city", "daytime_mape", "dry_mape", "wet_mape",
            "wet_minus_dry_pp", "mae", "rmse", "r2", "n_daytime",
        ]
    )

    # --- Write outputs ----------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "metrics" / "xgb_baseline.csv"
    summary_path = RESULTS_DIR / "tables" / "xgb_baseline_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[wrote] {metrics_path}  ({len(metrics_df)} rows)")
    print(f"[wrote] {summary_path}  ({len(summary_df)} rows)")

    # --- Print final tables ----------------------------------------------
    print("\n--- xgb_baseline.csv ---")
    print(metrics_df.to_string(index=False))
    print("\n--- xgb_baseline_summary.csv ---")
    print(summary_df.to_string(index=False))

    # --- Plausibility check (non-fatal) -----------------------------------
    expected = {
        "muntinlupa": (18.0, 21.0),
        "baguio":     (17.0, 20.0),
        "cebu":       (19.0, 22.0),
        "davao":      (14.0, 17.0),
    }
    hard_lo, hard_hi = 12.0, 25.0
    print("\n--- Plausibility check (daytime MAPE) ---")
    problems = []
    for _, r in summary_df.iterrows():
        c = r["city"]
        v = r["daytime_mape"]
        lo, hi = expected.get(c, (hard_lo, hard_hi))
        in_soft = lo <= v <= hi
        in_hard = hard_lo <= v <= hard_hi
        flag = "OK" if in_soft else ("SOFT-FAIL" if in_hard else "HARD-FAIL")
        print(
            f"  {c:<11} {v:6.2f}%  expected [{lo:.1f},{hi:.1f}]  "
            f"hard [{hard_lo:.0f},{hard_hi:.0f}]  -> {flag}"
        )
        if not in_hard:
            problems.append(c)
    if problems:
        print(
            f"\n  WARNING: cities outside hard range {problems}. "
            "Investigate before proceeding to Phase 11."
        )
    print(
        "\n  Benchmark (WESM, daytime MAPE < 18%): "
        + ", ".join(
            f"{r['city']}={'PASS' if r['daytime_mape']<18 else 'FAIL'}"
            for _, r in summary_df.iterrows()
        )
    )

    # --- End fingerprint snapshot ----------------------------------------
    print("\n[fingerprint] end snapshot:")
    after = snapshot_fingerprints()
    for k, v in after.items():
        print(f"  {v}  {k}")
    diff_fingerprints(before, after)

    print("\nPhase 10 complete.")


if __name__ == "__main__":
    main()