"""
scripts/07c_xgb_tuned.py

Phase 11: Model 3 — Tuned XGBoost (80 features / 77 predictors)

Trains XGBoost with the full leakage-free 77-predictor feature set
for each of the 4 Philippine cities. Hyperparameters are selected
via grid search on the validation split (144 combinations per city).
The best combination is retrained on train+val with
n_estimators = 1.2 * best_iteration, then evaluated on the test set.

Outputs (all under results/paper1/):
  metrics/xgb_tuned.csv                 (4 rows, 7 cols)
  tables/xgb_tuned_summary.csv          (4 rows, 9 cols)
  tables/xgb_tuned_hyperparams.csv      (4 rows, 13 cols)

Rules honoured:
  - 77 predictors only (all 80 columns minus time, local_time, ssrd_wm2)
  - Metrics come exclusively from src.metrics
  - Predictions clipped to >= 0 before metrics
  - Fingerprints snapshotted at start and end (fail on mismatch)
  - Writes only to results/paper1/
  - sha1_head_tail imported from src.io (not reimplemented)
  - No sklearn train_test_split
"""

import itertools
import sys
from pathlib import Path

# --- Path bootstrap: make `src` importable when running this file directly ---
# When invoked as `python scripts/07c_xgb_tuned.py`, Python puts scripts/ on
# sys.path, not the repo root, so `import src...` fails. Prepend the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Project imports (single source of truth for paths, hashing, metrics)
# ---------------------------------------------------------------------------
from src.io import SPLITS_DIR, RESULTS_DIR, sha1_head_tail
from src.metrics import all_metrics, mape_daytime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CITIES = ["muntinlupa", "baguio", "cebu", "davao"]
ORDER = CITIES  # fixed order required by the brief

NON_FEATURE_COLS = ["time", "local_time", "ssrd_wm2"]
TARGET_COL = "ssrd_wm2"
EXPECTED_FEATURE_COUNT = 77
DAYTIME_THRESHOLD_WM2 = 10  # kept for reference; metrics handles this

# Hyperparameter grid: 4 * 4 * 3 * 3 = 144 combinations
PARAM_GRID = {
    "learning_rate":    [0.01, 0.05, 0.10, 0.15],
    "max_depth":        [4, 6, 8, 10],
    "min_child_weight": [1, 3, 5],
    "subsample":        [0.7, 0.85, 1.0],
}
GRID_KEYS = list(PARAM_GRID.keys())
GRID_COMBOS = list(itertools.product(*(PARAM_GRID[k] for k in GRID_KEYS)))
N_COMBOS = len(GRID_COMBOS)  # 144

FIXED_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=1000,
    early_stopping_rounds=50,
    eval_metric="rmse",
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)

PROGRESS_EVERY = 12

# Expected daytime MAPE ranges for the plausibility check
EXPECTED_RANGES = {
    "muntinlupa": (4.5, 6.5),
    "baguio":     (5.5, 7.5),
    "cebu":       (5.0, 7.0),
    "davao":      (4.0, 6.0),
}
HARD_BOUNDS = (3.0, 10.0)  # STOP and flag if any city falls outside

# Phase 10 reference MAPEs (for the >= 10 pp improvement check)
PHASE10_MAPE = {
    "muntinlupa": 18.87,
    "baguio":     18.24,
    "cebu":       20.11,
    "davao":      15.50,
}

WESM_BENCHMARK = 18.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_paths(city: str) -> dict:
    """Return the three split CSV paths for a city."""
    base = Path(SPLITS_DIR)
    return {
        "train": base / f"{city}_train.csv",
        "val":   base / f"{city}_val.csv",
        "test":  base / f"{city}_test.csv",
    }


def fingerprint_inputs() -> dict:
    """Snapshot SHA1 head/tail of every split file for all cities.

    Uses sha1_head_tail from src.io — never reimplemented.
    """
    fps = {}
    for city in ORDER:
        for split_name, path in split_paths(city).items():
            key = f"{city}_{split_name}"
            fps[key] = sha1_head_tail(path)
    return fps


def load_city(city: str):
    """Load train/val/test for a city and derive the 77-predictor list."""
    paths = split_paths(city)
    train = pd.read_csv(paths["train"], parse_dates=["time", "local_time"])
    val   = pd.read_csv(paths["val"],   parse_dates=["time", "local_time"])
    test  = pd.read_csv(paths["test"],  parse_dates=["time", "local_time"])

    features = [c for c in train.columns if c not in NON_FEATURE_COLS]

    # Hard assertion: must be exactly 77 predictors
    if len(features) != EXPECTED_FEATURE_COUNT:
        print(
            f"FATAL: [{city}] feature count = {len(features)}, "
            f"expected {EXPECTED_FEATURE_COUNT}. STOP."
        )
        sys.exit(1)

    # Sanity: no non-feature column leaked in
    for col in NON_FEATURE_COLS:
        if col in features:
            print(f"FATAL: [{city}] non-feature column '{col}' in feature list. STOP.")
            sys.exit(1)

    return train, val, test, features


def grid_search_city(city, train, val, features):
    """Run the 144-combination grid search on the validation split.

    For each combination:
      - fit on train with early stopping on val
      - compute daytime MAPE on val via src.metrics.mape_daytime
    Returns (best_params_dict, best_val_mape, best_iteration).
    """
    X_train = train[features].values
    y_train = train[TARGET_COL].values
    X_val   = val[features].values
    y_val   = val[TARGET_COL].values

    # Pre-compute the daytime mask on val once (threshold on observed ssrd)
    val_daytime_mask = y_val >= DAYTIME_THRESHOLD_WM2

    best_val_mape = np.inf
    best_params = None
    best_iteration = None

    for i, combo in enumerate(GRID_COMBOS, start=1):
        params = dict(zip(GRID_KEYS, combo))
        model = xgb.XGBRegressor(**params, **FIXED_PARAMS)

        # Early stopping on the validation split (RMSE)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # best_iteration_ is the 0-based index of the best round
        best_it = int(model.best_iteration) + 1

        # Predict on val, clip >= 0, restrict to daytime, compute MAPE
        pred_val = np.clip(model.predict(X_val), 0.0, None)
        val_mape = mape_daytime(
            y_val[val_daytime_mask],
            pred_val[val_daytime_mask],
        )

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_params = params
            best_iteration = best_it

        if i % PROGRESS_EVERY == 0 or i == N_COMBOS:
            print(
                f"  [{city}] combos {i}/{N_COMBOS} done "
                f"(best so far: val MAPE = {best_val_mape:.2f}%)"
            )

    return best_params, best_val_mape, best_iteration


def retrain_and_evaluate(city, train, val, test, features,
                         best_params, best_iteration):
    """Retrain on train+val and evaluate on test.

    Retraining step:
      - Concatenate train + val (13,128 + 4,416 = 17,544 rows)
      - n_estimators = ceil(1.2 * best_iteration)
      - NO early stopping on the combined set (there is no held-out
        eval set any more; the 1.2x inflation is the standard
        heuristic to compensate for the larger training set).
        early_stopping_rounds is therefore STRIPPED from the params
        before the final fit, otherwise XGBoost raises
        "Must have at least 1 validation dataset for early stopping."
    """
    n_estimators_final = int(np.ceil(1.2 * best_iteration))

    train_val = pd.concat([train, val], axis=0, ignore_index=True)
    X_trainval = train_val[features].values
    y_trainval = train_val[TARGET_COL].values

    X_test = test[features].values
    y_test = test[TARGET_COL].values

    # Drop early_stopping_rounds for the final fit: no eval_set here.
    final_params = {k: v for k, v in FIXED_PARAMS.items()
                    if k != "early_stopping_rounds"}
    final_model = xgb.XGBRegressor(
        **best_params,
        **{**final_params, "n_estimators": n_estimators_final},
    )
    # No eval_set / no early stopping on the combined train+val set.
    final_model.fit(X_trainval, y_trainval, verbose=False)

    pred_test = np.clip(final_model.predict(X_test), 0.0, None)

    return pred_test, y_test, n_estimators_final


def seasonal_breakdown(test, y_test, pred_test):
    """Compute dry/wet MAPE from the season column on the test split.

    The brief guarantees a 'season' column exists; if not, return NaN
    so the row is still written (schema fixed at 9 columns).
    """
    if "season" not in test.columns:
        return np.nan, np.nan, np.nan, np.nan

    season = test["season"].astype(str).str.lower()
    # Accept common encodings: dry/wet, dry_season/wet_season, 0/1
    dry_mask = season.isin(["dry", "dry_season", "0", "false"])
    wet_mask = season.isin(["wet", "wet_season", "1", "true"])

    # Restrict to daytime for a fair seasonal comparison
    daytime = y_test >= DAYTIME_THRESHOLD_WM2

    dry_mape = (
        mape_daytime(y_test[dry_mask & daytime], pred_test[dry_mask & daytime])
        if (dry_mask & daytime).sum() > 0 else np.nan
    )
    wet_mape = (
        mape_daytime(y_test[wet_mask & daytime], pred_test[wet_mask & daytime])
        if (wet_mask & daytime).sum() > 0 else np.nan
    )

    if np.isnan(dry_mape) or np.isnan(wet_mape):
        gap = np.nan
    else:
        gap = wet_mape - dry_mape

    return dry_mape, wet_mape, gap, daytime.sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Phase 11: Tuned XGBoost (77 predictors, 144-combo grid search)")
    print("=" * 72)

    # ---- Fingerprint snapshot at start -----------------------------------
    print("\n[1/4] Snapshotting input fingerprints at start ...")
    fps_start = fingerprint_inputs()
    for k, v in fps_start.items():
        print(f"  {k:24s} {v}")

    # ---- Per-city loop ----------------------------------------------------
    metrics_rows = []
    summary_rows = []
    hyper_rows = []

    for city in ORDER:
        print("\n" + "-" * 72)
        print(f"[CITY] {city}")
        print("-" * 72)

        train, val, test, features = load_city(city)
        print(f"  train rows = {len(train):>6d} | "
              f"val rows = {len(val):>6d} | "
              f"test rows = {len(test):>6d}")
        print(f"  feature count = {len(features)} (must be 77)")

        # ---- Grid search --------------------------------------------------
        print(f"  Running grid search over {N_COMBOS} combinations ...")
        best_params, best_val_mape, best_iteration = grid_search_city(
            city, train, val, features
        )
        print(f"  BEST: {best_params}")
        print(f"  best_iteration = {best_iteration}, "
              f"val MAPE = {best_val_mape:.2f}%")

        # ---- Retrain on train+val, evaluate on test -----------------------
        pred_test, y_test, n_est_final = retrain_and_evaluate(
            city, train, val, test, features, best_params, best_iteration
        )
        print(f"  retrained n_estimators = {n_est_final} "
              f"(= ceil(1.2 * {best_iteration}))")

        # ---- Metrics ------------------------------------------------------
        # all_metrics returns: mae, rmse, r2, mape, n_daytime
        # (no n_total key — use len(y_test) for the total row count)
        m = all_metrics(y_test, pred_test)
        metrics_rows.append({
            "city": city,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "mape": m["mape"],
            "n_daytime": m["n_daytime"],
            "n_total": len(y_test),
        })

        # ---- Seasonal breakdown ------------------------------------------
        dry_mape, wet_mape, gap, n_day = seasonal_breakdown(
            test, y_test, pred_test
        )
        wet_minus_dry_pp = gap if not np.isnan(gap) else 0.0

        summary_rows.append({
            "city": city,
            "daytime_mape": m["mape"],
            "dry_mape": dry_mape,
            "wet_mape": wet_mape,
            "wet_minus_dry_pp": wet_minus_dry_pp,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "n_daytime": m["n_daytime"],
        })

        hyper_rows.append({
            "city": city,
            "learning_rate": best_params["learning_rate"],
            "max_depth": best_params["max_depth"],
            "min_child_weight": best_params["min_child_weight"],
            "subsample": best_params["subsample"],
            "best_iteration": best_iteration,
            "n_estimators_final": n_est_final,
            "val_mape_at_best": best_val_mape,
            # Phase 18: widen CSV to fully describe the final fit
            "objective":        FIXED_PARAMS["objective"],
            "eval_metric":      FIXED_PARAMS["eval_metric"],
            "colsample_bytree": FIXED_PARAMS["colsample_bytree"],
            "random_state":     FIXED_PARAMS["random_state"],
            "tree_method":      FIXED_PARAMS["tree_method"],
        })

        # Per-city summary
        print(f"  [TEST] daytime MAPE = {m['mape']:.2f}% | "
              f"MAE = {m['mae']:.2f} | RMSE = {m['rmse']:.2f} | "
              f"R2 = {m['r2']:.4f}")
        print(f"  [TEST] dry MAPE = {dry_mape:.2f}% | "
              f"wet MAPE = {wet_mape:.2f}% | "
              f"gap = {wet_minus_dry_pp:+.2f} pp")
        print(f"  [TEST] n_daytime = {m['n_daytime']} | "
              f"n_total = {len(y_test)}")

    # ---- Write outputs ----------------------------------------------------
    print("\n[3/4] Writing outputs ...")
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=["city", "mae", "rmse", "r2", "mape",
                 "n_daytime", "n_total"],
    )
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["city", "daytime_mape", "dry_mape", "wet_mape",
                 "wet_minus_dry_pp", "mae", "rmse", "r2", "n_daytime"],
    )
    hyper_df = pd.DataFrame(
        hyper_rows,
        columns=["city", "learning_rate", "max_depth",
                 "min_child_weight", "subsample",
                 "best_iteration", "n_estimators_final",
                 "val_mape_at_best",
                 "objective", "eval_metric", "colsample_bytree",
                 "random_state", "tree_method"],
    )

    metrics_path = Path(RESULTS_DIR) / "metrics" / "xgb_tuned.csv"
    summary_path = Path(RESULTS_DIR) / "tables" / "xgb_tuned_summary.csv"
    hyper_path   = Path(RESULTS_DIR) / "tables" / "xgb_tuned_hyperparams.csv"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    hyper_df.to_csv(hyper_path, index=False)

    print(f"  wrote {metrics_path}")
    print(f"  wrote {summary_path}")
    print(f"  wrote {hyper_path}")

    # ---- Global summary tables -------------------------------------------
    print("\n[4/4] Global summary")
    print("\n--- xgb_tuned.csv ---")
    print(metrics_df.to_string(index=False))
    print("\n--- xgb_tuned_summary.csv ---")
    print(summary_df.to_string(index=False))
    print("\n--- xgb_tuned_hyperparams.csv ---")
    print(hyper_df.to_string(index=False))

    # ---- Plausibility check ----------------------------------------------
    print("\n--- Plausibility check ---")
    all_ok = True
    for row in metrics_rows:
        city = row["city"]
        mape = row["mape"]
        lo, hi = EXPECTED_RANGES[city]
        hard_lo, hard_hi = HARD_BOUNDS
        phase10 = PHASE10_MAPE[city]
        improvement = phase10 - mape

        in_range = lo <= mape <= hi
        in_hard = hard_lo <= mape <= hard_hi
        beats_wesm = mape < WESM_BENCHMARK
        beats_phase10 = improvement >= 10.0

        status = "PASS" if (in_range and in_hard and beats_wesm
                            and beats_phase10) else "FAIL"
        if status == "FAIL":
            all_ok = False

        print(
            f"  {city:11s} MAPE = {mape:5.2f}%  "
            f"(expected [{lo}, {hi}])  "
            f"vs Phase10 {phase10:5.2f}% "
            f"(delta = -{improvement:4.2f} pp)  "
            f"WESM {'PASS' if beats_wesm else 'FAIL'}  "
            f"hard-bounds {'OK' if in_hard else 'VIOLATED'}  "
            f"-> {status}"
        )

    print(f"\n  ALL FOUR CITIES BELOW 18% WESM: "
          f"{'YES' if all(r['mape'] < WESM_BENCHMARK for r in metrics_rows) else 'NO'}")
    print(f"  ALL FOUR IMPROVE >= 10 pp OVER PHASE 10: "
          f"{'YES' if all(PHASE10_MAPE[r['city']] - r['mape'] >= 10.0 for r in metrics_rows) else 'NO'}")
    print(f"  ALL FOUR INSIDE HARD BOUNDS [3%, 10%]: "
          f"{'YES' if all(HARD_BOUNDS[0] <= r['mape'] <= HARD_BOUNDS[1] for r in metrics_rows) else 'NO'}")

    if not all_ok:
        print("\n  *** WARNING: one or more cities failed a plausibility "
              "check. Review before treating Phase 11 as complete. ***")

    # ---- Fingerprint check at end ----------------------------------------
    print("\n[END] Snapshotting input fingerprints again ...")
    fps_end = fingerprint_inputs()
    mismatch = False
    for k in fps_start:
        if fps_start[k] != fps_end[k]:
            print(f"  MISMATCH: {k}  start={fps_start[k]}  end={fps_end[k]}")
            mismatch = True
    if mismatch:
        print("FATAL: input fingerprints changed during the run. STOP.")
        sys.exit(1)
    print("  All input fingerprints unchanged (start == end).")

    # ---- Output verification ---------------------------------------------
    print("\n--- Output verification ---")
    for name, df, path in [
        ("xgb_tuned.csv", metrics_df, metrics_path),
        ("xgb_tuned_summary.csv", summary_df, summary_path),
        ("xgb_tuned_hyperparams.csv", hyper_df, hyper_path),
    ]:
        reread = pd.read_csv(path)
        n_nan = int(reread.isna().sum().sum())
        print(f"  {name}: rows={len(reread)} cols={len(reread.columns)} "
              f"NaN={n_nan}")

    print("\nPhase 11 complete.")


if __name__ == "__main__":
    main()