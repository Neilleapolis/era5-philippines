#!/usr/bin/env python
"""
scripts/10_generate_figures.py

Phase 17: Figures for Paper 1.

Produces 11 PNG figures at 300 DPI under
results/paper1/figures/{exploratory,results,diagnostics}/.

The ONLY model fit in this script is the diagnostic re-fit of
the tuned XGBoost (Config A) used to produce actual-vs-predicted
figures. No grid search, no tuning, no Config B/C fits.

Run standalone from project root:
    python scripts/10_generate_figures.py
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from PIL import Image

# ----------------------------------------------------------------------
# Path bootstrap (mirrors scripts/07c_xgb_tuned.py)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import sha1_head_tail
from src.metrics import mape_daytime
from config.paper1_config import (
    DAYTIME_THRESHOLD_WM2,
    WESM_BENCHMARK_MAPE,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CITIES = ["muntinlupa", "baguio", "cebu", "davao"]

CITY_COLORS = {
    "muntinlupa": "#1f77b4",
    "baguio":     "#2ca02c",
    "cebu":       "#d62728",
    "davao":      "#ff7f0e",
}

MODEL_COLORS = {
    "persistence":  "#7f7f7f",
    "xgb_baseline": "#9467bd",
    "xgb_tuned":    "#1f77b4",
    "ablation_A":   "#1f77b4",
    "ablation_B":   "#ff7f0e",
    "ablation_C":   "#d62728",
}

NON_FEATURE_COLS = ["time", "local_time", "ssrd_wm2"]
EXPECTED_FEATURE_COUNT = 77

# Hyperparameter columns that are metadata, not XGBoost kwargs.
HP_DROP = {"city", "best_iteration", "n_estimators_final", "val_mape_at_best"}

# Mirrors scripts/07c_xgb_tuned.py FIXED_PARAMS (lines 71-79).
# early_stopping_rounds is stripped by Phase 11 before the final fit
# (line 234), and n_estimators is overridden by n_estimators_final
# (line 238), so neither appears here.
PHASE11_FIXED_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "colsample_bytree": 0.8,
    "random_state":     42,
    "tree_method":      "hist",
}

# Paths
RESULTS = PROJECT_ROOT / "results" / "paper1"
METRICS = RESULTS / "metrics"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
FIG_EXPLORATORY = FIGURES / "exploratory"
FIG_RESULTS = FIGURES / "results"
FIG_DIAGNOSTICS = FIGURES / "diagnostics"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "paper1"

INPUT_METRIC_FILES = [
    METRICS / "persistence.csv",
    METRICS / "xgb_baseline.csv",
    METRICS / "xgb_tuned.csv",
    METRICS / "ablation_xgb.csv",
]
INPUT_TABLE_FILES = [
    TABLES / "master_results.csv",
    TABLES / "paper_table1_models.csv",
    TABLES / "paper_table2_ablation.csv",
    TABLES / "paper_table3_seasonal.csv",
    TABLES / "xgb_tuned_hyperparams.csv",
]

# ----------------------------------------------------------------------
# matplotlib rcParams
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
plt.style.use("seaborn-v0_8-whitegrid")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _assert_inputs_exist():
    missing = []
    for p in INPUT_METRIC_FILES + INPUT_TABLE_FILES:
        if not p.exists():
            missing.append(str(p))
    for city in CITIES:
        for split in ("train", "val", "test"):
            p = SPLIT_DIR / f"{city}_{split}.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n  " + "\n  ".join(missing)
        )


def _fingerprint_inputs():
    """Use src.io.sha1_head_tail for the input fingerprint (Rule 8)."""
    paths = list(INPUT_METRIC_FILES) + list(INPUT_TABLE_FILES)
    for city in CITIES:
        for split in ("train", "val", "test"):
            paths.append(SPLIT_DIR / f"{city}_{split}.csv")
    return {str(p): sha1_head_tail(str(p)) for p in paths}


def _load_split(city, split):
    p = SPLIT_DIR / f"{city}_{split}.csv"
    df = pd.read_csv(p)
    for col in NON_FEATURE_COLS:
        if col not in df.columns:
            raise KeyError(f"{p} missing required column '{col}'")
    df["local_time"] = pd.to_datetime(df["local_time"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def _predictor_cols(df):
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if len(feats) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"feature count = {len(feats)}, expected {EXPECTED_FEATURE_COUNT}"
        )
    for col in NON_FEATURE_COLS:
        if col in feats:
            raise RuntimeError(f"non-feature column '{col}' leaked into feature list")
    return feats


def _mape_pct(y_true, y_pred):
    """src.metrics.mape_daytime returns percent already; do not scale."""
    return float(mape_daytime(y_true, y_pred, threshold=DAYTIME_THRESHOLD_WM2))


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def _write_csv(rows, path, columns):
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


# ----------------------------------------------------------------------
# Diagnostic re-fit (the ONLY model fit in this script)
# ----------------------------------------------------------------------
def _load_hyperparams():
    p = TABLES / "xgb_tuned_hyperparams.csv"
    df = pd.read_csv(p)
    if "city" not in df.columns:
        raise KeyError(f"{p} missing 'city' column")
    return df.set_index("city")


def _diagnostic_refit(city, hp_row):
    train = _load_split(city, "train")
    val = _load_split(city, "val")
    test = _load_split(city, "test")

    features = _predictor_cols(train)
    for col in NON_FEATURE_COLS:
        if col in features:
            raise RuntimeError(f"non-feature column '{col}' in feature list")

    X_trainval = pd.concat([train[features], val[features]], axis=0).values
    y_trainval = pd.concat([train["ssrd_wm2"], val["ssrd_wm2"]], axis=0).values

    hp = {k: (v.item() if hasattr(v, "item") else v) for k, v in hp_row.items()}
    n_estimators = int(hp.pop("n_estimators_final"))
    for k in HP_DROP:
        hp.pop(k, None)
    for k in PHASE11_FIXED_PARAMS:
        hp.pop(k, None)

    # XGBoost requires integer-typed params (e.g. max_depth) to be
    # Python ints, not floats. Coerce any integral-valued entry.
    for k, v in list(hp.items()):
        if isinstance(v, float) and v.is_integer():
            hp[k] = int(v)

    model = XGBRegressor(
        n_estimators=n_estimators,
        **PHASE11_FIXED_PARAMS,
        **hp,
    )
    model.fit(X_trainval, y_trainval, verbose=False)

    X_test = test[features].values
    y_true = test["ssrd_wm2"].values
    y_pred = np.clip(model.predict(X_test), 0.0, None)

    return {
        "test": test,
        "y_true": y_true,
        "y_pred": y_pred,
        "features": features,
        "model": model,
    }


# ----------------------------------------------------------------------
# Figure 1 — Diurnal profiles by season
# ----------------------------------------------------------------------
def fig01_diurnal_profiles():
    print("[Figure 1] diurnal profiles by season")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, city in zip(axes, CITIES):
        df = _load_split(city, "train")
        df = df[df["ssrd_wm2"] > DAYTIME_THRESHOLD_WM2].copy()
        df["hour"] = df["local_time"].dt.hour
        df["month"] = df["local_time"].dt.month
        df["season"] = np.where(df["month"].isin([11, 12, 1, 2, 3, 4]), "dry", "wet")
        for season, color, ls in (("dry", "#d62728", "-"), ("wet", "#1f77b4", "--")):
            sub = df[df["season"] == season]
            if sub.empty:
                continue
            g = sub.groupby("hour")["ssrd_wm2"]
            mean = g.mean()
            std = g.std().fillna(0.0)
            ax.plot(mean.index, mean.values, color=color, linestyle=ls, label=season)
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(city.capitalize())
        ax.set_xlabel("Hour of day (local, 0-23)")
        ax.set_ylabel("ssrd_wm2 (W/m$^2$)")
        ax.set_xlim(0, 23)
        ax.legend(title="Season", fontsize=9, title_fontsize=9)
    fig.suptitle("Mean diurnal solar radiation by season (train split)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, FIG_EXPLORATORY / "fig01_diurnal_profiles.png")

    rows = []
    for city in CITIES:
        df = _load_split(city, "train")
        df = df[df["ssrd_wm2"] > DAYTIME_THRESHOLD_WM2].copy()
        df["hour"] = df["local_time"].dt.hour
        df["month"] = df["local_time"].dt.month
        df["season"] = np.where(df["month"].isin([11, 12, 1, 2, 3, 4]), "dry", "wet")
        for season in ("dry", "wet"):
            sub = df[df["season"] == season]
            for hour in range(24):
                h = sub[sub["hour"] == hour]["ssrd_wm2"]
                rows.append({
                    "city": city,
                    "season": season,
                    "hour_local": hour,
                    "mean_ssrd_wm2": float(h.mean()) if len(h) else np.nan,
                    "std_ssrd_wm2": float(h.std()) if len(h) > 1 else np.nan,
                    "n_hours": int(len(h)),
                })
    _write_csv(
        rows,
        FIG_EXPLORATORY / "fig01_data.csv",
        ["city", "season", "hour_local", "mean_ssrd_wm2",
         "std_ssrd_wm2", "n_hours"],
    )


# ----------------------------------------------------------------------
# Figure 2 — Monthly average solar radiation
# ----------------------------------------------------------------------
def fig02_monthly_radiation():
    print("[Figure 2] monthly average radiation")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    months = np.arange(1, 13)
    for ax, city in zip(axes, CITIES):
        df = _load_split(city, "train")
        df = df[df["ssrd_wm2"] > DAYTIME_THRESHOLD_WM2].copy()
        df["month"] = df["local_time"].dt.month
        df["season"] = np.where(df["month"].isin([11, 12, 1, 2, 3, 4]), "dry", "wet")
        means = df.groupby(["month", "season"])["ssrd_wm2"].mean().unstack(fill_value=np.nan)
        dry = means["dry"] if "dry" in means else pd.Series(np.nan, index=months)
        wet = means["wet"] if "wet" in means else pd.Series(np.nan, index=months)
        width = 0.4
        ax.bar(months - width / 2, dry.reindex(months).values, width,
               color="#d62728", label="dry")
        ax.bar(months + width / 2, wet.reindex(months).values, width,
               color="#1f77b4", label="wet")
        ax.set_title(city.capitalize())
        ax.set_xticks(months)
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean ssrd_wm2 (W/m$^2$)")
        ax.legend(title="Season", fontsize=9, title_fontsize=9)
    fig.suptitle("Monthly mean solar radiation by season (train split)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, FIG_EXPLORATORY / "fig02_monthly_radiation.png")

    rows = []
    for city in CITIES:
        df = _load_split(city, "train")
        df = df[df["ssrd_wm2"] > DAYTIME_THRESHOLD_WM2].copy()
        df["month"] = df["local_time"].dt.month
        for month in range(1, 13):
            sub = df[df["month"] == month]["ssrd_wm2"]
            season = "dry" if month in (11, 12, 1, 2, 3, 4) else "wet"
            rows.append({
                "city": city,
                "month": month,
                "season": season,
                "mean_ssrd_wm2": float(sub.mean()) if len(sub) else np.nan,
                "n_hours": int(len(sub)),
            })
    _write_csv(
        rows,
        FIG_EXPLORATORY / "fig02_data.csv",
        ["city", "month", "season", "mean_ssrd_wm2", "n_hours"],
    )


# ----------------------------------------------------------------------
# Figure 3 — Model progression
# ----------------------------------------------------------------------
def fig03_model_progression():
    print("[Figure 3] model progression")
    t1 = pd.read_csv(TABLES / "paper_table1_models.csv").set_index("city")
    values = {
        "persistence":  [float(t1.loc[c, "persistence_mape"])  for c in CITIES],
        "xgb_baseline": [float(t1.loc[c, "xgb_baseline_mape"]) for c in CITIES],
        "xgb_tuned":    [float(t1.loc[c, "xgb_tuned_mape"])    for c in CITIES],
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CITIES))
    width = 0.27
    for i, (name, vals) in enumerate(values.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, color=MODEL_COLORS[name], label=name)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(WESM_BENCHMARK_MAPE, color="red", linestyle="--", linewidth=1.2,
               label=f"WESM benchmark ({WESM_BENCHMARK_MAPE:.0f}%)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CITIES])
    ax.set_xlabel("City")
    ax.set_ylabel("Daytime MAPE (%, log scale)")
    ax.set_title("Model progression: persistence to tuned XGBoost")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, FIG_RESULTS / "fig03_model_progression.png")


# ----------------------------------------------------------------------
# Figure 4 — Ablation bar chart
# ----------------------------------------------------------------------
def fig04_ablation_bars():
    print("[Figure 4] ablation bars")
    ab = pd.read_csv(TABLES / "paper_table2_ablation.csv").set_index("city")
    cols = [("A", "config_a_mape", "ablation_A"),
            ("B", "config_b_mape", "ablation_B"),
            ("C", "config_c_mape", "ablation_C")]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CITIES))
    width = 0.27
    for i, (label, col, key) in enumerate(cols):
        vals = [float(ab.loc[c, col]) for c in CITIES]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, color=MODEL_COLORS[key],
                      label=f"Config {label}")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(WESM_BENCHMARK_MAPE, color="red", linestyle="--", linewidth=1.2,
               label=f"WESM benchmark ({WESM_BENCHMARK_MAPE:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CITIES])
    ax.set_xlabel("City")
    ax.set_ylabel("Daytime MAPE (%)")
    ax.set_title("Feature-configuration ablation (tuned XGBoost)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, FIG_RESULTS / "fig04_ablation_bars.png")


# ----------------------------------------------------------------------
# Figure 5 — Ablation value decomposition
# ----------------------------------------------------------------------
def fig05_ablation_values():
    print("[Figure 5] ablation value decomposition")
    ab = pd.read_csv(TABLES / "paper_table2_ablation.csv").set_index("city")
    fdir = [float(ab.loc[c, "fdir_value_pp"]) for c in CITIES]
    other = [float(ab.loc[c, "other_rad_value_pp"]) for c in CITIES]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(CITIES))

    axL.bar(x, fdir, color=[CITY_COLORS[c] for c in CITIES])
    axL.set_xticks(x)
    axL.set_xticklabels([c.capitalize() for c in CITIES])
    axL.set_xlabel("City")
    axL.set_ylabel("MAPE points (pp)")
    axL.set_title("Value of fdir_wm2 (Config A $\\to$ B)")
    axL.set_ylim(0, 10)
    for xi, v in zip(x, fdir):
        axL.annotate(f"{v:.3f}", (xi, v), ha="center", va="bottom", fontsize=9)

    axR.bar(x, other, color=[CITY_COLORS[c] for c in CITIES])
    axR.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axR.set_xticks(x)
    axR.set_xticklabels([c.capitalize() for c in CITIES])
    axR.set_xlabel("City")
    axR.set_ylabel("MAPE points (pp)")
    axR.set_title("Value of remaining radiation features (Config B $\\to$ C)")
    axR.set_ylim(-0.5, 0.5)
    for xi, v in zip(x, other):
        va = "bottom" if v >= 0 else "top"
        axR.annotate(f"{v:+.3f}", (xi, v), ha="center", va=va, fontsize=9)

    fig.suptitle("Value of radiation features (Config A $\\to$ B $\\to$ C)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, FIG_RESULTS / "fig05_ablation_values.png")


# ----------------------------------------------------------------------
# Figure 6 — Seasonal gap by city
# ----------------------------------------------------------------------
def fig06_seasonal_gap():
    print("[Figure 6] seasonal gap")
    seas = pd.read_csv(TABLES / "paper_table3_seasonal.csv")
    seas = seas[seas["model_or_config"] == "ablation_C"]
    if seas.empty:
        raise ValueError("paper_table3_seasonal.csv has no ablation_C rows")
    seas = seas.set_index("city")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CITIES))
    width = 0.35
    for i, (col, label, color) in enumerate(
        (("dry_mape", "dry", "#d62728"), ("wet_mape", "wet", "#1f77b4"))
    ):
        vals = [float(seas.loc[c, col]) for c in CITIES]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(WESM_BENCHMARK_MAPE, color="red", linestyle="--", linewidth=1.2,
               label=f"WESM benchmark ({WESM_BENCHMARK_MAPE:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CITIES])
    ax.set_xlabel("City")
    ax.set_ylabel("Daytime MAPE (%)")
    ax.set_title("Seasonal MAPE gap (Config C)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, FIG_RESULTS / "fig06_seasonal_gap.png")


# ----------------------------------------------------------------------
# Figure 7 — Cumulative MAPE distribution
# ----------------------------------------------------------------------
def _per_hour_ape(y_true, y_pred, mask, eps=1e-9):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    denom = np.where(np.abs(yt) < eps, np.nan, yt)
    ape = np.abs((yt - yp) / denom) * 100.0
    return ape[mask]


def fig07_cumulative_mape(diagnostics):
    print("[Figure 7] cumulative MAPE distribution")
    thresholds = np.linspace(0, 60, 301)

    def _cdf(ape):
        a = np.asarray(ape)
        return np.array([(a <= t).mean() for t in thresholds])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, city in zip(axes, CITIES):
        d = diagnostics[city]
        yt = d["y_true"]
        yp = d["y_pred"]
        mask = yt > DAYTIME_THRESHOLD_WM2

        ape_tuned = _per_hour_ape(yt, yp, mask)

        y_pers = pd.Series(yt).shift(24).values
        y_pers = np.nan_to_num(y_pers, nan=0.0)
        ape_pers = _per_hour_ape(yt, y_pers, mask)

        ax.plot(thresholds, _cdf(ape_pers), color=MODEL_COLORS["persistence"],
                linestyle="--", label="persistence")
        ax.plot(thresholds, _cdf(ape_tuned), color=MODEL_COLORS["xgb_tuned"],
                linestyle="-", label="xgb_tuned (Config A)")
        ax.axvline(WESM_BENCHMARK_MAPE, color="red", linestyle=":", linewidth=1.2,
                   label=f"{WESM_BENCHMARK_MAPE:.0f}%")
        ax.set_title(city.capitalize())
        ax.set_xlabel("MAPE threshold (%)")
        ax.set_ylabel("Fraction of daytime hours")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Cumulative distribution of daytime MAPE "
        f"(daytime = ssrd_wm2 > {DAYTIME_THRESHOLD_WM2:.0f} W/m$^2$)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, FIG_RESULTS / "fig07_cumulative_mape.png")

    rows = []
    for city in CITIES:
        d = diagnostics[city]
        yt = d["y_true"]
        yp = d["y_pred"]
        mask = yt > DAYTIME_THRESHOLD_WM2

        ape_tuned = _per_hour_ape(yt, yp, mask)
        y_pers = pd.Series(yt).shift(24).values
        y_pers = np.nan_to_num(y_pers, nan=0.0)
        ape_pers = _per_hour_ape(yt, y_pers, mask)

        for model, ape in (("persistence", ape_pers), ("xgb_tuned", ape_tuned)):
            a = np.asarray(ape)
            for t in thresholds:
                rows.append({
                    "city": city,
                    "model": model,
                    "mape_threshold_pct": float(t),
                    "fraction_below": float((a <= t).mean()),
                })
    _write_csv(
        rows,
        FIG_RESULTS / "fig07_data.csv",
        ["city", "model", "mape_threshold_pct", "fraction_below"],
    )


# ----------------------------------------------------------------------
# Figure 8 — Actual vs predicted time series (Muntinlupa)
# ----------------------------------------------------------------------
def _pick_extreme_week(df, want_highest):
    """First full Mon-Sun week (daytime rows) by mean ssrd_wm2."""
    df = df.copy()
    iso = df["local_time"].dt.isocalendar()
    df["iso_year"] = iso["year"].values
    df["iso_week"] = iso["week"].values

    candidates = []
    for (yr, wk), grp in df.groupby(["iso_year", "iso_week"]):
        dates = grp["local_time"].dt.date.unique()
        if len(dates) == 7:
            candidates.append(((yr, wk), grp))
    if not candidates:
        raise ValueError("no full Mon-Sun weeks in test set")

    means = {key: grp["ssrd_wm2"].mean() for key, grp in candidates}
    pick_key = (max if want_highest else min)(means, key=means.get)
    grp = dict(candidates)[pick_key]
    start = grp["local_time"].min().normalize()
    end = grp["local_time"].max().normalize()
    return start, end


def fig08_actual_vs_predicted(diagnostics):
    print("[Figure 8] actual vs predicted time series")
    city = "muntinlupa"
    d = diagnostics[city]
    test = d["test"].copy()
    test["y_pred_A"] = d["y_pred"]
    test["y_true"] = d["y_true"]
    test = test[test["y_true"] > DAYTIME_THRESHOLD_WM2].copy()

    dry_start, dry_end = _pick_extreme_week(test, want_highest=True)
    wet_start, wet_end = _pick_extreme_week(test, want_highest=False)
    print(f"  dry week: {dry_start.date()} -> {dry_end.date()}")
    print(f"  wet week: {wet_start.date()} -> {wet_end.date()}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for ax, (label, start, end) in zip(
        axes,
        (("dry", dry_start, dry_end), ("wet", wet_start, wet_end)),
    ):
        wk = test[(test["local_time"] >= start) &
                  (test["local_time"] < end + pd.Timedelta(days=1))]
        ax.plot(wk["local_time"], wk["y_true"], color="black", linewidth=1.2,
                label="actual")
        ax.plot(wk["local_time"], wk["y_pred_A"], color=MODEL_COLORS["xgb_tuned"],
                linewidth=1.0, label="xgb_tuned (Config A)")
        ax.set_title(f"{label.capitalize()} week: {start.date()} to {end.date()}")
        ax.set_xlabel("Local time")
        ax.set_ylabel("ssrd_wm2 (W/m$^2$)")
        ax.legend(fontsize=9)
    fig.suptitle(f"Actual vs predicted solar radiation — {city.capitalize()} "
                 "(test set, daytime hours)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, FIG_DIAGNOSTICS / "fig08_actual_vs_predicted.png")


# ----------------------------------------------------------------------
# Figure 9 — Scatter plots
# ----------------------------------------------------------------------
def fig09_scatter(diagnostics):
    print("[Figure 9] scatter actual vs predicted")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()
    for ax, city in zip(axes, CITIES):
        d = diagnostics[city]
        yt = d["y_true"]
        yp = d["y_pred"]
        mask = yt > DAYTIME_THRESHOLD_WM2
        ax.scatter(yt[mask], yp[mask], s=4, alpha=0.25,
                   color=CITY_COLORS[city], edgecolors="none")
        lim = max(np.nanmax(yt[mask]), np.nanmax(yp[mask])) * 1.05
        ax.plot([0, lim], [0, lim], color="red", linewidth=1.0, label="1:1")
        mape = _mape_pct(yt, yp)
        ax.set_title(f"{city.capitalize()} (MAPE = {mape:.2f}%)")
        ax.set_xlabel("Actual ssrd_wm2 (W/m$^2$)")
        ax.set_ylabel("Predicted ssrd_wm2 (W/m$^2$)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs actual solar radiation (tuned XGBoost, Config A)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, FIG_DIAGNOSTICS / "fig09_scatter_actual_predicted.png")


# ----------------------------------------------------------------------
# Figure 10 — Feature importance
# ----------------------------------------------------------------------
def fig10_feature_importance(diagnostics):
    print("[Figure 10] feature importance")
    gains = {}
    for city in CITIES:
        booster = diagnostics[city]["model"].get_booster()
        score = booster.get_score(importance_type="gain")
        feats = diagnostics[city]["features"]
        mapped = {}
        for k, v in score.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feats):
                    mapped[feats[idx]] = v
            else:
                mapped[k] = v
        gains[city] = mapped

    all_feats = set()
    for g in gains.values():
        all_feats.update(g.keys())
    avg = {f: float(np.mean([gains[c].get(f, 0.0) for c in CITIES])) for f in all_feats}
    top = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:15]

    names = [k for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, vals, color="#1f77b4")
    ax.set_xlabel("Mean gain (unitless)")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 features by gain (tuned XGBoost, Config A, mean of 4 cities)")
    fig.tight_layout()
    _save(fig, FIG_DIAGNOSTICS / "fig10_feature_importance.png")

    rows = []
    for rank, (feat, val) in enumerate(top, start=1):
        rows.append({"feature": feat, "mean_gain": float(val), "rank": rank})
    _write_csv(
        rows,
        FIG_DIAGNOSTICS / "fig10_data.csv",
        ["feature", "mean_gain", "rank"],
    )


# ----------------------------------------------------------------------
# Figure 11 — Residual distributions
# ----------------------------------------------------------------------
def fig11_residuals(diagnostics):
    print("[Figure 11] residual distributions")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, city in zip(axes, CITIES):
        d = diagnostics[city]
        yt = d["y_true"]
        yp = d["y_pred"]
        mask = yt > DAYTIME_THRESHOLD_WM2
        resid = yp[mask] - yt[mask]
        ax.hist(resid, bins=60, density=True,
                color=CITY_COLORS[city], alpha=0.8)
        ax.axvline(0, color="red", linewidth=1.0)
        ax.set_title(f"{city.capitalize()} (mean={resid.mean():.2f}, "
                     f"std={resid.std():.2f})")
        ax.set_xlabel("Residual (predicted $-$ actual, W/m$^2$)")
        ax.set_ylabel("Density (unitless)")
    fig.suptitle("Residual distributions (tuned XGBoost, Config A, daytime test)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, FIG_DIAGNOSTICS / "fig11_residuals.png")

    rows = []
    for city in CITIES:
        d = diagnostics[city]
        yt = d["y_true"]
        yp = d["y_pred"]
        mask = yt > DAYTIME_THRESHOLD_WM2
        resid = yp[mask] - yt[mask]

        counts, edges = np.histogram(resid, bins=60, density=False)
        mean_r = float(resid.mean())
        std_r = float(resid.std())
        for i, c in enumerate(counts):
            rows.append({
                "city": city,
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "count": int(c),
                "mean_residual": mean_r,
                "std_residual": std_r,
            })
    _write_csv(
        rows,
        FIG_DIAGNOSTICS / "fig11_data.csv",
        ["city", "bin_left", "bin_right", "count",
         "mean_residual", "std_residual"],
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 17 — Figure generation")
    print("=" * 60)
    print(f"DAYTIME_THRESHOLD_WM2 = {DAYTIME_THRESHOLD_WM2}")
    print(f"WESM_BENCHMARK_MAPE   = {WESM_BENCHMARK_MAPE}")

    _assert_inputs_exist()
    fp_before = _fingerprint_inputs()

    for d in (FIG_EXPLORATORY, FIG_RESULTS, FIG_DIAGNOSTICS):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Diagnostic re-fit] tuned XGBoost, Config A, one fit per city")
    hp = _load_hyperparams()
    tuned = pd.read_csv(METRICS / "xgb_tuned.csv").set_index("city")
    diagnostics = {}
    for city in CITIES:
        print(f"  fitting {city} ...", end=" ", flush=True)
        diagnostics[city] = _diagnostic_refit(city, hp.loc[city])
        d = diagnostics[city]
        mape_refit = _mape_pct(d["y_true"], d["y_pred"])
        mape_phase11 = float(tuned.loc[city, "mape"])
        delta = abs(mape_refit - mape_phase11)
        print(f"MAPE={mape_refit:.6f}% (Phase 11={mape_phase11:.6f}%, "
              f"delta={delta:.3e} pp)")
        if delta > 1e-6:
            print(f"  WARNING: {city} re-fit MAPE differs from Phase 11 "
                  f"by {delta:.3e} pp (> 1e-6)")

    print("\n[Figures]")
    fig01_diurnal_profiles()
    fig02_monthly_radiation()
    fig03_model_progression()
    fig04_ablation_bars()
    fig05_ablation_values()
    fig06_seasonal_gap()
    fig07_cumulative_mape(diagnostics)
    fig08_actual_vs_predicted(diagnostics)
    fig09_scatter(diagnostics)
    fig10_feature_importance(diagnostics)
    fig11_residuals(diagnostics)

    fp_after = _fingerprint_inputs()
    changed = [p for p in fp_before if fp_before[p] != fp_after[p]]
    if changed:
        raise RuntimeError(f"Input files modified: {changed}")

    print("\n" + "=" * 60)
    print("Figure summary")
    print("=" * 60)
    pngs = sorted(FIGURES.rglob("*.png"))
    for p in pngs:
        with Image.open(p) as im:
            w, h = im.size
        print(f"  {p.relative_to(PROJECT_ROOT)}  ({w} x {h} px)")
    print(f"\nTotal: {len(pngs)} PNG files")
    print("Input fingerprints unchanged: yes")


if __name__ == "__main__":
    main()