"""
src/features.py
===============
Leakage-free feature engineering for era5-philippines (Phase 6).

Each add_* function appends its category's columns to a DataFrame
and returns it. compute_all_features() applies them in the exact
order A → N specified in the Phase 6 brief.

Leakage contract (non-negotiable):
- No function reads current-hour ssrd_wm2 or fdir_wm2.
- All lags use shift >= 1.
- All rolling windows operate on ssrd_wm2.shift(1).
- Interaction features use lag_1, not current-hour ssrd_wm2.
- Geometry columns from Phase 5 are reused, never re-derived.
"""

import numpy as np
import pandas as pd

RAIN_MM_THRESHOLD = 0.1  # WMO; matches Phase 4


# ---------------------------------------------------------------------
# A. Time (10)
# ---------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    lt = pd.to_datetime(df["local_time"])
    df["hour_local"]  = lt.dt.hour
    df["month"]       = lt.dt.month
    df["day_of_year"] = lt.dt.dayofyear
    df["weekday"]     = lt.dt.weekday + 1  # Mon=1 .. Sun=7

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_local"] / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ---------------------------------------------------------------------
# B. Solar geometry — already in with_geometry. Skip.
# C. Direct radiation — already in with_geometry. Skip.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# D. Core weather derived (1)
# ---------------------------------------------------------------------
def add_core_weather(df: pd.DataFrame) -> pd.DataFrame:
    t = df["t2m_c"]
    d = df["d2m_c"]
    # Magnus-Tetens. Clip to [0,100] because supersaturation is
    # unphysical and would create noisy extremes.
    e  = np.exp((17.625 * d) / (243.04 + d))
    es = np.exp((17.625 * t) / (243.04 + t))
    df["rh"] = (100.0 * e / es).clip(0, 100)
    return df


# ---------------------------------------------------------------------
# E. Thermodynamic (3)
# ---------------------------------------------------------------------
def add_thermodynamic(df: pd.DataFrame) -> pd.DataFrame:
    t  = df["t2m_c"]
    td = df["d2m_c"]
    rh = df["rh"]

    def es(T):
        # Bolton (1980) saturation vapor pressure, hPa.
        return 6.112 * np.exp(17.67 * T / (T + 243.5))

    df["vapor_pressure_deficit"] = es(t) - es(td)

    # Stull (2011) wet-bulb approximation.
    Tw = (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
          + np.arctan(t + rh)
          - np.arctan(rh - 1.676331)
          + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
          - 4.686035)
    df["wet_bulb_temp"] = Tw

    # Specific humidity (g/kg). p = 1013.25 hPa assumed because
    # ERA5 surface pressure is not in our 15-variable set. This
    # is a documented approximation.
    e_actual = es(td)
    p = 1013.25
    df["specific_humidity"] = 622.0 * e_actual / (p - 0.378 * e_actual)
    return df


# ---------------------------------------------------------------------
# F. Cloud engineered (4)
# ---------------------------------------------------------------------
def add_cloud_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df["cloud_opacity"] = (0.45 * df["lcc"] + 0.30 * df["mcc"]
                           + 0.15 * df["hcc"] + 0.10 * df["tcc"])
    df["tclw_tciw_sum"] = df["tclw"] + df["tciw"]
    df["convective_flag"] = (
        (df["tclw"] + df["tciw"] > 0.05)
        & (df["tp_mm"] > RAIN_MM_THRESHOLD)
    ).astype(int)
    df["blh_rh_interaction"] = (df["blh"] / 3000.0) * (df["rh"] / 100.0)
    return df


# ---------------------------------------------------------------------
# G. Cloud dynamics (8) — all shifts are on cloud columns, never
#    the target. Depends on F for cloud_opacity.
# ---------------------------------------------------------------------
def add_cloud_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    df["cloud_trend_3h"] = df["tcc"] - df["tcc"].shift(3)
    df["cloud_trend_6h"] = df["tcc"] - df["tcc"].shift(6)
    df["cloud_lag_1"]    = df["tcc"].shift(1)
    df["cloud_lag_3"]    = df["tcc"].shift(3)
    df["cloud_lag_6"]    = df["tcc"].shift(6)
    df["lcc_lag_1"]      = df["lcc"].shift(1)
    df["lcc_lag_3"]      = df["lcc"].shift(3)
    df["opacity_lag_1"]  = df["cloud_opacity"].shift(1)
    return df


# ---------------------------------------------------------------------
# H. Rain (5)
# ---------------------------------------------------------------------
def add_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rain_binary"] = (df["tp_mm"] >= RAIN_MM_THRESHOLD).astype(int)

    # All precip sums use shift(1) so the current hour is NEVER
    # included. Range [t-3, t-1] = 3 past hours, shifted.
    tp_shift = df["tp_mm"].shift(1)
    df["precip_3h"]  = tp_shift.rolling(3,  min_periods=1).sum()
    df["precip_6h"]  = tp_shift.rolling(6,  min_periods=1).sum()
    df["precip_24h"] = tp_shift.rolling(24, min_periods=1).sum()

    # hours_since_rain recomputed from shifted rain_binary so the
    # current hour's rain does not reset the counter.
    rb = df["rain_binary"].shift(1).fillna(0).astype(int).values
    out = np.zeros(len(rb), dtype=int)
    counter = 72
    for i in range(len(rb)):
        if rb[i] == 1:
            counter = 0
        else:
            counter = min(counter + 1, 72)
        out[i] = counter
    df["hours_since_rain"] = out
    return df


# ---------------------------------------------------------------------
# I. Lag features (6) — strictly past target
# ---------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_1"]   = df["ssrd_wm2"].shift(1)
    df["lag_2"]   = df["ssrd_wm2"].shift(2)
    df["lag_3"]   = df["ssrd_wm2"].shift(3)
    df["lag_24"]  = df["ssrd_wm2"].shift(24)
    df["lag_48"]  = df["ssrd_wm2"].shift(48)
    df["lag_168"] = df["ssrd_wm2"].shift(168)
    return df


# ---------------------------------------------------------------------
# J. Rolling statistics (5) — MUST operate on shifted target
# ---------------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # CRITICAL: shift(1) first so the window never contains the
    # current-hour target. Rolling on raw ssrd_wm2 would leak.
    s = df["ssrd_wm2"].shift(1)
    df["rolling_mean_6"]  = s.rolling(6,  min_periods=1).mean()
    df["rolling_std_6"]   = s.rolling(6,  min_periods=1).std().fillna(0)
    df["rolling_mean_24"] = s.rolling(24, min_periods=1).mean()
    df["rolling_std_24"]  = s.rolling(24, min_periods=1).std().fillna(0)

    # Diurnal difference per thesis Appendix A.1:
    # lag-24 minus lag-25, both strictly past.
    df["diurnal_diff"] = (df["ssrd_wm2"].shift(24)
                          - df["ssrd_wm2"].shift(25))
    return df


# ---------------------------------------------------------------------
# K. Interactions (3) — use lag_1, never current ssrd_wm2
# ---------------------------------------------------------------------
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # lag_1 (previous hour) keeps this leakage-free. Using current
    # ssrd_wm2 here would push |r| to ~0.99+ and fail the audit.
    df["ssrd_x_t2m"] = df["lag_1"] * df["t2m_c"]
    df["ssrd_x_rh"]  = df["lag_1"] * df["rh"]
    df["t2m_x_rh"]   = df["t2m_c"] * df["rh"]
    return df


# ---------------------------------------------------------------------
# L. Seasonal lag splits (2)
# ---------------------------------------------------------------------
def add_seasonal_splits(df: pd.DataFrame) -> pd.DataFrame:
    dry = df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    wet = df["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df["lag_24_dry"] = df["lag_24"] * dry
    df["lag_24_wet"] = df["lag_24"] * wet
    return df


# ---------------------------------------------------------------------
# M. Regime categories (5)
# ---------------------------------------------------------------------
def add_regime_categories(df: pd.DataFrame) -> pd.DataFrame:
    tcc = df["tcc"]
    df["cloud_regime"] = np.select(
        [tcc <= 0.1, tcc <= 0.3, tcc <= 0.7],
        [0, 1, 2],
        default=3,
    ).astype(int)
    df["is_overcast"] = (tcc > 0.7).astype(int)

    t = df["t2m_c"]
    df["temp_cat"] = np.select(
        [t < 24, t < 27, t < 30], [0, 1, 2], default=3
    ).astype(int)

    rh = df["rh"]
    df["humidity_cat"] = np.select(
        [rh < 65, rh < 80, rh < 90], [0, 1, 2], default=3
    ).astype(int)

    ws = df["wind_speed"]
    df["wind_cat"] = np.select(
        [ws < 2, ws < 5, ws < 8], [0, 1, 2], default=3
    ).astype(int)
    return df


# ---------------------------------------------------------------------
# N. Season (1)
# ---------------------------------------------------------------------
def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df["season"] = df["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    return df


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Phase 6 feature functions in the exact order A → N.
    Categories B and C (solar geometry, direct radiation) are
    intentionally skipped — those columns already exist in the
    with_geometry input and must not be re-derived.
    """
    df = add_time_features(df)        # A
    df = add_core_weather(df)         # D
    df = add_thermodynamic(df)        # E
    df = add_cloud_engineered(df)     # F
    df = add_cloud_dynamics(df)       # G (needs F's cloud_opacity)
    df = add_rain_features(df)        # H
    df = add_lag_features(df)         # I
    df = add_rolling_features(df)     # J
    df = add_interactions(df)         # K (needs I's lag_1)
    df = add_seasonal_splits(df)      # L (needs I's lag_24)
    df = add_regime_categories(df)    # M
    df = add_season(df)               # N

    # Zero-fill warmup NaNs from lag/rolling columns. Tree models
    # treat 0 as "no history". Dropping rows would break the frozen
    # row-count contract (26,304 per city).
    df = df.fillna(0.0)
    return df


# ---------------------------------------------------------------------
# Locked list of new columns (A–N), in output order
# ---------------------------------------------------------------------
NEW_FEATURE_COLUMNS = [
    # A — time (10)
    "hour_local", "month", "day_of_year", "weekday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "month_sin", "month_cos",
    # D — core weather derived (1)
    "rh",
    # E — thermodynamic (3)
    "vapor_pressure_deficit", "wet_bulb_temp", "specific_humidity",
    # F — cloud engineered (4)
    "cloud_opacity", "tclw_tciw_sum", "convective_flag",
    "blh_rh_interaction",
    # G — cloud dynamics (8)
    "cloud_trend_3h", "cloud_trend_6h",
    "cloud_lag_1", "cloud_lag_3", "cloud_lag_6",
    "lcc_lag_1", "lcc_lag_3", "opacity_lag_1",
    # H — rain (5)
    "rain_binary", "precip_3h", "precip_6h", "precip_24h",
    "hours_since_rain",
    # I — lags of target (6)
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    # J — rolling stats on shifted target (5)
    "rolling_mean_6", "rolling_std_6",
    "rolling_mean_24", "rolling_std_24",
    "diurnal_diff",
    # K — interactions (3)
    "ssrd_x_t2m", "ssrd_x_rh", "t2m_x_rh",
    # L — seasonal lag splits (2)
    "lag_24_dry", "lag_24_wet",
    # M — regime categories (5)
    "cloud_regime", "is_overcast", "temp_cat",
    "humidity_cat", "wind_cat",
    # N — season (1)
    "season",
]

assert len(NEW_FEATURE_COLUMNS) == 53, (
    f"NEW_FEATURE_COLUMNS has {len(NEW_FEATURE_COLUMNS)} entries; "
    f"brief requires 53."
)

FORBIDDEN_COLUMNS = [
    "direct_fraction",
    "opacity_x_direct",
    "clearness_index",
    "cloud_modification_factor",
    "cos_zenith_x_opacity",
    "kt_lag_1", "kt_lag_24", "kt_cat", "kt_x_tcc",
]


# ---------------------------------------------------------------------
# Leakage audit helper
# ---------------------------------------------------------------------
def leakage_audit(df: pd.DataFrame,
                  train_end: pd.Timestamp,
                  test_start: pd.Timestamp,
                  top_n: int = 10) -> pd.DataFrame:
    """
    Pearson |r| of every numeric feature vs ssrd_wm2, on the
    train portion only (time < train_end). Also computes |r| on
    the test portion (time >= test_start) as a stability check.

    Returns a DataFrame with columns: feature, r_train, r_test.
    """
    tr = df[df["time"] < train_end]
    te = df[df["time"] >= test_start]

    y_tr = tr["ssrd_wm2"].values
    y_te = te["ssrd_wm2"].values

    rows = []
    for col in df.columns:
        if col in ("time", "local_time", "ssrd_wm2"):
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        x_tr = tr[col].values.astype(float)
        x_te = te[col].values.astype(float)
        if np.std(x_tr) == 0 or np.std(x_te) == 0:
            continue
        r_tr = np.corrcoef(x_tr, y_tr)[0, 1]
        r_te = np.corrcoef(x_te, y_te)[0, 1]
        rows.append((col, r_tr, r_te))

    audit = pd.DataFrame(rows, columns=["feature", "r_train", "r_test"])
    audit["abs_r_train"] = audit["r_train"].abs()
    audit = (audit.sort_values("abs_r_train", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))
    audit.insert(0, "rank", np.arange(1, len(audit) + 1))
    return audit