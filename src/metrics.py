"""
Evaluation metrics for Paper 1 (era5-philippines).

All four metrics (MAE, RMSE, R^2, MAPE_daytime) are computed on
DAYTIME HOURS ONLY, defined as rows where y_true > DAYTIME_THRESHOLD_WM2.

This module performs NO data loading. It receives y_true and y_pred
arrays from the caller. This keeps the reporting layer free of any
pipeline coupling and makes leakage impossible by construction.

Radiation units: W/m^2. MAE and RMSE are reported in W/m^2.
R^2 is dimensionless. MAPE is reported in percent.

MAPE formula (frozen decision for Paper 1):
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
This is the PER-ROW division variant, NOT WAPE. The two differ
substantially near sunrise/sunset, and the per-row version is the
one used in the thesis.
"""

from __future__ import annotations

import numpy as np

try:
    from config.paper1_config import DAYTIME_THRESHOLD_WM2
except ImportError:  # pragma: no cover - fallback when config not importable
    DAYTIME_THRESHOLD_WM2 = 10.0


_DEFAULT_THRESHOLD = DAYTIME_THRESHOLD_WM2


def _as_1d_float_arrays(y_true, y_pred):
    """Coerce inputs to 1D float numpy arrays and validate length match."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have equal length; "
            f"got {yt.shape[0]} and {yp.shape[0]}"
        )
    return yt, yp


def _daytime_mask(y_true, threshold):
    """Boolean mask of rows that count as daytime (y_true > threshold)."""
    return y_true > threshold


def mae(y_true, y_pred, threshold=_DEFAULT_THRESHOLD):
    """Mean Absolute Error on daytime rows only.

    Filters to rows where y_true > threshold, then returns
    mean(|y_true - y_pred|) in W/m^2.

    Returns np.nan if no daytime rows pass the threshold.
    Does not modify inputs.
    """
    yt, yp = _as_1d_float_arrays(y_true, y_pred)
    mask = _daytime_mask(yt, threshold)
    if not np.any(mask):
        return float(np.nan)
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def rmse(y_true, y_pred, threshold=_DEFAULT_THRESHOLD):
    """Root Mean Squared Error on daytime rows only.

    Filters to rows where y_true > threshold, then returns
    sqrt(mean((y_true - y_pred)^2)) in W/m^2.

    Returns np.nan if no daytime rows pass the threshold.
    Does not modify inputs.
    """
    yt, yp = _as_1d_float_arrays(y_true, y_pred)
    mask = _daytime_mask(yt, threshold)
    if not np.any(mask):
        return float(np.nan)
    diff = yt[mask] - yp[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def r2(y_true, y_pred, threshold=_DEFAULT_THRESHOLD):
    """Coefficient of determination (R^2) on daytime rows only.

    Filters to rows where y_true > threshold, then returns
        1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)^2) and
          SS_tot = sum((y_true - mean(y_true))^2).

    Returns np.nan if no daytime rows pass the threshold, or if
    SS_tot == 0 (undefined R^2).
    Does not modify inputs.
    """
    yt, yp = _as_1d_float_arrays(y_true, y_pred)
    mask = _daytime_mask(yt, threshold)
    if not np.any(mask):
        return float(np.nan)
    yt_m = yt[mask]
    yp_m = yp[mask]
    ss_res = float(np.sum((yt_m - yp_m) ** 2))
    ss_tot = float(np.sum((yt_m - np.mean(yt_m)) ** 2))
    if ss_tot == 0.0:
        return float(np.nan)
    return 1.0 - ss_res / ss_tot


def mape_daytime(y_true, y_pred, threshold=_DEFAULT_THRESHOLD):
    """Mean Absolute Percentage Error on daytime rows only (percent).

    Frozen formula for Paper 1 -- PER-ROW division:
        MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    This is NOT WAPE (sum|err| / sum|y_true| * 100).

    Filters to rows where y_true > threshold. Rows at or below
    threshold are excluded before the per-row ratio is taken, which
    is safe because the threshold (10 W/m^2) is strictly positive.

    Returns np.nan if no daytime rows pass the threshold.
    Does not modify inputs.
    """
    yt, yp = _as_1d_float_arrays(y_true, y_pred)
    mask = _daytime_mask(yt, threshold)
    if not np.any(mask):
        return float(np.nan)
    yt_m = yt[mask]
    yp_m = yp[mask]
    return float(np.mean(np.abs(yt_m - yp_m) / np.abs(yt_m)) * 100.0)


def all_metrics(y_true, y_pred, threshold=_DEFAULT_THRESHOLD):
    """Compute all four daytime metrics plus the daytime row count.

    Returns a dict with keys:
        'mae', 'rmse', 'r2', 'mape', 'n_daytime'

    If no daytime rows pass the threshold, all four metric values
    are np.nan and n_daytime is 0.
    Does not modify inputs.
    """
    yt, yp = _as_1d_float_arrays(y_true, y_pred)
    mask = _daytime_mask(yt, threshold)
    n_daytime = int(np.count_nonzero(mask))
    return {
        "mae": mae(yt, yp, threshold=threshold),
        "rmse": rmse(yt, yp, threshold=threshold),
        "r2": r2(yt, yp, threshold=threshold),
        "mape": mape_daytime(yt, yp, threshold=threshold),
        "n_daytime": n_daytime,
    }


__all__ = [
    "mae",
    "rmse",
    "r2",
    "mape_daytime",
    "all_metrics",
    "DAYTIME_THRESHOLD_WM2",
]
