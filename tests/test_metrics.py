"""
Focused test suite for src.metrics (Phase 8).

Runnable via either:
    python -m pytest tests/test_metrics.py
    python tests/test_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.metrics import (  # noqa: E402
    mae,
    rmse,
    r2,
    mape_daytime,
    all_metrics,
    DAYTIME_THRESHOLD_WM2,
)
from config.paper1_config import DAYTIME_THRESHOLD_WM2 as CFG_THRESHOLD  # noqa: E402

try:
    from sklearn.metrics import (
        mean_absolute_error as sk_mae,
        mean_squared_error as sk_rmse_sq,
        r2_score as sk_r2,
    )
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


TOL = 1e-9


def test_module_exposes_frozen_threshold():
    assert DAYTIME_THRESHOLD_WM2 == CFG_THRESHOLD
    assert DAYTIME_THRESHOLD_WM2 == 10


def test_mape_hand_computed_toy_example():
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = np.array([110.0, 180.0, 420.0])
    expected = ((10.0 / 100.0) + (20.0 / 200.0) + (20.0 / 400.0)) / 3.0 * 100.0
    got = mape_daytime(y_true, y_pred, threshold=10.0)
    assert abs(got - expected) < TOL, f"mape={got} expected={expected}"


def test_mape_simple_ten_percent_case():
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = np.array([110.0, 220.0, 440.0])
    got = mape_daytime(y_true, y_pred, threshold=10.0)
    assert abs(got - 10.0) < TOL, f"mape={got} expected=10.0"


def test_threshold_excludes_low_rows():
    y_true = np.array([5.0, 100.0, 200.0])
    y_pred = np.array([50.0, 90.0, 190.0])
    assert abs(mae(y_true, y_pred, threshold=10.0) - 10.0) < TOL


def test_threshold_boundary_is_strictly_greater():
    y_true = np.array([10.0, 10.0, 10.0, 11.0])
    y_pred = np.array([0.0, 0.0, 0.0, 1.0])
    assert abs(mae(y_true, y_pred, threshold=10.0) - 10.0) < TOL
    assert all_metrics(y_true, y_pred, threshold=10.0)["n_daytime"] == 1


def test_mae_rmse_r2_match_sklearn_on_daytime_subset():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0.0, 900.0, size=500)
    y_pred = y_true + rng.normal(0.0, 30.0, size=500)
    thr = 10.0
    mask = y_true > thr
    yt = y_true[mask]
    yp = y_pred[mask]

    got_mae = mae(y_true, y_pred, threshold=thr)
    got_rmse = rmse(y_true, y_pred, threshold=thr)
    got_r2 = r2(y_true, y_pred, threshold=thr)

    if _HAS_SKLEARN:
        exp_mae = sk_mae(yt, yp)
        exp_rmse = float(np.sqrt(sk_rmse_sq(yt, yp)))
        exp_r2 = sk_r2(yt, yp)
    else:  # pragma: no cover
        exp_mae = float(np.mean(np.abs(yt - yp)))
        exp_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        exp_r2 = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)

    assert abs(got_mae - exp_mae) < TOL
    assert abs(got_rmse - exp_rmse) < TOL
    assert abs(got_r2 - exp_r2) < TOL


def test_all_metrics_keys_and_n_daytime():
    y_true = np.array([0.0, 5.0, 10.0, 50.0, 100.0, 200.0])
    y_pred = np.array([0.0, 5.0, 10.0, 45.0, 90.0, 190.0])
    out = all_metrics(y_true, y_pred, threshold=10.0)
    assert set(out.keys()) == {"mae", "rmse", "r2", "mape", "n_daytime"}
    assert out["n_daytime"] == 3
    assert isinstance(out["n_daytime"], int)


def test_empty_daytime_returns_nan():
    y_true = np.array([0.0, 1.0, 5.0, 9.999])
    y_pred = np.array([0.0, 2.0, 6.0, 10.0])
    assert np.isnan(mae(y_true, y_pred, threshold=10.0))
    assert np.isnan(rmse(y_true, y_pred, threshold=10.0))
    assert np.isnan(r2(y_true, y_pred, threshold=10.0))
    assert np.isnan(mape_daytime(y_true, y_pred, threshold=10.0))
    out = all_metrics(y_true, y_pred, threshold=10.0)
    assert out["n_daytime"] == 0
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["r2"])
    assert np.isnan(out["mape"])


def test_inputs_not_mutated():
    y_true = np.array([5.0, 100.0, 200.0, 0.0, 50.0])
    y_pred = np.array([50.0, 90.0, 190.0, 0.0, 55.0])
    y_true_before = y_true.copy()
    y_pred_before = y_pred.copy()
    _ = mae(y_true, y_pred, threshold=10.0)
    _ = rmse(y_true, y_pred, threshold=10.0)
    _ = r2(y_true, y_pred, threshold=10.0)
    _ = mape_daytime(y_true, y_pred, threshold=10.0)
    _ = all_metrics(y_true, y_pred, threshold=10.0)
    assert np.array_equal(y_true, y_true_before)
    assert np.array_equal(y_pred, y_pred_before)


def test_length_mismatch_raises():
    try:
        mae(np.array([1.0, 2.0]), np.array([1.0]), threshold=10.0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError on length mismatch")


def test_mape_is_per_row_not_wape():
    y_true = np.array([1000.0, 20.0])
    y_pred = np.array([900.0, 10.0])
    got = mape_daytime(y_true, y_pred, threshold=10.0)
    assert abs(got - 30.0) < TOL, f"mape={got} expected=30.0 (per-row)"
    assert abs(got - 10.784313725490197) > 1.0


def _run_all():
    tests = [
        ("test_module_exposes_frozen_threshold",
         test_module_exposes_frozen_threshold),
        ("test_mape_hand_computed_toy_example",
         test_mape_hand_computed_toy_example),
        ("test_mape_simple_ten_percent_case",
         test_mape_simple_ten_percent_case),
        ("test_threshold_excludes_low_rows",
         test_threshold_excludes_low_rows),
        ("test_threshold_boundary_is_strictly_greater",
         test_threshold_boundary_is_strictly_greater),
        ("test_mae_rmse_r2_match_sklearn_on_daytime_subset",
         test_mae_rmse_r2_match_sklearn_on_daytime_subset),
        ("test_all_metrics_keys_and_n_daytime",
         test_all_metrics_keys_and_n_daytime),
        ("test_empty_daytime_returns_nan",
         test_empty_daytime_returns_nan),
        ("test_inputs_not_mutated",
         test_inputs_not_mutated),
        ("test_length_mismatch_raises",
         test_length_mismatch_raises),
        ("test_mape_is_per_row_not_wape",
         test_mape_is_per_row_not_wape),
    ]
    passed = 0
    failed = 0
    failures = []
    print("=" * 68)
    print("Phase 8 metrics test suite")
    print(f"sklearn available: {_HAS_SKLEARN}")
    print("=" * 68)
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
            failures.append((name, e))
    print("-" * 68)
    print(f"TOTAL: {passed + failed}   PASSED: {passed}   FAILED: {failed}")
    print("=" * 68)
    if failures:
        print("Failure details:")
        for name, e in failures:
            print(f"  - {name}: {e}")
    return passed, failed


if __name__ == "__main__":
    passed, failed = _run_all()
    sys.exit(0 if failed == 0 else 1)
