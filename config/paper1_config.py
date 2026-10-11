# config/paper1_config.py
# =============================================================================
# PAPER 1 CONFIGURATION - specific to the solar forecasting reimplementation
# =============================================================================

# -----------------------------------------------------------------------------
# Chronological split - FROZEN, do not change after Stage 7
# -----------------------------------------------------------------------------
TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-06-30"
VAL_START   = "2024-07-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-31"

# -----------------------------------------------------------------------------
# Metric constants
# -----------------------------------------------------------------------------
DAYTIME_THRESHOLD_WM2 = 10  # MAPE only computed where actual > this

# -----------------------------------------------------------------------------
# Ablation configs
# -----------------------------------------------------------------------------
CONFIG_A_REMOVE = []  # Full feature set
CONFIG_B_REMOVE = ["fdir_wm2"]
CONFIG_C_REMOVE = [
    "fdir_wm2",
    "clear_sky_ghi",
    "i0_horizontal",
    "clear_sky_lag_24",
]

# -----------------------------------------------------------------------------
# Model registry - for sanity checks and reporting
# -----------------------------------------------------------------------------
MODELS = {
    "persistence":        {"expected_mape": (36, 45)},
    "xgb_baseline":       {"expected_mape": (15, 20)},
    "xgb_tuned":          {"expected_mape": (4, 7)},
    "seasonal_ensemble":  {"expected_mape": (6, 7)},
    "hybrid_lstm":        {"expected_mape": (4, 7)},
    "lgbm_attention":     {"expected_mape": (4, 5)},
}

# -----------------------------------------------------------------------------
# WESM benchmark (for evaluation tables)
# -----------------------------------------------------------------------------
WESM_BENCHMARK_MAPE = 18.0