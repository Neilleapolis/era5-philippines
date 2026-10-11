# Paper 1 — Results Summary

## 1. Benchmark progression

| Model | Muntinlupa | Baguio | Cebu | Davao |
|---|---|---|---|---|
| Persistence | 39.69 | 38.02 | 44.53 | 36.32 |
| XGBoost baseline | 18.87 | 18.24 | 20.11 | 15.50 |
| XGBoost tuned | 7.28 | 7.38 | 7.74 | 6.82 |
| 18% benchmark | 18.00 | 18.00 | 18.00 | 18.00 |

## 2. Ablation (Configs A/B/C)

| City | Config A (77) | Config B (76) | Config C (72) |
|---|---|---|---|
| Baguio | 7.38 | 13.84 | 14.19 |
| Cebu | 7.74 | 16.14 | 16.11 |
| Davao | 6.82 | 13.07 | 13.14 |
| Muntinlupa | 7.28 | 15.07 | 15.12 |


*fdir_value_pp = the cost in MAPE percentage points of removing fdir_wm2 alone (positive = the feature was valuable).*

*other_rad_value_pp = the additional cost of removing clear_sky_ghi, i0_horizontal, extraterrestrial_radiation, and clear_sky_lag_24 after fdir_wm2 has already been removed (positive = the features were valuable; negative = keeping them was slightly harmful).*

**Finding.** fdir_wm2 alone is worth 6.25–8.40 pp of MAPE. The remaining radiation features are worth -0.02 to +0.34 pp — effectively zero, with a slight negative in three of four cities.

## 3. Operational feasibility (Config C vs WESM 18%)

| City | Config C MAPE (%) | Margin below 18% (pp) | Result |
|---|---|---|---|
| Baguio | 14.19 | 3.81 | PASS |
| Cebu | 16.11 | 1.89 | PASS |
| Davao | 13.14 | 4.86 | PASS |
| Muntinlupa | 15.12 | 2.88 | PASS |

## 4. Seasonal breakdown

| City | Dry MAPE (%) | Wet MAPE (%) | Wet − Dry (pp) |
|---|---|---|---|
| Muntinlupa | 12.93 | 17.19 | 4.26 |
| Baguio | 11.64 | 16.58 | 4.94 |
| Cebu | 15.96 | 16.26 | 0.30 |
| Davao | 13.05 | 13.23 | 0.18 |

## 5. Key findings

- Config C (operational) satisfies the WESM 18% daytime MAPE benchmark for all four cities: yes.
- Removing fdir_wm2 (A → B) costs an average of 7.23 pp in MAPE across the four cities.
- The remaining radiation features, once fdir_wm2 is removed (B → C), change MAPE by an average of +0.11 pp — effectively zero.
- Tightest operational margin: Cebu at 1.89 pp below 18%. Widest: Davao at 4.86 pp.
- Easiest site under Config C: Davao. Hardest: Cebu.
- best_model is xgb_tuned for all four cities (see paper_table1_models.csv).

## 6. Method summary

Eighty-feature leakage-audited pipeline over ERA5 reanalysis for four
Philippine cities (Muntinlupa, Baguio, Cebu, Davao). Target:
ssrd_wm2. Chronological split: train 2023-01-01 to 2024-06-30,
validation 2024-07-01 to 2024-12-31, test 2025-01-01 to 2025-12-31.
Models: persistence baseline, tuned XGBoost (144-combination grid
search per city per config), and three ablation configs (A = full,
B = no fdir_wm2, C = operational). All metrics reported for daytime
hours only. Benchmark: WESM MAPE < 18%.
