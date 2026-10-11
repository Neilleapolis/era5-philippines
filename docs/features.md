# Feature Set — Paper 1

## Overview
- Total: 80 columns per city
- Breakdown: 27 inherited from Phase 5 (`data/with_geometry/{city}.csv`),
  53 new in Phase 6 (`data/features/{city}.csv`)
- Target variable: `ssrd_wm2` (surface solar radiation downwards, W/m²)
- All features verified leakage-free (top |r| = 0.9752)
- Row count: 26,304 per city (105,216 total across 4 cities)
- Cities: muntinlupa, baguio, cebu, davao

---

## Columns by category

### Inherited from Phase 5 (27) — `with_geometry` order, LOCKED

| Column | Source | Notes |
|--------|--------|-------|
| time | Phase 4 | UTC timestamp |
| local_time | Phase 4 | Asia/Manila |
| ssrd_wm2 | ERA5 | **TARGET** |
| fdir_wm2 | ERA5 | Direct solar radiation, W/m² |
| tp_mm | ERA5 | Total precipitation, mm |
| t2m_c | ERA5 | 2 m temperature, °C |
| d2m_c | ERA5 | 2 m dewpoint, °C |
| u10 | ERA5 | 10 m u-wind |
| v10 | ERA5 | 10 m v-wind |
| wind_speed | ERA5 | sqrt(u10² + v10²) |
| wind_direction | ERA5 | Meteorological convention: atan2(−u10, −v10) × 180/π, range [0, 360) |
| tcc | ERA5 | Total cloud cover, 0–1 |
| lcc | ERA5 | Low cloud cover, 0–1 |
| mcc | ERA5 | Medium cloud cover, 0–1 |
| hcc | ERA5 | High cloud cover, 0–1 |
| tclw | ERA5 | Total column liquid water |
| tciw | ERA5 | Total column ice water |
| tcwv | ERA5 | Total column water vapour |
| blh | ERA5 | Boundary layer height, m |
| solar_zenith_angle | Phase 5 | degrees |
| solar_elevation | Phase 5 | degrees |
| cos_zenith | Phase 5 | cos(radians(solar_zenith_angle)), range [−1, 1], unclipped |
| daylight_flag | Phase 5 | 1 if elevation > 0 else 0 |
| extraterrestrial_radiation | Phase 5 | DNI_extra × clip(cos_zenith, 0, None) |
| i0_horizontal | Phase 5 | Numerically identical to extraterrestrial_radiation (Decision 5.4) |
| clear_sky_ghi | Phase 5 | Ineichen clear-sky GHI, W/m² |
| clear_sky_lag_24 | Phase 5 | shift(clear_sky_ghi, 24); first 24 rows zero-filled (Decision 5.5) |

### A. Time features (10)

| Column | Formula | Notes |
|--------|---------|-------|
| hour_local | `local_time.dt.hour` | 0–23 |
| month | `local_time.dt.month` | 1–12 |
| day_of_year | `local_time.dt.dayofyear` | 1–365/366 |
| weekday | `local_time.dt.weekday + 1` | Mon=1 .. Sun=7 |
| hour_sin | `sin(2π × hour_local / 24)` | cyclical encoding |
| hour_cos | `cos(2π × hour_local / 24)` | cyclical encoding |
| doy_sin | `sin(2π × day_of_year / 365)` | cyclical encoding |
| doy_cos | `cos(2π × day_of_year / 365)` | cyclical encoding |
| month_sin | `sin(2π × month / 12)` | cyclical encoding |
| month_cos | `cos(2π × month / 12)` | cyclical encoding |

### B. Solar geometry (0)
N/A — already in `with_geometry` (Phase 5). Not re-derived.

### C. Direct radiation (0)
N/A — already in `with_geometry` (Phase 5). Not re-derived.

### D. Core weather derived (1)

| Column | Formula | Notes |
|--------|---------|-------|
| rh | `100 × (exp((17.625 × d2m_c) / (243.04 + d2m_c)) / exp((17.625 × t2m_c) / (243.04 + t2m_c)))` | Magnus-Tetens, clipped to [0, 100] |

### E. Thermodynamic (3)

| Column | Formula | Notes |
|--------|---------|-------|
| vapor_pressure_deficit | `es(t2m_c) − es(d2m_c)` where `es(T) = 6.112 × exp(17.67 × T / (T + 243.5))` | hPa; non-negative by construction |
| wet_bulb_temp | `T × atan(0.151977 × sqrt(RH + 8.313659)) + atan(T + RH) − atan(RH − 1.676331) + 0.00391838 × RH^1.5 × atan(0.023101 × RH) − 4.686035` | °C; Stull (2011) |
| specific_humidity | `622 × e / (p − 0.378 × e)` where `e = es(d2m_c)` and `p = 1013.25 hPa` | g/kg; see approximation notes |

### F. Cloud engineered (4)

| Column | Formula | Notes |
|--------|---------|-------|
| cloud_opacity | `0.45 × lcc + 0.30 × mcc + 0.15 × hcc + 0.10 × tcc` | weighted cloud metric |
| tclw_tciw_sum | `tclw + tciw` | total column condensate |
| convective_flag | `1 if (tclw + tciw > 0.05) AND (tp_mm > 0.1) else 0` | binary |
| blh_rh_interaction | `(blh / 3000) × (rh / 100)` | boundary-layer × humidity |

### G. Cloud dynamics (8) — all shifts on cloud columns, never the target

| Column | Formula | Notes |
|--------|---------|-------|
| cloud_trend_3h | `tcc − shift(tcc, 3)` | 3-hour change |
| cloud_trend_6h | `tcc − shift(tcc, 6)` | 6-hour change |
| cloud_lag_1 | `shift(tcc, 1)` | past only |
| cloud_lag_3 | `shift(tcc, 3)` | past only |
| cloud_lag_6 | `shift(tcc, 6)` | past only |
| lcc_lag_1 | `shift(lcc, 1)` | past only |
| lcc_lag_3 | `shift(lcc, 3)` | past only |
| opacity_lag_1 | `shift(cloud_opacity, 1)` | depends on category F |

### H. Rain features (5)

| Column | Formula | Notes |
|--------|---------|-------|
| rain_binary | `1 if tp_mm >= 0.1 else 0` | WMO threshold |
| precip_3h | `Σ tp_mm over [t−3, t−1]` | shifted; past only |
| precip_6h | `Σ tp_mm over [t−6, t−1]` | shifted; past only |
| precip_24h | `Σ tp_mm over [t−24, t−1]` | shifted; past only |
| hours_since_rain | hours since last `rain_binary == 1`, computed from `rain_binary.shift(1)` | capped at 72 |

### I. Lag features (6) — lags of the target, strictly past

| Column | Formula | Notes |
|--------|---------|-------|
| lag_1 | `shift(ssrd_wm2, 1)` | first 1 row zero-filled |
| lag_2 | `shift(ssrd_wm2, 2)` | first 2 rows zero-filled |
| lag_3 | `shift(ssrd_wm2, 3)` | first 3 rows zero-filled |
| lag_24 | `shift(ssrd_wm2, 24)` | first 24 rows zero-filled |
| lag_48 | `shift(ssrd_wm2, 48)` | first 48 rows zero-filled |
| lag_168 | `shift(ssrd_wm2, 168)` | first 168 rows zero-filled |

### J. Rolling statistics (5) — MUST operate on shifted target

| Column | Formula | Notes |
|--------|---------|-------|
| rolling_mean_6 | `ssrd_wm2.shift(1).rolling(6, min_periods=1).mean()` | leakage-safe |
| rolling_std_6 | `ssrd_wm2.shift(1).rolling(6, min_periods=1).std().fillna(0)` | leakage-safe |
| rolling_mean_24 | `ssrd_wm2.shift(1).rolling(24, min_periods=1).mean()` | leakage-safe |
| rolling_std_24 | `ssrd_wm2.shift(1).rolling(24, min_periods=1).std().fillna(0)` | leakage-safe |
| diurnal_diff | `shift(ssrd_wm2, 24) − shift(ssrd_wm2, 25)` | thesis Appendix A.1 form |

### K. Interaction features (3) — use lag_1, never current ssrd_wm2

| Column | Formula | Notes |
|--------|---------|-------|
| ssrd_x_t2m | `lag_1 × t2m_c` | uses previous-hour target |
| ssrd_x_rh | `lag_1 × rh` | uses previous-hour target |
| t2m_x_rh | `t2m_c × rh` | weather × weather |

### L. Seasonal lag splits (2)

| Column | Formula | Notes |
|--------|---------|-------|
| lag_24_dry | `lag_24 × (1 if month in [11,12,1,2,3,4] else 0)` | dry season (Nov–Apr) |
| lag_24_wet | `lag_24 × (1 if month in [5,6,7,8,9,10] else 0)` | wet season (May–Oct) |

### M. Regime categories (5)

| Column | Formula | Notes |
|--------|---------|-------|
| cloud_regime | `0 if tcc ≤ 0.1; 1 if tcc ≤ 0.3; 2 if tcc ≤ 0.7; 3 otherwise` | 0–3 |
| is_overcast | `1 if tcc > 0.7 else 0` | binary |
| temp_cat | `0 if t2m_c < 24; 1 if < 27; 2 if < 30; 3 otherwise` | 0–3 |
| humidity_cat | `0 if rh < 65; 1 if < 80; 2 if < 90; 3 otherwise` | 0–3 |
| wind_cat | `0 if wind_speed < 2; 1 if < 5; 2 if < 8; 3 otherwise` | 0–3 |

### N. Season (1)

| Column | Formula | Notes |
|--------|---------|-------|
| season | `1 if month in [5,6,7,8,9,10] else 0` | wet=1, dry=0 |

---

## Leakage audit
- **Method**: Pearson |r| vs `ssrd_wm2` on training rows only
  (`time < 2024-07-01`), aggregated as mean |r| across the 4 cities.
  Test-portion |r| (`time >= 2025-01-01`) reported as a stability column.
- **Result**: highest |r| = **0.9752** (`fdir_wm2`)
- **Threshold**: 0.99; audit **passes**
- **Top-10 table** (from Phase 6 run output):

```
  rank  feature                            |r| train  |r| test (mean)
     1  fdir_wm2                              0.9752          0.9626
     2  lag_24                                0.9472          0.9205
     3  clear_sky_ghi                         0.9446          0.9256
     4  clear_sky_lag_24                      0.9438          0.9256
     5  i0_horizontal                         0.9412          0.9216
     6  extraterrestrial_radiation            0.9412          0.9216
     7  lag_48                                0.9343          0.8964
     8  lag_1                                 0.9334          0.9291
     9  ssrd_x_t2m                            0.9312          0.9277
    10  ssrd_x_rh                             0.9181          0.9189
```

---

## Excluded features (leakage)
The following were present in the original thesis Appendix A.1 but are
**excluded** from Paper 1 because they use the current-hour target:

- `direct_fraction` (= `fdir_wm2 / (ssrd_wm2 + 1)`) — uses `ssrd_wm2`
- `opacity_x_direct` (= `cloud_opacity × direct_fraction`) — uses `direct_fraction`
- `clearness_index` (= `ssrd_wm2 / (i0_horizontal + ε)`) — uses `ssrd_wm2`
- `cloud_modification_factor` (= `ssrd_wm2 / (clear_sky_ghi + ε)`) — uses `ssrd_wm2`
- `cos_zenith_x_opacity` (= `cos_zenith × cloud_opacity`) — safe in principle
  but removed for consistency with thesis Config C
- Any derivative of the above (`kt_lag_1`, `kt_lag_24`, `kt_cat`, `kt_x_tcc`)

---

## Approximation notes
- `specific_humidity` uses `p = 1013.25 hPa` (standard sea-level pressure),
  as ERA5's surface-pressure field is not in our 15-variable download set.
- `extraterrestrial_radiation` and `i0_horizontal` are numerically identical
  by construction (both `= DNI_extra × clip(cos_zenith, 0, None)`); both
  slots are written per the Phase 5 brief (Decision 5.4).
- Lag and rolling-stats warmup NaNs are zero-filled rather than dropped,
  to preserve the 26,304-row frozen shape contract. Affects only the
  leading rows (e.g. first 168 rows of `lag_168`).
- `wind_direction` uses the meteorological convention (direction from which
  the wind blows, measured clockwise from north): `atan2(−u10, −v10) × 180/π`,
  normalised to [0, 360). This differs from a mathematical vector angle
  (`atan2(u10, v10)`), which would measure the direction the wind blows
  towards. The meteorological convention is standard for weather data.