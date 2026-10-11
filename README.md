# ERA5-Philippines Solar Forecasting

## Overview

This project tests whether free public weather data alone can forecast
surface solar radiation well enough to meet the WESM benchmark of
MAPE < 18% during daytime hours. Using ERA5 reanalysis for four
Philippine cities — Muntinlupa, Baguio, Cebu, and Davao — we build a
leakage-audited pipeline that engineers solar-geometry and lag features,
trains persistence and gradient-boosted baselines, runs an ablation over
feature families, and evaluates every model against the same
forecast-time-available data constraint. All results are produced from
data that would be available at forecast time; no future information
enters any model.

## Repository layout

```
era5-philippines/
  config/       Configuration constants and paths
  data/         Raw, extracted, cleaned, feature, and split datasets (frozen)
  docs/         Protocol, decisions log, feature notes, fingerprints
  results/      Metrics, tables, and figures for Paper 1
  scripts/      Numbered pipeline stages 01-10 plus run_all.py
  src/          Shared library code (I/O, features, metrics)
  tests/        Unit tests
```

## Requirements

- Python 3.11 or newer
- Libraries listed in `requirements.txt`
- No GPU required; all stages run on CPU

## Data

Raw ERA5 reanalysis data is downloaded from the Copernicus Climate Data
Store (CDS). A free CDS account and API key are required. Configure your
key at `~/.cdsapirc` before running `scripts/01_download_era5.py`. The
raw data is ##not## included in this repository; the download step is
the only stage that requires network access.

## Reproducing the pipeline

Two paths are supported.

##(a) Full run from scratch## — requires CDS credentials and network access:

```bash
python scripts/run_all.py
```

##(b) From frozen intermediates## — if `data/` is already populated:

```bash
python scripts/06_split.py
python scripts/07a_persistence.py
python scripts/07b_xgb_baseline.py
python scripts/07c_xgb_tuned.py
python scripts/08_ablation.py
python scripts/09_evaluate.py
python scripts/10_generate_figures.py
```

Stages 01-05 populate `data/` from CDS. Stages 06-10 consume frozen
intermediates and produce everything under `results/paper1/`.

## What each script does

|Script|Description|Key outputs|
|---|---|---|
|`01_download_era5.py`|Downloads ERA5 reanalysis from CDS|`data/raw/`|
|`02_extract_to_csv.py`|Extracts grid cells to per-city CSVs|`data/extracted/`|
|`03_clean_and_convert.py`|Cleans, converts units, handles gaps|`data/cleaned/`|
|`04_solar_geometry.py`|Computes solar position features|`data/with_geometry/`|
|`05_feature_engineering.py`|Builds lag, rolling, interaction features|`data/features/`|
|`06_split.py`|Time-ordered train/val/test split|`data/splits/`|
|`07a_persistence.py`|Persistence baseline|`results/paper1/metrics/`|
|`07b_xgb_baseline.py`|XGBoost baseline|`results/paper1/metrics/`|
|`07c_xgb_tuned.py`|XGBoost with grid search|`results/paper1/metrics/`, `results/paper1/tables/xgb_tuned_hyperparams.csv`|
|`08_ablation.py`|Feature-family ablation|`results/paper1/metrics/`, `results/paper1/tables/`|
|`09_evaluate.py`|Aggregates metrics, writes paper tables|`results/paper1/tables/`|
|`10_generate_figures.py`|Generates all paper figures|`results/paper1/figures/`|

## Outputs

- `results/paper1/metrics/` — per-model MAPE tables
- `results/paper1/tables/` — paper-ready tables
- `results/paper1/figures/` — paper figures

## Frozen stages

Intermediate data files are frozen with recorded SHA1 fingerprints in
`docs/frozen_fingerprints.md`. Every model script verifies these
fingerprints at the start and end of its run, and fails if they differ.
This guarantees that results are reproducible from a known data state.

## Known limitations

- The Phase 11 hyperparameter CSV was widened in Phase 18 to include
  all `FIXED_PARAMS` keys. See `docs/reproducibility.md`.
- Figures 7 and 8 show persistence and xgb_tuned only, because per-hour
  predictions for other models are not persisted.
- Raw data is not bundled; users must download from CDS.

## License

MIT. See [LICENSE](LICENSE) for details.
