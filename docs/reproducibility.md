# Reproducibility

## Pipeline overview

Data flow:

```
data/raw → data/extracted → data/cleaned →
  data/with_geometry → data/features →
  data/splits → results/paper1/
```

## Frozen stages

Each stage in the pipeline is frozen at the end of its phase. The
SHA1 head-and-tail fingerprint of every frozen file is recorded in
`docs/frozen_fingerprints.md`.

|Folder|Frozen at end of phase|Fingerprint location|
|---|---|---|
|`data/raw/`|Phase 1|`docs/frozen_fingerprints.md`|
|`data/extracted/`|Phase 2|`docs/frozen_fingerprints.md`|
|`data/cleaned/`|Phase 3|`docs/frozen_fingerprints.md`|
|`data/with_geometry/`|Phase 4|`docs/frozen_fingerprints.md`|
|`data/features/`|Phase 5|`docs/frozen_fingerprints.md`|
|`data/splits/`|Phase 6|`docs/frozen_fingerprints.md`|

## Verifying frozen inputs

Every model script snapshots the fingerprints of its input files at
the start of a run, and again at the end. If any fingerprint differs,
the script fails. The helper `sha1_head_tail` in `src/io.py` records
a head-and-tail SHA1 digest so large CSVs can be verified without
reading them in full.

## Known gaps and their resolutions

1. The Phase 11 hyperparameter CSV `xgb_tuned_hyperparams.csv` was
   originally partial: it stored only the four grid-searched
   hyperparameters plus metadata, omitting `objective`, `eval_metric`,
   `colsample_bytree`, `random_state`, and `tree_method`, all of which
   were constants in `FIXED_PARAMS` and did enter the final fit.
   Phase 18 widened the CSV to 13 columns by appending the five
   FIXED_PARAMS keys (objective, eval_metric, colsample_bytree,
   random_state, tree_method). The CSV is now sufficient on its
   own to reconstruct the final Phase 11 model.

2. Phase 17 figures 7 and 8 show a subset of models because per-hour
   predictions for `xgb_baseline` and `ablation_C` are not persisted.
   Accepted as-is; persisting them would require re-running those
   stages, which Phase 18 does not do.

3. The widening of gap 1 was performed once, in Phase 18, without
   re-running the grid search. The Phase 11 model parameters in
   `FIXED_PARAMS` were carried directly into the CSV; no model
   was retrained, and no model output was recomputed.

## Pipeline runtime

Approximate wall-clock per stage on a CPU-only workstation:

|Stage|Approximate wall-clock|
|---|---|
|01 (download)|hours (CDS queue dependent)|
|02-06 (pipeline)|under 5 minutes combined|
|07a (persistence)|under 10 seconds|
|07b (xgb baseline)|about 2 minutes|
|07c (xgb tuned)|60-90 minutes (grid search)|
|08 (ablation)|about 60 minutes (B and C grid searches)|
|09 (evaluate)|under 30 seconds|
|10 (figures)|under 5 minutes|

Total from frozen intermediates: approximately 2.5 hours.
