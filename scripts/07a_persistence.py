#!/usr/bin/env python
"""
scripts/07a_persistence.py

Phase 9 — Model 1: Persistence Baseline
era5-philippines / Paper 1

Predicts ssrd_wm2 from lag_24 (strictly 24h in the past).
No training, no validation, no learning. Pure naive floor.

Reads:   data/splits/paper1/{city}_test.csv
         data/splits/paper1/{city}_{train,val}.csv  (fingerprint only)
Writes:  results/paper1/metrics/persistence.csv
         results/paper1/tables/persistence_summary.csv

Fails loudly if data/splits/paper1/*.csv changed during the run.
Uses src.io.sha1_head_tail — the same fingerprint function that
docs/frozen_fingerprints.md was written with.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Path bootstrap
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import all_metrics, mape_daytime          # noqa: E402
from src.io import sha1_head_tail, SPLITS_DIR              # noqa: E402

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
RESULTS_METRICS_DIR = REPO_ROOT / "results" / "paper1" / "metrics"
RESULTS_TABLES_DIR = REPO_ROOT / "results" / "paper1" / "tables"
FINGERPRINTS_MD = REPO_ROOT / "docs" / "frozen_fingerprints.md"

CITIES = ["muntinlupa", "baguio", "cebu", "davao"]
SPLIT_KINDS = ["train", "val", "test"]


# ----------------------------------------------------------------------
# Text reader (encoding-tolerant)
# ----------------------------------------------------------------------
def _read_text_any_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1252", errors="replace")


# ----------------------------------------------------------------------
# Snapshot splits using the pipeline's own fingerprint function
# ----------------------------------------------------------------------
def snapshot_splits() -> dict[str, str]:
    """{relative_path: full 40-hex sha1_head_tail} for all 12 split CSVs."""
    snap: dict[str, str] = {}
    for city in CITIES:
        for kind in SPLIT_KINDS:
            p = SPLITS_DIR / f"{city}_{kind}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing split file: {p}")
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            snap[rel] = sha1_head_tail(p)
    return snap


# ----------------------------------------------------------------------
# Parser for docs/frozen_fingerprints.md
# ----------------------------------------------------------------------
_HEADING_RE = re.compile(r"^##\s+(.*)$")
_ROW_RE = re.compile(
    r"^\|\s*([A-Za-z0-9_\-./]+\.csv)\s*\|\s*([\d,]+)\s*\|\s*([0-9a-fA-F]{6,40})\s*\|\s*$"
)


def _section_folder(heading: str) -> str | None:
    h = heading.lower()
    if "phase 7" in h and "split" in h:
        return "data/splits/paper1"
    if "cleaned" in h and "phase 4" in h:
        return "data/cleaned"
    if "with_geometry" in h and "phase 5" in h:
        return "data/with_geometry"
    return None


def parse_fingerprints_md(path: Path) -> dict[str, str]:
    """Return {rel_path: first-12-hex sha1} for sections we recognise."""
    if not path.exists():
        raise FileNotFoundError(f"Missing fingerprints doc: {path}")
    text = _read_text_any_encoding(path)

    out: dict[str, str] = {}
    current_folder: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        m = _HEADING_RE.match(line)
        if m:
            current_folder = _section_folder(m.group(1))
            continue
        if current_folder is None:
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        basename, _size, sha = m.group(1), m.group(2), m.group(3)
        out[f"{current_folder}/{basename}"] = sha.lower()[:12]

    return out


# ----------------------------------------------------------------------
# Fingerprint check
# ----------------------------------------------------------------------
def check_fingerprints(stage: str) -> None:
    current = snapshot_splits()
    expected_all = parse_fingerprints_md(FINGERPRINTS_MD)
    expected = {
        k: v for k, v in expected_all.items()
        if k.startswith("data/splits/paper1/")
    }

    if not expected:
        raise RuntimeError(
            f"[{stage}] No split fingerprints parsed from {FINGERPRINTS_MD}.\n"
            f"  Expected a '## Phase 7' section with rows of the form\n"
            f"  '| <basename>.csv | <size> | <12-hex sha1> |'."
        )

    mismatches: list[str] = []
    missing: list[str] = []
    for rel, full_sha in current.items():
        prefix = full_sha[:12]
        if rel not in expected:
            missing.append(rel)
        elif expected[rel] != prefix:
            mismatches.append(
                f"  {rel}:\n      disk = {prefix}\n      doc  = {expected[rel]}"
            )

    if mismatches or missing:
        parts = [f"\n[{stage}] FROZEN SPLIT FINGERPRINT MISMATCH — refusing to continue."]
        if mismatches:
            parts.append("  SHA1 head mismatches:")
            parts.extend(mismatches)
        if missing:
            parts.append("  On disk but not listed in fingerprints doc:")
            parts.extend(f"    {r}" for r in missing)
        raise RuntimeError("\n".join(parts) + "\n")

    print(
        f"[{stage}] Fingerprints OK — {len(current)} split files match "
        f"docs/frozen_fingerprints.md (src.io.sha1_head_tail)"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 9 — Persistence Baseline (lag_24)")
    print("=" * 72)

    check_fingerprints("start")

    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    summary_rows: list[dict] = []

    for city in CITIES:
        test_path = SPLITS_DIR / f"{city}_test.csv"
        print(f"\n--- {city} ---")
        print(f"  loading {test_path.relative_to(REPO_ROOT)}")
        test = pd.read_csv(test_path)

        for col in ("ssrd_wm2", "lag_24", "season"):
            if col not in test.columns:
                raise KeyError(f"{test_path} missing required column '{col}'")

        y_true = test["ssrd_wm2"].to_numpy(dtype=float)
        y_pred = np.clip(test["lag_24"].to_numpy(dtype=float), 0.0, None)

        n_total = int(len(y_true))
        if int(np.sum(y_pred < 0)) != 0:
            raise RuntimeError(f"{city}: negative predictions after clip (impossible)")

        m = all_metrics(y_true, y_pred)
        print(
            f"  n_total={n_total}  n_daytime={m['n_daytime']}  "
            f"mae={m['mae']:.4f}  rmse={m['rmse']:.4f}  "
            f"r2={m['r2']:.4f}  mape={m['mape']:.4f}%"
        )

        metric_rows.append({
            "city": city,
            "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
            "mape": m["mape"], "n_daytime": m["n_daytime"],
            "n_total": n_total,
        })

        season = test["season"].to_numpy()
        dry_mask = season == 0
        wet_mask = season == 1
        if dry_mask.sum() == 0 or wet_mask.sum() == 0:
            raise RuntimeError(
                f"{city}: empty seasonal split "
                f"(dry={int(dry_mask.sum())}, wet={int(wet_mask.sum())})"
            )

        mape_dry = mape_daytime(y_true[dry_mask], y_pred[dry_mask])
        mape_wet = mape_daytime(y_true[wet_mask], y_pred[wet_mask])
        if np.isnan(mape_dry) or np.isnan(mape_wet):
            raise RuntimeError(
                f"{city}: NaN seasonal MAPE (dry={mape_dry}, wet={mape_wet})"
            )

        wet_minus_dry = float(mape_wet - mape_dry)
        print(
            f"  dry_mape={mape_dry:.4f}%  wet_mape={mape_wet:.4f}%  "
            f"wet-dry={wet_minus_dry:+.4f} pp"
        )

        summary_rows.append({
            "city": city,
            "daytime_mape": m["mape"],
            "dry_mape": mape_dry,
            "wet_mape": mape_wet,
            "wet_minus_dry_pp": wet_minus_dry,
            "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
            "n_daytime": m["n_daytime"],
        })

    metrics_df = pd.DataFrame(
        metric_rows,
        columns=["city", "mae", "rmse", "r2", "mape", "n_daytime", "n_total"],
    )
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "city", "daytime_mape", "dry_mape", "wet_mape",
            "wet_minus_dry_pp", "mae", "rmse", "r2", "n_daytime",
        ],
    )

    if metrics_df.isna().any().any():
        raise RuntimeError("NaN detected in persistence.csv")
    if summary_df.isna().any().any():
        raise RuntimeError("NaN detected in persistence_summary.csv")

    metrics_path = RESULTS_METRICS_DIR / "persistence.csv"
    summary_path = RESULTS_TABLES_DIR / "persistence_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY — persistence.csv")
    print("=" * 72)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("SUMMARY — persistence_summary.csv")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    print(f"\nWrote {metrics_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")

    check_fingerprints("end")

    # Plausibility banner
    print("\n" + "=" * 72)
    print("PLAUSIBILITY CHECK")
    print("=" * 72)
    expected_ranges = {
        "muntinlupa": (39.0, 40.0),
        "baguio": (37.0, 39.0),
        "cebu": (43.0, 46.0),
        "davao": (35.0, 38.0),
    }
    hard_bounds = (25.0, 55.0)
    flags: list[str] = []
    for row in metric_rows:
        c, v = row["city"], row["mape"]
        lo, hi = expected_ranges[c]
        status = "OK" if lo <= v <= hi else "OUTSIDE EXPECTED"
        if not (hard_bounds[0] <= v <= hard_bounds[1]):
            status += " / OUTSIDE HARD BOUNDS [25,55]"
            flags.append(f"{c}={v:.4f}% outside [25,55]")
        beat = "BEATS 18%" if v < 18.0 else "above 18% (expected)"
        print(f"  {c:11s} mape={v:6.4f}%  expected=[{lo:.0f},{hi:.0f}]  {status}  [{beat}]")

    for row in summary_rows:
        if not (row["wet_mape"] > row["dry_mape"]):
            flags.append(
                f"{row['city']}: wet_mape ({row['wet_mape']:.4f}) "
                f"<= dry_mape ({row['dry_mape']:.4f})"
            )

    if flags:
        print("\nFLAGS (non-fatal, review manually):")
        for f in flags:
            print(f"  - {f}")
    else:
        print("\nAll plausibility checks passed.")

    print("\nPhase 9 complete.")


if __name__ == "__main__":
    main()