#!/usr/bin/env python
"""
Phase 7 — Train / Val / Test Split
==================================

Splits each city's frozen feature table into three chronological
subsets using the frozen date boundaries from
config/paper1_config.py. Writes 12 CSVs to data/splits/paper1/.

Boundary semantics
------------------
config/paper1_config.py stores the split boundaries as calendar-date
strings with INCLUSIVE end dates, e.g.:

    TRAIN_START = "2023-01-01"   TRAIN_END = "2024-06-30"
    VAL_START   = "2024-07-01"   VAL_END   = "2024-12-31"
    TEST_START  = "2025-01-01"   TEST_END  = "2025-12-31"

"TRAIN_END = 2024-06-30" means 2024-06-30 is the last included day.
This script converts each inclusive END date to an exclusive upper
bound (END + 1 day at 00:00 UTC) so the masks become clean
half-open intervals over UTC-aware hourly timestamps:

    train: [TRAIN_START 00:00,  TRAIN_END+1d 00:00)
    val:   [VAL_START   00:00,  VAL_END+1d   00:00)
    test:  [TEST_START  00:00,  TEST_END+1d  00:00)

The config is treated as frozen and is NOT modified by this script.

Non-negotiable rules enforced here:
  * Chronological split only. No shuffling.
  * No sklearn train_test_split.
  * Warmup rows (first 168 of each city) stay in TRAIN.
  * data/features/ is read-only. Fingerprints checked before/after.

At the end of a successful run the script emits a ready-to-paste
Phase 7 section for docs/frozen_fingerprints.md. The block is:
  (a) printed to stdout between BEGIN/END markers, and
  (b) appended to docs/frozen_fingerprints.md, but only once
      (idempotent: a re-run that finds the Phase 7 header in the
      file skips the append).

Run:  python scripts/06_split.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Repo root + import path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paper1_config import (  # noqa: E402
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
)
from src import io as sio  # noqa: E402


CITIES = ["muntinlupa", "baguio", "cebu", "davao"]

EXPECTED_ROWS = {"train": 13_128, "val": 4_416, "test": 8_760}
EXPECTED_TOTAL = 26_304
EXPECTED_COLS = 80

# Expected time ranges (inclusive on both ends, as printed).
EXPECTED_RANGES = {
    "train": ("2023-01-01 00:00:00+00:00", "2024-06-30 23:00:00+00:00"),
    "val":   ("2024-07-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00"),
    "test":  ("2025-01-01 00:00:00+00:00", "2025-12-31 23:00:00+00:00"),
}

PHASE7_HEADER = "## Phase 7 — Train/Val/Test Split"
FINGERPRINT_MARKER_BEGIN = "--- BEGIN PHASE 7 FINGERPRINTS ---"
FINGERPRINT_MARKER_END = "--- END PHASE 7 FINGERPRINTS ---"


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def _day_start_utc(date_str) -> pd.Timestamp:
    """Parse a date (or datetime) string as UTC midnight.

    paper1_config.py stores split boundaries as calendar-date strings
    like "2024-06-30". Anchor them to 00:00:00 UTC on that day.
    """
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _exclusive_upper_bound(end_date_str) -> pd.Timestamp:
    """Convert an inclusive END date to an exclusive upper bound.

    paper1_config.py uses inclusive END dates ("last day included"),
    e.g. TRAIN_END = "2024-06-30" means the last included day is
    2024-06-30. Convert to half-open interval by adding one day, so
    the mask becomes `time < 2024-07-01 00:00 UTC`.
    """
    return _day_start_utc(end_date_str) + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _fingerprint_features() -> dict[str, tuple[int, str]]:
    """Snapshot size + head/tail SHA1 for every features file."""
    fps: dict[str, tuple[int, str]] = {}
    for city in CITIES:
        path = sio.FEATURES_DIR / f"{city}.csv"
        size = path.stat().st_size
        sha = sio.sha1_head_tail(path)
        fps[city] = (size, sha)
    return fps


# ---------------------------------------------------------------------------
# Split correctness helpers
# ---------------------------------------------------------------------------

def _assert_masks_partition(
    df: pd.DataFrame, masks: dict[str, pd.Series]
) -> None:
    """Verify the three masks are mutually exclusive and cover all rows."""
    names = list(masks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = masks[names[i]], masks[names[j]]
            if (a & b).any():
                n = int((a & b).sum())
                raise AssertionError(
                    f"Masks {names[i]!r} and {names[j]!r} overlap on "
                    f"{n} rows"
                )
    covered = masks[names[0]]
    for n in names[1:]:
        covered = covered | masks[n]
    n_uncovered = int((~covered).sum())
    if n_uncovered:
        bad = df.loc[~covered, "time"].head(5).tolist()
        raise AssertionError(
            f"{n_uncovered} rows not covered by any mask. "
            f"First uncovered timestamps: {bad}"
        )


def split_city(
    city: str,
    train_start, train_end_excl,
    val_start,   val_end_excl,
    test_start,  test_end_excl,
) -> dict:
    """Load, mask, verify, and save one city's three splits."""
    print(f"\n{'=' * 68}")
    print(f"CITY: {city}")
    print(f"{'=' * 68}")

    df = sio.load_features(city)
    n_rows, n_cols = df.shape
    print(f"  Loaded {n_rows:,} rows × {n_cols} cols from "
          f"data/features/{city}.csv")
    print("  Locked column order preserved (no reorder).")

    if n_cols != EXPECTED_COLS:
        raise AssertionError(
            f"{city}: expected {EXPECTED_COLS} columns, got {n_cols}"
        )

    train_mask = (df["time"] >= train_start) & (df["time"] < train_end_excl)
    val_mask   = (df["time"] >= val_start)   & (df["time"] < val_end_excl)
    test_mask  = (df["time"] >= test_start)  & (df["time"] < test_end_excl)

    masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    _assert_masks_partition(df, masks)

    splits = {
        "train": df.loc[train_mask].copy(),
        "val":   df.loc[val_mask].copy(),
        "test":  df.loc[test_mask].copy(),
    }

    for name, part in splits.items():
        n = len(part)
        if n != EXPECTED_ROWS[name]:
            raise AssertionError(
                f"{city}/{name}: expected {EXPECTED_ROWS[name]} rows, "
                f"got {n}"
            )
        if part.shape[1] != n_cols:
            raise AssertionError(
                f"{city}/{name}: column count drifted to {part.shape[1]}"
            )
        if list(part.columns) != list(df.columns):
            raise AssertionError(
                f"{city}/{name}: column order changed vs parent "
                f"features file"
            )
        if not part["time"].is_monotonic_increasing:
            raise AssertionError(f"{city}/{name}: time is not monotonic")
        if part["time"].duplicated().any():
            n_dup = int(part["time"].duplicated().sum())
            raise AssertionError(
                f"{city}/{name}: {n_dup} duplicate timestamps"
            )

    if not (splits["train"]["time"].max() < splits["val"]["time"].min()):
        raise AssertionError(f"{city}: train/val temporal overlap")
    if not (splits["val"]["time"].max() < splits["test"]["time"].min()):
        raise AssertionError(f"{city}: val/test temporal overlap")

    total = sum(len(s) for s in splits.values())
    if total != EXPECTED_TOTAL:
        raise AssertionError(
            f"{city}: split total {total} != {EXPECTED_TOTAL}"
        )

    parent_nan_cols = set(df.columns[df.isna().any()])
    for name, part in splits.items():
        nan_cols = set(part.columns[part.isna().any()])
        new_nan = nan_cols - parent_nan_cols
        if new_nan:
            raise AssertionError(
                f"{city}/{name}: introduced NaN in columns "
                f"{sorted(new_nan)}"
            )

    for name, part in splits.items():
        path = sio.save_split(city, name, part)
        print(f"  wrote {path.relative_to(ROOT)}")

    for name in ("train", "val", "test"):
        part = splits[name]
        t0 = part["time"].iloc[0]
        t1 = part["time"].iloc[-1]
        exp0, exp1 = EXPECTED_RANGES[name]
        ok = (str(t0) == exp0) and (str(t1) == exp1)
        flag = "OK " if ok else "!! "
        print(f"  [{flag}] {name:<5}  rows={len(part):>6,}  "
              f"{t0}  →  {t1}")

    return {
        "city": city,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows": {k: len(v) for k, v in splits.items()},
    }


# ---------------------------------------------------------------------------
# Fingerprint markdown emission
# ---------------------------------------------------------------------------

def _build_fingerprint_block(
    fp_before: dict[str, tuple[int, str]],
    fp_after: dict[str, tuple[int, str]],
) -> str:
    """Build the Phase 7 markdown section as a string."""
    drift = [c for c in CITIES if fp_before[c] != fp_after[c]]
    if drift:
        raise AssertionError(
            f"Cannot emit fingerprint markdown: features drifted "
            f"for {drift}"
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def feat_row(city: str) -> str:
        size, sha = fp_after[city]
        name = f"{city}.csv"
        return f"| {name:<42} | {size:>12,} | {sha} |"

    def split_row(city: str, split: str) -> str:
        path = sio.SPLITS_DIR / f"{city}_{split}.csv"
        size = path.stat().st_size
        sha = sio.sha1_head_tail(path)
        return f"| {path.name:<42} | {size:>12,} | {sha} |"

    header = (
        "| file                                       | "
        "size (bytes) | sha1_head_tail |"
    )
    sep = (
        "|--------------------------------------------|"
        "--------------|----------------|"
    )

    lines: list[str] = []
    lines.append(PHASE7_HEADER)
    lines.append("")
    lines.append(f"Frozen at: {now}")
    lines.append("Script: scripts/06_split.py")
    lines.append(
        "Fingerprints taken BEFORE and AFTER the split; identical, "
        "so the frozen Phase 6 features files are unchanged by "
        "this stage."
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for c in CITIES:
        lines.append(feat_row(c))
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for c in CITIES:
        for s in ("train", "val", "test"):
            lines.append(split_row(c, s))
    lines.append("")
    lines.append(
        "Boundary note: paper1_config.py stores inclusive END dates "
        "(e.g. TRAIN_END = '2024-06-30' means the last included day). "
        "This script converts each to an exclusive upper bound "
        "(END + 1 day at 00:00 UTC) before masking, yielding the "
        "half-open intervals:"
    )
    lines.append(
        "  train: [2023-01-01 00:00 UTC, 2024-07-01 00:00 UTC)"
    )
    lines.append(
        "  val:   [2024-07-01 00:00 UTC, 2025-01-01 00:00 UTC)"
    )
    lines.append(
        "  test:  [2025-01-01 00:00 UTC, 2026-01-01 00:00 UTC)"
    )
    lines.append(
        "No shuffling, no sklearn train_test_split. The warmup rows "
        "(first 168 per city, zero-filled lag_168) are contained "
        "entirely within the train split."
    )

    return "\n".join(lines)


def _emit_fingerprint_markdown(
    fp_before: dict[str, tuple[int, str]],
    fp_after: dict[str, tuple[int, str]],
) -> None:
    """Print a ready-to-paste Phase 7 section to stdout."""
    block = _build_fingerprint_block(fp_before, fp_after)
    print("\n" + "=" * 68)
    print("FINGERPRINT BLOCK FOR docs/frozen_fingerprints.md")
    print("=" * 68)
    print("Copy everything between the BEGIN/END markers into")
    print("docs/frozen_fingerprints.md, appended at the end.")
    print(FINGERPRINT_MARKER_BEGIN)
    print(block)
    print(FINGERPRINT_MARKER_END)


def _emit_fingerprint_block_to_file(
    fp_before: dict[str, tuple[int, str]],
    fp_after: dict[str, tuple[int, str]],
    target: Path,
) -> None:
    """Append the Phase 7 block to target, but only once (idempotent)."""
    existing = target.read_text() if target.exists() else ""
    if PHASE7_HEADER in existing:
        print(f"  ({target.name} already contains "
              f"{PHASE7_HEADER!r}; skipping append)")
        return

    block = _build_fingerprint_block(fp_before, fp_after)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write("\n" + block + "\n")
    print(f"  appended Phase 7 section to "
          f"{target.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()

    train_start = _day_start_utc(TRAIN_START)
    train_end_excl = _exclusive_upper_bound(TRAIN_END)
    val_start = _day_start_utc(VAL_START)
    val_end_excl = _exclusive_upper_bound(VAL_END)
    test_start = _day_start_utc(TEST_START)
    test_end_excl = _exclusive_upper_bound(TEST_END)

    print("Phase 7 — Train/Val/Test Split")
    print("  paper1_config boundaries (inclusive END dates):")
    print(f"    TRAIN: {TRAIN_START}  →  {TRAIN_END}")
    print(f"    VAL:   {VAL_START}  →  {VAL_END}")
    print(f"    TEST:  {TEST_START}  →  {TEST_END}")
    print("  Derived half-open intervals (UTC):")
    print(f"    train: [{train_start}, {train_end_excl})")
    print(f"    val:   [{val_start}, {val_end_excl})")
    print(f"    test:  [{test_start}, {test_end_excl})")

    print("\nFrozen-input fingerprints (BEFORE):")
    fp_before = _fingerprint_features()
    for city in CITIES:
        size, sha = fp_before[city]
        print(f"  {city:<11} size={size:>9,}  sha1_head_tail={sha}")

    summaries = []
    for city in CITIES:
        summaries.append(split_city(
            city,
            train_start, train_end_excl,
            val_start,   val_end_excl,
            test_start,  test_end_excl,
        ))

    print("\nFrozen-input fingerprints (AFTER):")
    fp_after = _fingerprint_features()
    for city in CITIES:
        size, sha = fp_after[city]
        print(f"  {city:<11} size={size:>9,}  sha1_head_tail={sha}")

    drift = [c for c in CITIES if fp_before[c] != fp_after[c]]
    if drift:
        print(f"\nFATAL: frozen features changed for cities: {drift}")
        return 2
    print("\nFrozen-input integrity: PASS (all 4 features files "
          "unchanged)")

    print("\n" + "=" * 68)
    print("AGGREGATE")
    print("=" * 68)
    print(f"{'city':<11} {'train':>7} {'val':>6} {'test':>7} {'total':>7}")
    for s in summaries:
        r = s["rows"]
        tot = r["train"] + r["val"] + r["test"]
        print(f"{s['city']:<11} {r['train']:>7,} {r['val']:>6,} "
              f"{r['test']:>7,} {tot:>7,}")

    _emit_fingerprint_markdown(fp_before, fp_after)
    _emit_fingerprint_block_to_file(
        fp_before,
        fp_after,
        ROOT / "docs" / "frozen_fingerprints.md",
    )

    elapsed = time.time() - t_start
    print(f"\nRuntime: {elapsed:.2f}s")
    print("Phase 7 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())