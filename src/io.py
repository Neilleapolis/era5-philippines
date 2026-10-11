"""
src/io.py
=========
Small, focused I/O helpers for the era5-philippines pipeline.

Established in Phase 4. Extended in Phase 5 (solar geometry) with
load_cleaned / save_with_geometry. Extended in Phase 6 (feature
engineering) with load_with_geometry / save_features and a small
fingerprint helper used for frozen-input integrity checks. Extended
in Phase 7 (train/val/test split) with load_features / save_split.
Extended in Phase 9 (persistence baseline) with strict text I/O.

Design rules:
- Paths are relative to project root (scripts are run from root).
- No transformation logic lives here — only load/save/fingerprint.
- Function names are stable; downstream scripts import these.
"""

import hashlib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Path anchors (relative to project root)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CLEANED_DIR = DATA_DIR / "cleaned"
WITH_GEOMETRY_DIR = DATA_DIR / "with_geometry"   # Phase 5 output
FEATURES_DIR = DATA_DIR / "features"             # Phase 6 output
SPLITS_DIR = DATA_DIR / "splits" / "paper1"      # Phase 7 output
RESULTS_DIR = DATA_DIR.parent / "results" / "paper1"


# ---------------------------------------------------------------------
# Extracted (Phase 3 output)
# ---------------------------------------------------------------------
def load_extracted(city: str) -> pd.DataFrame:
    """
    Load data/extracted/{city}.csv with `time` parsed as UTC datetime.

    Parameters
    ----------
    city : str
        One of: 'muntinlupa', 'baguio', 'cebu', 'davao'.

    Returns
    -------
    pd.DataFrame
        Raw extracted ERA5 values (J/m², K, m, etc.) — NOT converted.
    """
    path = EXTRACTED_DIR / f"{city}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Extracted file not found: {path}. Run Phase 3 first."
        )
    df = pd.read_csv(path, parse_dates=["time"])
    # Phase 3 wrote UTC timestamps; make the tz explicit so arithmetic
    # with Timedelta is unambiguous.
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------
# Cleaned (Phase 4 output)
# ---------------------------------------------------------------------
def save_cleaned(city: str, df: pd.DataFrame) -> None:
    """
    Save the cleaned DataFrame to data/cleaned/{city}.csv.

    The `time` and `local_time` columns are written in ISO-8601 so
    they round-trip cleanly through pandas on later reads.
    """
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    path = CLEANED_DIR / f"{city}.csv"
    df.to_csv(path, index=False)
    print(f"  [io] wrote {path.relative_to(PROJECT_ROOT)}  "
          f"({len(df):,} rows, {df.shape[1]} cols)")


# ---------------------------------------------------------------------
# Cleaned (Phase 4 output) — read side, added Phase 5
# ---------------------------------------------------------------------
def load_cleaned(city: str) -> pd.DataFrame:
    """
    Load data/cleaned/{city}.csv (frozen Phase 4 output).

    Both `time` and `local_time` are parsed back to tz-aware
    datetimes (UTC), matching how Phase 4 saved them.

    Read-only. Never writes back to data/cleaned/.
    """
    path = CLEANED_DIR / f"{city}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {path}. Run Phase 4 first."
        )
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df


# ---------------------------------------------------------------------
# With-geometry (Phase 5 output)
# ---------------------------------------------------------------------
def save_with_geometry(city: str, df: pd.DataFrame) -> None:
    """
    Save the solar-geometry DataFrame to data/with_geometry/{city}.csv.

    New folder — never touches data/cleaned/. Creates the folder if
    it does not exist. `time` and `local_time` are written in
    ISO-8601 so they round-trip cleanly on later reads.
    """
    WITH_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path = WITH_GEOMETRY_DIR / f"{city}.csv"
    df.to_csv(path, index=False)
    print(f"  [io] wrote {path.relative_to(PROJECT_ROOT)}  "
          f"({len(df):,} rows, {df.shape[1]} cols)")


# ---------------------------------------------------------------------
# With-geometry (Phase 5 output) — read side, added Phase 6
# ---------------------------------------------------------------------
def load_with_geometry(city: str) -> pd.DataFrame:
    """
    Load data/with_geometry/{city}.csv (frozen Phase 5 output).

    Both `time` and `local_time` are parsed back to tz-aware
    datetimes (UTC), matching how Phase 5 saved them.

    Read-only. Never writes back to data/with_geometry/.
    """
    path = WITH_GEOMETRY_DIR / f"{city}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"With-geometry file not found: {path}. Run Phase 5 first."
        )
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["local_time"] = pd.to_datetime(df["local_time"], utc=True)
    return df


# ---------------------------------------------------------------------
# Features (Phase 6 output)
# ---------------------------------------------------------------------
def save_features(city: str, df: pd.DataFrame) -> None:
    """
    Save the engineered-features DataFrame to data/features/{city}.csv.

    New folder — never touches data/with_geometry/ or any other
    frozen folder. Creates the folder if it does not exist. `time`
    and `local_time` are written in ISO-8601 so they round-trip
    cleanly on later reads.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / f"{city}.csv"
    df.to_csv(path, index=False)
    print(f"  [io] wrote {path.relative_to(PROJECT_ROOT)}  "
          f"({len(df):,} rows, {df.shape[1]} cols)")


# ---------------------------------------------------------------------
# Fingerprint helper — added Phase 6
# ---------------------------------------------------------------------
def sha1_head_tail(path: Path, n_bytes: int = 4096) -> str:
    """
    Cheap fingerprint of a file: SHA1 over
    (size || first n_bytes || last n_bytes).

    Used by downstream scripts to snapshot frozen inputs before and
    after a run, so any accidental write to a frozen folder is
    detected loudly. Full-file SHA1 is avoided to keep runtime low
    on the ~10 MB with_geometry CSVs.
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(n_bytes)
        f.seek(max(0, size - n_bytes))
        tail = f.read(n_bytes)
    h = hashlib.sha1()
    h.update(str(size).encode())
    h.update(head)
    h.update(tail)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Strict text I/O — added Phase 9
#
# Rationale: a fallback-chain reader (utf-8 → cp1252 → latin-1) can
# silently re-encode a file on write, turning a UTF-8 em-dash into a
# cp1252 byte. That bug cost Phase 9 two rounds of debugging. These
# helpers declare one encoding and fail loudly on mismatch, so any
# encoding drift is caught at the moment it happens.
# ---------------------------------------------------------------------
def read_text_strict(path: Path, encoding: str = "utf-8") -> str:
    """
    Read a text file with a single declared encoding.

    Raises UnicodeDecodeError if the file is not valid in `encoding`.
    No fallback. Use this for docs, config, and any file whose
    encoding should be stable across platforms.
    """
    return Path(path).read_text(encoding=encoding)


def write_text_strict(
    path: Path, text: str, encoding: str = "utf-8"
) -> None:
    """
    Write text with a single declared encoding, LF newlines, no BOM.

    `newline=""` prevents Python from translating `\\n` to the
    platform default on write, so bytes on disk match `text` exactly.
    """
    Path(path).write_text(text, encoding=encoding, newline="")


# ---------------------------------------------------------------------
# Features (Phase 6 output) — read side, added Phase 7
# ---------------------------------------------------------------------
def load_features(city: str) -> pd.DataFrame:
    """
    Load data/features/{city}.csv (frozen Phase 6 output).

    `time` is parsed back to a tz-aware UTC Timestamp so that
    comparisons against TRAIN_END / VAL_END / TEST_END (also UTC)
    are exact. The locked column order is preserved; we do NOT
    reorder columns here.

    Read-only. Never writes back to data/features/.
    """
    path = FEATURES_DIR / f"{city}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen feature file: {path}. Run Phase 6 first."
        )
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"{path} has no `time` column")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ---------------------------------------------------------------------
# Splits (Phase 7 output)
# ---------------------------------------------------------------------
def save_split(city: str, split_name: str, df: pd.DataFrame) -> Path:
    """
    Save one split to data/splits/paper1/{city}_{split_name}.csv.

    split_name must be one of {'train', 'val', 'test'}. New folder —
    never touches data/features/ or any other frozen folder. Creates
    the splits directory if it does not exist. Returns the written
    path so callers can log it.
    """
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split_name: {split_name!r}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS_DIR / f"{city}_{split_name}.csv"

    # index=False: the features file's row order is positional; we
    # do not want to persist a pandas index.
    df.to_csv(out_path, index=False)
    return out_path