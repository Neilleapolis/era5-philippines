"""Phase 18 convenience runner for the full era5-philippines pipeline.

Script name : run_all.py
Phase       : 18 (Code cleanup and reproducibility)
Inputs      : data/raw/ (for stage 01, optional)
Outputs     : delegates to each stage's own outputs
Rules       : no pipeline logic; subprocess-only; abort on first failure

Run from project root:
    python scripts/run_all.py
    python scripts/run_all.py --from 06
    python scripts/run_all.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGES = [
    "01_download_era5.py",
    "02_extract_to_csv.py",
    "03_clean_and_convert.py",
    "04_solar_geometry.py",
    "05_feature_engineering.py",
    "06_split.py",
    "07a_persistence.py",
    "07b_xgb_baseline.py",
    "07c_xgb_tuned.py",
    "08_ablation.py",
    "09_evaluate.py",
    "10_generate_figures.py",
]


def _stage_prefix(name: str) -> str:
    return name.split("_", 1)[0]


def _select_stages(start: str | None) -> list[str]:
    if start is None:
        return list(STAGES)
    for i, s in enumerate(STAGES):
        if _stage_prefix(s) == start:
            return STAGES[i:]
    raise SystemExit(f"--from {start!r} matched no stage")


def _raw_data_nonempty() -> bool:
    raw = PROJECT_ROOT / "data" / "raw"
    if not raw.is_dir():
        return False
    return any(raw.iterdir())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="start", default=None,
                   help="stage prefix to start from, e.g. 06")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sequence without executing")
    args = p.parse_args(argv)

    stages = _select_stages(args.start)

    if args.dry_run:
        print("Would run, in order:")
        for s in stages:
            print(f"  scripts/{s}")
        return 0

    for i, name in enumerate(stages, 1):
        path = PROJECT_ROOT / "scripts" / name
        if not path.exists():
            print(f"ERROR: missing {path}")
            return 2

        if name == "01_download_era5.py" and _raw_data_nonempty():
            print("data/raw/ already contains files.")
            reply = input("Re-download? [y/N] ").strip().lower()
            if reply != "y":
                print("Skipping 01_download_era5.py")
                continue

        print("=" * 60)
        print(f"STAGE {i}/{len(stages)}: {name}")
        print("=" * 60)
        try:
            subprocess.run(
                [sys.executable, str(path)],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.CalledProcessError as exc:
            print(f"STAGE FAILED: {name} (exit {exc.returncode})")
            return exc.returncode
        print(f"--- {name} complete ---")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
