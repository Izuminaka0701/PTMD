#!/usr/bin/env python3
"""
Chạy toàn bộ pipeline PTMD trên VM (CPU only).

Usage:
  python scripts/run_pipeline.py --step all
  python scripts/run_pipeline.py --step fetch --limit 3 --download
  python scripts/run_pipeline.py --step train --epochs 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="PTMD full pipeline runner (CPU only)")
    parser.add_argument(
        "--step",
        choices=["check", "fetch", "download", "extract", "train", "eval", "all"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--verify-dar", action="store_true")
    args = parser.parse_args()

    steps = {
        "check": lambda: run([PYTHON, "scripts/check_api.py"]),
        "fetch": lambda: run(
            [PYTHON, "scripts/01_fetch_sample_metadata.py", "--limit", str(args.limit)]
            + (["--download"] if args.download else [])
        ),
        "download": lambda: run([PYTHON, "scripts/00_download_samples.py"]),
        "extract": lambda: run(
            [PYTHON, "scripts/02_extract_features.py"]
            + (["--verify-dar"] if args.verify_dar else [])
        ),
        "train": lambda: run(
            [PYTHON, "scripts/03_train_model.py", "--epochs", str(args.epochs)]
        ),
        "eval": lambda: run([PYTHON, "scripts/04_evaluate.py"]),
    }

    if args.step == "all":
        for name in ["check", "fetch", "extract", "train", "eval"]:
            steps[name]()
    else:
        steps[args.step]()


if __name__ == "__main__":
    main()
