#!/usr/bin/env python3
"""
Convert API Monitor CSV export → PTMD sandbox log JSON.

Usage (trên VM):
  python scripts/convert_api_monitor.py --input api_log.csv --output data/sandbox_logs/sample.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import save_json


def parse_csv(csv_path: Path) -> list[dict]:
    calls = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            api = row.get("API Name") or row.get("api") or row.get("Function") or ""
            module = row.get("Module") or row.get("module") or row.get("DLL") or "unknown"
            ts = float(row.get("Time of Day", row.get("timestamp", i * 0.001)))
            args = []
            for key in row:
                if key.startswith("Arg") or key.startswith("Parameter"):
                    if row[key]:
                        args.append(row[key])
            calls.append({
                "timestamp": ts,
                "api": api.strip(),
                "module": module.strip(),
                "args": args,
            })
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert API Monitor CSV to PTMD log JSON")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    calls = parse_csv(Path(args.input))
    save_json({"sandbox": "api_monitor", "api_calls": calls}, args.output)
    print(f"Converted {len(calls)} API calls → {args.output}")


if __name__ == "__main__":
    main()
