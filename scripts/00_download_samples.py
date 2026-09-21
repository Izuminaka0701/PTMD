#!/usr/bin/env python3
"""
Tải sample từ manifest qua MalwareBazaar API.

Usage (trên VM):
  python scripts/00_download_samples.py
  python scripts/00_download_samples.py --manifest data/metadata/sample_manifest.json --limit 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.integrations.malwarebazaar import MalwareBazaarClient, MalwareBazaarError
from src.utils.io import load_json, save_json
from src.utils.pe_utils import find_sample_file, is_pe_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Download samples from MalwareBazaar")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số sample tải")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--yes", "-y", action="store_true", help="Bỏ qua confirm prompt")
    args = parser.parse_args()

    cfg = load_config()
    manifest_path = Path(args.manifest or cfg["paths"]["manifest_file"])
    manifest = load_json(manifest_path)
    samples = manifest.get("samples", [])

    if args.limit:
        samples = samples[: args.limit]

    try:
        client = MalwareBazaarClient()
    except MalwareBazaarError as ex:
        print(f"ERROR: {ex}")
        sys.exit(1)

    samples_dir = Path(cfg["paths"]["samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)

    if not args.yes:
        print(f"\n\u26a0  WARNING: Bạn sắp tải tối đa {len(samples)} malware sample thật.")
        print("   Chỉ nên chạy trên VM cô lập (isolated VM).")
        confirm = input("   Tiếp tục? (y/N): ").strip().lower()
        if confirm != "y":
            print("Đã hủy download.")
            sys.exit(0)

    ok, skip, fail = 0, 0, 0
    for i, s in enumerate(samples, 1):
        sha = s["sha256"]
        existing = find_sample_file(samples_dir, sha, s.get("filename", ""))
        if args.skip_existing and existing and is_pe_file(existing):
            print(f"  [{i}] SKIP (exists) {sha[:16]}...")
            skip += 1
            continue

        print(f"  [{i}] Downloading {sha[:16]}... ({s.get('family_name', '?')})")
        try:
            path = client.download_file(sha, samples_dir)
            if path:
                s["local_path"] = str(path)
                s["filename"] = path.name
                ok += 1
                print(f"       → {path.name}")
            else:
                fail += 1
                print("       → FAILED (empty response)")
        except Exception as ex:
            fail += 1
            print(f"       → ERROR: {ex}")
        time.sleep(0.3)

    save_json(manifest, manifest_path)
    print(f"\nDone: downloaded={ok}, skipped={skip}, failed={fail}")
    print(f"Manifest updated → {manifest_path}")


if __name__ == "__main__":
    main()
