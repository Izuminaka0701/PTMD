#!/usr/bin/env python3
"""
Bước 1 — Lấy metadata sample từ MalwareBazaar (cần ABUSECH_API_KEY trong .env).

Usage (trên VM):
  python scripts/01_fetch_sample_metadata.py
  python scripts/01_fetch_sample_metadata.py --family PlugX --limit 10
  python scripts/01_fetch_sample_metadata.py --download
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
from src.utils.family_map import FAMILY_TAGS, load_family_registry, resolve_family
from src.utils.io import load_json, save_json


def entry_to_manifest(entry: dict, family: dict, tag: str) -> dict:
    sha = entry.get("sha256_hash", "")
    return {
        "sha256": sha,
        "filename": entry.get("file_name", f"{sha[:16]}.bin"),
        "family_id": family["id"],
        "family_name": family["name"],
        "group": family["group"],
        "source": "malwarebazaar",
        "tag": tag,
        "first_seen": entry.get("first_seen"),
        "file_type": entry.get("file_type"),
        "file_size": entry.get("file_size"),
        "sandbox_log": f"data/sandbox_logs/{sha}.json",
    }


def fetch_for_family(client: MalwareBazaarClient, family: dict, limit: int) -> list[dict]:
    name = family["name"]
    tags = FAMILY_TAGS.get(name, [name.replace(" ", "")])
    results: list[dict] = []
    seen: set[str] = set()

    for tag in tags:
        entries = client.fetch_by_tags([tag], limit_per_tag=limit)
        for e in entries:
            sha = e.get("sha256_hash", "")
            if sha and sha not in seen:
                seen.add(sha)
                results.append(entry_to_manifest(e, family, tag))
        if results:
            break  # đủ sample từ tag đầu tiên thành công
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch malware metadata from MalwareBazaar")
    parser.add_argument("--family", type=str, help="Lọc theo tên MITRE family")
    parser.add_argument("--limit", type=int, default=5, help="Max sample mỗi family")
    parser.add_argument("--download", action="store_true", help="Tải luôn sample về data/samples/")
    parser.add_argument("--yes", "-y", action="store_true", help="Bỏ qua confirm prompt (dùng cho automation)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg["api"]["malwarebazaar_key_set"]:
        print("ERROR: ABUSECH_API_KEY chưa được cấu hình.")
        print("  Copy .env.example → .env và điền API key từ https://auth.abuse.ch/")
        sys.exit(1)

    try:
        client = MalwareBazaarClient()
    except MalwareBazaarError as ex:
        print(f"ERROR: {ex}")
        sys.exit(1)

    meta = load_json(cfg["paths"]["metadata_file"])
    families = meta["families"]

    if args.family:
        families = [f for f in families if f["name"].lower() == args.family.lower()]
        if not families:
            print(f"Family not found: {args.family}")
            sys.exit(1)

    all_samples: list[dict] = []
    print(f"Querying MalwareBazaar ({len(families)} families)...")

    for fam in families:
        print(f"  → {fam['name']} ({fam['group']})")
        found = fetch_for_family(client, fam, args.limit)
        print(f"    Found {len(found)} samples")
        all_samples.extend(found)

    if args.download and all_samples:
        if not args.yes:
            print(f"\n\u26a0  WARNING: Bạn sắp tải {len(all_samples)} malware sample thật.")
            print("   Chỉ nên chạy trên VM cô lập (isolated VM).")
            confirm = input("   Tiếp tục? (y/N): ").strip().lower()
            if confirm != "y":
                print("Đã hủy download. Chỉ lưu metadata.")
                out_path = Path(args.output or cfg["paths"]["manifest_file"])
                save_json({"samples": all_samples, "benign_samples": []}, out_path)
                print(f"Saved {len(all_samples)} entries → {out_path}")
                return
        samples_dir = Path(cfg["paths"]["samples_dir"])
        samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {len(all_samples)} samples...")
        for i, s in enumerate(all_samples, 1):
            sha = s["sha256"]
            print(f"  [{i}/{len(all_samples)}] {sha[:16]}...")
            path = client.download_file(sha, samples_dir)
            if path:
                s["local_path"] = str(path)
                s["filename"] = path.name
            time.sleep(0.5)

    out_path = Path(args.output or cfg["paths"]["manifest_file"])
    save_json({"samples": all_samples, "benign_samples": []}, out_path)
    print(f"\nSaved {len(all_samples)} entries → {out_path}")
    print("\nNext steps:")
    print("  1. Dynamic analysis → data/sandbox_logs/<sha256>.json")
    print("  2. python scripts/02_extract_features.py")
    print("  3. python scripts/03_train_model.py --device cuda")


if __name__ == "__main__":
    main()
