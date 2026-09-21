#!/usr/bin/env python3
"""Kiểm tra MalwareBazaar API key (không tải malware)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, get_api_key
from src.integrations.malwarebazaar import MalwareBazaarClient, MalwareBazaarError


def main() -> None:
    cfg = load_config()
    key = get_api_key()

    if not key:
        print("FAIL: ABUSECH_API_KEY chưa được cấu hình trong .env")
        sys.exit(1)

    print(f"API key: {key[:8]}...{key[-4:]} ({len(key)} chars)")
    print(f"API URL: {cfg['api']['malwarebazaar_url']}")

    try:
        client = MalwareBazaarClient()
        results = client.get_taginfo("PlugX", limit=1)
        if results:
            sample = results[0]
            print(f"OK: API hoạt động — test query PlugX")
            print(f"  SHA256: {sample.get('sha256_hash', '?')[:32]}...")
            print(f"  File:   {sample.get('file_name', '?')}")
        else:
            print("OK: API phản hồi (không có sample cho tag PlugX)")
    except MalwareBazaarError as ex:
        print(f"FAIL: {ex}")
        sys.exit(1)
    except Exception as ex:
        print(f"FAIL: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
