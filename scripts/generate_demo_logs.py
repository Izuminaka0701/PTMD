#!/usr/bin/env python3
"""
Tạo sandbox logs mô phỏng theo pattern G1/G2/G3 cho từng MITRE family.
Dùng để test pipeline khi chưa có dynamic analysis thật.

Usage (trên VM):
  python scripts/generate_demo_logs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.io import load_json, save_json

# Behavioral templates theo nhóm kỹ thuật
TEMPLATES = {
    "G1": [
        {"timestamp": 0.01, "api": "LoadLibraryA", "module": "kernel32.dll", "args": ["advapi32.dll"]},
        {"timestamp": 0.02, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["RegOpenKeyExA"]},
        {"timestamp": 0.03, "api": "LoadLibraryW", "module": "kernel32.dll", "args": ["ws2_32.dll"]},
        {"timestamp": 0.04, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["WSAStartup"]},
        {"timestamp": 0.05, "api": "VirtualAlloc", "module": "kernel32.dll", "args": ["0", "4096", "0x3000", "0x04"]},
        {"timestamp": 0.06, "api": "CreateThread", "module": "kernel32.dll", "args": []},
    ],
    "G2": [
        {"timestamp": 0.01, "api": "NtQueryInformationProcess", "module": "ntdll.dll", "args": ["-1", "0", "0", "0", "0"]},
        {"timestamp": 0.02, "api": "VirtualAlloc", "module": "kernel32.dll", "args": ["0", "8192", "0x3000", "0x40"]},
        {"timestamp": 0.03, "api": "VirtualProtect", "module": "kernel32.dll", "args": ["0x10000000", "8192", "0x20"]},
        {"timestamp": 0.04, "api": "NtQuerySystemInformation", "module": "ntdll.dll", "args": ["11", "0", "0", "0"]},
        {"timestamp": 0.05, "api": "CreateThread", "module": "kernel32.dll", "args": []},
        {"timestamp": 0.06, "api": "WriteProcessMemory", "module": "kernel32.dll", "args": []},
    ],
    "G3": [
        {"timestamp": 0.01, "api": "VirtualAlloc", "module": "kernel32.dll", "args": ["0", "4096", "0x3000", "0x40"]},
        {"timestamp": 0.02, "api": "VirtualProtect", "module": "kernel32.dll", "args": ["0x10000000", "4096", "0x20"]},
        {"timestamp": 0.03, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["0x7FFE0000", "0x6A4ABC12"]},
        {"timestamp": 0.04, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["0x7FFE0000", "0x91F2D8E3"]},
        {"timestamp": 0.05, "api": "LdrLoadDll", "module": "ntdll.dll", "args": ["0xDEADBEEF"]},
        {"timestamp": 0.06, "api": "CreateRemoteThread", "module": "kernel32.dll", "args": []},
    ],
}

# G3 variants theo hash algorithm
G3_VARIANTS = {
    "DJB2 variant": [{"timestamp": 0.03, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["0x7FFE0000", "0x8FECDFFB"]}],
    "XOR": [{"timestamp": 0.03, "api": "LdrGetProcedureAddress", "module": "ntdll.dll", "args": ["0xAA", "0xBB"]}],
    "Custom seed 131313": [{"timestamp": 0.03, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["0x7FFE0000", "0x00131313"]}],
    "ROR13": [{"timestamp": 0.02, "api": "NtQueryInformationProcess", "module": "ntdll.dll", "args": []}],
}


def generate_log(group: str, family_name: str, hash_algo: str | None) -> dict:
    base = list(TEMPLATES.get(group, TEMPLATES["G1"]))
    if group == "G3" and hash_algo:
        for key, extra in G3_VARIANTS.items():
            if key.lower() in (hash_algo or "").lower():
                base.extend(extra)
                break
    return {
        "sample_family": family_name,
        "technique_group": group,
        "sandbox": "synthetic_demo",
        "api_calls": base,
    }


def main() -> None:
    cfg = load_config()
    meta = load_json(cfg["paths"]["metadata_file"])
    logs_dir = Path(cfg["paths"]["sandbox_logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["paths"]["manifest_file"])
    manifest = load_json(manifest_path) if manifest_path.exists() else {"samples": [], "benign_samples": []}

    for fam in meta["families"]:
        sha = f"demo_{fam['id'].lower()}"
        log = generate_log(fam["group"], fam["name"], fam.get("hash_algorithm"))
        log_path = logs_dir / f"{sha}.json"
        save_json(log, log_path)

        manifest["samples"] = [s for s in manifest.get("samples", []) if s.get("sha256") != sha]
        manifest["samples"].append({
            "sha256": sha,
            "filename": f"{sha}.exe",
            "family_id": fam["id"],
            "family_name": fam["name"],
            "group": fam["group"],
            "source": "synthetic_demo",
            "sandbox_log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
            "notes": "Demo log — thay bằng sandbox log thật sau khi dynamic analysis",
        })

    save_json(manifest, manifest_path)
    print(f"Generated {len(meta['families'])} demo logs → {logs_dir}")
    print(f"Updated manifest → {manifest_path}")
    print("\nLưu ý: Cần đặt file PE tương ứng vào data/samples/demo_s1149.exe, ...")
    print("       Hoặc thay sha256 bằng hash thật sau khi tải từ MalwareBazaar.")


if __name__ == "__main__":
    main()
