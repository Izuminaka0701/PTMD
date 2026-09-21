"""Load YAML configuration and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Load .env nếu có (python-dotenv optional)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    _env_file = ROOT / ".env"
    if _env_file.exists():
        for line in _env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def get_api_key() -> str:
    return os.environ.get("ABUSECH_API_KEY", os.environ.get("MALWAREBAZAAR_API_KEY", ""))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in (
        "samples_dir", "sandbox_logs_dir", "processed_dir",
        "metadata_file", "manifest_file", "checkpoint_dir",
    ):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)

    cfg["api"] = {
        "malwarebazaar_key_set": bool(get_api_key()),
        "malwarebazaar_url": cfg.get("sample_sources", {}).get("malwarebazaar", {}).get(
            "api_url", "https://mb-api.abuse.ch/api/v1/"
        ),
    }
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])
