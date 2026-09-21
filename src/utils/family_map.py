"""Unified MITRE family name → id + G1/G2/G3 mapping."""

from __future__ import annotations

from src.utils.io import load_json
from src.config import ROOT

# Alias cho tên family trong MalwareBazaar / repo dataset
FAMILY_ALIASES: dict[str, str] = {
    "Raccoon": "Raccoon Stealer",
    "CobaltStrike": "Cobalt Strike",
    "Cobalt Strike Beacon": "Cobalt Strike",
    "BruteRatel": "Brute Ratel C4",
    "BazarLoader": "Bazar",
    "LODEINFO": "LODEINFO",
    "Lazarus": "Lazarus Group",
}

# MalwareBazaar search tags
FAMILY_TAGS: dict[str, list[str]] = {
    "AvosLocker": ["AvosLocker", "avos"],
    "Bazar": ["BazarLoader", "bazar"],
    "Brute Ratel C4": ["BruteRatel", "brute-ratel"],
    "CANONSTAGER": ["CANONSTAGER"],
    "CHIMNEYSWEEP": ["CHIMNEYSWEEP"],
    "CLAIMLOADER": ["CLAIMLOADER"],
    "HiddenFace": ["HiddenFace"],
    "HTTPTroy": ["HTTPTroy"],
    "Latrodectus": ["Latrodectus", "unidentified_111"],
    "LODEINFO": ["LODEINFO", "lodeinfo"],
    "PlugX": ["PlugX", "plugx"],
    "Raccoon Stealer": ["Raccoon", "raccoon"],
    "Samurai": ["Samurai", "samurai"],
    "SplatDropper": ["SplatDropper"],
    "TONESHELL": ["TONESHELL"],
    "Cobalt Strike": ["CobaltStrike", "cobalt-strike"],
    "WannaCry": ["WannaCry", "wannacry"],
}


def load_family_registry() -> dict[str, dict]:
    """name → {id, group, hash_algorithm, type}"""
    meta = load_json(ROOT / "data" / "metadata" / "mitre_families.json")
    registry: dict[str, dict] = {}
    for fam in meta["families"]:
        registry[fam["name"]] = {
            "id": fam["id"],
            "group": fam["group"],
            "hash_algorithm": fam.get("hash_algorithm"),
            "type": fam.get("type"),
        }
    return registry


# APT groups / aliases không có trong mitre_families.json
APT_FALLBACK: dict[str, dict] = {
    "Lazarus": {"id": "G0032", "group": "G2", "type": "APT"},
    "Lazarus Group": {"id": "G0032", "group": "G2", "type": "APT"},
    "Kimsuky": {"id": "G0094", "group": "G3", "type": "APT"},
    "Mustang Panda": {"id": "G0129", "group": "G2", "type": "APT"},
}


def resolve_family(name: str) -> dict | None:
    registry = load_family_registry()
    canonical = FAMILY_ALIASES.get(name, name)
    if canonical in registry:
        return registry[canonical]
    return APT_FALLBACK.get(name) or APT_FALLBACK.get(canonical)
