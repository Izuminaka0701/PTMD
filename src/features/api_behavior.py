"""
Stream 2 — API Call Behavior sequence from sandbox dynamic analysis logs.

Input format (JSON):
{
  "api_calls": [
    {"timestamp": 0.1, "api": "LoadLibraryA", "module": "kernel32.dll", "args": ["user32.dll"]},
    {"timestamp": 0.2, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["MessageBoxA"]},
    ...
  ]
}

Phát hiện behavioral patterns:
  G1: LoadLibrary → GetProcAddress chains
  G2: NtQuery* / memory reads without prior LoadLibrary
  G3: VirtualProtect + indirect calls, hash-like constants in args
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.hashing import HashAlgorithms

# Vocabulary categories for embedding
API_CATEGORIES = {
    "loader": {"LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW", "LdrLoadDll"},
    "resolver": {"GetProcAddress", "LdrGetProcedureAddress", "GetModuleHandleA", "GetModuleHandleW"},
    "memory": {"VirtualAlloc", "VirtualAllocEx", "VirtualProtect", "VirtualProtectEx", "HeapAlloc"},
    "process": {"CreateProcessA", "CreateProcessW", "WriteProcessMemory", "CreateRemoteThread"},
    "ntdll": {"NtQueryInformationProcess", "NtQuerySystemInformation", "NtWriteVirtualMemory"},
    "crypto": {"CryptDecrypt", "CryptEncrypt", "BCryptDecrypt"},
    "network": {"InternetOpenA", "HttpSendRequestA", "WSAStartup", "connect"},
    "file": {"CreateFileA", "WriteFile", "ReadFile", "DeleteFileA"},
}

CATEGORY_LIST = list(API_CATEGORIES.keys()) + ["other"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}


class APIBehaviorExtractor:
    """Convert sandbox API log into fixed-length behavior tensor."""

    def __init__(self, max_seq_len: int = 512, embed_dim: int = 128):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self._build_api_vocab()

    def _build_api_vocab(self) -> None:
        common_apis = set()
        for apis in API_CATEGORIES.values():
            common_apis.update(apis)
        common_apis.update(G1_CHAIN_APIS := {
            "LoadLibraryA", "GetProcAddress", "VirtualAlloc", "VirtualProtect",
            "CreateThread", "WriteProcessMemory", "NtQueryInformationProcess",
        })
        self.api_vocab = sorted(common_apis)
        self.api_to_idx = {a: i + 1 for i, a in enumerate(self.api_vocab)}  # 0 = PAD
        self.vocab_size = len(self.api_vocab) + 1

    def extract_from_log(self, log_path: str | Path) -> dict[str, Any]:
        from src.utils.io import load_json

        log_path = Path(log_path)
        data = load_json(log_path)
        calls = data.get("api_calls", data if isinstance(data, list) else [])
        return self.extract(calls)

    def extract(self, api_calls: list[dict[str, Any]]) -> dict[str, Any]:
        seq_indices: list[int] = []
        seq_features: list[list[float]] = []
        timestamps: list[float] = []

        for call in api_calls[: self.max_seq_len]:
            api = call.get("api", call.get("function", "unknown"))
            module = call.get("module", call.get("dll", ""))
            ts = float(call.get("timestamp", call.get("time", 0.0)))
            args = call.get("args", [])

            idx = self.api_to_idx.get(api, 0)
            cat_idx = self._categorize(api)
            hash_feats = HashAlgorithms.hash_feature_vector(api) if api != "unknown" else [0.0] * 6

            feat = [
                float(idx) / max(self.vocab_size, 1),
                cat_idx / len(CATEGORY_LIST),
                float("kernel32" in module.lower()),
                float("ntdll" in module.lower()),
                float(self._looks_like_hash_arg(args)),
                float(self._is_indirect_pattern(api, args)),
                ts,
            ] + hash_feats

            seq_indices.append(idx)
            seq_features.append(feat)
            timestamps.append(ts)

        length = len(seq_indices)
        pad_len = self.max_seq_len - length

        indices_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        features_arr = np.zeros((self.max_seq_len, len(seq_features[0]) if seq_features else 13), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        if length > 0:
            indices_arr[:length] = seq_indices
            features_arr[:length] = seq_features
            mask[:length] = 1.0

        behavior_stats = self._compute_behavior_stats(api_calls)

        return {
            "api_indices": indices_arr,
            "api_features": features_arr,
            "attention_mask": mask,
            "seq_length": length,
            "behavior_stats": behavior_stats,
            "patterns": self._detect_patterns(api_calls),
        }

    def _categorize(self, api: str) -> int:
        for cat, apis in API_CATEGORIES.items():
            if api in apis:
                return CATEGORY_TO_IDX[cat]
        return CATEGORY_TO_IDX["other"]

    @staticmethod
    def _looks_like_hash_arg(args: list[Any]) -> bool:
        for a in args:
            s = str(a)
            if re.match(r"^0x[0-9a-fA-F]{6,}$", s):
                return True
            try:
                v = int(s, 0)
                if v > 0xFFFF:
                    return True
            except ValueError:
                pass
        return False

    @staticmethod
    def _is_indirect_pattern(api: str, args: list[Any]) -> float:
        if api in ("VirtualProtect", "VirtualAlloc") and args:
            return 1.0
        return 0.0

    def _compute_behavior_stats(self, calls: list[dict]) -> dict[str, float]:
        apis = [c.get("api", c.get("function", "")) for c in calls]
        n = max(len(apis), 1)

        g1_chain = self._count_g1_chains(calls)
        g2_peb = sum(1 for a in apis if a in API_CATEGORIES["ntdll"]) / n
        g3_hash_args = sum(1 for c in calls if self._looks_like_hash_arg(c.get("args", []))) / n

        return {
            "g1_chain_ratio": g1_chain / n,
            "g2_peb_ratio": g2_peb,
            "g3_hash_arg_ratio": g3_hash_args,
            "loader_count": sum(1 for a in apis if a in API_CATEGORIES["loader"]) / n,
            "resolver_count": sum(1 for a in apis if a in API_CATEGORIES["resolver"]) / n,
            "total_calls": float(len(calls)),
        }

    @staticmethod
    def _count_g1_chains(calls: list[dict]) -> int:
        chains = 0
        for i in range(len(calls) - 1):
            a1 = calls[i].get("api", "")
            a2 = calls[i + 1].get("api", "")
            if a1 in API_CATEGORIES["loader"] and a2 in API_CATEGORIES["resolver"]:
                chains += 1
        return chains

    def _detect_patterns(self, calls: list[dict]) -> list[str]:
        patterns: list[str] = []
        stats = self._compute_behavior_stats(calls)

        if stats["g1_chain_ratio"] > 0.05:
            patterns.append("G1_LoadLibrary_GetProcAddress_chain")
        if stats["g2_peb_ratio"] > 0.02:
            patterns.append("G2_PEB_NtQuery_pattern")
        if stats["g3_hash_arg_ratio"] > 0.03:
            patterns.append("G3_hash_constant_in_args")
        if stats["loader_count"] > 0 and stats["resolver_count"] == 0:
            patterns.append("G2_no_standard_resolver")

        return patterns
