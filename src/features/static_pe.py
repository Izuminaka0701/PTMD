"""
Stream 1 — Static PE / IAT features for Dynamic API Resolution detection.

Phát hiện dấu hiệu:
  G1: LoadLibrary/GetProcAddress trong IAT
  G2: Sparse imports, thiếu kernel32 exports trực tiếp
  G3: Entropy cao, ít import, suspicious sections
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pefile
except ImportError:
    pefile = None  # type: ignore

# APIs liên quan trực tiếp đến dynamic resolution
G1_APIS = {
    "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW",
    "GetProcAddress", "GetModuleHandleA", "GetModuleHandleW",
    "LdrLoadDll", "LdrGetProcedureAddress",
}
G2_INDICATOR_APIS = {
    "NtQueryInformationProcess", "NtQuerySystemInformation",
    "RtlGetVersion", "ZwQueryInformationProcess",
}
G3_SUSPICIOUS_APIS = {
    "VirtualAlloc", "VirtualAllocEx", "VirtualProtect", "VirtualProtectEx",
    "WriteProcessMemory", "CreateRemoteThread", "NtWriteVirtualMemory",
    "RtlDecompressBuffer", "CryptDecrypt",
}

FEATURE_NAMES = [
    "num_sections", "num_imports", "num_import_dlls",
    "has_loadlibrary", "has_getprocaddress", "has_ldr_load_dll",
    "g1_api_ratio", "g2_indicator_ratio", "g3_suspicious_ratio",
    "text_entropy", "data_entropy", "overall_entropy",
    "import_table_size_ratio", "is_dll", "is_64bit",
    "has_reloc", "has_tls", "num_exports",
    "suspicious_section_count", "iat_sparse_score",
    "missing_kernel32_direct", "entry_point_section_idx",
    "overlay_size_ratio", "rich_header_present",
    "num_resources", "debug_present",
    # padding to 64 dims with derived features
]


class StaticPEExtractor:
    """Extract fixed-size static feature vector from PE file."""

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim

    def extract(self, pe_path: str | Path) -> dict[str, Any]:
        if pefile is None:
            raise ImportError("pefile required: pip install pefile")

        pe_path = Path(pe_path)
        pe = pefile.PE(str(pe_path), fast_load=True)
        pe.parse_data_directories()

        imports = self._collect_imports(pe)
        sections = pe.sections or []

        num_imports = len(imports)
        import_dlls = {dll for dll, _ in imports}
        import_names = {api for _, api in imports}

        g1_hits = sum(1 for a in import_names if a in G1_APIS)
        g2_hits = sum(1 for a in import_names if a in G2_INDICATOR_APIS)
        g3_hits = sum(1 for a in import_names if a in G3_SUSPICIOUS_APIS)

        entropies = [self._section_entropy(s) for s in sections]
        text_ent = entropies[0] if entropies else 0.0
        data_ent = entropies[1] if len(entropies) > 1 else 0.0

        file_size = max(pe_path.stat().st_size, 1)
        overlay = max(0, file_size - pe.OPTIONAL_HEADER.SizeOfImage)

        has_kernel32 = any("kernel32" in d.lower() for d in import_dlls)
        kernel32_direct = any(
            dll.lower().startswith("kernel32") and api in G1_APIS
            for dll, api in imports
        )

        raw = [
            float(len(sections)),
            float(num_imports),
            float(len(import_dlls)),
            float("LoadLibraryA" in import_names or "LoadLibraryW" in import_names),
            float("GetProcAddress" in import_names),
            float("LdrLoadDll" in import_names),
            g1_hits / max(num_imports, 1),
            g2_hits / max(num_imports, 1),
            g3_hits / max(num_imports, 1),
            text_ent,
            data_ent,
            float(np.mean(entropies)) if entropies else 0.0,
            self._import_table_ratio(pe, file_size),
            float(pe.FILE_HEADER.IMAGE_FILE_DLL != 0),
            float(pe.FILE_HEADER.Machine == 0x8664),
            float(hasattr(pe, "DIRECTORY_ENTRY_RELOC")),
            float(hasattr(pe, "DIRECTORY_ENTRY_TLS")),
            float(len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else 0),
            float(sum(1 for e in entropies if e > 7.0)),
            self._iat_sparse_score(num_imports, len(sections)),
            float(has_kernel32 and not kernel32_direct),
            float(self._entry_point_section(pe, sections)),
            overlay / file_size,
            float(getattr(pe, "RICH_HEADER", getattr(pe, "RICH_HEADER_DATA", None)) is not None),
            float(len(pe.DIRECTORY_ENTRY_RESOURCE.entries) if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE") else 0),
            float(hasattr(pe, "DIRECTORY_ENTRY_DEBUG")),
        ]

        # DAR heuristic score (0–1): sparse IAT + runtime resolver APIs
        has_loadlib = bool(import_names & {"LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW"})
        has_getproc = "GetProcAddress" in import_names
        dar_score = 0.0
        if num_imports < 20 and has_loadlib and has_getproc:
            dar_score = 1.0 - min(num_imports / 20.0, 1.0)

        vec = self._pad_to_dim(raw)
        pe.close()

        return {
            "feature_vector": vec,
            "feature_names": FEATURE_NAMES[: len(raw)],
            "num_imports": num_imports,
            "import_dlls": sorted(import_dlls),
            "g1_apis_found": sorted(import_names & G1_APIS),
            "dar_score": dar_score,
            "metadata": {"path": str(pe_path), "sha256_hint": pe_path.stem},
        }

    def _collect_imports(self, pe: Any) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            return result
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode(errors="ignore") if entry.dll else "unknown"
            for imp in entry.imports or []:
                if imp.name:
                    result.append((dll, imp.name.decode(errors="ignore")))
                elif imp.ordinal:
                    result.append((dll, f"ordinal_{imp.ordinal}"))
        return result

    @staticmethod
    def _section_entropy(section: Any) -> float:
        data = section.get_data()
        if not data:
            return 0.0
        freq = np.bincount(np.frombuffer(data[:65536], dtype=np.uint8), minlength=256)
        freq = freq[freq > 0] / freq.sum()
        return float(-np.sum(freq * np.log2(freq)))

    @staticmethod
    def _import_table_ratio(pe: Any, file_size: int) -> float:
        try:
            imp_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[1]  # IMAGE_DIRECTORY_ENTRY_IMPORT
            return imp_dir.Size / max(file_size, 1)
        except Exception:
            return 0.0

    @staticmethod
    def _iat_sparse_score(num_imports: int, num_sections: int) -> float:
        """Few imports relative to sections suggests dynamic resolution (G2/G3)."""
        if num_sections == 0:
            return 0.0
        expected = num_sections * 3
        return float(max(0.0, 1.0 - num_imports / expected))

    @staticmethod
    def _entry_point_section(pe: Any, sections: list) -> int:
        ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        for i, s in enumerate(sections):
            start = s.VirtualAddress
            end = start + s.Misc_VirtualSize
            if start <= ep < end:
                return i
        return -1

    def _pad_to_dim(self, raw: list[float]) -> np.ndarray:
        arr = np.array(raw, dtype=np.float32)
        if len(arr) < self.feature_dim:
            # polynomial interaction terms for extra capacity
            extra = []
            n = len(arr)
            for i in range(min(n, self.feature_dim)):
                for j in range(i + 1, min(i + 6, n)):
                    extra.append(arr[i] * arr[j])
                    if n + len(extra) >= self.feature_dim:
                        break
                if n + len(extra) >= self.feature_dim:
                    break
            arr = np.concatenate([arr, np.array(extra[: self.feature_dim - len(arr)], dtype=np.float32)])
            # zero-pad if still short (fallback guarantee)
            if len(arr) < self.feature_dim:
                arr = np.concatenate([arr, np.zeros(self.feature_dim - len(arr), dtype=np.float32)])
        return arr[: self.feature_dim]
