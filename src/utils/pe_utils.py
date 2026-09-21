"""PE sample discovery, zip extraction, DAR heuristic verification."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

ZIP_PASSWORD = b"infected"
PE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr", ".cpl", ".bin", ".vir"}

G1_RESOLVER_APIS = {
    "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW",
    "GetProcAddress", "LdrLoadDll", "LdrGetProcedureAddress",
}


def find_sample_file(samples_dir: Path, sha256: str, filename: str = "") -> Path | None:
    """Tìm file sample theo sha256 hoặc filename."""
    samples_dir = Path(samples_dir)
    candidates: list[Path] = []

    if filename:
        candidates.append(samples_dir / filename)
        candidates.append(samples_dir / Path(filename).name)

    candidates.extend([
        samples_dir / f"{sha256}.exe",
        samples_dir / f"{sha256}.dll",
        samples_dir / f"{sha256}.bin",
        samples_dir / f"{sha256}.vir.zip",
        samples_dir / f"{sha256}.zip",
    ])

    for c in candidates:
        if c.exists() and c.is_file():
            if c.suffix.lower() in (".zip",):
                extracted = extract_zip_sample(c, samples_dir, sha256)
                return extracted or c
            return c

    for c in sorted(samples_dir.glob(f"{sha256}*")):
        if c.is_file():
            if c.suffix.lower() in (".zip",):
                extracted = extract_zip_sample(c, samples_dir, sha256)
                return extracted or c
            return c
    return None


def extract_zip_sample(zip_path: Path, output_dir: Path, sha256: str) -> Path | None:
    """Giải nén .vir.zip / .zip (password: infected)."""
    output_dir = Path(output_dir)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                return None
            inner = names[0]
            try:
                data = zf.read(inner, pwd=ZIP_PASSWORD)
            except RuntimeError:
                data = zf.read(inner)

            out = output_dir / f"{sha256}{Path(inner).suffix or '.bin'}"
            out.write_bytes(data)
            return out
    except (zipfile.BadZipFile, OSError):
        return None


def is_pe_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def verify_dynamic_api_resolution(pe_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """
    Heuristic DAR (giống malware-api-dataset/verify_samples.py):
    IAT sparse (<20 imports) + LoadLibrary/GetProcAddress.
    """
    import pefile

    pe = pefile.PE(str(pe_path), fast_load=True)
    pe.parse_data_directories()

    imports: list[tuple[str, str]] = []
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode(errors="ignore") if entry.dll else "unknown"
            for imp in entry.imports or []:
                if imp.name:
                    imports.append((dll, imp.name.decode(errors="ignore")))

    api_names = {a for _, a in imports}
    has_loadlib = any("LoadLibrary" in a for a in api_names)
    has_getproc = "GetProcAddress" in api_names
    total = len(imports)
    pe.close()

    meta = {
        "total_imports": total,
        "has_loadlibrary": has_loadlib,
        "has_getprocaddress": has_getproc,
    }

    if total < 20 and has_loadlib and has_getproc:
        return True, f"IAT sparse ({total} imports) + runtime resolution", meta
    return False, f"Total imports: {total}", meta
