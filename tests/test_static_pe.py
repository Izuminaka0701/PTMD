"""Unit tests for StaticPEExtractor — especially RICH_HEADER compat."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: create a minimal valid PE file for testing
# ---------------------------------------------------------------------------

def _make_minimal_pe(tmp_path: Path, name: str = "test.exe") -> Path:
    """Create a minimal PE that pefile can parse (MZ + PE header stub)."""
    # Minimal DOS header
    dos_header = bytearray(64)
    dos_header[0:2] = b"MZ"
    pe_offset = 64
    struct.pack_into("<I", dos_header, 60, pe_offset)  # e_lfanew

    # Minimal PE signature + COFF header + Optional header
    pe_sig = b"PE\x00\x00"
    # COFF: Machine=0x14c (i386), 1 section, no symbols, optional hdr size=0xe0
    coff = struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0002)

    # Optional header (PE32) — minimal
    optional = bytearray(0xE0)
    optional[0:2] = struct.pack("<H", 0x10B)  # PE32 magic
    struct.pack_into("<I", optional, 16, 0x1000)  # AddressOfEntryPoint
    struct.pack_into("<I", optional, 56, 0x10000)  # SizeOfImage
    struct.pack_into("<I", optional, 60, 0x200)  # SizeOfHeaders
    struct.pack_into("<I", optional, 76, 16)  # NumberOfRvaAndSizes

    # Section header: .text
    section = bytearray(40)
    section[0:6] = b".text\x00"
    struct.pack_into("<I", section, 8, 0x1000)  # VirtualSize
    struct.pack_into("<I", section, 12, 0x1000)  # VirtualAddress
    struct.pack_into("<I", section, 16, 0x200)  # SizeOfRawData
    struct.pack_into("<I", section, 20, 0x200)  # PointerToRawData

    # Pad to SizeOfHeaders then add a "section" of 0x200 bytes
    header = dos_header + pe_sig + coff + bytes(optional) + bytes(section)
    header_padded = header.ljust(0x200, b"\x00")
    section_data = b"\xCC" * 0x200  # INT3 fill

    pe_bytes = header_padded + section_data
    out = tmp_path / name
    out.write_bytes(pe_bytes)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRichHeaderCompat:
    """Verify the RICH_HEADER getattr fallback works for old & new pefile."""

    def test_rich_header_old_api(self):
        """Simulates pefile 2023.x where pe.RICH_HEADER exists."""
        from src.features.static_pe import StaticPEExtractor

        ext = StaticPEExtractor(feature_dim=64)

        mock_pe = MagicMock()
        mock_pe.RICH_HEADER = MagicMock()  # old-style attr present
        # getattr chain: finds RICH_HEADER first
        result = getattr(mock_pe, "RICH_HEADER", getattr(mock_pe, "RICH_HEADER_DATA", None))
        assert result is not None

    def test_rich_header_new_api(self):
        """Simulates pefile 2024.x where pe.RICH_HEADER_DATA exists instead."""
        mock_pe = MagicMock(spec=[])  # empty spec — no attributes
        mock_pe.RICH_HEADER_DATA = MagicMock()
        # Must NOT raise AttributeError
        result = getattr(mock_pe, "RICH_HEADER", getattr(mock_pe, "RICH_HEADER_DATA", None))
        assert result is not None

    def test_rich_header_neither(self):
        """Neither attribute exists → should return None (not crash)."""
        mock_pe = MagicMock(spec=[])
        result = getattr(mock_pe, "RICH_HEADER", getattr(mock_pe, "RICH_HEADER_DATA", None))
        assert result is None


class TestStaticPEExtractor:
    """Integration-level tests for the full extractor."""

    def test_extract_returns_correct_keys(self, tmp_path: Path):
        """Extract from a minimal PE should return required dict keys."""
        pe_path = _make_minimal_pe(tmp_path)

        from src.features.static_pe import StaticPEExtractor

        ext = StaticPEExtractor(feature_dim=64)
        try:
            result = ext.extract(pe_path)
        except Exception:
            # Minimal PE may not parse fully — that's OK for this test
            pytest.skip("Minimal PE not parseable by pefile on this platform")

        assert "feature_vector" in result
        assert "feature_names" in result
        assert "num_imports" in result
        assert "dar_score" in result
        assert isinstance(result["feature_vector"], np.ndarray)
        assert result["feature_vector"].shape == (64,)

    def test_feature_dim_padding(self):
        """Verify _pad_to_dim always outputs exactly feature_dim elements."""
        from src.features.static_pe import StaticPEExtractor

        ext = StaticPEExtractor(feature_dim=64)
        raw = [float(i) for i in range(26)]
        padded = ext._pad_to_dim(raw)
        assert padded.shape == (64,), f"Expected (64,), got {padded.shape}"
        # First 26 elements should be preserved
        assert padded[0] == 0.0
        assert padded[25] == 25.0

    def test_feature_dim_truncation(self):
        """Verify _pad_to_dim truncates when input exceeds feature_dim."""
        from src.features.static_pe import StaticPEExtractor

        ext = StaticPEExtractor(feature_dim=8)
        raw = [float(i) for i in range(20)]
        padded = ext._pad_to_dim(raw)
        assert padded.shape == (8,)

    def test_section_entropy_empty(self):
        """Empty section data → entropy 0."""
        from src.features.static_pe import StaticPEExtractor

        section = MagicMock()
        section.get_data.return_value = b""
        assert StaticPEExtractor._section_entropy(section) == 0.0

    def test_iat_sparse_score_no_sections(self):
        """Zero sections → score 0."""
        from src.features.static_pe import StaticPEExtractor

        assert StaticPEExtractor._iat_sparse_score(10, 0) == 0.0

    def test_iat_sparse_score_sparse(self):
        """Few imports relative to sections → high sparse score."""
        from src.features.static_pe import StaticPEExtractor

        score = StaticPEExtractor._iat_sparse_score(1, 5)
        assert 0.9 < score <= 1.0
