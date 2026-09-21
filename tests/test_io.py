"""Unit tests for I/O utilities — UTF-8 handling and json round-trip."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestLoadJson:
    """Test load_json handles UTF-8 correctly (Bug #2 — Unicode decode)."""

    def test_roundtrip_utf8(self, tmp_path: Path):
        from src.utils.io import load_json, save_json

        data = {
            "name": "Phân tích Mã độc",
            "family": "CHIMNEYSWEEP — S1149",
            "unicode_chars": "日本語テスト 🔬",
            "nested": {"key": "giá trị có dấu"},
        }

        out = tmp_path / "test_utf8.json"
        save_json(data, out)

        # Verify file is actually UTF-8
        raw = out.read_bytes()
        assert b"Ph\xc3\xa2n t\xc3\xadch" in raw  # UTF-8 encoded Vietnamese

        loaded = load_json(out)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        from src.utils.io import save_json

        deep = tmp_path / "a" / "b" / "c" / "test.json"
        save_json({"ok": True}, deep)
        assert deep.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path):
        from src.utils.io import load_json

        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nope.json")

    def test_save_ensure_ascii_false(self, tmp_path: Path):
        """Verify ensure_ascii=False so Unicode chars are preserved as-is."""
        from src.utils.io import save_json

        data = {"msg": "Cảnh báo"}
        out = tmp_path / "test.json"
        save_json(data, out)

        text = out.read_text(encoding="utf-8")
        assert "Cảnh báo" in text  # Not escaped as \\uXXXX


class TestSetSeed:
    def test_reproducible_random(self):
        import random

        from src.utils.io import set_seed

        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b
