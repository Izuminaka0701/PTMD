"""Unit tests for APIBehaviorExtractor."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_G1_CALLS = [
    {"timestamp": 0.01, "api": "LoadLibraryA", "module": "kernel32.dll", "args": ["user32.dll"]},
    {"timestamp": 0.02, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["MessageBoxA"]},
    {"timestamp": 0.03, "api": "LoadLibraryW", "module": "kernel32.dll", "args": ["advapi32.dll"]},
    {"timestamp": 0.04, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["RegOpenKeyExA"]},
]

SAMPLE_G3_CALLS = [
    {"timestamp": 0.01, "api": "VirtualAlloc", "module": "kernel32.dll", "args": ["0x00000000", "0x1000"]},
    {"timestamp": 0.02, "api": "VirtualProtect", "module": "kernel32.dll", "args": ["0xDEADBEEF", "0x1000"]},
    {"timestamp": 0.03, "api": "CreateThread", "module": "kernel32.dll", "args": ["0x00400000"]},
]

EMPTY_CALLS: list[dict] = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAPIBehaviorExtractor:

    def test_extract_g1_chain_detected(self):
        from src.features.api_behavior import APIBehaviorExtractor

        ext = APIBehaviorExtractor(max_seq_len=64, embed_dim=32)
        result = ext.extract(SAMPLE_G1_CALLS)

        assert result["seq_length"] == 4
        stats = result["behavior_stats"]
        assert stats["g1_chain_ratio"] > 0, "Should detect LoadLibrary→GetProcAddress chains"
        assert "G1_LoadLibrary_GetProcAddress_chain" in result["patterns"]

    def test_extract_empty_calls(self):
        from src.features.api_behavior import APIBehaviorExtractor

        ext = APIBehaviorExtractor(max_seq_len=64, embed_dim=32)
        result = ext.extract(EMPTY_CALLS)

        assert result["seq_length"] == 0
        assert isinstance(result["api_indices"], np.ndarray)
        assert result["api_indices"].shape == (64,)
        assert np.all(result["api_indices"] == 0)

    def test_extract_returns_required_keys(self):
        from src.features.api_behavior import APIBehaviorExtractor

        ext = APIBehaviorExtractor(max_seq_len=64, embed_dim=32)
        result = ext.extract(SAMPLE_G1_CALLS)

        required = {"api_indices", "api_features", "attention_mask", "seq_length",
                     "behavior_stats", "patterns"}
        assert required <= set(result.keys())

    def test_attention_mask_correct(self):
        from src.features.api_behavior import APIBehaviorExtractor

        ext = APIBehaviorExtractor(max_seq_len=64, embed_dim=32)
        result = ext.extract(SAMPLE_G1_CALLS)

        mask = result["attention_mask"]
        assert mask[:4].sum() == 4.0, "First 4 positions should be 1"
        assert mask[4:].sum() == 0.0, "Remaining positions should be 0"

    def test_hash_arg_detection(self):
        from src.features.api_behavior import APIBehaviorExtractor

        assert APIBehaviorExtractor._looks_like_hash_arg(["0xDEADBEEF"]) is True
        assert APIBehaviorExtractor._looks_like_hash_arg(["hello"]) is False
        assert APIBehaviorExtractor._looks_like_hash_arg([]) is False

    def test_categorize(self):
        from src.features.api_behavior import APIBehaviorExtractor, CATEGORY_TO_IDX

        ext = APIBehaviorExtractor(max_seq_len=16, embed_dim=16)
        assert ext._categorize("LoadLibraryA") == CATEGORY_TO_IDX["loader"]
        assert ext._categorize("GetProcAddress") == CATEGORY_TO_IDX["resolver"]
        assert ext._categorize("RandomUnknownAPI") == CATEGORY_TO_IDX["other"]

    def test_g3_hash_pattern(self):
        from src.features.api_behavior import APIBehaviorExtractor

        ext = APIBehaviorExtractor(max_seq_len=64, embed_dim=32)
        result = ext.extract(SAMPLE_G3_CALLS)

        stats = result["behavior_stats"]
        assert stats["g3_hash_arg_ratio"] > 0, "Should detect hash-like args"

    def test_max_seq_len_truncation(self):
        """Verify sequences longer than max_seq_len are truncated."""
        from src.features.api_behavior import APIBehaviorExtractor

        ext = APIBehaviorExtractor(max_seq_len=2, embed_dim=16)
        result = ext.extract(SAMPLE_G1_CALLS)

        assert result["seq_length"] == 2
        assert result["api_indices"].shape == (2,)
