"""
Unified Feature Extractor — gộp 3 nguồn features thành structured groups cho MCAFF.

Vẫn giữ nguyên logic trích xuất từ 3 nguồn (StaticPE, APIBehavior, APIGraph),
nhưng output là dict chứa 3 feature groups:
  - static_x:   [64] — PE/IAT features
  - behavior_x: [15] — Behavior statistics + pattern flags + derived features
  - graph_x:    [5]  — Graph topology statistics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.features.api_behavior import APIBehaviorExtractor
from src.features.api_graph import APIGraphBuilder
from src.features.static_pe import StaticPEExtractor


# Behavior pattern flags
BEHAVIOR_PATTERNS = [
    "G1_LoadLibrary_GetProcAddress_chain",
    "G2_PEB_NtQuery_pattern",
    "G3_hash_constant_in_args",
    "G2_no_standard_resolver",
]


class UnifiedFeatureExtractor:
    """Extract and combine features from PE file + sandbox log into structured groups."""

    # Feature dimensions for each group
    STATIC_DIM = 64
    BEHAVIOR_DIM = 15
    GRAPH_DIM = 5

    def __init__(
        self,
        static_feature_dim: int = 64,
        behavior_max_seq_len: int = 512,
        behavior_embed_dim: int = 128,
        graph_max_nodes: int = 256,
        graph_node_feature_dim: int = 32,
    ):
        self.static_ext = StaticPEExtractor(feature_dim=static_feature_dim)
        self.behavior_ext = APIBehaviorExtractor(
            max_seq_len=behavior_max_seq_len,
            embed_dim=behavior_embed_dim,
        )
        self.graph_builder = APIGraphBuilder(
            max_nodes=graph_max_nodes,
            node_feature_dim=graph_node_feature_dim,
        )
        self.STATIC_DIM = static_feature_dim

    def extract(
        self,
        pe_path: str | Path | None = None,
        api_calls: list[dict[str, Any]] | None = None,
        static_data: dict[str, Any] | None = None,
        behavior_data: dict[str, Any] | None = None,
        graph_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Extract unified features from PE file and/or API call log.

        Can also accept pre-extracted data from individual extractors.

        Returns:
            dict with keys:
              - static_x:   np.ndarray [static_dim]
              - behavior_x: np.ndarray [15]
              - graph_x:    np.ndarray [5]
              - metadata:   dict with extraction details
        """
        # Extract from sources if not provided
        if static_data is None and pe_path is not None:
            static_data = self.static_ext.extract(pe_path)

        if behavior_data is None:
            behavior_data = self.behavior_ext.extract(api_calls or [])

        if graph_data is None:
            graph_data = self.graph_builder.build(
                static_data=static_data,
                api_calls=api_calls or [],
            )

        # Build feature groups
        static_x = self._build_static_features(static_data)
        behavior_x = self._build_behavior_features(behavior_data)
        graph_x = self._build_graph_features(graph_data)

        return {
            "static_x": static_x,
            "behavior_x": behavior_x,
            "graph_x": graph_x,
            "metadata": {
                "static_dim": len(static_x),
                "behavior_dim": len(behavior_x),
                "graph_dim": len(graph_x),
                "has_pe": pe_path is not None,
                "has_api_calls": bool(api_calls),
                "num_api_calls": len(api_calls) if api_calls else 0,
            },
        }

    def _build_static_features(self, static_data: dict[str, Any] | None) -> np.ndarray:
        """Static PE features — feature_vector từ StaticPEExtractor."""
        if static_data is None:
            return np.zeros(self.STATIC_DIM, dtype=np.float32)
        vec = static_data["feature_vector"]
        if isinstance(vec, list):
            vec = np.array(vec, dtype=np.float32)
        return vec.astype(np.float32)

    def _build_behavior_features(self, behavior_data: dict[str, Any] | None) -> np.ndarray:
        """
        Behavior features — gộp behavior_stats + pattern flags + derived.

        Layout [15 dims]:
          [0-5]   behavior_stats: g1_chain_ratio, g2_peb_ratio, g3_hash_arg_ratio,
                                  loader_count, resolver_count, total_calls (normalized)
          [6-9]   pattern_flags:  4 binary flags cho detected patterns
          [10-14] derived:        loader_resolver_ratio, g1_vs_g2_diff, g1_vs_g3_diff,
                                  max_group_ratio, has_any_pattern
        """
        if behavior_data is None:
            return np.zeros(self.BEHAVIOR_DIM, dtype=np.float32)

        stats = behavior_data.get("behavior_stats", {})
        patterns = behavior_data.get("patterns", [])

        # [0-5] Behavior statistics
        g1_chain = stats.get("g1_chain_ratio", 0.0)
        g2_peb = stats.get("g2_peb_ratio", 0.0)
        g3_hash = stats.get("g3_hash_arg_ratio", 0.0)
        loader = stats.get("loader_count", 0.0)
        resolver = stats.get("resolver_count", 0.0)
        total = stats.get("total_calls", 0.0)
        total_norm = min(total / 1000.0, 1.0)  # Normalize

        # [6-9] Pattern flags
        pattern_set = set(patterns)
        pattern_flags = [
            float(p in pattern_set) for p in BEHAVIOR_PATTERNS
        ]

        # [10-14] Derived features
        loader_resolver_ratio = loader / max(resolver, 1e-6) if resolver > 0 else loader
        loader_resolver_ratio = min(loader_resolver_ratio, 10.0) / 10.0  # Normalize

        g1_vs_g2 = g1_chain - g2_peb
        g1_vs_g3 = g1_chain - g3_hash
        max_group = max(g1_chain, g2_peb, g3_hash)
        has_any_pattern = float(len(patterns) > 0)

        vec = [
            g1_chain, g2_peb, g3_hash, loader, resolver, total_norm,
            *pattern_flags,
            loader_resolver_ratio, g1_vs_g2, g1_vs_g3, max_group, has_any_pattern,
        ]

        return np.array(vec, dtype=np.float32)

    def _build_graph_features(self, graph_data: dict[str, Any] | None) -> np.ndarray:
        """
        Graph-derived features — topology statistics từ APIGraphBuilder.

        Layout [5 dims]:
          num_modules, num_apis, num_hash_nodes, num_resolves, num_hash_edges
          (tất cả normalized)
        """
        if graph_data is None:
            return np.zeros(self.GRAPH_DIM, dtype=np.float32)

        stats = graph_data.get("graph_stats", {})

        vec = [
            min(stats.get("num_modules", 0) / 50.0, 1.0),
            min(stats.get("num_apis", 0) / 100.0, 1.0),
            min(stats.get("num_hash_nodes", 0) / 20.0, 1.0),
            min(stats.get("num_resolves", 0) / 50.0, 1.0),
            min(stats.get("num_hash_edges", 0) / 20.0, 1.0),
        ]

        return np.array(vec, dtype=np.float32)

    @classmethod
    def get_feature_dims(cls) -> dict[str, int]:
        """Return dimension of each feature group."""
        return {
            "static": cls.STATIC_DIM,
            "behavior": cls.BEHAVIOR_DIM,
            "graph": cls.GRAPH_DIM,
            "total": cls.STATIC_DIM + cls.BEHAVIOR_DIM + cls.GRAPH_DIM,
        }
