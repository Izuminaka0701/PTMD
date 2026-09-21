"""
Stream 3 — API Resolution Graph for GNN.

Nodes: modules + resolved API functions
Edges:
  - IMPORTS (module → api)
  - RESOLVES (LoadLibrary/GetProcAddress chain)
  - CALLS (temporal call order from sandbox)
  - HASH_RESOLVE (constant/hash → api inference for G3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.hashing import HashAlgorithms

EDGE_TYPES = ["imports", "resolves", "calls", "hash_resolve"]
EDGE_TYPE_TO_IDX = {e: i for i, e in enumerate(EDGE_TYPES)}

NODE_TYPE_MODULE = 0
NODE_TYPE_API = 1
NODE_TYPE_HASH_CONST = 2


class APIGraphBuilder:
    """Build PyG-compatible graph from static imports + dynamic API log."""

    def __init__(self, max_nodes: int = 256, node_feature_dim: int = 32):
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim

    def build(
        self,
        static_data: dict[str, Any] | None = None,
        behavior_data: dict[str, Any] | None = None,
        api_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[tuple[int, int, int]] = []  # src, dst, edge_type
        node_index: dict[str, int] = {}

        def add_node(key: str, ntype: int, label: str, extra: dict | None = None) -> int:
            if key in node_index:
                return node_index[key]
            if len(nodes) >= self.max_nodes:
                return -1
            idx = len(nodes)
            node_index[key] = idx
            nodes.append({"key": key, "type": ntype, "label": label, **(extra or {})})
            return idx

        # --- Static import edges ---
        if static_data:
            for dll in static_data.get("import_dlls", []):
                mid = add_node(f"dll:{dll}", NODE_TYPE_MODULE, dll)
                for api in static_data.get("g1_apis_found", []):
                    aid = add_node(f"api:{api}", NODE_TYPE_API, api)
                    if mid >= 0 and aid >= 0:
                        edges.append((mid, aid, EDGE_TYPE_TO_IDX["imports"]))

        # --- Dynamic resolution + call edges ---
        if api_calls:
            prev_api_idx = -1
            last_loader = -1

            for call in api_calls:
                api = call.get("api", call.get("function", ""))
                module = call.get("module", call.get("dll", "unknown"))
                args = call.get("args", [])

                if not api:
                    continue

                mid = add_node(f"dll:{module}", NODE_TYPE_MODULE, module)
                aid = add_node(f"api:{api}", NODE_TYPE_API, api)

                if mid >= 0 and aid >= 0:
                    edges.append((mid, aid, EDGE_TYPE_TO_IDX["calls"]))

                if api in ("LoadLibraryA", "LoadLibraryW", "LdrLoadDll"):
                    last_loader = aid
                elif api in ("GetProcAddress", "LdrGetProcedureAddress") and last_loader >= 0:
                    edges.append((last_loader, aid, EDGE_TYPE_TO_IDX["resolves"]))
                    for arg in args:
                        resolved = add_node(f"api:{arg}", NODE_TYPE_API, str(arg))
                        if resolved >= 0:
                            edges.append((aid, resolved, EDGE_TYPE_TO_IDX["resolves"]))

                for arg in args:
                    if self._is_hash_constant(arg):
                        hid = add_node(f"hash:{arg}", NODE_TYPE_HASH_CONST, str(arg))
                        if hid >= 0 and aid >= 0:
                            edges.append((hid, aid, EDGE_TYPE_TO_IDX["hash_resolve"]))

                if prev_api_idx >= 0 and aid >= 0:
                    edges.append((prev_api_idx, aid, EDGE_TYPE_TO_IDX["calls"]))
                prev_api_idx = aid

        # GNN cần ít nhất 1 node
        if not nodes:
            nodes = [{"key": "placeholder", "type": NODE_TYPE_API, "label": "unknown"}]

        x = self._build_node_features(nodes)
        edge_index, edge_attr = self._build_edge_tensors(edges, len(nodes))

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": len(nodes),
            "node_labels": [n["label"] for n in nodes],
            "edge_types": EDGE_TYPES,
            "graph_stats": {
                "num_modules": sum(1 for n in nodes if n["type"] == NODE_TYPE_MODULE),
                "num_apis": sum(1 for n in nodes if n["type"] == NODE_TYPE_API),
                "num_hash_nodes": sum(1 for n in nodes if n["type"] == NODE_TYPE_HASH_CONST),
                "num_resolves": sum(1 for _, _, t in edges if t == EDGE_TYPE_TO_IDX["resolves"]),
                "num_hash_edges": sum(1 for _, _, t in edges if t == EDGE_TYPE_TO_IDX["hash_resolve"]),
            },
        }

    def _build_node_features(self, nodes: list[dict]) -> np.ndarray:
        x = np.zeros((max(len(nodes), 1), self.node_feature_dim), dtype=np.float32)
        for i, node in enumerate(nodes[: self.max_nodes]):
            label = node["label"]
            ntype = node["type"]
            hash_feats = HashAlgorithms.hash_feature_vector(label) if ntype != NODE_TYPE_HASH_CONST else [0.0] * 6

            base = [
                float(ntype) / 2.0,
                float("kernel32" in label.lower()),
                float("ntdll" in label.lower()),
                float(node["type"] == NODE_TYPE_HASH_CONST),
                float(label.startswith("Load") or label.startswith("GetProc") or label.startswith("Ldr")),
                len(label) / 64.0,
            ] + hash_feats

            # pad/truncate to node_feature_dim
            feat = base[: self.node_feature_dim]
            if len(feat) < self.node_feature_dim:
                feat = feat + [0.0] * (self.node_feature_dim - len(feat))
            x[i] = feat
        return x

    @staticmethod
    def _build_edge_tensors(
        edges: list[tuple[int, int, int]], num_nodes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not edges:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, len(EDGE_TYPES)), dtype=np.float32),
            )

        src = [e[0] for e in edges if e[0] < num_nodes and e[1] < num_nodes]
        dst = [e[1] for e in edges if e[0] < num_nodes and e[1] < num_nodes]
        types = [e[2] for e in edges if e[0] < num_nodes and e[1] < num_nodes]

        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.zeros((len(types), len(EDGE_TYPES)), dtype=np.float32)
        for i, t in enumerate(types):
            if 0 <= t < len(EDGE_TYPES):
                edge_attr[i, t] = 1.0
        return edge_index, edge_attr

    @staticmethod
    def _is_hash_constant(val: Any) -> bool:
        s = str(val)
        try:
            v = int(s, 0)
            return v > 0xFFFF
        except ValueError:
            return False
