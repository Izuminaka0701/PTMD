"""
DAR Dataset — loads preprocessed unified features for G1/G2/G3 classification.

Dùng cho MCAFF model — mỗi sample chứa 3 feature groups:
  - static_x:   [static_dim]
  - behavior_x: [behavior_dim]
  - graph_x:    [graph_dim]
  - y_group:    int 0=G1, 1=G2, 2=G3
  - y_family:   int family index (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import load_json


class DARDataset(Dataset):
    """
    Each item contains:
      - static_x:    [static_dim]   — PE/IAT features
      - behavior_x:  [behavior_dim] — Behavior statistics + patterns
      - graph_x:     [graph_dim]    — Graph topology statistics
      - y_group:     int 0=G1, 1=G2, 2=G3
      - y_family:    int family index (optional)
    """

    GROUP_MAP = {"G1": 0, "G2": 1, "G3": 2, "benign": -1}

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        family_to_idx: dict[str, int] | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        index_path = self.processed_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing {index_path}. Run: python scripts/02_extract_features.py"
            )

        index = load_json(index_path)
        self.samples = [s for s in index["samples"] if s.get("split") == split]
        self.family_to_idx = family_to_idx or index.get("family_to_idx", {})
        self.num_families = len(self.family_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.samples[idx]
        data = load_json(self.processed_dir / f"{meta['sha256']}.json")

        y_group = self.GROUP_MAP.get(meta["group"], 0)
        y_family = self.family_to_idx.get(meta.get("family_id", ""), 0)

        return {
            "sha256": meta["sha256"],
            "static_x": torch.tensor(data["static_x"], dtype=torch.float32),
            "behavior_x": torch.tensor(data["behavior_x"], dtype=torch.float32),
            "graph_x": torch.tensor(data["graph_x"], dtype=torch.float32),
            "y_group": torch.tensor(y_group, dtype=torch.long),
            "y_family": torch.tensor(y_family, dtype=torch.long),
        }


def collate_dar_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple collate — no graph batching needed for MCAFF."""
    return {
        "sha256": [b["sha256"] for b in batch],
        "static_x": torch.stack([b["static_x"] for b in batch]),
        "behavior_x": torch.stack([b["behavior_x"] for b in batch]),
        "graph_x": torch.stack([b["graph_x"] for b in batch]),
        "y_group": torch.stack([b["y_group"] for b in batch]),
        "y_family": torch.stack([b["y_family"] for b in batch]),
    }
