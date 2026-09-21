"""Stratified train/val/test split by G1/G2/G3 group."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def stratified_split(
    samples: list[dict],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    group_key: str = "group",
) -> list[dict]:
    """Giữ tỷ lệ G1/G2/G3 trong mỗi split."""
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = defaultdict(list)

    for i, s in enumerate(samples):
        by_group[s.get(group_key, "G1")].append(i)

    test_idx: set[int] = set()
    val_idx: set[int] = set()

    for group, indices in by_group.items():
        idx_arr = np.array(indices)
        rng.shuffle(idx_arr)
        n = len(idx_arr)
        n_test = max(1, int(n * test_split)) if n >= 3 else (1 if n > 1 else 0)
        n_val = max(1, int(n * val_split)) if n >= 3 else (1 if n > 2 else 0)

        test_idx.update(idx_arr[:n_test].tolist())
        val_idx.update(idx_arr[n_test : n_test + n_val].tolist())

    for i, s in enumerate(samples):
        if i in test_idx:
            s["split"] = "test"
        elif i in val_idx:
            s["split"] = "val"
        else:
            s["split"] = "train"
    return samples
