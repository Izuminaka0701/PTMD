#!/usr/bin/env python3
"""
Bước 3 — Train MCAFF (Multi-Head Cross-Attention Feature Fusion) model.

Usage (trên VM — CPU only):
  python scripts/03_train_model.py
  python scripts/03_train_model.py --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from src.config import load_config
from src.datasets import DARDataset, collate_dar_batch
from src.models import MCAFF
from src.training import Trainer
from src.training.trainer import compute_class_weights
from src.utils.io import load_json, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MCAFF model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    if args.device:
        cfg["training"]["device"] = args.device
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    processed_dir = Path(cfg["paths"]["processed_dir"])
    index = load_json(processed_dir / "index.json")

    train_ds = DARDataset(processed_dir, split="train", family_to_idx=index["family_to_idx"])
    val_ds = DARDataset(processed_dir, split="val", family_to_idx=index["family_to_idx"])

    if len(train_ds) == 0:
        print("No training samples. Run scripts/02_extract_features.py first.")
        sys.exit(1)

    batch_size = min(cfg["training"]["batch_size"], len(train_ds))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_dar_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dar_batch
    )

    class_weights = None
    if not args.no_class_weights:
        labels = [DARDataset.GROUP_MAP.get(s["group"], 0) for s in index["samples"] if s.get("split") == "train"]
        if labels:
            class_weights = compute_class_weights(labels)
            print(f"Class weights (G1/G2/G3): {class_weights.tolist()}")

    # Get feature dimensions from index or config
    feature_dims = index.get("feature_dims", {})
    model_cfg = cfg["model"]

    model = MCAFF(
        static_dim=feature_dims.get("static", model_cfg["static_dim"]),
        behavior_dim=feature_dims.get("behavior", model_cfg["behavior_dim"]),
        graph_dim=feature_dims.get("graph", model_cfg["graph_dim"]),
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_attention_layers=model_cfg["num_attention_layers"],
        ffn_ratio=model_cfg["ffn_ratio"],
        num_classes=model_cfg["num_classes"],
        num_families=max(len(index["family_to_idx"]), 1),
        family_classifier=model_cfg["family_classifier"],
        dropout=model_cfg["dropout"],
        pool_strategy=model_cfg.get("pool_strategy", "mean"),
    )

    print(f"Model: MCAFF (Multi-Head Cross-Attention Feature Fusion)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Feature dims: static={feature_dims.get('static', '?')}, "
          f"behavior={feature_dims.get('behavior', '?')}, "
          f"graph={feature_dims.get('graph', '?')}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Device: {cfg['training']['device']}")

    trainer = Trainer(model, cfg, device=args.device, class_weights=class_weights)
    result = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    save_json(result, Path(cfg["paths"]["checkpoint_dir"]) / "training_history.json")
    print(f"\nBest validation F1: {result['best_f1']:.4f}")


if __name__ == "__main__":
    main()
