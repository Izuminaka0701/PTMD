#!/usr/bin/env python3
"""
Bước 4 — Evaluate MCAFF model trên test set.

Usage (trên VM — CPU only):
  python scripts/04_evaluate.py
  python scripts/04_evaluate.py --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.datasets import DARDataset, collate_dar_batch
from src.models import MCAFF
from src.training.metrics import GROUP_NAMES, compute_metrics
from src.utils.io import load_json, save_json


def plot_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    arr = np.array(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=GROUP_NAMES, yticklabels=GROUP_NAMES,
    )
    plt.title("Confusion Matrix — G1/G2/G3 Classification (MCAFF)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_attention_weights(attn_weights: torch.Tensor, out_path: Path) -> None:
    """Visualize cross-attention weights between feature groups (for research paper)."""
    group_labels = ["Static PE", "Behavior", "Graph"]

    if attn_weights.dim() == 4:
        # [num_layers, batch, 3, 3] → average over batch
        attn = attn_weights.mean(dim=1).cpu().numpy()
    else:
        attn = attn_weights.cpu().numpy()

    num_layers = attn.shape[0]
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.heatmap(
            attn[i], annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=group_labels, yticklabels=group_labels,
            ax=ax, vmin=0, vmax=1,
        )
        ax.set_title(f"Cross-Attention Layer {i + 1}")
        ax.set_ylabel("Query (from)")
        ax.set_xlabel("Key (to)")

    plt.suptitle("MCAFF Cross-Attention Weights between Feature Groups", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MCAFF model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    index = load_json(processed_dir / "index.json")

    test_ds = DARDataset(processed_dir, split="test", family_to_idx=index["family_to_idx"])
    if len(test_ds) == 0:
        print("No test samples available.")
        sys.exit(1)

    loader = DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_dar_batch
    )

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

    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    all_true, all_pred, all_sha = [], [], []
    all_attn_weights = []

    for batch in loader:
        out = model(
            batch["static_x"].to(args.device),
            batch["behavior_x"].to(args.device),
            batch["graph_x"].to(args.device),
        )
        preds = out["logits_group"].argmax(dim=-1).cpu().numpy()
        labels = batch["y_group"].cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.tolist())
        all_sha.extend(batch["sha256"])

        if "attn_weights" in out:
            all_attn_weights.append(out["attn_weights"])

    metrics = compute_metrics(np.array(all_true), np.array(all_pred))
    metrics["predictions"] = [
        {"sha256": s, "true": GROUP_NAMES[t], "pred": GROUP_NAMES[p]}
        for s, t, p in zip(all_sha, all_true, all_pred)
    ]

    out_dir = Path(cfg["paths"]["checkpoint_dir"])
    save_json(metrics, out_dir / "test_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")

    # Plot attention weights for research paper
    if all_attn_weights:
        avg_attn = torch.stack(all_attn_weights).mean(dim=0)
        plot_attention_weights(avg_attn, out_dir / "attention_weights.png")
        print(f"Saved: {out_dir / 'attention_weights.png'}")

    print(json.dumps({k: v for k, v in metrics.items() if k != "predictions"}, indent=2))
    print(f"\nSaved: {out_dir / 'test_metrics.json'}")
    print(f"Saved: {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
