#!/usr/bin/env python3
"""
Bước 5 — Dự đoán nhóm kỹ thuật G1/G2/G3 cho sample mới bằng MCAFF.

Usage (trên VM — CPU only):
  python scripts/05_predict.py --sample data/samples/unknown.exe --log data/sandbox_logs/unknown.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features import UnifiedFeatureExtractor
from src.models import MCAFF
from src.training.metrics import GROUP_NAMES
from src.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict G1/G2/G3 for a sample using MCAFF")
    parser.add_argument("--sample", type=str, required=True, help="Path to PE file")
    parser.add_argument("--log", type=str, default=None, help="Sandbox API log JSON")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config()
    features_cfg = cfg["features"]

    unified_ext = UnifiedFeatureExtractor(
        static_feature_dim=features_cfg["static"]["feature_dim"],
        behavior_max_seq_len=features_cfg["behavior"]["max_seq_len"],
        behavior_embed_dim=features_cfg["behavior"]["embed_dim"],
        graph_max_nodes=features_cfg["graph"]["max_nodes"],
        graph_node_feature_dim=features_cfg["graph"]["node_feature_dim"],
    )

    api_calls = []
    if args.log:
        log_data = load_json(args.log)
        api_calls = log_data.get("api_calls", log_data if isinstance(log_data, list) else [])

    features = unified_ext.extract(pe_path=args.sample, api_calls=api_calls)

    index_path = Path(cfg["paths"]["processed_dir"]) / "index.json"
    index = load_json(index_path) if index_path.exists() else {}

    feature_dims = index.get("feature_dims", UnifiedFeatureExtractor.get_feature_dims())
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
        num_families=max(len(index.get("family_to_idx", {})), 1),
        family_classifier=model_cfg["family_classifier"],
        dropout=model_cfg["dropout"],
        pool_strategy=model_cfg.get("pool_strategy", "mean"),
    )

    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        out = model(
            torch.tensor(features["static_x"], dtype=torch.float32).unsqueeze(0).to(args.device),
            torch.tensor(features["behavior_x"], dtype=torch.float32).unsqueeze(0).to(args.device),
            torch.tensor(features["graph_x"], dtype=torch.float32).unsqueeze(0).to(args.device),
        )

    probs = torch.softmax(out["logits_group"], dim=-1).cpu().numpy()[0]
    pred = int(probs.argmax())

    print(f"\nSample: {args.sample}")
    print(f"Model: MCAFF (Multi-Head Cross-Attention Feature Fusion)")
    print(f"Predicted group: {GROUP_NAMES[pred]}")
    print("Probabilities:")
    for i, name in enumerate(GROUP_NAMES):
        print(f"  {name}: {probs[i]:.4f}")

    # Show attention weights (which feature groups contributed most)
    if "attn_weights" in out:
        group_labels = ["Static PE", "Behavior", "Graph"]
        print("\nCross-Attention weights (last layer, mean over heads):")
        attn = out["attn_weights"][-1].squeeze(0).cpu().numpy()
        for i, from_g in enumerate(group_labels):
            for j, to_g in enumerate(group_labels):
                print(f"  {from_g} → {to_g}: {attn[i][j]:.4f}")

    print(f"\nFeature metadata: {features['metadata']}")


if __name__ == "__main__":
    main()
