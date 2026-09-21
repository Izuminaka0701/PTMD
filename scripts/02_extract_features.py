#!/usr/bin/env python3
"""
Bước 2 — Trích xuất unified features từ sample + sandbox log cho MCAFF.

Usage (trên VM):
  python scripts/02_extract_features.py
  python scripts/02_extract_features.py --verify-dar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features import UnifiedFeatureExtractor
from src.features.api_behavior import APIBehaviorExtractor
from src.features.api_graph import APIGraphBuilder
from src.features.static_pe import StaticPEExtractor
from src.utils.io import load_json, save_json, set_seed
from src.utils.pe_utils import find_sample_file, is_pe_file, verify_dynamic_api_resolution
from src.utils.splits import stratified_split


def load_api_calls(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    data = load_json(log_path)
    return data.get("api_calls", data if isinstance(data, list) else [])


def extract_one(
    sample: dict,
    samples_dir: Path,
    unified_ext: UnifiedFeatureExtractor,
    behavior_ext: APIBehaviorExtractor,
    graph_builder: APIGraphBuilder,
    static_ext: StaticPEExtractor,
    verify_dar: bool = False,
) -> dict | None:
    sha256 = sample["sha256"]
    pe_path = find_sample_file(samples_dir, sha256, sample.get("filename", ""))
    if pe_path is None:
        print(f"  [!] Sample not found: {sha256[:16]}...")
        return None

    if not is_pe_file(pe_path):
        print(f"  [!] Not a PE file: {pe_path.name}")
        return None

    log_path = Path(sample.get("sandbox_log", f"data/sandbox_logs/{sha256}.json"))
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    api_calls = load_api_calls(log_path)

    # Extract with individual extractors (for metadata)
    static_data = static_ext.extract(pe_path)
    behavior_data = behavior_ext.extract(api_calls)
    graph_data = graph_builder.build(static_data=static_data, api_calls=api_calls)

    # Build unified features for MCAFF
    unified = unified_ext.extract(
        pe_path=pe_path,
        api_calls=api_calls,
        static_data=static_data,
        behavior_data=behavior_data,
        graph_data=graph_data,
    )

    dar_info = {}
    if verify_dar:
        is_dar, reason, dar_meta = verify_dynamic_api_resolution(pe_path)
        dar_info = {
            "is_dynamic_api_resolution": is_dar,
            "dar_reason": reason,
            **dar_meta,
        }

    return {
        "sha256": sha256,
        "family_id": sample.get("family_id"),
        "family_name": sample.get("family_name"),
        "group": sample.get("group"),
        "has_sandbox_log": len(api_calls) > 0,
        "dar_verification": dar_info,
        # Unified features for MCAFF (3 groups)
        "static_x": unified["static_x"].tolist(),
        "behavior_x": unified["behavior_x"].tolist(),
        "graph_x": unified["graph_x"].tolist(),
        # Metadata for analysis/reporting
        "metadata": {
            "g1_apis_found": static_data["g1_apis_found"],
            "import_dlls": static_data["import_dlls"],
            "num_imports": static_data["num_imports"],
            "dar_score": static_data.get("dar_score", 0.0),
            "behavior_stats": behavior_data["behavior_stats"],
            "behavior_patterns": behavior_data["patterns"],
            "graph_stats": graph_data["graph_stats"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unified features for MCAFF")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--verify-dar", action="store_true", help="Chạy heuristic DAR verification")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    manifest_path = Path(args.manifest or cfg["paths"]["manifest_file"])
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_json(manifest_path)
    samples = manifest.get("samples", [])
    if not samples:
        print("No samples in manifest.")
        sys.exit(1)

    features_cfg = cfg["features"]
    static_ext = StaticPEExtractor(feature_dim=features_cfg["static"]["feature_dim"])
    behavior_ext = APIBehaviorExtractor(
        max_seq_len=features_cfg["behavior"]["max_seq_len"],
        embed_dim=features_cfg["behavior"]["embed_dim"],
    )
    graph_builder = APIGraphBuilder(
        max_nodes=features_cfg["graph"]["max_nodes"],
        node_feature_dim=features_cfg["graph"]["node_feature_dim"],
    )
    unified_ext = UnifiedFeatureExtractor(
        static_feature_dim=features_cfg["static"]["feature_dim"],
        behavior_max_seq_len=features_cfg["behavior"]["max_seq_len"],
        behavior_embed_dim=features_cfg["behavior"]["embed_dim"],
        graph_max_nodes=features_cfg["graph"]["max_nodes"],
        graph_node_feature_dim=features_cfg["graph"]["node_feature_dim"],
    )

    samples_dir = Path(cfg["paths"]["samples_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    samples = stratified_split(
        samples,
        cfg["training"]["val_split"],
        cfg["training"]["test_split"],
        cfg["project"]["seed"],
    )

    family_ids = sorted({s.get("family_id", "") for s in samples if s.get("family_id")})
    family_to_idx = {fid: i for i, fid in enumerate(family_ids)}

    ok, fail = 0, 0
    processed_samples = []

    for sample in samples:
        sha = sample["sha256"]
        print(f"Processing {sha[:16]}... ({sample.get('family_name', '?')} / {sample.get('group', '?')})")
        result = extract_one(
            sample, samples_dir, unified_ext, behavior_ext, graph_builder, static_ext, args.verify_dar
        )
        if result:
            save_json(result, processed_dir / f"{sha}.json")
            processed_samples.append(sample)
            ok += 1
            if result.get("dar_verification"):
                dv = result["dar_verification"]
                tag = "DYN-API" if dv.get("is_dynamic_api_resolution") else "normal"
                print(f"  DAR verify: {tag} — {dv.get('dar_reason', '')}")
        else:
            fail += 1

    feature_dims = UnifiedFeatureExtractor.get_feature_dims()
    index = {
        "samples": [
            {
                "sha256": s["sha256"],
                "family_id": s.get("family_id"),
                "family_name": s.get("family_name"),
                "group": s.get("group"),
                "split": s.get("split", "train"),
            }
            for s in processed_samples
        ],
        "family_to_idx": family_to_idx,
        "feature_dims": feature_dims,
        "stats": {
            "total_manifest": len(samples),
            "extracted": ok,
            "failed": fail,
            "with_sandbox_log": sum(
                1 for s in processed_samples
                if load_api_calls(ROOT / f"data/sandbox_logs/{s['sha256']}.json")
            ),
        },
    }
    save_json(index, processed_dir / "index.json")
    print(f"\nExtracted {ok}/{len(samples)} (failed={fail}) → {processed_dir}")
    print(f"Feature dims: static={feature_dims['static']}, behavior={feature_dims['behavior']}, graph={feature_dims['graph']}")


if __name__ == "__main__":
    main()
