#!/usr/bin/env python3
"""
Bước 2 — Trích xuất unified features từ sample + sandbox log cho MCAFF.

Usage (trên VM):
  python scripts/02_extract_features.py
  python scripts/02_extract_features.py --verify-dar
  python scripts/02_extract_features.py --demo-only   # Chỉ dùng sandbox logs, không cần PE files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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


def _synthesize_static_features(
    api_calls: list[dict], group: str, feature_dim: int = 64
) -> dict:
    """
    Tạo synthetic static features từ sandbox log khi không có PE file thật.
    Dùng cho demo/testing — giả lập IAT features dựa trên behavioral pattern.
    """
    api_names = {c.get("api", c.get("function", "")) for c in api_calls}
    modules = {c.get("module", c.get("dll", "")) for c in api_calls}

    from src.features.static_pe import G1_APIS, G2_INDICATOR_APIS, G3_SUSPICIOUS_APIS

    g1_hits = sum(1 for a in api_names if a in G1_APIS)
    g2_hits = sum(1 for a in api_names if a in G2_INDICATOR_APIS)
    g3_hits = sum(1 for a in api_names if a in G3_SUSPICIOUS_APIS)
    num_imports = len(api_names)
    n = max(num_imports, 1)

    # Synthetic IAT sparse indicators per group
    group_profiles = {
        "G1": {"num_sections": 4, "has_loadlib": 1.0, "has_getproc": 1.0, "entropy": 5.5},
        "G2": {"num_sections": 5, "has_loadlib": 0.0, "has_getproc": 0.0, "entropy": 6.5},
        "G3": {"num_sections": 3, "has_loadlib": 0.0, "has_getproc": 0.0, "entropy": 7.5},
    }
    profile = group_profiles.get(group, group_profiles["G1"])

    raw = [
        profile["num_sections"],        # num_sections
        float(num_imports),             # num_imports
        float(len(modules)),            # num_import_dlls
        profile["has_loadlib"],         # has_loadlibrary
        profile["has_getproc"],         # has_getprocaddress
        0.0,                            # has_ldr_load_dll
        g1_hits / n,                    # g1_api_ratio
        g2_hits / n,                    # g2_indicator_ratio
        g3_hits / n,                    # g3_suspicious_ratio
        profile["entropy"],             # text_entropy
        profile["entropy"] * 0.8,       # data_entropy
        profile["entropy"] * 0.9,       # overall_entropy
        0.01,                           # import_table_size_ratio
        0.0,                            # is_dll
        1.0,                            # is_64bit
        1.0,                            # has_reloc
        0.0,                            # has_tls
        0.0,                            # num_exports
        float(sum(1 for _ in [] if True)),  # suspicious_section_count
        max(0.0, 1.0 - num_imports / 15),   # iat_sparse_score
        0.0,                            # missing_kernel32_direct
        0.0,                            # entry_point_section_idx
        0.0,                            # overlay_size_ratio
        1.0,                            # rich_header_present
        0.0,                            # num_resources
        0.0,                            # debug_present
    ]

    # Pad to feature_dim with polynomial interaction terms (same as StaticPEExtractor)
    arr = np.array(raw, dtype=np.float32)
    if len(arr) < feature_dim:
        extra = []
        n_raw = len(arr)
        for i in range(min(n_raw, feature_dim)):
            for j in range(i + 1, min(i + 6, n_raw)):
                extra.append(arr[i] * arr[j])
                if n_raw + len(extra) >= feature_dim:
                    break
            if n_raw + len(extra) >= feature_dim:
                break
        arr = np.concatenate([arr, np.array(extra[: feature_dim - len(arr)], dtype=np.float32)])
        if len(arr) < feature_dim:
            arr = np.concatenate([arr, np.zeros(feature_dim - len(arr), dtype=np.float32)])
    arr = arr[:feature_dim]

    # DAR score
    has_loadlib = bool(api_names & {"LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW"})
    has_getproc = "GetProcAddress" in api_names
    dar_score = 0.0
    if num_imports < 20 and has_loadlib and has_getproc:
        dar_score = 1.0 - min(num_imports / 20.0, 1.0)

    return {
        "feature_vector": arr,
        "feature_names": [],
        "num_imports": num_imports,
        "import_dlls": sorted(modules),
        "g1_apis_found": sorted(api_names & G1_APIS),
        "dar_score": dar_score,
        "metadata": {"path": "synthetic", "sha256_hint": "demo"},
    }


def extract_one(
    sample: dict,
    samples_dir: Path,
    unified_ext: UnifiedFeatureExtractor,
    behavior_ext: APIBehaviorExtractor,
    graph_builder: APIGraphBuilder,
    static_ext: StaticPEExtractor,
    verify_dar: bool = False,
    demo_only: bool = False,
) -> dict | None:
    sha256 = sample["sha256"]

    log_path = Path(sample.get("sandbox_log", f"data/sandbox_logs/{sha256}.json"))
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    api_calls = load_api_calls(log_path)

    # Try to find real PE file
    pe_path = find_sample_file(samples_dir, sha256, sample.get("filename", ""))
    has_pe = pe_path is not None and is_pe_file(pe_path) if pe_path else False

    if has_pe:
        # Real PE file available — full extraction
        static_data = static_ext.extract(pe_path)
    elif demo_only or api_calls:
        # No PE file but we have sandbox log — synthesize static features
        print(f"  [i] No PE file, using synthetic static features for {sha256[:16]}...")
        static_data = _synthesize_static_features(
            api_calls, sample.get("group", "G1"), static_ext.feature_dim
        )
    else:
        print(f"  [!] No PE file and no sandbox log: {sha256[:16]}...")
        return None

    behavior_data = behavior_ext.extract(api_calls)
    graph_data = graph_builder.build(static_data=static_data, api_calls=api_calls)

    # Build unified features for MCAFF
    unified = unified_ext.extract(
        pe_path=pe_path if has_pe else None,
        api_calls=api_calls,
        static_data=static_data,
        behavior_data=behavior_data,
        graph_data=graph_data,
    )

    dar_info = {}
    if verify_dar and has_pe:
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
        "has_real_pe": has_pe,
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
    parser.add_argument(
        "--demo-only", action="store_true",
        help="Chỉ dùng sandbox logs, tạo synthetic static features (không cần PE files)",
    )
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

    # In demo-only mode, filter to samples that have sandbox logs
    if args.demo_only:
        before = len(samples)
        samples = [
            s for s in samples
            if (ROOT / Path(s.get("sandbox_log", f"data/sandbox_logs/{s['sha256']}.json"))).exists()
        ]
        print(f"Demo-only mode: {len(samples)}/{before} samples have sandbox logs")

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
            sample, samples_dir, unified_ext, behavior_ext, graph_builder,
            static_ext, args.verify_dar, demo_only=args.demo_only,
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

    if ok == 0:
        print("\nKhông trích xuất được sample nào!")
        if not args.demo_only:
            print("Thử chạy với --demo-only để dùng synthetic static features.")
        sys.exit(1)

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
    print(f"\nExtracted {ok}/{len(samples)} (failed={fail}) -> {processed_dir}")
    print(f"Feature dims: static={feature_dims['static']}, behavior={feature_dims['behavior']}, graph={feature_dims['graph']}")


if __name__ == "__main__":
    main()

