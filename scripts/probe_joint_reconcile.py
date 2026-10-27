#!/usr/bin/env python3
"""Deterministic CPU-only candidate-graph fixture; no OCR or GT metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CANONICAL_FLOAT_SIGFIGS = 12

from babaorum.experimental.joint_reconcile import (  # noqa: E402
    CropTransform, LineProposal, ReconcileConfig,
    build_reconciliation_graph, graph_sha256,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_float_tree(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical graph hashes require finite floats")
        return {"$float": "0" if value == 0 else format(value, f".{CANONICAL_FLOAT_SIGFIGS}g")}
    if isinstance(value, dict):
        return {key: _canonical_float_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_float_tree(item) for item in value]
    return value


def canonical_graph_sha256(value) -> str:
    canonical = json.dumps(
        _canonical_float_tree(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fixture_proposals() -> list[LineProposal]:
    page_hash = sha("synthetic-page-pixels-v1")
    page = dict(page_key="synthetic:reconcile/page-1", document_key="synthetic:issue-1",
                page_size_wh=(240.0, 120.0), page_sha256=page_hash)
    common = dict(checkpoint_sha256=None, geometry_status="available", orientation="ltr",
                  band_normal_negative_px=15.0, band_normal_positive_px=5.0,
                  text_status="missing", text=None, token_ids=None, ctc_status="unavailable")
    crop_a = CropTransform((0.0, 0.0, 130.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    crop_b = CropTransform((110.0, 0.0, 240.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    return [
        LineProposal(proposal_id="global:q0", source="global_page", query_id="q0",
                     input_sha256=sha("synthetic-global-input"), baseline_xy=((10.0, 30.0), (230.0, 30.0)),
                     block_id=None, crop_transform=None, **page, **common),
        LineProposal(proposal_id="local:a:q0", source="local_block", query_id="q0",
                     input_sha256=sha("synthetic-crop-a"), baseline_xy=((10.0, 30.0), (125.0, 30.0)),
                     block_id="block-a", crop_transform=crop_a, **page, **common),
        LineProposal(proposal_id="local:b:q0", source="local_block", query_id="q0",
                     input_sha256=sha("synthetic-crop-b"), baseline_xy=((0.0, 30.0), (120.0, 30.0)),
                     block_id="block-b", crop_transform=crop_b, **page, **common),
        LineProposal(proposal_id="local:a:q1", source="local_block", query_id="q1",
                     input_sha256=sha("synthetic-crop-a"), baseline_xy=((112.0, 60.0), (123.0, 60.0)),
                     block_id="block-a", crop_transform=crop_a, **page, **common),
        LineProposal(proposal_id="local:b:q1", source="local_block", query_id="q1",
                     input_sha256=sha("synthetic-crop-b"), baseline_xy=((2.0, 60.0), (13.0, 60.0)),
                     block_id="block-b", crop_transform=crop_b, **page, **common),
    ]


def crop_scale_diagnostics(config: ReconcileConfig) -> dict:
    """Exercise crop-scale factors in edge epsilon and fallback band extent."""
    page_hash = sha("synthetic-page-pixels-crop-scale-v1")
    rect = (50.0, 10.0, 150.0, 110.0)
    padding = (4.0, 6.0, 2.0, 3.0)
    scales = (1.5, 3.0)
    lines = {
        "near": ((50.5, 50.0), (80.0, 50.0)),  # 0.5 page px from the left crop edge
        "band": ((60.0, 20.0), (120.0, 20.0)),  # fallback 1h band reaches the top crop edge
        "explicit_band": ((60.0, 25.0), (120.0, 25.0)),
    }
    fixture = fixture_proposals()
    global_template, local_template = fixture[0], fixture[1]
    proposals: list[LineProposal] = []
    for name, points in lines.items():
        global_line = replace(
            global_template,
            proposal_id=f"scale:global:{name}", query_id=f"scale:{name}",
            page_size_wh=(200.0, 120.0), page_sha256=page_hash,
            input_sha256=sha(f"scale-global-{name}"), baseline_xy=points,
            band_normal_negative_px=None, band_normal_positive_px=None,
        )
        proposals.append(global_line)
        for scale in scales:
            crop = CropTransform(rect, scale, padding)
            local_points = tuple(
                (padding[0] + (x - rect[0]) * scale,
                 padding[1] + (y - rect[1]) * scale)
                for x, y in points
            )
            explicit_band = name == "explicit_band"
            local_line = replace(
                local_template,
                proposal_id=f"scale:local:{name}:{scale:g}", query_id=f"scale:{name}",
                page_size_wh=(200.0, 120.0), page_sha256=page_hash,
                input_sha256=sha(f"scale-local-{name}-{scale:g}"), baseline_xy=local_points,
                crop_transform=crop,
                band_normal_negative_px=(20.0 * scale if explicit_band else None),
                band_normal_positive_px=((20.0 / 3.0) * scale if explicit_band else None),
            )
            proposals.append(local_line)

    graph = build_reconciliation_graph(proposals, config)
    nodes = {node["proposal"]["proposal_id"]: node for node in graph["nodes"]}
    near_flags = [nodes[f"scale:local:near:{scale:g}"]["crop_flags"] for scale in scales]
    band_flags = [nodes[f"scale:local:band:{scale:g}"]["crop_flags"] for scale in scales]
    explicit_flags = [nodes[f"scale:local:explicit_band:{scale:g}"]["crop_flags"] for scale in scales]
    edge_map = {tuple(item["source_ids"]): item for item in graph["edges"]}
    near_edges = [edge_map[("scale:global:near", f"scale:local:near:{scale:g}")]
                  for scale in scales]
    band_edges = [edge_map[("scale:global:band", f"scale:local:band:{scale:g}")]
                  for scale in scales]
    explicit_edges = [edge_map[("scale:global:explicit_band",
                                f"scale:local:explicit_band:{scale:g}")] for scale in scales]
    checks = {
        "crop_scale_epsilon_flags_invariant_and_active": (
            near_flags[0] == near_flags[1]
            and all(flags["touches_internal_crop_edge"] for flags in near_flags)
        ),
        "crop_scale_band_extension_flags_invariant_and_active": (
            band_flags[0] == band_flags[1]
            and all(flags["may_be_truncated_by_crop_band"] for flags in band_flags)
            and all(not flags["touches_internal_crop_edge"] for flags in band_flags)
        ),
        "explicit_band_extents_scale_with_crop": (
            explicit_flags[0] == explicit_flags[1]
            and all(flags["may_be_truncated_by_crop_band"] for flags in explicit_flags)
            and all(not flags["touches_internal_crop_edge"] for flags in explicit_flags)
            and nodes["scale:local:explicit_band:3"]["proposal"]["band_normal_negative_px"]
            == 2 * nodes["scale:local:explicit_band:1.5"]["proposal"]["band_normal_negative_px"]
        ),
        "crop_scale_candidate_labels_invariant": (
            near_edges[0]["labels"] == near_edges[1]["labels"]
            and band_edges[0]["labels"] == band_edges[1]["labels"]
            and explicit_edges[0]["labels"] == explicit_edges[1]["labels"]
            and all("crop_truncation_possible" in item["labels"]
                    for item in (*near_edges, *band_edges, *explicit_edges))
        ),
        "crop_scale_page_provenance_shared_and_inputs_distinct": (
            len({item.page_sha256 for item in proposals}) == 1
            and len({item.input_sha256 for item in proposals}) == len(proposals)
        ),
        "canonical_hash_ignores_sub_precision_float_noise": (
            canonical_graph_sha256({"x": 500.0})
            == canonical_graph_sha256({"x": 500.0000000002})
        ),
        "canonical_hash_preserves_resolvable_float_change": (
            canonical_graph_sha256({"x": 0.001})
            != canonical_graph_sha256({"x": 0.0010000000001})
        ),
        "canonical_hash_exposes_rounding_boundary": (
            format(1.0000000000049, ".12g") == "1"
            and format(1.0000000000051, ".12g") == "1.00000000001"
            and canonical_graph_sha256({"x": 1.0000000000049})
            != canonical_graph_sha256({"x": 1.0000000000051})
        ),
        "canonical_hash_normalizes_signed_zero": (
            canonical_graph_sha256({"x": -0.0}) == canonical_graph_sha256({"x": 0.0})
        ),
        "canonical_hash_refuses_nonfinite": _canonical_rejects_nonfinite(),
    }
    return {
        "checks": checks,
        "graph_sha256": graph_sha256(graph),
        "graph_sha256_canonical_12sig": canonical_graph_sha256(graph),
        "local_nodes": [
            {"proposal_id": proposal_id,
             "band_normal_negative_px": nodes[proposal_id]["proposal"]["band_normal_negative_px"],
             "band_normal_positive_px": nodes[proposal_id]["proposal"]["band_normal_positive_px"],
             "crop_flags": nodes[proposal_id]["crop_flags"]}
            for proposal_id in sorted(nodes) if proposal_id.startswith("scale:local:")
        ],
        "paired_edges": [
            {"source_ids": list(item["source_ids"]), "labels": item["labels"],
             "coverage_left_on_right": item["coverage_left_on_right"],
             "coverage_right_on_left": item["coverage_right_on_left"],
             "endpoint_distance_heights": item["endpoint_distance_heights"]}
            for item in sorted(graph["edges"], key=lambda edge: edge["source_ids"])
            if "scale:global:" in item["source_ids"][0]
            and "scale:local:" in item["source_ids"][1]
            and item["source_ids"][0].split(":")[-1] == item["source_ids"][1].split(":")[2]
        ],
        "scope": "constructed_same_page_coordinates_only_no_image_no_ground_truth",
    }


def _canonical_rejects_nonfinite() -> bool:
    try:
        canonical_graph_sha256({"x": math.nan})
    except ValueError:
        return True
    return False


def run(config_data: dict) -> dict:
    config = ReconcileConfig(**config_data)
    proposals = fixture_proposals()
    graph = build_reconciliation_graph(proposals, config)
    reversed_graph = build_reconciliation_graph(list(reversed(proposals)), config)
    edge_map = {tuple(e["source_ids"]): e for e in graph["edges"]}
    checks = {
        "all_five_proposals_retained": len(graph["nodes"]) == 5,
        "global_to_two_local_partials": all(
            "partial_overlap_candidate" in edge_map[("global:q0", local_id)]["labels"]
            for local_id in ("local:a:q0", "local:b:q0")
        ),
        "local_to_local_duplicate_candidate": (
            "duplicate_candidate" in edge_map[("local:a:q1", "local:b:q1")]["labels"]
        ),
        "global_spanned_by_two_local_blocks": len(graph["groups"]) == 1 and
            graph["groups"][0]["global_proposal_id"] == "global:q0" and
            graph["groups"][0]["block_ids"] == ["block-a", "block-b"] and
            graph["groups"][0]["global_arc_coverage_union"] >= config.multi_local_group_coverage and
            graph["groups"][0]["increment_over_best_single_coverage"] > 0.05,
        "group_reports_overlap_and_member_edges": len(graph["groups"]) == 1 and
            graph["groups"][0]["redundant_coverage_sum_minus_union"] >= 0 and
            len(graph["groups"][0]["member_edges"]) == 2,
        "selection_is_null": graph["selection"] is None,
        "input_permutation_invariant": graph_sha256(graph) == graph_sha256(reversed_graph),
        "page_source_hash_shared": len({p.page_sha256 for p in proposals}) == 1,
        "local_crop_input_hashes_distinct": len({p.input_sha256 for p in proposals if p.source == "local_block"}) == 2,
    }
    crop_scale = crop_scale_diagnostics(config)
    checks.update(crop_scale["checks"])
    return {
        "result": "PASS_SYNTHETIC_WIRING_ONLY" if all(checks.values()) else "FAIL",
        "data_scope": "constructed_coordinates_only_no_image_no_ground_truth",
        "device": "cpu",
        "checks": checks,
        "crop_scale_diagnostics": crop_scale,
        "graph_sha256": graph_sha256(graph),
        "graph_sha256_canonical_12sig": canonical_graph_sha256(graph),
        "graph": graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_reconcile_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_reconcile_probe_scale_v6.json")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config_data = json.loads(config_bytes)
    report = run(config_data)
    report["provenance"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256((ROOT / "src/babaorum/experimental/joint_reconcile.py").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config_data,
        "historical_probe_note": (
            "The preceding report references script SHA256 0a473f8e...; that source snapshot was not archived. "
            "Its graph SHA256 3695885c4d309a879d35c382c94c0c1d8ff1ae8335706287c882985ab5e9e0a4 "
            "is reproduced by this probe and matches reports/joint_reconcile_probe.json."
        ),
        "runtime": {
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "raw_hash_note": "Exact-float graph_sha256 can vary across Python runtimes.",
            "canonical_hash_float_precision": f"{CANONICAL_FLOAT_SIGFIGS} significant digits; nonfinite values refused.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": report["result"], "checks": report["checks"],
                      "graph_sha256": report["graph_sha256"], "output": str(args.output)}, indent=2))
    if report["result"] != "PASS_SYNTHETIC_WIRING_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
