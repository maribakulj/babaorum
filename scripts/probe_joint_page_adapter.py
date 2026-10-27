#!/usr/bin/env python3
"""Tiny encoded-image/page-graph bridge probe; no training or OCR benchmark.

Geometry, presence, text paths and cyclic order are injected interface fixtures.
Existing reports and the companion encoded PNG are never overwritten.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time

import PIL
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_curved_page import TinyCurvedJointPage  # noqa: E402
from babaorum.experimental.joint_page_adapter import (  # noqa: E402
    adapt_curved_page, geometry_predictions_from_graph,
)
from babaorum.line_geometry import GroundTruthLine, evaluate_line_geometry  # noqa: E402
from babaorum.page_graph import validate_page_graph  # noqa: E402


def image_input(config):
    width, height = config["image_size_wh"]
    raw = bytes((17 * x + 11 * y) % 256 for y in range(height) for x in range(width))
    output = BytesIO()
    Image.frombytes("L", (width, height), raw).save(output, format="PNG")
    encoded = output.getvalue()
    with Image.open(BytesIO(encoded)) as image:
        decoded = image.convert("L").tobytes()
    tensor = torch.tensor(list(decoded), dtype=torch.float64).reshape(1, 1, height, width) / 255.0
    normalized_bytes = array("d", (pixel / 255.0 for pixel in decoded)).tobytes()
    binding = {"decoded_uint8_pixels_sha256": hashlib.sha256(decoded).hexdigest(),
               "decoded_normalized_tensor_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
               "tensor_shape": [1, 1, height, width], "tensor_dtype": "float64", "tensor_order": "C",
               "byte_order": sys.byteorder, "normalization": "uint8 / 255.0",
               "normalized_reference_method": "stdlib array double from separately decoded PNG pixels"}
    return encoded, tensor, binding


def prediction_fixture(config, model, image, *, fault=None, filtered=False):
    baseline = torch.tensor(config["predicted_baselines_xy"], dtype=torch.float64)
    heights = torch.tensor(config["predicted_heights_px"], dtype=torch.float64)
    query = config["fault_query"]
    if fault == "collapsed":
        baseline[query] = torch.tensor(config["collapsed_point_xy"], dtype=torch.float64)
    elif fault == "cap":
        baseline[query] = torch.tensor(config["cap_baseline_xy"], dtype=torch.float64)
    with torch.no_grad():
        forward_input = {"sha256": hashlib.sha256(image.contiguous().numpy().tobytes(order="C")).hexdigest(),
                         "shape": list(image.shape), "dtype": str(image.dtype), "order": "C", "byte_order": sys.byteorder}
        prediction = model(image, baseline_override=baseline, height_override=heights)
    prediction.presence_logits = torch.tensor(
        config["filtered_presence_logits" if filtered else "selected_presence_logits"], dtype=torch.float64,
    )
    prediction.order_logits = torch.tensor(config["unresolved_cyclic_order_logits"], dtype=torch.float64)
    for item, path in zip(prediction.queries, config["forced_ctc_paths"]):
        if item.ctc_logits is None:
            continue
        logits = torch.full_like(item.ctc_logits, -config["forced_logit_magnitude"])
        indices = list(path) + [0] * (len(logits) - len(path))
        for step, label in enumerate(indices):
            logits[step, 0, label] = config["forced_logit_magnitude"]
        item.ctc_logits = logits
    return prediction, forward_input


def run(config, encoded, image, config_digest, binding):
    torch.manual_seed(config["seed"])
    names = ("num_queries", "control_points", "channels", "alphabet_size", "band_rows",
             "baseline_fraction", "spacing_px", "max_samples", "penalty_scale_px")
    model = TinyCurvedJointPage(**{name: config[name] for name in names}).double()
    components = []
    for name in ("joint_curved_page", "joint_page_adapter"):
        source_digest = hashlib.sha256((ROOT / "src/babaorum/experimental" / (name + ".py")).read_bytes()).hexdigest()
        components.append({"name": name, "code_revision": "source_sha256:" + source_digest,
                           "config_sha256": config_digest, "checkpoint_sha256": None, "license": None})
    kwargs = {name: config[name] for name in ("page_key", "document_key", "title_key", "run_id", "presence_threshold")}
    kwargs.update(png_bytes=encoded, components=components)
    alphabet = tuple(config["alphabet"])
    predictions, forward_inputs = {}, {}
    for name, fault, filtered in (("valid", None, False), ("collapsed", "collapsed", False),
                                 ("cap", "cap", False), ("filtered", None, True)):
        predictions[name], forward_inputs[name] = prediction_fixture(config, model, image, fault=fault, filtered=filtered)
    outputs = {name: adapt_curved_page(prediction, alphabet=alphabet, **kwargs)
               for name, prediction in predictions.items()}
    outputs["no_alphabet"] = adapt_curved_page(predictions["valid"], alphabet=None, **kwargs)
    outputs["unknown_document"] = adapt_curved_page(
        predictions["valid"], alphabet=alphabet, **dict(kwargs, document_key=None),
    )
    for output in outputs.values():
        validate_page_graph(output.graph)
    valid, invalid, cap = (outputs[name].graph for name in ("valid", "collapsed", "cap"))
    expected_points = [[[x + .5, y + .5] for x, y in points] for points in config["predicted_baselines_xy"]]
    fraction = config["baseline_fraction"]
    checks = {
        "all_six_graphs_validate": True,
        "decoded_png_matches_every_forward_tensor": all(item["sha256"] == binding["decoded_normalized_tensor_sha256"] and
                                                       item["shape"] == binding["tensor_shape"] and item["dtype"] == "torch.float64"
                                                       for item in forward_inputs.values()),
        "encoded_png_hash": all(out.graph["image"]["sha256"] == hashlib.sha256(encoded).hexdigest() for out in outputs.values()),
        "encoded_dimensions": valid["image"]["width"] == image.shape[-1] and valid["image"]["height"] == image.shape[-2],
        "center_to_edge_conversion": [line["baseline"]["points"] for line in valid["lines"]] == expected_points,
        "coordinate_round_trip": [[[x - .5, y - .5] for x, y in line["baseline"]["points"]] for line in valid["lines"]] == config["predicted_baselines_xy"],
        "normal_distances_unchanged": all(line["baseline_band"]["normal_negative_px"] == fraction * h and
                                         line["baseline_band"]["normal_positive_px"] == (1 - fraction) * h
                                         for line, h in zip(valid["lines"], config["predicted_heights_px"])),
        "predicted_baseline_sources_only": all(line["baseline"]["source"] == "predicted" for line in valid["lines"]),
        "checkpoint_explicitly_absent": all(component["checkpoint_sha256"] is None for component in valid["provenance"]["components"]),
        "no_composition_contours": all(line["polygon"] == {"status": "missing", "points": None, "source": None}
                                       for output in outputs.values() for line in output.graph["lines"]),
        "no_words_invented": all(output.graph["words"] == [] for output in outputs.values()),
        "cyclic_logits_not_exported_as_order": all(output.graph["relations"] == [] for output in outputs.values()),
        "no_complete_coverage_claim": all(record["status"] != "complete" and record["expected_count"] is None
                                          for output in outputs.values() for record in output.graph["coverage"].values()),
        "blank_only_is_available_empty": valid["lines"][0]["text"] == {"status": "available", "value": ""},
        "ctc_repeat_blank_collapse": valid["lines"][1]["text"] == {"status": "available", "value": "AB"},
        "space_is_not_stripped": valid["lines"][2]["text"] == {"status": "available", "value": " "},
        "empty_text_in_available_count": valid["coverage"]["transcription"]["available_count"] == 3,
        "missing_alphabet_is_missing_text": all(line["text"] == {"status": "missing", "value": None}
                                                for line in outputs["no_alphabet"].graph["lines"]),
        "unknown_document_remains_null": outputs["unknown_document"].graph["identity"]["document_key"] is None,
        "presence_filter_keeps_query_ids": [line["id"] for line in outputs["filtered"].graph["lines"]]
                                          == [valid["lines"][index]["id"] for index in (0, 2)],
        "selected_invalid_query_still_emitted": len(invalid["lines"]) == 3,
        "invalid_baseline_error_without_points": invalid["lines"][1]["baseline"] == {"status": "error", "points": None, "source": None},
        "invalid_line_counted_as_error": invalid["coverage"]["lines"]["status"] == "error" and invalid["coverage"]["lines"]["error_count"] == 1,
        "cap_preserves_available_baseline": cap["lines"][1]["baseline"]["status"] == "available",
        "cap_band_validity_unknown": cap["lines"][1]["baseline_band"]["status"] == "missing",
        "cap_transcription_error": cap["lines"][1]["text"] == {"status": "error", "value": None},
    }
    # GT exists only in this diagnostic scorer, after adaptation. It is not an
    # independent OCR benchmark: these exact fixture locations were injected.
    gt = [GroundTruthLine(f"gt_{index}", [points[0], points[-1]], h, "ground_truth")
          for index, (points, h) in enumerate(zip(expected_points, config["predicted_heights_px"]))]
    metric = evaluate_line_geometry(gt, geometry_predictions_from_graph(invalid))
    numerator, denominator = config["expected_invalid_geometry_precision"]
    checks.update({
        "invalid_geometry_remains_precision_denominator": metric.n_pred == denominator and metric.matched_count == numerator,
        "precision_two_over_three": math.isclose(metric.precision, numerator / denominator),
        "unavailable_geometry_explicit_in_metric": metric.pred_geometry_unavailable_count == 1,
        "cap_baseline_retained_by_metric_bridge": geometry_predictions_from_graph(cap)[1].geometry_status == "available",
    })
    return {"checks": checks, "graphs": {name: output.graph for name, output in outputs.items()},
            "diagnostics": {name: output.diagnostics for name, output in outputs.items()},
            "fixture_geometry_metric": asdict(metric),
            "input_link": dict(binding, forward_inputs=forward_inputs),
            "prediction_fixture_scope": "untrained forward plus explicit baseline/presence/text/order overrides; not predicted OCR performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_page_adapter_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_page_adapter_probe.json")
    args = parser.parse_args()
    png_path = args.output.with_suffix(".png")
    if args.output.exists() or png_path.exists():
        parser.error("Output or companion PNG exists; use a distinct --output to preserve evidence")
    started = time.monotonic()
    report = {"schema_version": "joint_page_adapter_probe_v1", "result": "FAIL", "synthetic_only": True,
              "ocr_performance_claim": False, "pretrained_weights": False, "device": "cpu",
              "created_at": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__,
              "pillow_version": PIL.__version__}
    report["corrections_before_first_probe"] = [
        "Tests exposed that the metric bridge initially accepted lines[*].baseline.source='ground_truth' from the generic page graph. The prediction-only bridge now rejects that source, preventing GT geometry from being returned as a prediction.",
        "Node status clarified before the probe: partial when baseline is available despite band/text failure, error when baseline is unavailable; component statuses and metric denominators remain explicit.",
        "Added independent normalized decoded-PNG hash and hashes at every forward call to verify this probe's image-to-tensor linkage.",
    ]
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        if (config["device"], config["threads"], config["dtype"], config["synthetic_only"]) != ("cpu", 1, "float64", True):
            raise ValueError("Only single-thread CPU float64 synthetic execution is permitted")
        if config["image_size_wh"] != [64, 48] or config["num_queries"] != 3:
            raise ValueError("This adapter probe is bounded to one 64x48 image with three queries")
        torch.set_num_threads(1)
        digest = hashlib.sha256(raw).hexdigest()
        report.update(config=config, config_sha256=digest)
        sources = [Path(__file__)] + [ROOT / "src/babaorum/experimental" / name for name in (
            "joint_page_adapter.py", "joint_curved_page.py", "joint_curve_repair.py", "joint_curve_sampling.py",
            "joint_curves.py", "joint_geometry.py")]
        sources += [ROOT / "src/babaorum" / name for name in ("page_graph.py", "line_geometry.py")]
        report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        encoded, image, binding = image_input(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(encoded)
        report["encoded_image"] = {"path": str(png_path), "sha256": hashlib.sha256(encoded).hexdigest(),
                                   "bytes": len(encoded), "width": image.shape[-1], "height": image.shape[-2]}
        report["adapter"] = run(config, encoded, image, digest, binding)
        if all(report["adapter"]["checks"].values()):
            report["result"] = "PASS_PAGE_GRAPH_ADAPTER_ONLY"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"result": report["result"], "output": str(args.output), "error": report.get("error"),
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["result"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
