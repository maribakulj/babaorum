#!/usr/bin/env python3
"""Fixed synthetic CPU checks of curved geometry and masked sampling.

Reports preserve failure and cannot overwrite an earlier execution. This is
neither an OCR benchmark nor an experiment on the historical page test sets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_geometry import FeatureFrame  # noqa: E402
from babaorum.experimental.joint_curves import curve_geometry_loss, resample_arclength  # noqa: E402
from babaorum.experimental.joint_curve_sampling import (  # noqa: E402
    curved_band_grid, sample_curved_band, masked_curve_ctc_loss,
)


def tensor(value, *, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def norm(value):
    return 0.0 if value is None else float(value.detach().norm())


def scalar(value):
    return float(value.detach())


def densify(points, fractions):
    pieces = []
    for start, end, local in zip(points[:-1], points[1:], fractions):
        for fraction in local:
            pieces.append(start + fraction * (end - start))
    return torch.stack(pieces + [points[-1]])


def geometric_checks(config):
    baseline = tensor(config["baseline_xy"])
    target = baseline + tensor([0., config["target_offset_y_px"]])
    dense_pred = densify(baseline, config["densify_fractions_by_segment"])
    dense_target = densify(target, config["densify_fractions_by_segment"])
    options = {"target_height_px": tensor(config["height_px"]),
               "spacing_px": config["spacing_px"], "orientation": "ltr",
               "max_samples": config["max_samples"]}
    sparse = curve_geometry_loss(baseline, target, **options)
    dense = curve_geometry_loss(dense_pred, dense_target, **options)
    factor = config["scale_factor"]
    scaled = curve_geometry_loss(baseline * factor, target * factor,
                                target_height_px=options["target_height_px"] * factor,
                                spacing_px=options["spacing_px"] * factor, orientation="ltr")
    reference = tensor([[1., 4.], [21., 4.]])
    translation = config["normal_translation_px"]
    translated = curve_geometry_loss(
        reference + tensor([0., translation]), reference,
        target_height_px=tensor(config["normal_translation_gt_height_px"]),
        spacing_px=config["spacing_px"], orientation="ltr",
    )
    reverse = curve_geometry_loss(reference.flip(0), reference, **options)
    partial = curve_geometry_loss(tensor([[7., 4.], [15., 4.]]), reference, **options)
    arc_sparse = resample_arclength(baseline, spacing_px=config["spacing_px"])
    arc_dense = resample_arclength(dense_pred, spacing_px=config["spacing_px"])
    tolerance = config["invariance_absolute_tolerance"]
    differences = {
        "loss_densification": abs(scalar(sparse.total) - scalar(dense.total)),
        "loss_scale": abs(scalar(sparse.total) - scalar(scaled.total)),
        "arc_densification_max_px": float((arc_sparse.points_xy - arc_dense.points_xy).abs().max()),
    }
    checks = {
        "loss_invariant_densification": differences["loss_densification"] <= tolerance,
        "normalized_loss_invariant_scale": differences["loss_scale"] <= tolerance,
        "arc_invariant_densification": differences["arc_densification_max_px"] <= tolerance,
        "normal_distance_analytic": abs(scalar(translated.curve) - translation / config["normal_translation_gt_height_px"]) <= tolerance,
        "reverse_penalized_by_ordered_endpoints": scalar(reverse.curve) <= tolerance and scalar(reverse.endpoints) > 0,
        "partial_coverage_penalized": scalar(partial.pred_to_target) <= tolerance and scalar(partial.target_to_pred) > 0,
    }
    return {"checks": checks, "differences": differences,
            "sampling_points": len(arc_sparse.points_xy),
            "normal_translation_curve_loss": scalar(translated.curve),
            "reverse_endpoint_loss": scalar(reverse.endpoints),
            "partial_pred_to_target": scalar(partial.pred_to_target),
            "partial_target_to_pred": scalar(partial.target_to_pred)}


def make_field(width, height):
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                            torch.arange(width, dtype=torch.float64), indexing="ij")
    return torch.stack((torch.zeros_like(xx),
                        4. - .035 * (xx - 8.).square() - .07 * (yy - 10.).square(),
                        4. - .035 * (xx - 23.).square() - .07 * (yy - 10.).square()))[None]


def ctc_loss(sample):
    # Validate eligibility before converting the sampled evidence into logits.
    sample.as_ctc_sequence()
    logits = sample.features.mean(2).permute(2, 0, 1)
    return torch.nn.functional.ctc_loss(
        logits.log_softmax(-1), torch.tensor([1, 2]), torch.tensor([len(logits)]),
        torch.tensor([2]), blank=0, reduction="mean", zero_infinity=False,
    )


def sampling_checks(config):
    width, height = config["page_size_wh"]
    frame = FeatureFrame((width, height), (0., 0.), (1., 1.))
    baseline = tensor(config["baseline_xy"], grad=True)
    line_height = tensor(config["height_px"], grad=True)
    field = make_field(width, height).requires_grad_()
    grid_options = {"band_rows": config["band_rows"], "spacing_px": config["spacing_px"],
                    "baseline_fraction": config["baseline_fraction"], "orientation": "ltr",
                    "max_samples": config["max_samples"],
                    "penalty_scale_px": config["penalties"]["normal_scale_px"]}
    grid = curved_band_grid(baseline, line_height, frame, (height, width), **grid_options)
    live = sample_curved_band(field, grid, frame)
    detached = sample_curved_band(field, grid, frame, detach_geometry=True)
    live_loss = ctc_loss(live)
    stopped_loss = ctc_loss(detached)
    infeasible_logits = live.features.mean(2).permute(2, 0, 1)
    infeasible = masked_curve_ctc_loss(infeasible_logits, live, [1] * len(infeasible_logits))
    variables = (baseline, line_height, field)
    gradients_live = torch.autograd.grad(live_loss, variables, allow_unused=True, retain_graph=True)
    gradients_stopped = torch.autograd.grad(stopped_loss, variables, allow_unused=True, retain_graph=True)
    names = ("baseline", "height", "features")
    gradients = {mode: {name: {"connected": gradient is not None, "norm": norm(gradient)}
                        for name, gradient in zip(names, values)}
                 for mode, values in (("coupled", gradients_live), ("detached", gradients_stopped))}
    dense_baseline = densify(baseline, config["densify_fractions_by_segment"])
    dense_grid = curved_band_grid(dense_baseline, line_height, frame, (height, width), **grid_options)
    dense_sample = sample_curved_band(field, dense_grid, frame)
    grid_difference = float((grid.pixel_xy - dense_grid.pixel_xy).abs().max().detach())
    feature_difference = float((live.features - dense_sample.features).abs().max().detach())
    minimum = config["gradient_minimum_norm"]
    tolerance = config["invariance_absolute_tolerance"]
    checks = {
        "valid_baseline": bool(grid.line_valid),
        "same_ctc_forward_both_modes": bool(torch.equal(live.features, detached.features)),
        "ctc_to_baseline_and_height": all(gradients["coupled"][name]["norm"] > minimum for name in ("baseline", "height")),
        "ctc_geometry_stopped": all(not gradients["detached"][name]["connected"] for name in ("baseline", "height")),
        "ctc_to_features_both_modes": all(gradients[mode]["features"]["norm"] > minimum for mode in gradients),
        "grid_invariant_densification": grid_difference <= tolerance,
        "features_invariant_densification": feature_difference <= tolerance,
        "predicted_length_infeasible_ctc_excluded_and_counted": (
            infeasible.status == "excluded_infeasible_length"
            and infeasible.excluded_lines == 1 and infeasible.included_lines == 0
            and scalar(infeasible.loss) == 0.0
        ),
    }
    return {"checks": checks, "ctc_gradients": gradients,
            "grid_densification_max_error_px": grid_difference,
            "feature_densification_max_error": feature_difference,
            "ctc_loss": scalar(live_loss), "sample_shape": list(live.features.shape)}


def validity_checks(config):
    width, height = config["page_size_wh"]
    frame = FeatureFrame((width, height), (0., 0.), (1., 1.))
    baseline = tensor(config["oob_baseline_xy"], grad=True)
    line_height = tensor(config["oob_height_px"], grad=True)
    grid = curved_band_grid(
        baseline, line_height, frame, (height, width), band_rows=config["band_rows"],
        spacing_px=config["spacing_px"], baseline_fraction=config["baseline_fraction"],
        orientation="ltr", penalty_scale_px=config["penalties"]["oob_scale_px"],
    )
    sampled = sample_curved_band(make_field(width, height), grid, frame)
    logits = sampled.features.mean(2).permute(2, 0, 1)
    excluded = masked_curve_ctc_loss(logits, sampled, [1, 2])
    boundary_gradient = torch.autograd.grad(grid.boundary_loss, baseline)[0]
    refused = False
    try:
        sampled.as_ctc_sequence()
    except ValueError:
        refused = True
    fixture = config["fold_fixture"]
    arc_width, arc_height = fixture["page_size_wh"]
    arc_frame = FeatureFrame((arc_width, arc_height), (0., 0.), (1., 1.))
    angles = torch.linspace(-math.pi / 2, math.pi / 2, fixture["control_points"], dtype=torch.float64)
    radial = torch.stack((angles.cos(), angles.sin()), -1)
    center = tensor(fixture["center_xy"])
    arcs = {}
    for name, radius, band_height in (
        ("folded", fixture["radius_px"], fixture["height_px"]),
        ("wide_radius", config["valid_arc_fixture"]["radius_px"], config["valid_arc_fixture"]["height_px"]),
    ):
        arc_grid = curved_band_grid(center + radius * radial, tensor(band_height), arc_frame,
                                   (arc_height, arc_width), band_rows=config["band_rows"],
                                   spacing_px=fixture["spacing_px"], baseline_fraction=fixture["baseline_fraction"],
                                   orientation="ttb", penalty_scale_px=band_height)
        arcs[name] = {"line_valid": bool(arc_grid.line_valid),
                      "all_cells_valid": bool(arc_grid.cell_valid.all()),
                      "fold_loss": scalar(arc_grid.fold_loss)}
    checks = {
        "oob_line_invalid": not bool(grid.line_valid),
        "oob_samples_explicitly_masked": not bool(grid.sample_valid.all()),
        "strict_ctc_conversion_refuses_invalid_line": refused,
        "training_excludes_invalid_line_without_raising": (
            excluded.status == "excluded_invalid_geometry" and excluded.included_lines == 0
            and excluded.excluded_lines == 1 and scalar(excluded.loss) == 0.0
        ),
        "boundary_loss_positive": scalar(grid.boundary_loss) > 0,
        "boundary_gradient_moves_inward": float(boundary_gradient[:, 1].sum()) < 0,
        "small_radius_fold_detected": not arcs["folded"]["line_valid"],
        "large_radius_valid": arcs["wide_radius"]["line_valid"],
    }
    return {"checks": checks, "boundary_loss": scalar(grid.boundary_loss),
            "training_ctc_status": excluded.status, "training_excluded_lines": excluded.excluded_lines,
            "boundary_gradient_sum_y": float(boundary_gradient[:, 1].sum()),
            "invalid_sample_count": int((~grid.sample_valid).sum()), "arcs": arcs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_curves_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_curves_probe.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a distinct --output to preserve prior evidence")
    started = time.monotonic()
    report = {"schema_version": "joint_curves_probe_v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "synthetic_only": True, "ocr_performance_claim": False,
              "torch_version": torch.__version__, "result": "FAIL"}
    report["changes_after_initial_observation"] = [
        "Review K1: a valid geometry with insufficient predicted CTC time steps is now excluded and counted, not raised as an iteration-ending error.",
        "A dedicated infeasible-length exclusion check was added after the initial passing run; no configuration or original check threshold changed.",
    ]
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        if config["device"] != "cpu" or config["threads"] != 1 or not config["synthetic_only"]:
            raise ValueError("Only single-thread synthetic CPU execution is permitted")
        if config["max_samples"] > 256 or config["page_size_wh"] != [32, 24]:
            raise ValueError("This probe is bounded to its declared tiny geometry")
        if config["spacing_px"] != config["spacing_fraction_of_gt_height"] * config["height_px"]:
            raise ValueError("Primary-curve spacing must match the declared GT-height fraction")
        torch.set_num_threads(1)
        report["config"] = config
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        sources = (Path(__file__), ROOT / "src/babaorum/experimental/joint_curves.py",
                   ROOT / "src/babaorum/experimental/joint_curve_sampling.py",
                   ROOT / "src/babaorum/experimental/joint_geometry.py")
        report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        report["geometry"] = geometric_checks(config)
        report["sampling"] = sampling_checks(config)
        report["validity"] = validity_checks(config)
        passed = all(check for section in ("geometry", "sampling", "validity")
                     for check in report[section]["checks"].values())
        report["result"] = "PASS_CURVE_MECHANISMS_ONLY" if passed else "FAIL"
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
