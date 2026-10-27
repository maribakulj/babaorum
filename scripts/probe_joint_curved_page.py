#!/usr/bin/env python3
"""Shared-feature curved page probe with independently injected query faults.

Synthetic CPU only. Exclusions retain GT denominators and are not OCR scores.
An existing output is never overwritten, including failed runs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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

from babaorum.experimental.joint_curved_page import (  # noqa: E402
    CurvedPageTargets, TinyCurvedJointPage, curved_page_losses, export_curved_page,
)


def tensor(value, *, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def number(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else value


def norms(loss, variables):
    gradients = torch.autograd.grad(loss, tuple(variables.values()), allow_unused=True, retain_graph=True)
    return {name: {"connected": value is not None,
                   "norm": 0.0 if value is None else float(value.detach().norm())}
            for name, value in zip(variables, gradients)}


def inputs(config):
    width, height = config["image_size_wh"]
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                            torch.arange(width, dtype=torch.float64), indexing="ij")
    image = (torch.sin(xx * .21) + torch.cos(yy * .37))[None, None]
    targets = CurvedPageTargets(
        baseline_xy=[tensor(points) for points in config["target_baselines_xy"]],
        height_px=tensor(config["target_heights_px"]),
        texts=config["target_token_ids"], order=torch.tensor(config["target_order"]),
    )
    return image, targets


def setup_model(config):
    torch.manual_seed(config["seed"])
    names = ("num_queries", "control_points", "channels", "alphabet_size", "band_rows",
             "baseline_fraction", "spacing_px", "max_samples", "penalty_scale_px")
    model = TinyCurvedJointPage(**{name: config[name] for name in names}).double()
    if model.min_increment_px != config["min_increment_px"] or model.min_segment_length_px != config["min_segment_length_px"]:
        raise ValueError("Model numeric geometry thresholds differ from the declared probe configuration")
    return model


def natural_forward(config, model, image, targets):
    results, predictions, gradients = {}, {}, {}
    for name, detached in (("coupled", False), ("detached", True)):
        pred = model(image, detach_geometry=detached)
        losses = curved_page_losses(pred, targets)
        results[name], predictions[name] = losses, pred
        gradients[name] = norms(losses.diagnostics["ctc_signal"], {
            "baseline": pred.baseline_xy, "height": pred.height_px,
            "geometry_head": model.geometry_head.weight,
            "encoder": next(model.encoder.parameters()), "reader": model.ctc_head.weight,
        })
    minimum = config["gradient_minimum_norm"]
    checks = {
        "all_natural_queries_included": all(result.counts["included_ctc"] == config["num_queries"] for result in results.values()),
        "same_matching": results["coupled"].matching == results["detached"].matching,
        "same_forward_logits": all(torch.equal(first.ctc_logits, second.ctc_logits)
                                    for first, second in zip(predictions["coupled"].queries, predictions["detached"].queries)),
        "geometry_gradients_live": all(gradients["coupled"][name]["norm"] > minimum for name in ("baseline", "height", "geometry_head")),
        "geometry_gradients_stopped": all(not gradients["detached"][name]["connected"] for name in ("baseline", "height", "geometry_head")),
        "encoder_reader_both_active": all(gradients[mode][name]["norm"] > minimum for mode in gradients for name in ("encoder", "reader")),
    }
    return {"checks": checks, "gradients": gradients,
            "counts": {name: result.counts for name, result in results.items()},
            "losses": {name: {key: number(value) for key, value in result.losses.items()} for name, result in results.items()},
            "parameters": sum(parameter.numel() for parameter in model.parameters())}


def fault_case(config, model, image, targets, kind):
    baseline = tensor(config["predicted_baselines_xy"])
    height = tensor(config["predicted_heights_px"], grad=True)
    query = config["fault_query"]
    expected = {
        "collapsed": "excluded_degenerate_curve", "near_collapsed": "excluded_degenerate_curve",
        "out_of_bounds": "excluded_invalid_geometry", "sampling_cap": "excluded_sampling_cap_exceeded",
        "infeasible_ctc": "excluded_infeasible_length",
        "double_cap": "excluded_sampling_cap_exceeded",
        "double_cap_in_page": "excluded_sampling_cap_exceeded",
    }
    if kind in ("collapsed", "near_collapsed"):
        baseline[query] = tensor(config["collapse_point_xy"])
        if kind == "near_collapsed":
            baseline[query, :, 0] += torch.arange(config["control_points"]) * config["near_collapse_dx_px"]
    elif kind == "out_of_bounds":
        baseline[query, :, 1] = config["oob_baseline_y"]
    elif kind == "sampling_cap":
        baseline[query] = tensor(config["cap_baseline_xy"])
    elif kind == "infeasible_ctc":
        baseline[query] = tensor(config["infeasible_baseline_xy"])
    elif kind == "double_cap":
        extension = config["review_extension"]
        baseline[query] = tensor(extension["double_cap_baseline_xy"])
        height = height.detach().clone()
        height[query] = extension["double_cap_height_px"]
        height.requires_grad_()
    elif kind == "double_cap_in_page":
        extension = config["in_page_double_cap"]
        baseline[query] = tensor(extension["baseline_xy"])
        height = height.detach().clone()
        height[query] = extension["predicted_height_px"]
        height.requires_grad_()
        target_heights = targets.height_px.clone()
        target_heights[query] = extension["target_height_px"]
        targets = replace(targets, height_px=target_heights)
    else:
        raise ValueError(f"Unknown fault {kind}")
    baseline.requires_grad_()
    prediction = model(image, baseline_override=baseline, height_override=height)
    result = curved_page_losses(prediction, targets)
    n = len(targets.baseline_xy)
    charge = config["max_samples"] * math.log(config["alphabet_size"]) / n
    signal = result.diagnostics["ctc_signal"]
    other_gradients = norms(signal, {
        f"query_{index}_ctc": item.ctc_logits
        for index, item in enumerate(prediction.queries) if index != query
    })
    presence_grad = torch.autograd.grad(result.losses["presence"], prediction.presence_logits, retain_graph=True)[0]
    geometry_objective = sum(result.losses[key] for key in ("geometry", "repair", "boundary", "fold"))
    geometry_grad = torch.autograd.grad(geometry_objective, baseline, retain_graph=True)[0]
    minimum = config["gradient_minimum_norm"]
    checks = {
        "two_other_lines_keep_ctc": result.counts["included_ctc"] == 2,
        "one_line_exclusion_counted": result.counts["excluded_ctc"] == 1 and result.line_statuses[query] == expected[kind],
        "all_gt_kept_in_denominator": result.counts["ctc_denominator"] == n and result.counts["matched"] == n,
        "token_support_preserved": result.counts["supported_target_tokens"] + result.counts["excluded_target_tokens"] == result.counts["ground_truth_tokens"],
        "declared_constant_charge": math.isclose(result.diagnostics["ctc_exclusion_charge"], charge, rel_tol=1e-12),
        "charged_ctc_equals_signal_plus_charge": math.isclose(number(result.losses["ctc"]), number(signal) + charge, rel_tol=1e-12),
        "other_queries_ctc_gradients_live": all(item["norm"] > minimum for item in other_gradients.values()),
        "excluded_presence_gradient_live": abs(float(presence_grad[query])) > minimum,
        "excluded_geometry_gradient_live": float(geometry_grad[query].norm()) > minimum,
        "page_loss_finite": all(math.isfinite(number(value)) for value in result.losses.values()),
    }
    diagnostics = {key: number(value) for key, value in result.diagnostics.items()}
    report = {"checks": checks, "counts": result.counts, "matching": result.matching,
              "line_statuses": result.line_statuses, "geometry_statuses": result.geometry_statuses,
              "losses": {key: number(value) for key, value in result.losses.items()},
              "diagnostics": diagnostics, "other_ctc_gradients": other_gradients,
              "excluded_presence_gradient": float(presence_grad[query]),
              "excluded_geometry_gradient_norm": float(geometry_grad[query].norm())}
    if kind in ("collapsed", "near_collapsed"):
        repair_grad = torch.autograd.grad(result.losses["repair"], baseline, retain_graph=True)[0]
        repaired = baseline.detach() - config["repair_step_learning_rate"] * repair_grad
        distances = torch.linalg.vector_norm(repaired[query, 1:] - repaired[query, :-1], dim=-1)
        checks["repair_step_separates_all_controls"] = bool((distances > config["min_segment_length_px"]).all())
        report["one_step_minimum_segment_px"] = float(distances.min())
        report["repair_gradient_xy"] = repair_grad[query].tolist()
    if kind in ("double_cap", "double_cap_in_page"):
        repair_grad, height_grad = torch.autograd.grad(result.losses["repair"], (baseline, height), retain_graph=True)
        extension = config["in_page_double_cap"] if kind == "double_cap_in_page" else config["review_extension"]
        lr = extension["repair_step_learning_rate"]
        repaired = baseline.detach() - lr * repair_grad
        old_length = torch.linalg.vector_norm(baseline[query, 1:] - baseline[query, :-1], dim=-1).sum()
        new_length = torch.linalg.vector_norm(repaired[query, 1:] - repaired[query, :-1], dim=-1).sum()
        checks.update({
            "reader_and_geometry_caps_counted": result.counts["excluded_sampling_cap_exceeded"] == 1 and result.counts["excluded_geometry_sampling_cap"] == 1,
            "explicit_cap_repair": result.geometry_statuses[query] == "repair_cap" and result.counts["repair_cap_geometry_lines"] == 1,
            "fixed_step_shortens_endpoint_span": float(new_length) < float(old_length.detach()),
            "height_gradient_toward_target": float(height_grad[query]) > minimum,
        })
        report["repair_step"] = {"learning_rate": lr, "length_before_px": float(old_length.detach()),
                                 "length_after_px": float(new_length), "height_gradient": float(height_grad[query]),
                                 "scope": "excess_endpoint_span_only_not_interior_excursion_repair"}
        if kind == "double_cap_in_page":
            width, page_height = config["image_size_wh"]
            points = baseline[query].detach()
            fraction = config["baseline_fraction"]
            h = float(height[query].detach())
            gt_curve = targets.baseline_xy[query]
            gt_length = float(torch.linalg.vector_norm(gt_curve[1:] - gt_curve[:-1], dim=-1).sum())
            gt_steps = math.ceil(gt_length / (float(targets.height_px[query]) * 0.5)) + 1
            checks.update({
                "straight_band_wholly_inside_page": bool((points[:, 1] == points[0, 1]).all()) and bool(
                    ((points[:, 0] >= 0) & (points[:, 0] <= width - 1)
                     & (points[:, 1] - fraction * h >= 0)
                     & (points[:, 1] + (1 - fraction) * h <= page_height - 1)).all()),
                "target_within_geometry_sampling_budget": gt_steps <= 256,
            })
            report["gt_geometry_required_steps"] = gt_steps
            report["fixture_scope"] = extension["reason"]
        else:
            report["combined_faults"] = "outside page plus both sample caps; reader cap takes precedence and no validity grid is computed"
    if kind == "collapsed":
        exported = export_curved_page(
            prediction, image_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            page_key="synthetic:curved-page-29", document_key="synthetic:curved-issue",
        )
        exported["image_sha256_scope"] = "synthetic_float64_tensor_bytes_C_order_not_encoded_image"
        exported["input_tensor"] = {"shape": list(image.shape), "dtype": str(image.dtype), "byte_order": sys.byteorder}
        report["diagnostic_export"] = exported
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_curved_page_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_curved_page_probe.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a distinct --output to preserve prior evidence")
    start = time.monotonic()
    report = {"schema_version": "joint_curved_page_probe_v1", "result": "FAIL", "device": "cpu",
              "synthetic_only": True, "ocr_performance_claim": False, "pretrained_weights": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__}
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        if config["device"] != "cpu" or config["threads"] != 1 or config["dtype"] != "float64" or not config["synthetic_only"]:
            raise ValueError("This probe only permits single-thread synthetic CPU float64 execution")
        if config["num_queries"] != 3 or config["image_size_wh"] != [64, 48] or config["max_samples"] > 256:
            raise ValueError("Probe dimensions must stay within the declared tiny integration scope")
        if config["exclusion_charge_rule"] != "max_samples_times_log_alphabet_size":
            raise ValueError("Unexpected CTC exclusion charge rule")
        torch.set_num_threads(1)
        report["config"] = config
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        sources = [Path(__file__)] + [ROOT / "src/babaorum/experimental" / name for name in (
            "joint_curved_page.py", "joint_curve_repair.py", "joint_curve_sampling.py", "joint_curves.py", "joint_geometry.py")]
        report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        report["fault_injection_scope"] = {
            "faults_use_geometry_overrides": True,
            "natural_exclusion_frequency_measured": False,
            "natural_constraints": "positive x increments prevent collapse at ordinary pixel precision; band validity and sample caps are still checked",
            "no_natural_training_claim": True,
        }
        image, targets = inputs(config)
        model = setup_model(config)
        report["natural_forward"] = natural_forward(config, model, image, targets)
        report["faults"] = {kind: fault_case(config, model, image, targets, kind) for kind in (
            "collapsed", "near_collapsed", "out_of_bounds", "sampling_cap", "infeasible_ctc")}
        if "review_extension" in config:
            report["faults"]["double_cap"] = fault_case(config, model, image, targets, "double_cap")
            report["changes_after_initial_v1_observation"] = [
                "P1: endpoint/height fallback for geometry sampling cap; new double-cap fixture and fixed-step contraction check declared after review",
                "P2: typed cap and numerical exceptions replace matching exception message strings",
                "P3: explicit fault-injection scope; no natural exclusion-frequency or training claim",
            ]
        if "in_page_double_cap" in config:
            report["faults"]["double_cap_in_page"] = fault_case(config, model, image, targets, "double_cap_in_page")
            report["changes_after_initial_v1_observation"].append(
                "Second review: additional double-cap fixture fully inside the page, with explicitly subpixel synthetic GT height; prior reviewed report preserved"
            )
        passed = all(report["natural_forward"]["checks"].values()) and all(
            value for case in report["faults"].values() for value in case["checks"].values())
        report["result"] = "PASS_CURVED_PAGE_MECHANISMS_ONLY" if passed else "FAIL"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"result": report["result"], "output": str(args.output), "error": report.get("error"),
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["result"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
