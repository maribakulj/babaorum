#!/usr/bin/env python3
"""One CTC-only SGD step across separate/shared/detached geometry paths.

Tiny untrained synthetic CPU mechanism, never an OCR performance comparison.
The predeclared configuration and prior negative experiments stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_causal import run_one_step_ablation  # noqa: E402
from babaorum.experimental.joint_curved_page import CurvedPageTargets, TinyCurvedJointPage  # noqa: E402


def inputs(config):
    width, height = config["image_size_wh"]
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                            torch.arange(width, dtype=torch.float64), indexing="ij")
    image = (torch.sin(xx * .21) + torch.cos(yy * .37))[None, None]
    targets = CurvedPageTargets(
        baseline_xy=[torch.tensor(points, dtype=torch.float64) for points in config["target_baselines_xy"]],
        height_px=torch.tensor(config["target_heights_px"], dtype=torch.float64),
        texts=config["target_token_ids"], order=torch.tensor(config["target_order"]),
    )
    torch.manual_seed(config["seed"])
    names = ("num_queries", "control_points", "channels", "alphabet_size", "band_rows",
             "baseline_fraction", "spacing_px", "max_samples", "penalty_scale_px")
    model = TinyCurvedJointPage(**{name: config[name] for name in names}).double()
    return model, image, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_causal_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_causal_probe.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a distinct --output to preserve prior evidence")
    started = time.monotonic()
    report = {"schema_version": "joint_causal_probe_v1", "result": "FAIL", "device": "cpu",
              "synthetic_only": True, "pretrained_weights": False, "ocr_performance_claim": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__,
              "scope": "one update of the CTC term only, not full scientific training arms S2/B3/B4",
              "prior_capture_drift_failure_preserved": "reports/joint_page_probe_initial.json"}
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        if (config["device"], config["threads"], config["dtype"], config["synthetic_only"]) != ("cpu", 1, "float64", True):
            raise ValueError("Only single-thread float64 synthetic CPU execution is permitted")
        if config["image_size_wh"] != [64, 48] or config["num_queries"] != 3 or config["control_points"] != 4:
            raise ValueError("This causal probe is bounded to one 64x48 image and three four-point queries")
        if (config["optimizer"] != "one_step_plain_SGD" or config["momentum"] != 0 or config["weight_decay"] != 0
                or not config["ctc_signal_only"] or config["geometry_loss_weight"] != 0):
            raise ValueError("Only one plain SGD step of the CTC signal is in scope")
        torch.set_num_threads(1)
        report.update(config=config, config_sha256=hashlib.sha256(raw).hexdigest())
        sources = [Path(__file__)] + [ROOT / "src/babaorum/experimental" / name for name in (
            "joint_causal.py", "joint_curved_page.py", "joint_curve_repair.py", "joint_curve_sampling.py",
            "joint_curves.py", "joint_geometry.py")]
        report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        model, image, targets = inputs(config)
        report["input_tensor"] = {"sha256": hashlib.sha256(image.numpy().tobytes(order="C")).hexdigest(),
                                  "shape": list(image.shape), "dtype": str(image.dtype), "byte_order": sys.byteorder,
                                  "hash_scope": "synthetic_tensor_bytes_C_order_not_encoded_image"}
        def execute(rate):
            result = run_one_step_ablation(
                model, image, targets, learning_rate=rate,
                minimum_change_norm=config["minimum_change_norm"],
                maximum_relative_residual=config["maximum_relative_residual"],
                residual_floor=config["residual_floor"],
            )
            return result, result.to_report()

        ablation, main_report = execute(config["learning_rate"])
        _, half_report = execute(config["half_learning_rate"])
        report["ablation"] = main_report
        half_fields = {}
        for field in ("b3_residual", "residual_delta43"):
            full_norm = float(torch.linalg.vector_norm(ablation.decomposition[field]))
            half_norm = float(torch.linalg.vector_norm(
                torch.tensor(half_report["decomposition"][field], dtype=torch.float64)
            ))
            half_fields[field] = {"full_learning_rate_absolute_l2_px": full_norm,
                                  "half_learning_rate_absolute_l2_px": half_norm,
                                  "ratio_half_over_full": half_norm / max(full_norm, config["residual_floor"]),
                                  "passes_predeclared_factor": None}
        absolute_factor = config["maximum_absolute_half_step_factor"]
        for item in half_fields.values():
            item["passes_predeclared_factor"] = item["half_learning_rate_absolute_l2_px"] <= (
                absolute_factor * item["full_learning_rate_absolute_l2_px"] + config["residual_floor"]
            )
        relative_halving = {name: {"full": main_report["decomposition"][name],
                                   "half": half_report["decomposition"][name]}
                            for name in ("relative_residual_b3", "relative_residual_delta43")}
        report["halved_step_diagnostic"] = {
            "scope": "predeclared numerical check; each arm receives one step at each fixed learning rate from the unchanged base",
            "absolute_remainders": half_fields,
            "relative_residuals_reported_separately": relative_halving,
            "relative_rule_at_each_rate": config["maximum_relative_residual"],
            "criteria": config["absolute_remainder_test"],
            "check_abs_rule": all(item["passes_predeclared_factor"] for item in half_fields.values()),
            "check_relative_main_and_half": all(main_report["checks"].values()) and all(half_report["checks"].values()),
        }
        report["halved_step_diagnostic"]["checks"] = {
            "absolute_remainders_quadratic": report["halved_step_diagnostic"]["check_abs_rule"],
            "relative_residual_under_declared_limit_at_both_rates": report["halved_step_diagnostic"]["check_relative_main_and_half"],
        }
        if all(main_report["checks"].values()) and all(half_report["checks"].values()) and all(report["halved_step_diagnostic"]["checks"].values()):
            report["result"] = "PASS_ONE_STEP_CAUSAL_MECHANISM_ONLY"
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
