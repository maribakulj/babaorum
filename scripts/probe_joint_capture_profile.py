#!/usr/bin/env python3
"""Predeclared synthetic profile of point and Gaussian-expected CTC gradients."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from babaorum.experimental.joint_geometry import FeatureFrame, sample_baseline_bands  # noqa: E402


def ctc_at_y(y, field, x, height, frame, config):
    baseline = torch.stack((x, torch.ones_like(x) * y), dim=-1)[None]
    sample = sample_baseline_bands(
        field, baseline, height, frame,
        output_size=(config["band_rows"], config["sequence_steps"]),
        baseline_fraction=config["baseline_fraction"],
    )
    logits = sample.features.mean(dim=2).permute(2, 0, 1)
    return F.ctc_loss(
        logits.log_softmax(-1), torch.tensor([1, 2]),
        torch.tensor([config["sequence_steps"]]), torch.tensor([2]),
        blank=0, reduction="mean", zero_infinity=False,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_capture_profile.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_capture_profile.json")
    args = parser.parse_args()
    started = time.monotonic()
    report = {
        "schema_version": "joint_capture_profile_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True,
        "device": "cpu",
        "ocr_performance_claim": False,
        "pretrained_weights": False,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        report["config"] = config
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        source_paths = (Path(__file__), ROOT / "src/babaorum/experimental/joint_geometry.py")
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }
        if config["synthetic_only"] is not True or config["device"] != "cpu" or config["threads"] != 1:
            raise ValueError("only single-thread synthetic CPU execution is allowed")
        if (config["quadrature_order"] != 15 or config["image_size_wh"][0] != 16
                or not 64 <= config["image_size_wh"][1] <= 128):
            raise ValueError("the quadrature is fixed; synthetic canvas must remain within the bounded 16x64..128 range")
        step = config["offset_step_heights"]
        count = round((config["offset_max_heights"] - config["offset_min_heights"]) / step)
        if count != 16 or not math.isclose(count * step, 4.0, abs_tol=1e-12):
            raise ValueError("offset grid must remain the 17-point predeclared grid")
        if config["gaussian_sigma_fractions_of_height"] != [0.25, 0.5, 1.0, 1.5]:
            raise ValueError("Gaussian widths are fixed by preregistration")
        torch.set_num_threads(1)
        torch.manual_seed(config["seed"])
        dtype = torch.float64
        width, page_height = config["image_size_wh"]
        h = config["height_px"]
        target_y = config["target_baseline_y"]
        evidence_y = config["evidence_center_y"]
        yy, xx = torch.meshgrid(
            torch.arange(page_height, dtype=dtype),
            torch.arange(width, dtype=dtype), indexing="ij",
        )
        field = torch.stack((
            torch.zeros_like(xx),
            5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - evidence_y).square(),
            5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - evidence_y).square(),
        ))[None]
        frame = FeatureFrame((width, page_height), (0.0, 0.0), (1.0, 1.0))
        x = torch.tensor([2.0, 13.0], dtype=dtype)
        height = torch.tensor([h], dtype=dtype)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(config["quadrature_order"])
        nodes = torch.tensor(nodes_np * math.sqrt(2.0), dtype=dtype)
        weights = torch.tensor(weights_np / math.sqrt(math.pi), dtype=dtype)
        offsets_h = [config["offset_min_heights"] + i * step for i in range(count + 1)]
        tau = config["prior_sigma_fraction_of_height"] * h
        epsilon = config["directional_gradient_epsilon"]

        def expected_ctc(mu, sigma):
            terms = [ctc_at_y(mu + sigma * z, field, x, height, frame, config) for z in nodes]
            return torch.stack(terms).dot(weights)

        def kl_mean_gradient(mu):
            return (mu - target_y) / (tau * tau)

        profiles = []
        for delta_h in offsets_h:
            y0 = torch.tensor(target_y + delta_h * h, dtype=dtype, requires_grad=True)
            point_loss = ctc_at_y(y0, field, x, height, frame, config)
            point_grad = torch.autograd.grad(point_loss, y0)[0]
            toward = (
                float(point_grad.detach()) < -epsilon if float(y0.detach()) < evidence_y
                else float(point_grad.detach()) > epsilon if float(y0.detach()) > evidence_y
                else abs(float(point_grad.detach())) <= epsilon
            )
            row = {
                "offset_heights_from_target": delta_h,
                "baseline_y": float(y0.detach()),
                "point_ctc": float(point_loss.detach()),
                "point_dloss_dy": float(point_grad.detach()),
                "point_gradient_toward_evidence": toward,
                "gaussian": [],
            }
            for sigma_fraction in config["gaussian_sigma_fractions_of_height"]:
                sigma = sigma_fraction * h
                mu = torch.tensor(target_y + delta_h * h, dtype=dtype, requires_grad=True)
                smooth_loss = expected_ctc(mu, sigma)
                smooth_grad = torch.autograd.grad(smooth_loss, mu)[0]
                kl = math.log(tau / sigma) + (sigma * sigma + (float(mu.detach()) - target_y) ** 2) / (2 * tau * tau) - 0.5
                total_grad = float(smooth_grad.detach()) + kl_mean_gradient(mu).detach().item()
                smooth_toward = (
                    float(smooth_grad.detach()) < -epsilon if float(mu.detach()) < evidence_y
                    else float(smooth_grad.detach()) > epsilon if float(mu.detach()) > evidence_y
                    else abs(float(smooth_grad.detach())) <= epsilon
                )
                total_toward = (
                    total_grad < -epsilon if float(mu.detach()) < evidence_y
                    else total_grad > epsilon if float(mu.detach()) > evidence_y
                    else abs(total_grad) <= epsilon
                )
                row["gaussian"].append({
                    "sigma_fraction_of_height": sigma_fraction,
                    "sigma_px": sigma,
                    "expected_ctc": float(smooth_loss.detach()),
                    "d_expected_ctc_dmu": float(smooth_grad.detach()),
                    "kl": kl,
                    "ctc_plus_kl": float(smooth_loss.detach()) + kl,
                    "d_ctc_plus_kl_dmu": total_grad,
                    "ctc_gradient_toward_evidence": smooth_toward,
                    "total_gradient_toward_evidence": total_toward,
                })
            profiles.append(row)

        def contiguous_interval(rows, selector):
            zero_idx = offsets_h.index(0.0)
            if not selector(rows[zero_idx]):
                return {
                    "left_offset_h": 0.0, "right_offset_h": 0.0,
                    "span_h": 0.0, "left_edge_censored": False,
                    "right_edge_censored": False,
                }
            left = right = zero_idx
            while left > 0 and selector(rows[left - 1]):
                left -= 1
            while right + 1 < len(rows) and selector(rows[right + 1]):
                right += 1
            return {
                "left_offset_h": rows[left]["offset_heights_from_target"],
                "right_offset_h": rows[right]["offset_heights_from_target"],
                "span_h": rows[right]["offset_heights_from_target"] - rows[left]["offset_heights_from_target"],
                "left_edge_censored": left == 0,
                "right_edge_censored": right == len(rows) - 1,
            }

        intervals = {"point_ctc": contiguous_interval(profiles, lambda r: r["point_gradient_toward_evidence"])}
        for index, sigma_fraction in enumerate(config["gaussian_sigma_fractions_of_height"]):
            intervals[f"gaussian_{sigma_fraction:g}h_ctc"] = contiguous_interval(
                profiles, lambda r, j=index: r["gaussian"][j]["ctc_gradient_toward_evidence"]
            )
            intervals[f"gaussian_{sigma_fraction:g}h_ctc_plus_kl"] = contiguous_interval(
                profiles, lambda r, j=index: r["gaussian"][j]["total_gradient_toward_evidence"]
            )

        all_numbers = []
        for row in profiles:
            all_numbers.extend((row["point_ctc"], row["point_dloss_dy"]))
            for item in row["gaussian"]:
                all_numbers.extend(item[key] for key in (
                    "expected_ctc", "d_expected_ctc_dmu", "kl", "ctc_plus_kl",
                    "d_ctc_plus_kl_dmu",
                ))
        checks = {
            "quadrature_weight_sum_one": abs(float(weights.sum()) - 1.0) <= 1e-14,
            "quadrature_standard_normal_second_moment_one": abs(float((weights * nodes.square()).sum()) - 1.0) <= 1e-12,
            "all_profile_values_finite": all(math.isfinite(value) for value in all_numbers),
            "all_requested_bands_in_sampler_support": len(profiles) == len(offsets_h),
        }
        report["checks"] = checks
        report["result"] = "PROFILE_COMPUTED_SYNTHETIC_ONLY" if all(checks.values()) else "PROFILE_ERROR"
        report["offset_grid_heights"] = offsets_h
        report["gauss_hermite"] = {
            "order": config["quadrature_order"], "nodes_normal": nodes.tolist(),
            "weights_normal": weights.tolist(), "weight_sum": float(weights.sum()),
            "second_moment": float((weights * nodes.square()).sum()),
        }
        report["contiguous_directional_intervals_from_target"] = intervals
        report["profiles"] = profiles
        report["interpretation"] = (
            "Intervals are discrete-grid regions around the target where the initial gradient descent direction points toward the constructed evidence peak. They do not measure convergence, capture probability, line recall, or OCR quality. The KL gradient pulls toward the target and is reported separately from expected CTC."
        )
    except Exception as exc:
        report["result"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"result": report["result"], "output": str(args.output),
                      "error": report.get("error"), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["result"].startswith("PROFILE_COMPUTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
