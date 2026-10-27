#!/usr/bin/env python3
"""Exploratory multi-start trajectories for point and Gaussian-band CTC."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from babaorum.experimental.joint_geometry import FeatureFrame  # noqa: E402
from scripts.probe_joint_capture_profile import ctc_at_y  # noqa: E402


def kl_normal(mu, sigma, target_y, prior_sigma):
    return math.log(prior_sigma / sigma) + (
        sigma * sigma + (mu - target_y).square()
    ) / (2 * prior_sigma * prior_sigma) - 0.5


def run(config):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    dtype = torch.float64
    width, page_height = config["image_size_wh"]
    target_y, evidence_y = config["target_baseline_y"], config["evidence_center_y"]
    h = config["height_px"]
    yy, xx = torch.meshgrid(
        torch.arange(page_height, dtype=dtype), torch.arange(width, dtype=dtype), indexing="ij"
    )
    field = torch.stack((
        torch.zeros_like(xx),
        5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - evidence_y).square(),
        5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - evidence_y).square(),
    ))[None]
    frame = FeatureFrame((width, page_height), (0.0, 0.0), (1.0, 1.0))
    x = torch.tensor([2.0, 13.0], dtype=dtype)
    height = torch.tensor([h], dtype=dtype)
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(15)
    nodes = torch.tensor(nodes_np * math.sqrt(2.0), dtype=dtype)
    weights = torch.tensor(weights_np / math.sqrt(math.pi), dtype=dtype)
    sigma = config["fixed_q_sigma_fraction_of_height"] * h
    tau = config["prior_sigma_fraction_of_height"] * h

    def expected_ctc(mu):
        values = [
            ctc_at_y(mu + sigma * node, field, x, height, frame, config)
            for node in nodes
        ]
        return torch.stack(values).dot(weights)

    result = {}
    for offset_h in config["start_offsets_heights"]:
        start_y = target_y + offset_h * h
        arms = {}
        point_y = torch.nn.Parameter(torch.tensor(start_y, dtype=dtype))
        point_optimizer = torch.optim.SGD([point_y], lr=config["learning_rate"])
        q_mu = torch.nn.Parameter(torch.tensor(start_y, dtype=dtype))
        q_optimizer = torch.optim.SGD([q_mu], lr=config["learning_rate"])
        trajectories = {"point_ctc": [], "gaussian_expected_ctc_plus_kl": []}

        def record(step):
            point_loss = ctc_at_y(point_y, field, x, height, frame, config)
            q_ctc = expected_ctc(q_mu)
            q_kl = kl_normal(q_mu, sigma, target_y, tau)
            trajectories["point_ctc"].append({
                "step": step, "y": float(point_y.detach()), "ctc": float(point_loss.detach())
            })
            trajectories["gaussian_expected_ctc_plus_kl"].append({
                "step": step, "mu": float(q_mu.detach()),
                "expected_ctc": float(q_ctc.detach()), "kl": float(q_kl.detach()),
                "objective": float((q_ctc + q_kl).detach()),
            })

        record(0)
        record_steps = set(config["record_steps"])
        for step in range(1, config["steps"] + 1):
            point_optimizer.zero_grad()
            point_loss = ctc_at_y(point_y, field, x, height, frame, config)
            point_loss.backward()
            point_optimizer.step()

            q_optimizer.zero_grad()
            ctc_loss = expected_ctc(q_mu)
            objective = ctc_loss + kl_normal(q_mu, sigma, target_y, tau)
            objective.backward()
            q_optimizer.step()
            if step in record_steps:
                record(step)

        final_point = point_y.detach().clone().requires_grad_(True)
        point_final_loss = ctc_at_y(final_point, field, x, height, frame, config)
        point_final_grad = torch.autograd.grad(point_final_loss, final_point)[0]
        final_mu = q_mu.detach().clone().requires_grad_(True)
        q_final_ctc = expected_ctc(final_mu)
        q_final_kl = kl_normal(final_mu, sigma, target_y, tau)
        q_final_grad = torch.autograd.grad(q_final_ctc + q_final_kl, final_mu)[0]
        arms["point_ctc"] = {
            "final_y": float(point_y.detach()),
            "final_ctc": float(point_final_loss.detach()),
            "final_dloss_dy": float(point_final_grad.detach()),
            "displacement_from_target_h": (float(point_y.detach()) - target_y) / h,
        }
        arms["gaussian_expected_ctc_plus_kl"] = {
            "final_mu": float(q_mu.detach()),
            "final_sigma_px": sigma,
            "final_expected_ctc": float(q_final_ctc.detach()),
            "final_ctc_at_mean": float(ctc_at_y(final_mu, field, x, height, frame, config).detach()),
            "final_kl": float(q_final_kl.detach()),
            "final_dobjective_dmu": float(q_final_grad.detach()),
            "displacement_from_target_h": (float(q_mu.detach()) - target_y) / h,
        }
        finite = all(
            math.isfinite(float(value))
            for arm in trajectories.values()
            for row in arm
            for value in row.values()
            if isinstance(value, (float, int))
        )
        finite = finite and all(
            math.isfinite(value)
            for arm in arms.values()
            for value in arm.values()
            if isinstance(value, (float, int))
        )
        result[f"start_{offset_h:+g}h"] = {
            "start_y": start_y,
            "trajectories": trajectories,
            "final": arms,
            "all_values_finite": finite,
        }
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_capture_trajectories.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_capture_trajectories.json")
    args = parser.parse_args()
    started = time.monotonic()
    report = {
        "schema_version": "joint_capture_trajectories_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True, "device": "cpu", "ocr_performance_claim": False,
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        report["config"] = config
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        sources = (
            Path(__file__), ROOT / "scripts/probe_joint_capture_profile.py",
            ROOT / "src/babaorum/experimental/joint_geometry.py",
        )
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }
        if config["synthetic_only"] is not True or config["device"] != "cpu" or config["threads"] != 1:
            raise ValueError("only single-thread synthetic CPU execution is allowed")
        if config["steps"] != 40 or config["record_steps"] != [0, 1, 5, 10, 20, 40]:
            raise ValueError("optimizer and observation times are fixed by this exploratory lot")
        if config["start_offsets_heights"] != [-1.0, 0.0, 0.25, 0.5, 1.0, 1.5]:
            raise ValueError("the exploratory start grid is fixed")
        if config["fixed_q_sigma_fraction_of_height"] != 0.5:
            raise ValueError("the Gaussian width is fixed")
        torch.set_num_threads(1)
        report["trajectories"] = run(config)
        report["checks"] = {
            "all_start_arms_finite": all(
                item["all_values_finite"] for item in report["trajectories"].values()
            ),
            "all_six_declared_starts_present": len(report["trajectories"]) == 6,
        }
        report["result"] = (
            "COMPUTED_EXPLORATORY_TRAJECTORIES"
            if all(report["checks"].values()) else "FAIL_NONFINITE_OR_MISSING_ARM"
        )
        report["interpretation"] = (
            "Exploratory multi-start paths in one fixed smooth synthetic field. Start offsets were chosen after inspecting the T4b profile. These paths do not estimate capture probability, OCR performance, or generalization."
        )
    except Exception as exc:
        report["result"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, args.output)
    print(json.dumps({"result": report["result"], "output": str(args.output),
                      "error": report.get("error"), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["result"].startswith("COMPUTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
