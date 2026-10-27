#!/usr/bin/env python3
"""Exploratory CPU toy probe of a Gaussian soft band with a KL geometry prior.

This compares point CTC optimization with expected CTC under a Gaussian offset
distribution plus KL(q || target-centered prior). It is a constructed mechanism
check only; it consumes no page data or pretrained reader.
"""

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
    sampled = sample_baseline_bands(
        field,
        baseline,
        height,
        frame,
        output_size=(config["band_rows"], config["sequence_steps"]),
        baseline_fraction=config["baseline_fraction"],
    )
    logits = sampled.features.mean(dim=2).permute(2, 0, 1)
    target = torch.tensor([1, 2], dtype=torch.long)
    return F.ctc_loss(
        logits.log_softmax(-1), target,
        torch.tensor([config["sequence_steps"]]), torch.tensor([2]),
        blank=0, reduction="mean", zero_infinity=False,
    )


def gaussian_kl(mu, sigma, target_y, prior_sigma):
    return (
        torch.log(prior_sigma / sigma)
        + (sigma.square() + (mu - target_y).square()) / (2.0 * prior_sigma**2)
        - 0.5
    )


def run_probe(config):
    torch.set_num_threads(config["threads"])
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
    # Smooth synthetic logits with two character-specific horizontal peaks.
    field = torch.stack((
        torch.zeros_like(xx),
        5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - evidence_y).square(),
        5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - evidence_y).square(),
    ))[None]
    frame = FeatureFrame((width, page_height), (0.0, 0.0), (1.0, 1.0))
    x = torch.tensor([2.0, 13.0], dtype=dtype)
    height = torch.tensor([h], dtype=dtype)
    point_y = torch.nn.Parameter(torch.tensor(target_y, dtype=dtype))
    point_opt = torch.optim.SGD([point_y], lr=config["learning_rate"])
    q_mu = torch.nn.Parameter(torch.tensor(target_y, dtype=dtype))
    initial_sigma = config["initial_q_sigma_fraction_of_height"] * h
    q_log_sigma = torch.nn.Parameter(torch.tensor(math.log(initial_sigma), dtype=dtype))
    q_opt = torch.optim.SGD([q_mu, q_log_sigma], lr=config["learning_rate"])

    nodes_np, weights_np = np.polynomial.hermite.hermgauss(config["quadrature_order"])
    nodes = torch.tensor(nodes_np * math.sqrt(2.0), dtype=dtype)
    weights = torch.tensor(weights_np / math.sqrt(math.pi), dtype=dtype)

    def expected_ctc(mu, sigma):
        return torch.stack([
            ctc_at_y(mu + sigma * node, field, x, height, frame, config)
            for node in nodes
        ]).dot(weights)

    point_initial = ctc_at_y(point_y, field, x, height, frame, config)
    point_initial_grad = torch.autograd.grad(point_initial, point_y, retain_graph=True)[0]
    q_initial_sigma = q_log_sigma.exp()
    q_initial_ctc = expected_ctc(q_mu, q_initial_sigma)
    q_initial_grad = torch.autograd.grad(q_initial_ctc, q_mu, retain_graph=True)[0]
    prior_sigma = config["prior_sigma_fraction_of_height"] * h
    q_initial_kl = gaussian_kl(q_mu, q_initial_sigma, target_y, prior_sigma)

    for _ in range(config["steps"]):
        point_opt.zero_grad()
        point_loss = ctc_at_y(point_y, field, x, height, frame, config)
        point_loss.backward()
        point_opt.step()

        q_opt.zero_grad()
        sigma = q_log_sigma.exp()
        expected = expected_ctc(q_mu, sigma)
        kl = gaussian_kl(q_mu, sigma, target_y, prior_sigma)
        objective = expected + kl
        objective.backward()
        q_opt.step()

    point_final = ctc_at_y(point_y.detach(), field, x, height, frame, config)
    point_final_y = point_y.detach().clone().requires_grad_(True)
    point_final_for_grad = ctc_at_y(point_final_y, field, x, height, frame, config)
    point_final_grad = torch.autograd.grad(point_final_for_grad, point_final_y)[0]
    q_final_sigma = q_log_sigma.detach().exp()
    q_final_mu = q_mu.detach().clone().requires_grad_(True)
    q_final_ctc = expected_ctc(q_final_mu, q_final_sigma)
    q_final_point_ctc = ctc_at_y(q_final_mu, field, x, height, frame, config)
    q_final_kl = gaussian_kl(q_final_mu, q_final_sigma, target_y, prior_sigma)
    q_final_grad_mu = torch.autograd.grad(q_final_ctc, q_final_mu, retain_graph=True)[0]

    point_displacement = abs(float(point_y.detach()) - target_y)
    q_displacement = abs(float(q_mu.detach()) - target_y)
    checks = {
        "variational_moves_farther_than_point_by_declared_margin": (
            q_displacement - point_displacement
            >= config["declared_checks"]["minimum_variational_minus_point_displacement_px"]
        ),
        "point_control_is_near_stationary_at_final_position": (
            abs(float(point_final_grad))
            <= config["declared_checks"]["maximum_point_gradient_abs_at_final_point"]
        ),
        "all_losses_and_gradients_finite": all(
            math.isfinite(float(value.detach())) for value in (
                point_initial, point_initial_grad, point_final, point_final_grad,
                q_initial_ctc, q_initial_grad, q_initial_kl,
                q_final_ctc, q_final_point_ctc, q_final_kl, q_final_grad_mu,
                q_mu.detach(), q_final_sigma,
            )
        ),
        "kl_nonnegative_with_declared_tolerance": (
            float(q_initial_kl.detach()) >= -config["declared_checks"]["require_nonnegative_kl_with_tolerance"]
            and float(q_final_kl.detach()) >= -config["declared_checks"]["require_nonnegative_kl_with_tolerance"]
        ),
    }
    return {
        "result": "PASS_EXPLORATORY_MECHANISM_ONLY" if all(checks.values()) else "FAIL_EXPLORATORY_CHECK",
        "checks": checks,
        "geometry": {
            "target_baseline_y": target_y,
            "evidence_center_y": evidence_y,
            "height_px": h,
            "prior_sigma_px": prior_sigma,
            "initial_q_sigma_px": initial_sigma,
        },
        "quadrature": {
            "rule": "Gauss-Hermite physicists nodes transformed to N(0,1)",
            "order": config["quadrature_order"],
            "weight_sum": float(weights.sum()),
            "nodes": nodes.tolist(),
            "weights": weights.tolist(),
        },
        "point_ctc_arm": {
            "initial_loss": float(point_initial.detach()),
            "initial_dloss_dy": float(point_initial_grad.detach()),
            "final_y": float(point_y.detach()),
            "final_loss": float(point_final.detach()),
            "final_dloss_dy": float(point_final_grad.detach()),
            "displacement_px": point_displacement,
        },
        "gaussian_band_arm": {
            "initial_expected_ctc": float(q_initial_ctc.detach()),
            "initial_d_expected_ctc_dmu": float(q_initial_grad.detach()),
            "initial_kl": float(q_initial_kl.detach()),
            "final_mu": float(q_mu.detach()),
            "final_sigma_px": float(q_final_sigma),
            "final_expected_ctc": float(q_final_ctc.detach()),
            "final_ctc_at_mean": float(q_final_point_ctc.detach()),
            "final_kl": float(q_final_kl.detach()),
            "final_d_expected_ctc_dmu": float(q_final_grad_mu.detach()),
            "displacement_px": q_displacement,
            "optimizer": "plain SGD on mu and log(sigma); no clipping or tuning",
        },
        "interpretation": "Exploratory only: the displacement margin was chosen after a transient inline feasibility run. One smooth synthetic evidence field and one initialization. No confirmatory claim, capture-radius guarantee, OCR result, or claim that the variational objective is optimal.",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_variational_band_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_variational_band_probe.json")
    args = parser.parse_args()
    started = time.monotonic()
    report = {
        "schema_version": "joint_variational_band_probe_v1",
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
        if config["synthetic_only"] is not True or config["device"] != "cpu" or config["threads"] != 1:
            raise ValueError("only single-thread synthetic CPU execution is allowed")
        if config["quadrature_order"] != 15 or config["steps"] != 40:
            raise ValueError("quadrature and optimization budget are fixed by the preregistration")
        if config["image_size_wh"] != [16, 64] or config["sequence_steps"] != 12:
            raise ValueError("probe dimensions are fixed by the preregistration")
        report["config"] = config
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        source_paths = (
            Path(__file__), ROOT / "src/babaorum/experimental/joint_geometry.py",
        )
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }
        report["probe"] = run_probe(config)
        report["result"] = report["probe"]["result"]
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
    return 0 if report["result"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
