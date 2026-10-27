#!/usr/bin/env python3
"""T4-bis: capture radius of reading-based baseline refinement (synthetic CPU).

Point CTC L(y) is tabulated per feature stride with the real sampler; Gaussian
expectations and their mu/sigma derivatives are exact trapezoid integrals over
that tabulation. No ground truth enters the objectives. No OCR claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from babaorum.experimental.joint_geometry import FeatureFrame  # noqa: E402
from probe_joint_capture_profile import ctc_at_y  # noqa: E402


def field_for_stride(config, stride):
    dtype = torch.float64
    width, height = config["image_size_wh"]
    ev = config["evidence_center_y"]
    yy, xx = torch.meshgrid(torch.arange(height, dtype=dtype), torch.arange(width, dtype=dtype), indexing="ij")
    field = torch.stack((
        torch.zeros_like(xx),
        5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - ev).square(),
        5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - ev).square(),
    ))[None]
    if stride > 1:
        field = F.avg_pool2d(field, kernel_size=stride, stride=stride)
    origin = (stride - 1) / 2.0
    frame = FeatureFrame((width, height), (origin, origin), (float(stride), float(stride)))
    return field, frame


class Table:
    def __init__(self, config, stride):
        field, frame = field_for_stride(config, stride)
        x = torch.tensor([2.0, 13.0], dtype=torch.float64)
        h = torch.tensor([config["height_px"]], dtype=torch.float64)
        lo, hi = config["tabulation_y_range"]
        self.y = np.arange(lo, hi + 1e-12, config["tabulation_step_px"])
        with torch.no_grad():
            self.l = np.array([float(ctc_at_y(torch.tensor(v, dtype=torch.float64), field, x, h, frame, config))
                               for v in self.y])
        self.dl = np.gradient(self.l, self.y)
        self.y_star = float(self.y[np.argmin(self.l)])

    def point_grad(self, mu):
        if not self.y[0] <= mu <= self.y[-1]:
            raise ValueError("point iterate left the tabulated range")
        return float(np.interp(mu, self.y, self.dl))

    def gaussian(self, mu, sigma):
        """E_q[L], dE/dmu, dE/dsigma for q = N(mu, sigma^2)."""
        lo, hi = mu - 6 * sigma, mu + 6 * sigma
        if lo < self.y[0] or hi > self.y[-1]:
            raise ValueError("Gaussian support left the tabulated range")
        m = (self.y >= lo) & (self.y <= hi)
        y, l = self.y[m], self.l[m]
        z = (y - mu) / sigma
        phi = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))
        norm = np.trapezoid(phi, y)
        e = np.trapezoid(l * phi, y) / norm
        d_mu = np.trapezoid(l * phi * z / sigma, y) / norm
        d_sigma = np.trapezoid(l * phi * (z * z - 1) / sigma, y) / norm
        return float(e), float(d_mu), float(d_sigma)


def optimise(table, arm, start, config):
    h = config["height_px"]
    lr, steps = config["optimizer"]["learning_rate"], config["optimizer"]["steps"]
    mu = start
    log_sigma = math.log(1.0 * h)
    tau = 1.0 * h
    sigma0, sigma_end = 1.0 * h, 0.1 * h
    try:
        for k in range(steps):
            if arm == "point":
                mu -= lr * table.point_grad(mu)
            elif arm == "annealed":
                sigma = sigma0 * (sigma_end / sigma0) ** (k / (steps - 1))
                mu -= lr * table.gaussian(mu, sigma)[1]
            else:
                sigma = math.exp(log_sigma)
                _, d_mu, d_sigma = table.gaussian(mu, sigma)
                d_mu += (mu - start) / tau ** 2
                d_sigma += sigma / tau ** 2 - 1.0 / sigma
                mu -= lr * d_mu
                log_sigma -= lr * d_sigma * sigma
                log_sigma = min(max(log_sigma, math.log(0.02 * h)), math.log(1.5 * h))
    except ValueError as exc:
        return {"status": "left_support", "detail": str(exc), "mu_final": mu}
    out = {"status": "ok", "mu_final": mu}
    if arm == "learned_with_detector_prior":
        out["sigma_final_h"] = math.exp(log_sigma) / h
    return out


def radius(offsets, captured):
    i0 = offsets.index(0.0)
    if not captured[i0]:
        return {"lower_h": None, "upper_h": None}
    lo = i0
    while lo > 0 and captured[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(offsets) - 1 and captured[hi + 1]:
        hi += 1
    return {"lower_h": -offsets[lo], "upper_h": offsets[hi],
            "lower_censored": lo == 0, "upper_censored": hi == len(offsets) - 1}


def run(config):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    h = config["height_px"]
    sc = config["starts"]
    n = round((sc["offset_max_heights"] - sc["offset_min_heights"]) / sc["offset_step_heights"])
    offsets = [round(sc["offset_min_heights"] + i * sc["offset_step_heights"], 10) for i in range(n + 1)]
    results = {}
    for stride in config["feature_strides"]:
        table = Table(config, stride)
        per_arm = {}
        for arm in config["arms"]:
            runs = [optimise(table, arm, table.y_star + off * h, config) for off in offsets]
            captured = [r["status"] == "ok" and abs(r["mu_final"] - table.y_star) <= 0.25 * h for r in runs]
            per_arm[arm] = {"radius": radius(offsets, captured),
                            "captured_offsets_h": [o for o, c in zip(offsets, captured) if c],
                            "runs": [{"offset_h": o, **r} for o, r in zip(offsets, runs)]}
        local_minima = [float(table.y[i]) for i in range(1, len(table.y) - 1)
                        if table.l[i] < table.l[i - 1] and table.l[i] < table.l[i + 1]]
        results[f"stride_{stride}"] = {"y_star": table.y_star, "point_ctc_local_minima_y": local_minima,
                                       "arms": per_arm}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_capture_tbis.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_capture_tbis.json")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if config["synthetic_only"] is not True or config["device"] != "cpu" or config["threads"] != 1:
        raise ValueError("only single-thread synthetic CPU execution is allowed")
    started = time.monotonic()
    report = {
        "schema_version": "joint_capture_tbis_v1", "status": "EXPLORATORY_MECHANISM_SYNTHETIC_ONLY",
        "ocr_performance_claim": False, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), ROOT / "scripts/probe_joint_capture_profile.py",
            ROOT / "src/babaorum/experimental/joint_geometry.py")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform()},
    }
    report["results"] = run(config)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
