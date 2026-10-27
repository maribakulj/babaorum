#!/usr/bin/env python3
"""Corrective dense-quadrature re-analysis of T4b/T4c (synthetic, CPU).

The point CTC L(y) is tabulated once on a fine grid with the same field and
sampler as T4b/T4c. Gaussian expectations E_sigma(mu) and their exact
mu-derivatives are trapezoid integrals over that tabulation, which avoids the
15-node Gauss-Hermite error on the piecewise-smooth L(y). Point arms use the
real autograd CTC. No OCR, no data, no performance claim.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from babaorum.experimental.joint_geometry import FeatureFrame  # noqa: E402
from probe_joint_capture_profile import ctc_at_y  # noqa: E402


def build_field(config):
    dtype = torch.float64
    width, page_height = config["image_size_wh"]
    ev = config["evidence_center_y"]
    yy, xx = torch.meshgrid(torch.arange(page_height, dtype=dtype),
                            torch.arange(width, dtype=dtype), indexing="ij")
    field = torch.stack((
        torch.zeros_like(xx),
        5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - ev).square(),
        5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - ev).square(),
    ))[None]
    frame = FeatureFrame((width, page_height), (0.0, 0.0), (1.0, 1.0))
    x = torch.tensor([2.0, 13.0], dtype=dtype)
    height = torch.tensor([config["height_px"]], dtype=dtype)
    return field, frame, x, height


def point_loss(y, ctx, config):
    field, frame, x, height = ctx
    return ctc_at_y(y, field, x, height, frame, config)


class Tabulated:
    def __init__(self, ctx, config):
        tab = config["tabulation"]
        self.y = np.arange(tab["y_min"], tab["y_max"] + 1e-12, tab["step_px"])
        with torch.no_grad():
            self.l = np.array([float(point_loss(torch.tensor(v, dtype=torch.float64), ctx, config))
                               for v in self.y])

    def expected(self, mu, sigma):
        """Return E[L(mu+sigma z)] and its exact mu-derivative on the tabulation."""
        lo, hi = mu - 6 * sigma, mu + 6 * sigma
        if lo < self.y[0] or hi > self.y[-1]:
            raise ValueError("Gaussian support leaves the tabulated range")
        mask = (self.y >= lo) & (self.y <= hi)
        y, l = self.y[mask], self.l[mask]
        z = (y - mu) / sigma
        phi = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))
        norm = np.trapezoid(phi, y)
        value = np.trapezoid(l * phi, y) / norm
        derivative = np.trapezoid(l * phi * (y - mu) / sigma ** 2, y) / norm
        return float(value), float(derivative)


def contiguous_interval(offsets, toward):
    """Maximal run of True containing offset 0, with censoring flags."""
    i0 = offsets.index(0.0)
    if not toward[i0]:
        return {"contains_target": False, "left_h": None, "right_h": None,
                "left_censored": False, "right_censored": False}
    left = i0
    while left > 0 and toward[left - 1]:
        left -= 1
    right = i0
    while right < len(offsets) - 1 and toward[right + 1]:
        right += 1
    return {"contains_target": True, "left_h": offsets[left], "right_h": offsets[right],
            "left_censored": left == 0, "right_censored": right == len(offsets) - 1}


def run(config):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    ctx = build_field(config)
    h, tgt = config["height_px"], config["target_baseline_y"]
    tau = config["prior_sigma_fraction_of_height"] * h
    tab = Tabulated(ctx, config)
    point_min_y = float(tab.y[np.argmin(tab.l)])
    references = {"evidence_center_y": config["evidence_center_y"],
                  "point_ctc_global_minimum_y": point_min_y}

    pc = config["profile"]
    count = round((pc["offset_max_heights"] - pc["offset_min_heights"]) / pc["offset_step_heights"])
    offsets = [round(pc["offset_min_heights"] + i * pc["offset_step_heights"], 10) for i in range(count + 1)]
    eps = pc["directional_gradient_epsilon"]
    profile = {}
    for frac in pc["sigma_fractions_of_height"]:
        sigma = frac * h
        rows = []
        for off in offsets:
            mu = tgt + off * h
            e, de = tab.expected(mu, sigma)
            kl_grad = (mu - tgt) / tau ** 2
            rows.append({"offset_h": off, "mu": mu, "expected_ctc": e, "d_expected_ctc": de,
                         "d_ctc_plus_kl": de + kl_grad})
        result = {"rows": rows, "intervals": {}}
        for name, ref in references.items():
            for key in ("d_expected_ctc", "d_ctc_plus_kl"):
                toward = [((ref - r["mu"]) * (-r[key]) > eps * abs(ref - r["mu"])) if abs(ref - r["mu"]) > 1e-12
                          else abs(r[key]) <= eps for r in rows]
                result["intervals"][f"{key}__toward_{name}"] = contiguous_interval(offsets, toward)
        profile[f"sigma_{frac}h"] = result

    # Stationary points of E+KL and E on a fine mu grid (sigma fixed as T4c).
    tc = config["trajectories"]
    sigma_q = tc["fixed_q_sigma_fraction_of_height"] * h
    grid = np.arange(tgt - 2.0 * h, tgt + 2.0 * h + 1e-9, 0.01)
    d_e = np.array([tab.expected(m, sigma_q)[1] for m in grid])
    d_ekl = d_e + (grid - tgt) / tau ** 2

    def zeros(d):
        out = []
        for i in range(len(grid) - 1):
            if d[i] == 0 or d[i] * d[i + 1] < 0:
                kind = "minimum" if d[i] < d[i + 1] else "maximum"
                out.append({"mu": float(grid[i] - d[i] * (grid[i + 1] - grid[i]) / (d[i + 1] - d[i])), "kind": kind})
        return out

    stationary = {"expected_ctc_plus_kl": zeros(d_ekl), "expected_ctc": zeros(d_e)}

    def step_arm(arm, value):
        if arm in ("point_ctc", "point_ctc_plus_kl"):
            y = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            loss = point_loss(y, ctx, config)
            loss.backward()
            g = float(y.grad)
            if arm == "point_ctc_plus_kl":
                g += (value - tgt) / tau ** 2
            return g
        if arm == "kl_only":
            return (value - tgt) / tau ** 2
        _, de = tab.expected(value, sigma_q)
        return de + ((value - tgt) / tau ** 2 if arm == "expected_ctc_plus_kl" else 0.0)

    trajectories = {}
    lr = tc["learning_rate"]
    for off in tc["start_offsets_heights"]:
        start = tgt + off * h
        arms = {}
        for arm in tc["arms"]:
            value, at_short = start, None
            for step in range(1, tc["long_steps"] + 1):
                value -= lr * step_arm(arm, value)
                if step == tc["short_steps"]:
                    at_short = value
            arms[arm] = {"after_short_steps": at_short, "after_long_steps": value,
                         "final_gradient": step_arm(arm, value)}
        trajectories[f"start_{off:+g}h"] = arms
    return {"references": references, "profile": profile,
            "stationary_points_sigma_fixed": stationary, "trajectories": trajectories,
            "tabulation_points": int(len(tab.y))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_capture_dense_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_capture_dense_v2.json")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if config["synthetic_only"] is not True or config["device"] != "cpu" or config["threads"] != 1:
        raise ValueError("only single-thread synthetic CPU execution is allowed")
    started = time.monotonic()
    report = {
        "schema_version": "joint_capture_dense_v2", "status": "CORRECTIVE_EXPLORATORY_SYNTHETIC_ONLY",
        "ocr_performance_claim": False, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), ROOT / "scripts/probe_joint_capture_profile.py",
            ROOT / "src/babaorum/experimental/joint_geometry.py")},
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
    }
    report["results"] = run(config)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
