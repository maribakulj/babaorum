#!/usr/bin/env python3
"""T4-ter v2 reanalysis: flat plateau, support and capture by loss level.

Read-only with respect to the T4-ter config and report. The CTC table is rebuilt
with the T4-ter Table class; no optimisation is rerun. Synthetic CPU, no OCR claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_joint_capture_tter import Table, radius  # noqa: E402

PLATEAU_TOL = 1e-3
BACKGROUND_TOL = 1e-6


def contiguous(mask, i0):
    """Largest contiguous True interval of mask containing index i0."""
    lo = i0
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    return lo, hi


def analyse_stride(config, report_stride, stride):
    h = config["height_px"]
    table = Table(config, stride)
    l, y = table.l, table.y
    l_min = float(l.min())
    l_bg = float(l.max())
    i_star = int(np.argmin(l))

    plateau_mask = l <= l_min + PLATEAU_TOL
    p_lo, p_hi = contiguous(plateau_mask, i_star)
    centre = 0.5 * (y[p_lo] + y[p_hi])

    support_mask = l < l_bg - BACKGROUND_TOL
    plateau_in_support = bool(support_mask[p_lo:p_hi + 1].all())
    if plateau_in_support:
        # extend the plateau outwards while the support condition holds
        s_lo = p_lo
        while s_lo > 0 and support_mask[s_lo - 1]:
            s_lo -= 1
        s_hi = p_hi
        while s_hi < len(y) - 1 and support_mask[s_hi + 1]:
            s_hi += 1
        support = {
            "lo_px": float(y[s_lo]), "hi_px": float(y[s_hi]),
            "half_width_lower_h": float((centre - y[s_lo]) / h),
            "half_width_upper_h": float((y[s_hi] - centre) / h),
        }
    else:
        support = None

    arms = {}
    for arm, arm_report in report_stride["arms"].items():
        runs = arm_report["runs"]
        offsets = [float(r["offset_h"]) for r in runs]
        captured = []
        for r in runs:
            if r["status"] != "ok":
                captured.append(False)
                continue
            l_at_mu = float(np.interp(r["mu_final"], y, l))
            captured.append(l_at_mu <= l_min + PLATEAU_TOL)
        arms[arm] = {
            "radius_v2": radius(offsets, captured),
            "captured_offsets_h": [o for o, c in zip(offsets, captured) if c],
        }

    return {
        "y_star_table": float(y[i_star]),
        "y_star_report": report_stride["y_star"],
        "L_min": l_min,
        "L_bg": l_bg,
        "plateau": {
            "lo_px": float(y[p_lo]), "hi_px": float(y[p_hi]),
            "centre_px": float(centre),
            "width_h": float((y[p_hi] - y[p_lo]) / h),
        },
        "y_star_in_plateau": bool(p_lo <= i_star <= p_hi),
        "support": support,
        "plateau_in_support": plateau_in_support,
        "arms": arms,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_capture_tter.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/joint_capture_tter.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_capture_tter_analysis_v2.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_bytes())
    if config["synthetic_only"] is not True or config["device"] != "cpu" or config["threads"] != 1:
        raise ValueError("only single-thread synthetic CPU execution is allowed")
    torch.set_num_threads(config["threads"])
    report = json.loads(args.report.read_bytes())

    probe = ROOT / "scripts/probe_joint_capture_tter.py"
    strides = {}
    for stride in config["feature_strides"]:
        key = f"stride_{stride}"
        strides[key] = analyse_stride(config, report["results"][key], stride)

    out = {
        "schema_version": "joint_capture_tter_analysis_v2",
        "status": "REANALYSIS_NO_NEW_OPTIMIZATION",
        "constants": {"plateau_tolerance_L": PLATEAU_TOL, "background_tolerance_L": BACKGROUND_TOL},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {
            str(args.config.relative_to(ROOT)): sha256(args.config),
            str(args.report.relative_to(ROOT)): sha256(args.report),
            str(probe.relative_to(ROOT)): sha256(probe),
        },
        "strides": strides,
    }
    args.output.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
