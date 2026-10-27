#!/usr/bin/env python3
"""Generate the predeclared analytic-only v2 hybrid recall report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from babaorum.experimental.joint_hybrid_bounds import build_full_grid, line_recall_bounds


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/joint_hybrid_bounds_v2.json"
SOURCES = [
    "scripts/probe_joint_hybrid_bounds_v2.py",
    "src/babaorum/experimental/joint_hybrid_bounds.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_hybrid_bounds_v2.json")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    rows = build_full_grid(config)
    example = line_recall_bounds(0.5, 0.5, 0.5)
    tolerance = config["numeric_tolerance"]
    report = {
        "schema_version": "joint_hybrid_bounds_report_v2",
        "result": "PASS_ANALYTIC_CEILING_ONLY",
        "scope": "line-weighted closed-form recall model; no real data, images, training, or learned reconciliation",
        "config_sha256": sha256(CONFIG),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "example_p_q_r_half": example,
        "rows": rows,
        "diagnostics": {
            "row_count": len(rows),
            "frechet_bounds_valid": all(
                row["frechet_union_lower"] <= row["independent_oracle_union_recall"] + tolerance
                and row["independent_oracle_union_recall"] <= row["frechet_union_upper"] + tolerance
                for row in rows
            ),
            "oracle_union_dominates_best_single": all(
                row["independent_oracle_union_recall"] + tolerance >= row["best_single_recall"]
                for row in rows
            ),
            "frechet_best_path_gain_lower_zero": all(
                abs(row["frechet_gain_over_best_single_lower"]) <= tolerance
                for row in rows
            ),
            "max_independent_gain_over_best_single": max(
                row["independent_gain_over_best_single"] for row in rows
            ),
            "max_frechet_gain_over_best_single": max(
                row["frechet_gain_over_best_single_upper"] for row in rows
            ),
        },
        "limitations": [
            "p must be measured weighted by the number of GT lines per block.",
            "Independent errors are an optimistic scenario when difficult lines are missed by both passes.",
            "Oracle matching and no false positives overstate realizable reconciliation.",
            "Partial crops, overlap, merges, precision, geometry, order, and compute are not modeled.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['result']}: {len(rows)} analytic cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
