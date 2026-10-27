#!/usr/bin/env python3
"""Write an analytic-only report for cascade/global-pass recall ceilings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from babaorum.experimental.hybrid_bounds import build_grid, recall_bounds


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/hybrid_miss_ceiling.json"
SOURCES = [
    "scripts/probe_hybrid_miss_ceiling.py",
    "src/babaorum/experimental/hybrid_bounds.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hybrid_miss_ceiling.json")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    rows = build_grid(config)
    report = {
        "schema_version": "hybrid_miss_ceiling_report_v1",
        "result": "PASS_ANALYTIC_CEILING_ONLY",
        "scope": "closed-form miss-probability model; no dataset, image, training, or learned reconciliation",
        "config_sha256": sha256(CONFIG),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "rows": rows,
        "diagnostics": {
            "row_count": len(rows),
            "max_independent_gain": max(row["independent_gain_over_cascade"] for row in rows),
            "frechet_bounds_valid": all(
                row["frechet_union_lower"] <= row["independent_oracle_union_recall"]
                <= row["frechet_union_upper"]
                for row in rows
            ),
            "oracle_union_dominates_each_path": all(
                row["independent_oracle_union_recall"] + 1e-12 >= max(
                    row["cascade_recall"], row["global_recall"]
                )
                for row in rows
            ),
            "edge_cases": {
                "perfect_all": recall_bounds(0.0, 0.0, 0.0),
                "global_useless": recall_bounds(0.2, 0.3, 1.0),
                "cascade_useless_global_perfect": recall_bounds(1.0, 1.0, 0.0),
            },
        },
        "limitations": [
            "No empirical rates or page-level distribution are estimated.",
            "Independent errors are only a scenario; shared features can correlate failures.",
            "Oracle reconciliation and no false positives overstate a realizable detector.",
            "No precision, geometry, text, order, or compute costs are measured.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['result']}: {len(rows)} analytic cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
