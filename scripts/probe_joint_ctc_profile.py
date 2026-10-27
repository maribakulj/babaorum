#!/usr/bin/env python3
"""CTC loss profile over baseline y for the drift-witness field (exploratory).

Re-builds the constructed field of probe_joint_page.drift_witness from the same
config and evaluates the CTC loss on a fixed y grid. No training, no OCR data.
Positions whose band leaves the page are recorded as rejected, not skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_geometry import FeatureFrame, sample_baseline_bands  # noqa: E402

# 0.00 .. 15.75 step 0.25. Chosen during exploration AFTER the initial v1
# witness FAIL; not a preregistered grid.
Y_GRID = [i / 4 for i in range(0, 64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_page_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_ctc_profile.json")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    cfg = config["drift_witness"]
    torch.set_num_threads(1)
    yy, xx = torch.meshgrid(torch.arange(16, dtype=torch.float64),
                            torch.arange(16, dtype=torch.float64), indexing="ij")
    y_evidence = cfg["evidence_center_y"]
    field = torch.stack((torch.zeros_like(xx),
                         5 - .15 * (xx - 4).square() - .3 * (yy - y_evidence).square(),
                         5 - .15 * (xx - 11).square() - .3 * (yy - y_evidence).square()))[None]
    frame = FeatureFrame((16, 16), (0., 0.), (1., 1.))
    height = torch.tensor([cfg["height_px"]], dtype=torch.float64)
    x = torch.tensor([2., 13.], dtype=torch.float64)
    rows = []
    for y in Y_GRID:
        baseline = torch.stack((x, torch.full_like(x, y)), dim=-1)[None]
        try:
            bands = sample_baseline_bands(field, baseline, height, frame, output_size=(3, 12),
                                          baseline_fraction=config["baseline_fraction"])
        except ValueError as exc:
            rows.append({"y": y, "ctc": None, "rejected": str(exc)})
            continue
        logits = bands.features.mean(dim=2).permute(2, 0, 1)
        ctc = torch.nn.functional.ctc_loss(
            logits.log_softmax(-1), torch.tensor([1, 2]), torch.tensor([12]),
            torch.tensor([2]), blank=0, reduction="mean", zero_infinity=False)
        rows.append({"y": y, "ctc": float(ctc), "rejected": None})
    valid = [r for r in rows if r["ctc"] is not None]
    local_minima = [valid[i]["y"] for i in range(1, len(valid) - 1)
                    if valid[i]["ctc"] < valid[i - 1]["ctc"] and valid[i]["ctc"] < valid[i + 1]["ctc"]]
    best = min(valid, key=lambda r: r["ctc"])
    report = {
        "schema_version": "joint_ctc_profile_v1", "author": "Claude w2:p6",
        "status": "EXPLORATORY_DIAGNOSTIC_SYNTHETIC_ONLY", "ocr_performance_claim": False,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "src/babaorum/experimental/joint_geometry.py")},
        "torch_version": torch.__version__,
        "band": {"height_px": cfg["height_px"], "baseline_fraction": config["baseline_fraction"],
                 "band_center_offset_px": -(config["baseline_fraction"] - 0.5) * cfg["height_px"]},
        "witness_start_y": cfg["initial_baseline_y"], "evidence_center_y": y_evidence,
        "minimum_scope": "sampled_grid_only", "exploration_after_initial_failure": True,
        "grid_step_y": 0.25, "global_minimum": best, "local_minima_y": local_minima, "profile": rows,
    }
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"minimum_scope": "sampled_grid_only", "exploration_after_initial_failure": True,
        "grid_step_y": 0.25, "global_minimum": best, "local_minima_y": local_minima}))


if __name__ == "__main__":
    main()
