#!/usr/bin/env python3
"""Scale and tile diagnostic for the stage S smoke pages (S-L5b step 1). No training.

Reports, per page, the original size, the scale factor applied by train_stage_s,
the size after scaling, the median band height before/after scaling, the median
baseline arc length after scaling, and, for random 512-px tiles, how many lines
intersect the tile versus how many lie entirely inside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_stage_s import lines_in_tile, load_scaled_page  # noqa: E402
from babaorum.experimental.stage_s_data import load_mf_page  # noqa: E402

CONFIG = ROOT / "config/stage_s_smoke.json"
OUTPUT = ROOT / "reports/stage_s_scale_diagnostic.json"
TILE_SIZES = (512, 768)
TILES_PER_PAGE = 20


def arc_length(baseline):
    b = np.asarray(baseline, dtype=np.float64)
    return float(np.hypot(*(b[1:] - b[:-1]).T).sum())


def intersects_tile(line, x0, y0, size):
    b = np.asarray(line["baseline"], dtype=np.float64)
    ymin = b[:, 1].min() - line["normal_negative_px"]
    ymax = b[:, 1].max() + line["normal_positive_px"]
    return not (b[:, 0].max() < x0 or b[:, 0].min() > x0 + size
                or ymax < y0 or ymin > y0 + size)


def main():
    cfg = json.loads(CONFIG.read_text())
    root = Path(cfg["source"]["root"]).expanduser()
    target = cfg["scale"]["target_median_height_px"]
    page_ids = cfg["split"]["train_pages"] + cfg["split"]["eval_pages"]
    rng = np.random.default_rng(cfg["seed"])

    pages, tiles = [], []
    for pid in page_ids:
        raw = load_mf_page(root / "pages" / f"{pid}.json", root / "images" / f"{pid}.png")
        h0, w0 = raw["image"].shape
        scaled = load_scaled_page(root, pid, target)
        h1, w1 = scaled["image"].shape
        before = [ln["normal_negative_px"] + ln["normal_positive_px"] for ln in raw["lines"]]
        after = [ln["normal_negative_px"] + ln["normal_positive_px"] for ln in scaled["lines"]]
        arcs = [arc_length(ln["baseline"]) for ln in scaled["lines"]]
        pages.append({
            "page_id": pid, "original_wh": [w0, h0], "factor": scaled["factor"],
            "scaled_wh": [w1, h1], "n_lines": len(raw["lines"]),
            "median_band_px_before": float(np.median(before)),
            "median_band_px_after": float(np.median(after)),
            "median_baseline_arc_px_after": float(np.median(arcs)),
            "p90_baseline_arc_px_after": float(np.percentile(arcs, 90)),
        })
        for tile in TILE_SIZES:
            for _ in range(TILES_PER_PAGE):
                x0 = int(rng.integers(0, w1 - tile + 1))
                y0 = int(rng.integers(0, h1 - tile + 1))
                inter = sum(intersects_tile(ln, x0, y0, tile) for ln in scaled["lines"])
                inside = len(lines_in_tile(scaled["lines"], x0, y0, tile))
                tiles.append({"tile_px": tile, "page_id": pid, "x0": x0, "y0": y0,
                              "intersecting": inter, "entirely_inside": inside})

    summary = {}
    for tile in TILE_SIZES:
        sel = [t for t in tiles if t["tile_px"] == tile]
        inter = [t["intersecting"] for t in sel]
        inside = [t["entirely_inside"] for t in sel]
        summary[str(tile)] = {
            "mean_intersecting_per_tile": float(np.mean(inter)),
            "mean_entirely_inside_per_tile": float(np.mean(inside)),
            "tiles_with_zero_entirely_inside": int(sum(1 for x in inside if x == 0)),
            "n_tiles": len(sel),
        }
    report = {
        "schema_version": "stage_s_scale_diagnostic_v2",
        "status": "DIAGNOSTIC_ONLY_NO_TRAINING",
        "tile_sizes_px": list(TILE_SIZES),
        "tiles_per_page_per_size": TILES_PER_PAGE,
        "pages": pages,
        "tiles": tiles,
        "summary": summary,
    }
    OUTPUT.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
