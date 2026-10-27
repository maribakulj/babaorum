"""Round-trip checks for stage S targets on canonical Mille Feuilles page mf_0000 (read in place)."""

from __future__ import annotations

from pathlib import Path

import pytest

from babaorum.experimental.stage_s_targets import extract_polylines, rasterize_baselines

DATA = Path("/Users/marcel/heritage-synth/runs/accept-layout-v2/reproduced")
PAGE_JSON = DATA / "pages/mf_0000.json"
IMAGE = DATA / "images/mf_0000.png"

if not (PAGE_JSON.exists() and IMAGE.exists()):
    pytest.skip(f"canonical page not found under {DATA}", allow_module_level=True)

from babaorum.experimental.stage_s_data import load_mf_page  # noqa: E402
from babaorum.line_geometry import (  # noqa: E402
    GroundTruthLine, PredictedLine, evaluate_line_geometry,
)

PAGE = load_mf_page(PAGE_JSON, IMAGE)
H, W = PAGE["image"].shape
RASTER = rasterize_baselines(PAGE["lines"], (H, W), stride=4)
POLYLINES = extract_polylines(RASTER["baseline"], stride=4, threshold=0.5, min_cells=4)


def _gt_lines():
    out = []
    for line in PAGE["lines"]:
        height = line["normal_negative_px"] + line["normal_positive_px"]
        if height > 0:
            out.append(GroundTruthLine(id=line["id"], baseline=line["baseline"],
                                       height_px=height, baseline_source="ground_truth"))
    return out


def _pred_lines():
    return [PredictedLine(id=f"pred_{i:04d}", baseline=poly, baseline_source="predicted")
            for i, poly in enumerate(POLYLINES)]


def test_round_trip_meets_declared_criterion_on_mf_0000():
    result = evaluate_line_geometry(_gt_lines(), _pred_lines())
    mean_norm = (sum(m.normalized_symmetric_distance for m in result.matches) / len(result.matches)
                 if result.matches else float("nan"))
    print(f"\nrecall={result.recall} precision={result.precision} "
          f"matched={result.matched_count} n_gt={result.n_gt} n_pred={result.n_pred} "
          f"mean_normalized_distance={mean_norm:.4f}")
    assert result.recall is not None and result.recall >= 0.95
    assert result.precision is not None and result.precision >= 0.95


def test_collisions_are_counted_and_reported_without_threshold():
    assert isinstance(RASTER["collisions"], int)
    assert RASTER["collisions"] >= 0
    print(f"\ncollisions={RASTER['collisions']}")


def test_extraction_is_deterministic():
    again = extract_polylines(RASTER["baseline"], stride=4, threshold=0.5, min_cells=4)
    assert again == POLYLINES
