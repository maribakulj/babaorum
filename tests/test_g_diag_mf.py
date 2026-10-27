"""Fast checks of the G-diag rules (synthetic, no BLLA): gutter, classes, partition, both-rule
reporting, control against a reference, time limit, and the --confirmed-go guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import g_diag_mf as g  # noqa: E402

GT = [[100.0, 200.0], [300.0, 200.0]]   # horizontal GT line, length 200
H = 20.0                                # band height; tolerance 0.5 h = 10 px
BLOCK_X = (90.0, 310.0)                 # GT block x-range
GAP = 20.0                              # page gutter used in the classification tests


def test_coverage_and_near_on_a_straight_line():
    pts = g.gt_samples(GT)
    assert g.coverage(pts, [[100.0, 200.0], [300.0, 200.0]], 10.0) == pytest.approx(1.0)
    assert 0.45 < g.coverage(pts, [[100.0, 200.0], [200.0, 200.0]], 10.0) < 0.6
    far = [[100.0, 260.0], [300.0, 260.0]]
    assert g.coverage(pts, far, 10.0) == 0.0 and not g.near(pts, far, 10.0)


def test_gutter_is_measured_on_vertically_overlapping_neighbours_only():
    blocks = [(0.0, 0.0, 100.0, 50.0), (130.0, 0.0, 230.0, 50.0),   # gap 30, overlap in y
              (0.0, 200.0, 100.0, 250.0), (400.0, 300.0, 500.0, 350.0)]  # block 4: no vertical neighbour
    assert g.measure_gutter(blocks) == pytest.approx(30.0)
    assert g.measure_gutter([(0.0, 0.0, 10.0, 10.0)]) is None


def test_class_a_absent_when_no_prediction_is_near():
    assert g.classify_miss(GT, H, BLOCK_X, [[[100.0, 400.0], [300.0, 400.0]]], GAP) == "a_absent"
    assert g.classify_miss(GT, H, BLOCK_X, [], GAP) == "a_absent"


def test_class_b_uses_the_candidate_and_the_measured_gutter():
    fused = [[60.0, 200.0], [340.0, 200.0]]   # covers the GT line; extends 30 px beyond x 90 and 310
    assert g.classify_miss(GT, H, BLOCK_X, [fused], GAP) == "b_fusion"      # overhang 30 px > gutter 20
    assert g.classify_miss(GT, H, BLOCK_X, [fused], 40.0) == "d_other"      # overhang 30 px <= gutter 40


def test_class_b_not_assigned_without_a_measured_gutter():
    fused = [[60.0, 200.0], [340.0, 200.0]]
    assert g.classify_miss(GT, H, BLOCK_X, [fused], None) == "d_other"


def test_class_c_partial_when_no_prediction_reaches_0_75():
    half = [[100.0, 200.0], [190.0, 200.0]]
    assert g.classify_miss(GT, H, BLOCK_X, [half], GAP) == "c_partial"


def test_class_d_other_when_a_prediction_covers_but_is_not_a_fusion():
    inside = [[95.0, 200.0], [305.0, 200.0]]  # covers the GT line, stays within the block
    assert g.classify_miss(GT, H, BLOCK_X, [inside], GAP) == "d_other"


def test_partition_of_misses_by_class_sums_to_the_number_of_misses():
    cases = [
        ([[[100.0, 400.0], [300.0, 400.0]]], "a_absent"),
        ([[[60.0, 200.0], [340.0, 200.0]]], "b_fusion"),
        ([[[100.0, 200.0], [190.0, 200.0]]], "c_partial"),
        ([[[95.0, 200.0], [305.0, 200.0]]], "d_other"),
    ]
    counts = {"a_absent": 0, "b_fusion": 0, "c_partial": 0, "d_other": 0}
    for preds, expected in cases:
        cls = g.classify_miss(GT, H, BLOCK_X, preds, GAP)
        assert cls == expected
        counts[cls] += 1
    assert sum(counts.values()) == len(cases)


def test_band_predictions_are_mapped_by_origin_and_kept_by_core_centroid():
    band = [[[10.0, 5.0], [90.0, 5.0]], [[10.0, 60.0], [90.0, 60.0]]]
    kept = g.keep_band_predictions(band, y0=1000.0, core0=1000.0, core1=1050.0)
    assert kept == [[[10.0, 1005.0], [90.0, 1005.0]]]


def test_band_specs_round_to_page_rows():
    specs = g.band_specs(3698, [[0.0, 0.55], [0.45, 1.0]], [[0.0, 0.5], [0.5, 1.0]])
    assert specs[0] == (0, round(0.55 * 3698), 0, round(0.5 * 3698))
    assert specs[1] == (round(0.45 * 3698), 3698, round(0.5 * 3698), 3698)


def test_both_rules_are_reported_and_may_hold_together():
    both = g.decision(0.8, 0.3, 0.6)
    assert both["rule_scale_D1_le_half_D0"] is True and both["rule_columns_b_ge_half"] is True
    assert "both rules hold" in both["text"]
    neither = g.decision(0.8, 0.7, 0.1)
    assert neither["rule_scale_D1_le_half_D0"] is False and neither["rule_columns_b_ge_half"] is False
    assert "volet 3" in neither["text"]
    assert "horizontal band helps" in both["reading_D1"]


def test_control_flags_any_disagreement_with_the_reference():
    ref = [{"id": "a", "global_hit": True}, {"id": "b", "global_hit": False}]
    assert g.control_status({"a"}, ref)["status"] == "OK"
    res = g.control_status({"a", "b"}, ref)
    assert res["status"] == "MISMATCH" and res["mismatches"] == ["b"]


def test_time_limit_is_forty_five_real_minutes():
    assert g.TIME_LIMIT_S == 45 * 60
    assert g.guard_exceeded(45 * 60 + 1) is True
    assert g.guard_exceeded(45 * 60 - 1) is False


def test_run_refuses_without_confirmed_go(tmp_path):
    with pytest.raises(SystemExit):
        g.main(["--output", str(tmp_path / "x.json")])
