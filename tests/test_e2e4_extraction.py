"""Fast checks of the E2E-4 pure rules: gap() (amendment 1, one function for measurement and fusion),
fusion conditions, deterministic chains, G rule, H-ESP on g/e, readings R5 and R6, guards.
No torch, no pages."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e2e4_extraction as e  # noqa: E402

H = 32.0
G_BIG = 100.0


def seg(x0, x1, y=100.0):
    return [[x0, y], [x1, y]]


# ---- gap(A, B) = x(start B) - x(end A): the single function of amendment 1

def test_gap_is_start_of_b_minus_end_of_a():
    assert e.gap(seg(0, 100), seg(130, 200)) == pytest.approx(30.0)


def test_gap_is_negative_for_overlaps_and_zero_for_touching():
    assert e.gap(seg(0, 100), seg(95, 200)) == pytest.approx(-5.0)
    assert e.gap(seg(0, 100), seg(100, 200)) == pytest.approx(0.0)


def test_gap_uses_reading_order_whatever_the_point_order():
    assert e.gap(seg(100, 0), seg(200, 130)) == pytest.approx(30.0)


# ---- fusion conditions, all through the same gap()

def test_aligned_pair_with_hole_below_G_is_fused():
    assert e.fuse_pair(seg(0, 100), seg(130, 200), G=40, h_target=H) == pytest.approx(30.0)


def test_hole_above_G_is_not_fused():
    assert e.fuse_pair(seg(0, 100), seg(130, 200), G=20, h_target=H) is None


def test_overlap_is_never_fused_even_with_large_G():
    assert e.fuse_pair(seg(0, 100), seg(95, 200), G=G_BIG, h_target=H) is None


def test_vertical_offset_above_half_the_target_height_is_not_fused():
    b = [[130.0, 100.0 + 0.5 * H + 1], [200.0, 100.0 + 0.5 * H + 1]]
    assert e.fuse_pair(seg(0, 100), b, G=G_BIG, h_target=H) is None


def test_angle_difference_above_ten_degrees_is_not_fused():
    import math
    b = [[130.0, 100.0], [200.0, 100.0 + 100.0 * math.tan(math.radians(12))]]
    assert e.fuse_pair(seg(0, 100), b, G=G_BIG, h_target=H) is None


def test_chain_of_three_is_fused_in_reading_order():
    polys = [seg(0, 100), seg(130, 200), seg(230, 300)]
    fused, members = e.fuse_chain(polys, G=40, h_target=H)
    assert len(fused) == 1 and members == [[0, 1, 2]]
    assert fused[0][0] == [0.0, 100.0] and fused[0][-1] == [300.0, 100.0]


def test_chain_is_deterministic():
    polys = [seg(230, 300), seg(0, 100), seg(130, 200)]
    assert e.fuse_chain(polys, G=40, h_target=H) == e.fuse_chain(polys, G=40, h_target=H)


def test_G_zero_leaves_the_extraction_unchanged():
    polys = [seg(0, 100), seg(130, 200)]
    fused, members = e.fuse_chain(polys, G=0.0, h_target=H)
    assert members == [[0], [1]] and len(fused) == 2


# ---- measurement of holes on a line (§2) and word gaps, with the same gap()

def test_line_gaps_keep_only_positive_holes():
    assert e.line_gaps([seg(0, 30), seg(40, 60)]) == [pytest.approx(10.0)]
    assert e.line_gaps([seg(0, 50), seg(40, 60)]) == []


def test_word_gaps_use_gap_between_consecutive_boxes():
    gaps = e.word_gaps({"L1": [(0.0, 10.0), (20.0, 30.0), (25.0, 40.0)]})
    assert gaps == [pytest.approx(10.0)]   # 10 -> 20 is a hole; 30 -> 25 overlaps and is not counted


# ---- G rule (§3): 90th percentile, capped at h_target, reported in px and in e

def test_G_is_the_ninetieth_percentile_of_the_gaps():
    res = e.compute_G([float(v) for v in range(1, 11)], h_target=H)
    assert res["p90"] == pytest.approx(9.1)
    assert res["G"] == pytest.approx(9.1)


def test_G_is_capped_at_the_target_height():
    assert e.compute_G([100.0, 200.0], h_target=H)["G"] == pytest.approx(H)


def test_G_without_gaps_is_zero():
    assert e.compute_G([], h_target=H)["G"] == 0.0


# ---- H-ESP on the ratio g / e (amendment 1)

def test_hesp_supported_when_median_ratio_and_half_are_in_interval():
    res = e.hesp_reading([10.0, 10.0, 10.0, 25.0], e_px=10.0)
    assert res["supported"] is True and res["n"] == 4


def test_hesp_not_supported_when_gaps_are_much_larger_than_e():
    res = e.hesp_reading([40.0, 40.0, 40.0, 40.0], e_px=10.0)
    assert res["supported"] is False and "non expliquées" in res["text"]


def test_hesp_not_supported_when_gaps_are_much_smaller_than_e():
    res = e.hesp_reading([2.0, 2.0, 2.0, 25.0], e_px=10.0)
    assert res["supported"] is False


# ---- readings R5 and R6

def test_R5_sufficient_needs_precision_half_and_recall_within_two_points():
    assert e.r5_reading(0.6, 0.56)["sufficient"] is True
    assert e.r5_reading(0.6, 0.55)["sufficient"] is False    # recall below 0.5768 - 0.02
    assert e.r5_reading(0.4, 0.9)["sufficient"] is False     # precision below 0.5


def test_R6_drop_above_two_points_degrades_S2():
    assert e.r6_reading(0.7960 - 0.015, 0.9677, 0.7960, 0.9677)["degraded"] is False
    assert e.r6_reading(0.7960 - 0.025, 0.9677, 0.7960, 0.9677)["degraded"] is True
    assert e.r6_reading(0.7960, 0.9677 - 0.025, 0.7960, 0.9677)["degraded"] is True


def test_R6_exactly_two_points_is_not_a_degradation():
    assert e.r6_reading(0.7960 - 0.02, 0.9677, 0.7960, 0.9677)["degraded"] is False


# ---- guards

def test_run_refuses_without_confirmed_go():
    with pytest.raises(SystemExit):
        e.main(["run"])


def test_run_refuses_an_existing_report(tmp_path):
    existing = tmp_path / "report.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit):
        e.main(["run", "--confirmed-go", "--clone", str(tmp_path), "--tmp", str(tmp_path),
                "--bundle-zip", str(tmp_path / "b.zip"), "--calibration", str(tmp_path),
                "--report", str(existing)])


# ---- amendment 2 (§8): nothing under --tmp, cap 150 min, guard recognises e2e4 runs

def test_G_fixed_is_written_next_to_the_report_when_tmp_is_read_only(tmp_path):
    import os
    tmp = tmp_path / "ro_tmp"
    tmp.mkdir()
    report = tmp_path / "out" / "report.json"
    if os.geteuid() == 0:
        pytest.skip("root ignores the read-only bit")
    os.chmod(tmp, 0o555)
    try:
        out = e.write_G_fixed(report, {"G": {"G": 9.1}}, tmp)
    finally:
        os.chmod(tmp, 0o755)
    assert out == tmp_path / "out" / "e2e4_G_fixed.json" and out.exists()
    assert list(tmp.iterdir()) == []


def test_G_fixed_refuses_a_path_under_tmp(tmp_path):
    with pytest.raises(SystemExit):
        e.write_G_fixed(tmp_path / "report.json", {}, tmp_path)


def test_cap_is_150_minutes_in_the_config():
    import json
    cfg = json.loads(e.CONFIG.read_text())
    assert cfg["budget"]["cap_minutes"] == 150


def test_live_guard_sees_an_e2e4_run_and_skips_the_current_chain():
    ps = ("100 1 /bin/zsh\n"
          "200 100 python3 scripts/e2e4_extraction.py run --confirmed-go\n"
          "300 200 python3 scripts/e2e4_extraction.py check\n"
          "400 1 python3 scripts/e2e4_extraction.py run --confirmed-go\n")
    alive = e.live_e2e4_processes(ps, own_pid=300)
    # 200 and 100 are ancestors of the current process (excluded); 400 is an independent live run
    assert [line.split()[0] for line in alive] == ["400"]


def test_smoke_refuses_without_both_id_lists():
    with pytest.raises(SystemExit):
        e.main(["run", "--confirmed-go", "--setting-ids", "mf_0000", "--clone", "x", "--tmp", "x",
                "--bundle-zip", "x", "--calibration", "x", "--report", "x"])
