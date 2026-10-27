"""Fast T5 checks, no BLLA: phi and gain formulas on a known 2x2 table, crop round trip, bootstrap."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t5_mf  # noqa: E402


def _table_indicators(both, only_g, only_l, neither):
    miss_g = [1] * both + [1] * only_g + [0] * only_l + [0] * neither
    miss_l = [1] * both + [0] * only_g + [1] * only_l + [0] * neither
    return miss_g, miss_l


def test_phi_and_gain_on_known_table():
    # 400 lines: P(both miss)=0.375, P(only G)=0.125, P(only L)=0.125, P(neither)=0.375
    # r = P(miss_G) = 0.5, m = P(miss_L) = 0.5, phi = (0.375 - 0.25) / 0.25 = 0.5
    miss_g, miss_l = _table_indicators(150, 50, 50, 150)
    s = t5_mf.stats_from_indicators(miss_g, miss_l)
    assert s["r_miss_global"] == pytest.approx(0.5)
    assert s["m_miss_local"] == pytest.approx(0.5)
    assert s["p_both_miss"] == pytest.approx(0.375)
    assert s["phi"] == pytest.approx(0.5)
    assert s["G_observed"] == pytest.approx(0.125)
    assert s["G_reconstructed"] == pytest.approx(0.125)


def test_gain_is_zero_when_one_pass_contains_the_other_misses():
    # local misses are a subset of global misses: union adds nothing over the best single pass
    miss_g, miss_l = _table_indicators(50, 150, 0, 200)
    s = t5_mf.stats_from_indicators(miss_g, miss_l)
    assert s["G_observed"] == pytest.approx(0.0, abs=1e-12)


def test_crop_transform_round_trip_is_identity():
    polygon = [[864.0, 94.0], [1816.0, 94.0], [1816.0, 202.0], [864.0, 202.0]]
    tr = t5_mf.crop_transform(polygon, 2680, 3698)
    assert tr["scale"] == pytest.approx(1800.0 / 3698.0)
    pts = [[900.0, 120.0], [1500.0, 150.5], [1800.0, 200.0]]
    canvas = t5_mf.page_to_crop_canvas(pts, tr)
    back = t5_mf.crop_to_page(canvas, tr)
    np.testing.assert_allclose(np.asarray(back), np.asarray(pts), atol=1e-9)


def test_crop_is_clipped_to_the_page_and_keeps_margin():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
    tr = t5_mf.crop_transform(polygon, 2680, 3698)
    assert tr["origin"] == (0.0, 0.0)  # margin would go negative: clipped
    assert tr["size"] == (124, 74)     # 100 + 24 and 50 + 24


def test_bootstrap_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(1)
    pages = []
    for _ in range(5):
        g = (rng.random(20) < 0.4).astype(int)
        l = (rng.random(20) < 0.4).astype(int)
        pages.append((g, l))
    a = t5_mf.bootstrap_by_page(pages, resamples=50, seed=17)
    b = t5_mf.bootstrap_by_page(pages, resamples=50, seed=17)
    assert a == b
    lo, hi = a["G_observed_ci95"]
    assert lo <= hi


def test_full_run_is_refused_while_run_gate_is_closed(tmp_path):
    cfg = {"run_gate": {"full_run": "CLOSED"}}
    with pytest.raises(RuntimeError, match="full run refused"):
        t5_mf.check_run_gate(cfg, None, tmp_path / t5_mf.FULL_OUTPUT)
    with pytest.raises(RuntimeError, match="full run refused"):
        t5_mf.check_run_gate(cfg, 30, tmp_path / t5_mf.FULL_OUTPUT)


def test_one_page_chronometer_writes_only_its_own_report(tmp_path):
    cfg = {"run_gate": {"full_run": "CLOSED"}}
    t5_mf.check_run_gate(cfg, 1, tmp_path / t5_mf.CHRONO_OUTPUT)  # allowed
    with pytest.raises(RuntimeError, match="must write"):
        t5_mf.check_run_gate(cfg, 1, tmp_path / t5_mf.FULL_OUTPUT)


def test_open_run_gate_allows_full_run_only_to_full_output(tmp_path):
    cfg = {"run_gate": {"full_run": "OPEN"}}
    t5_mf.check_run_gate(cfg, None, tmp_path / t5_mf.FULL_OUTPUT)
    with pytest.raises(RuntimeError, match="must write"):
        t5_mf.check_run_gate(cfg, None, tmp_path / t5_mf.CHRONO_OUTPUT)


# ---- E2E-2b: caps refused in full run, INCOMPLETE status, local GT by crop box,
# ---- resume guarded by sha256, separate timings, five-hour guard

import json as _json  # noqa: E402

V2 = t5_mf.CONFIG


def _cfg():
    return _json.loads(V2.read_text())


def test_caps_are_refused_outside_the_one_page_chronometer(tmp_path):
    with pytest.raises(RuntimeError, match="refused outside"):
        t5_mf.validate_caps(None, 3, None)
    with pytest.raises(RuntimeError, match="refused outside"):
        t5_mf.validate_caps(30, None, 300.0)
    t5_mf.validate_caps(1, 3, 300.0)  # chronometer: allowed
    cfg = _cfg()
    with pytest.raises(RuntimeError, match="refused outside"):
        t5_mf.run(cfg, tmp_path / t5_mf.FULL_OUTPUT, pages_limit=None, skip_blla=True, max_crops=3)


def test_interruption_or_guard_gives_incomplete_and_no_statistic_is_read():
    assert t5_mf.finalize_status([{"page_id": "p"}], False, chrono=False) == "INCOMPLETE"
    assert t5_mf.finalize_status([], True, chrono=False) == "INCOMPLETE"
    assert t5_mf.finalize_status([], False, chrono=False) == "COMPLETE"
    assert t5_mf.finalize_status([], False, chrono=True) == "COST_PROBE_ONLY_NO_RESULT"


def test_guard_hit_on_full_run_writes_incomplete_without_statistics(tmp_path):
    cfg = _cfg()
    out = tmp_path / t5_mf.FULL_OUTPUT
    report = t5_mf.run(cfg, out, pages_limit=None, skip_blla=True, guard_seconds=-1.0)
    assert report["status"] == "INCOMPLETE" and report["guard_hit"] is True
    assert report["pages_done"] == []
    assert "primary_tolerance_0.5h" not in report and "bootstrap_by_page" not in report
    assert "NOT READ" in report["statistics"]


def test_local_ground_truth_is_every_line_with_a_point_inside_the_crop_box():
    box = (100.0, 100.0, 300.0, 200.0)
    inside = t5_mf.GroundTruthLine(id="a", baseline=[[50.0, 150.0], [120.0, 150.0]],
                                   height_px=10.0, baseline_source="ground_truth")   # one point inside
    outside = t5_mf.GroundTruthLine(id="b", baseline=[[10.0, 10.0], [20.0, 10.0]],
                                    height_px=10.0, baseline_source="ground_truth")
    edge = t5_mf.GroundTruthLine(id="c", baseline=[[300.0, 150.0], [400.0, 150.0]],
                                 height_px=10.0, baseline_source="ground_truth")   # on the boundary
    got = [g.id for g in t5_mf.gt_lines_in_box([inside, outside, edge], box)]
    assert got == ["a", "c"]


def test_resume_only_with_identical_sha256_of_config_script_and_model():
    meta = {"config_sha256": "c", "script_sha256": "s", "model_sha256": "m"}
    assert t5_mf.resume_allowed(meta, dict(meta)) is True
    assert t5_mf.resume_allowed(meta, dict(meta, script_sha256="other")) is False
    assert t5_mf.resume_allowed(meta, dict(meta, config_sha256="other")) is False
    assert t5_mf.resume_allowed(meta, dict(meta, model_sha256="other")) is False


def test_timings_are_split_per_page_and_recorded(tmp_path):
    cfg = _cfg()
    cfg["sampling"]["pages"]["n"] = 1
    report = t5_mf.run(cfg, tmp_path / t5_mf.CHRONO_OUTPUT, pages_limit=1, skip_blla=True)
    p = report["per_page"][0]
    for key in ("seconds_global", "seconds_crops", "seconds_page", "n_crops"):
        assert key in p
    assert p["n_crops"] == 0 and p["seconds_global"] == 0.0 and p["seconds_crops"] == 0.0
    assert report["status"] == "COST_PROBE_ONLY_NO_RESULT"
    assert "primary_tolerance_0.5h" not in report


# ---- E2E-2b corrections (v3 report): interpretation rule, exclusion, resume guard, module hashes

def test_interpretation_rule_fixed_before_run():
    assert t5_mf.FULL_OUTPUT == "g_intra_mf_v2.json"
    assert t5_mf.FUTILITY_UPPER == 0.02
    oracle = t5_mf.interpret_G(0.01, 0.05)
    assert oracle["oracle_reframing_effect"] is True and oracle["not_distinguished_from_0"] is False
    zero = t5_mf.interpret_G(-0.02, 0.03)
    assert zero["not_distinguished_from_0"] is True and zero["futile_upper_below_0_02"] is False
    futile = t5_mf.interpret_G(-0.05, 0.01)
    assert futile["futile_upper_below_0_02"] is True and futile["oracle_reframing_effect"] is False


def test_line_without_baseline_point_in_crop_box_is_excluded_not_counted():
    crop = {"ln_in": {"gt_ids": ["ln_in"], "matched": {"ln_in"}}}
    sampled = [{"id": "ln_in", "block_id": "b1"}, {"id": "ln_out", "block_id": "b1"}]
    rows, excluded = t5_mf.build_rows(sampled, {"b1": {"gt_ids": ["ln_in"], "matched": {"ln_in"}}},
                                      {"ln_in", "ln_out"}, set())
    assert [r["id"] for r in rows] == ["ln_in"]
    assert excluded == [{"id": "ln_out", "block_id": "b1", "reason": "excluded_not_in_crop"}]


def _write_partial(path, meta, guard_hit=False, elapsed=0.0):
    path.write_text(json.dumps({"meta": meta, "per_page": [], "elapsed_cumulative_s": elapsed,
                                "guard_hit": guard_hit}))


def _current_meta():
    return {"config_sha256": t5_mf.sha256(t5_mf.CONFIG), "script_sha256": t5_mf.sha256(Path(t5_mf.__file__)),
            "model_sha256": None, "module_sha256": t5_mf.module_hashes()}


def test_resume_refused_after_a_guard_hit(tmp_path):
    out = tmp_path / t5_mf.FULL_OUTPUT
    _write_partial(out.with_name(out.stem + ".partial.json"), _current_meta(), guard_hit=True)
    with pytest.raises(RuntimeError, match="time-guard hit"):
        t5_mf.run(_cfg(), out, pages_limit=None, skip_blla=True)


def test_resume_refused_when_a_module_hash_differs(tmp_path):
    out = tmp_path / t5_mf.FULL_OUTPUT
    meta = _current_meta()
    meta["module_sha256"] = dict(meta["module_sha256"], **{"src/babaorum/line_geometry.py": "0" * 64})
    _write_partial(out.with_name(out.stem + ".partial.json"), meta)
    with pytest.raises(RuntimeError, match="refuse to resume"):
        t5_mf.run(_cfg(), out, pages_limit=None, skip_blla=True)


def test_clock_is_cumulative_across_resumes_and_guard_is_definitive(tmp_path):
    out = tmp_path / t5_mf.FULL_OUTPUT
    part = out.with_name(out.stem + ".partial.json")
    _write_partial(part, _current_meta(), elapsed=5 * 3600.0 + 1.0)  # already past the 5-hour guard
    report = t5_mf.run(_cfg(), out, pages_limit=None, skip_blla=True)
    assert report["status"] == "INCOMPLETE" and report["guard_hit"] is True
    assert "primary_tolerance_0.5h" not in report
    assert json.loads(part.read_text())["guard_hit"] is True  # the next resume will be refused


def test_crop_record_and_partial_file_are_json_serialisable(tmp_path):
    entry = t5_mf.crop_entry({"ln_a", "ln_b"}, ["ln_a", "ln_b", "ln_c"], 2, (1.0, 2.0, 3.0, 4.0))
    text = json.dumps({"crops": {"b1": entry}}, allow_nan=False, default=float)
    back = json.loads(text)["crops"]["b1"]
    assert back["matched"] == ["ln_a", "ln_b"] and back["gt_ids"] == ["ln_a", "ln_b", "ln_c"]
    rows, excluded = t5_mf.build_rows([{"id": "ln_a", "block_id": "b1"}], {"b1": entry}, {"ln_a"}, set())
    assert rows[0]["local_hit"] is True
    t5_mf.write_partial(tmp_path / "x.partial.json", {"config_sha256": "c"}, [{"page_id": "p", "crops": {"b1": entry}}],
                        1.0, guard_hit=False)
    assert json.loads((tmp_path / "x.partial.json").read_text())["per_page"][0]["crops"]["b1"]["matched"] == ["ln_a", "ln_b"]
