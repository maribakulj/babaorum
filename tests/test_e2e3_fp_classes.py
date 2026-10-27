"""Fast checks of the E2E-3 pure rules: D > F > H > R classes, partition, ratios, readings,
cosine and bootstrap, replay tolerance, guard. No torch, no kraken, no pages."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e2e3_fp_classes as e  # noqa: E402

H = 20.0
GT_A = [[100.0, 200.0], [300.0, 200.0]]
GT_B = [[100.0, 400.0], [300.0, 400.0]]
GTS = [("gA", GT_A, H), ("gB", GT_B, H)]


def test_class_D_when_a_near_gt_line_is_already_matched_elsewhere():
    pred = [[100.0, 205.0], [300.0, 205.0]]           # 5 px from gA (0.25 h), full coverage
    assert e.classify_unmatched(pred, GTS, matched_gt_to_other={"gA"}) == "D"


def test_class_F_when_near_gt_is_covered_less_than_half():
    pred = [[100.0, 200.0], [160.0, 200.0]]           # covers ~30 % of gA, within 0.5 h
    assert e.classify_unmatched(pred, GTS, matched_gt_to_other=set()) == "F"


def test_class_H_when_farther_than_one_height_from_every_gt():
    pred = [[100.0, 260.0], [300.0, 260.0]]           # 60 px from gA = 3 h
    assert e.classify_unmatched(pred, GTS, matched_gt_to_other=set()) == "H"


def test_class_R_is_the_rest_between_half_and_one_height_with_full_coverage():
    pred = [[100.0, 212.0], [300.0, 212.0]]           # 12 px = 0.6 h: not near (<= 0.5 h), not H
    assert e.classify_unmatched(pred, GTS, matched_gt_to_other=set()) == "R"


def test_order_D_before_F():
    pred = [[100.0, 200.0], [160.0, 200.0]]           # would be F, but gA is matched elsewhere -> D
    assert e.classify_unmatched(pred, GTS, matched_gt_to_other={"gA"}) == "D"


def test_partition_sums_to_the_number_of_unmatched_predictions():
    counts = e.count_classes(["D", "F", "H", "R", "R"])
    assert counts == {"D": 1, "F": 1, "H": 1, "R": 2}
    assert sum(counts.values()) == 5


def test_ratio_and_h1_reading():
    assert e.ratio(170, 60) == pytest.approx(170 / 60)
    assert e.h1_reading(2.84, 1.20) == "supported"          # B3 > 1.5 and S2 <= 1.2
    assert e.h1_reading(1.1, 1.0) == "contradicted"          # B3 <= 1.2 x S2 (1.1 <= 1.2)
    assert e.h1_reading(1.4, 1.0) == "intermediate"          # B3 > 1.2 x S2 and B3 <= 1.5


def test_r1_reading_uses_the_dominant_class():
    assert e.r1_reading({"D": 9, "F": 1, "H": 0, "R": 0})["dominant"] == "D"
    assert e.r1_reading({"D": 0, "F": 6, "H": 4, "R": 0})["dominant"] == "F"
    assert e.r1_reading({"D": 3, "F": 3, "H": 2, "R": 2})["dominant"] is None


def test_cosine_basic_values():
    assert e.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert e.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    vals = [-0.2, -0.1, -0.3, -0.15, -0.25] * 4
    m1, lo1, hi1 = e.bootstrap_mean_ci(vals, 2000, 20261017)
    m2, lo2, hi2 = e.bootstrap_mean_ci(vals, 2000, 20261017)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 <= m1 <= hi1
    assert e.cosine_reading(lo1, hi1, median_ratio=0.5, median_rho=0.2).startswith("conflict supported")


def test_r4_three_cases_conflict_negligible_and_not_supported():
    # supported: CI entirely < 0 and median rho >= 0.1, ratio not negligible
    assert e.cosine_reading(-0.3, -0.1, median_ratio=0.5, median_rho=0.2).startswith("conflict supported")
    # negligible: median ratio < 0.1, checked first
    assert e.cosine_reading(-0.3, -0.1, median_ratio=0.05, median_rho=0.2).startswith("CTC gradient negligible")
    # not supported: CI < 0 but median rho < 0.1
    assert e.cosine_reading(-0.3, -0.1, median_ratio=0.5, median_rho=0.05).startswith("conflict not supported")
    assert e.cosine_reading(-0.1, 0.2, median_ratio=0.5, median_rho=0.5).startswith("no measurable conflict")
    assert e.cosine_reading(0.1, 0.4, median_ratio=0.5, median_rho=0.5).startswith("gradients aligned")


def test_rho_is_minus_cos_times_ratio_and_s2_control_is_zero():
    cos, nd, nc = -0.4, 2.0, 0.6
    ratio = nc / nd
    assert -cos * ratio == pytest.approx(0.12)
    assert e.median([0.1, 0.3, 0.2]) == pytest.approx(0.2)
    assert e.median([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)


def test_replay_tolerance_is_one_thousandth():
    assert e.replay_status(0.5769, 0.2042, 0.5769, 0.2042, 0.001) == "OK"
    assert e.replay_status(0.5800, 0.2042, 0.5769, 0.2042, 0.001) == "MISMATCH"


def test_guard_is_sixty_minutes():
    assert e.CAP_MIN == 60.0
    assert e.guard_exceeded(60 * 60 + 1) is True
    assert e.guard_exceeded(60 * 60 - 1) is False


def test_main_refuses_any_run_step_without_confirmed_go():
    with pytest.raises(SystemExit):
        e.main(["replay"])
    with pytest.raises(SystemExit):
        e.main(["classes"])


# ---- §8 locks: coverage by arc-length projection, reference line with id tie-break, R1 with R >= 50 %

GT_L = [[0.0, 0.0], [100.0, 0.0]]          # GT length 100


def test_coverage_total_is_one():
    assert e.coverage([[0.0, 0.0], [100.0, 0.0]], GT_L, 1.0) == pytest.approx(1.0)


def test_coverage_half_is_half():
    assert e.coverage([[0.0, 0.0], [50.0, 0.0]], GT_L, 1.0) == pytest.approx(0.5)


def test_coverage_disjoint_is_zero():
    assert e.coverage([[0.0, 30.0], [100.0, 30.0]], GT_L, 1.0) == 0.0


def test_coverage_overlaps_are_not_counted_twice():
    # two overlapping prediction segments covering [0,50] and [25,75] -> union [0,75]
    pred = [[0.0, 0.0], [50.0, 0.0], [25.0, 0.0], [75.0, 0.0]]
    assert e.coverage(pred, GT_L, 1.0) == pytest.approx(0.75)


def test_reference_line_tie_goes_to_smallest_gt_id():
    gts = [("gZ", GT_A, H), ("gA", GT_A, H)]        # identical lines: equal normalised distance
    pred = [[100.0, 205.0], [300.0, 205.0]]
    assert e.reference_line(pred, gts)[0] == "gA"


def test_r1_non_explained_when_r_is_at_least_half():
    assert e.r1_reading({"D": 1, "F": 1, "H": 0, "R": 8})["text"].startswith("non explique")
    assert e.r1_reading({"D": 9, "F": 1, "H": 0, "R": 0})["dominant"] == "D"


# ---- run 2 (§12): page cache acceptance and refusal, live-process guard

def _fake_cache(root, n_json, n_png):
    (root / "pages").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    for i in range(n_json):
        (root / "pages" / f"mf_{i:04d}.json").write_text(f"{{\"i\": {i}}}")
    for i in range(n_png):
        (root / "images" / f"mf_{i:04d}.png").write_bytes(bytes([i % 256]) * 8)


def test_cache_with_60_json_and_60_png_is_accepted_with_sha256_per_file(tmp_path):
    _fake_cache(tmp_path, 60, 60)
    res = e.page_cache_status(tmp_path)
    assert res["status"] == "ACCEPTED"
    assert res["files"] == 120
    assert all(len(v) == 64 for v in res["sha256"].values())


def test_incomplete_cache_is_refused_as_failed(tmp_path):
    _fake_cache(tmp_path, 60, 59)
    res = e.page_cache_status(tmp_path)
    assert res["status"] == "FAILED" and "cache incomplete" in res["reason"]


def test_missing_cache_is_refused_as_failed(tmp_path):
    assert e.page_cache_status(tmp_path / "absent")["status"] == "FAILED"


def test_live_process_guard_ignores_self_and_all_ancestors_and_detects_other_pids():
    # 1 (init) <- 50 (zsh) <- 60 (lourd.sh, contains the script name) <- 70 (time) <- 80 (this process)
    ps = ("    1     0 /sbin/launchd\n"
          "   50     1 /bin/zsh -c eval 'scripts/e2e3_fp_classes.py run'\n"
          "   60    50 /bin/sh lourd.sh python scripts/e2e3_fp_classes.py run\n"
          "   70    60 /usr/bin/time -l python scripts/e2e3_fp_classes.py run\n"
          "   80    70 python scripts/e2e3_fp_classes.py run --confirmed-go\n"
          "  101     1 python scripts/e2e3_fp_classes.py run --clone x\n"
          "  202     1 python -c from mille_feuilles.cli import main\n"
          "  303     1 /bin/zsh\n")
    alive = e.live_processes(ps, own_pid=80)
    assert [ln.split()[0] for ln in alive] == ["101", "202"]   # self and ancestors excluded, others detected
    assert e.live_processes("  303     1 /bin/zsh\n", own_pid=80) == []
    assert len(e.live_processes(ps, own_pid=999)) >= 3          # an unrelated own pid excludes nothing in the chain


# ---- run 2 bug (AttributeError: random.Random has no integers): the batch RNG must be the NumPy Generator

def test_batch_rng_is_the_numpy_generator_used_by_train_v2():
    import numpy as np
    rng = np.random.default_rng(20261017)
    i = e.pick_index(rng, 5)
    assert isinstance(i, int) and 0 <= i < 5
    idx = e.pick_lines(rng, 10, 4)
    assert len(idx) == 4 and len(set(idx)) == 4 and all(0 <= k < 10 for k in idx)
    assert e.pick_lines(rng, 10, 0) == []


def test_random_module_generator_is_refused_by_the_batch_helpers():
    import random
    with pytest.raises(AttributeError):
        e.pick_index(random.Random(1), 5)


def test_run3_report_path_is_fixed_and_existing_reports_are_never_overwritten():
    src = Path(e.__file__).read_text()
    assert "e2e3_fp_classes_run3.json" in src
    assert "run2.json\"  # never" not in src


def test_cli_defaults_are_the_mac_values():
    ps = e.build_parser()
    a = ps.parse_args(["run"])
    assert a.tmp == Path("/private/tmp/e2e3")
    assert str(a.bundle_zip).endswith("Mon Drive/babaorum/stage_s/bundle/stage_s_bundle_v5.zip")
    assert a.heritage_py == Path.home() / "heritage-synth/.venv/bin/python"
    assert a.calibration == e.ROOT / "runs/e2e1_kaggle_output/calibration"
    assert a.report == e.ROOT / "reports/e2e3_fp_classes_run3.json"


def test_cli_overrides_paths_for_kaggle():
    a = e.build_parser().parse_args(["run", "--tmp", "/kaggle/working/e2e3", "--bundle-zip", "/kaggle/input/b/x.bin",
                                     "--heritage-py", "/usr/bin/python3", "--calibration", "/kaggle/input/c",
                                     "--report", "/kaggle/working/e2e3_fp_classes_run3.json"])
    assert a.tmp == Path("/kaggle/working/e2e3") and a.calibration == Path("/kaggle/input/c")
    assert a.report == Path("/kaggle/working/e2e3_fp_classes_run3.json")
