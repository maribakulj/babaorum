"""LD-0R checks (docs/END_TO_END_LD0R_LOT.txt §4, §8 (3), amendements 1, 2, 5 et 7). CPU, no model, no VT text read.

(1) the LD-0 script imported by LD-0R is the frozen gel 6 (sha 4fae48f8…);
(2) the 26 pages of the config are the 26 juge files and the pool of the tirage (Mac only: skipped elsewhere);
(3) _band on a rectangle and an inclined polygon; a line where _band fails is counted apart;
(4) the text field is dropped when the JSON is read;
(5) a page whose sha differs is refused;
(6) reading rules: N=32, N=64 by p95, N=64 by share, ZONES_TROP_CHARGEES by share, band-failure ceiling exactly 5 %,
    Z-GT control, zones of mean a below 0.92;
(7) Z-GT boxes are built by region;
(8) INCOMPLETE on an error or fewer than 26 pages; the 26-page guard of the full run;
(9) the HF environment is set at import, before any huggingface import;
(10) the script reuses the LD-0 functions and copies none.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import ld0r_real_coverage as R  # noqa: E402

CFG = json.loads((ROOT / "config" / "ld0r_pages.json").read_text())
LD0_SHA = "4fae48f8616a6177819bfb2a6a636d140117681c536407216d92076a4da05913"


def test_ld0_script_is_the_frozen_gel6():
    assert R.sha256(ROOT / "scripts" / "ld0_zone_coverage.py") == LD0_SHA
    assert CFG["ld0_script"]["sha256"] == LD0_SHA


def test_config_lists_the_26_pages_of_the_juge_directory_and_of_the_pool():
    assert len(CFG["pages"]) == 26
    names = [p["page"] for p in CFG["pages"]]
    assert len(set(names)) == 26 and not any(n.startswith("mf_") for n in names)
    juge = Path(CFG["juge_dir"])
    if not juge.is_dir():
        pytest.skip("juge directory absent (Mac only)")
    assert sorted(p.stem for p in juge.glob("*.json")) == sorted(names)
    tirage = json.loads(Path(CFG["pool"]["tirage"]).read_text())
    assert sorted(tirage["pool"]) == sorted(names)


def test_band_on_a_rectangle_gives_exact_neg_pos_h():
    band = R.band_fn()
    assert band([[0, 10], [100, 10]], [[0, 0], [100, 0], [100, 20], [0, 20]]) == pytest.approx((10.0, 10.0))


def test_band_on_an_inclined_polygon_gives_exact_neg_pos_h():
    band = R.band_fn()
    neg, pos = band([[0, 0], [100, 0]], [[0, -5], [100, -3], [100, 7], [0, 5]])
    assert (neg, pos) == pytest.approx((5.0, 7.0))
    assert neg + pos == pytest.approx(12.0)


def test_a_line_where_band_fails_is_counted_apart_and_excluded():
    band = R.band_fn()
    true_lines = [
        {"gt_line_id": "a", "region": "r", "baseline": [[0, 10], [100, 10]],
         "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        {"gt_line_id": "b", "region": "r", "baseline": [[100, 10], [0, 10]],
         "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        {"gt_line_id": "c", "region": "r", "baseline": None, "polygon": [[0, 0]]},
    ]
    ok, n_fail = R.vt_lines(true_lines, band)
    assert [l["id"] for l in ok] == ["a"] and n_fail == 2
    assert ok[0]["h"] == pytest.approx(20.0)


def test_the_text_field_is_dropped_when_the_json_is_read(tmp_path):
    p = tmp_path / "page.json"
    p.write_text(json.dumps({"true_lines": [{"gt_line_id": "x", "text": "SECRET", "baseline": [[0, 0], [1, 0]]}]}))
    data = R.read_json_no_text(p)
    assert "SECRET" not in json.dumps(data) and "text" not in data["true_lines"][0]


def test_a_page_whose_image_sha_differs_is_refused(tmp_path):
    runs = tmp_path / "runs"
    juge = runs / "juge"
    juge.mkdir(parents=True)
    img = tmp_path / "img.png"
    Image.new("RGB", (8, 8), "white").save(img)
    run = runs / "p1.json"
    run.write_text(json.dumps({"image": str(img), "image_sha256": "0" * 64}))
    (juge / "p1.json").write_text(json.dumps({"true_lines": []}))
    entry = {"page": "p1", "run_sha256": R.sha256(run), "juge_sha256": R.sha256(juge / "p1.json"),
             "image_sha256": "0" * 64}
    with pytest.raises(R.ImageRefused):
        R.load_page_inputs(entry, juge)


def _page(a16, lpz, n_lines, a_gt=1.0):
    m = {"n_lines": n_lines, "a": a16, "lines_per_zone": lpz}
    return {"measures": {"Z-PP-blocs": {"marge16": m}, "Z-GT": {"marge16": {"a": a_gt, "n_lines": n_lines}}}}


def test_reading_n32_when_p95_and_big_zone_share_are_small():
    res = R.reading([_page(0.95, [3] * 99 + [20], 1000)], 0, 1000, CFG, smoke=False)
    assert res["status"] == "OK" and res["N"] == 32


def test_reading_n64_when_p95_between_32_and_64_and_share_above_64_is_small():
    # correction du test (§8 (3)) : comptes cohérents (somme des zones <= lignes validées)
    res = R.reading([_page(0.95, [3] * 90 + [40] * 10, 1000)], 0, 1000, CFG, smoke=False)
    assert res["status"] == "OK_N64" and res["N"] == 64


def test_reading_n64_when_p95_at_most_32_but_share_in_zones_above_32_exceeds_two_percent():
    res = R.reading([_page(0.95, [3] * 96 + [40] * 4, 1000)], 0, 1000, CFG, smoke=False)
    assert res["status"] == "OK_N64" and res["N"] == 64 and res["share_lines_in_zones_gt32"] > 0.02


def test_reading_zones_too_loaded_when_p95_at_most_64_but_share_above_64_exceeds_two_percent():
    res = R.reading([_page(0.95, [3] * 96 + [70] * 4, 1000)], 0, 1000, CFG, smoke=False)
    assert res["status"] == "ZONES_TROP_CHARGEES"


def test_reading_zones_too_loaded_when_p95_above_64():
    res = R.reading([_page(0.95, [3] * 80 + [90] * 20, 3000)], 0, 3000, CFG, smoke=False)
    assert res["status"] == "ZONES_TROP_CHARGEES"


def test_band_failure_ceiling_is_exactly_five_percent_then_adapter_suspect():
    good = [_page(0.99, [3] * 10, 100)]
    at = R.reading(good, n_band_fail=5, n_lines=95, cfg=CFG, smoke=False)     # 5 / 100 = 0.05
    assert at["status"] != "ADAPTER_SUSPECT" and at["band_failure_share"] == pytest.approx(0.05)
    above = R.reading(good, n_band_fail=6, n_lines=94, cfg=CFG, smoke=False)  # 6 / 100 = 0.06
    assert above["status"] == "ADAPTER_SUSPECT"


def test_z_gt_control_below_099_is_adapter_suspect():
    res = R.reading([_page(0.99, [3] * 10, 100, a_gt=0.5)], 0, 100, CFG, smoke=False)
    assert res["status"] == "ADAPTER_SUSPECT"


def test_reading_mean_a_below_092_is_zones_insuffisantes():
    res = R.reading([_page(0.80, [5] * 10, 100)], 0, 100, CFG, smoke=False)
    assert res["status"] == "ZONES_INSUFFISANTES_REEL"


def test_z_gt_boxes_are_built_by_region():
    lines = [
        {"region": "A", "polygon": [[0, 0], [10, 0], [10, 5]]},
        {"region": "A", "polygon": [[20, 20], [30, 30]]},
        {"region": "B", "polygon": [[100, 100], [110, 110]]},
    ]
    assert sorted(R.z_gt_boxes(lines)) == [(0.0, 0.0, 30.0, 30.0), (100.0, 100.0, 110.0, 110.0)]


def test_status_is_incomplete_on_an_error_or_under_twenty_six_pages():
    assert R.decide_status(25, 0, 26) == "INCOMPLETE"
    assert R.decide_status(26, 1, 26) == "INCOMPLETE"
    assert R.decide_status(26, 0, 26) == "COMPLETE"
    assert R.decide_status(1, 0, 1) == "COMPLETE"


def test_full_run_refuses_fewer_than_26_pages_or_a_page_subset():
    cfg25 = dict(CFG, pages=CFG["pages"][:25])
    with pytest.raises(SystemExit):
        R._select_pages(cfg25, argparse.Namespace(smoke=False, pages=None, confirmed_go=True))
    with pytest.raises(SystemExit):
        R._select_pages(CFG, argparse.Namespace(smoke=False, pages=["L_OEuvre_4613537_T0000001"], confirmed_go=True))
    with pytest.raises(SystemExit):
        R._select_pages(CFG, argparse.Namespace(smoke=False, pages=None, confirmed_go=False))


def test_hf_environment_is_set_at_import_before_any_huggingface_import():
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "60" and os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"


def test_the_script_imports_the_ld0_functions_and_copies_none():
    src = (ROOT / "scripts" / "ld0r_real_coverage.py").read_text()
    assert "import ld0_zone_coverage as L0" in src
    for name in ("def measure_lines", "def lines_per_zone", "def fraction_in_small_zones", "def zones_from_blocks"):
        assert name not in src
