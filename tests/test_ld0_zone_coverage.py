"""LD-0 checks (docs/END_TO_END_LD0_LOT.txt §10 and §11). CPU, synthetic pages, no model download.

Names containing "axel" compare with ~/axel at commit 7b75303 (Mac only: deselected in the Kaggle notebook).
(1) measures (a), (a'), (b), (c) and the margin on toy boxes;
(2) the frame: zone native x (W_s/W, H_s/H); scaled_wh = round(native_wh x factor);
(3) zones: Z-PP-blocs (one per block), Z-PP-colonnes (merge_columns + union), equal to axel zones_of on toy boxes;
(4) the reading rules of §11 (10);
(5) the config: 50 pages, cap 90 min, licence, vendored files;
(6) the vendored package is identical to commit 7b75303 of ~/axel (Mac only);
(7) a full run on 50 synthetic pages: no page mf_0050..mf_0059 is listed, opened, loaded or hashed (spies);
(8) output guards, licence guard, resume by input sha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "ld0_vendor"))
sys.path.insert(0, str(ROOT / "src"))

import ld0_zone_coverage as L  # noqa: E402
L._load_deps()  # axel vendor and babaorum imports, as at run time

AXEL = Path.home() / "axel"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- (1) measures on toy boxes

def _line(points):
    return np.asarray(points, dtype=float)


def test_line_inside_one_zone_counts_for_a_b_and_not_c():
    samples = [_line([(10, 10), (20, 10), (30, 10)])]
    m = L.measure_lines(samples, [(0, 0, 100, 100)], margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m["a"] == 1.0 and m["b"] == 1.0 and m["c"] == 0.0 and m["a_prime"] == 1.0


def test_line_cut_between_two_zones_is_in_b_not_in_a_and_counts_in_c():
    samples = [_line([(10, 10), (20, 10), (30, 10), (40, 10)])]
    zones = [(0, 0, 25, 100), (25, 0, 100, 100)]     # left half of the points in each zone
    m = L.measure_lines(samples, zones, margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m["a"] == 0.0 and m["b"] == 1.0 and m["c"] == 1.0
    assert L.lines_per_zone(m, 2) == [0, 0]          # a cut line is in no zone


def test_line_outside_every_zone_counts_nowhere():
    samples = [_line([(500, 500), (510, 500)])]
    m = L.measure_lines(samples, [(0, 0, 100, 100)], margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m["a"] == 0.0 and m["b"] == 0.0 and m["c"] == 0.0


def test_margin_16_brings_a_line_just_outside_into_the_zone():
    samples = [_line([(104, 50), (108, 50), (112, 50)])]     # zone ends at x = 100; 116 with the margin
    m0 = L.measure_lines(samples, [(0, 0, 100, 100)], margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    m16 = L.measure_lines(samples, [(0, 0, 100, 100)], margin=16, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m0["a"] == 0.0 and m16["a"] == 1.0


def test_a_prime_uses_seventy_five_percent():
    samples = [_line([(10, 10), (20, 10), (30, 10), (200, 10)])]   # 3 of 4 points inside: 0.75
    m = L.measure_lines(samples, [(0, 0, 100, 100)], margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m["a"] == 0.0 and m["a_prime"] == 1.0


def test_denominator_is_all_validated_lines():
    samples = [_line([(10, 10), (20, 10)]), _line([(500, 500), (510, 500)])]
    m = L.measure_lines(samples, [(0, 0, 100, 100)], margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m["n_lines"] == 2 and m["a"] == 0.5


# ---- (2) frame

def test_scaled_frame_is_native_times_factor_with_rounded_size():
    native = (101, 77)
    factor = 1.25
    scaled = (round(native[0] * factor), round(native[1] * factor))
    assert scaled == (126, 96)
    box = L.scale_box((10, 20, 30, 40), scaled[0] / native[0], scaled[1] / native[1])
    assert box == pytest.approx((10 * 126 / 101, 20 * 96 / 77, 30 * 126 / 101, 40 * 96 / 77))


def test_factor_rule_is_the_load_scaled_page_rule():
    lines = [{"normal_negative_px": 4.0, "normal_positive_px": 8.0},
             {"normal_negative_px": 5.0, "normal_positive_px": 7.0},
             {"normal_negative_px": 3.0, "normal_positive_px": 9.0}]
    assert L._page_factor(lines, 32.0) == pytest.approx(32.0 / 12.0)


# ---- (3) zones

def test_z_pp_blocs_is_one_zone_per_block_and_z_pp_colonnes_is_the_union_of_members():
    blocks = [(10, 10, 50, 30), (10, 32, 50, 52), (200, 10, 240, 30)]
    divisions, groups = L.zones_from_blocks(blocks, H_native=400, merge_gap_fraction=0.02, max_h=2400)
    assert divisions["Z-PP-blocs"] == blocks
    assert sorted(len(g) for g in groups) == [1, 2]
    assert (10, 10, 50, 52) in divisions["Z-PP-colonnes"]
    assert len(divisions["Z-PP-colonnes"]) == 2


@pytest.mark.skipif(not AXEL.exists(), reason="Mac only: needs ~/axel")
def test_axel_union_of_members_is_the_same_as_zones_of():
    """Same union as axel scripts/e12_kraken.py zones_of (lines 269-275), on toy boxes."""
    from axel.layout.chains import merge_columns
    boxes = [(10, 10, 50, 30), (12, 33, 52, 52), (10, 60, 50, 80), (300, 10, 340, 30)]
    groups = merge_columns(boxes, 0.02 * 400, 2400.0)
    ours = [L.union_box([boxes[i] for i in g]) for g in groups]
    axel = []
    for z in groups:
        bb = [boxes[i] for i in z]
        x0, y0 = min(b[0] for b in bb), min(b[1] for b in bb)
        x1, y1 = max(b[2] for b in bb), max(b[3] for b in bb)
        axel.append((x0, y0, x1, y1))
    assert ours == axel


# ---- (4) reading rules (§11 (10))

CFG_READING = {"usable_mean_a": 0.92, "N_if_p95_ok": 32, "N_else": 64}


def _stats(a_blocs, p_blocs, a_col, p_col):
    return {"Z-PP-blocs": {"a_mean": a_blocs, "p95": p_blocs}, "Z-PP-colonnes": {"a_mean": a_col, "p95": p_col}}


def test_no_cutting_division_usable_gives_correction():
    r = L.reading(_stats(0.80, 10, 0.85, 10), CFG_READING)
    assert r["choice"] is None and "correction" in r["action"]


def test_only_one_usable_is_retained_and_n_64_if_p95_too_high():
    r = L.reading(_stats(0.95, 40, 0.50, 10), CFG_READING)
    assert r["choice"] == "Z-PP-blocs" and r["N"] == 64


def test_both_usable_with_p95_ok_keeps_the_larger_a():
    r = L.reading(_stats(0.93, 20, 0.95, 20), CFG_READING)
    assert r["choice"] == "Z-PP-colonnes" and r["N"] == 32


def test_both_usable_only_one_p95_ok_keeps_that_one():
    r = L.reading(_stats(0.99, 40, 0.93, 20), CFG_READING)
    assert r["choice"] == "Z-PP-colonnes" and r["N"] == 32


def test_tie_goes_to_z_pp_blocs():
    r = L.reading(_stats(0.95, 20, 0.95, 20), CFG_READING)
    assert r["choice"] == "Z-PP-blocs"


# ---- (5) config

def test_config_lists_fifty_pages_cap_90_licence_and_vendor_shas():
    cfg = json.loads(L.CONFIG.read_text())
    assert cfg["pages"]["list"] == [f"mf_{i:04d}" for i in range(50)]
    assert cfg["budget"]["cap_minutes"] == 90
    assert cfg["license"]["text"] == "apache-2.0" and cfg["license"]["date"] == "2026-10-18"
    assert cfg["detector"]["revision"] == "e4f489dc536556fc1be02b973ba06756e0d4a2ac"
    assert cfg["required_outside_image"] == ["transformers==5.18.0", "huggingface_hub==1.33.0"]
    for rel, want in cfg["vendor"]["files"].items():
        assert _sha(ROOT / "scripts" / "ld0_vendor" / rel) == want, rel


@pytest.mark.skipif(not AXEL.exists(), reason="Mac only: needs ~/axel")
def test_axel_vendored_files_are_identical_to_commit_7b75303():
    for rel in json.loads(L.CONFIG.read_text())["vendor"]["files"]:
        src = subprocess.run(["git", "-C", str(AXEL), "show", f"7b75303:src/{rel}"],
                             capture_output=True, check=True).stdout
        assert src == (ROOT / "scripts" / "ld0_vendor" / rel).read_bytes(), rel


# ---- (6) guards and resume, on synthetic pages

def _make_page(root: Path, pid: str, W: int = 60, H: int = 80) -> None:
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(pid[3:]))
    img = Image.fromarray(rng.integers(0, 255, (H, W), dtype=np.uint8), mode="L")
    png = root / "images" / f"{pid}.png"
    img.save(png)
    line_id = f"{pid}_l0"
    polygon = [[5, 34], [50, 36], [50, 46], [5, 44]]
    page = {
        "page_id": pid,
        "image": {"sha256": _sha(png), "width": W, "height": H},
        "lines": [{"id": line_id, "block_id": f"{pid}_b0", "baseline": [[5, 40], [50, 41]],
                   "polygon": polygon, "text": "ab", "legibility": "readable"}],
        "blocks": [{"id": f"{pid}_b0", "category": "texte", "polygon": [[4, 33], [52, 33], [52, 47], [4, 47]],
                    "line_ids": [line_id]}],
    }
    (root / "pages" / f"{pid}.json").write_text(json.dumps(page))


def _reference(root: Path, pages) -> Path:
    ref = {}
    for pid in pages:
        ref[f"pages/{pid}.json"] = _sha(root / "pages" / f"{pid}.json")
        ref[f"images/{pid}.png"] = _sha(root / "images" / f"{pid}.png")
    path = root.parent / "e2e4_reference.json"
    path.write_text(json.dumps({"steps": {"page_cache": {"sha256": ref}}}))
    return path


class _Cfg:
    def __init__(self):
        self.id2label = {0: "text", 1: "title"}


class _Model:
    config = _Cfg()


def _fake_detector(monkeypatch):
    monkeypatch.setattr(L, "load_detector", lambda cfg_det, cache: (None, _Model(), "stub"))

    def fake_detect(proc, model, image_rgb, floor):
        w, h = image_rgb.size
        return {"width": w, "height": h,
                "blocks": [{"id": "b0", "label": "text", "score": 0.9,
                            "polygon": [[4, 33], [52, 33], [52, 47], [4, 47]]}]}
    monkeypatch.setattr(L, "detect", fake_detect)


@pytest.fixture
def no_reference_prefix(tmp_path, monkeypatch):
    """Synthetic pages: the E2E-4 reference prefix of the real config does not apply (the test has its own)."""
    cfg = json.loads(L.CONFIG.read_text())
    cfg["pages"]["reference_sha256_prefix"] = ""
    path = tmp_path / "cfg_test.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(L, "CONFIG", path)


def _args(root, out, reference, pages=None):
    return argparse.Namespace(confirmed_go=True, root=str(root), cache_dir=str(root.parent / "hf"),
                              out_dir=str(out), reference_report=str(reference), pages=pages)


@pytest.fixture
def spied(monkeypatch):
    seen = []
    real_load, real_read, real_sha = L._load_page, L._read_page_json, L.sha256

    def record(name, fn):
        def wrapper(*args, **kwargs):
            seen.append(" ".join(str(a) for a in args))
            return fn(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(L, "_load_page", record("load", real_load))
    monkeypatch.setattr(L, "_read_page_json", record("read", real_read))
    monkeypatch.setattr(L, "sha256", record("sha", real_sha))

    def forbidden(*a, **k):
        raise AssertionError("directory listing is forbidden in LD-0 (only explicit paths)")
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    return seen


def _numbers(seen):
    import re
    return {int(n) for text in seen for n in re.findall(r"mf_(\d{4})", text)}


def test_full_run_on_fifty_pages_never_touches_pages_50_to_59(tmp_path, monkeypatch, spied, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = [f"mf_{i:04d}" for i in range(50)]
    for pid in pages:
        _make_page(root, pid)
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    report = L.run(_args(root, tmp_path / "out", ref))
    assert report["status"] in ("COMPLETE", "MEASURE_CHECK_FAILED")
    assert len(report["pages"]) == 50
    numbers = _numbers(spied)
    assert numbers and max(numbers) <= 49
    assert all((tmp_path / "out" / "zones" / d / "mf_0007.json").exists() for d in L.DIVISIONS)


def test_resume_skips_pages_with_the_same_input_sha(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000", "mf_0001"]
    for pid in pages:
        _make_page(root, pid)
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    L.run(_args(root, tmp_path / "out", ref, pages=pages))
    again = L.run(_args(root, tmp_path / "out", ref, pages=pages))
    assert again["pages_resumed_from_disk"] == pages


def test_output_inside_the_input_folder_is_refused(tmp_path):
    root = tmp_path / "ld0_pages"
    with pytest.raises(SystemExit):
        L.check_output_dir(root / "out", root, smoke=True)


def test_full_run_is_refused_outside_kaggle_working(tmp_path):
    with pytest.raises(SystemExit):
        L.check_output_dir(tmp_path / "out", tmp_path / "ld0_pages", smoke=False)


def test_full_run_is_refused_with_empty_licence(tmp_path, monkeypatch):
    cfg = json.loads(L.CONFIG.read_text())
    cfg["license"]["text"] = ""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(L, "CONFIG", path)
    with pytest.raises(SystemExit):
        L.run(_args(tmp_path / "ld0_pages", tmp_path / "out", tmp_path / "ref.json"))


def test_page_list_contains_no_page_from_50_to_59():
    cfg = json.loads(L.CONFIG.read_text())
    assert not any(p in {f"mf_{i:04d}" for i in range(50, 60)} for p in cfg["pages"]["list"])


def test_notebook_install_cell_is_the_common_block_except_the_declared_lines():
    """§12 (1), §14 (1) and (8): only the declared lines differ from the common cell; -q is gone; UTC prints, flush."""
    import difflib
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    common = builder.common_install_cell().splitlines()
    ours = builder.install_cell().splitlines()
    changed = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=common, b=ours, autojunk=False).get_opcodes():
        if tag != "equal":
            changed.append((tag, common[i1:i2], ours[j1:j2]))
    assert all(tag in ("replace", "insert") for tag, _, _ in changed)
    assert any("timeout=900" in ln for _, _, new_lines in changed for ln in new_lines)
    text = "\n".join(ours)
    assert "'-q'" not in text
    assert "'pip', 'install', '--timeout', '30', '--retries', '1', pkg]" in text   # §14 (9)
    assert "_say('pip start', pkg, _utc())" in text
    assert "_say('pip end', pkg, _utc(), round(time.monotonic() - _t0, 1), 's')" in text
    assert 'REQUIRED_OUTSIDE_IMAGE = ["transformers==5.18.0", "huggingface_hub==1.33.0"]' in text
    assert "'transformers', 'huggingface_hub', 'torchvision', 'cv2'" in text


def test_run_cell_prints_the_utc_hour_before_each_heavy_import():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, dict(cfg["vendor"]["files"]),
                       list(cfg["pages"]["list"]))
    cell = "".join(nb["cells"][-1]["source"])
    order = [cell.index(f"_imp(\"{m}\")") for m in ("torch", "torchvision", "cv2", "transformers")]
    assert order == sorted(order)
    assert "print('import', name, _utc(), flush=True)" in cell
    assert cell.index("print('ld0 cell 2 start'") < order[0]


def test_notebook_in_the_repository_equals_build_output():
    """§12 (2): the notebook file is byte for byte the output of build() with the current kit."""
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    kit = {rel: _sha(ROOT / rel) for rel in builder.KIT_FILES}
    vendor = dict(cfg["vendor"]["files"])
    expected = json.dumps(builder.build(kit, vendor, list(cfg["pages"]["list"])), indent=1, ensure_ascii=False)
    actual = (ROOT / "scripts/stage_s_notebooks/kaggle_ld0_zone_coverage.ipynb").read_text()
    assert actual == expected


# ---- §12 (4): errors are recorded, a failed run still writes its report

def test_a_page_in_error_is_recorded_and_the_others_go_on(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000", "mf_0001"]
    for pid in pages:
        _make_page(root, pid)
    page = json.loads((root / "pages" / "mf_0001.json").read_text())
    page["lines"][0]["baseline"] = [[50, 41], [5, 40]]          # right-to-left: the geometry refuses it
    (root / "pages" / "mf_0001.json").write_text(json.dumps(page))
    ref = _reference(root, pages)                                # reference of the sha of the modified file
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    report = L.run(_args(root, tmp_path / "out", ref, pages=pages))
    assert report["status"] == "FAILED_PAGES"
    assert [e["page_id"] for e in report["errors"]] == ["mf_0001"]
    assert report["pages"] == ["mf_0000"] and report["reading"] is None


def test_an_uncaught_exception_writes_a_failed_report_and_keeps_the_measures(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000"]
    _make_page(root, "mf_0000")
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)

    def boom(cfg_det, cache):
        raise RuntimeError("detector unavailable (test)")
    monkeypatch.setattr(L, "load_detector", boom)
    out = tmp_path / "out"
    rc = L.main(["--confirmed-go", "--root", str(root), "--cache-dir", str(tmp_path / "hf"),
                 "--reference-report", str(ref), "--out-dir", str(out), "--pages", "mf_0000"])
    assert rc == 1
    report = json.loads((out / "ld0_zone_coverage_report.json").read_text())
    assert report["status"] == "FAILED" and "detector unavailable" in report["reason"]
    assert "Traceback" not in report["reason"] and report["traceback"]


# ---- §12 (7): factor formula = train_stage_s.load_scaled_page, on mf_0000 only (Mac, when the page is there)

@pytest.mark.skipif(not os.environ.get("LD0_MF0000_DIR"), reason="Mac only: LD0_MF0000_DIR must hold mf_0000 (pages/, images/)")
def test_factor_formula_equals_load_scaled_page_on_mf_0000():
    from train_stage_s import load_scaled_page  # noqa: E402  (imports torch: Mac venv only)
    root = Path(os.environ["LD0_MF0000_DIR"])
    cfg = json.loads(L.CONFIG.read_text())
    target = cfg["measure"]["target_median_height_px"]
    page = L._load_page(root, "mf_0000")
    ref = load_scaled_page(root, "mf_0000", target)
    assert L._page_factor(page["lines"], target) == pytest.approx(ref["factor"], abs=1e-12)
    raw = json.loads((root / "pages" / "mf_0000.json").read_text())
    W, H = raw["image"]["width"], raw["image"]["height"]
    Ws, Hs = round(W * ref["factor"]), round(H * ref["factor"])
    assert ref["image"].shape == (Hs, Ws)



# ---- §13 (1) and p6 precision: discovery never lists or traverses e2e3/mf/pages or e2e3/mf/images

RESERVED = ("e2e3/mf/pages", "e2e3/mf/images")


def _input_tree(tmp_path: Path) -> Path:
    base = tmp_path / "input"
    (base / "dsA").mkdir(parents=True)
    (base / "dsA" / "stage_s_bundle_v6.zip.bin").write_bytes(b"zip")
    (base / "dsB" / "scripts").mkdir(parents=True)
    (base / "dsB" / "scripts" / "ld0_zone_coverage.py").write_text("# kit")
    (base / "dsC").mkdir()
    (base / "dsC" / "e2e4_extraction_report.json").write_text("{}")
    mf = base / "dsD" / "e2e3" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "images").mkdir(parents=True)
    for pid in ("mf_0000", "mf_0050"):          # the decoy mf_0050 must never be named
        (mf / "pages" / f"{pid}.json").write_text("{}")
        (mf / "images" / f"{pid}.png").write_bytes(b"png")
    return base


@pytest.fixture
def listing_spies(monkeypatch):
    import glob as globmod
    import os
    seen = []

    def rec(name, fn):
        def wrapper(*args, **kwargs):
            seen.append((name, " ".join(str(a) for a in args)))
            return fn(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(os, "scandir", rec("os.scandir", os.scandir))
    monkeypatch.setattr(os, "listdir", rec("os.listdir", os.listdir))
    monkeypatch.setattr(os, "walk", rec("os.walk", os.walk))
    monkeypatch.setattr(globmod, "glob", rec("glob.glob", globmod.glob))
    monkeypatch.setattr(Path, "iterdir", rec("Path.iterdir", Path.iterdir))
    monkeypatch.setattr(Path, "glob", rec("Path.glob", Path.glob))
    monkeypatch.setattr(Path, "rglob", rec("Path.rglob", Path.rglob))
    return seen


def test_discovery_never_lists_or_traverses_the_reserved_pages(tmp_path, listing_spies):
    base = _input_tree(tmp_path)
    found = L.discover_inputs(base)
    assert set(found) == {"bundle", "kit", "reference", "run3_mf_dir", "roots_used"}
    assert found["run3_mf_dir"] == base / "dsD" / "e2e3" / "mf"
    assert listing_spies, "the spies must have seen the discovery"
    for name, arg in listing_spies:
        for bad in RESERVED:
            assert bad not in arg, (name, arg)
        assert "mf_005" not in arg, (name, arg)


def test_discovery_refuses_a_white_list_entry_under_the_reserved_pages(tmp_path, monkeypatch):
    cfg = json.loads(L.CONFIG.read_text())
    cfg["discovery"]["relpaths"]["run3_mf_dir"] = "e2e3/mf/pages"
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(L, "CONFIG", path)
    with pytest.raises(ValueError):
        L.discover_inputs(_input_tree(tmp_path))


def test_discovery_requires_each_explicit_path_exactly_once(tmp_path):
    base = _input_tree(tmp_path)
    (base / "dsC2").mkdir()
    (base / "dsC2" / "e2e4_extraction_report.json").write_text("{}")
    with pytest.raises(RuntimeError):
        L.discover_inputs(base)


def test_whitelist_in_the_config_names_no_reserved_page_path():
    cfg = json.loads(L.CONFIG.read_text())
    for rel in cfg["discovery"]["relpaths"].values():
        assert not any(bad in rel for bad in RESERVED), rel


def test_no_recursive_search_in_the_notebook_cell():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, dict(cfg["vendor"]["files"]),
                       list(cfg["pages"]["list"]))
    cell = "".join(nb["cells"][-1]["source"])
    for pattern in ("**", "rglob", "os.walk", "iterdir", "listdir", "scandir", ".glob(", "glob.glob("):
        assert pattern not in cell, pattern


# ---- §13 (2): exclusive assignment, declared; non-exclusive count descriptive

def test_overlapping_zones_are_assigned_to_one_zone_and_nonexclusive_counts_are_reported():
    z0, z1 = (0.0, 0.0, 100.0, 100.0), (50.0, 0.0, 200.0, 100.0)
    line_a = np.array([[70.0, 50.0]] * 20)                                  # in z0 and z1, both 1.0 -> z0 (tie)
    line_b = np.array([[60.0 + i, 50.0] for i in range(19)] + [[150.0, 50.0]])  # z0 0.95, z1 1.0 -> z1
    m = L.measure_lines([line_a, line_b], [z0, z1], margin=0, a_thr=0.9, a_prime_thr=0.75, b_thr=0.9)
    assert m["a"] == 1.0
    assert L.lines_per_zone(m, 2) == [1, 1]                  # exclusive: A to z0, B to z1
    assert m["nonexclusive_counts"] == [2, 2]                # descriptive: every zone counts every admissible line


# ---- §13 (3): share of lines in zones of at most 32 and 64 lines

def test_fraction_of_lines_in_zones_of_at_most_32_and_64_lines():
    m = {"n_lines": 4, "covered_one": np.array([True, True, True, False]),
         "best_zone": np.array([0, 0, 1, -1])}
    counts = [2, 40]
    assert L.fraction_in_small_zones(m, counts, 32) == pytest.approx(0.5)
    assert L.fraction_in_small_zones(m, counts, 64) == pytest.approx(0.75)


# ---- §13 (4): a measure from another launch is refused

def test_resume_refuses_a_measure_from_another_launch(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000", "mf_0001"]
    for pid in pages:
        _make_page(root, pid)
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    L.run(_args(root, tmp_path / "out", ref, pages=pages))
    mp = tmp_path / "out" / "measures" / "mf_0000.json"
    stored = json.loads(mp.read_text())
    stored["identity"]["script_sha256"] = "0" * 64
    mp.write_text(json.dumps(stored))
    with pytest.raises(L.ResumeIncompatible):
        L.run(_args(root, tmp_path / "out", ref, pages=pages))


# ---- §13 (5): report and manifest

def test_report_has_height_distribution_is_read_and_manifest_lists_measures(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000", "mf_0001"]
    for pid in pages:
        _make_page(root, pid)
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    report = L.run(_args(root, tmp_path / "out", ref, pages=pages))
    assert "lines_per_zone_by_height" in report["aggregate"]["Z-PP-blocs"]
    assert set(report["aggregate"]["Z-PP-blocs"]["lines_per_zone_by_height"]) == {"le_32", "gt_32"}
    assert report["classes_table"]["0"] == {"label": "text", "is_read": True}
    assert "INDICATEUR" in report["indicator_caveat"] and "0,90" in report["indicator_caveat"]
    manifest = json.loads((tmp_path / "out" / "ld0_manifest.json").read_text())
    assert "measures/mf_0000.json" in manifest["files_sha256"]
    assert "measures/mf_0001.json" in manifest["files_sha256"]
    assert "zones/Z-GT/mf_0000.json" in manifest["files_sha256"]


# ---- §13 (6): aggregate-only after the hard stop

def test_aggregate_only_writes_an_incomplete_report_without_reading(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000", "mf_0001"]
    for pid in pages:
        _make_page(root, pid)
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    L.run(_args(root, tmp_path / "out", ref, pages=pages))
    # the old run report must not survive: the hard-stop run rewrites it as INCOMPLETE
    rep = L.aggregate_only(argparse.Namespace(out_dir=str(tmp_path / "out")))
    assert rep["status"] == "INCOMPLETE" and rep["reading"] is None
    assert rep["pages"] == pages
    assert json.loads((tmp_path / "out" / "ld0_zone_coverage_report.json").read_text())["status"] == "INCOMPLETE"


def test_aggregate_only_refuses_measures_from_another_launch(tmp_path, monkeypatch, no_reference_prefix):
    root = tmp_path / "ld0_pages"
    pages = ["mf_0000"]
    _make_page(root, "mf_0000")
    ref = _reference(root, pages)
    _fake_detector(monkeypatch)
    monkeypatch.setattr(L, "KAGGLE_OUT", tmp_path)
    L.run(_args(root, tmp_path / "out", ref, pages=pages))
    mp = tmp_path / "out" / "measures" / "mf_0000.json"
    stored = json.loads(mp.read_text())
    stored["identity"]["config_sha256"] = "1" * 64
    mp.write_text(json.dumps(stored))
    with pytest.raises(L.ResumeIncompatible):
        L.aggregate_only(argparse.Namespace(out_dir=str(tmp_path / "out")))


# ---- §14 (6): timeouts, process groups, logs, heartbeat, exit record (fake subprocesses; no network, no model)

def test_timeout_kills_the_whole_group_and_returns_in_time(tmp_path):
    import time as _t
    gpid_file = tmp_path / "grandchild.pid"
    code = ("import subprocess, sys, time, os; "
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            "open(os.environ['GPID'], 'w').write(str(g.pid)); time.sleep(120)")
    env = dict(os.environ, GPID=str(gpid_file))
    t0 = _t.monotonic()
    r = L.run_step("sleeper", [sys.executable, "-c", code], 2.0, tmp_path / "logs", env=env, kill_grace_s=1.0)
    elapsed = _t.monotonic() - t0
    assert r["rc"] == "TIMEOUT"
    assert elapsed < 2.0 + 15.0
    gpid = int(gpid_file.read_text())
    _t.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(gpid, 0)                        # the grandchild that held the group is gone


def test_step_logs_are_written_in_files(tmp_path):
    code = "import sys; print('hello-out'); print('hello-err', file=sys.stderr)"
    r = L.run_step("talker", [sys.executable, "-c", code], 30, tmp_path / "logs")
    assert r["rc"] == 0
    assert "hello-out" in (tmp_path / "logs" / "talker.stdout").read_text()
    assert "hello-err" in (tmp_path / "logs" / "talker.stderr").read_text()
    assert "hello-err" in r["stderr_tail"]


def test_heartbeat_writes_at_least_one_line_with_a_short_interval(tmp_path):
    import time as _t
    hb = L.Heartbeat(tmp_path, interval_s=0.05)
    hb.start()
    hb.set_step("measure")
    _t.sleep(0.3)
    hb.stop()
    lines = [json.loads(x) for x in (tmp_path / "heartbeat.jsonl").read_text().splitlines()]
    assert len(lines) >= 2
    assert set(lines[0]) == {"utc", "step", "seconds_in_step", "measures_json"}
    assert lines[-1]["step"] == "measure"


def test_exit_record_is_written_after_an_exception_in_the_middle(tmp_path):
    cleaned = []

    def boom():
        raise RuntimeError("exception in the middle (test)")
    steps = [{"name": "first", "fn": lambda: None}, {"name": "middle", "fn": boom},
             {"name": "never", "fn": lambda: cleaned.append("never-ran")}]
    rec = L.execute_pipeline(tmp_path / "out", steps, heartbeat_interval_s=60, cleanup=lambda r: cleaned.append("cleanup"))
    exit_file = json.loads((tmp_path / "out" / "ld0_run_exit.json").read_text())
    assert exit_file["status"] == "FAILED" and exit_file["step"] == "middle"
    assert "RuntimeError" in exit_file["exception"]
    assert "never-ran" not in cleaned and "cleanup" in cleaned
    assert rec["status"] == "FAILED"


def test_timeout_runs_only_the_aggregate_step_after_measure(tmp_path):
    later = []
    slow = [sys.executable, "-c", "import time; time.sleep(60)"]
    agg = [sys.executable, "-c", "print('aggregate-ran')"]
    steps = [{"name": "measure", "argv": slow, "timeout_s": 1, "after_timeout": {"name": "aggregate", "argv": agg,
                                                                                  "timeout_s": 30}},
             {"name": "later", "fn": lambda: later.append("later")}]
    rec = L.execute_pipeline(tmp_path / "out", steps, heartbeat_interval_s=60)
    assert rec["status"] == "TIMEOUT" and rec["step"] == "measure" and rec["exit_code"] == "TIMEOUT"
    assert later == []
    assert "aggregate-ran" in (tmp_path / "out" / "logs" / "aggregate.stdout").read_text()
    assert json.loads((tmp_path / "out" / "ld0_run_exit.json").read_text())["status"] == "TIMEOUT"


def test_timeouts_and_heartbeat_are_in_the_config_and_the_manifest_values():
    cfg = json.loads(L.CONFIG.read_text())
    assert cfg["timeouts_s"] == {"pip_per_package": 900, "download": 600, "unzip": 900, "pytest": 900,
                                 "measure": 6600, "aggregate": 600}
    assert cfg["download_retries"] == 1 and cfg["faulthandler_seconds"] == 600
    assert cfg["heartbeat_interval_s"] == 60


def test_notebook_execution_cell_has_no_subprocess_call_and_no_capture():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, dict(cfg["vendor"]["files"]),
                       list(cfg["pages"]["list"]))
    cell = "".join(nb["cells"][-1]["source"])
    assert "subprocess.run" not in cell and "capture_output" not in cell   # the DESELECT names are not calls
    assert "execute_pipeline" in cell


# ---- §14 (10) and (11): HF download as its own step (fake downloads, no network), faulthandler, trace, heartbeat

def test_fake_download_that_hangs_is_killed_and_retried_once_in_time(tmp_path):
    import time as _t
    hang = [sys.executable, "-c", "import time; time.sleep(120)"]
    cleaned = []
    steps = [{"name": "download", "argv": hang, "timeout_s": 2, "retry": 1,
              "before_retry": lambda: cleaned.append("partial removed")}]
    t0 = _t.monotonic()
    rec = L.execute_pipeline(tmp_path / "out", steps, heartbeat_interval_s=60)
    elapsed = _t.monotonic() - t0
    assert elapsed < 2 * (2 + 15)
    assert rec["attempts"]["download"] == 2 and cleaned == ["partial removed"]
    assert rec["status"] == "TIMEOUT" and rec["step"] == "download"


def test_after_two_failed_downloads_the_exit_record_says_failed_download_with_the_cause(tmp_path):
    fail = [sys.executable, "-c", "import sys; print('network gone', file=sys.stderr); sys.exit(3)"]
    later = []
    steps = [{"name": "download", "argv": fail, "timeout_s": 30, "retry": 1, "before_retry": lambda: None},
             {"name": "measure", "fn": lambda: later.append("measure")}]
    rec = L.execute_pipeline(tmp_path / "out", steps, heartbeat_interval_s=60)
    exit_rec = json.loads((tmp_path / "out" / "ld0_run_exit.json").read_text())
    assert exit_rec["status"] == "FAILED" and exit_rec["step"] == "download"
    assert exit_rec["attempts"]["download"] == 2
    assert exit_rec["cause"]["rc"] == 3 and "network gone" in exit_rec["cause"]["stderr_tail"]
    assert later == []                       # the measure is not launched


def test_hf_timeouts_are_set_before_the_first_import_of_huggingface_hub():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, dict(cfg["vendor"]["files"]),
                       list(cfg["pages"]["list"]))
    cell2 = "".join(nb["cells"][2]["source"])
    first = cell2.index("os.environ.update(")
    assert first < cell2.index("import datetime")
    for key in ("HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT", "HF_HUB_DISABLE_XET"):
        assert f'"{key}"' in cell2[:first + 200]
    cell1 = builder.install_cell()
    assert cell1.index("os.environ.update(") < cell1.index("for mod in (")
    assert L.HF_ENV == {"HF_HUB_DOWNLOAD_TIMEOUT": "60", "HF_HUB_ETAG_TIMEOUT": "30", "HF_HUB_DISABLE_XET": "1"}


def test_measure_runs_offline_and_download_runs_with_disable_xet():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, dict(cfg["vendor"]["files"]),
                       list(cfg["pages"]["list"]))
    cell2 = "".join(nb["cells"][2]["source"])
    assert 'ENV_MEASURE = dict(ENV_DOWNLOAD, HF_HUB_OFFLINE="1")' in cell2
    assert '"name": "measure", "argv": MEASURE_ARGV, "cwd": str(ROOT_LD0), "env": ENV_MEASURE' in cell2
    assert "ENV_DOWNLOAD = dict(ENV)" in cell2 and '"HF_HUB_DISABLE_XET": "1"' in cell2


def test_faulthandler_is_armed_around_the_import_loop_of_cell_1():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cell1 = builder.install_cell()
    a = cell1.index("faulthandler.dump_traceback_later(600, exit=True")
    b = cell1.index("for mod in (")
    c = cell1.index("faulthandler.cancel_dump_traceback_later()")
    assert a < b < c


def test_cell_1_trace_file_receives_the_lines_with_flush(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cell1 = builder.install_cell()
    head = cell1[:cell1.index("_say('ld0 install start'")]
    trace = tmp_path / "ld0_cell1_trace.txt"
    head = head.replace("/kaggle/working/ld0_cell1_trace.txt", str(trace)).replace("os.makedirs('/kaggle/working', exist_ok=True)", "")
    ns = {}
    exec(head + "\n_say('ld0 install start', _utc())\n_say('import module', 'numpy', _utc())\n", ns)
    ns["_TRACE"].close()
    lines = trace.read_text().splitlines()
    assert lines[0].startswith("ld0 install start") and lines[1].startswith("import module numpy")


def test_heartbeat_construction_error_still_writes_the_exit_record(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("heartbeat cannot be built (test)")
    monkeypatch.setattr(L, "Heartbeat", broken)
    rec = L.execute_pipeline(tmp_path / "out", [{"name": "x", "fn": lambda: None}], heartbeat_interval_s=60)
    exit_rec = json.loads((tmp_path / "out" / "ld0_run_exit.json").read_text())
    assert exit_rec["status"] == "FAILED" and "heartbeat cannot be built" in exit_rec["exception"]
    assert rec["status"] == "FAILED"


def test_heartbeat_loop_logs_its_own_errors_and_keeps_going(tmp_path, monkeypatch):
    import time as _t
    calls = {"n": 0}
    hb = L.Heartbeat(tmp_path, interval_s=0.05)

    def flaky():
        calls["n"] += 1
        raise OSError("disk full (test)")
    monkeypatch.setattr(hb, "beat", flaky)
    hb.start()
    _t.sleep(0.3)
    hb.stop()
    assert calls["n"] >= 2
    assert "disk full" in (tmp_path / "heartbeat_errors.txt").read_text()


def test_clean_partial_cache_removes_only_incomplete_blobs(tmp_path):
    blobs = tmp_path / "models--PaddlePaddle--PP-DocLayoutV3_safetensors" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"partial")
    (blobs / "def").write_bytes(b"complete")
    assert L.clean_partial_cache(tmp_path, "PaddlePaddle/PP-DocLayoutV3_safetensors") == 1
    assert sorted(p.name for p in blobs.iterdir()) == ["def"]


# ---- §14 (12): both mounting formats, decoys at level 3, kit found by explicit candidates

def _tree_format(tmp_path: Path, fmt: str) -> Path:
    """fmt 'old': /kaggle/input/<slug>; fmt 'new': /kaggle/input/notebooks/mabaku/<slug> and
    /kaggle/input/datasets/mabaku/<slug>. Decoys: mf_0050 under e2e3/mf/pages and as a level-3 folder."""
    base = tmp_path / "input"
    if fmt == "old":
        bundle_root, kit_root, ref_root, run3_root, decoy_root = (base / "dsA", base / "babaorum-ld0-kit",
                                                                  base / "dsC", base / "dsD", base / "dsE")
    else:
        bundle_root = base / "datasets" / "mabaku" / "babaorum-stage-s-bundle-v6"
        kit_root = base / "datasets" / "mabaku" / "babaorum-ld0-kit"
        ref_root = base / "notebooks" / "mabaku" / "babaorum-e2e4-v3"
        run3_root = base / "notebooks" / "mabaku" / "babaorum-e2e3-run3-fp-classes"
        decoy_root = base / "notebooks" / "mabaku"
    bundle_root.mkdir(parents=True)
    (bundle_root / "stage_s_bundle_v6.zip.bin").write_bytes(b"zip")
    (kit_root / "scripts").mkdir(parents=True)
    (kit_root / "scripts" / "ld0_zone_coverage.py").write_text("# kit")
    ref_root.mkdir(parents=True)
    (ref_root / "e2e4_extraction_report.json").write_text("{}")
    mf = run3_root / "e2e3" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "images").mkdir(parents=True)
    for pid in ("mf_0000", "mf_0050"):
        (mf / "pages" / f"{pid}.json").write_text("{}")
        (mf / "images" / f"{pid}.png").write_bytes(b"png")
    decoy = decoy_root / "mf_0050"                   # a level-3 decoy named like a reserved page
    decoy.mkdir(parents=True, exist_ok=True)
    (decoy / "mf_0050.json").write_text("{}")
    return base


@pytest.mark.parametrize("fmt", ["old", "new"])
def test_discovery_with_decoys_in_both_formats_never_lists_or_opens_reserved_pages(tmp_path, fmt, monkeypatch):
    import glob as globmod
    import os
    base = _tree_format(tmp_path, fmt)
    seen, opened = [], []

    def rec(name, fn):
        def wrapper(*args, **kwargs):
            seen.append((name, " ".join(str(a) for a in args)))
            return fn(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(os, "scandir", rec("os.scandir", os.scandir))
    monkeypatch.setattr(os, "listdir", rec("os.listdir", os.listdir))
    monkeypatch.setattr(os, "walk", rec("os.walk", os.walk))
    monkeypatch.setattr(globmod, "glob", rec("glob.glob", globmod.glob))
    monkeypatch.setattr(Path, "iterdir", rec("Path.iterdir", Path.iterdir))
    monkeypatch.setattr(Path, "glob", rec("Path.glob", Path.glob))
    monkeypatch.setattr(Path, "rglob", rec("Path.rglob", Path.rglob))
    real_open = Path.read_bytes

    def watch_read(self):
        opened.append(str(self))
        return real_open(self)
    monkeypatch.setattr(Path, "read_bytes", watch_read)
    found = L.discover_inputs(base)
    assert found["run3_mf_dir"].name == "mf" and found["run3_mf_dir"].parent.name == "e2e3"
    assert "mf_0050" not in " ".join(str(v) for k, v in found.items() if k != "roots_used")
    for name, arg in seen:
        assert "e2e3/mf" not in arg, (name, arg)           # nothing listed inside, or of, e2e3/mf
    assert not any(arg.rstrip("/").endswith("mf_0050") for _, arg in seen)
    assert not any("mf_0050" in o for o in opened)          # the decoy is never opened, hashed or read
    assert not any(str(o).endswith("mf_0050") for o in opened)
    assert found["roots_used"]                              # roots are printed and kept, names only
    assert all(not r.startswith("/") for r in found["roots_used"])


def test_kit_is_found_by_explicit_candidates_in_both_formats(tmp_path):
    for fmt in ("old", "new"):
        base = _tree_format(tmp_path / fmt, fmt)
        kit_script = L.discover_inputs(base)["kit"]
        candidates = [base / "babaorum-ld0-kit" / "scripts" / "ld0_zone_coverage.py",
                      base / "datasets" / "mabaku" / "babaorum-ld0-kit" / "scripts" / "ld0_zone_coverage.py"]
        assert L.find_kit_script([str(c) for c in candidates]) == kit_script


def test_kit_candidates_refuse_zero_or_two_hits(tmp_path):
    (tmp_path / "a" / "scripts").mkdir(parents=True)
    (tmp_path / "a" / "scripts" / "ld0_zone_coverage.py").write_text("#")
    (tmp_path / "b" / "scripts").mkdir(parents=True)
    (tmp_path / "b" / "scripts" / "ld0_zone_coverage.py").write_text("#")
    with pytest.raises(RuntimeError):
        L.find_kit_script([str(tmp_path / "a" / "scripts" / "ld0_zone_coverage.py"),
                           str(tmp_path / "b" / "scripts" / "ld0_zone_coverage.py")])
    with pytest.raises(RuntimeError):
        L.find_kit_script([str(tmp_path / "missing.py")])


def test_notebook_kit_candidates_are_the_two_explicit_paths_of_the_config_and_exactly_one_must_exist(tmp_path):
    import textwrap
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld0 as builder  # noqa: E402
    cfg = json.loads(L.CONFIG.read_text())
    nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, dict(cfg["vendor"]["files"]),
                       list(cfg["pages"]["list"]))
    cell2 = "".join(nb["cells"][2]["source"])
    assert json.dumps(cfg["discovery"]["kit_dataset_candidates"]) in cell2 or \
        all(c in cell2 for c in cfg["discovery"]["kit_dataset_candidates"])
    assert "/kaggle/input/babaorum-ld0-kit/scripts/ld0_zone_coverage.py" in cell2
    assert "/kaggle/input/datasets/mabaku/babaorum-ld0-kit/scripts/ld0_zone_coverage.py" in cell2
    start = cell2.index("    _hits = [Path(c)")
    end = cell2.index('    print("kit script"')
    snippet = textwrap.dedent(cell2[start:end])
    for case, make in (("one", ["one"]), ("zero", []), ("two", ["one", "two"])):
        d = tmp_path / case
        cands = [str(d / "a" / "scripts" / "ld0_zone_coverage.py"), str(d / "b" / "scripts" / "ld0_zone_coverage.py")]
        for name in make:
            target = d / ("a" if name == "one" else "b") / "scripts" / "ld0_zone_coverage.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("#")
        ns = {"KIT_CANDIDATES": cands, "Path": Path, "KIT_DIR": None}
        if case == "one":
            exec(snippet, ns)
            assert ns["KIT_DIR"] == d / "a"
        else:
            with pytest.raises(AssertionError):
                exec(snippet, ns)
