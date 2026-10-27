"""DATA-1 checks (docs/END_TO_END_DATA1_LOT.txt §2 as amended by §5 and §6). CPU, synthetic, no GPU, no real page.

(1) the cell components are the components of extract_polylines (same polylines, same order);
(2) scores are the mean of the detection probability over the cells; n_cells counts them;
(3) a mismatch between the two extractions is refused;
(4) no fusion and no second refinement (amendment 2 §6), no G, no e2e4_G_fixed.json, no --g-file;
(5) the output format: before list only;
(6) the config: pages mf_0000..mf_0049 in 3 shards, cap 480 min;
(7) full run and shards with spies: each shard lists only its pages; the union is exactly mf_0000..mf_0049
    with no duplicate; no page mf_0050..mf_0059 is listed, opened, read or hashed;
(8) the output never goes under the input folder; a full run writes only under /kaggle/working;
(9) the install cell of the notebooks is the block of KAGGLE_INSTALL.txt; three notebooks, one per shard.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import data1_fragments as D  # noqa: E402
import e2e4_extraction as E4  # noqa: E402
import e2e3_fp_classes as E3  # noqa: E402
from babaorum.experimental.stage_s_targets import extract_polylines  # noqa: E402


def _tuples(polys):
    return [[tuple(p) for p in poly] for poly in polys]


def _random_prob(seed, shape=(30, 60)):
    rng = np.random.default_rng(seed)
    blobs = rng.random(shape)
    for _ in range(2):
        blobs = (blobs + np.roll(blobs, 1, 0) + np.roll(blobs, 1, 1)) / 3.0
    return blobs


# ---- (1) same components as extract_polylines

@pytest.mark.parametrize("seed,threshold,min_cells", [(0, 0.5, 4), (1, 0.45, 4), (2, 0.55, 2), (3, 0.5, 9)])
def test_cell_components_are_the_components_of_extract_polylines(seed, threshold, min_cells):
    prob = _random_prob(seed)
    official = [p for p in extract_polylines(prob, stride=4, threshold=threshold, min_cells=min_cells) if len(p) >= 2]
    mine = [poly for poly, _ in D.components_with_cells(prob, stride=4, threshold=threshold, min_cells=min_cells)
            if len(poly) >= 2]
    assert len(official) > 0
    assert _tuples(mine) == _tuples(official)


# ---- (2) scores and n_cells

def test_score_is_the_mean_over_the_cells_of_a_component_and_small_blobs_are_dropped():
    prob = np.zeros((10, 10))
    prob[2:5, 2:5] = 0.9            # 9 cells, one component
    prob[7, 7:9] = 0.8              # 2 cells: below min_cells = 4, dropped
    comps = D.components_with_cells(prob, stride=4, threshold=0.5, min_cells=4)
    assert len(comps) == 1
    _, cells = comps[0]
    assert len(cells) == 9
    assert np.mean([prob[r, c] for r, c in cells]) == pytest.approx(0.9)


# ---- (3) refusal on mismatch (stub model with the shapes of the real one)

class _StubModel:
    def __init__(self, prob):
        import torch
        self._logit = torch.logit(torch.as_tensor(prob, dtype=torch.float32).clamp(1e-4, 1 - 1e-4))

    def enc_det(self, x):
        import torch
        return torch.zeros((1, 4, self._logit.shape[0], self._logit.shape[1]))

    def head_d(self, features):
        return self._logit[None, None], None, None


def test_mismatch_between_the_two_extractions_is_refused(monkeypatch):
    prob = _random_prob(5, shape=(20, 30))
    model = _StubModel(prob)
    ext = {"stride": 4, "threshold": 0.5, "min_cells": 4}
    extracted, scores, _ = D.infer_fragments(model, np.zeros((4, 4)), ext)
    assert len(extracted) == len(scores) > 0
    real = D.extract_polylines
    monkeypatch.setattr(D, "extract_polylines", lambda *a, **k: real(*a, **k)[:-1] or [[(0.0, 0.0), (1.0, 1.0)]])
    with pytest.raises(RuntimeError):
        D.infer_fragments(model, np.zeros((4, 4)), ext)


# ---- (4) no fusion, no second refinement, no G (amendment 2 §6)

def test_script_has_no_fusion_no_second_refine_no_G():
    src = (ROOT / "scripts/data1_fragments.py").read_text()
    assert "fuse_chain" not in src and "fused_score" not in src
    assert src.count("E4.refine(") == 1
    assert "e2e4_G_fixed" not in src and "--g-file" not in src and "G_file" not in src


# ---- (5) output format: before list only

def test_payload_has_the_format_without_after_and_without_G():
    ext = {"stride": 4, "threshold": 0.5, "min_cells": 4}
    extracted = [[(1.0, 2.0), (3.0, 2.0)], [(1.0, 9.0), (3.0, 9.0)]]
    scores = [(0.9, 4), (0.7, 6)]
    refined_b = [[[1.0, 2.0]], [[1.0, 9.0]]]
    sizes = {"factor": 1.25, "native_wh": [101, 77], "scaled_wh": [126, 96]}
    pay = D.assemble_payload("S2", "mf_0007", sizes, ext, extracted, scores, refined_b)
    assert set(pay) == {"page_id", "bras", "échelle", "factor", "native_wh", "scaled_wh", "stride", "seuil", "before"}
    assert pay["page_id"] == "mf_0007" and pay["bras"] == "S2" and pay["échelle"] == 1.25 == pay["factor"]
    assert pay["native_wh"] == [101, 77] and pay["scaled_wh"] == [126, 96]
    assert pay["stride"] == 4 and pay["seuil"] == 0.5
    assert [b["id"] for b in pay["before"]] == ["S2:mf_0007:b0", "S2:mf_0007:b1"]
    for b in pay["before"]:
        assert set(b) == {"id", "polyline_px", "refined_baseline_px", "score", "n_cells"}
    assert pay["before"][0]["n_cells"] == 4 and pay["before"][1]["score"] == 0.7
    json.dumps(pay, ensure_ascii=True)


# ---- (6) config

def test_config_pages_shards_and_cap():
    dcfg = json.loads(D.DATA_CONFIG.read_text())
    assert dcfg["pages"]["used"] == [f"mf_{i:04d}" for i in range(50)]
    assert "G" not in dcfg
    assert dcfg["budget"]["cap_minutes"] == 480
    assert dcfg["shards"] == [[0, 16], [17, 33], [34, 49]]
    assert dcfg["checkpoints"]["B3"]["sha256_prefix"] == "738ca116bdfcef16"
    assert dcfg["checkpoints"]["S2"]["sha256_prefix"] == "12fca335"
    assert dcfg["page_reference"]["sha256_prefix"] == "714bc571"
    assert D.ARMS == ("B3", "S2")
    assert not set(dcfg["pages"]["used"]) & D.FORBIDDEN


# ---- (7) runs with spies (I/O wrappers replaced; the inputs are real small files)

def _fake_input(tmp_path, pages):
    """Fake input folder: pages/ and images/ for the given pages, the E2E-4 reference, the checkpoints."""
    mf = tmp_path / "in" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "images").mkdir(parents=True)
    for pid in pages:
        (mf / "pages" / f"{pid}.json").write_text(json.dumps({"lines": [{"text": "ab", "legibility": "readable"}]}))
        (mf / "images" / f"{pid}.png").write_bytes(b"png-" + pid.encode())
    ref_sha = {}
    for pid in pages:
        ref_sha[f"pages/{pid}.json"] = E3.sha256(mf / "pages" / f"{pid}.json")
        ref_sha[f"images/{pid}.png"] = E3.sha256(mf / "images" / f"{pid}.png")
    report = tmp_path / "e2e4_extraction_report.json"
    report.write_text(json.dumps({"alphabet_size": 2, "steps": {"page_cache": {"sha256": ref_sha}}}))
    calib = tmp_path / "calibration"
    for arm in ("B3", "S2"):
        (calib / arm).mkdir(parents=True)
        (calib / arm / "checkpoint_last.pt").write_bytes(b"ckpt")
    return mf.parent, report, calib


def _no_prefix_config(tmp_path):
    cfg = json.loads(D.DATA_CONFIG.read_text())
    cfg["page_reference"]["sha256_prefix"] = ""
    for arm in ("B3", "S2"):
        cfg["checkpoints"][arm]["sha256_prefix"] = ""
    path = tmp_path / "data1_test_config.json"
    path.write_text(json.dumps(cfg))
    return path


def _args(root, report, calib, out, pages=None, shard=None):
    return argparse.Namespace(confirmed_go=True, tmp=str(root), calibration=str(calib),
                              e2e4_report=str(report), out_dir=str(out), pages=pages, shard=shard)


@pytest.fixture
def spies(monkeypatch):
    seen = []

    def record(name, fn):
        def wrapper(*args, **kwargs):
            seen.append((name, " ".join(str(a) for a in args)))   # every argument: the page id may be the 2nd
            return fn(*args, **kwargs)
        return wrapper

    def load_page(root, pid, target):
        seen.append(("load_page", pid))
        # native 101 x 77 px, factor 1.25: scaled_wh = (round(126.25), round(96.25)) = (126, 96)
        return {"image": np.zeros((96, 126), dtype=np.float32), "factor": 1.25,
                "native_wh": [101, 77], "scaled_wh": [126, 96]}

    monkeypatch.setattr(D, "_read_json", record("read_json", D._read_json))
    monkeypatch.setattr(D, "_sha", record("sha", D._sha))
    monkeypatch.setattr(D, "_load_page", load_page)
    monkeypatch.setattr(D, "_prefix_ok", lambda path, prefix: True)
    monkeypatch.setattr(D, "_load_model", lambda arm, ckpt, n: object())
    monkeypatch.setattr(D, "infer_fragments", lambda model, image, ext: (
        [[(1.0, 2.0), (3.0, 2.0)], [(1.0, 9.0), (3.0, 9.0)]], [(0.9, 4), (0.5, 5)], None))
    monkeypatch.setattr(E4, "refine", lambda model, features, polys, cfg: [[[x, y] for x, y in p] for p in polys])
    return seen


def _listed(seen):
    """Every mf_XXXX named by a read, a hash or a page load (the spies)."""
    return sorted({pid for _, text in seen for pid in re.findall(r"mf_\d{4}", text)})


def test_full_run_never_lists_opens_reads_or_hashes_pages_50_to_59(tmp_path, monkeypatch, spies):
    monkeypatch.setattr(D, "KAGGLE_OUT", tmp_path / "kaggle_working")
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    out = tmp_path / "kaggle_working" / "out"
    manifest = D.run(_args(root_parent, report, calib, out))
    assert manifest["status"] == "OK"
    assert manifest["pages_written"] == 100                   # 50 pages x 2 arms
    numbers = {int(p[3:]) for p in _listed(spies)}
    assert numbers and max(numbers) <= 49
    assert len(list((out / "fragments" / "B3").glob("*.json"))) == 50
    assert len(list((out / "fragments" / "S2").glob("*.json"))) == 50


def test_smoke_run_on_mf_0000_only(tmp_path, monkeypatch, spies):
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    manifest = D.run(_args(root_parent, report, calib, tmp_path / "smoke", pages=["mf_0000"]))
    assert manifest["status"] == "SMOKE"
    assert manifest["pages_written"] == 2
    assert [pid for name, pid in spies if name == "load_page"] == ["mf_0000"]
    assert not any(name == "read_json" for name, _ in spies)
    assert not any(50 <= int(p[3:]) <= 59 for p in _listed(spies))


def test_shards_are_the_three_consecutive_blocks_and_their_union_is_exactly_0000_0049(tmp_path, monkeypatch, spies):
    dcfg = json.loads(D.DATA_CONFIG.read_text())
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    monkeypatch.setattr(D, "KAGGLE_OUT", tmp_path / "kaggle_working")
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    union, sizes = [], []
    for k in range(3):
        spies.clear()
        out = tmp_path / "kaggle_working" / f"shard{k}"
        manifest = D.run(_args(root_parent, report, calib, out, shard=f"{k}/3"))
        assert manifest["status"] == "OK"
        assert manifest["shard"]["index"] == k and manifest["shard"]["of"] == 3
        listed = _listed(spies)
        first, last = dcfg["shards"][k]
        assert listed == [f"mf_{i:04d}" for i in range(first, last + 1)] == manifest["pages_fragments"]
        assert manifest["pages_written"] == 2 * len(listed)
        union += listed
        sizes.append(len(listed))
    assert sorted(union) == [f"mf_{i:04d}" for i in range(50)]
    assert len(union) == len(set(union)) == 50
    assert sizes == [17, 17, 16]
    assert not any(p in D.FORBIDDEN for p in union)


def test_shard_smoke_on_mf_0000_only(tmp_path, monkeypatch, spies):
    """The smoke of amendment 2: --shard 0/3 --pages mf_0000."""
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    manifest = D.run(_args(root_parent, report, calib, tmp_path / "smoke", pages=["mf_0000"], shard="0/3"))
    assert manifest["status"] == "SMOKE" and manifest["pages_written"] == 2
    assert [pid for name, pid in spies if name == "load_page"] == ["mf_0000"]


def test_pages_outside_the_shard_are_refused(tmp_path, monkeypatch, spies):
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    with pytest.raises(AssertionError):
        D.run(_args(root_parent, report, calib, tmp_path / "smoke", pages=["mf_0000"], shard="1/3"))


def test_forbidden_page_is_refused_even_in_smoke(tmp_path, monkeypatch, spies):
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    with pytest.raises(AssertionError):
        D.run(_args(root_parent, report, calib, tmp_path / "smoke", pages=["mf_0050"]))


# ---- (8) output guards

def test_output_never_under_the_input_folder(tmp_path):
    root = tmp_path / "e2e3" / "mf"
    with pytest.raises(SystemExit):
        D.check_output_dir(root / "out", root, smoke=True)
    with pytest.raises(SystemExit):
        D.check_output_dir(tmp_path / "e2e3" / "out", root, smoke=True)


def test_full_run_refused_outside_kaggle_working(tmp_path):
    with pytest.raises(SystemExit):
        D.check_output_dir(tmp_path / "out", tmp_path / "mf", smoke=False)


def test_run_refuses_without_confirmed_go(tmp_path):
    with pytest.raises(SystemExit):
        D.main(["--tmp", "x", "--calibration", "x", "--e2e4-report", "x", "--out-dir", "x"])


# ---- (10) LISEZMOI and page size fields (amendment 3 §7)

def test_readme_exists_and_page_size_fields_are_present_and_consistent(tmp_path, monkeypatch, spies):
    monkeypatch.setattr(D, "DATA_CONFIG", _no_prefix_config(tmp_path))
    root_parent, report, calib = _fake_input(tmp_path, [f"mf_{i:04d}" for i in range(50)])
    out = tmp_path / "smoke"
    manifest = D.run(_args(root_parent, report, calib, out, pages=["mf_0000"], shard="0/3"))
    readme = out / "LISEZMOI.txt"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    for fragment in ("factor = 32 / médiane", "round(W·factor)", "pixels de la PAGE MISE À L'ÉCHELLE",
                     "x_natif = x · W / W_s", "AUCUNE correspondance", "Pas de fusion", "mf_0000 à mf_0000"):
        assert fragment in text, fragment
    assert "<shard>" not in text and "<target>" not in text and "<stride>" not in text and "<K>" not in text
    assert manifest["status"] == "SMOKE"
    for arm in ("B3", "S2"):
        page = json.loads((out / "fragments" / arm / "mf_0000.json").read_text())
        assert set(page["native_wh"]) and len(page["native_wh"]) == 2
        assert len(page["scaled_wh"]) == 2 and page["factor"] == page["échelle"]
        W, H = page["native_wh"]
        Ws, Hs = page["scaled_wh"]
        f = page["factor"]
        assert [Ws, Hs] == [round(W * f), round(H * f)]
        assert abs(Ws / W - f) < 1 / W and abs(Hs / H - f) < 1 / H


# ---- (9) install cell and notebooks

def _builder():
    sys.path.insert(0, str(ROOT / "scripts/stage_s_notebooks"))
    import build_kaggle_data1 as builder  # noqa: E402
    return builder


def test_install_cell_of_the_builder_is_the_block_of_kaggle_install_txt():
    builder = _builder()
    txt = (ROOT / "scripts/stage_s_notebooks/KAGGLE_INSTALL.txt").read_text().splitlines()
    start = txt.index("Cellule (première cellule de code) :") + 1
    end = txt.index("Lancement des scripts (PYTHONPATH vers les sources du bundle, jamais d'import depuis le dossier courant) :")
    block = "\n".join(ln[4:] for ln in txt[start:end] if ln.startswith("    ")) + "\n"
    assert builder.install_cell() == block


def test_three_notebooks_each_with_its_shard_and_the_install_cell_first():
    builder = _builder()
    for k in range(3):
        nb = builder.build({rel: "0" * 64 for rel in builder.KIT_FILES}, k)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert "".join(code_cells[0]["source"]) == builder.install_cell()
        run_source = "".join(code_cells[1]["source"])
        assert f'SHARD = "{k}/3"' in run_source and '"--shard", SHARD' in run_source
        assert "--g-file" not in run_source and "e2e4_G_fixed" not in run_source
    assert builder.SHARDS == [0, 1, 2]
