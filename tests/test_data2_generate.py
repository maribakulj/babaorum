"""DATA-2 checks (docs/END_TO_END_DATA2_LOT.txt §1, §2, §3, §9, §11). CPU, synthetic pages, no generator run.

(1) rename: only page_id and image.path change; PNG bytes unchanged; file names are mfn_XXXX; manifest sha = file sha;
(2) config: seed 20261021, 20 pages, commit 23875ab, pilot sha, the generator command arguments;
(3) index rg.blind.gt-index.v1: 20 entries, equal to the manifest and to the files;
(4) overlap: sha equality -> OVERLAP_PAGE; 5 % threshold (strictly above) -> OVERLAP_TEXT_FLAGGED; NFC normalisation;
    the reads are mf_0000..0049 JSON only (spies: no mf_005x read, listed, hashed; no glob, no listdir);
(5) DATA-1 amended mode (--generation-manifest): the page list and the sha come from the manifest; the E2E-4 report is
    never read;
(6) data2_common.py is a verbatim copy of the frozen LD-0 functions (provenance, §11 (4));
(7) every kit file is under 300 KB (§11 (5)).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import data2_generate as G  # noqa: E402
import data2_common as C  # noqa: E402

RESERVED = re.compile(r"mf_005\d")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _raw_page(raw: Path, mf: str, seed: int) -> None:
    (raw / "pages").mkdir(parents=True, exist_ok=True)
    (raw / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (40, 30), dtype=np.uint8), mode="L").save(raw / "images" / f"{mf}.png")
    page = {
        "schema_version": "0.3.0", "page_id": mf,
        "image": {"path": f"images/{mf}.png", "sha256": _sha(raw / "images" / f"{mf}.png"),
                  "width": 30, "height": 40, "dpi": 150, "color_mode": "L"},
        "lines": [{"id": f"{mf}_l00000", "block_id": f"{mf}_b0000", "baseline": [[2, 20], [28, 21]],
                   "polygon": [[2, 15], [28, 16], [28, 25], [2, 24]], "text": f"ligne  {seed}",
                   "legibility": "readable", "extensions": {}, "word_ids": []}],
        "blocks": [{"id": f"{mf}_b0000", "category": "texte", "polygon": [[1, 14], [29, 15], [29, 26], [1, 25]],
                    "line_ids": [f"{mf}_l00000"], "article_id": "a0"}],
        "provenance": {"seed": seed},
    }
    (raw / "pages" / f"{mf}.json").write_text(json.dumps(page))


# ---- (2) configuration and generator command

def test_config_matches_the_lot():
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    g = cfg["generator"]
    assert g["commit"] == "23875ab05425ce2aa07742bf314d02436c2f5642"
    assert g["seed"] == 20261021 and g["pages"] == 20 and g["pilot_profile"] == "pilot-v0.2"
    assert g["render"] == {"width": 2680, "height": 3698, "dpi": 150, "degradation": "mixed", "columns": None}
    assert cfg["pages"]["list"] == [f"mfn_{i:04d}" for i in range(20)]
    assert cfg["overlap"]["text_share_threshold"] == 0.05
    assert cfg["overlap"]["mf_pages"] == [f"mf_{i:04d}" for i in range(50)]
    assert cfg["pilot_config_sha256"]["sha256"] == "b984106f76b09fb7b92539f1ec682f50a031d72c2f9e3d3b40ac9d339e28e4c7"
    assert len(g["mille_feuilles_sha256_from_bundle_manifest"]) == 51
    assert cfg["timeouts_s"]["generate"] == 2700 and cfg["timeouts_s"]["overlap"] == 600


def test_generator_command_carries_seed_pages_render_and_no_mf_output_name(tmp_path):
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    cmd = " ".join(G.generate_command(tmp_path, tmp_path / "raw", cfg))
    for part in ("--seed 20261021", "--pages 20", "--width 2680", "--height 3698", "--dpi 150",
                 "--degradation mixed", "--jobs 1"):
        assert part in cmd, part
    assert "mfn" not in cmd                       # the generator writes the raw names; they are renamed afterwards


# ---- (1) rename

def test_rename_changes_only_page_id_and_image_path_and_keeps_png_bytes(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for i in range(20):
        _raw_page(raw, f"mf_{i:04d}", seed=i)
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    manifest = G.rename_all(raw, out, cfg, {"test": True})
    for row in manifest["pages"]:
        assert row["fields_changed"] == ["page_id", "image.path"], row["fields_changed"]
        mfn = row["page_id"]
        new = json.loads((out / "mfn" / "pages" / f"{mfn}.json").read_text())
        old = json.loads((raw / "pages" / f"{row['mf_name']}.json").read_text())
        assert new["page_id"] == mfn and new["image"]["path"] == f"images/{mfn}.png"
        assert new["lines"] == old["lines"] and new["blocks"] == old["blocks"]
        restored = json.loads(json.dumps(new))
        restored["page_id"] = old["page_id"]
        restored["image"]["path"] = old["image"]["path"]
        assert restored == old                  # nothing else differs from the raw page
        assert _sha(raw / "images" / f"{row['mf_name']}.png") == _sha(out / "mfn" / "images" / f"{mfn}.png")


def test_rename_manifest_hashes_are_the_file_hashes_and_no_output_name_is_mf(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for i in range(20):
        _raw_page(raw, f"mf_{i:04d}", seed=i)
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    manifest = G.rename_all(raw, out, cfg, {})
    for key, value in manifest["files_sha256"].items():
        assert _sha(out / "mfn" / key) == value, key
    names = [p.relative_to(out) for p in out.rglob("*") if p.is_file()]
    assert not any("mf_" in n.name for n in names)
    assert all(n.name.startswith("mfn_") or n.name.endswith(".json") for n in names)
    assert len(manifest["files_sha256"]) == 40


# ---- (3) index

def test_index_has_the_rg_format_and_equals_the_manifest_and_the_files(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for i in range(20):
        _raw_page(raw, f"mf_{i:04d}", seed=i)
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    G.rename_all(raw, out, cfg, {})
    idx = G.build_index(out)
    assert idx["schema"] == "rg.blind.gt-index.v1"
    assert idx["generation_manifest_sha256"] == _sha(out / "data2_generation_manifest.json")
    assert list(idx["pages"]) == [f"mfn_{i:04d}" for i in range(20)]
    for mfn, sha in idx["pages"].items():
        assert sha == _sha(out / "mfn" / "pages" / f"{mfn}.json")
    on_disk = json.loads((out / "rg-v2-mfn-gt-index.json").read_text())
    assert on_disk == idx


# ---- (4) overlap

def _mf_reference(mf_dir: Path, images: dict) -> dict:
    sha = {}
    for name, png in images.items():
        sha[f"images/{name}.png"] = _sha(png)
    return {"steps": {"page_cache": {"sha256": sha}}}


def _make_overlap_fixture(tmp_path: Path, mf_text_lines: list, mfn_text_lines: list, same_image: bool = False):
    """mf_0000..0049 JSON with the given texts (first page), mfn_0000..0019 with mfn texts (10 lines per page)."""
    run3 = tmp_path / "run3" / "e2e3" / "mf"
    (run3 / "pages").mkdir(parents=True)
    (run3 / "images").mkdir(parents=True)
    for i in range(50):
        lines = [{"text": t} for t in mf_text_lines] if i == 0 else [{"text": f"mf page {i} other"}]
        (run3 / "pages" / f"mf_{i:04d}.json").write_text(json.dumps({"lines": lines}))
        Image.fromarray(np.full((8, 8), i, dtype=np.uint8), mode="L").save(run3 / "images" / f"mf_{i:04d}.png")
    images = {f"mf_{i:04d}": run3 / "images" / f"mf_{i:04d}.png" for i in range(50)}
    ref = _mf_reference(run3, images)
    out = tmp_path / "out"
    for i in range(20):
        (out / "mfn" / "pages").mkdir(parents=True, exist_ok=True)
        (out / "mfn" / "images").mkdir(parents=True, exist_ok=True)
        page_lines = [{"text": t} for t in mfn_text_lines[i]]
        (out / "mfn" / "pages" / f"mfn_{i:04d}.json").write_text(json.dumps({"page_id": f"mfn_{i:04d}", "lines": page_lines}))
        if same_image and i == 3:
            Image.fromarray(np.full((8, 8), 0, dtype=np.uint8), mode="L").save(out / "mfn" / "images" / f"mfn_{i:04d}.png")
        else:
            Image.fromarray(np.full((8, 8), 200 + i, dtype=np.uint8), mode="L").save(out / "mfn" / "images" / f"mfn_{i:04d}.png")
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    return out, run3, ref, cfg


def test_overlap_same_image_sha_gives_overlap_page(tmp_path):
    out, mf_dir, ref, cfg = _make_overlap_fixture(tmp_path, ["abc"], [["x"] * 10 for _ in range(20)], same_image=True)
    # make the generated image 3 identical to mf_0000's bytes
    (out / "mfn" / "images" / "mfn_0003.png").write_bytes((mf_dir / "images" / "mf_0000.png").read_bytes())
    rep = G.overlap_check(out, mf_dir, ref, cfg)
    assert rep["status"] == "OVERLAP_PAGE" and rep["image_sha_matches"] == ["mfn_0003"]


def test_overlap_text_threshold_is_five_percent_strictly_above(tmp_path):
    mf_texts = [f"mf line {k}" for k in range(10)]                  # 10 texts in mf_0000
    # 200 mfn lines in total: 10 in mf (5 %) -> OK ; 11 in mf (5.5 %) -> flagged
    base = [[f"fresh {i} {k}" for k in range(10)] for i in range(20)]
    out, mf_dir, ref, cfg = _make_overlap_fixture(tmp_path / "ok", mf_texts, [list(r) for r in base])
    ok_lines = [list(r) for r in base]
    ok_lines[0][:10] = mf_texts
    for i, lines in enumerate(ok_lines):
        page = out / "mfn" / "pages" / f"mfn_{i:04d}.json"
        page.write_text(json.dumps({"page_id": f"mfn_{i:04d}", "lines": [{"text": t} for t in lines]}))
    rep = G.overlap_check(out, mf_dir, ref, cfg)
    assert rep["lines_in_mf_text"] == 10 and rep["text_share_global"] == pytest.approx(0.05)
    assert rep["status"] == "OK"
    flagged = [list(r) for r in base]
    flagged[0][:10] = mf_texts
    flagged[1][0] = mf_texts[0]
    for i, lines in enumerate(flagged):
        (out / "mfn" / "pages" / f"mfn_{i:04d}.json").write_text(json.dumps({"page_id": f"mfn_{i:04d}", "lines": [{"text": t} for t in lines]}))
    rep2 = G.overlap_check(out, mf_dir, ref, cfg)
    assert rep2["lines_in_mf_text"] == 11 and rep2["status"] == "OVERLAP_TEXT_FLAGGED"


def test_overlap_normalises_nfc_and_spaces_but_keeps_case(tmp_path):
    assert G.normalise_text("Café  du   Nord ") == "Café du Nord"
    assert G.normalise_text("Abc") != G.normalise_text("abc")


def test_overlap_reads_only_mf_0000_to_0049_by_explicit_paths(tmp_path, monkeypatch):
    out, mf_dir, ref, cfg = _make_overlap_fixture(tmp_path, ["abc"], [["x"] * 10 for _ in range(20)])
    seen = []

    def spy(name, fn):
        def w(self, *a, **k):
            seen.append((name, str(self)))
            return fn(self, *a, **k)
        return w

    def forbidden(*a, **k):
        raise AssertionError("listing is forbidden in the overlap check")
    monkeypatch.setattr(Path, "read_text", spy("read_text", Path.read_text))
    monkeypatch.setattr(Path, "read_bytes", spy("read_bytes", Path.read_bytes))
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    G.overlap_check(out, mf_dir, ref, cfg)
    for name, path in seen:
        assert not RESERVED.search(path), (name, path)


def test_overlap_refuses_a_reference_not_listing_the_fifty_pages(tmp_path):
    out, mf_dir, ref, cfg = _make_overlap_fixture(tmp_path, ["abc"], [["x"] * 10 for _ in range(20)])
    cfg["overlap"]["mf_pages"] = [f"mf_{i:04d}" for i in range(60)]
    with pytest.raises(AssertionError):
        G.overlap_check(out, mf_dir, ref, cfg)


# ---- (5) DATA-1 in generation-manifest mode (§3 of the lot): no E2E-4 report read

@pytest.mark.skipif(not os.environ.get("DATA2_WITH_TORCH"),
                    reason="imports DATA-1 (torch): runs in the Kaggle pytest of the fragments carnet")
def test_data1_generation_mode_takes_pages_and_reference_from_the_manifest(tmp_path, monkeypatch):
    import data1_fragments as D1
    raw, out = tmp_path / "raw", tmp_path / "gen"
    for i in range(20):
        _raw_page(raw, f"mf_{i:04d}", seed=i)
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    manifest = G.rename_all(raw, out, cfg, {})
    gm = out / "data2_generation_manifest.json"
    seen_reports = []
    real_read = Path.read_text

    def watch(self, *a, **k):
        if str(self).endswith("e2e4_extraction_report.json"):   # the kit config e2e4_extraction.json is allowed
            seen_reports.append(str(self))
        return real_read(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", watch)
    monkeypatch.setattr(D1, "_load_model", lambda arm, ckpt, n: object())
    monkeypatch.setattr(D1, "_prefix_ok", lambda path, prefix: True)
    monkeypatch.setattr(D1, "_load_page", lambda root, pid, target: {"image": np.zeros((4, 4), np.float32),
                                                                       "factor": 1.25, "native_wh": [30, 40],
                                                                       "scaled_wh": [38, 50]})
    monkeypatch.setattr(D1, "infer_fragments", lambda model, image, ext: ([[(1.0, 2.0), (3.0, 2.0)]],
                                                                           [(0.9, 4)], None))
    monkeypatch.setattr(D1.E4, "refine", lambda model, features, polys, c: [[[x, y] for x, y in p] for p in polys])
    monkeypatch.setattr(D1.E3, "alphabet_from_json", lambda pages: ["<blank>", "a"])
    monkeypatch.setattr(D1, "KAGGLE_OUT", tmp_path)
    for arm in ("B3", "S2"):                       # the checkpoint files must exist (their prefixes are not checked here)
        (tmp_path / arm).mkdir(exist_ok=True)
        (tmp_path / arm / "checkpoint_last.pt").write_bytes(b"ckpt")
    args = argparse.Namespace(confirmed_go=True, tmp=str(out), calibration=str(tmp_path), e2e4_report=None,
                              out_dir=str(tmp_path / "frag"), pages=["mfn_0000"], shard=None,
                              generation_manifest=str(gm))
    res = D1.run(args)
    assert res["status"] == "SMOKE"
    assert res["inputs"]["generation_manifest_sha256"] == _sha(gm)
    assert seen_reports == []                  # the E2E-4 report is never read in generation mode
    assert res["pages_fragments"] == ["mfn_0000"]


# ---- (6) provenance: the copied LD-0 functions are the frozen text

def test_data2_common_functions_are_verbatim_copies_of_frozen_ld0():
    ld0_src = (ROOT / "scripts" / "ld0_zone_coverage.py").read_text()
    tree = ast.parse(ld0_src)
    ld0 = {n.name: ast.get_source_segment(ld0_src, n) for n in tree.body if getattr(n, "name", None)}
    common_src = (ROOT / "scripts" / "data2_common.py").read_text()
    common = {n.name: ast.get_source_segment(common_src, n) for n in ast.parse(common_src).body
              if getattr(n, "name", None)}
    for name in ("_child_dirs", "find_kit_script", "_group_alive", "_signal_group",
                 "_kill_group", "_tail", "run_step", "Heartbeat", "execute_pipeline"):
        expected = ld0[name]
        if name == "execute_pipeline":
            expected = expected.replace("ld0_run_exit.json", "data2_run_exit.json")
        assert common[name] == expected, name
    # discover_inputs: verbatim except the declared optional argument relpaths (see the header of data2_common)
    d_expected = ld0["discover_inputs"]
    d_expected = d_expected.replace('input_root: Path = Path("/kaggle/input")) -> dict:',
                                    'input_root: Path = Path("/kaggle/input"), relpaths: dict = None) -> dict:')
    d_expected = d_expected.replace('    relpaths = cfg["discovery"]["relpaths"]\n',
                                    '    relpaths = relpaths if relpaths is not None else cfg["discovery"]["relpaths"]\n')
    assert common["discover_inputs"] == d_expected, "discover_inputs"
    assert "4fae48f8616a6177819bfb2a6a636d140117681c536407216d92076a4da05913" in common_src[:900]


# ---- (7) size of the code files

def test_kit_files_are_under_300_kb():
    for rel in ("scripts/data2_generate.py", "scripts/data2_common.py", "config/data2_pages.json",
                "tests/test_data2_generate.py", "scripts/data1_fragments.py", "tests/test_data1_fragments.py"):
        assert (ROOT / rel).stat().st_size < 300 * 1024, rel


# ---- notebooks (text checks; the Kaggle cells are compiled, not run here)

def test_three_notebook_cells_compile_and_the_smoke_has_one_pipeline():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_data2 as B  # noqa: E402
    nbs = B.build_all()
    assert set(nbs) == {"kaggle_data2_generation.ipynb", "kaggle_data2_fragments.ipynb", "kaggle_data2_smoke.ipynb"}
    for name, nb in nbs.items():
        for cell in nb["cells"][1:]:
            compile("".join(cell["source"]), name, "exec")
    smoke = "".join(nbs["kaggle_data2_smoke.ipynb"]["cells"][2]["source"])
    assert smoke.count("C.execute_pipeline(") == 1
    assert "STEPS = STEPS_GEN + STEPS_F" in smoke and "STEPS_F = [" in smoke and "STEPS_GEN = [" in smoke


def test_install_cell_lists_the_generator_packages_with_timeout_and_no_quiet():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_data2 as B  # noqa: E402
    cell = B.install_cell(B.GENERATOR_REQUIRED, B.GENERATOR_MODULES)
    assert '"Pillow>=12.3,<13"' in cell and "'--timeout', '30', '--retries', '1'" in cell
    assert "timeout=900" in cell and "'-q'" not in cell
    assert "faulthandler.dump_traceback_later(600, exit=True" in cell and "faulthandler.cancel_dump_traceback_later()" in cell


# ---- §11 (3) of the lot: discovery spies, both mounting formats, the three carnets and their white lists

FORBIDDEN_LISTING = ("e2e3/mf/pages", "e2e3/mf/images")


def _tree_for_format(base: Path, fmt: str) -> Path:
    """Every file the three white lists need, once each, plus decoys. fmt 'old': /kaggle/input/<slug>;
    fmt 'new': /kaggle/input/notebooks/mabaku/<slug> and /kaggle/input/datasets/mabaku/<slug>."""
    if fmt == "old":
        roots = {k: base / f"ds_{k}" for k in ("bundle", "kit", "ref", "run3", "calib", "gen")}
    else:
        roots = {"bundle": base / "datasets" / "mabaku" / "babaorum-stage-s-bundle-v5",
                 "kit": base / "datasets" / "mabaku" / "babaorum-data2-kit",
                 "ref": base / "notebooks" / "mabaku" / "babaorum-e2e4-v3",
                 "run3": base / "notebooks" / "mabaku" / "babaorum-e2e3-run3",
                 "calib": base / "notebooks" / "mabaku" / "babaorum-e2e1-calibration",
                 "gen": base / "notebooks" / "mabaku" / "babaorum-data2-generation"}
    for k in roots.values():
        k.mkdir(parents=True, exist_ok=True)
    (roots["bundle"] / "stage_s_bundle_v5.zip.bin").write_bytes(b"zip")
    (roots["kit"] / "scripts").mkdir(exist_ok=True)
    (roots["kit"] / "scripts" / "data2_generate.py").write_text("#")
    (roots["kit"] / "scripts" / "data1_fragments.py").write_text("#")
    (roots["ref"] / "e2e4_extraction_report.json").write_text("{}")
    mf = roots["run3"] / "e2e3" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "images").mkdir(parents=True)
    for pid in ("mf_0000", "mf_0050"):                       # decoy mf_0050 under e2e3/mf/pages and images
        (mf / "pages" / f"{pid}.json").write_text("{}")
        (mf / "images" / f"{pid}.png").write_bytes(b"png")
    (roots["calib"] / "calibration" / "B3").mkdir(parents=True)
    (roots["calib"] / "calibration" / "S2").mkdir(parents=True)
    (roots["calib"] / "calibration" / "B3" / "checkpoint_last.pt").write_bytes(b"c")
    (roots["calib"] / "calibration" / "S2" / "checkpoint_last.pt").write_bytes(b"c")
    (roots["gen"] / "gen_out" / "mfn").mkdir(parents=True, exist_ok=True)     # B2: the outputs live under gen_out/
    (roots["gen"] / "gen_out" / "data2_generation_manifest.json").write_text("{}")
    (roots["gen"] / "gen_out" / "data2_overlap_report.json").write_text("{}")
    # level-3 decoy named like a reserved page, in a mounted root
    decoy = base / ("notebooks/mabaku/mf_0050" if fmt == "new" else "mf_0050")
    decoy.mkdir(parents=True, exist_ok=True)
    (decoy / "mf_0050.json").write_text("{}")
    return base


def _spies(monkeypatch):
    import glob as globmod
    seen_listed, seen_read = [], []

    def rec(store, name, fn):
        def wrapper(*args, **kwargs):
            store.append((name, str(args[0]) if args else ""))
            return fn(*args, **kwargs)
        return wrapper
    monkeypatch.setattr(os, "scandir", rec(seen_listed, "os.scandir", os.scandir))
    monkeypatch.setattr(os, "listdir", rec(seen_listed, "os.listdir", os.listdir))
    monkeypatch.setattr(os, "walk", rec(seen_listed, "os.walk", os.walk))
    monkeypatch.setattr(globmod, "glob", rec(seen_listed, "glob.glob", globmod.glob))
    monkeypatch.setattr(Path, "iterdir", rec(seen_listed, "Path.iterdir", Path.iterdir))
    monkeypatch.setattr(Path, "glob", rec(seen_listed, "Path.glob", Path.glob))
    monkeypatch.setattr(Path, "rglob", rec(seen_listed, "Path.rglob", Path.rglob))
    monkeypatch.setattr(Path, "open", rec(seen_read, "Path.open", Path.open))
    return seen_listed, seen_read


def _check_no_reserved_listing(listed):
    for name, arg in listed:
        assert not any(bad in arg for bad in FORBIDDEN_LISTING), (name, arg)
        p = Path(arg)
        assert not (p / "e2e3" / "mf").is_dir(), ("listed a folder that is or contains e2e3/mf", name, arg)
        assert not p.name.startswith("mf"), ("listed a page-like folder", name, arg)


def _carnet_relpaths():
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    d = cfg["discovery"]
    return {"generation": d["relpaths"], "fragments": d["relpaths_fragments"],
            "smoke_fragments": d["relpaths_smoke_fragments"]}


@pytest.mark.parametrize("fmt", ["old", "new"])
@pytest.mark.parametrize("carnet", ["generation", "fragments", "smoke_fragments"])
def test_discovery_spies_for_each_carnet_and_format_never_list_or_read_reserved_pages(tmp_path, monkeypatch, fmt, carnet):
    import data2_common as C2
    base = _tree_for_format(tmp_path / fmt, fmt)
    listed, opened = _spies(monkeypatch)
    rel = _carnet_relpaths()[carnet]
    found = C2.discover_inputs(base, relpaths=rel)
    for key, path in found.items():
        if key == "roots_used":
            continue
        assert "mf_0050" not in str(path), (key, path)
    _check_no_reserved_listing(listed)
    assert not any("mf_005" in arg for _, arg in listed)
    assert not any(any(bad in arg for bad in FORBIDDEN_LISTING) for _, arg in opened)
    assert not any("mf_005" in arg for _, arg in opened)
    assert found["roots_used"]


def test_the_three_white_lists_name_no_reserved_page_path():
    for carnet, rel in _carnet_relpaths().items():
        for key, value in rel.items():
            assert not any(bad in value for bad in FORBIDDEN_LISTING), (carnet, key, value)


def test_the_three_notebook_cells_name_no_reserved_page_path_and_no_recursive_search():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_data2 as B  # noqa: E402
    for name, nb in B.build_all().items():
        cell = "".join(nb["cells"][2]["source"])
        assert not any(bad in cell for bad in FORBIDDEN_LISTING), name
        for pattern in ("**", "rglob", "os.walk", "iterdir", "listdir", "scandir", ".glob(", "glob.glob("):
            assert pattern not in cell, (name, pattern)


def test_overlap_check_builds_only_the_mf_0000_to_0049_paths_and_nothing_else_under_e2e3_mf(tmp_path, monkeypatch):
    out, mf_dir, ref, cfg = _make_overlap_fixture(tmp_path, ["abc"], [["x"] * 10 for _ in range(20)])
    # decoys: mf_0050 JSON and PNG exist next to the real pages
    (mf_dir / "pages" / "mf_0050.json").write_text(json.dumps({"lines": [{"text": "decoy"}]}))
    Image.fromarray(np.full((8, 8), 99, dtype=np.uint8), mode="L").save(mf_dir / "images" / "mf_0050.png")
    listed, opened = _spies(monkeypatch)
    rep = G.overlap_check(out, mf_dir, ref, cfg)
    assert rep["mf_pages_read"] == [f"mf_{i:04d}" for i in range(50)]
    assert not any("mf_005" in arg for _, arg in opened)
    assert not any(any(bad in arg for bad in FORBIDDEN_LISTING) and "mf_" in arg and "mf_0050" in arg
                   for _, arg in opened)
    read_under_mf = [arg for _, arg in opened if "e2e3/mf" in arg]
    assert all(Path(a).name.startswith("mf_00") and int(Path(a).stem[3:]) <= 49 for a in read_under_mf), read_under_mf
    _check_no_reserved_listing(listed)


# ---- B1 (relecture of gel 2): execute_pipeline really runs, in process and in a subprocess step

def test_execute_pipeline_really_runs_an_in_process_and_a_subprocess_step(tmp_path):
    import data2_common as C3
    marks = []
    steps = [{"name": "inproc", "fn": lambda: marks.append("ran")},
             {"name": "child", "argv": [sys.executable, "-c", "print('child-ok')"], "timeout_s": 60}]
    rec = C3.execute_pipeline(tmp_path / "out", steps, heartbeat_interval_s=60)
    assert rec["status"] == "DONE" and marks == ["ran"]
    assert "child-ok" in (tmp_path / "out" / "logs" / "child.stdout").read_text()
    exit_rec = json.loads((tmp_path / "out" / "data2_run_exit.json").read_text())
    assert exit_rec["status"] == "DONE" and set(exit_rec["durations_s"]) == {"inproc", "child"}


def test_fragments_carnet_refuses_unless_the_overlap_report_is_ok_and_prefixes_the_smoke_steps():
    import build_kaggle_data2 as B  # noqa: E402  (scripts/stage_s_notebooks is on the path from the tests above)
    nbs = B.build_all()
    frag = "".join(nbs["kaggle_data2_fragments.ipynb"]["cells"][2]["source"])
    assert 'st != "OK"' in frag and "OVERLAP_TEXT_FLAGGED" in frag and "BLOCKED_OVERLAP" in frag
    assert '"gen_out/data2_generation_manifest.json"' in json.dumps(json.loads((ROOT / "config" / "data2_pages.json").read_text())["discovery"]["relpaths_fragments"])
    smoke = "".join(nbs["kaggle_data2_smoke.ipynb"]["cells"][2]["source"])
    assert '{"name": "gen_controls"' in smoke and '{"name": "frag_controls"' in smoke
    assert "durations_s" in "".join(nbs["kaggle_data2_generation.ipynb"]["cells"][2]["source"])


def test_sha_of_the_e2e4_report_and_alphabet_size_are_written_in_the_cells():
    import build_kaggle_data2 as B  # noqa: E402
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    for name, nb in B.build_all().items():
        cell = "".join(nb["cells"][2]["source"])
        assert cfg["page_reference"]["sha256_full"] in cell, name
        assert "alphabet_size\"] == 74" in cell, name


# ---- amendement 5 (§13 (2)): no name read in a notebook cell may be undefined (static, by ast; nothing is run)

NB_NAMES = ["kaggle_data2_generation.ipynb", "kaggle_data2_fragments.ipynb", "kaggle_data2_smoke.ipynb"]


def _cell_sources(name):
    nb = json.loads((ROOT / "scripts" / "stage_s_notebooks" / name).read_text())
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]


def _bound_names(tree):
    """Every name bound anywhere in the cell: assignments, defs, imports, loop/with/except targets, arguments."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _read_names(tree):
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def test_no_notebook_cell_reads_an_undefined_name():
    """§13 (2): in each carnet, cells in order; a name read in a cell must be bound in that cell or an earlier one,
    or be a builtin. Gel 3 fails here: BUNDLE_V5_SHA is read in d_discover and bound nowhere in the notebook."""
    import builtins
    problems = {}
    for name in NB_NAMES:
        bound = {"__file__", "__name__"}
        for i, src in enumerate(_cell_sources(name)):
            tree = ast.parse(src)
            bound |= _bound_names(tree)
            missing = sorted(n for n in _read_names(tree) if n not in bound and not hasattr(builtins, n))
            if missing:
                problems.setdefault(name, []).append((i, missing))
    assert not problems, f"names read but never bound in the notebook: {problems}"


# ---- amendement 6 (§14): smoke-only flag, real acceptance key, text whitelist of the fragments

import data2_text_guard as TG  # noqa: E402

GUARD_NAME = "overlap_text_flag_accepted_by_pA"


def test_the_smoke_flag_is_absent_from_the_real_config_and_the_real_cells():
    """§14 (1): the smoke flag lives in config/data2_smoke.json only; it must not appear in the real DATA-2 config or in
    the generation and fragments cells."""
    assert GUARD_NAME not in (ROOT / "config" / "data2_pages.json").read_text()
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    nbs = {n: json.loads((ROOT / "scripts" / "stage_s_notebooks" / n).read_text())
           for n in ("kaggle_data2_generation.ipynb", "kaggle_data2_fragments.ipynb")}
    for n, nb in nbs.items():
        assert GUARD_NAME not in json.dumps(nb), n
    assert GUARD_NAME in (ROOT / "config" / "data2_smoke.json").read_text()


def test_the_real_acceptance_lifts_only_text_flagged_and_carries_the_mention():
    """§14 (2): the key overlap_text_accepted_real lifts OVERLAP_TEXT_FLAGGED only; the mention is written."""
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    real = cfg["overlap_text_accepted_real"]
    assert real["accepted"] is True and real["by"] == "pA" and real["lifts_only"] == "OVERLAP_TEXT_FLAGGED"
    assert real["never_lifts"] == "OVERLAP_PAGE"
    assert real["mention"].startswith("Les textes des lignes mfn sont presque tous identiques")
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_data2 as B  # noqa: E402
    frag = "".join(json.loads((ROOT / "scripts" / "stage_s_notebooks" / "kaggle_data2_fragments.ipynb")
                              .read_text())["cells"][1 + 1]["source"])
    assert "OVERLAP_TEXT_FLAGGED" in frag and "OVERLAP_PAGE" in frag and "BLOCKED_OVERLAP" in frag
    assert real["mention"] in frag and "OVERLAP_ACCEPT = {" in frag


def _payload():
    return {"page_id": "mfn_0000", "bras": "B3", "échelle": 0.64, "factor": 0.64, "native_wh": [5891, 8610],
            "scaled_wh": [3770, 5510], "stride": 4, "seuil": 0.5,
            "before": [{"id": "B3:mfn_0000:b0", "polyline_px": [[1.0, 2.0], [3.0, 4.0]],
                        "refined_baseline_px": [[1.0, 2.0]], "score": 0.9, "n_cells": 12}]}


def test_text_whitelist_accepts_a_geometry_only_payload_and_manifest():
    assert TG.check_payload(_payload()) == []
    assert TG.check_manifest({"schema_version": "data1_manifest_v1", "status": "OK", "smoke": True,
                              "kit_sha256": "0" * 64, "arms": ["B3", "S2"]}) == []


def test_text_whitelist_rejects_a_text_key_and_a_string_value():
    bad = _payload()
    bad["text"] = "un mot"
    assert any("'text'" in p for p in TG.check_payload(bad))
    bad2 = _payload()
    bad2["before"][0]["id"] = "abc"
    assert any("hors liste blanche" in p for p in TG.check_payload(bad2))
    bad3 = _payload()
    bad3["before"][0]["n_cells"] = "12 caractères"
    assert any("n_cells" in p for p in TG.check_payload(bad3))


def test_text_whitelist_rejects_the_alphabet_keys_of_the_manifest():
    """§14 (3): the alphabet is text-derived and must never leave. This is what currently blocks the DATA-1 manifest."""
    probs = TG.check_manifest({"schema_version": "data1_manifest_v1", "alphabet": "placeholder of 74"})
    assert any("alphabet" in p for p in probs)


def test_fragments_of_the_smoke_page_pass_the_whitelist_when_they_exist():
    """Runs only where the smoke output is given (DATA2_FRAG_DIR); the Kaggle carnet runs the same check (BLOCKED_TEXT_FIELD)."""
    d = os.environ.get("DATA2_FRAG_DIR")
    if not d:
        pytest.skip("no fragment output given (DATA2_FRAG_DIR)")
    root = Path(d)
    manifest = json.loads((root / "data1_manifest.json").read_text())
    payloads = [json.loads(p.read_text()) for p in sorted(root.rglob("*.json"))
                if p.name != "data1_manifest.json" and "page_id" in json.loads(p.read_text())]
    assert payloads, "no payload"
    assert TG.check_all(payloads, manifest) == []


# ---- §15 (p6, 24/10/2026): sanitisation of the DATA-1 manifest before pR; text_guard after sanitize

def _data1_like_manifest():
    return {"schema_version": "data1_manifest_v1", "status": "OK", "smoke": True, "shard": None,
            "pages_fragments": ["mfn_0000"], "arms": ["B3", "S2"], "alphabet": "<blank> c0 c1 … c72",
            "alphabet_size": 74, "kit_sha256": {"scripts/data1_fragments.py": "0" * 64},
            "counts": {"B3": {"before_total": 3}}, "versions": {"python": "3.12.1"},
            "config_sha256": {"data1": "1" * 64, "e2e4": "2" * 64},
            "outputs_sha256": {"fragments/B3/mfn_0000.json": "3" * 64}}


def test_sanitize_removes_exactly_the_two_keys_and_keeps_the_rest_byte_for_byte():
    man = _data1_like_manifest()
    clean, removed = TG.sanitize_manifest(man, "a" * 64)
    assert removed == ["alphabet", "alphabet_size"]
    assert "alphabet" not in clean and "alphabet_size" not in clean
    rest = {k: v for k, v in man.items() if k not in ("alphabet", "alphabet_size")}
    assert json.dumps({k: clean[k] for k in rest}, sort_keys=True) == json.dumps(rest, sort_keys=True)
    assert clean["removed_keys"] == ["alphabet", "alphabet_size"] and clean["original_manifest_sha256"] == "a" * 64


def test_text_guard_fails_before_sanitize_and_passes_after(tmp_path):
    man = _data1_like_manifest()
    assert TG.check_manifest(man) != []                       # before: alphabet keys refused
    p = tmp_path / "data1_manifest.json"
    p.write_text(json.dumps(man))
    TG.sanitize_file(p)
    assert TG.check_manifest(json.loads(p.read_text())) == []  # after: whitelist passes, nothing exempted


def test_the_original_manifest_is_absent_from_the_final_output(tmp_path):
    p = tmp_path / "data1_manifest.json"
    p.write_text(json.dumps(_data1_like_manifest()))
    original_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    info = TG.sanitize_file(p)
    assert info["original_sha256"] == original_sha
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert hashlib.sha256(f.read_bytes()).hexdigest() != original_sha
            assert "alphabet" not in json.loads(f.read_text()) if f.suffix == ".json" else True


def test_the_notebook_runs_sanitize_before_text_guard_and_consigns_the_original_sha():
    nb = json.loads((ROOT / "scripts" / "stage_s_notebooks" / "kaggle_data2_fragments.ipynb").read_text())
    src = "".join(nb["cells"][2]["source"])
    assert src.index('"name": "sanitize"') < src.index('"name": "text_guard"') < len(src)
    assert src.index('"name": "fragments"') < src.index('"name": "sanitize"')
    assert "original_manifest_sha256" in src and "sanitize_file" in src


# ---- §16 (amendement 7): total whitelist, real DATA-1 manifest (B1), exact LISEZMOI, closed output list (B2 in kit)

FIX = ROOT / "tests" / "fixtures" / "data1_shard0"


def _real_manifest():
    return json.loads((FIX / "data1_manifest.json").read_text())


def test_real_shard0_manifest_passes_sanitize_then_text_guard(tmp_path):
    """§16 (1) B1 : le VRAI manifeste DATA-1 (fixture), après sanitize, passe la liste blanche totale."""
    p = tmp_path / "data1_manifest.json"
    p.write_text((FIX / "data1_manifest.json").read_text())
    assert TG.check_manifest(json.loads(p.read_text())) != []      # avant : alphabet refusé
    TG.sanitize_file(p)
    assert TG.check_manifest(json.loads(p.read_text())) == []      # après : passe
    assert "config_sha256" in _real_manifest() and "outputs_sha256" in _real_manifest()


def test_real_shard0_readme_matches_the_template_except_the_run_date():
    """§16 (2) : LISEZMOI.txt exact, reconstruit à partir du gabarit (lu sans importer torch) et du manifeste."""
    template = TG.template_from_source((ROOT / "scripts" / "data1_fragments.py").read_text())
    cfg = json.loads((ROOT / "config" / "e2e4_extraction.json").read_text())
    text = (FIX / "LISEZMOI.txt").read_text(encoding="utf-8")
    assert TG.readme_matches(text, _real_manifest(), template, cfg)
    assert not TG.readme_matches(text.replace("Statut : OK", "Statut : KO"), _real_manifest(), template, cfg)


def test_the_whitelist_is_recursive_and_refuses_a_text_key_at_depth():
    man = _data1_like_manifest()
    man["counts"]["B3"]["label_text"] = 5
    assert any("label_text" in p for p in TG.check_manifest(man))


def test_the_whitelist_refuses_a_free_string_value_at_depth():
    man = _data1_like_manifest()
    man["inputs"] = {"e2e4_report_sha256": "une phrase libre"}
    assert any("phrase libre" in p for p in TG.check_manifest(man))


def test_reason_is_admitted_only_when_the_status_is_not_done_or_complete():
    base, _removed = TG.sanitize_manifest(_data1_like_manifest(), "a" * 64)
    ok_status = dict(base, status="OK", reason="échec partiel")
    assert TG.check_manifest(ok_status) == []
    done = dict(base, status="DONE", reason="échec partiel")
    assert any("reason interdit" in p for p in TG.check_manifest(done))


def test_lisezmon_data2_must_be_exactly_the_mention(tmp_path):
    cfg = json.loads((ROOT / "config" / "data2_pages.json").read_text())
    mention = cfg["overlap_text_accepted_real"]["mention"]
    assert TG.exact_text_matches(mention + "\n", mention + "\n")
    assert not TG.exact_text_matches(mention + "\nun mot en trop", mention + "\n")


def test_output_of_the_fragments_carnet_contains_only_the_listed_files(tmp_path):
    man = _real_manifest()
    listed = TG.expected_output(man, overlap_text_accepted=True)
    for rel in listed:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    assert TG.unexpected_output(tmp_path, listed) == []
    (tmp_path / "stray.log").write_text("x")
    assert TG.unexpected_output(tmp_path, listed) == ["stray.log"]


def test_the_kit_carries_the_smoke_config_the_notebooks_and_the_fixtures():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_data2 as B  # noqa: E402
    for rel in ("config/data2_smoke.json", "scripts/stage_s_notebooks/kaggle_data2_fragments.ipynb",
                "scripts/stage_s_notebooks/kaggle_data2_generation.ipynb",
                "scripts/stage_s_notebooks/kaggle_data2_smoke.ipynb",
                "tests/fixtures/data1_shard0/data1_manifest.json", "tests/fixtures/data1_shard0/LISEZMOI.txt"):
        assert rel in B.FRAG_KIT, rel


# ---- §17 (p6, 24/10/2026): times_s.alphabet_models is renamed times_s.models_load in sanitize; no name exception

def test_times_alphabet_models_is_renamed_and_recorded_in_renamed_keys():
    raw = _real_manifest()
    assert raw["times_s"]["alphabet_models"] >= 0
    clean, _removed = TG.sanitize_manifest(raw, "a" * 64)
    assert "alphabet_models" not in clean["times_s"]
    assert clean["times_s"]["models_load"] == raw["times_s"]["alphabet_models"]
    assert clean["renamed_keys"] == [["times_s.alphabet_models", "times_s.models_load"]]
    assert TG.check_manifest(clean) == []


def test_raw_alphabet_models_key_is_refused_and_other_renames_are_refused():
    raw = _real_manifest()
    assert any("alphabet_models" in p for p in TG.check_manifest(raw))
    bad = dict(TG.sanitize_manifest(raw, "a" * 64)[0], renamed_keys=[["times_s.x", "times_s.y"]])
    assert any("renamed_keys" in p for p in TG.check_manifest(bad))


# ---- §20 (amendement 9) : fumée v3 — generation_manifest_sha256 et SMOKE acceptés par text_guard

SMOKE_V3 = ROOT / "tests" / "fixtures" / "data2_smoke_v3" / "data1_manifest.json"


def test_text_guard_passes_on_the_real_smoke_v3_manifest():
    man = json.loads(SMOKE_V3.read_text())
    assert man["status"] == "SMOKE"
    assert "generation_manifest_sha256" in man["inputs"]
    assert TG.check_manifest(man) == []


def test_the_whitelist_refuses_an_unknown_status_and_a_non_sha_generation_value():
    base = _data1_like_manifest()
    assert any("statut" in p for p in TG.check_manifest(dict(base, status="PRELIM")))
    bad = dict(base, inputs={"generation_manifest_sha256": "pas-un-sha"})
    assert any("generation_manifest_sha256" in p for p in TG.check_manifest(bad))
