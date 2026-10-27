"""LD-3 + DATA-3 : tests légers (docs/END_TO_END_LD3_PROPOSITION.txt §2-§8). Fonctions pures et petits tableaux synthétiques :
aucun modèle chargé, aucun entraînement, aucun détecteur, aucun générateur, aucune page réelle. Les tests de carnets sont ignorés
si le constructeur est absent du kit Kaggle."""
import ast
import builtins
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shlex
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import data3_generate as D3  # noqa: E402
import ld1b_border_diag as LB  # noqa: E402
import ld2_core as C2  # noqa: E402

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
import eval_ld3 as E3  # noqa: E402
import train_ld2 as TD  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402

CARNETS = ROOT / "scripts" / "stage_s_notebooks"
LD3 = CARNETS / "ld3"
DOC = ROOT / "docs" / "END_TO_END_LD3_PROPOSITION.txt"
CFG = E3.load_cfg()
NEEDS_CARNET = pytest.mark.skipif(not (CARNETS / "build_kaggle_ld3.py").is_file() or not (LD3 / "kaggle_ld3.ipynb").is_file(),
                                  reason="build_kaggle_ld3.py / ld3/*.ipynb absents du kit (Kaggle)")
NEEDS_DOC = pytest.mark.skipif(not DOC.is_file(), reason="docs/ absent du kit (Kaggle)")


def _norm(t):
    return re.sub(r"\s+", " ", t)


def _b():
    import build_kaggle_ld3 as B3
    return B3


# ---------- DATA-3 : noms, graines, drapeau, dossier de sortie ----------

def test_real_names_are_exactly_mf_0060_to_mf_0069_and_the_seed_is_the_pre_registered_one():
    assert D3.real_names(CFG) == [f"mf_00{i}" for i in range(60, 70)] and CFG["pages"]["seed"] == 20261026
    assert D3.decoy_names(CFG) == ["decoy_0000"] and D3.decoy_names(CFG, 3) == ["decoy_0000", "decoy_0001", "decoy_0002"]
    assert D3.check_names(D3.real_names(CFG), "real", CFG) and D3.check_names(["decoy_0007"], "decoy", CFG)


@pytest.mark.parametrize("bad", ["mf_0000", "mf_0049", "mf_0050", "mf_0059", "mfn_0000", "mfn_0019"])
def test_forbidden_output_names_are_refused_in_both_modes(bad):
    for mode, base in (("real", D3.real_names(CFG)), ("decoy", ["decoy_0000"])):
        with pytest.raises(SystemExit, match="interdits"):
            D3.check_names(base[:-1] + [bad], mode, CFG)


def test_real_mode_needs_exactly_the_ten_names_and_decoy_mode_only_decoy_names():
    with pytest.raises(SystemExit, match="exactement"):
        D3.check_names(D3.real_names(CFG)[:9], "real", CFG)
    with pytest.raises(SystemExit, match="exactement"):
        D3.check_names(["mf_0070"] + D3.real_names(CFG)[1:], "real", CFG)
    with pytest.raises(SystemExit, match="leurre"):
        D3.check_names(D3.real_names(CFG), "decoy", CFG)                       # noms réels avec le mode leurre
    with pytest.raises(SystemExit, match="leurre"):
        D3.check_names(["page_0000"], "decoy", CFG)
    with pytest.raises(SystemExit, match="inconnu"):
        D3.check_names(["decoy_0000"], "autre", CFG)


def test_the_real_seed_is_refused_with_decoy_names_and_any_other_seed_with_real_names():
    assert D3.check_seed(20261026, "real", CFG) == 20261026 and D3.check_seed(1, "decoy", CFG) == 1
    with pytest.raises(SystemExit, match="graine réelle"):
        D3.check_seed(20261026, "decoy", CFG)
    with pytest.raises(SystemExit, match="exige la graine"):
        D3.check_seed(1, "real", CFG)


def test_real_mode_is_refused_unless_the_eval_flag_is_explicit_and_the_generator_never_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(D3, "generate", lambda *a, **k: pytest.fail("le générateur ne doit pas démarrer"))
    for flag in (False, None, 1, "oui"):
        with pytest.raises(SystemExit, match="phase EVAL de LD-3"):
            D3.produce_pages("real", CFG, tmp_path / "src", tmp_path / "w", tmp_path / "p", called_from_eval_ld3=flag)
    with pytest.raises(SystemExit, match="jamais lancé en ligne de commande"):               # F4 : mode réel refusé entièrement
        D3.main(["--mode", "real", "--src", str(tmp_path), "--work", str(tmp_path / "w"), "--pages-root", str(tmp_path / "p")])
    with pytest.raises(SystemExit):                                                          # l'option --from-eval-ld3 n'existe plus
        D3.main(["--mode", "real", "--from-eval-ld3", "--src", str(tmp_path), "--work", str(tmp_path / "w"),
                 "--pages-root", str(tmp_path / "p")])
    assert 'add_argument("--from-eval-ld3"' not in (ROOT / "scripts" / "data3_generate.py").read_text()
    with pytest.raises(SystemExit, match="graine"):                                        # le drapeau passe, la garde de graine reste
        D3.produce_pages("real", CFG, tmp_path, tmp_path / "w", tmp_path / "p", called_from_eval_ld3=True, seed=5)
    with pytest.raises(SystemExit, match="graine réelle"):
        D3.produce_pages("decoy", CFG, tmp_path, tmp_path / "w", tmp_path / "p", seed=20261026)


def test_a_non_empty_output_folder_is_refused_before_any_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(D3, "generate", lambda *a, **k: pytest.fail("le générateur ne doit pas démarrer"))
    out = tmp_path / "p"
    out.mkdir()
    (out / "x").write_text("1")
    with pytest.raises(SystemExit, match="non vide"):
        D3.produce_pages("decoy", CFG, tmp_path, tmp_path / "w", out, seed=1)
    D3.check_out_empty(tmp_path / "absent")                                                # absent ou vide : accepté


def test_the_generator_block_equals_the_data2_one_except_seed_pages_and_names():
    d2 = json.loads((ROOT / "config" / "data2_pages.json").read_text())["generator"]
    g = CFG["generator"]
    assert g["commit"] == d2["commit"] and g["profile"] == d2["pilot_profile"] and g["render"] == d2["render"]
    assert g["pilot_config_sha256"] == json.loads((ROOT / "config" / "data2_pages.json").read_text())["pilot_config_sha256"]["sha256"]
    BD2 = pytest.importorskip("build_kaggle_data2")                                       # absent du kit Kaggle
    assert g["bundle_sha256"] in d2["repo_in_bundle"] and g["required_pip"] == BD2.GENERATOR_REQUIRED
    assert CFG["pages"]["seed"] != d2["seed"] and CFG["overlap"]["text_share_threshold"] == 0.05


def _write_raw(raw: Path, n: int):
    (raw / "pages").mkdir(parents=True)
    (raw / "images").mkdir()
    for i in range(n):
        name = f"mf_{i:04d}"
        Image.fromarray(np.full((40, 60), 255 - i, dtype=np.uint8)).save(raw / "images" / f"{name}.png")
        sha = hashlib.sha256((raw / "images" / f"{name}.png").read_bytes()).hexdigest()
        (raw / "pages" / f"{name}.json").write_text(json.dumps({
            "page_id": name, "image": {"sha256": sha, "height": 40, "width": 60, "path": f"images/{name}.png"},
            "lines": [{"id": f"{name}_l0", "text": f"texte {i}", "baseline": [[1, 2], [3, 4]], "block_id": "b", "polygon": [],
                       "legibility": "readable"}], "blocks": []}))


def _mf_dir(tmp_path, texts=("autre",)):
    mf = tmp_path / "mf"
    (mf / "pages").mkdir(parents=True)
    for p in CFG["overlap"]["mf_pages"]:
        (mf / "pages" / f"{p}.json").write_text(json.dumps({"lines": [{"text": t} for t in texts]}))
    return mf


def test_produce_pages_in_decoy_mode_renames_and_writes_the_layout_the_loaders_read(tmp_path, monkeypatch):
    calls = []

    def fake_generate(src, raw, cfg, pages, seed, extra_libs=None, timeout_s=None):
        calls.append((pages, seed, extra_libs))
        _write_raw(raw, pages)
        return 0

    monkeypatch.setattr(D3, "generate", fake_generate)
    out = tmp_path / "pages_root"
    m = D3.produce_pages("decoy", CFG, tmp_path / "src", tmp_path / "work", out, seed=7, n_pages=2, mf_dir=_mf_dir(tmp_path),
                         extra_libs=tmp_path / "libs")
    assert calls == [(2, 7, tmp_path / "libs")] and m["names"] == ["decoy_0000", "decoy_0001"] and m["seed"] == 7
    assert sorted(p.name for p in (out / "pages").iterdir()) == ["decoy_0000.json", "decoy_0001.json"]
    assert sorted(p.name for p in (out / "images").iterdir()) == ["decoy_0000.png", "decoy_0001.png"]
    page = json.loads((out / "pages" / "decoy_0001.json").read_text())
    assert page["page_id"] == "decoy_0001" and page["image"]["path"] == "images/decoy_0001.png"
    for rel, sha in m["files_sha256"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == sha
    assert hashlib.sha256((out / "images" / "decoy_0001.png").read_bytes()).hexdigest() == page["image"]["sha256"]
    man = json.loads((tmp_path / "work" / "data3_generation_manifest.json").read_text())
    assert man["generator"]["commit"] == CFG["generator"]["commit"] and man["generator"]["render"] == CFG["generator"]["render"]
    assert man["generator"]["bundle_sha256"] == CFG["generator"]["bundle_sha256"] and man["mode"] == "decoy"
    assert (tmp_path / "work" / "data3_overlap_report.json").is_file()
    assert man["overlap"]["status"] == "OK" and man["overlap"]["blocking"] is False
    with pytest.raises(SystemExit, match="non vide"):                                       # une seconde production dans le même dossier
        D3.produce_pages("decoy", CFG, tmp_path, tmp_path / "w2", out, seed=7)


def test_produce_pages_in_real_mode_with_the_flag_produces_exactly_the_ten_names(tmp_path, monkeypatch):
    monkeypatch.setattr(D3, "generate", lambda src, raw, cfg, pages, seed, extra_libs=None, timeout_s=None: (
        _write_raw(raw, pages), 0 if (pages, seed) == (10, 20261026) else 9)[1])
    m = D3.produce_pages("real", CFG, tmp_path, tmp_path / "w", tmp_path / "p", called_from_eval_ld3=True, mf_dir=_mf_dir(tmp_path))
    assert m["names"] == [f"mf_00{i}" for i in range(60, 70)] and m["seed"] == 20261026 and len(m["files_sha256"]) == 20


def test_the_overlap_report_flags_text_but_does_not_block_and_blocks_an_identical_image(tmp_path):
    root = tmp_path / "p"
    _write_raw(root, 2)
    # deux pages (fichiers du dossier synthétique) de textes « texte 0 » / « texte 1 » ; réservoir qui contient les deux -> texte signalé
    names = ["mf_0000", "mf_0001"]
    rep = D3.overlap_check(root, names, _mf_dir(tmp_path, ("texte 0", "texte 1")), None, CFG)
    assert rep["status"] == "OVERLAP_TEXT_FLAGGED" and rep["blocking"] is False and rep["text_share_global"] == 1.0
    assert D3.overlap_check(root, names, _mf_dir(tmp_path / "x", ("zzz",)), None, CFG)["status"] == "OK"
    sha = hashlib.sha256((root / "images" / "mf_0000.png").read_bytes()).hexdigest()
    ref = {"steps": {"page_cache": {"sha256": {f"images/{p}.png": sha for p in CFG["overlap"]["mf_pages"]}}}}
    rep2 = D3.overlap_check(root, names, _mf_dir(tmp_path / "y", ("zzz",)), ref, CFG)
    assert rep2["status"] == "OVERLAP_PAGE" and rep2["blocking"] is True and rep2["image_sha_matches"] == ["mf_0000"]


# ---------- marge : contexte, identité de M16, M32 ----------

def test_margin_context_sets_and_always_restores_the_module_value_even_on_exception():
    assert LD.MARGIN_H == 0.5
    with E3.margin_h(1.0):
        assert LD.MARGIN_H == 1.0
        with E3.margin_h(0.25):
            assert LD.MARGIN_H == 0.25
        assert LD.MARGIN_H == 1.0
    assert LD.MARGIN_H == 0.5
    with pytest.raises(RuntimeError, match="boum"):
        with E3.margin_h(2.0):
            assert LD.MARGIN_H == 2.0
            raise RuntimeError("boum")
    assert LD.MARGIN_H == 0.5
    with pytest.raises(ValueError):
        with E3.margin_h(1.0):
            raise ValueError("x")
    assert LD.MARGIN_H == 0.5


def test_m16_is_bit_identical_to_an_untouched_zone_canvas_and_m32_widens_the_crop_by_32_px():
    rng = np.random.default_rng(0)
    page = rng.random((900, 2000)).astype(np.float32)
    box = (120.0, 140.0, 700.0, 260.0)
    c0, m0 = LD.zone_canvas(page, box)
    with E3.margin_h(E3.arms(CFG)["M16"]):
        c16, m16 = LD.zone_canvas(page, box)
    assert np.array_equal(c0, c16) and m0 == m16 and c0.tobytes() == c16.tobytes()
    with E3.margin_h(E3.arms(CFG)["M32"]):
        c32, m32 = LD.zone_canvas(page, box)
    assert m32.crop_w - m0.crop_w == 32 and m32.crop_h - m0.crop_h == 32 and m32.ix0 == m0.ix0 - 16 and m32.iy0 == m0.iy0 - 16
    assert (CFG["margins"]["M16"], CFG["margins"]["M32"]) == (0.5, 1.0)


def test_a_big_zone_is_reduced_more_by_m32_and_the_reduction_summary_counts_it():
    page = np.ones((900, 2200), np.float32)
    box = (10.0, 10.0, 1560.0, 400.0)                                                # plus large que le canevas à 16 px de marge
    with E3.margin_h(0.5):
        _c, a = LD.zone_canvas(page, box)
    with E3.margin_h(1.0):
        _c, b = LD.zone_canvas(page, box)
    fa, fb = 0.5 * (a.fx + a.fy), 0.5 * (b.fx + b.fy)
    assert fb < fa < 1.0
    red = E3.zone_reduction([a, b, LD.zone_canvas(page, (10.0, 10.0, 300.0, 60.0))[1]])
    assert red["zones"] == 3 and red["reduced"] == 2 and abs(red["mean_f_reduced"] - (fa + fb) / 2) < 1e-12
    assert E3.zone_reduction([]) == {"zones": 0, "reduced": 0, "reduced_share": None, "mean_f_reduced": None, "min_f": None}


# ---------- scoring : égalité avec train_ld2.score_lines, matches ----------

def _gt(i, x, y=10.0, text="ab", leg="readable"):
    return {"id": f"g{i}", "baseline": np.array([[x, y], [x + 300.0, y]]), "neg": 12.0, "pos": 4.0, "text": text, "legibility": leg}


def _kept(lines, zone=0):
    return [{"pts": np.array(g["baseline"]), "zone": zone, "query": i, "score": 0.9, "h": 16.0} for i, g in enumerate(lines)]


def test_the_new_scoring_helper_equals_train_ld2_score_lines_and_returns_the_matched_gt_ids():
    page = {"gt_lines": [_gt(0, 0.0), _gt(1, 400.0), _gt(2, 800.0), _gt(3, 1200.0, text="", leg="illegible")]}
    kept = _kept(page["gt_lines"][:3]) + [{"pts": np.array([[5000.0, 900.0], [5200.0, 900.0]]), "zone": 0, "query": 9, "score": 0.5, "h": 16.0}]
    raws = [(0, None, {"hyp": {0: "ab", 1: "ax", 2: None, 9: "zz"}, "hyp_none": {2: "band_outside_support"}})]
    mine, matched = E3.score_with_matches(page, kept, raws)
    assert mine == TD.score_lines(page, kept, raws) and matched == {"g0", "g1", "g2"}
    assert mine["cer"]["edits"] == 1 and mine["cer"]["skipped_by_cause"]["band_outside_support"] == 1 and mine["n_pred"] == 4
    empty, none = E3.score_with_matches(page, [], [])
    assert empty == TD.score_lines(page, [], []) and none == set() and empty["precision"] == 0.0


def test_the_scoring_helper_calls_evaluate_line_geometry_once(monkeypatch):
    calls = []
    real = E3.evaluate_line_geometry
    monkeypatch.setattr(E3, "evaluate_line_geometry", lambda *a, **k: calls.append(1) or real(*a, **k))
    page = {"gt_lines": [_gt(0, 0.0)]}
    E3.score_with_matches(page, _kept(page["gt_lines"]), [(0, None, {"hyp": {0: "ab"}})])
    assert calls == [1]


# ---------- table de bord : affectation, distance, tranches ----------

def test_the_zone_assignment_follows_ld1b_rules_largest_share_first_index_on_ties_none_is_outside():
    lines = [{"id": "a", "baseline": np.array([[10.0, 10.0], [90.0, 10.0]]), "neg": 5.0, "pos": 5.0},
             {"id": "tie", "baseline": np.array([[0.0, 10.0], [200.0, 10.0]]), "neg": 5.0, "pos": 5.0},
             {"id": "none", "baseline": np.array([[900.0, 900.0], [950.0, 900.0]]), "neg": 5.0, "pos": 5.0}]
    zones = [(7, (0.0, 0.0, 100.0, 50.0)), (3, (100.0, 0.0, 200.0, 50.0)), (4, (0.0, 0.0, 100.0, 50.0))]
    owner = E3.owner_pp(lines, zones)
    assert owner == {"a": 7, "tie": 7}                                                    # 'a' : deux zones égales -> la première de la liste
    page = {"blocks": [{"lines": [{"id": "x"}, {"id": "y"}]}, {"lines": [{"id": "z"}]}]}
    assert E3.owner_gt(page) == {"x": 0, "y": 0, "z": 1}


def test_edge_counts_bins_are_the_ld1b_bins_matched_are_counted_and_outside_lines_are_apart():
    box = (0.0, 0.0, 400.0, 100.0)
    def line(i, y, neg=10.0, pos=10.0):
        return {"id": f"l{i}", "baseline": np.array([[10.0, y], [300.0, y]]), "neg": neg, "pos": pos}
    # distances haut = y - neg - 0, bas = 100 - (y + pos) : minimum des deux
    lines = [line(0, 15.0), line(1, 25.0), line(2, 40.0), line(3, 50.0), line(4, 85.0), line(5, 50.0)]
    owner = {f"l{i}": 0 for i in range(5)}                                               # l5 n'a pas de zone
    res = E3.edge_counts(lines, owner, {0: box}, matched={"l0", "l2", "l3", "l5"})
    b = res["bins"]
    assert (b["0-8"]["gt_n"], b["0-8"]["gt_matched"]) == (2, 1)      # l0 : d = 5 ; l4 : d = 100 - 95 = 5
    assert (b["8-16"]["gt_n"], b["8-16"]["gt_matched"]) == (1, 0)    # l1 : d = 15
    assert (b["16-32"]["gt_n"], b["16-32"]["gt_matched"]) == (1, 1)  # l2 : d = 30
    assert (b[">32"]["gt_n"], b[">32"]["gt_matched"]) == (1, 1)      # l3 : d = 40
    assert res["gt_hors_zone"] == 1 and res["gt_in_zones"] == 5
    assert LB.BIN_NAMES == ["0-8", "8-16", "16-32", ">32"] and LB.bin_of(-3.0) == "0-8" and LB.bin_of(8.0) == "8-16"
    # un résultat par page agrège comme LD-1B (moyenne non pondérée par page, seules les pages avec lignes dans la tranche)
    agg = LB.aggregate([b, b])
    assert agg["0-8"]["recall_mean"] == 0.5 and agg["16-32"]["recall_mean"] == 1.0


# ---------- critère (§5) ----------

def _rows(P, R, ids=None, n_pred=5):
    return [{"page": p, "precision": P[i] if hasattr(P, "__len__") else P, "recall": R[i] if hasattr(R, "__len__") else R,
             "n_pred": n_pred} for i, p in enumerate(ids or [f"mf_00{k}" for k in range(60, 70)])]


IDS = [f"mf_00{k}" for k in range(60, 70)]


def test_principal_is_met_iff_the_low_bound_of_dR_is_positive_and_the_low_bound_of_dP_is_at_least_minus_0_01():
    base = np.full(10, 0.90)
    ok = E3.principal_criterion(_rows(0.95, base), _rows(0.945, base + 0.02), IDS, CFG)
    assert ok["met"] and ok["verdict"] == "PRINCIPAL_REMPLI" and ok["bootstrap"]["dR"]["low"] > 0
    flat = E3.principal_criterion(_rows(0.95, base), _rows(0.95, base), IDS, CFG)                 # ΔR = 0 : borne basse = 0, pas > 0
    assert not flat["met"] and flat["verdict"] == "PRINCIPAL_NON_REMPLI"
    precision_loss = E3.principal_criterion(_rows(0.95, base), _rows(0.93, base + 0.02), IDS, CFG)   # ΔP = -0,02 < -0,01
    assert not precision_loss["met"] and precision_loss["bootstrap"]["dP"]["low"] < -0.01
    edge = E3.principal_criterion(_rows(0.95, base), _rows(0.9402, base + 0.02), IDS, CFG)         # ΔP = -0,0098 : borne >= -0,01
    assert edge["met"]
    assert ok["M16"]["R"] == pytest.approx(0.9) and ok["M32"]["R"] == pytest.approx(0.92)


def test_principal_uses_the_ld2_bootstrap_function_seed_draws_and_percentiles():
    rng = np.random.default_rng(3)
    p16, r16 = rng.uniform(0.9, 1.0, 10), rng.uniform(0.85, 0.95, 10)
    p32, r32 = p16 + rng.normal(0, 0.01, 10), r16 + rng.normal(0.01, 0.01, 10)
    out = E3.principal_criterion(_rows(p16, r16), _rows(p32, r32), IDS, CFG)
    idx = np.random.default_rng(20261018).integers(0, 10, size=(10000, 10))
    lo, hi = np.percentile((r32 - r16)[idx].mean(axis=1), [2.5, 97.5])
    assert (out["bootstrap"]["dR"]["low"], out["bootstrap"]["dR"]["high"]) == (float(lo), float(hi))
    assert out["bootstrap"] == C2.paired_bootstrap(list(p32 - p16), list(r32 - r16), CFG)
    assert CFG["bootstrap"] == {"draws": 10000, "seed": 20261018, "percentiles": [2.5, 97.5]}
    assert (CFG["criterion"]["principal"]["dr_low_gt"], CFG["criterion"]["principal"]["dp_low_min"]) == (0.0, -0.01)


def test_n_pred_zero_pages_count_as_p_r_zero_in_the_principal_deltas():
    r16, r32 = _rows(0.9, 0.9), _rows(0.9, 0.9)
    r32[0].update(n_pred=0, precision=None, recall=None)
    out = E3.principal_criterion(r16, r32, IDS, CFG)
    assert out["M32"]["P"] == pytest.approx(0.81) and out["M32"]["R"] == pytest.approx(0.81) and not out["met"]


def test_secondary_is_the_ld2_criterion_unchanged_m32_against_the_in_run_b3():
    ref = {p: (0.71, 0.923) for p in IDS}
    assert CFG["criterion"]["a"] == {"precision_min": 0.8, "recall_min": 0.9, "pages": 10}
    assert CFG["criterion"]["b"] == {"dp_low_gt": 0.0, "dr_low_min": -0.02}
    ok = E3.secondary_criterion(_rows(0.95, 0.93), ref, IDS, CFG)
    assert ok["verdict"] == "CRITERE_REMPLI" and ok["a"]["met"] and ok["b"]["met"]
    bad = E3.secondary_criterion(_rows(0.95, 0.89), ref, IDS, CFG)                         # R < 0,90 et ΔR < -0,02
    assert bad["verdict"] == "CRITERE_NON_REMPLI" and not bad["a"]["met"] and not bad["b"]["met"]
    assert ok == C2.criterion(_rows(0.95, 0.93), ref, IDS, CFG)


# ---------- identités, refus, marqueur ----------

def test_the_code_pins_of_ld1_ld2_and_e2e5_match_the_files_and_a_tampered_pin_is_refused():
    out = E3.check_code_identity(CFG)
    assert len(out) == len(CFG["frozen_files_sha256"]) + 8 and CFG["frozen_files_sha256"]["src/babaorum/experimental/line_detr.py"].startswith("7715a8e3")
    bad = json.loads(json.dumps(CFG))
    bad["e2e5"]["code_pins"]["scripts/train_stage_s_v2.py"] = "0" * 64
    with pytest.raises(SystemExit, match="train_stage_s_v2"):
        E3.check_code_identity(bad)


def test_the_ld2_files_are_untouched():
    for rel, prefix in (("scripts/train_ld2.py", "bb301d04"), ("scripts/ld2_core.py", "8c9b0753"), ("config/ld2.json", "6907f17b"),
                        ("tests/test_ld2.py", "9352adc7"), ("src/babaorum/experimental/line_detr.py", "7715a8e3"),
                        ("scripts/train_line_detr.py", "3552c072"), ("scripts/stage_s_notebooks/build_kaggle_ld2.py", "8d9d0946"),
                        ("scripts/stage_s_notebooks/ld2/kaggle_ld2.ipynb", "2bfc0120")):
        if (ROOT / rel).is_file():
            assert C2.sha256_file(ROOT / rel).startswith(prefix), rel


def _ld2_dir(tmp_path, **over):
    d = tmp_path / "ld2"
    d.mkdir()
    (d / "ld2_main_step003271.pt").write_bytes(b"ck")
    sha = hashlib.sha256(b"ck").hexdigest()
    man = {"S2": 3271, "alphabet": ["<blank>", "a"],
           "freeze": {"checkpoint": "ld2_main_step003271.pt", "checkpoint_sha256": sha, "threshold": 0.6}}
    man.update(over.get("man", {}))
    (d / "ld2_manifest.json").write_text(json.dumps(man))
    (d / "ld2_report.json").write_text(json.dumps({"checkpoint_sha256": over.get("rep_sha", sha)}))
    cfg = json.loads(json.dumps(CFG))
    cfg["ld2"]["checkpoint_sha256"] = sha
    return d, cfg, sha


def test_ld2_identity_accepts_a_consistent_output_and_refuses_each_mismatch(tmp_path):
    d, cfg, sha = _ld2_dir(tmp_path)
    ident = E3.check_ld2_identity(d, cfg)
    assert ident["threshold"] == 0.6 and ident["alphabet"] == ["<blank>", "a"] and ident["checkpoint_sha256"] == sha
    with pytest.raises(SystemExit, match="sha différent"):
        E3.check_ld2_identity(d, dict(cfg, ld2=dict(cfg["ld2"], checkpoint_sha256="0" * 64)))
    (tmp_path / "b").mkdir()
    d2, cfg2, _ = _ld2_dir(tmp_path / "b", man={"freeze": {"checkpoint": "x.pt"}})
    with pytest.raises(SystemExit, match="freeze"):
        E3.check_ld2_identity(d2, cfg2)
    (tmp_path / "c").mkdir()
    d3, cfg3, _ = _ld2_dir(tmp_path / "c", rep_sha="1" * 64)
    with pytest.raises(SystemExit, match="rapport de LD-2"):
        E3.check_ld2_identity(d3, cfg3)
    (tmp_path / "d").mkdir()
    d4, cfg4, sha4 = _ld2_dir(tmp_path / "d", man={"freeze": {"checkpoint": "ld2_main_step003271.pt",
                                                             "checkpoint_sha256": hashlib.sha256(b"ck").hexdigest(), "threshold": 0.5}})
    with pytest.raises(SystemExit, match="seuil"):
        E3.check_ld2_identity(d4, cfg4)
    (d4 / "ld2_report.json").unlink()
    with pytest.raises(SystemExit):
        E3.check_ld2_identity(d4, cfg4)


def _eval_args(tmp_path, **kw):
    base = dict(smoke=True, decoy_reference=True, work=str(tmp_path / "w"), gen_src=str(tmp_path), extra_libs=None, e2e4_report=None,
                ld2_dir=None, e2e5_checkpoint=None, cache_dir=str(tmp_path / "hf"))
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_eval_refuses_when_the_report_or_the_marker_exists_and_a_decoy_reference_outside_the_smoke(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld3"
    out.mkdir(parents=True)
    monkeypatch.setattr(E3, "make_context", lambda *a, **k: pytest.fail("aucun modèle ne doit être chargé"))
    (out / "ld3_report.json").write_text("{}")
    with pytest.raises(SystemExit, match="existe déjà"):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    (out / "ld3_report.json").unlink()
    (out / E3.EVAL_STARTED).write_text("{}")
    with pytest.raises(SystemExit, match="décision explicite de pA"):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert (out / E3.EVAL_STARTED).is_file()                                              # jamais supprimé
    (out / E3.EVAL_STARTED).unlink()
    with pytest.raises(SystemExit, match="fumée"):
        E3.phase_EVAL(_eval_args(tmp_path, smoke=False, decoy_reference=True), CFG, None, out, tmp_path)


def _fake_page_row(pid, p16=0.9, r16=0.9, p32=0.9, r32=0.92):
    def score(P, R):
        return {"n_pred": 5, "n_gt": 5, "precision": P, "recall": R, "precision_none": False, "splits": 0, "merges": 1,
                "cer": {"edits": 1, "chars": 10, "lines": 2, "skipped": 0, "matched_readable_lines": 2, "matched_readable_chars": 10,
                        "skipped_by_cause": {"degenerate_chord": 0, "band_outside_support": 0, "reader_exception": 0, "unknown": 0}},
                "n_zones": 3, "reduction": {"zones": 3, "reduced": 1, "reduced_share": 1 / 3, "mean_f_reduced": 0.9, "min_f": 0.9},
                "edge": {"bins": LB.page_counts([("0-8", True), ("0-8", False), (">32", True)], []), "gt_hors_zone": 0, "gt_in_zones": 3}}
    row = {"page": pid, "seconds": {"load": 0.1, "detector": 0.1, "total": 1.0, "B3": 0.2}}
    for arm, (P, R) in (("M16", (p16, r16)), ("M32", (p32, r32))):
        row[arm] = {src: score(P, R) for src in E3.SOURCES}
        for src in E3.SOURCES:
            row["seconds"][f"{arm}/{src}"] = 0.3
    row["B3"] = {"precision": 0.71, "recall": 0.923, "n_pred": 6, "n_gt": 5, "matched": 5, "seconds": 0.2}
    return row


def test_the_marker_is_created_before_the_first_page_and_an_interrupted_eval_cannot_be_restarted(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld3"
    out.mkdir(parents=True)
    ident = {"checkpoint_sha256": "ab" * 32, "alphabet": ["<blank>"]}
    monkeypatch.setattr(E3, "make_context", lambda *a, **k: {"ident": ident, "pages_root": None})
    seen = []

    def fake_produce(mode, cfg, src, work, pages_root, **kw):
        seen.append(("produce", (out / E3.EVAL_STARTED).is_file(), mode, kw["called_from_eval_ld3"]))
        return {"mode": mode, "seed": 1, "names": ["decoy_0000", "decoy_0001"], "files_sha256": {}, "generator": {}, "seconds_total": 0,
                "overlap": {"status": "OK", "blocking": False}}

    def fake_page(pid, root, ctx):
        seen.append(("page", pid, (out / E3.EVAL_STARTED).is_file()))
        if pid == "decoy_0001":
            raise RuntimeError("interruption")
        return _fake_page_row(pid)

    monkeypatch.setattr(E3.D3, "produce_pages", fake_produce)
    monkeypatch.setattr(E3, "evaluate_page_ld3", fake_page)
    with pytest.raises(RuntimeError, match="interruption"):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert seen == [("produce", True, "decoy", True), ("page", "decoy_0000", True), ("page", "decoy_0001", True)]
    started = json.loads((out / E3.EVAL_STARTED).read_text())
    assert started["checkpoint_sha256"] == "ab" * 32 and started["mode"] == "decoy" and "started_utc" in started
    assert (out / "ld3_eval_pages.jsonl").read_text().count("\n") == 1 and not (out / "ld3_report.json").exists()
    with pytest.raises(SystemExit, match="décision explicite de pA"):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert len(seen) == 3 and (out / E3.EVAL_STARTED).is_file()


def test_a_blocking_overlap_stops_before_any_page_is_evaluated(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld3"
    out.mkdir(parents=True)
    monkeypatch.setattr(E3, "make_context", lambda *a, **k: {"ident": {"checkpoint_sha256": "ab" * 32}, "pages_root": None})
    monkeypatch.setattr(E3.D3, "produce_pages", lambda *a, **k: {
        "mode": "decoy", "seed": 1, "names": ["decoy_0000"], "files_sha256": {}, "generator": {}, "seconds_total": 0,
        "overlap": {"status": "OVERLAP_PAGE", "blocking": True}})
    monkeypatch.setattr(E3, "evaluate_page_ld3", lambda *a, **k: pytest.fail("aucune page ne doit être évaluée"))
    with pytest.raises(SystemExit, match="recouvrement bloquant"):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)


def test_a_real_eval_builds_the_mf_0060_pages_only_through_data3_with_the_flag(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld3"
    out.mkdir(parents=True)
    got = {}
    monkeypatch.setattr(E3, "make_context", lambda *a, **k: {"ident": {"checkpoint_sha256": "ab" * 32}, "pages_root": None})

    def fake_produce(mode, cfg, src, work, pages_root, **kw):
        got.update(mode=mode, flag=kw["called_from_eval_ld3"], n=kw["n_pages"])
        raise RuntimeError("stop")

    monkeypatch.setattr(E3.D3, "produce_pages", fake_produce)
    with pytest.raises(RuntimeError, match="stop"):
        E3.phase_EVAL(_eval_args(tmp_path, smoke=False, decoy_reference=False), CFG, None, out, tmp_path)
    assert got == {"mode": "real", "flag": True, "n": None}


# ---------- rapport : réserves devant, prédiction recopiée, verdicts ----------

def test_the_report_has_reserves_in_front_the_prediction_copied_and_both_verdicts():
    ids = IDS
    rows = [_fake_page_row(p, 0.90, 0.90, 0.895, 0.92) for p in ids]
    ident = {"checkpoint_sha256": "e5afd143" + "0" * 56, "alphabet": ["<blank>"], "threshold": 0.6}
    rep = E3.build_report(rows, ids, CFG, {"mode": "real"}, ident, False)
    assert list(rep)[:3] == ["schema", "checkpoint_sha256", "reserves"] and rep["checkpoint_sha256"] == ident["checkpoint_sha256"]
    for r in CFG["reserves"]:
        assert r in rep["reserves"]
    assert C2.RESERVES[6] in rep["reserves"] and C2.RESERVES[7] in rep["reserves"] and rep["reserves"][-1] == C2.CER_SCOPE
    assert rep["prediction"] == {"source": CFG["prediction"]["source"], "lines": CFG["prediction"]["lines"],
                                 "amendment": CFG["prediction"]["amendment"]}
    assert rep["principal"]["verdict"] in ("PRINCIPAL_REMPLI", "PRINCIPAL_NON_REMPLI")
    assert rep["secondary"]["verdict"] in ("CRITERE_REMPLI", "CRITERE_NON_REMPLI")
    assert rep["principal"]["met"] is True and rep["secondary"]["verdict"] == "CRITERE_REMPLI"
    assert set(rep["arms"]) == {"M16", "M32"} and set(rep["arms"]["M32"]) == {"Z-PP-blocs", "Z-GT"} and rep["pages"] == ids
    assert rep["edge_totals"]["M32"]["Z-PP-blocs"]["0-8"] == {"gt_n": 20, "gt_matched": 10}
    assert rep["zone_reduction"]["M16"]["Z-GT"]["reduced"] == 10 and rep["B3"]["P"] == pytest.approx(0.71) and rep["B3"]["decoy"] is False
    assert rep["mechanism_bin_0_8"]["M32"]["Z-PP-blocs"]["recall_mean"] == 0.5
    json.dumps(rep)                                                                      # sérialisable


@NEEDS_DOC
def test_the_reserves_and_the_prediction_are_in_the_proposition_verbatim():
    doc = _norm(DOC.read_text() + (ROOT / "docs" / "END_TO_END_LD3_AMENDEMENT2.txt").read_text())      # proposition + amendement 2
    for line in CFG["reserves"][:6] + CFG["prediction"]["lines"]:
        assert _norm(line) in doc, line[:60]
    assert "20261026" in doc and "mf_0060-0069" in doc


def test_no_number_of_the_pre_registration_is_hardcoded_in_the_new_code():
    banned = {20261026, 20261018, 10000, 9000, 0.6, -0.01, 2.5, 97.5, 0.8, 0.9, -0.02}
    for rel in ("scripts/eval_ld3.py", "scripts/data3_generate.py"):
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                assert node.value not in banned, (rel, node.value, node.lineno)


def test_the_new_code_names_the_reserved_pages_only_in_docstrings_and_comments():
    for rel in ("scripts/eval_ld3.py", "scripts/data3_generate.py", "scripts/stage_s_notebooks/build_kaggle_ld3.py"):
        if not (ROOT / rel).is_file():
            continue
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"mf_00[5-6]\d|mfn_", node.value):
                assert len(node.value) > 60 and "\n" in node.value or "mf_0060" in node.value, (rel, node.value[:60])


# ---------- carnets ----------

def _code(name):
    nb = json.loads((LD3 / name).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@NEEDS_CARNET
def test_the_notebooks_on_disk_equal_the_builder_output_and_are_small():
    B3 = _b()
    assert json.loads((LD3 / "kaggle_ld3.ipynb").read_text()) == B3.build_notebook()
    assert json.loads((LD3 / "kaggle_ld3_smoke.ipynb").read_text()) == B3.build_smoke_notebook()
    assert json.loads((LD3 / "full_kernel" / "notebook.ipynb").read_text()) == B3.build_notebook()
    assert json.loads((LD3 / "smoke_kernel" / "notebook.ipynb").read_text()) == B3.build_smoke_notebook()
    assert _code("kaggle_ld3.ipynb") == B3.cell("full") and _code("kaggle_ld3_smoke.ipynb") == B3.cell("smoke")
    assert json.loads((LD3 / "kit_files.json").read_text()) == B3.kit_files()
    for p in (LD3 / "full_kernel" / "notebook.ipynb", LD3 / "smoke_kernel" / "notebook.ipynb"):
        assert p.stat().st_size < 300 * 1024
    assert json.loads((LD3 / "full_kernel" / "kernel-metadata.json").read_text()) == B3.kernel_metadata(
        "mabaku/babaorum-ld3-run", "babaorum-ld3-run", gpu=True)
    assert json.loads((LD3 / "smoke_kernel" / "kernel-metadata.json").read_text()) == B3.kernel_metadata(
        "mabaku/babaorum-ld3-smoke", "babaorum-ld3-smoke")


@NEEDS_CARNET
def test_metadata_gpu_for_the_full_run_cpu_for_the_smoke_and_the_sources():
    full = json.loads((LD3 / "full_kernel" / "kernel-metadata.json").read_text())
    smoke = json.loads((LD3 / "smoke_kernel" / "kernel-metadata.json").read_text())
    assert full["enable_gpu"] is True and smoke["enable_gpu"] is False and full["is_private"] and full["code_file"] == "notebook.ipynb"
    assert set(full["dataset_sources"]) == {"mabaku/babaorum-ld3-kit", "mabaku/babaorum-stage-s-bundle-v5"}
    assert set(full["kernel_sources"]) == {"mabaku/babaorum-ld2-run", "mabaku/babaorum-e2e5-lambda",
                                           "mabaku/babaorum-e2e3-run3-fp-classes", "mabaku/babaorum-e2e4-extraction"}


@NEEDS_CARNET
def test_the_kit_has_every_file_the_code_reads_and_the_install_cell_text_is_unchanged():
    files = set(json.loads((LD3 / "kit_files.json").read_text()))
    for rel in files:
        assert (ROOT / rel).is_file(), rel
    for rel in ("scripts/eval_ld3.py", "scripts/data3_generate.py", "scripts/data2_generate.py", "scripts/train_ld2.py",
                "scripts/ld2_core.py", "scripts/ld1b_border_diag.py", "config/ld3.json", "config/ld2.json",
                "config/ld1_line_detr.json", "config/stage_s_smoke_v2.json", "config/data2_pages.json",
                "src/babaorum/experimental/stage_s_arms.py", "tests/test_ld3.py"):
        assert rel in files, rel
    assert hashlib.sha256((CARNETS / "KAGGLE_INSTALL.txt").read_bytes()).hexdigest().startswith("a0cda2416808a1ab")


@NEEDS_CARNET
def test_the_cells_never_name_a_reserved_page_never_list_a_directory_and_the_smoke_is_decoy_only():
    for name in ("kaggle_ld3.ipynb", "kaggle_ld3_smoke.ipynb"):
        code = _code(name)
        for bad in ("glob(", "rglob", "listdir", "scandir", "os.walk", "iterdir", "mf_005", "mf_006", "mfn_"):
            assert bad not in code, (name, bad)
    smoke = _code("kaggle_ld3_smoke.ipynb")
    assert '"--smoke", "--decoy-reference"' in smoke and '"decoy_0000"' in smoke and "--from-eval" not in smoke
    assert "5400" in smoke and "10800" in smoke                                         # plafonds CPU de la fumée seulement
    full = _code("kaggle_ld3.ipynb")
    assert "--smoke" not in full and "--decoy-reference" not in full and "'deadline_s': 9000" in full


def _data2_common():
    spec = importlib.util.spec_from_file_location("data2_common_ld3test", ROOT / "scripts" / "data2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Run:
    """Exécute le vrai FULL_BODY avec un faux run_step et une fausse horloge. plan[étape] = {"rc", "dur", "estimate"}."""

    def __init__(self, tmp_path, monkeypatch, plan, cuda=True, start=0.0):
        B3 = _b()
        self.calls, self.clock, self.plan = [], [start], plan
        self.out = tmp_path / "work"
        self.out.mkdir()
        self.C = _data2_common()
        monkeypatch.setattr(self.C, "run_step", self.run_step)
        tree = ast.parse(B3.cell("full"))
        lits = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("LIMITS", "TIMEOUTS", "REQUIRED_PIP", "COMMIT", "RUN_FILES")}
        self.limits, self.timeouts = lits["LIMITS"], lits["TIMEOUTS"]
        ns = {"time": types.SimpleNamespace(monotonic=lambda: self.clock[0]), "json": json, "os": os, "sys": sys, "shlex": shlex,
              "Path": Path, "OUT": self.out, "C": self.C, "C2": C2, "T_START": 0.0, "KIT": Path("/k"), "PAGES": Path("/p"),
              "CKPT": Path("/c"), "LD2DIR": Path("/d"), "E2E4": Path("/e"), "BUNDLE_ZIP": Path("/b"), "sha256": lambda p: "0" * 64,
              "_utc": lambda: "00:00:00", "LIMITS": self.limits, "TIMEOUTS": self.timeouts, "REQUIRED_PIP": lits["REQUIRED_PIP"],
              "COMMIT": lits["COMMIT"], "RUN_FILES": lits["RUN_FILES"], "shutil": __import__("shutil")}
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        fake_torch.__version__ = "9.9"
        fake_torch.version = types.SimpleNamespace(cuda="12.x")
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(importlib.metadata, "version", lambda name: f"v-{name}")
        ns["RUN"] = self.out / "ld3"
        ns["CAL_OUT"] = self.out / "ld3_cal"
        exec(compile(B3._argv_builder("cuda"), "<_argv_builder>", "exec"), ns)
        monkeypatch.setitem(ns, "BUNDLE_DIR", tmp_path / "bundle")
        monkeypatch.setitem(ns, "SRC", tmp_path / "src")
        monkeypatch.setitem(ns, "LIBS", tmp_path / "libs")
        monkeypatch.setitem(ns, "WORK", tmp_path / "wk")
        ns["d_mille_check"] = lambda: None
        for st in ns["GEN_PREP"]:
            if st["name"] == "mille_check":
                st["fn"] = ns["d_mille_check"]
        exec(compile(B3.FULL_BODY, "<FULL_BODY>", "exec"), ns)
        self.ns = ns

    def run_step(self, name, argv, timeout_s, logs_dir, cwd=None, env=None, kill_grace_s=10.0):
        p = {"rc": 0, "dur": 1.0, "estimate": 100.0, **self.plan.get(name, {})}
        self.calls.append((name, timeout_s, list(argv)))
        self.clock[0] += p["dur"]
        if name == "CAL" and p["rc"] == 0:
            (self.out / "ld3_cal").mkdir(parents=True, exist_ok=True)
            (self.out / "ld3_cal" / "ld3_cal.json").write_text(json.dumps({"estimate_s": p["estimate"], "page": "mf_0045"}))
        return {"step": name, "rc": p["rc"], "seconds": p["dur"], "stdout": "", "stderr": "", "stderr_tail": ""}

    @property
    def exit(self):
        return self.ns["EXIT"]

    @property
    def names(self):
        return [c[0] for c in self.calls]

    def timeout(self, name):
        return dict((c[0], c[1]) for c in self.calls)[name]


@NEEDS_CARNET
def test_the_full_run_order_caps_and_the_9000_s_global_deadline(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"pip": {"dur": 100.0}, "CAL": {"dur": 200.0, "estimate": 900.0}, "D": {"dur": 50.0}})
    assert r.exit["status"] == "DONE"
    assert r.names == ["pip", "unzip", "clone", "checkout", "D", "controls", "CAL", "EVAL"]
    lim, to = r.limits, r.timeouts
    assert lim["deadline_s"] == 9000 and r.timeout("pip") == to["pip"] and r.timeout("CAL") == to["CAL"]
    assert r.timeout("EVAL") == 9000 - (100.0 + 1 + 1 + 1 + 50.0 + 1 + 200.0) - lim["eval_margin_s"]
    est = json.loads((r.out / "ld3_eval_estimate.json").read_text())
    assert est["estimate_s"] == 900.0 * 1.5 + lim["generation_estimate_s"] and est["factor"] == 1.5
    pip = dict((c[0], c[2]) for c in r.calls)["pip"]
    assert pip[:4] == [sys.executable, "-m", "pip", "install"] and "--target" in pip                    # bibliothèques isolées


@NEEDS_CARNET
def test_the_estimation_gate_stops_before_the_pages_are_generated_or_read(tmp_path, monkeypatch):
    lim = _b()._limits()
    avail_guess = 9000 - 7 - lim["eval_margin_s"]
    r = Run(tmp_path, monkeypatch, {"CAL": {"estimate": (avail_guess - lim["generation_estimate_s"]) / 1.5 + 100.0}})
    assert "EVAL" not in r.names and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "estimate"
    est = json.loads((r.out / "ld3_eval_estimate.json").read_text())
    assert est["estimate_s"] > est["available_s"]
    (tmp_path / "ok").mkdir()
    ok = Run(tmp_path / "ok", monkeypatch, {"CAL": {"estimate": 100.0}})
    assert "EVAL" in ok.names and ok.exit["status"] == "DONE"


@NEEDS_CARNET
def test_cuda_is_required_and_a_failed_step_stops_the_run_with_incomplete_at_that_step(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {}, cuda=False)
    assert r.names == [] and r.exit["status"] == "FAILED" and "CUDA" in r.exit["exception"]
    (tmp_path / "a").mkdir()
    r = Run(tmp_path / "a", monkeypatch, {"CAL": {"rc": "TIMEOUT"}})
    assert r.names[-1] == "CAL" and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "CAL"
    (tmp_path / "b").mkdir()
    r = Run(tmp_path / "b", monkeypatch, {"D": {"dur": 9000.0}})
    assert r.names[-1] == "D" and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "controls"


@NEEDS_CARNET
def test_the_eval_step_is_an_argv_step_without_smoke_flag_and_every_long_step_is_not_fn(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    argv = dict((c[0], c[2]) for c in r.calls)["EVAL"]
    assert argv[1].endswith("eval_ld3.py") and argv[argv.index("--phase") + 1] == "EVAL" and "--smoke" not in argv
    fn_steps = [s["name"] for s in r.ns["STEPS"] if "fn" in s]
    assert fn_steps == ["pre", "mille_check", "estimate"]


def _mount_tree(tmp_path, fmt):
    inp = tmp_path / "input"
    base = (lambda slug, sub: inp / slug) if fmt == "A" else (lambda slug, sub: inp / sub / "mabaku" / slug)
    kit = base("babaorum-ld3-kit", "datasets")
    (kit / "config").mkdir(parents=True)
    (kit / "scripts").mkdir()
    (kit / "src").mkdir()
    import shutil
    shutil.copy(ROOT / "scripts" / "data2_common.py", kit / "scripts")
    shutil.copy(ROOT / "scripts" / "ld2_core.py", kit / "scripts")
    shutil.copy(ROOT / "config" / "data2_pages.json", kit / "config")
    (kit / "config" / "ld3.json").write_text("{}")
    mf = base("babaorum-e2e3-run3-fp-classes", "notebooks") / "e2e3" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "pages" / "mf_0000.json").write_text("{}")
    e5 = base("babaorum-e2e5-lambda", "notebooks")
    (e5 / "logs").mkdir(parents=True)
    (e5 / "logs" / "checkpoint_last.pt").write_bytes(b"e5")
    ld2 = base("babaorum-ld2-run", "notebooks") / "ld2"
    ld2.mkdir(parents=True)
    (ld2 / "ld2_manifest.json").write_text("{}")
    (ld2 / "ld2_main_step003271.pt").write_bytes(b"l2")
    e4 = base("babaorum-e2e4-extraction", "notebooks")
    e4.mkdir(parents=True)
    (e4 / "e2e4_extraction_report.json").write_bytes(b"e4")
    bundle = base("babaorum-stage-s-bundle-v5", "datasets")
    bundle.mkdir(parents=True)
    (bundle / "stage_s_bundle_v5.zip.bin").write_bytes(b"bd")
    return inp


@NEEDS_CARNET
@pytest.mark.parametrize("fmt", ["A", "B"])
def test_the_mount_resolution_takes_the_real_roots_in_both_formats_and_never_lists(tmp_path, monkeypatch, fmt):
    B3 = _b()
    inp = _mount_tree(tmp_path, fmt)
    sha = lambda b: hashlib.sha256(b).hexdigest()                                         # noqa: E731
    src = B3.cell("full", out=str(tmp_path / "out"),
                  kit=[str(inp / "babaorum-ld3-kit"), str(inp / "datasets" / "mabaku" / "babaorum-ld3-kit")],
                  pages=[str(inp / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf"),
                         str(inp / "notebooks" / "mabaku" / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf")],
                  ckpt=[str(inp / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt"),
                        str(inp / "notebooks" / "mabaku" / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt")],
                  ld2=[str(inp / "babaorum-ld2-run" / "ld2"), str(inp / "notebooks" / "mabaku" / "babaorum-ld2-run" / "ld2")],
                  e2e4=[str(inp / "babaorum-e2e4-extraction" / "e2e4_extraction_report.json"),
                        str(inp / "notebooks" / "mabaku" / "babaorum-e2e4-extraction" / "e2e4_extraction_report.json")],
                  bundle=[str(inp / "babaorum-stage-s-bundle-v5" / "stage_s_bundle_v5.zip.bin"),
                          str(inp / "datasets" / "mabaku" / "babaorum-stage-s-bundle-v5" / "stage_s_bundle_v5.zip.bin")],
                  e2e5_sha=sha(b"e5"), ld2_sha=sha(b"l2"), e2e4_sha=sha(b"e4"), bundle_sha=sha(b"bd"))
    head = src.split("\nBUNDLE_DIR = ")[0]
    touched = []
    monkeypatch.setattr(os, "listdir", lambda *a, **k: touched.append("listdir") or [])
    monkeypatch.setattr(os, "scandir", lambda *a, **k: touched.append("scandir") or iter(()))
    for name in ("glob", "iterdir", "rglob"):
        monkeypatch.setattr(Path, name, lambda *a, _n=name, **k: touched.append(_n) or iter(()))
    ns = {"__name__": "ld3_mount_test"}
    saved = list(sys.path)
    exec(compile(head, "ld3_head", "exec"), ns)
    monkeypatch.undo()
    sys.path[:] = saved
    sys.modules.pop("ld2_core", None)
    sys.modules.pop("data2_common", None)
    assert touched == [] and ns["KIT"].name == "babaorum-ld3-kit" and ns["LD2DIR"].name == "ld2" and ns["BUNDLE_ZIP"].name.endswith(".bin")
    assert str(ns["PAGES"]).endswith("e2e3/mf")
    sys.path[:] = saved


@NEEDS_CARNET
def test_the_mount_resolution_refuses_a_wrong_sha_and_two_candidates(tmp_path):
    B3 = _b()
    inp = _mount_tree(tmp_path, "A")
    sha = lambda b: hashlib.sha256(b).hexdigest()                                         # noqa: E731
    base = dict(out=str(tmp_path / "o"), kit=[str(inp / "babaorum-ld3-kit")],
                pages=[str(inp / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf")],
                ckpt=[str(inp / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt")], ld2=[str(inp / "babaorum-ld2-run" / "ld2")],
                e2e4=[str(inp / "babaorum-e2e4-extraction" / "e2e4_extraction_report.json")],
                bundle=[str(inp / "babaorum-stage-s-bundle-v5" / "stage_s_bundle_v5.zip.bin")],
                e2e5_sha=sha(b"e5"), ld2_sha=sha(b"l2"), e2e4_sha=sha(b"e4"), bundle_sha=sha(b"bd"))
    head = lambda **kw: B3.cell("full", **{**base, **kw}).split("\nBUNDLE_DIR = ")[0]        # noqa: E731
    for key, msg in (("e2e5_sha", "E2E-5"), ("ld2_sha", "LD-2"), ("bundle_sha", "bundle")):
        with pytest.raises(AssertionError, match=msg):
            exec(compile(head(**{key: "0" * 64}), "x", "exec"), {})
        sys.modules.pop("ld2_core", None)
    with pytest.raises(RuntimeError):
        exec(compile(head(kit=[str(inp / "babaorum-ld3-kit")] * 2), "x", "exec"), {})
    sys.modules.pop("ld2_core", None)


@NEEDS_CARNET
def test_the_smoke_cell_runs_the_decoy_flags_with_cpu_caps(tmp_path, monkeypatch):
    B3 = _b()
    inp = _mount_tree(tmp_path, "A")
    sha = lambda b: hashlib.sha256(b).hexdigest()                                         # noqa: E731
    calls, caps = [], {}

    def fake_execute(out, steps, heartbeat_interval_s=60.0, cleanup=None):
        for st in steps:
            if "fn" in st:
                continue                                                           # contrôle des fichiers du générateur : pas ici
            argv = st["argv"]() if callable(st["argv"]) else st["argv"]
            calls.append((st["name"], [str(a) for a in argv]))
            caps[st["name"]] = st["timeout_s"]
        return {"status": "DONE", "step": None, "exit_code": 0, "durations_s": {}}

    real_spec = importlib.util.spec_from_file_location

    def spec_with_fake_loader(name, path, *a, **k):
        spec = real_spec(name, path, *a, **k)
        if name == "data2_common":
            spec.loader = types.SimpleNamespace(create_module=lambda s: None,
                                                exec_module=lambda m: setattr(m, "execute_pipeline", fake_execute))
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec_with_fake_loader)
    src = B3.cell("smoke", out=str(tmp_path / "out"),
                  kit=[str(inp / "babaorum-ld3-kit")], pages=[str(inp / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf")],
                  ckpt=[str(inp / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt")], ld2=[str(inp / "babaorum-ld2-run" / "ld2")],
                  e2e4=[str(inp / "babaorum-e2e4-extraction" / "e2e4_extraction_report.json")],
                  bundle=[str(inp / "babaorum-stage-s-bundle-v5" / "stage_s_bundle_v5.zip.bin")],
                  e2e5_sha=sha(b"e5"), ld2_sha=sha(b"l2"), e2e4_sha=sha(b"e4"), bundle_sha=sha(b"bd"))
    saved = list(sys.path)
    exec(compile(src, "ld3_smoke_cell", "exec"), {"__name__": "smoke"})
    sys.path[:] = saved
    sys.modules.pop("ld2_core", None)
    by = dict(calls)
    assert [c[0] for c in calls] == ["pip", "unzip", "clone", "checkout", "D", "pytest", "controls", "CAL", "EVAL"]
    assert not any(re.search(r"mf_00[5-6]\d|mfn_", a) for _n, argv in calls for a in argv)
    assert "--smoke" in by["EVAL"] and "--decoy-reference" in by["EVAL"] and "--smoke" in by["CAL"]
    assert by["CAL"][by["CAL"].index("--pages") + 1] == "mf_0000" and by["CAL"][by["CAL"].index("--eval-pages") + 1] == "decoy_0000"
    assert by["EVAL"][by["EVAL"].index("--device") + 1] == "cpu" and caps["CAL"] == 5400 and caps["EVAL"] == 10800
    assert "tests/test_ld3.py" in " ".join(by["pytest"]) and "-k" not in by["pytest"]


def test_evaluate_page_applies_each_arm_margin_computes_pp_zones_once_and_shares_them(monkeypatch):
    """Intégration légère : modèle, détecteur et lecture factices ; on vérifie la mécanique (marge par bras, zones PP uniques,
    structure de la ligne de page, table de bord, B3)."""
    g = _gt(0, 100.0, y=60.0)
    page = {"gt_lines": [g], "blocks": [{"lines": [{"id": "g0"}], "box": (50.0, 20.0, 500.0, 120.0)}],
            "image": np.ones((200, 600), np.float32)}
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: page)
    pp_calls, margins = [], []
    monkeypatch.setattr(E3.TL, "pp_zones", lambda *a, **k: pp_calls.append(1) or [(0, (50.0, 20.0, 500.0, 120.0))])

    def fake_zone_raws(model, pg, zones, thr, alphabet, device):
        margins.append(LD.MARGIN_H)
        pts = np.stack([np.linspace(100.0, 400.0, 8), np.full(8, 60.0)], axis=-1)
        raw = {"prob": np.array([0.9, 0.1]), "pts": np.stack([pts, pts]), "neg": np.array([12.0, 12.0]), "pos": np.array([4.0, 4.0]),
               "hyp": {0: "ab"}, "hyp_none": {}}
        meta = LD.ZoneMeta(ix0=0, iy0=0, fx=1.0, fy=1.0, crop_w=600, crop_h=200)
        return [(zid, meta, raw) for zid, _box in zones]

    monkeypatch.setattr(TD, "zone_raws", fake_zone_raws)
    ctx = {"threshold": 0.6, "alphabet": ["<blank>", "a", "b"], "device": None, "model": object(), "cache": None, "L0": None,
           "det_fn": None, "arms": E3.arms(CFG), "b3": lambda pid: {"precision": 0.7, "recall": 0.9, "n_pred": 3, "n_gt": 1}}
    row = E3.evaluate_page_ld3("p", Path("."), ctx)
    assert pp_calls == [1] and margins == [0.5, 0.5, 1.0, 1.0] and LD.MARGIN_H == 0.5       # PP une fois ; M16 puis M32, restauré
    for arm in ("M16", "M32"):
        for src in E3.SOURCES:
            s = row[arm][src]
            assert s["recall"] == 1.0 and s["n_pred"] == 1 and s["cer"]["lines"] == 1 and s["reduction"]["reduced"] == 0
            assert s["edge"]["bins"]["16-32"]["gt_matched"] == 1 and s["edge"]["gt_hors_zone"] == 0       # d = min(60-12-20, 120-64) = 28
    assert row["B3"]["recall"] == 0.9 and {"load", "detector", "M16/Z-PP-blocs", "M32/Z-GT", "B3", "total"} <= set(row["seconds"])


# ---------- gel 2 : F1 à F7 ----------

def test_f1_every_code_file_not_part_of_ld3_is_pinned_and_a_changed_one_is_refused():
    pins = CFG["frozen_files_sha256"]
    must = ["scripts/train_ld2.py", "scripts/ld2_core.py", "config/ld2.json", "scripts/ld1b_border_diag.py", "scripts/data2_generate.py",
            "scripts/data2_common.py", "config/data2_pages.json", "config/ld1_line_detr.json", "scripts/ld0_zone_coverage.py",
            "src/babaorum/line_geometry.py", "src/babaorum/experimental/line_detr.py", "scripts/train_line_detr.py"]
    for rel in must:
        assert rel in pins and pins[rel] == C2.sha256_file(ROOT / rel), rel
    assert pins["scripts/train_ld2.py"] == "bb301d04445624c0e2ba759f4c9a882fe6315c47e2e1b3a6e13ab5e6380d08a7"
    assert pins["scripts/ld2_core.py"] == "8c9b0753ba99ac0e441ef86a20bb04bac1120eda6ea141a0f680354108fb4f49"
    assert pins["config/ld2.json"] == "6907f17b6c74e2e7179d03cef75ad2b43f990161e3f2cf7284435c35d7f85896"
    for rel in ("scripts/data2_generate.py", "src/babaorum/line_geometry.py", "scripts/ld1b_border_diag.py"):
        bad = json.loads(json.dumps(CFG))
        bad["frozen_files_sha256"][rel] = "0" * 64
        with pytest.raises(SystemExit, match=Path(rel).name):
            E3.check_code_identity(bad)


def _toy_pages(texts):
    return lambda root, pid, target_h: {"lines": [{"legibility": "readable", "text": texts[pid]},
                                                    {"legibility": "illegible", "text": "ZZ"}]}


def test_f2_the_alphabet_is_rebuilt_like_run_v2_and_must_equal_the_ld2_manifest_element_by_element(monkeypatch):
    from babaorum.experimental.stage_s_data import build_alphabet
    cfg = json.loads(json.dumps(CFG))
    cfg["e2e5"]["train_pages"] = ["p0", "p1"]
    cfg["e2e5"]["alphabet_size"] = 6
    monkeypatch.setattr(E3, "load_scaled_page", _toy_pages({"p0": "ba c", "p1": "dd"}))
    rebuilt = E3.rebuild_e2e5_alphabet("/racine", cfg, 32.0)
    assert rebuilt == build_alphabet([{"lines": [{"legibility": "readable", "text": "ba c"}]}, {"lines": [{"legibility": "readable", "text": "dd"}]}])
    assert rebuilt == ["<blank>", " ", "a", "b", "c", "d"]
    assert E3.check_alphabet(rebuilt, list(rebuilt), cfg) == rebuilt
    swapped = list(rebuilt)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    with pytest.raises(SystemExit, match="diffère"):
        E3.check_alphabet(rebuilt, swapped, cfg)                                              # même contenu, ordre différent
    with pytest.raises(SystemExit, match="diffère"):
        E3.check_alphabet(rebuilt, rebuilt + ["e"], cfg)
    cfg["e2e5"]["alphabet_size"] = 74
    with pytest.raises(SystemExit, match="taille"):
        E3.check_alphabet(rebuilt, list(rebuilt), cfg)
    assert CFG["e2e5"]["train_pages"] == [f"mf_{i:04d}" for i in range(50)] and CFG["e2e5"]["alphabet_size"] == 74
    src = (ROOT / "scripts" / "eval_ld3.py").read_text()
    assert "check_alphabet(rebuild_e2e5_alphabet(args.root" in src and 'ident["alphabet"] = check_alphabet' in src


def test_f3_the_report_reserves_come_in_the_requested_order():
    rows = [_fake_page_row(p) for p in IDS]
    rep = E3.build_report(rows, IDS, CFG, {"mode": "real"}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False)
    r = rep["reserves"]
    assert r[:2] == [C2.RESERVES[6], C2.RESERVES[7]] and r[2] == CFG["reserves"][0]                 # G2 : « même distribution » en 3e
    assert r[3:7] == [C2.RESERVES[i] for i in (0, 3, 2, 5)]
    assert r[7:-1] == CFG["reserves"][1:] and r[-1] == C2.CER_SCOPE and len(r) == 6 + len(CFG["reserves"]) + 1
    assert "0,934" in r[0] and "Le_Matin_0554721_T0000004" in r[0] and "LD-1B" in r[1] and "0,843" in r[1]
    assert r[2].startswith("même distribution que mf_0000-0059")


def test_f5_the_cal_page_must_be_a_training_page():
    assert E3.check_cal_page("mf_0045", CFG) == "mf_0045" and E3.check_cal_page("mf_0000", CFG) == "mf_0000"
    for bad in ("mf_0060", "mf_0069", "mf_0050", "mfn_0000", "decoy_0000", "mf_0049x", "page", "mf_005"):
        with pytest.raises(SystemExit):
            E3.check_cal_page(bad, CFG)


def test_f6_real_ids_must_be_exactly_the_ten_data3_names_and_smoke_ids_decoy_names():
    real = D3.real_names(CFG)
    assert E3.check_eval_ids(real, "real", CFG) == real
    for bad in (real[:9], real[::-1], real + ["mf_0070"], ["decoy_0000"], [f"mf_00{i}" for i in range(50, 60)]):
        with pytest.raises(SystemExit):
            E3.check_eval_ids(bad, "real", CFG)
    assert E3.check_eval_ids(["decoy_0000", "decoy_0012"], "smoke", CFG) == ["decoy_0000", "decoy_0012"]
    for bad in ([], ["mf_0060"], ["decoy_000"], ["mf_0000"], ["decoy_0000", "mfn_0001"]):
        with pytest.raises(SystemExit, match="leurres"):
            E3.check_eval_ids(bad, "smoke", CFG)


def test_f6_eval_stops_before_the_first_page_when_the_generated_ids_are_wrong(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld3"
    out.mkdir(parents=True)
    monkeypatch.setattr(E3, "make_context", lambda *a, **k: {"ident": {"checkpoint_sha256": "ab" * 32}, "pages_root": None})
    monkeypatch.setattr(E3.D3, "produce_pages", lambda *a, **k: {
        "mode": "real", "seed": 1, "names": ["mf_0060", "mf_0061"], "files_sha256": {}, "generator": {}, "seconds_total": 0,
        "overlap": {"status": "OK", "blocking": False}})
    monkeypatch.setattr(E3, "evaluate_page_ld3", lambda *a, **k: pytest.fail("aucune page ne doit être évaluée"))
    with pytest.raises(SystemExit, match="exactement mf_0060"):
        E3.phase_EVAL(_eval_args(tmp_path, smoke=False, decoy_reference=False), CFG, None, out, tmp_path)
    (out / E3.EVAL_STARTED).unlink()
    monkeypatch.setattr(E3.D3, "produce_pages", lambda *a, **k: {
        "mode": "decoy", "seed": 1, "names": ["mf_0000"], "files_sha256": {}, "generator": {}, "seconds_total": 0,
        "overlap": {"status": "OK", "blocking": False}})
    with pytest.raises(SystemExit, match="leurres"):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)


def test_f7_the_run_identity_records_the_sha_of_the_ld3_files_and_the_torch_cuda_versions():
    ident = E3.run_identity(CFG)
    assert set(ident["files_sha256"]) == {"scripts/eval_ld3.py", "scripts/data3_generate.py", "config/ld3.json", "tests/test_ld3.py"}
    for rel, sha in ident["files_sha256"].items():
        assert sha == C2.sha256_file(ROOT / rel), rel
    assert ident["torch"] == torch.__version__ and "cuda" in ident and "gpu" in ident and "cuda_available" in ident
    rep = E3.build_report([_fake_page_row(p) for p in IDS], IDS, CFG, {}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False, ident)
    assert rep["run_identity"] == ident
    src = (ROOT / "scripts" / "eval_ld3.py").read_text()
    assert 'rec["run_identity"] = run_identity(cfg)' in src and '"run_identity": run_identity(cfg)' in src   # sortie d'étape et controls
    assert CFG["run_identity_files"] == list(ident["files_sha256"])


# ---------- gel 3 : G1 à G4 ----------

CTX1 = ("Borne théorique : si LD-2 ressemble à LD-1B, environ 27 % des lignes sont dans la tranche 0-8 px avec un rappel d'environ 0,84. "
        "Effacer tout le défaut de bord donnerait au plus environ +0,04 de rappel.")
CTX2 = "l'indice de page de LD-2 (r = +0,64 entre la part de lignes de bord et le rappel) va plutôt CONTRE l'hypothèse du bord."


def test_g1_the_prediction_has_the_two_context_sentences_and_the_amended_readings():
    lines = CFG["prediction"]["lines"]
    assert lines[0] == CTX1 and lines[1] == CTX2 and len(lines) == 10
    assert lines[2].startswith("ΔR(M32 − M16)") and lines[6].startswith("Critère SECONDAIRE")      # les cinq chiffres inchangés
    assert lines[7].startswith("principal rempli : la marge compte")
    assert lines[8] == "Principal non rempli = PISTE DU BORD CLOSE (décision pA), quelle que soit l'évolution du rappel de la tranche 0-8 px."
    assert lines[9].startswith("Le rappel de la tranche 0-8 px est rapporté tel quel, comme un DESCRIPTIF.")
    assert not any("LD-3b" in l and "non rempli" in l for l in lines) and not any("masqué par ΔP" in l for l in lines)
    am = CFG["prediction"]["amendment"]
    assert am == {"file": "docs/END_TO_END_LD3_AMENDEMENT2.txt",
                  "sha256": "9ce2756e7dc0bf1a7affcac8215cbedac29540933abd3e3c72b7e6cbfedc9085"}
    f = ROOT / am["file"]
    if f.is_file():
        assert C2.sha256_file(f) == am["sha256"]
        assert _norm(CTX2) in _norm(f.read_text()) and _norm(CTX1) in _norm(f.read_text())
        for l in (lines[8], lines[9]):
            assert _norm(l) in _norm(f.read_text())
    if DOC.is_file():
        assert _norm(CTX1) in _norm(DOC.read_text()) and _norm(lines[7]) in _norm(DOC.read_text())


def test_g1_the_reading_depends_only_on_the_principal_verdict_and_is_in_the_report():
    assert E3.reading_from_principal({"met": True}) == "LD3B_SUITE_LOGIQUE"
    assert E3.reading_from_principal({"met": False}) == "PISTE_DU_BORD_CLOSE"
    assert E3.reading_from_principal({"met": False, "mechanism": 0.99, "M32": {"R": 1.0}}) == "PISTE_DU_BORD_CLOSE"     # rien d'autre n'entre
    rows_hi = [_fake_page_row(p, 0.90, 0.90, 0.895, 0.92) for p in IDS]
    rows_lo = [_fake_page_row(p, 0.90, 0.90, 0.90, 0.90) for p in IDS]
    ident = {"checkpoint_sha256": "ab" * 32, "alphabet": []}
    hi = E3.build_report(rows_hi, IDS, CFG, {}, ident, False)
    lo = E3.build_report(rows_lo, IDS, CFG, {}, ident, False)
    assert hi["principal"]["met"] and hi["reading"] == "LD3B_SUITE_LOGIQUE" and not lo["principal"]["met"]
    assert lo["reading"] == "PISTE_DU_BORD_CLOSE"
    for r in (hi, lo):                                                                       # le rappel 0-8 px est identique : même mécanisme
        assert r["mechanism_bin_0_8"]["M32"]["Z-PP-blocs"]["recall_mean"] == 0.5
    assert hi["prediction"]["amendment"]["sha256"].startswith("9ce2756e")
    assert list(hi).index("reading") == list(hi).index("principal") + 1
    src = (ROOT / "scripts" / "eval_ld3.py").read_text()
    body = src[src.index("def reading_from_principal"):src.index("def mechanism")]
    assert "principal[\"met\"]" in body and "mechanism" not in body.split('"""')[2]


def _failing_cmd(code):
    return lambda src_root, raw, cfg, pages, seed: [sys.executable, "-c", code]


def test_g3_a_failing_generator_raises_with_the_last_2000_chars_and_writes_the_failure_json(tmp_path, monkeypatch):
    code = ("import sys; sys.stderr.write('A' * 5000 + 'FIN_STDERR'); sys.stdout.write('B' * 3000 + 'FIN_STDOUT'); sys.exit(3)")
    monkeypatch.setattr(D3.D2, "generate_command", _failing_cmd(code))
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(D3.GenerationFailed) as ei:
        D3.produce_pages("decoy", CFG, src, tmp_path / "work", tmp_path / "p", seed=7)
    msg = str(ei.value)
    assert "code de sortie 3" in msg and "FIN_STDERR" in msg and "FIN_STDOUT" in msg and isinstance(ei.value, RuntimeError)
    rec = json.loads((tmp_path / "work" / "data3_generate_failure.json").read_text())
    assert rec["rc"] == 3 and len(rec["stderr_tail"]) == 2000 and rec["stderr_tail"].endswith("FIN_STDERR")
    assert len(rec["stdout_tail"]) == 2000 and rec["stdout_tail"].endswith("FIN_STDOUT")
    assert not (tmp_path / "p").exists() or not any((tmp_path / "p").iterdir())             # aucune page produite


def test_g4_a_generator_timeout_gives_the_same_failure_record(tmp_path, monkeypatch):
    monkeypatch.setattr(D3.D2, "generate_command",
                        _failing_cmd("import sys, time; sys.stderr.write('lent'); sys.stderr.flush(); time.sleep(30)"))
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(D3.GenerationFailed, match="TIMEOUT"):
        D3.produce_pages("decoy", CFG, src, tmp_path / "work", tmp_path / "p", seed=7, timeout_s=0.5)
    rec = json.loads((tmp_path / "work" / "data3_generate_failure.json").read_text())
    assert rec["rc"] == "TIMEOUT" and rec["timeout_s"] == 0.5 and rec["stderr_tail"] == "lent"


def test_g3_g4_eval_copies_the_failure_json_keeps_the_marker_and_ends_incomplete_with_the_generate_timeout(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld3"
    out.mkdir(parents=True)
    monkeypatch.setattr(E3, "make_context", lambda *a, **k: {"ident": {"checkpoint_sha256": "ab" * 32}, "pages_root": None})
    seen = {}

    def failing(mode, cfg, src, work, pages_root, **kw):
        seen["timeout_s"] = kw["timeout_s"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "data3_generate_failure.json").write_text(json.dumps({"rc": "TIMEOUT", "stderr_tail": "x", "stdout_tail": ""}))
        raise D3.GenerationFailed("générateur : code de sortie TIMEOUT")

    monkeypatch.setattr(E3.D3, "produce_pages", failing)
    with pytest.raises(D3.GenerationFailed):
        E3.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert seen["timeout_s"] == CFG["robustness"]["timeouts_s"]["generate"] == 1500 and CFG["robustness"]["generation_estimate_s"] == 600
    assert json.loads((out / "data3_generate_failure.json").read_text())["rc"] == "TIMEOUT"
    assert (out / E3.EVAL_STARTED).is_file()
    src = (ROOT / "scripts" / "eval_ld3.py").read_text()
    assert 'status="INCOMPLETE"' in src and "except Exception as exc" in src                # GenerationFailed (RuntimeError) -> INCOMPLETE
    assert issubclass(D3.GenerationFailed, RuntimeError) and not issubclass(D3.GenerationFailed, SystemExit)


# ---------- gel 4 : H1 (échéance globale sur TOUTES les étapes), H2 (facteur moyen pondéré) ----------

@NEEDS_CARNET
def test_h1_late_entry_with_200_s_left_pip_gets_at_most_80_s_and_the_run_ends_before_9000_s(tmp_path, monkeypatch):
    lim = _b()._limits()
    assert lim["deadline_s"] == 9000 and lim["stop_margin_s"] == 120
    r = Run(tmp_path, monkeypatch, {"pip": {"dur": 80.0}}, start=9000.0 - 200.0)
    assert r.names == ["pip"] and r.timeout("pip") == 80.0                                  # min(1500, 200 - 120)
    assert r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "unzip"                  # 120 s restants : unzip n'est pas lancé
    assert r.clock[0] <= 9000.0 and 9000.0 - r.clock[0] >= lim["stop_margin_s"] - 1e-9     # la marge d'arrêt est conservée


@NEEDS_CARNET
def test_h1_deadline_already_reached_launches_no_subprocess_at_all_and_is_incomplete(tmp_path, monkeypatch):
    for start in (9000.0, 9000.0 - 120.0, 9500.0):
        (tmp_path / str(int(start))).mkdir()
        r = Run(tmp_path / str(int(start)), monkeypatch, {}, start=start)
        assert r.calls == [] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "pre"


@NEEDS_CARNET
def test_h1_every_argv_step_gets_max_0_min_cap_remaining_minus_the_stop_margin(tmp_path, monkeypatch):
    lim = _b()._limits()
    r = Run(tmp_path, monkeypatch, {"pip": {"dur": 1000.0}, "unzip": {"dur": 800.0}, "clone": {"dur": 500.0}, "checkout": {"dur": 400.0},
                                   "D": {"dur": 300.0}, "controls": {"dur": 100.0}, "CAL": {"dur": 4000.0, "estimate": 10.0}})
    to, m = r.timeouts, lim["stop_margin_s"]
    t = 0.0
    for name, cap, dur in (("pip", to["pip"], 1000.0), ("unzip", to["unzip"], 800.0), ("clone", to["clone"], 500.0),
                           ("checkout", to["clone"], 400.0), ("D", to["D"], 300.0), ("controls", to["controls"], 100.0),
                           ("CAL", to["CAL"], 4000.0)):
        assert r.timeout(name) == max(0.0, min(cap, 9000.0 - t - m)), name
        t += dur
    # CAL a consommé 4000 s : l'estimation est écartée par la porte ou EVAL reçoit restant - eval_margin_s
    if "EVAL" in r.names:
        assert r.timeout("EVAL") == 9000.0 - t - lim["eval_margin_s"]


@NEEDS_CARNET
def test_h1_the_fn_steps_are_guarded_and_the_smoke_keeps_its_fixed_caps(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 8900.0}})                                  # reste 100 s < marge : controls refusé
    assert r.names[-1] == "D" and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "controls"
    src = _b().FULL_BODY
    assert 'guard_fn("pre", s_pre)' in src and 'guard_fn("estimate", s_estimate)' in src and 'guard_fn("mille_check"' in src
    assert "marge d'arrêt" in src and "SIGKILL" in src                                      # grâce du kill et nettoyage documentés
    smoke = _b().SMOKE_BODY
    assert "FUMÉE" in smoke and "plafonds FIXES" in smoke and "step_argv" not in smoke


def test_h2_the_reduction_factor_is_weighted_by_the_reduced_zones_and_min_f_is_kept():
    rows = [_fake_page_row(p) for p in IDS]
    for i, r in enumerate(rows):
        for arm in ("M16", "M32"):
            for src in E3.SOURCES:
                f = 0.2 if i == 0 else 0.8
                r[arm][src]["reduction"] = {"zones": 3, "reduced": 1, "reduced_share": 1 / 3, "mean_f_reduced": f, "min_f": f}
    rep = E3.build_report(rows, IDS, CFG, {}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False)
    red = rep["zone_reduction"]["M32"]["Z-PP-blocs"]
    assert red["reduced"] == 10 and red["mean_f_reduced"] == pytest.approx(0.74) and red["min_f"] == 0.2     # 1 x 0,2 et 9 x 0,8
    assert red["zones"] == 30 and red["reduced_share"] == pytest.approx(1 / 3)
    # pondération par le nombre de zones réduites de chaque page
    unequal = E3.reduction_total([{"zones": 5, "reduced": 3, "mean_f_reduced": 0.5, "min_f": 0.4},
                                  {"zones": 5, "reduced": 1, "mean_f_reduced": 0.9, "min_f": 0.9},
                                  {"zones": 4, "reduced": 0, "mean_f_reduced": None, "min_f": None}])
    assert unequal["mean_f_reduced"] == pytest.approx((3 * 0.5 + 1 * 0.9) / 4) and unequal["min_f"] == 0.4 and unequal["zones"] == 14
    assert E3.reduction_total([{"zones": 2, "reduced": 0, "mean_f_reduced": None, "min_f": None}])["mean_f_reduced"] is None
