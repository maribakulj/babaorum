"""LD-4 : tests légers (docs/END_TO_END_LD4_PROPOSITION.txt). Petits tableaux synthétiques : aucun modèle réel, aucun détecteur, aucun générateur,
aucune page réelle. Les tests de carnets sont ignorés si le constructeur est absent du kit Kaggle."""
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
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
import eval_ld3 as E3  # noqa: E402
import eval_ld4 as E4  # noqa: E402
import train_ld2 as TD  # noqa: E402
import train_stage_s_v2 as T2  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

CARNETS = ROOT / "scripts" / "stage_s_notebooks"
NB = CARNETS / "ld4"
DOC = ROOT / "docs" / "END_TO_END_LD4_PROPOSITION.txt"
CFG = E4.load_cfg()
IDS = [f"mf_00{i}" for i in range(70, 80)]
NEEDS_CARNET = pytest.mark.skipif(not (CARNETS / "build_kaggle_ld4.py").is_file() or not (NB / "kaggle_ld4.ipynb").is_file(),
                                  reason="build_kaggle_ld4.py / ld4/*.ipynb absents du kit (Kaggle)")
NEEDS_DOC = pytest.mark.skipif(not DOC.is_file(), reason="docs/ absent du kit (Kaggle)")


def _norm(t):
    return re.sub(r"\s+", " ", t)


def _b():
    import build_kaggle_ld4 as B4
    return B4


# ---------- jouets ----------

def line(y, x0=100.0, x1=400.0, n=8):
    return np.stack([np.linspace(x0, x1, n), np.full(n, float(y))], axis=-1)


def gt(i, y, x0=100.0, x1=400.0):
    return {"id": f"g{i}", "baseline": line(y, x0, x1, 2), "neg": 12.0, "pos": 4.0, "text": "ab", "legibility": "readable"}


def ln(y, x0=100.0, x1=400.0, h=16.0, zone="B3", query=0, score=1.0):
    return {"score": score, "zone": zone, "query": query, "pts": line(y, x0, x1), "h": h, "neg": 0.75 * h, "pos": 0.25 * h}


# ---------- fusion (§2) ----------

def test_fuse_keeps_all_detr_lines_and_adds_only_the_b3_lines_no_detr_line_covers():
    detr = [ln(60.0, zone=0, query=0, score=0.9), ln(200.0, zone=0, query=1, score=0.8)]
    covered = ln(62.0, query=0)                       # sur la 1re ligne DETR
    free = ln(130.0, query=1)                         # loin de toute ligne DETR
    free_twin = ln(130.0, query=2)                    # jumelle d'une ligne B3 ajoutée : PAS de NMS entre lignes B3
    fusion, added, cov = E4.fuse(detr, [covered, free, free_twin])
    assert fusion[:2] == detr and fusion[2:] == [free, free_twin] and added == [free, free_twin] and cov == [covered]
    assert E4.fuse(detr, []) == (detr, [], []) and E4.fuse([], [free])[0] == [free]
    f2, a2, c2 = E4.fuse(detr, [covered])
    assert f2 == detr and a2 == [] and c2 == [covered]


def test_fuse_boundaries_follow_the_nms_rule_t_at_most_half_the_mean_height_and_o_strictly_above_half_the_shorter_line():
    k = ln(60.0, x0=100.0, x1=400.0, h=16.0, zone=0)
    # t : décalage vertical 8 = 0,5 x (16 + 16)/2 -> recouverte (<=) ; 8,01 -> non recouverte
    assert E4.fuse([k], [ln(68.0)])[2] and not E4.fuse([k], [ln(68.01)])[2]
    # o : b = [1, 201] contre k = [100, 400] : o = 101 > 0,5 x min(200, 300) = 100 -> recouverte ; b = [0, 200] : o = 100, pas > 100
    assert E4.fuse([k], [ln(60.0, x0=1.0, x1=201.0)])[2] and not E4.fuse([k], [ln(60.0, x0=0.0, x1=200.0)])[2]
    # hauteurs : la moyenne de b.h et k.h compte (b : 40, k : 16 -> 0,5 x 28 = 14)
    assert E4.fuse([k], [ln(74.0, h=40.0)])[2] and not E4.fuse([k], [ln(75.0, h=40.0)])[2]


def test_fuse_uses_the_unchanged_overlap_rule_with_b3_as_candidate_and_detr_as_kept(monkeypatch):
    calls = []
    real = LD._overlap_rule
    monkeypatch.setattr(LD, "_overlap_rule", lambda a, b, ha, hb: calls.append((a, b, ha, hb)) or real(a, b, ha, hb))
    k, b = ln(60.0, h=20.0, zone=0), ln(300.0, h=12.0)
    E4.fuse([k], [b])
    assert len(calls) == 1 and calls[0][0] is b["pts"] and calls[0][1] is k["pts"] and calls[0][2:] == (12.0, 20.0)
    assert C2.sha256_file(ROOT / "src/babaorum/experimental/line_detr.py").startswith("7715a8e3")


# ---------- mesure ----------

def test_score_geometry_equals_the_ld2_numbers_and_geometry_status_does_not_change_the_result():
    page = {"gt_lines": [gt(0, 60.0), gt(1, 100.0), gt(2, 140.0)]}
    kept = [ln(60.0, zone=0, query=0), ln(101.0, zone=0, query=1), ln(500.0, zone=0, query=2)]
    s, matched, mp = E4.score_geometry(page, kept)
    raws = [(0, None, {"hyp": {}, "hyp_none": {}})]
    ref = TD.score_lines(page, kept, raws)
    assert all(s[k] == ref[k] for k in ("n_pred", "n_gt", "precision", "recall", "splits", "merges", "precision_none"))
    assert matched == {"g0", "g1"} and mp == {0, 1} and s["matched"] == 2
    gt_plain = [GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]), baseline_source="ground_truth")
                for g in page["gt_lines"]]                                                       # construction d'evaluate_full_page (sans statut)
    pred_plain = [PredictedLine(id=f"p{i:05d}", baseline=k["pts"].tolist(), baseline_source="predicted") for i, k in enumerate(kept)]
    res = evaluate_line_geometry(gt_plain, pred_plain)
    assert (res.recall, res.precision, res.matched_count) == (s["recall"], s["precision"], s["matched"])
    zero, zm, zp = E4.score_geometry(page, [])
    assert (zero["precision"], zero["recall"], zero["n_pred"], zm, zp) == (0.0, 0.0, 0, set(), set())


def test_the_b3_lines_must_reproduce_the_metrics_after_exactly_or_the_run_is_refused():
    page = {"gt_lines": [gt(0, 60.0), gt(1, 100.0)]}
    lines = [ln(60.0), ln(100.0), ln(500.0)]
    s, _m, _p = E4.score_geometry(page, lines)
    after = {"recall": s["recall"], "precision": s["precision"], "matched": s["matched"], "n_pred": s["n_pred"], "n_gt": 2}
    E4.check_b3_exact(s, after)
    for key, val in (("recall", 0.5), ("precision", 0.5), ("matched", 1), ("n_pred", 4)):
        with pytest.raises(SystemExit, match="exactement metrics.after"):
            E4.check_b3_exact(s, dict(after, **{key: val}))
    empty, _a, _b = E4.score_geometry(page, [])
    E4.check_b3_exact(empty, {"recall": 0.0, "precision": None, "matched": 0, "n_pred": 0, "n_gt": 2})
    with pytest.raises(SystemExit):
        E4.check_b3_exact(empty, {"recall": 0.0, "precision": 0.3, "matched": 0, "n_pred": 2, "n_gt": 2})


class FakeB3(torch.nn.Module):
    """Faux modèle B3 : mêmes attributs que StageSModel pour l'extraction d'evaluate_full_page ; extraction et raffinement sont remplacés."""
    K = 8
    arm = "B3"

    def __init__(self):
        super().__init__()
        self.enc_det = lambda image: torch.zeros(1, 4, 5, 6)
        self.head_d = lambda feats: (torch.zeros(1, 1, 5, 6), None, None)
        self.head_g = object()
        self.reader = None


def test_b3_lines_follow_the_extraction_and_refinement_path_of_evaluate_full_page(monkeypatch):
    cfg_e2e5 = json.loads((ROOT / "config" / "stage_s_smoke_v2.json").read_text())
    polys = [[[100.0, 60.0], [400.0, 60.0]], [[100.0, 100.0], [400.0, 100.0]]]
    seen = {"extract": [], "refine": []}

    def fake_extract(prob, stride, threshold, min_cells):
        seen["extract"].append((tuple(prob.shape), stride, threshold, min_cells))
        return polys + [[[1.0, 1.0]]]                                           # un polyligne à 1 point : écarté (len >= 2)

    def fake_refine(features, extracted, head_g, head_d, K):
        seen["refine"].append((len(extracted), K))
        return {"baseline": torch.tensor([line(60.0).tolist(), line(100.0).tolist()]), "neg": torch.tensor([12.0, 10.0]),
                "pos": torch.tensor([4.0, 6.0])}

    for mod in (T2, E4):
        monkeypatch.setattr(mod, "extract_polylines", fake_extract)
        monkeypatch.setattr(mod, "refine_lines", fake_refine)
    page = {"image": np.ones((20, 24), np.float32), "page_id": "p", "alphabet": ["<blank>"],
            "lines": [{"id": "g0", "baseline": polys[0], "normal_negative_px": 12.0, "normal_positive_px": 4.0, "text": "", "legibility": "x"},
                      {"id": "g1", "baseline": polys[1], "normal_negative_px": 10.0, "normal_positive_px": 6.0, "text": "", "legibility": "x"}]}
    model = FakeB3()
    lines = E4.b3_lines(model, page, cfg_e2e5, torch.device("cpu"))
    full = T2.evaluate_full_page(model, page, cfg_e2e5, torch.device("cpu"))
    assert seen["extract"][0] == seen["extract"][1] == ((5, 6), cfg_e2e5["extraction"]["stride"], cfg_e2e5["extraction"]["threshold"],
                                                         cfg_e2e5["extraction"]["min_cells"])
    assert seen["refine"][0] == seen["refine"][1] == (2, 8)
    assert [l["h"] for l in lines] == [16.0, 16.0] and [l["zone"] for l in lines] == ["B3", "B3"] and lines[0]["pts"].shape == (8, 2)
    score, _m, _p = E4.score_geometry({"gt_lines": [gt(0, 60.0), gt(1, 100.0)]}, lines)
    E4.check_b3_exact(score, full["metrics"]["after"])                                   # même appariement que evaluate_full_page
    monkeypatch.setattr(E4, "extract_polylines", lambda *a, **k: [])
    assert E4.b3_lines(model, page, cfg_e2e5, torch.device("cpu")) == []


# ---------- critère (§5) ----------

def _row(pid, p_d, r_d, p_f, r_f, b3=(0.71, 0.923)):
    def sc(P, R):
        return {"n_pred": 5, "n_gt": 5, "precision": P, "recall": R, "precision_none": False, "matched": 5, "splits": 0, "merges": 0,
                "edge": {"bins": LB.page_counts([("0-8", True), ("0-8", False), (">32", True)], []), "gt_hors_zone": 0, "gt_in_zones": 3}}
    f = dict(sc(p_f, r_f), b3_total=4, b3_covered=3, b3_added=1, b3_added_matched=1, b3_added_unmatched=0)
    return {"page": pid, "DETR": sc(p_d, r_d), "B3": sc(b3[0], b3[1]), "FUSION": f, "n_zones": 3,
            "b3_after": {"recall": b3[1], "precision": b3[0], "matched": 5, "n_pred": 6, "n_gt": 5}, "seconds": {"total": 1.0}}


def test_principal_is_the_ld2_criterion_unchanged_for_the_fusion_against_the_in_run_b3():
    rows = [_row(p, 0.97, 0.91, 0.88, 0.95) for p in IDS]
    ok = E4.principal_criterion(rows, IDS, CFG)
    ref = {p: (0.71, 0.923) for p in IDS}
    assert ok == C2.criterion([dict(r["FUSION"], page=r["page"]) for r in rows], ref, IDS, CFG)
    assert ok["verdict"] == "CRITERE_REMPLI" and ok["a"]["met"] and ok["b"]["met"]
    low_p = E4.principal_criterion([_row(p, 0.97, 0.91, 0.79, 0.95) for p in IDS], IDS, CFG)           # (a) : P < 0,80
    assert not low_p["a"]["met"] and low_p["verdict"] == "CRITERE_NON_REMPLI"
    low_r = E4.principal_criterion([_row(p, 0.97, 0.91, 0.90, 0.89) for p in IDS], IDS, CFG)           # R < 0,90 et ΔR < -0,02
    assert not low_r["a"]["met"] and not low_r["b"]["met"]
    assert CFG["criterion"]["a"] == {"precision_min": 0.8, "recall_min": 0.9, "pages": 10} and CFG["criterion"]["b"] == {"dp_low_gt": 0.0, "dr_low_min": -0.02}
    assert CFG["bootstrap"] == {"draws": 10000, "seed": 20261018, "percentiles": [2.5, 97.5]}


def test_the_decoy_reference_replaces_b3_values_only_in_the_smoke_and_the_reading_depends_only_on_the_verdict():
    rows = [_row(p, 0.97, 0.91, 0.88, 0.95) for p in IDS]
    dec = E4.principal_criterion(rows, IDS, CFG, smoke_decoy=True)
    assert dec["b"]["reference_per_page"][IDS[0]] == list(C2.page_pr(CFG["decoy_b3"]))
    assert E4.principal_criterion(rows, IDS, CFG)["b"]["reference_per_page"][IDS[0]] == [0.71, 0.923]
    assert E4.reading_from_principal({"verdict": "CRITERE_REMPLI"}, CFG) == "FUSION_CANDIDATE_GEOMETRIE"
    assert E4.reading_from_principal({"verdict": "CRITERE_NON_REMPLI", "a": {"met": True}}, CFG) == "O5_SUR_DECISION_PA"
    rep_hi = E4.build_report(rows, IDS, CFG, {}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False)
    rep_lo = E4.build_report([_row(p, 0.97, 0.91, 0.70, 0.95) for p in IDS], IDS, CFG, {}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False)
    assert rep_hi["reading"] == "FUSION_CANDIDATE_GEOMETRIE" and rep_lo["reading"] == "O5_SUR_DECISION_PA"
    assert rep_hi["b3_reference"]["decoy_reference_used"] is False


def test_the_descriptive_fusion_minus_detr_uses_the_same_bootstrap_and_decides_nothing():
    rows = [_row(p, 0.97, 0.91, 0.88, 0.95) for p in IDS]
    d = E4.descriptive_fusion_vs_detr(rows, IDS, CFG)
    assert d["descriptive_only"] and abs(d["bootstrap"]["dR"]["mean"] - 0.04) < 1e-12
    assert abs(d["bootstrap"]["dP"]["mean"] + 0.09) < 1e-12 and d["FUSION"]["R"] == pytest.approx(0.95) and d["DETR"]["P"] == pytest.approx(0.97)
    rep = E4.build_report(rows, IDS, CFG, {}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False)
    assert rep["descriptive_fusion_vs_detr"]["bootstrap"]["draws"] == 10000 and "verdict" not in rep["descriptive_fusion_vs_detr"]


def test_the_report_has_the_reserves_in_the_section_7_order_the_prediction_copied_counts_edge_tables_and_identity():
    rows = [_row(p, 0.97, 0.91, 0.88, 0.95) for p in IDS]
    ident = {"checkpoint_sha256": "e5afd143" + "0" * 56, "alphabet": ["<blank>"], "threshold": 0.6}
    rep = E4.build_report(rows, IDS, CFG, {"mode": "real"}, ident, False, E4.run_identity(CFG))
    assert list(rep)[:3] == ["schema", "checkpoint_sha256", "reserves"] and rep["checkpoint_sha256"] == ident["checkpoint_sha256"]
    r = rep["reserves"]
    assert r[:2] == [C2.RESERVES[6], C2.RESERVES[7]] and r[2:] == CFG["reserves"]
    assert "0,934" in r[0] and "0,843" in r[1] and r[2].startswith("Même distribution") and "B3 réévalué" in r[3] and "chaîne" in r[4] and "CER non mesuré" in r[5]
    assert rep["prediction"] == {"source": CFG["prediction"]["source"], "lines": CFG["prediction"]["lines"]}
    assert set(rep["outputs"]) == {"DETR", "B3", "FUSION"} and rep["outputs"]["FUSION"]["pages"] == 10
    assert rep["fusion_counts"] == {"b3_total": 40, "b3_covered": 30, "b3_added": 10, "b3_added_matched": 10, "b3_added_unmatched": 0}
    assert set(rep["edge_totals"]) == {"DETR", "B3", "FUSION"} and rep["edge_totals"]["FUSION"]["0-8"] == {"gt_n": 20, "gt_matched": 10}
    assert rep["run_identity"]["files_sha256"]["scripts/eval_ld4.py"] == C2.sha256_file(ROOT / "scripts/eval_ld4.py")
    assert rep["pages"] == IDS and len(rep["per_page"]) == 10
    json.dumps(rep)


@NEEDS_DOC
def test_the_prediction_and_reserves_are_in_the_proposition_verbatim_and_the_sha_is_the_pre_registered_one():
    doc = _norm(DOC.read_text())
    for l_ in CFG["prediction"]["lines"] + CFG["reserves"][:4]:
        assert _norm(l_) in doc, l_[:60]
    assert C2.sha256_file(DOC).startswith("e04b67bb9d96539e")


# ---------- pages DATA-3 avec la config de LD-4 ----------

def test_ld4_pages_are_mf_0070_to_0079_with_the_new_seed_and_the_extended_forbidden_regex():
    assert D3.real_names(CFG) == IDS and CFG["pages"]["seed"] == 20261028 and CFG["pages"]["real_first"] == 70
    assert D3.check_names(IDS, "real", CFG) == IDS and D3.check_names(["decoy_0000"], "decoy", CFG)
    for bad in ["mf_0000", "mf_0049", "mf_0050", "mf_0059", "mf_0060", "mf_0069", "mfn_0000"]:
        with pytest.raises(SystemExit, match="interdits"):
            D3.check_names(IDS[:-1] + [bad], "real", CFG)
    with pytest.raises(SystemExit, match="interdits"):
        D3.check_names([f"mf_00{i}" for i in range(60, 70)], "real", CFG)                 # les pages de LD-3 sont interdites à LD-4
    assert D3.check_seed(20261028, "real", CFG) and D3.check_seed(1, "decoy", CFG)
    with pytest.raises(SystemExit, match="exige la graine"):
        D3.check_seed(20261026, "real", CFG)                                              # la graine de LD-3 est refusée
    with pytest.raises(SystemExit, match="graine réelle"):
        D3.check_seed(20261028, "decoy", CFG)
    assert CFG["generator"] == json.loads((ROOT / "config" / "ld3.json").read_text())["generator"]
    assert CFG["pages"]["cal_name_regex"] == "^mf_00[0-4][0-9]$"


def test_the_real_mode_is_refused_without_the_eval_flag_and_decoy_names_never_use_the_real_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(D3, "generate", lambda *a, **k: pytest.fail("le générateur ne doit pas démarrer"))
    with pytest.raises(SystemExit, match="phase EVAL"):
        D3.produce_pages("real", CFG, tmp_path, tmp_path / "w", tmp_path / "p")
    with pytest.raises(SystemExit, match="graine réelle"):
        D3.produce_pages("decoy", CFG, tmp_path, tmp_path / "w", tmp_path / "p", seed=20261028)
    for bad_ids in ([f"mf_00{i}" for i in range(60, 70)], ["decoy_0000"], IDS[::-1], IDS[:9]):
        with pytest.raises(SystemExit):
            E3.check_eval_ids(bad_ids, "real", CFG)
    assert E3.check_eval_ids(IDS, "real", CFG) == IDS and E3.check_eval_ids(["decoy_0000"], "smoke", CFG) == ["decoy_0000"]
    for bad in ("mf_0060", "mf_0070", "decoy_0000", "mfn_0001"):
        with pytest.raises(SystemExit):
            E3.check_cal_page(bad, CFG)
    assert E3.check_cal_page("mf_0045", CFG) == "mf_0045"


# ---------- identités, pins, marqueur ----------

def test_the_pins_cover_ld1_ld2_ld3_and_ldo1_files_and_a_changed_one_is_refused():
    pins = CFG["frozen_files_sha256"]
    for rel in ("scripts/train_ld2.py", "scripts/ld2_core.py", "config/ld2.json", "src/babaorum/experimental/line_detr.py", "scripts/train_line_detr.py",
                "scripts/eval_ld3.py", "scripts/data3_generate.py", "config/ld3.json", "scripts/diag_ld_o1.py", "config/ld_o1.json",
                "scripts/ld1b_border_diag.py", "src/babaorum/line_geometry.py"):
        assert pins[rel] == C2.sha256_file(ROOT / rel), rel
    assert len(E4.E3.check_code_identity(CFG)) == len(pins) + 8
    bad = json.loads(json.dumps(CFG))
    bad["frozen_files_sha256"]["scripts/eval_ld3.py"] = "0" * 64
    with pytest.raises(SystemExit, match="eval_ld3"):
        E3.check_code_identity(bad)
    assert CFG["ld2"]["checkpoint_sha256"].startswith("e5afd143") and CFG["e2e5"]["checkpoint_sha256"].startswith("ef5e8268")
    assert CFG["margin_h"] == 0.5 and CFG["threshold"] == 0.6


def _eval_args(tmp_path, **kw):
    base = dict(smoke=True, decoy_reference=True, work=str(tmp_path / "w"), gen_src=str(tmp_path), extra_libs=None, e2e4_report=None,
                ld2_dir=None, e2e5_checkpoint=None, cache_dir=str(tmp_path / "hf"))
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_eval_refuses_report_marker_and_a_decoy_reference_outside_the_smoke(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld4"
    out.mkdir(parents=True)
    monkeypatch.setattr(E4, "make_context", lambda *a, **k: pytest.fail("aucun modèle ne doit être chargé"))
    (out / "ld4_report.json").write_text("{}")
    with pytest.raises(SystemExit, match="existe déjà"):
        E4.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    (out / "ld4_report.json").unlink()
    (out / E4.EVAL_STARTED).write_text("{}")
    with pytest.raises(SystemExit, match="décision explicite de pA"):
        E4.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert (out / E4.EVAL_STARTED).is_file()
    (out / E4.EVAL_STARTED).unlink()
    with pytest.raises(SystemExit, match="fumée"):
        E4.phase_EVAL(_eval_args(tmp_path, smoke=False, decoy_reference=True), CFG, None, out, tmp_path)


def test_the_marker_is_created_before_generation_and_the_first_page_and_an_interruption_cannot_be_restarted(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld4"
    out.mkdir(parents=True)
    monkeypatch.setattr(E4, "make_context", lambda *a, **k: {"ident": {"checkpoint_sha256": "ab" * 32}})
    seen = []

    def fake_produce(mode, cfg, src, work, pages_root, **kw):
        seen.append(("produce", (out / E4.EVAL_STARTED).is_file(), mode, kw["called_from_eval_ld3"], kw["timeout_s"]))
        return {"mode": mode, "seed": 1, "names": ["decoy_0000", "decoy_0001"], "files_sha256": {}, "generator": {}, "seconds_total": 0,
                "overlap": {"status": "OK", "blocking": False}}

    def fake_page(pid, root, ctx):
        seen.append(("page", pid, (out / E4.EVAL_STARTED).is_file()))
        if pid == "decoy_0001":
            raise RuntimeError("interruption")
        return _row(pid, 0.9, 0.9, 0.9, 0.9)

    monkeypatch.setattr(E4.D3, "produce_pages", fake_produce)
    monkeypatch.setattr(E4, "evaluate_page_ld4", fake_page)
    with pytest.raises(RuntimeError, match="interruption"):
        E4.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert seen == [("produce", True, "decoy", True, 1500), ("page", "decoy_0000", True), ("page", "decoy_0001", True)]
    started = json.loads((out / E4.EVAL_STARTED).read_text())
    assert started["checkpoint_sha256"] == "ab" * 32 and started["mode"] == "decoy" and not (out / "ld4_report.json").exists()
    with pytest.raises(SystemExit, match="décision explicite de pA"):
        E4.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert len(seen) == 3


def test_a_generation_failure_copies_the_failure_json_and_a_blocking_overlap_or_wrong_ids_stop_before_any_page(tmp_path, monkeypatch):
    out = tmp_path / "out" / "ld4"
    out.mkdir(parents=True)
    monkeypatch.setattr(E4, "make_context", lambda *a, **k: {"ident": {"checkpoint_sha256": "ab" * 32}})
    monkeypatch.setattr(E4, "evaluate_page_ld4", lambda *a, **k: pytest.fail("aucune page ne doit être évaluée"))

    def failing(mode, cfg, src, work, pages_root, **kw):
        work.mkdir(parents=True, exist_ok=True)
        (work / "data3_generate_failure.json").write_text(json.dumps({"rc": 3}))
        raise D3.GenerationFailed("x")

    monkeypatch.setattr(E4.D3, "produce_pages", failing)
    with pytest.raises(D3.GenerationFailed):
        E4.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)
    assert json.loads((out / "data3_generate_failure.json").read_text()) == {"rc": 3}
    for names, overlap, msg in ((["decoy_0000"], {"status": "OVERLAP_PAGE", "blocking": True}, "recouvrement bloquant"),
                                (["mf_0060"], {"status": "OK", "blocking": False}, "leurres")):
        (out / E4.EVAL_STARTED).unlink()
        monkeypatch.setattr(E4.D3, "produce_pages", lambda *a, _n=names, _o=overlap, **k: {
            "mode": "decoy", "seed": 1, "names": _n, "files_sha256": {}, "generator": {}, "seconds_total": 0, "overlap": _o})
        with pytest.raises(SystemExit, match=msg):
            E4.phase_EVAL(_eval_args(tmp_path), CFG, None, out, tmp_path)


def test_page_frames_must_agree_between_detr_and_b3(tmp_path, monkeypatch):
    page = {"factor": 1.5, "image": np.ones((10, 10), np.float32), "gt_lines": [], "blocks": []}
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: page)
    monkeypatch.setattr(E4, "load_scaled_page", lambda root, pid, h: {"factor": 1.6, "image": np.ones((10, 10), np.float32)})
    ctx = {"threshold": 0.6, "alphabet": [], "device": None, "cfg_e2e5": {"scale": {"target_median_height_px": 32.0}}}
    with pytest.raises(SystemExit, match="échelles différentes"):
        E4.evaluate_page_ld4("p", Path("."), ctx)


def test_evaluate_page_runs_detr_b3_and_fusion_on_a_toy_page(monkeypatch):
    g0, g1, g2 = gt(0, 60.0), gt(1, 100.0), gt(2, 140.0)
    page = {"factor": 1.0, "image": np.ones((200, 600), np.float32), "gt_lines": [g0, g1, g2],
            "blocks": [{"lines": [{"id": "g0"}, {"id": "g1"}, {"id": "g2"}], "box": (50.0, 20.0, 450.0, 170.0)}]}
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: page)
    monkeypatch.setattr(E4, "load_scaled_page", lambda root, pid, h: {"factor": 1.0, "image": np.ones((200, 600), np.float32)})
    pp = [(0, (50.0, 20.0, 450.0, 170.0))]
    monkeypatch.setattr(E4.TL, "pp_zones", lambda *a, **k: pp)
    detr_pts = [line(60.0), line(100.0)]
    monkeypatch.setattr(E4.TD, "zone_raws", lambda *a, **k: [(0, None, {})])
    monkeypatch.setattr(E4.TL, "candidates_for_threshold", lambda raws, thr: [
        {"score": 0.9, "zone": 0, "query": i, "pts": p, "h": 16.0} for i, p in enumerate(detr_pts)])
    b3 = [ln(61.0), ln(140.0), ln(300.0, query=2)]
    b3_score, _m, _p = E4.score_geometry(page, b3)
    monkeypatch.setattr(E4, "b3_lines", lambda *a, **k: b3)
    after = {"recall": b3_score["recall"], "precision": b3_score["precision"], "matched": b3_score["matched"], "n_pred": 3, "n_gt": 3}
    monkeypatch.setattr(E4.T2, "evaluate_full_page", lambda *a, **k: {"metrics": {"after": after}})
    ctx = {"threshold": 0.6, "alphabet": [], "device": None, "model": object(), "b3_model": object(), "cache": None, "L0": None, "det_fn": None,
           "cfg_e2e5": {"scale": {"target_median_height_px": 32.0}}}
    row = E4.evaluate_page_ld4("p", Path("."), ctx)
    assert row["DETR"]["n_pred"] == 2 and row["DETR"]["recall"] == pytest.approx(2 / 3)
    assert row["FUSION"]["n_pred"] == 4 and row["FUSION"]["recall"] == 1.0 and row["FUSION"]["b3_covered"] == 1 and row["FUSION"]["b3_added"] == 2
    assert row["FUSION"]["b3_added_matched"] == 1 and row["FUSION"]["b3_added_unmatched"] == 1 and row["FUSION"]["b3_total"] == 3
    assert row["B3"]["recall"] == b3_score["recall"] and row["b3_after"]["n_pred"] == 3
    assert row["FUSION"]["edge"]["bins"]["16-32"]["gt_n"] + row["FUSION"]["edge"]["bins"][">32"]["gt_n"] == 3
    monkeypatch.setattr(E4, "b3_lines", lambda *a, **k: b3[:2])                                     # B3 ≠ evaluate_full_page : refus
    with pytest.raises(SystemExit, match="exactement metrics.after"):
        E4.evaluate_page_ld4("p", Path("."), ctx)


# ---------- carnets ----------

def _code(name):
    nb = json.loads((NB / name).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@NEEDS_CARNET
def test_the_notebooks_on_disk_equal_the_builder_have_no_trace_of_ld3_and_the_deadline_is_5400():
    B4 = _b()
    assert json.loads((NB / "kaggle_ld4.ipynb").read_text()) == B4.build_notebook()
    assert json.loads((NB / "kaggle_ld4_smoke.ipynb").read_text()) == B4.build_smoke_notebook()
    assert json.loads((NB / "full_kernel" / "notebook.ipynb").read_text()) == B4.build_notebook()
    assert json.loads((NB / "smoke_kernel" / "notebook.ipynb").read_text()) == B4.build_smoke_notebook()
    assert json.loads((NB / "kit_files.json").read_text()) == B4.kit_files()
    for n in ("kaggle_ld4.ipynb", "kaggle_ld4_smoke.ipynb"):
        code = _code(n)
        assert (NB / n).stat().st_size < 300 * 1024 and "ld3" not in code and "LD-3" not in code
        for bad in ("glob(", "rglob", "listdir", "scandir", "os.walk", "iterdir", "mf_005", "mf_006", "mf_007", "mfn_"):
            assert bad not in code, (n, bad)
    full, smoke = _code("kaggle_ld4.ipynb"), _code("kaggle_ld4_smoke.ipynb")
    assert "'deadline_s': 5400" in full and "eval_ld4.py" in full and "--smoke" not in full and "--decoy-reference" not in full
    assert '"--smoke", "--decoy-reference"' in smoke and '"decoy_0000"' in smoke and "5400" in smoke and "10800" in smoke
    full_meta = json.loads((NB / "full_kernel" / "kernel-metadata.json").read_text())
    assert full_meta["id"] == "mabaku/babaorum-ld4-run" and full_meta["enable_gpu"] is True and "mabaku/babaorum-ld4-kit" in full_meta["dataset_sources"]
    smoke_meta = json.loads((NB / "smoke_kernel" / "kernel-metadata.json").read_text())
    assert smoke_meta["id"] == "mabaku/babaorum-ld4-smoke" and smoke_meta["enable_gpu"] is False
    for rel in json.loads((NB / "kit_files.json").read_text()):
        assert (ROOT / rel).is_file(), rel


def _data2_common():
    spec = importlib.util.spec_from_file_location("data2_common_ld4test", ROOT / "scripts" / "data2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Run:
    """Exécute le vrai FULL_BODY (texte LD-3 -> LD-4) avec un faux run_step et une fausse horloge."""

    def __init__(self, tmp_path, monkeypatch, plan, cuda=True, start=0.0):
        B4 = _b()
        self.calls, self.clock, self.plan = [], [start], plan
        self.out = tmp_path / "work"
        self.out.mkdir()
        self.C = _data2_common()
        monkeypatch.setattr(self.C, "run_step", self.run_step)
        tree = ast.parse(B4.cell("full"))
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
        ns["RUN"], ns["CAL_OUT"] = self.out / "ld4", self.out / "ld4_cal"
        argv_builder = B4._t(B4.L3._argv_builder("cuda"))
        exec(compile(argv_builder, "<_argv_builder>", "exec"), ns)
        for k, v in (("BUNDLE_DIR", tmp_path / "bundle"), ("SRC", tmp_path / "src"), ("LIBS", tmp_path / "libs"), ("WORK", tmp_path / "wk")):
            ns[k] = v
        ns["d_mille_check"] = lambda: None
        exec(compile(B4._t(B4.L3.FULL_BODY), "<FULL_BODY>", "exec"), ns)
        self.ns = ns

    def run_step(self, name, argv, timeout_s, logs_dir, cwd=None, env=None, kill_grace_s=10.0):
        p = {"rc": 0, "dur": 1.0, "estimate": 100.0, **self.plan.get(name, {})}
        self.calls.append((name, timeout_s, list(argv)))
        self.clock[0] += p["dur"]
        if name == "CAL" and p["rc"] == 0:
            (self.out / "ld4_cal").mkdir(parents=True, exist_ok=True)
            (self.out / "ld4_cal" / "ld4_cal.json").write_text(json.dumps({"estimate_s": p["estimate"], "page": "mf_0045"}))
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
def test_the_full_run_order_caps_the_5400_s_deadline_and_the_estimation_gate(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"pip": {"dur": 100.0}, "CAL": {"dur": 200.0, "estimate": 900.0}, "D": {"dur": 50.0}})
    assert r.exit["status"] == "DONE" and r.names == ["pip", "unzip", "clone", "checkout", "D", "controls", "CAL", "EVAL"]
    lim, to = r.limits, r.timeouts
    assert lim["deadline_s"] == 5400 and to["generate"] == 1500 and lim["generation_estimate_s"] == 600
    assert r.timeout("EVAL") == 5400 - (100.0 + 3 + 50.0 + 1 + 200.0) - lim["eval_margin_s"]
    est = json.loads((r.out / "ld4_eval_estimate.json").read_text())
    assert est["estimate_s"] == 900.0 * 1.5 + 600
    argv = dict((c[0], c[2]) for c in r.calls)["EVAL"]
    assert argv[1].endswith("eval_ld4.py") and "--smoke" not in argv and argv[argv.index("--phase") + 1] == "EVAL"
    (tmp_path / "g").mkdir()
    gate = Run(tmp_path / "g", monkeypatch, {"CAL": {"estimate": 3200.0}})                          # 3200 x 1,5 + 600 > 5400 - 7 - 300
    assert "EVAL" not in gate.names and gate.exit["status"] == "INCOMPLETE" and gate.exit["step"] == "estimate"


@NEEDS_CARNET
def test_every_argv_step_is_bounded_by_the_global_deadline_and_a_reached_deadline_launches_nothing(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"pip": {"dur": 80.0}}, start=5400.0 - 200.0)
    assert r.names == ["pip"] and r.timeout("pip") == 80.0 and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "unzip" and r.clock[0] <= 5400.0
    for i, start in enumerate((5400.0, 5280.0, 6000.0)):
        (tmp_path / str(i)).mkdir()
        z = Run(tmp_path / str(i), monkeypatch, {}, start=start)
        assert z.calls == [] and z.exit["status"] == "INCOMPLETE" and z.exit["step"] == "pre"
    (tmp_path / "c").mkdir()
    c = Run(tmp_path / "c", monkeypatch, {"CAL": {"rc": "TIMEOUT"}})
    assert c.names[-1] == "CAL" and c.exit["status"] == "INCOMPLETE"


@NEEDS_CARNET
def test_the_smoke_cell_uses_cpu_one_decoy_page_and_the_decoy_reference_flag(tmp_path, monkeypatch):
    B4 = _b()
    src = B4.cell("smoke")
    assert '"cpu"' in src and "--decoy-reference" in src and "decoy_0000" in src and "tests/test_ld4.py" in src and "test_ld3" not in src
    assert B4.SMOKE_CAL_S == 5400 and B4.SMOKE_EVAL_S == 10800


# ---------- gel 2 : K1 no_grad, K2 déterminisme, K3 note de génération ----------

def test_k1_b3_lines_runs_inside_no_grad_like_evaluate_full_page(monkeypatch):
    cfg_e2e5 = json.loads((ROOT / "config" / "stage_s_smoke_v2.json").read_text())
    seen = {}

    class Spy(FakeB3):
        def __init__(self):
            super().__init__()
            self.enc_det = lambda image: seen.setdefault("enc", torch.is_grad_enabled()) and torch.zeros(1, 4, 5, 6) or torch.zeros(1, 4, 5, 6)
            self.head_d = lambda feats: (torch.zeros(1, 1, 5, 6), None, None)

    def fake_refine(features, extracted, head_g, head_d, K):
        seen["refine"] = torch.is_grad_enabled()
        return {"baseline": torch.tensor([line(60.0).tolist()]), "neg": torch.tensor([12.0]), "pos": torch.tensor([4.0])}

    monkeypatch.setattr(E4, "extract_polylines", lambda *a, **k: [[[100.0, 60.0], [400.0, 60.0]]])
    monkeypatch.setattr(E4, "refine_lines", fake_refine)
    page = {"image": np.ones((20, 24), np.float32)}
    assert torch.is_grad_enabled()
    E4.b3_lines(Spy(), page, cfg_e2e5, torch.device("cpu"))
    assert seen == {"enc": False, "refine": False} and torch.is_grad_enabled()                # désactivé dedans, rétabli dehors
    src = (ROOT / "scripts" / "train_stage_s_v2.py").read_text()
    assert "with torch.no_grad():" in src[src.index("def evaluate_full_page"):src.index("def evaluate_all")]


def test_k2_cudnn_is_deterministic_without_benchmark_at_the_start_of_controls_cal_and_eval_and_recorded(tmp_path, monkeypatch):
    saved = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
        assert E4.set_determinism() == {"cudnn_deterministic": True, "cudnn_benchmark": False}
        assert E4.run_identity(CFG)["cudnn_deterministic"] is True and E4.run_identity(CFG)["cudnn_benchmark"] is False
        order = []
        for phase, patch, call in (
                ("controls", ("check_code_identity", E3), lambda: E4.phase_controls(None, CFG, torch.device("cpu"))),
                ("CAL", ("check_cal_page", E3), lambda: E4.phase_CAL(types.SimpleNamespace(pages=["mf_0045"]), CFG, torch.device("cpu"), tmp_path)),
                ("EVAL", None, lambda: E4.phase_EVAL(_eval_args(tmp_path), CFG, torch.device("cpu"), tmp_path, tmp_path))):
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
            real = E4.set_determinism
            monkeypatch.setattr(E4, "set_determinism", lambda _p=phase: order.append(_p) or real())
            if patch:
                monkeypatch.setattr(patch[1], patch[0], lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop")))
            (tmp_path / "ld4_report.json").write_text("{}")                                   # EVAL : refus immédiat après set_determinism
            with pytest.raises(SystemExit):
                call()
            assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
            monkeypatch.undo()
        assert order == ["controls", "CAL", "EVAL"]
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved


def test_k2_the_report_records_the_flags_and_that_the_exact_check_ran_in_cal_before_the_marker():
    rows = [_row(p, 0.97, 0.91, 0.88, 0.95) for p in IDS]
    rep = E4.build_report(rows, IDS, CFG, {}, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, False, E4.run_identity(CFG))
    assert rep["b3_exact_checked_in_CAL"] is True and set(rep["determinism"]) == {"cudnn_deterministic", "cudnn_benchmark"}
    assert rep["run_identity"]["cudnn_deterministic"] is rep["determinism"]["cudnn_deterministic"]
    src = (ROOT / "scripts" / "eval_ld4.py").read_text()
    assert '"b3_exact_checked": True' in src and "AVANT le marqueur" in src                    # CAL : INCOMPLETE, aucune page générée
    page_fn = src[src.index("def evaluate_page_ld4"):src.index("# ---------- critère")]
    assert "check_b3_exact(score, after)" in page_fn                                            # contrôle strict conservé (§2)


def test_k3_the_generation_summary_notes_the_ld3_config_sha_field_and_gives_the_ld4_config_sha():
    gen = {"mode": "real", "seed": 20261028, "names": IDS, "files_sha256": {}, "seconds_total": 1.0,
           "generator": {"commit": "x", "ld3_config_sha256": "ee" * 32}, "overlap": {"status": "OK", "per_page": [1], "blocking": False}}
    s = E4.generation_summary(gen, CFG)
    assert s["ld4_config_sha256"] == C2.sha256_file(ROOT / "config/ld4.json") and "ld3_config_sha256" in s["note"] and "config/ld3.json" in s["note"]
    assert s["generator"]["ld3_config_sha256"] == "ee" * 32 and "per_page" not in s["overlap"]
    assert C2.sha256_file(ROOT / "scripts/data3_generate.py") == CFG["frozen_files_sha256"]["scripts/data3_generate.py"]       # fichier inchangé


# ---------- gel 3 : B1 / M3 (fichiers épinglés présents dans le kit), M2 (nom de l'enregistrement de sortie) ----------

@NEEDS_CARNET
def test_every_pinned_or_identity_file_is_in_the_kit_and_exists_locally():
    kit = set(json.loads((NB / "kit_files.json").read_text()))
    needed = set(CFG["frozen_files_sha256"]) | set(CFG["e2e5"]["code_pins"]) | set(CFG["run_identity_files"])
    for rel in sorted(needed):
        assert (ROOT / rel).is_file(), rel
        assert rel in kit, f"{rel} manque dans kit_files.json"
    assert "scripts/diag_ld_o1.py" in kit and "config/ld_o1.json" in kit
    assert CFG["robustness"]["exit_record"] == "ld4_run_exit.json" and "ld4_run_exit.json" in _code("kaggle_ld4.ipynb")
    assert "check_b3_exact(score, after)" in (ROOT / "scripts" / "eval_ld4.py").read_text()          # contrôle exact toujours strict
