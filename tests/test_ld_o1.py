"""LD-O1 : tests légers (docs/END_TO_END_LD_O1_LOT.txt). Petits tableaux synthétiques : aucun modèle, aucun détecteur, aucune page réelle.
Les tests de carnet sont ignorés si le constructeur est absent du kit Kaggle."""
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import ld1b_border_diag as LB  # noqa: E402
import ld2_core as C2  # noqa: E402

torch = pytest.importorskip("torch")
import diag_ld_o1 as D  # noqa: E402
import eval_ld3 as E3  # noqa: E402
import train_ld2 as TD  # noqa: E402
import train_line_detr as TL  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402

CARNETS = ROOT / "scripts" / "stage_s_notebooks"
OUT_NB = CARNETS / "ld_o1"
DOC = ROOT / "docs" / "END_TO_END_LD_O1_LOT.txt"
CFG = D.load_cfg()
THR = CFG["threshold"]
NEEDS_CARNET = pytest.mark.skipif(not (CARNETS / "build_kaggle_ld_o1.py").is_file() or not (OUT_NB / "notebook.ipynb").is_file(),
                                  reason="build_kaggle_ld_o1.py / ld_o1/ absents du kit (Kaggle)")
NEEDS_DOC = pytest.mark.skipif(not DOC.is_file(), reason="docs/ absent du kit (Kaggle)")


def _norm(t):
    return re.sub(r"\s+", " ", t)


def _b():
    import build_kaggle_ld_o1 as B
    return B


# ---------- jouets ----------

def line(y, x0=100.0, x1=400.0, n=8):
    return np.stack([np.linspace(x0, x1, n), np.full(n, float(y))], axis=-1)


def gt(i, y, text="ab", x0=100.0, x1=400.0):
    return {"id": f"g{i}", "baseline": line(y, x0, x1, 2), "neg": 12.0, "pos": 4.0, "text": text, "legibility": "readable"}


def cand(score, zone, query, y, h=16.0, x0=100.0, x1=400.0):
    return {"score": score, "zone": zone, "query": query, "pts": line(y, x0, x1), "h": h, "neg": h * 0.75, "pos": h * 0.25}


META = LD.ZoneMeta(ix0=0, iy0=0, fx=1.0, fy=1.0, crop_w=700, crop_h=500)


def raw_of(queries):
    """Sorties brutes d'une zone (repère canevas = repère page, méta identité) à partir de (prob, y, h)."""
    n = len(queries)
    return {"prob": np.array([q[0] for q in queries]), "pts": np.stack([line(q[1]) for q in queries]),
            "neg": np.array([q[2] * 0.75 for q in queries]), "pos": np.array([q[2] * 0.25 for q in queries]),
            "hyp": {i: "ab" for i in range(n)}, "hyp_none": {}}


# ---------- NMS instrumentée ----------

def test_the_instrumented_nms_keeps_exactly_what_nms_lines_keeps_and_names_the_first_suppressing_kept():
    rng = np.random.default_rng(1)
    preds = []
    for z in range(4):
        for q in range(12):
            y = float(rng.choice([60.0, 66.0, 72.0, 200.0, 207.0, 330.0]))
            preds.append(cand(float(rng.choice([0.61, 0.7, 0.8, 0.95])), z, q, y + float(rng.normal(0, 1)), h=float(rng.choice([16.0, 30.0])),
                              x0=float(rng.choice([100.0, 150.0])), x1=float(rng.choice([380.0, 420.0]))))
    kept, sup = D.instrumented_nms(preds)
    ref = LD.nms_lines(preds)
    assert [(k["zone"], k["query"]) for k in kept] == [(k["zone"], k["query"]) for k in ref] and all(a is b for a, b in zip(kept, ref))
    assert len(kept) + len(sup) == len(preds) and len(sup) > 0
    order = [(k["zone"], k["query"]) for k in kept]
    for c, s in sup:
        assert LD._overlap_rule(c["pts"], s["pts"], c["h"], s["h"])
        # s est le PREMIER gardé (dans l'ordre des gardés) qui recouvre c, parmi ceux gardés avant c
        earlier = [k for k in kept if (-k["score"], k["zone"], k["query"]) < (-c["score"], c["zone"], c["query"])]
        first = next(k for k in earlier if LD._overlap_rule(c["pts"], k["pts"], c["h"], k["h"]))
        assert s is first and order.index((s["zone"], s["query"])) == order.index((first["zone"], first["query"]))


def test_the_instrumented_nms_uses_the_unchanged_overlap_rule(monkeypatch):
    calls = []
    real = LD._overlap_rule
    monkeypatch.setattr(LD, "_overlap_rule", lambda *a: calls.append(1) or real(*a))
    D.instrumented_nms([cand(0.9, 0, 0, 60.0), cand(0.8, 1, 0, 62.0)])
    assert calls and D.LD._overlap_rule is not real
    assert C2.sha256_file(ROOT / "src/babaorum/experimental/line_detr.py").startswith("7715a8e3")


def test_all_queries_keeps_every_query_in_page_coordinates_like_candidates_for_threshold():
    meta = LD.ZoneMeta(ix0=10, iy0=20, fx=0.5, fy=0.5, crop_w=400, crop_h=200)
    raws = [(3, meta, raw_of([(0.9, 60.0, 16.0), (0.2, 80.0, 16.0)]))]
    qs = D.all_queries(raws)
    assert len(qs) == 2 and [q["score"] for q in qs] == [0.9, 0.2] and all(q["zone"] == 3 for q in qs)
    ref = TL.candidates_for_threshold(raws, THR)
    assert len(ref) == 1 and np.array_equal(ref[0]["pts"], qs[0]["pts"]) and ref[0]["h"] == qs[0]["h"]


# ---------- classes ----------

PF = CFG["prefilter"]


def _cls(g, queries, kept, suppressed, cfg=CFG):
    return D.classify_missed(g, queries, THR, kept, suppressed, cfg)


def test_classes_c1_to_c5_in_the_lot_order():
    g = gt(0, 60.0)
    match = cand(0.9, 0, 0, 60.0)
    far = cand(0.99, 1, 0, 300.0)
    low = cand(0.4, 0, 1, 61.0)
    c1 = _cls(g, [match, far], [far], [(match, far)])                                       # supprimé par une AUTRE zone
    assert c1["class"] == "C1_SUPPR_AUTRE_ZONE" and c1["suppressor_zone"] == 1 and c1["candidate_zone"] == 0
    far_same = cand(0.99, 0, 5, 300.0)
    c2 = _cls(g, [match, far_same], [far_same], [(match, far_same)])                        # même zone
    assert c2["class"] == "C2_SUPPR_MEME_ZONE"
    both = _cls(g, [match, far, far_same], [far, far_same], [(match, far_same), (cand(0.8, 0, 7, 61.0), far)])
    assert both["class"] == "C1_SUPPR_AUTRE_ZONE" and both["n_suppressed_matching"] == 2          # C1 d'abord
    c3 = _cls(g, [match], [match], [])                                                      # gardé qui apparie seul
    assert c3["class"] == "C3_CONFLIT" and c3["kept_zone"] == 0
    c4 = _cls(g, [low, cand(0.5, 2, 0, 59.0), far], [far], [])
    assert c4["class"] == "C4_SOUS_SEUIL" and c4["best_score"] == 0.5 and c4["n_below_threshold_matching"] == 2
    c5 = _cls(g, [far, cand(0.3, 1, 1, 70.0)], [far], [])                                   # 10 px sur h = 16 : distance 0,625 > 0,5, pas d'appariement
    assert c5["class"] == "C5_AUCUNE" and c5["nearest_distance_norm"] == pytest.approx(0.625)
    assert _cls(g, [], [], [])["class"] == "C5_AUCUNE" and _cls(g, [], [], [])["nearest_distance_norm"] is None


def test_the_prefilter_changes_no_class():
    g = gt(0, 60.0)
    pool = [cand(0.9, 0, 0, 60.0), cand(0.4, 1, 0, 61.0), cand(0.99, 2, 0, 400.0), cand(0.7, 3, 0, 62.0, x0=900.0, x1=1200.0),
            cand(0.95, 4, 0, 130.0)]
    for kept, sup in (([pool[2]], [(pool[0], pool[2])]), ([pool[0]], []), ([pool[2]], []), ([], [(pool[3], pool[2])])):
        on = _cls(g, pool, kept, sup)
        off = _cls(g, pool, kept, sup, dict(CFG, prefilter=dict(PF, enabled=False)))
        assert on == off
    assert D.plausible(g, pool[0], PF) and not D.plausible(g, pool[2], PF) and not D.plausible(g, pool[3], PF)


def test_alone_match_is_a_single_line_line_geometry_match():
    g = gt(0, 60.0)
    assert D.alone_match(g, cand(0.9, 0, 0, 60.0)) and D.alone_match(g, cand(0.9, 0, 0, 62.0))
    assert not D.alone_match(g, cand(0.9, 0, 0, 90.0)) and not D.alone_match(g, cand(0.9, 0, 0, 60.0, x0=100.0, x1=180.0))


# ---------- une source : classes, tranches, P/R recalculés, annexe ----------

def _toy_source():
    g0, g1 = gt(0, 60.0), gt(1, 72.0)                      # 12 px d'écart ; h = 16 -> tolérance 8 : une prédiction n'apparie que sa ligne
    page = {"gt_lines": [g0, g1]}
    z0, z1 = (50.0, 20.0, 450.0, 150.0), (50.0, 20.0, 450.0, 150.0)
    zones = [(0, z0), (1, z1)]
    # zone 0 : q0 (0,95) sur g0 ; zone 1 : q0 (0,70) sur g1, de hauteur prédite 30 -> recouvre la ligne gardée de la zone 0 (t = 12 <= 15)
    raws = [(0, META, raw_of([(0.95, 60.0, 30.0), (0.30, 300.0, 16.0)])), (1, META, raw_of([(0.70, 72.0, 30.0), (0.10, 400.0, 16.0)]))]
    return page, zones, raws, {"g0": 0, "g1": 1}


def test_diagnose_source_classifies_the_missed_line_checks_nms_and_pr_and_counts_by_bin():
    page, zones, raws, owner = _toy_source()
    res = D.diagnose_source(page, zones, raws, owner, THR, CFG)
    assert res["n_queries"] == 4 and res["n_candidates"] == 2 and res["n_kept"] == 1 and res["n_suppressed"] == 1
    assert res["score"]["recall"] == 0.5 and res["score"]["precision"] == 1.0 and res["score"]["n_pred"] == 1
    assert [(r["id"], r["class"]) for r in res["missed"]] == [("g1", "C1_SUPPR_AUTRE_ZONE")]
    # g1 : zone 1, boîte (50, 20, 450, 150) ; haut de bande 72 - 12 = 60 -> 40 ; bas 72 + 4 = 76 -> 74 ; d = 40 -> tranche > 32
    assert res["missed"][0]["bin"] == ">32" and res["counts"]["C1_SUPPR_AUTRE_ZONE"][">32"] == 1
    assert sum(sum(v.values()) for v in res["counts"].values()) == 1 and res["missed_by_bin"][">32"] == 1
    assert res["gt_by_bin"][">32"] == 1 and res["gt_by_bin"]["16-32"] == 1 and res["gt_by_bin"]["hors_zone"] == 0       # g0 : d = 28
    assert res["score"] == {k: TD.score_lines(page, TL.candidates_for_threshold(raws, THR), raws)[k]
                            for k in ("n_pred", "n_gt", "precision", "recall", "splits", "merges")}
    ann = res["annex_owner_filter"]
    assert ann["n_removed"] == 0 and ann["recall"] == 0.5 and ann["precision"] == 1.0           # milieu de g0 dans la boîte


def test_a_line_without_owner_zone_is_counted_apart_and_edge_bins_use_ld1b_definitions():
    page, zones, raws, owner = _toy_source()
    res = D.diagnose_source(page, zones, raws, {"g0": 0}, THR, CFG)                             # g1 sans zone propriétaire
    assert res["missed"][0]["bin"] == "hors_zone" and res["gt_by_bin"]["hors_zone"] == 1 and res["counts"]["C1_SUPPR_AUTRE_ZONE"]["hors_zone"] == 1
    assert D.BIN_KEYS == LB.BIN_NAMES + ["hors_zone"] and D.SOURCES == ("Z-PP-blocs", "Z-GT")
    g = {"baseline": line(30.0), "neg": 12.0, "pos": 4.0}
    assert LB.bin_of(LB.band_distance(g["baseline"], g["neg"], g["pos"], (0.0, 20.0, 500.0, 200.0))) == "0-8"        # haut de bande : 18 -> -2


def test_the_owner_filter_keeps_a_line_only_if_its_midpoint_is_inside_the_zone_box():
    inside, outside = cand(0.9, 0, 0, 60.0), cand(0.9, 0, 1, 60.0, x0=400.0, x1=900.0)
    box = (50.0, 20.0, 450.0, 150.0)
    assert D.owner_filter([inside, outside], {0: box}) == [inside]                              # milieu de la seconde : x = 650 > 450
    assert D.owner_filter([inside], {0: (50.0, 70.0, 450.0, 150.0)}) == []                      # milieu au-dessus de la boîte


def test_the_diagnosis_refuses_when_the_instrumented_nms_or_the_pr_would_differ(monkeypatch):
    page, zones, raws, owner = _toy_source()
    monkeypatch.setattr(D, "instrumented_nms", lambda preds: ([], []))
    with pytest.raises(RuntimeError, match="NMS instrumentée"):
        D.diagnose_source(page, zones, raws, owner, THR, CFG)


# ---------- agrégat, rapport ----------

def _page_result(pid):
    page, zones, raws, owner = _toy_source()
    res = {"page": pid, "checkpoint_sha256": "ab" * 32, "seconds": 1.0}
    for src in D.SOURCES:
        res[src] = dict(D.diagnose_source(page, zones, raws, owner, THR, CFG), n_zones=2)
    return res


def test_the_report_has_reserves_first_the_prediction_copied_totals_and_the_run_identity():
    pages = [_page_result(p) for p in CFG["pages"]]
    ident = {"checkpoint_sha256": "ab" * 32, "alphabet": ["<blank>"], "threshold": 0.6}
    rep = D.build_report(pages, CFG, ident, D.run_identity(CFG))
    assert list(rep)[:3] == ["schema", "checkpoint_sha256", "reserves"]
    assert rep["reserves"] == [CFG["reserves"][0], C2.RESERVES[6], C2.RESERVES[7], CFG["reserves"][1]]
    assert rep["reserves"][0].startswith("LD-2 a été entraîné sur mf_0000-0049") and "0,934" in rep["reserves"][1] and "0,843" in rep["reserves"][2]
    assert rep["prediction"]["lines"] == CFG["prediction"]["lines"] and rep["descriptive_only"] is True
    tot = rep["totals"]["Z-PP-blocs"]
    assert tot["missed_total"] == 5 and tot["gt_total"] == 10 and tot["counts"]["C1_SUPPR_AUTRE_ZONE"][">32"] == 5
    assert tot["shares_of_missed_by_bin"][">32"]["C1_SUPPR_AUTRE_ZONE"] == 1.0 and tot["shares_of_missed_by_bin"]["0-8"] is None
    assert tot["final"] == {"P": 1.0, "R": 0.5} and tot["annex_owner_filter"]["dR"] == 0.0 and tot["annex_owner_filter"]["descriptive_only"]
    assert rep["pages"] == CFG["pages"] and len(rep["per_page"]) == 5
    ri = rep["run_identity"]
    assert ri["torch"] == torch.__version__ and set(ri["files_sha256"]) == {"scripts/diag_ld_o1.py", "config/ld_o1.json", "tests/test_ld_o1.py"}
    assert ri["frozen_files_sha256"]["src/babaorum/experimental/line_detr.py"].startswith("7715a8e3")
    json.dumps(rep)


@NEEDS_DOC
def test_the_prediction_and_the_first_reserve_are_in_the_lot_verbatim():
    doc = _norm(DOC.read_text())
    for line_ in CFG["prediction"]["lines"] + [CFG["reserves"][0], CFG["reserves"][1]]:
        assert _norm(line_) in doc or line_ == CFG["reserves"][1], line_[:60]
    assert "DESCRIPTIF : il ne décide rien et ne règle rien" in doc and "3cce0cd2790395723dd0275ad1543c3fe4aa2d10264f55c857c65d8c316620cd" == C2.sha256_file(DOC)


# ---------- gardes ----------

def test_pages_must_be_among_mf_0045_to_0049_and_the_report_needs_exactly_the_five():
    assert D.check_pages(["mf_0045"], CFG) == ["mf_0045"] and D.check_pages(CFG["pages"][1:], CFG) == CFG["pages"][1:]
    assert D.check_pages(CFG["pages"], CFG, exact=True) == CFG["pages"] and CFG["pages"] == [f"mf_00{i}" for i in range(45, 50)]
    for bad in (["mf_0044"], ["mf_0050"], ["mf_0059"], ["mf_0060"], ["mfn_0000"], ["decoy_0000"], ["mf_0045", "mf_0045"], [], ["mf_0045", "mf_0050"]):
        with pytest.raises(SystemExit):
            D.check_pages(bad, CFG)
    with pytest.raises(SystemExit, match="cinq pages"):
        D.check_pages(CFG["pages"][:4], CFG, exact=True)


def test_run_refuses_a_reserved_page_before_loading_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "make_model", lambda *a, **k: pytest.fail("aucun modèle ne doit être chargé"))
    args = types.SimpleNamespace(pages=["mf_0050"], root=str(tmp_path), ld2_dir=None, cache_dir=str(tmp_path))
    with pytest.raises(SystemExit):
        D.phase_RUN(args, CFG, tmp_path)
    args.pages = ["mf_0045"]
    monkeypatch.setattr(LD, "MARGIN_H", 1.0)
    with pytest.raises(SystemExit, match="MARGIN_H"):
        D.phase_RUN(args, CFG, tmp_path)


def test_the_report_phase_refuses_a_page_computed_with_another_checkpoint(tmp_path, monkeypatch):
    for p in CFG["pages"]:
        C2.write_json(tmp_path / f"ld_o1_page_{p}.json", dict(_page_result(p), checkpoint_sha256="cd" * 32))
    monkeypatch.setattr(E3, "check_ld2_identity", lambda d, cfg: {"checkpoint_sha256": "ab" * 32, "alphabet": []})
    with pytest.raises(SystemExit, match="autre checkpoint"):
        D.phase_REPORT(types.SimpleNamespace(ld2_dir="x"), CFG, tmp_path)
    for p in CFG["pages"]:
        C2.write_json(tmp_path / f"ld_o1_page_{p}.json", _page_result(p))
    assert D.phase_REPORT(types.SimpleNamespace(ld2_dir="x"), CFG, tmp_path) == {"report": "ld_o1_report.json"}
    assert json.loads((tmp_path / "ld_o1_report.json").read_text())["status"] == "COMPLETE"
    (tmp_path / "ld_o1_page_mf_0049.json").unlink()
    with pytest.raises(FileNotFoundError):
        D.phase_REPORT(types.SimpleNamespace(ld2_dir="x"), CFG, tmp_path)


def test_the_pins_match_the_files_and_the_frozen_ld_files_are_untouched():
    D.check_pins(CFG)
    for rel in ("scripts/train_ld2.py", "scripts/ld2_core.py", "config/ld2.json", "src/babaorum/experimental/line_detr.py",
                "scripts/train_line_detr.py", "scripts/eval_ld3.py", "scripts/ld1b_border_diag.py"):
        assert rel in CFG["frozen_files_sha256"] and CFG["frozen_files_sha256"][rel] == C2.sha256_file(ROOT / rel)
    bad = json.loads(json.dumps(CFG))
    bad["frozen_files_sha256"]["scripts/eval_ld3.py"] = "0" * 64
    with pytest.raises(SystemExit, match="eval_ld3"):
        D.check_pins(bad)
    assert CFG["ld2"]["checkpoint_sha256"].startswith("e5afd143") and CFG["threshold"] == 0.6 and CFG["margin_h"] == 0.5


def test_the_code_names_the_reserved_pages_only_in_comments_and_no_number_of_the_lot_is_hardcoded():
    for rel in ("scripts/diag_ld_o1.py", "scripts/stage_s_notebooks/build_kaggle_ld_o1.py"):
        if not (ROOT / rel).is_file():
            continue
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.search(r"mf_005\d|mf_006\d|mfn_\d", node.value) or len(node.value) > 60, (rel, node.value[:50])
    for node in ast.walk(ast.parse((ROOT / "scripts" / "diag_ld_o1.py").read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, float) and not isinstance(node.value, bool):
            assert node.value not in (0.6, 0.5, 7200.0), node.lineno


# ---------- carnet ----------

def _code():
    nb = json.loads((OUT_NB / "notebook.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@NEEDS_CARNET
def test_the_notebook_on_disk_equals_the_builder_and_is_cpu_with_the_right_sources():
    B = _b()
    assert json.loads((OUT_NB / "notebook.ipynb").read_text()) == B.build_notebook() and _code() == B.cell()
    meta = json.loads((OUT_NB / "kernel-metadata.json").read_text())
    assert meta == B.kernel_metadata() and meta["enable_gpu"] is False and meta["id"] == "mabaku/babaorum-ld-o1"
    assert set(meta["kernel_sources"]) == {"mabaku/babaorum-ld2-run", "mabaku/babaorum-e2e3-run3-fp-classes"}
    assert (OUT_NB / "notebook.ipynb").stat().st_size < 300 * 1024
    files = set(json.loads((OUT_NB / "kit_files.json").read_text()))
    assert files == set(B.kit_files()) and all((ROOT / f).is_file() for f in files)
    for rel in ("scripts/diag_ld_o1.py", "config/ld_o1.json", "tests/test_ld_o1.py", "scripts/eval_ld3.py", "scripts/ld1b_border_diag.py",
                "scripts/train_ld2.py", "config/ld3.json"):
        assert rel in files, rel
    code = _code()
    for bad in ("glob(", "rglob", "listdir", "scandir", "os.walk", "iterdir", "mf_005", "mf_006", "mfn_"):
        assert bad not in code, bad


def _data2_common():
    spec = importlib.util.spec_from_file_location("data2_common_ldo1test", ROOT / "scripts" / "data2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Run:
    """Exécute le vrai corps du carnet avec un faux run_step et une fausse horloge."""

    def __init__(self, tmp_path, monkeypatch, plan, start=0.0):
        B = _b()
        self.calls, self.clock, self.plan = [], [start], plan
        self.out = tmp_path / "work"
        self.out.mkdir()
        self.C = _data2_common()
        monkeypatch.setattr(self.C, "run_step", self.run_step)
        tree = ast.parse(B.cell())
        lits = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("LIMITS", "TIMEOUTS", "PAGE_IDS", "RUN_FILES")}
        self.__dict__.update(limits=lits["LIMITS"], timeouts=lits["TIMEOUTS"])
        ns = {"time": types.SimpleNamespace(monotonic=lambda: self.clock[0]), "json": json, "os": os, "sys": sys, "Path": Path,
              "OUT": self.out, "C": self.C, "C2": C2, "T_START": 0.0, "KIT": Path("/k"), "PAGES": Path("/p"), "LD2DIR": Path("/d"),
              "sha256": lambda p: "0" * 64, "_utc": lambda: "00:00:00", "LIMITS": self.limits, "TIMEOUTS": self.timeouts,
              "PAGE_IDS": lits["PAGE_IDS"], "RUN_FILES": lits["RUN_FILES"]}
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "9.9"
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        exec(compile(B.BODY, "<BODY>", "exec"), ns)
        self.ns = ns

    def run_step(self, name, argv, timeout_s, logs_dir, cwd=None, env=None, kill_grace_s=10.0):
        p = {"rc": 0, "dur": 1.0, **self.plan.get(name, {})}
        self.calls.append((name, timeout_s, list(argv)))
        self.clock[0] += p["dur"]
        return {"step": name, "rc": p["rc"], "seconds": p["dur"], "stdout": "", "stderr": "", "stderr_tail": ""}

    @property
    def exit(self):
        return self.ns["EXIT"]

    @property
    def names(self):
        return [c[0] for c in self.calls]

    def argv(self, name):
        return dict((c[0], c[2]) for c in self.calls)[name]

    def timeout(self, name):
        return dict((c[0], c[1]) for c in self.calls)[name]


@NEEDS_CARNET
def test_the_smoke_page_runs_alone_first_then_the_four_others_then_the_report(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    assert r.exit["status"] == "DONE" and r.names == ["D", "RUN1", "RUN4", "REPORT"]
    a1, a4 = r.argv("RUN1"), r.argv("RUN4")
    assert a1[a1.index("--pages") + 1:] == ["mf_0045"]
    assert a4[a4.index("--pages") + 1:] == ["mf_0046", "mf_0047", "mf_0048", "mf_0049"] and "--phase" in a1 and a1[a1.index("--phase") + 1] == "RUN"
    assert r.argv("REPORT")[r.argv("REPORT").index("--phase") + 1] == "REPORT" and "--pages" not in r.argv("REPORT")
    assert r.limits["deadline_s"] == 7200 and r.limits["stop_margin_s"] == 120
    assert r.timeout("D") == r.timeouts["D"] and r.timeout("RUN4") == r.timeouts["RUN4"]


@NEEDS_CARNET
def test_if_the_smoke_page_fails_nothing_else_runs(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"RUN1": {"rc": 1}})
    assert r.names == ["D", "RUN1"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "RUN1"
    (tmp_path / "t").mkdir()
    r = Run(tmp_path / "t", monkeypatch, {"RUN1": {"rc": "TIMEOUT"}})
    assert r.names == ["D", "RUN1"] and r.exit["status"] == "INCOMPLETE"


@NEEDS_CARNET
def test_every_step_is_bounded_by_the_global_deadline_minus_the_stop_margin(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 80.0}}, start=7200.0 - 200.0)                  # entrée tardive : 200 s restantes
    assert r.names == ["D"] and r.timeout("D") == 80.0 and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "RUN1"
    assert r.clock[0] <= 7200.0
    for i, start in enumerate((7200.0, 7080.0, 9000.0)):
        (tmp_path / str(i)).mkdir()
        z = Run(tmp_path / str(i), monkeypatch, {}, start=start)
        assert z.calls == [] and z.exit["status"] == "INCOMPLETE" and z.exit["step"] == "pre"
    (tmp_path / "n").mkdir()
    n = Run(tmp_path / "n", monkeypatch, {"D": {"dur": 100.0}, "RUN1": {"dur": 1500.0}, "RUN4": {"dur": 4000.0}})
    assert n.timeout("RUN4") == min(n.timeouts["RUN4"], 7200.0 - 101.0 - 1500.0 - 120.0) and n.exit["status"] == "DONE"


def _mount_tree(tmp_path, fmt):
    inp = tmp_path / "input"
    base = (lambda slug, sub: inp / slug) if fmt == "A" else (lambda slug, sub: inp / sub / "mabaku" / slug)
    kit = base("babaorum-ld-o1-kit", "datasets")
    (kit / "config").mkdir(parents=True)
    (kit / "scripts").mkdir()
    import shutil
    shutil.copy(ROOT / "scripts" / "data2_common.py", kit / "scripts")
    shutil.copy(ROOT / "scripts" / "ld2_core.py", kit / "scripts")
    shutil.copy(ROOT / "config" / "data2_pages.json", kit / "config")
    (kit / "config" / "ld_o1.json").write_text("{}")
    mf = base("babaorum-e2e3-run3-fp-classes", "notebooks") / "e2e3" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "pages" / "mf_0000.json").write_text("{}")
    ld2 = base("babaorum-ld2-run", "notebooks") / "ld2"
    ld2.mkdir(parents=True)
    (ld2 / "ld2_manifest.json").write_text("{}")
    (ld2 / "ld2_main_step003271.pt").write_bytes(b"l2")
    (inp / "x" / "y").mkdir(parents=True)
    (inp / "x" / "y" / "mf_0050.json").write_text("{}")                                        # leurre au niveau 3
    return inp


@NEEDS_CARNET
@pytest.mark.parametrize("fmt", ["A", "B"])
def test_the_mounts_resolve_in_both_formats_without_listing_and_a_wrong_sha_is_refused(tmp_path, monkeypatch, fmt):
    B = _b()
    inp = _mount_tree(tmp_path, fmt)
    kw = dict(out=str(tmp_path / "out"), kit=[str(inp / "babaorum-ld-o1-kit"), str(inp / "datasets" / "mabaku" / "babaorum-ld-o1-kit")],
              pages=[str(inp / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf"),
                     str(inp / "notebooks" / "mabaku" / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf")],
              ld2=[str(inp / "babaorum-ld2-run" / "ld2"), str(inp / "notebooks" / "mabaku" / "babaorum-ld2-run" / "ld2")],
              ld2_sha=hashlib.sha256(b"l2").hexdigest())
    head = B.cell(**kw).split("\nRUN = OUT")[0]
    touched = []
    monkeypatch.setattr(os, "listdir", lambda *a, **k: touched.append("listdir") or [])
    monkeypatch.setattr(os, "scandir", lambda *a, **k: touched.append("scandir") or iter(()))
    for name in ("glob", "iterdir", "rglob"):
        monkeypatch.setattr(Path, name, lambda *a, _n=name, **k: touched.append(_n) or iter(()))
    ns = {"__name__": "ldo1_mount"}
    saved = list(sys.path)
    exec(compile(head, "ldo1_head", "exec"), ns)
    monkeypatch.undo()
    sys.path[:] = saved
    sys.modules.pop("ld2_core", None)
    assert touched == [] and ns["KIT"].name == "babaorum-ld-o1-kit" and ns["LD2DIR"].name == "ld2" and str(ns["PAGES"]).endswith("e2e3/mf")
    with pytest.raises(AssertionError, match="LD-2"):
        exec(compile(B.cell(**dict(kw, ld2_sha="0" * 64)).split("\nRUN = OUT")[0], "x", "exec"), {})
    sys.modules.pop("ld2_core", None)
    sys.path[:] = saved


# ---------- relecture : M1 (règle de lecture, comptes de suppression), M2 (C5 proche / loin) ----------

def test_m1_each_missed_line_records_the_suppression_counts_and_c1_priority_is_explicit():
    g = gt(0, 60.0)
    match_a, match_b = cand(0.9, 0, 0, 60.0), cand(0.8, 0, 7, 61.0)
    far1, far0 = cand(0.99, 1, 0, 300.0), cand(0.98, 0, 5, 300.0)
    both = _cls(g, [match_a, match_b], [far1, far0], [(match_a, far1), (match_b, far0)])        # une suppression autre zone, une même zone
    assert both["class"] == "C1_SUPPR_AUTRE_ZONE" and both["n_suppressed_other_zone"] == 1 and both["n_suppressed_same_zone"] == 1
    only_same = _cls(g, [match_a], [far0], [(match_a, far0)])
    assert only_same["class"] == "C2_SUPPR_MEME_ZONE" and (only_same["n_suppressed_other_zone"], only_same["n_suppressed_same_zone"]) == (0, 1)
    c3 = _cls(g, [match_a], [match_a], [])
    assert c3["class"] == "C3_CONFLIT" and (c3["n_suppressed_other_zone"], c3["n_suppressed_same_zone"]) == (0, 0)
    for r in (_cls(g, [cand(0.4, 0, 1, 61.0)], [far1], []), _cls(g, [far1], [far1], [])):          # C4, C5 : compteurs présents, à zéro
        assert (r["n_suppressed_other_zone"], r["n_suppressed_same_zone"]) == (0, 0)
    # C3 seulement si aucun candidat supprimé n'apparie : une suppression + un gardé qui apparie -> C1/C2, pas C3
    mixed = _cls(g, [match_a, match_b], [match_b], [(match_a, far0)])
    assert mixed["class"] == "C2_SUPPR_MEME_ZONE"


def test_m1_the_report_states_the_reading_rules_and_counts_c1_lines_that_also_have_a_same_zone_suppression():
    pages = [_page_result(p) for p in CFG["pages"]]
    for p in pages:                                                                             # g1 : C1 ; on lui ajoute une suppression de même zone
        for src in D.SOURCES:
            p[src]["missed"][0]["n_suppressed_same_zone"] = 1
    rep = D.build_report(pages, CFG, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, D.run_identity(CFG))
    rules = " ".join(rep["reading_rules"]["classes"])
    for needle in ("au moins un candidat supprimé", "AUTRE zone", "sinon C2", "premier gardé", "ordre des scores", "C3 seulement si AUCUN",
                   "c1_with_same_zone_too"):
        assert needle in rules, needle
    ex = rep["totals"]["Z-PP-blocs"]["extras_by_bin"]
    assert ex[">32"]["c1_with_same_zone_too"] == 5 and ex["0-8"]["c1_with_same_zone_too"] == 0 and set(ex) == set(D.BIN_KEYS)
    base = D.build_report([_page_result(p) for p in CFG["pages"]], CFG, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, D.run_identity(CFG))
    assert base["totals"]["Z-PP-blocs"]["extras_by_bin"][">32"]["c1_with_same_zone_too"] == 0      # la lecture C2 prioritaire se calcule du rapport
    assert all("n_suppressed_other_zone" in r and "n_suppressed_same_zone" in r for r in base["per_page"][0]["Z-PP-blocs"]["missed"])


def test_m2_c5_lines_are_split_into_near_far_and_no_query_with_the_merge_split_note():
    def row(i, d, b="0-8"):
        return {"id": f"g{i}", "bin": b, "class": "C5_AUCUNE", "nearest_distance_norm": d, "n_suppressed_other_zone": 0, "n_suppressed_same_zone": 0}
    page = _page_result("mf_0045")
    page["Z-PP-blocs"]["missed"] = [row(0, 0.2), row(1, 0.5), row(2, 0.51), row(3, 2.0), row(4, None), row(5, 0.3, ">32")]
    for src in D.SOURCES:
        page[src]["counts"] = {c: {b: 0 for b in D.BIN_KEYS} for c in CFG["classes"]}
        page[src]["counts"]["C5_AUCUNE"]["0-8"], page[src]["counts"]["C5_AUCUNE"][">32"] = 5, 1
        page[src]["missed_by_bin"] = {b: 0 for b in D.BIN_KEYS}
        page[src]["missed_by_bin"]["0-8"], page[src]["missed_by_bin"][">32"] = 5, 1
    page["Z-GT"]["missed"] = []
    agg = D.aggregate_source([page], "Z-PP-blocs", CFG)["extras_by_bin"]
    assert agg["0-8"] == {"c1_with_same_zone_too": 0, "c5_near": 2, "c5_far": 2, "c5_no_query": 1}    # 0,2 et 0,5 proches ; 0,51 et 2,0 loin
    assert agg[">32"]["c5_near"] == 1 and CFG["c5_near_threshold"] == 0.5
    note = CFG["reading_rules"]["c5_note"]
    assert "c5_near" in note and "fusions" in note and "coupures" in note and "75 %" in note
    rep = D.build_report([_page_result(p) for p in CFG["pages"]], CFG, {"checkpoint_sha256": "ab" * 32, "alphabet": []}, D.run_identity(CFG))
    assert rep["reading_rules"]["c5_note"] == note
