"""LD-2 : tests légers, SANS torch (docs/END_TO_END_LD2_LOT.txt §0-§7). Lançables en local et dans la fumée Kaggle.
Les tests de carnets (égalité avec le constructeur) sont ignorés si le constructeur est absent du kit Kaggle."""
import ast
import builtins
import hashlib
import shlex
import subprocess
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import ld2_core as C2  # noqa: E402

CARNETS = ROOT / "scripts" / "stage_s_notebooks"
LD2 = CARNETS / "ld2"
LOT = ROOT / "docs" / "END_TO_END_LD2_LOT.txt"
REF_REPORT = ROOT / "runs" / "e2e5_kaggle_output_v2" / "e2e5_lambda_run_report.json"
NEEDS_CARNET = pytest.mark.skipif(not (CARNETS / "build_kaggle_ld2.py").is_file() or not (LD2 / "kaggle_ld2.ipynb").is_file(),
                                  reason="build_kaggle_ld2.py / ld2/*.ipynb absents du kit (Kaggle)")
NEEDS_LOT = pytest.mark.skipif(not LOT.is_file(), reason="docs/ absent du kit (Kaggle)")
CFG = C2.load_config()


def _b():
    import build_kaggle_ld2 as B2
    return B2


# ---------- formules de pas (§5 (1), (2)) ----------

@pytest.mark.parametrize("s,expect", [(0.5, 16800), (0.42, 20000), (3.0, 2800), (2.2438474916250004, 3743), (8400.0, 1)])
def test_S2_is_min_of_20000_and_floor_of_8400_over_seconds_per_step(s, expect):
    assert C2.steps_main(s, CFG) == expect


@pytest.mark.parametrize("s2,s,expect", [(3000, 2.0, 2100), (3000, 1.0, 3000), (3000, 0.1, 3000), (20000, 4.0, 1050), (10, 4200.0, 1)])
def test_S_t_is_min_of_S2_and_floor_of_4200_over_witness_seconds_per_step(s2, s, expect):
    assert C2.steps_witness(s2, s, CFG) == expect


def test_the_step_numbers_come_from_the_config_not_from_the_code():
    cfg = json.loads(json.dumps(CFG))
    cfg["steps"].update(max_steps=100, main_budget_s=50, witness_budget_s=30)
    assert C2.steps_main(0.1, cfg) == 100 and C2.steps_main(1.0, cfg) == 50 and C2.steps_witness(100, 1.0, cfg) == 30


@NEEDS_CARNET
def test_gel6_the_built_full_notebook_carries_the_19800_s_deadline_and_the_decision_is_recorded():
    B2 = _b()
    lim = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(_code("kaggle_ld2.ipynb")).body
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "LIMITS"}["LIMITS"]
    assert lim["deadline_s"] == 19800 and CFG["robustness"]["deadline_s"] == 19800 and "19 800" in CFG["decisions"]["gel6"]
    assert CFG["decisions"]["gel6"].startswith("p6 25/10 : lot §9")


def test_the_lot_numbers_are_in_the_config():
    st, rb, fz = CFG["steps"], CFG["robustness"], CFG["freeze"]
    assert (st["timing_steps"], st["max_steps"], st["main_budget_s"], st["witness_budget_s"]) == (200, 20000, 8400, 4200)
    assert (rb["timeouts_s"]["E"], rb["timeouts_s"]["EW"], rb["deadline_s"]) == (9000, 4800, 19800)
    assert st["main_budget_s"] + 600 == rb["timeouts_s"]["E"]                 # 600 s de marge pour les checkpoints et le démarrage
    assert fz["threshold"] == 0.6 and fz["checkpoint_rule"].startswith("le DERNIER")
    assert (CFG["bootstrap"]["draws"], CFG["bootstrap"]["seed"], CFG["bootstrap"]["percentiles"]) == (10000, 20261018, [2.5, 97.5])
    assert CFG["criterion"]["a"]["precision_min"] == 0.8 and CFG["criterion"]["a"]["recall_min"] == 0.9
    assert CFG["criterion"]["b"] == {"dp_low_gt": 0.0, "dr_low_min": -0.02}
    assert CFG["lambda"]["value"] == "1/6" and abs(CFG["lambda"]["float"] - 1 / 6) < 1e-15
    assert (CFG["model"]["n_queries_main"], CFG["model"]["n_queries_witness"], CFG["model"]["k_points"]) == (32, 512, 8)
    assert CFG["reference"]["sha256_prefix"] == "3abfe0d7" and CFG["witness"]["descriptive"] is True
    assert CFG["init"]["decoder"]["sha256_full"] == "a1c5fbab288c99e4911789f926d2d568815fd2a165310da5020ac7c140f8197b"
    assert CFG["init"]["decoder"]["file"] == "ld1_step001871.pt" and CFG["init"]["decoder"]["step"] == 1871
    assert CFG["init"]["encoder"]["sha256_full"] == "ef5e826803c347f73e49c83c0f21f61399f9ea06ee985905bd17644dff3844fa"
    assert CFG["init"]["encoder"]["step"] == 2000


def test_no_lot_number_is_hardcoded_in_the_code():
    """Les nombres du lot sont lus dans config/ld2.json : aucune constante numérique du lot dans le code."""
    banned = {8400, 4200, 20000, 20261018, 20261023, 10000, 9000, 4800, 19800, 0.6, 0.8, 0.9, -0.02, 2.5, 97.5}
    for rel in ("scripts/ld2_core.py", "scripts/train_ld2.py"):
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                assert node.value not in banned, (rel, node.value, node.lineno)


# ---------- refus : shas (§7 (3), §0 (b)) ----------

def test_require_sha_refuses_a_different_file_and_returns_the_sha_otherwise(tmp_path):
    f = tmp_path / "x.pt"
    f.write_bytes(b"abc")
    assert C2.require_sha(f, hashlib.sha256(b"abc").hexdigest(), "x") == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(SystemExit, match="sha différent"):
        C2.require_sha(f, "0" * 64, "checkpoint du décodeur")


def test_a_missing_extra_or_misshaped_reader_key_is_refused_before_loading():
    want = {"conv.weight": (128, 512, 3), "out.bias": (40,)}
    C2.check_reader_keys(dict(want), want)
    with pytest.raises(SystemExit, match="clés reader"):
        C2.check_reader_keys({"conv.weight": (128, 512, 3)}, want)
    with pytest.raises(SystemExit, match="clés reader"):
        C2.check_reader_keys({**want, "extra": (1,)}, want)
    with pytest.raises(SystemExit, match="formes reader"):
        C2.check_reader_keys({**want, "out.bias": (41,)}, want)


def test_the_gel2_decisions_and_the_witness_limit_sentence_are_in_the_config():
    d = CFG["decisions"]
    assert set(d) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q7", "Q8", "Q9", "Q10", "Q12", "gel3", "gel6"} and all("p6 25/10" in v for v in d.values())
    assert CFG["witness"]["limit"] == ("témoin : recouvrement de tuile 384 px choisi par p6 (absent de la config d'E2E-5) ; "
                                       "lignes de plus de ~700 px souvent coupées, témoin pessimiste")
    assert "reader.*" in CFG["init"]["reader"] and "E2E-5" in CFG["init"]["reader"]


def _ref_file(tmp_path, ids, recall=0.5, precision=0.4, n_pred=5, name="rep.json"):
    f = tmp_path / name
    f.write_text(json.dumps({"eval": [{"page_id": p, "metrics": {"after": {"recall": recall, "precision": precision, "n_pred": n_pred}}}
                                      for p in ids]}))
    return f


def test_the_reference_report_is_refused_unless_its_sha_starts_with_3abfe0d7(tmp_path):
    ids = C2.eval_page_ids()
    f = _ref_file(tmp_path, ids)
    with pytest.raises(SystemExit, match="rapport de référence"):
        C2.load_reference(f, ids, CFG)
    cfg = json.loads(json.dumps(CFG))
    cfg["reference"]["sha256_prefix"] = C2.sha256_file(f)[:8]
    ref = C2.load_reference(f, ids, cfg)
    assert ref["mf_0050"] == (0.4, 0.5) and len(ref) == 10
    assert C2.load_reference(f, ids[:2], CFG, smoke=True)["mf_0051"] == (0.4, 0.5)     # leurre de la fumée : pas de sha
    with pytest.raises(SystemExit, match="absentes"):
        C2.load_reference(f, ids + ["mf_0099"], cfg)


def test_the_reference_applies_the_same_zero_convention_as_r7(tmp_path):
    ids = ["mf_0050", "mf_0051"]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"eval": [
        {"page_id": "mf_0050", "metrics": {"after": {"recall": 0.0, "precision": None, "n_pred": 0}}},
        {"page_id": "mf_0051", "metrics": {"after": {"recall": 0.9, "precision": None, "n_pred": 4}}}]}))
    ref = C2.load_reference(f, ids, CFG, smoke=True)
    assert ref == {"mf_0050": (0.0, 0.0), "mf_0051": (0.0, 0.9)}


@pytest.mark.skipif(not REF_REPORT.is_file(), reason="rapport E2E-5 absent (Mac seulement)")
def test_the_real_reference_report_has_the_pre_registered_sha_and_the_ten_pages():
    assert C2.sha256_file(REF_REPORT).startswith("3abfe0d7")
    ref = C2.load_reference(REF_REPORT, C2.eval_page_ids(), CFG)
    assert len(ref) == 10 and all(0.0 <= p <= 1.0 and 0.0 <= r <= 1.0 for p, r in ref.values())


# ---------- bootstrap (§0 (b)) ----------

def test_the_bootstrap_is_deterministic_with_the_seed_and_changes_with_another_seed():
    rng = np.random.default_rng(5)
    dp, dr = rng.normal(0.1, 0.05, 10), rng.normal(0.0, 0.03, 10)
    a, b = C2.paired_bootstrap(dp, dr, CFG), C2.paired_bootstrap(dp, dr, CFG)
    assert a == b and a["draws"] == 10000 and a["seed"] == 20261018 and a["percentiles"] == [2.5, 97.5]
    cfg = json.loads(json.dumps(CFG))
    cfg["bootstrap"]["seed"] = 1
    assert C2.paired_bootstrap(dp, dr, cfg)["dP"]["low"] != a["dP"]["low"]
    # valeur recalculée à la main avec la même construction
    idx = np.random.default_rng(20261018).integers(0, 10, size=(10000, 10))
    lo, hi = np.percentile(dp[idx].mean(axis=1), [2.5, 97.5])
    assert (a["dP"]["low"], a["dP"]["high"]) == (float(lo), float(hi)) and a["dP"]["mean"] == float(dp.mean())


def test_the_bootstrap_is_paired_the_same_pages_are_drawn_for_dP_and_dR():
    d = np.arange(10, dtype=float)
    out = C2.paired_bootstrap(d, d, CFG)
    assert out["dP"] == out["dR"]
    const = C2.paired_bootstrap([0.1] * 10, [0.2] * 10, CFG)
    assert abs(const["dP"]["low"] - 0.1) < 1e-12 and abs(const["dR"]["high"] - 0.2) < 1e-12


# ---------- critère (a) + (b) (§0, conception §10 (5), §11) ----------

def _rows(P, R, n_pred=5, ids=None):
    ids = ids or C2.eval_page_ids()
    return [{"page": p, "precision": P, "recall": R, "n_pred": n_pred} for p in ids]


def _ref(P, R, ids=None):
    return {p: (P, R) for p in (ids or C2.eval_page_ids())}


def test_criterion_met_when_a_and_b_hold_and_each_part_can_fail_alone():
    ids = C2.eval_page_ids()
    ok = C2.criterion(_rows(0.9, 0.95), _ref(0.7, 0.92), ids, CFG)
    assert ok["a"]["met"] and ok["b"]["met"] and ok["verdict"] == "CRITERE_REMPLI"
    low_p = C2.criterion(_rows(0.79, 0.99), _ref(0.5, 0.9), ids, CFG)
    assert not low_p["a"]["met"] and low_p["b"]["met"] and low_p["verdict"] == "CRITERE_NON_REMPLI"
    low_r = C2.criterion(_rows(0.95, 0.89), _ref(0.5, 0.88), ids, CFG)
    assert not low_r["a"]["met"] and low_r["verdict"] == "CRITERE_NON_REMPLI"
    no_gain = C2.criterion(_rows(0.9, 0.95), _ref(0.9, 0.95), ids, CFG)
    assert no_gain["a"]["met"] and not no_gain["b"]["met"]                      # borne basse de ΔP = 0, pas > 0
    recall_loss = C2.criterion(_rows(0.9, 0.95), _ref(0.5, 0.95 + 0.0201), ids, CFG)
    assert recall_loss["a"]["met"] and not recall_loss["b"]["met"]               # borne basse de ΔR < -0,02
    recall_ok = C2.criterion(_rows(0.9, 0.95), _ref(0.5, 0.95 + 0.0199), ids, CFG)
    assert recall_ok["b"]["met"] and recall_ok["verdict"] == "CRITERE_REMPLI"


def test_n_pred_zero_gives_P_and_R_zero_with_the_denominator_ten():
    ids = C2.eval_page_ids()
    rows = _rows(1.0, 1.0)
    rows[0].update(n_pred=0, precision=None, recall=None)
    agg = C2.mean_pr(rows, 10)
    assert agg["P"] == 0.9 and agg["R"] == 0.9 and agg["pages_without_prediction"] == 1
    assert C2.criterion(rows, _ref(0.1, 0.1), ids, CFG)["a"]["met"]               # 0,9 / 0,9 : bornes incluses
    rows[1].update(n_pred=0, precision=0.0, recall=0.0)
    assert C2.mean_pr(rows, 10)["R"] == 0.8 and not C2.criterion(rows, _ref(0.1, 0.1), ids, CFG)["a"]["met"]
    only_empty = _rows(None, None, n_pred=0)
    assert C2.mean_pr(only_empty, 10) == dict(P=0.0, R=0.0, pages=10, pages_without_prediction=10, precision_none_pages=0)
    assert C2.page_pr({"n_pred": 3, "precision": None, "recall": 0.5}) == (0.0, 0.5)
    assert C2.mean_pr([{"n_pred": 3, "precision": None, "recall": 0.5}] * 10, 10)["precision_none_pages"] == 10


def test_the_mean_needs_exactly_the_ten_pages():
    with pytest.raises(ValueError):
        C2.mean_pr(_rows(1.0, 1.0)[:9], 10)


def test_the_criterion_uses_only_the_main_rows_the_witness_has_no_input():
    sig = C2.criterion.__code__.co_varnames[:C2.criterion.__code__.co_argcount]
    assert sig == ("main_rows", "ref", "page_ids", "cfg")


def test_the_eval_page_ids_are_the_ten_reserved_pages_and_only_the_core_builds_them():
    assert C2.eval_page_ids() == [f"mf_00{i}" for i in range(50, 60)]


# ---------- gel (§4 (1), §7 (2)) ----------

def test_the_freeze_record_has_the_threshold_nms_checkpoint_sha_and_nothing_from_the_evaluation_pages():
    rec = C2.freeze_record(CFG, "ld2_main_step002800.pt", "ab" * 32, 2800, {"f": "1"})
    assert rec["threshold"] == 0.6 and rec["nms"]["max_median_distance_in_h"] == 0.5
    assert rec["checkpoint"] == "ld2_main_step002800.pt" and rec["checkpoint_sha256"] == "ab" * 32
    assert rec["checkpoint_rule"] == "le DERNIER (pas S2), sans sélection" and rec["written_before_EVAL"] is True
    assert "mf_005" not in json.dumps(rec)


def test_the_frozen_ld1_ld0_data2_files_are_untouched():
    for rel, prefix in (("src/babaorum/experimental/line_detr.py", "7715a8e3"), ("scripts/train_line_detr.py", "3552c072"),
                        ("config/ld1_line_detr.json", "8c05f43a"), ("scripts/data2_common.py", "55b18e30"),
                        ("scripts/ld0_zone_coverage.py", "4fae48f8"), ("config/stage_s_smoke_v2.json", "a464e1b6")):
        assert C2.sha256_file(ROOT / rel).startswith(prefix), rel
    assert CFG["freeze"]["frozen_files_sha256_prefix"] == {"src/babaorum/experimental/line_detr.py": "7715a8e3",
                                                          "scripts/train_line_detr.py": "3552c072"}
    assert C2.sha256_file(ROOT / CFG["inherit_ld1"]["path"]) == CFG["inherit_ld1"]["sha256"]


def test_allowed_is_max_0_min_cap_remaining_minus_reserve():
    assert C2.allowed(9000, 20000, 2520) == 9000 and C2.allowed(9000, 5000, 2520) == 2480
    assert C2.allowed(9000, 2000, 2520) == 0.0 and C2.allowed(100, 50) == 50


# ---------- réserves du rapport (§0, §5 (4), §6, §7 (5)) ----------

def _norm(t):
    return re.sub(r"\s+", " ", t)


@NEEDS_LOT
def test_every_reserve_is_in_the_lot_verbatim():
    lot = _norm(LOT.read_text())
    for r in C2.RESERVES + [C2.ALTO_MENTION]:
        assert _norm(r) in lot, r[:60]
    assert len(C2.RESERVES) == 8


def test_the_reserves_cover_section_0_6_and_7_5_and_the_alto_mention_is_exact():
    text = " ".join(C2.RESERVES)
    for needle in ("mise à l'échelle par la GT", "vues trois fois", "bootstrap peu puissant", "Gallica et NewsEye", "LD-0R",
                   "0,934", "0,901", "0,756", "Le_Matin_0554721_T0000004", "plafonné vers 0,93", "couverture 0,998",
                   "LD-1B (rapport e7750d10…)", "0,843", "1,000", "piste de bord citée, sans effet sur ce lot"):
        assert needle in text, needle
    assert C2.ALTO_MENTION == "pas d'ALTO livré dans LD-2, règle du 21/10 (texte brut ET ALTO) non applicable"
    assert C2.CER_SCOPE.startswith("CER du texte brut des lignes appariées, " + C2.ALTO_MENTION) and "texte brut" in C2.CER_SCOPE
    assert C2.CER_SCOPE.endswith(" ; conditionnel aux lectures obtenues")                  # gel 8 (M1) : ajout, texte existant inchangé


# ---------- statique : exposition des pages d'évaluation ----------

def test_the_code_names_the_reserved_pages_only_in_comments_and_docstrings():
    for rel in ("scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/stage_s_notebooks/build_kaggle_ld2.py"):
        if not (ROOT / rel).is_file():                                         # constructeur absent du kit Kaggle
            continue
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "mf_005" in node.value:
                assert re.fullmatch(r"(?s).*mf_0050-0059.*", node.value), (rel, node.value[:60])
                assert len(node.value) > 60, (rel, node.value)               # une phrase de docstring, jamais un identifiant


@NEEDS_CARNET
def test_the_carnets_never_name_the_reserved_pages_or_list_a_directory():
    for name in ("kaggle_ld2.ipynb", "kaggle_ld2_smoke.ipynb"):
        code = _code(name).replace('RUN.glob("ld2_*_step*.pt")', "")           # nettoyage de NOS checkpoints de fumée
        for bad in ("glob(", "rglob", "listdir", "scandir", "os.walk", "iterdir", "mf_005"):
            assert bad not in code, (name, bad)


def _code(name):
    nb = json.loads((LD2 / name).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


# ---------- carnets : égalité avec le constructeur, métadonnées, taille, kit ----------

@NEEDS_CARNET
def test_the_carnets_on_disk_equal_the_builder_output():
    B2 = _b()
    assert json.loads((LD2 / "kaggle_ld2.ipynb").read_text()) == B2.build_notebook()
    assert json.loads((LD2 / "kaggle_ld2_smoke.ipynb").read_text()) == B2.build_smoke_notebook()
    assert json.loads((LD2 / "full_kernel" / "notebook.ipynb").read_text()) == B2.build_notebook()
    assert json.loads((LD2 / "smoke_kernel" / "notebook.ipynb").read_text()) == B2.build_smoke_notebook()
    assert json.loads((LD2 / "full_kernel" / "kernel-metadata.json").read_text()) == B2.kernel_metadata(
        "mabaku/babaorum-ld2-run", "babaorum-ld2-run", gpu=True)
    assert json.loads((LD2 / "smoke_kernel" / "kernel-metadata.json").read_text()) == B2.kernel_metadata(
        "mabaku/babaorum-ld2-smoke", "babaorum-ld2-smoke")
    assert json.loads((LD2 / "kit_files.json").read_text()) == B2.kit_files()


@NEEDS_CARNET
def test_the_cell_of_the_full_notebook_is_the_builder_cell_and_the_notebooks_are_small():
    B2 = _b()
    assert _code("kaggle_ld2.ipynb") == B2.cell("full") and _code("kaggle_ld2_smoke.ipynb") == B2.cell("smoke")
    for p in (LD2 / "full_kernel" / "notebook.ipynb", LD2 / "smoke_kernel" / "notebook.ipynb"):
        assert p.stat().st_size < 300 * 1024


@NEEDS_CARNET
def test_kernel_metadata_gpu_for_the_full_run_cpu_for_the_smoke_and_the_three_sources():
    full = json.loads((LD2 / "full_kernel" / "kernel-metadata.json").read_text())
    smoke = json.loads((LD2 / "smoke_kernel" / "kernel-metadata.json").read_text())
    assert full["enable_gpu"] is True and smoke["enable_gpu"] is False and full["enable_internet"] and smoke["enable_internet"]
    assert full["is_private"] is True and full["code_file"] == "notebook.ipynb" and full["id"] == "mabaku/babaorum-ld2-run"
    assert full["dataset_sources"] == ["mabaku/babaorum-ld2-kit"]
    assert set(full["kernel_sources"]) == {"mabaku/babaorum-e2e5-lambda", "mabaku/babaorum-e2e3-run3-fp-classes",
                                           "mabaku/babaorum-ld1-run"}


@NEEDS_CARNET
def test_the_kit_contains_the_closure_the_configs_and_the_scripts_and_every_file_exists():
    files = set(json.loads((LD2 / "kit_files.json").read_text()))
    for rel in files:
        assert (ROOT / rel).is_file(), rel
    for rel in ("scripts/train_ld2.py", "scripts/ld2_core.py", "config/ld2.json", "config/ld1_line_detr.json",
                "config/stage_s_smoke_v2.json", "config/data2_pages.json", "scripts/data2_common.py", "scripts/ld0_zone_coverage.py",
                "scripts/train_line_detr.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py",
                "src/babaorum/experimental/line_detr.py", "src/babaorum/experimental/stage_s_arms.py",
                "src/babaorum/experimental/stage_s_reader.py", "src/babaorum/experimental/stage_s_model.py",
                "src/babaorum/experimental/stage_s_targets.py", "src/babaorum/experimental/joint_curves.py",
                "src/babaorum/line_geometry.py", "tests/test_ld2.py", "tests/test_ld2_heavy.py",
                "scripts/stage_s_notebooks/KAGGLE_INSTALL.txt"):
        assert rel in files, rel
    for rel in (CFG["inherit_ld1"]["path"], CFG["stage_s"]["config"]):
        assert rel in files


def test_kaggle_install_is_unchanged():
    install = (CARNETS / "KAGGLE_INSTALL.txt").read_bytes()
    assert hashlib.sha256(install).hexdigest().startswith("a0cda2416808a1ab")


# ---------- montages : exactement un candidat, aucun listage, leurre mf_0050 au niveau 3, deux formats ----------

def _mount_tree(tmp_path, fmt):
    """fmt A : /kaggle/input/<slug> ; fmt B : /kaggle/input/{notebooks,datasets}/mabaku/<slug> (niveau 3)."""
    inp = tmp_path / "input"
    base = (lambda slug, sub: inp / slug) if fmt == "A" else (lambda slug, sub: inp / sub / "mabaku" / slug)
    kit = base("babaorum-ld2-kit", "datasets")
    for sub in ("config", "scripts", "src"):
        (kit / sub).mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "data2_common.py", kit / "scripts")
    shutil.copy(ROOT / "scripts" / "ld2_core.py", kit / "scripts")
    shutil.copy(ROOT / "config" / "data2_pages.json", kit / "config")
    (kit / "config" / "ld2.json").write_text("{}")
    mf = base("babaorum-e2e3-run3-fp-classes", "notebooks") / "e2e3" / "mf"
    (mf / "pages").mkdir(parents=True)
    (mf / "pages" / "mf_0000.json").write_text("{}")
    (mf / "pages" / "mf_0050.json").write_text("{}")                         # leurre sous e2e3/mf/pages
    e5 = base("babaorum-e2e5-lambda", "notebooks")
    (e5 / "logs").mkdir(parents=True)
    (e5 / "logs" / "checkpoint_last.pt").write_bytes(b"enc")
    (e5 / "e2e5_lambda_run_report.json").write_text("{}")
    ld1 = base("babaorum-ld1-run", "notebooks") / "ld1"
    ld1.mkdir(parents=True)
    (ld1 / "ld1_step001871.pt").write_bytes(b"dec")
    (ld1 / "ld1_step001000.pt").write_bytes(b"old")                          # jamais lu
    decoy3 = inp / "x" / "y" / "mf_0050.json"                                # leurre au niveau 3
    decoy3.parent.mkdir(parents=True)
    decoy3.write_text("{}")
    return inp, kit, mf


@NEEDS_CARNET
@pytest.mark.parametrize("fmt", ["A", "B"])
def test_the_mount_resolution_of_the_cell_never_lists_and_never_touches_a_reserved_page(tmp_path, monkeypatch, fmt):
    B2 = _b()
    inp, kit, mf = _mount_tree(tmp_path, fmt)
    cands = lambda a, b: [str(inp / a), str(inp / b)]                         # noqa: E731
    src = B2.cell("full", out=str(tmp_path / "out"), init_sha=hashlib.sha256(b"enc").hexdigest(),
                  dec_sha=hashlib.sha256(b"dec").hexdigest(),
                  kit=[str(inp / "babaorum-ld2-kit"), str(inp / "datasets" / "mabaku" / "babaorum-ld2-kit")],
                  pages=[str(inp / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf"),
                         str(inp / "notebooks" / "mabaku" / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf")],
                  ckpt=[str(inp / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt"),
                        str(inp / "notebooks" / "mabaku" / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt")],
                  ld1=[str(inp / "babaorum-ld1-run" / "ld1" / "ld1_step001871.pt"),
                       str(inp / "notebooks" / "mabaku" / "babaorum-ld1-run" / "ld1" / "ld1_step001871.pt")],
                  ref=[str(inp / "babaorum-e2e5-lambda" / "e2e5_lambda_run_report.json"),
                       str(inp / "notebooks" / "mabaku" / "babaorum-e2e5-lambda" / "e2e5_lambda_run_report.json")])
    head = src.split("\ndef _argv(")[0]                                       # en-tête + résolveurs, sans le pipeline
    touched, reads = [], []
    real_open, real_text, real_bytes = io.open, Path.read_text, Path.read_bytes
    monkeypatch.setattr(os, "listdir", lambda *a, **k: touched.append("listdir") or [])
    monkeypatch.setattr(os, "scandir", lambda *a, **k: touched.append("scandir") or iter(()))
    monkeypatch.setattr(os, "walk", lambda *a, **k: touched.append("walk") or iter(()))
    for name in ("glob", "iterdir", "rglob"):
        monkeypatch.setattr(Path, name, lambda *a, _n=name, **k: touched.append(_n) or iter(()))
    spy = lambda f, *a, **k: (reads.append(os.fspath(f)), real_open(f, *a, **k))[1]                     # noqa: E731
    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(os.fspath(self)), real_text(self, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: (reads.append(os.fspath(self)), real_bytes(self, *a, **k))[1])
    ns = {"__name__": "ld2_mount_test"}
    saved = list(sys.path)
    exec(compile(head, "ld2_head", "exec"), ns)
    monkeypatch.undo()
    sys.path[:] = saved
    sys.modules.pop("ld2_core", None)
    assert touched == [] and not any("mf_005" in r for r in reads)
    assert str(ns["PAGES"]).endswith("e2e3/mf") and "step001000" not in str(ns["LD1CK"]) and ns["LD1CK"].name == "ld1_step001871.pt"
    assert ns["KIT"].name == "babaorum-ld2-kit"
    assert not any(str(r).endswith("ld1_step001000.pt") for r in reads)


@NEEDS_CARNET
def test_the_mount_resolution_refuses_a_wrong_checkpoint_sha_and_two_candidates(tmp_path):
    B2 = _b()
    inp, kit, mf = _mount_tree(tmp_path, "A")
    base = dict(out=str(tmp_path / "o"), kit=[str(inp / "babaorum-ld2-kit")], pages=[str(mf)],
                ckpt=[str(inp / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt")],
                ld1=[str(inp / "babaorum-ld1-run" / "ld1" / "ld1_step001871.pt")], ref=[], init_sha=hashlib.sha256(b"enc").hexdigest(),
                dec_sha=hashlib.sha256(b"dec").hexdigest())
    head = lambda **kw: B2.cell("full", **{**base, **kw}).split("\ndef _argv(")[0]                           # noqa: E731
    with pytest.raises(AssertionError, match="décodeur"):
        exec(compile(head(dec_sha="0" * 64), "x", "exec"), {})
    sys.modules.pop("ld2_core", None)
    with pytest.raises(AssertionError, match="encodeur"):
        exec(compile(head(init_sha="0" * 64), "x", "exec"), {})
    sys.modules.pop("ld2_core", None)
    with pytest.raises(RuntimeError):
        exec(compile(head(kit=[str(inp / "babaorum-ld2-kit")] * 2), "x", "exec"), {})
    sys.modules.pop("ld2_core", None)


# ---------- comportement du vrai corps du carnet complet (faux run_step, fausse horloge) ----------

def _data2_common():
    spec = importlib.util.spec_from_file_location("data2_common_ld2test", ROOT / "scripts" / "data2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Run:
    """Exécute le vrai FULL_BODY. plan[étape] = {"rc", "dur", "exit_status", "estimate"} ; l'horloge avance de dur à chaque étape."""

    def __init__(self, tmp_path, monkeypatch, plan, cuda=True, missing_pkg=None, ref_ok=True):
        B2 = _b()
        self.B2, self.calls, self.clock, self.plan = B2, [], [0.0], plan
        self.out = tmp_path / "work"
        self.out.mkdir()
        self.C = _data2_common()
        monkeypatch.setattr(self.C, "run_step", self.run_step)
        tree = ast.parse(B2.cell("full"))
        lits = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("LIMITS", "TIMEOUTS")}
        self.limits, self.timeouts = lits["LIMITS"], lits["TIMEOUTS"]
        ns = {"time": types.SimpleNamespace(monotonic=lambda: self.clock[0]), "json": json, "os": os, "sys": sys, "Path": Path,
              "OUT": self.out, "C": self.C, "C2": C2, "T_START": 0.0, "CKPT": tmp_path / "e.pt", "LD1CK": tmp_path / "d.pt",
              "REF_CANDIDATES": [str(tmp_path / "ref.json")], "resolve_file": lambda c: Path(c[0]), "shlex": shlex,
              "REF_PREFIX": "3abfe0d7", "EST_FACTOR": CFG["eval"]["estimate_factor"], "KIT": Path("/k"), "PAGES": Path("/p"),
              "LIMITS": self.limits, "TIMEOUTS": self.timeouts, "INIT_STEP": 2000, "DEC_STEP": 1871,
              "sha256": lambda p: ("3abfe0d7" if ref_ok or Path(p).name != "ref.json" else "deadbeef") + "0" * 56,
              "_utc": lambda: "00:00:00"}
        exec(compile(B2._argv_builder("cuda"), "<_argv_builder>", "exec"), ns)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        fake_torch.load = lambda p, **k: {"model": {}, "step": 1871 if Path(p).name == "d.pt" else 2000}
        fake_torch.__version__ = "9.9"
        fake_torch.version = types.SimpleNamespace(cuda="12.x")
        ld = types.ModuleType("babaorum.experimental.line_detr")
        ld.LineDetr = lambda: types.SimpleNamespace(state_dict=lambda: {})
        ld.load_enc_det_from_e2e5 = lambda *a, **k: None
        for name, mod in (("torch", fake_torch), ("babaorum", types.ModuleType("babaorum")),
                          ("babaorum.experimental", types.ModuleType("babaorum.experimental")),
                          ("babaorum.experimental.line_detr", ld)):
            monkeypatch.setitem(sys.modules, name, mod)

        def version(name):
            if name == missing_pkg:
                raise importlib.metadata.PackageNotFoundError(name)
            return f"v-{name}"

        monkeypatch.setattr(importlib.metadata, "version", version)
        exec(compile(B2.FULL_BODY, "<FULL_BODY>", "exec"), ns)
        self.ns = ns

    def run_step(self, name, argv, timeout_s, logs_dir, cwd=None, env=None, kill_grace_s=10.0):
        p = {"rc": 0, "dur": 1.0, "exit_status": None, "estimate": 100.0, **self.plan.get(name, {})}
        self.calls.append((name, timeout_s, list(argv)))
        self.clock[0] += p["dur"]
        run = self.out / "ld2"
        if name == "T" and p["rc"] == 0:
            run.mkdir(parents=True, exist_ok=True)
            (run / "ld2_manifest.json").write_text(json.dumps({"S2": 2800, "S_t": 0}))
        if name == "TW" and p["rc"] == 0 and argv[0] == "sh" and (run / "ld2_manifest.json").is_file():
            man = json.loads((run / "ld2_manifest.json").read_text())
            man["S_t"] = 1400
            (run / "ld2_manifest.json").write_text(json.dumps(man))
        if name == "CAL" and p["rc"] == 0:
            (self.out / "ld2_cal").mkdir(parents=True, exist_ok=True)
            (self.out / "ld2_cal" / "ld2_cal.json").write_text(json.dumps({"estimate_s": p["estimate"], "page": "mf_0045"}))
        if p["exit_status"]:
            ex = self.out / "exits"
            ex.mkdir(exist_ok=True)
            (ex / f"ld2_run_exit_{name}.json").write_text(json.dumps({"status": p["exit_status"]}))
        return {"step": name, "rc": p["rc"], "seconds": p["dur"], "stdout": "", "stderr": "", "stderr_tail": ""}

    @property
    def exit(self):
        return self.ns["EXIT"]

    @property
    def names(self):
        return [c[0] for c in self.calls]

    def argv_of(self, name):
        return dict((c[0], c[2]) for c in self.calls)[name]

    def deadline_of(self, name):
        words = shlex.split(self.argv_of(name)[2])
        return float(words[words.index("--deadline-s") + 1].rstrip(";"))

    def timeout(self, name):
        return dict((c[0], c[1]) for c in self.calls)[name]


@NEEDS_CARNET
def test_the_normal_run_goes_D_T_TW_E_EW_FREEZE_CAL_EVAL_with_caps_derived_from_the_remaining_time(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 60.0}, "T": {"dur": 400.0}, "TW": {"dur": 300.0}, "E": {"dur": 8700.0},
                                   "EW": {"dur": 3000.0}, "FREEZE": {"dur": 5.0}, "CAL": {"dur": 200.0, "estimate": 900.0}})
    assert r.exit["status"] == "DONE" and r.names == ["D", "T", "TW", "E", "EW", "FREEZE", "CAL", "EVAL"]
    lim, to = r.limits, r.timeouts
    tail = lim["tail_reserve_s"] + lim["stop_margin_s"]
    assert (lim["deadline_s"], to["E"], to["EW"], to["T"], to["TW"]) == (19800, 9000, 4800, 2700, 1500)
    assert r.timeout("D") == to["D"] and r.timeout("T") == to["T"]
    assert r.timeout("E") == min(9000, 19800 - 760.0 - tail)                         # D + T + TW déjà consommés
    assert r.deadline_of("EW") == min(4800, 19800 - 9460.0 - tail) and r.timeout("EW") == r.deadline_of("EW") + lim["ew_grace_s"]
    assert r.deadline_of("TW") == min(1500, 19800 - 460.0 - tail) and r.timeout("TW") == r.deadline_of("TW") + lim["ew_grace_s"]
    assert r.timeout("EVAL") == 19800 - (60.0 + 400.0 + 300.0 + 8700.0 + 3000.0 + 5.0 + 200.0) - lim["eval_margin_s"]
    est = json.loads((r.out / "ld2_eval_estimate.json").read_text())
    assert est["estimate_s"] == 900.0 * 1.5 and est["factor"] == 1.5 and est["margin_s"] == lim["eval_margin_s"]


@NEEDS_CARNET
def test_E_gets_the_remaining_time_minus_the_tail_reserve_when_less_than_its_cap(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 9000.0}})
    tail = r.limits["tail_reserve_s"] + r.limits["stop_margin_s"]
    assert r.timeout("E") == 19800 - 9002.0 - tail < 9000


@NEEDS_CARNET
def test_the_eval_cap_is_what_remains_minus_the_margin_after_the_estimation_gate(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    lim = r.limits
    expect = 19800 - 7.0 - lim["eval_margin_s"]                                    # 7 étapes de 1 s avant EVAL
    assert r.timeout("EVAL") == expect


@NEEDS_CARNET
def test_every_long_phase_is_an_argv_step_not_an_fn_step(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    steps = r.ns["STEPS"]
    fn_steps = [s["name"] for s in steps if "fn" in s]
    argv_steps = [s["name"] for s in steps if "argv" in s]
    assert fn_steps == ["controls", "estimate"] and argv_steps == ["D", "T", "TW", "E", "EW", "FREEZE", "CAL", "EVAL"]


@NEEDS_CARNET
def test_E_timeout_is_INCOMPLETE_at_E_and_nothing_after_it_runs_so_no_evaluation(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"E": {"rc": "TIMEOUT"}})
    assert r.names == ["D", "T", "TW", "E"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "E"
    assert not any(n in r.names for n in ("EW", "FREEZE", "CAL", "EVAL"))


@NEEDS_CARNET
def test_a_step_failure_before_E_stops_the_run_and_a_zero_delay_gives_INCOMPLETE_without_launching(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"T": {"rc": 1}})
    assert r.names == ["D", "T"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "T"
    (tmp_path / "b").mkdir()
    r = Run(tmp_path / "b", monkeypatch, {"D": {"dur": 19800.0}})
    assert r.names == ["D"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "T"
    (tmp_path / "c").mkdir()
    r = Run(tmp_path / "c", monkeypatch, {"T": {"exit_status": "FAILED", "rc": 1}})              # erreur de device à T : FAILED
    assert r.exit["status"] == "FAILED" and r.exit["step"] == "T"


@NEEDS_CARNET
def test_without_time_the_witness_is_not_launched_but_the_main_evaluation_continues(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"E": {"dur": 17300.0}})                         # reste 2 498 s < réserve de queue (2 520 s)
    assert r.names == ["D", "T", "TW", "E", "EW", "FREEZE", "CAL", "EVAL"]
    ew = dict((c[0], c[2]) for c in r.calls)["EW"]
    assert ew[-2:] == ["-c", "pass"]                                                # étape factice : rien ne tourne
    st = json.loads((r.out / "ld2" / "ld2_witness_status.json").read_text())
    assert st["status"] == "NOT_RUN" and st["S2"] == 2800 and st["S_t"] == 1400 and r.exit["status"] == "DONE"


@NEEDS_CARNET
def test_the_estimation_above_the_remaining_budget_is_INCOMPLETE_before_any_evaluation_page_is_opened(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"CAL": {"estimate": 14000.0}})                  # x 1,5 = 21 000 s ; disponible : 19800 - 7 - 300 = 19 493 s
    assert r.names == ["D", "T", "TW", "E", "EW", "FREEZE", "CAL"] and "EVAL" not in r.names
    assert r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "estimate"
    est = json.loads((r.out / "ld2_eval_estimate.json").read_text())
    assert est["estimate_s"] > est["available_s"] and est["available_s"] == 19800 - 7.0 - r.limits["eval_margin_s"]
    (tmp_path / "ok").mkdir()
    fits = Run(tmp_path / "ok", monkeypatch, {"CAL": {"estimate": est["available_s"] / 1.5 - 1.0}})
    assert "EVAL" in fits.names and fits.exit["status"] == "DONE"


@NEEDS_CARNET
def test_gel3_6_the_estimate_is_multiplied_by_the_config_factor_before_the_comparison(tmp_path, monkeypatch):
    assert CFG["eval"]["estimate_factor"] == 1.5
    unscaled = 14000.0                                                              # tiendrait sans le facteur (19 493 s disponibles)
    r = Run(tmp_path, monkeypatch, {"CAL": {"estimate": unscaled}})
    assert "EVAL" not in r.names and r.exit["step"] == "estimate"
    assert json.loads((r.out / "ld2_eval_estimate.json").read_text())["estimate_s"] == unscaled * 1.5


@NEEDS_CARNET
def test_gel3_4_controls_refuse_a_reference_report_whose_sha_does_not_start_with_3abfe0d7(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {}, ref_ok=False)
    assert r.names == [] and r.exit["status"] == "FAILED" and r.exit["step"] == "controls" and "référence" in r.exit["exception"]
    (tmp_path / "ok").mkdir()
    assert Run(tmp_path / "ok", monkeypatch, {}).exit["status"] == "DONE"
    assert _b().cell("full").count("REF_PREFIX = '3abfe0d7'") == 1 and "mf_005" not in _b().cell("full")


@NEEDS_CARNET
def test_gel3_1_T_and_TW_are_separate_steps_T_before_TW_before_E_and_S_t_stays_zero_without_TW(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 17400.0}})                          # reste 2 399 s : TW (réserve de queue) non lancé
    assert r.names[:3] == ["D", "T", "TW"] and r.argv_of("TW")[-2:] == ["-c", "pass"]
    man = json.loads((r.out / "ld2" / "ld2_manifest.json").read_text())
    assert man["S2"] == 2800 and man["S_t"] == 0                                      # S2 écrit par T seul ; S_t = 0
    (tmp_path / "n").mkdir()
    n = Run(tmp_path / "n", monkeypatch, {})
    assert json.loads((n.out / "ld2" / "ld2_manifest.json").read_text())["S_t"] == 1400
    assert n.names.index("T") < n.names.index("TW") < n.names.index("E")


@NEEDS_CARNET
def test_gel3_2_TW_and_EW_commands_are_wrapped_so_the_step_always_returns_zero(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    for ph in ("TW", "EW"):
        argv = r.argv_of(ph)
        assert argv[:2] == ["sh", "-c"] and f"ld2_{ph}_rc.txt" in argv[2] and argv[2].rstrip().endswith("exit 0")
        assert "train_ld2.py" in argv[2] and f"--phase {ph}" in argv[2]
    for ph in ("D", "T", "E", "FREEZE", "CAL", "EVAL"):
        assert r.argv_of(ph)[0] == sys.executable                                    # les autres étapes ne sont pas enveloppées


def _timeout_on_path(tmp_path, monkeypatch):
    """coreutils `timeout` si présent ; sinon (macOS) un substitut minimal avec la même syntaxe `timeout -s KILL secs cmd...`."""
    import shutil
    if shutil.which("timeout"):
        return "coreutils"
    shim = tmp_path / "bin" / "timeout"
    shim.parent.mkdir()
    shim.write_text(f"#!{sys.executable}\nimport subprocess, sys\n"
                    "secs, cmd = float(sys.argv[3]), sys.argv[4:]\n"
                    "p = subprocess.Popen(cmd)\n"
                    "try:\n    rc = p.wait(timeout=secs)\n    sys.exit(128 - rc if rc < 0 else rc)\n"
                    "except subprocess.TimeoutExpired:\n    p.kill(); p.wait(); sys.exit(137)\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}{os.pathsep}{os.environ['PATH']}")
    return "substitut"


@NEEDS_CARNET
def test_gel3_2_a_wrapped_step_returns_zero_after_a_crash_and_keeps_the_rc_in_a_file(tmp_path, monkeypatch):
    _timeout_on_path(tmp_path, monkeypatch)
    ns = {"sys": sys, "shlex": shlex, "Path": Path, "KIT": Path("/k"), "PAGES": Path("/p"), "CKPT": Path("/c"), "LD1CK": Path("/d"),
          "RUN": tmp_path / "out" / "ld2"}
    exec(compile(_b()._argv_builder("cuda"), "x", "exec"), ns)
    for code, want in (("raise SystemExit(7)", "7"), ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", "137"), ("pass", "0")):
        argv = ns["wrapped"]("EW", [sys.executable, "-c", code], 30)
        assert subprocess.run(argv).returncode == 0
        assert (tmp_path / "out" / "exits" / "ld2_EW_rc.txt").read_text().strip() == want


@NEEDS_CARNET
def test_gel5_a_command_still_running_at_inner_timeout_is_killed_before_the_notebook_timeout_and_sh_writes_137(tmp_path, monkeypatch):
    kind = _timeout_on_path(tmp_path, monkeypatch)
    ns = {"sys": sys, "shlex": shlex, "Path": Path, "KIT": Path("/k"), "PAGES": Path("/p"), "CKPT": Path("/c"), "LD1CK": Path("/d"),
          "RUN": tmp_path / "out" / "ld2"}
    exec(compile(_b()._argv_builder("cuda"), "x", "exec"), ns)
    argv = ns["wrapped"]("TW", [sys.executable, "-c", "import time; time.sleep(30)"], 0.5)
    t0 = __import__("time").monotonic()
    assert subprocess.run(argv, timeout=15).returncode == 0                       # sh survit à la mort de la commande
    assert __import__("time").monotonic() - t0 < 10
    assert (tmp_path / "out" / "exits" / "ld2_TW_rc.txt").read_text().strip() == "137", kind


@NEEDS_CARNET
def test_gel5_the_built_argv_has_timeout_kill_strictly_between_the_watchdog_and_the_notebook_timeout(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    grace = r.limits["ew_grace_s"]
    for ph in ("TW", "EW"):
        words = shlex.split(r.argv_of(ph)[2])
        assert words[:3] == ["timeout", "-s", "KILL"]
        inner, deadline = float(words[3]), r.deadline_of(ph)
        assert inner == round(deadline + 0.75 * grace, 1)
        assert deadline + 0.5 * grace < inner < r.timeout(ph) == deadline + grace


@NEEDS_CARNET
def test_the_run_exit_is_written_in_a_finally_and_named_ld2_and_the_controls_refuse_without_cuda(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {}, cuda=False)
    assert r.names == [] and r.exit["status"] == "FAILED" and r.exit["step"] == "controls" and "CUDA" in r.exit["exception"]
    tree = ast.parse(_code("kaggle_ld2.ipynb"))
    finals = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert "ld2_run_exit.json" in " ".join(ast.unparse(f) for t in finals for f in t.finalbody)
    (tmp_path / "e").mkdir()
    r = Run(tmp_path / "e", monkeypatch, {}, missing_pkg="transformers")
    assert r.names == [] and r.exit["status"] == "FAILED" and "transformers" in r.exit["exception"]
    (tmp_path / "f").mkdir()
    r = Run(tmp_path / "f", monkeypatch, {})
    assert set(json.loads((r.out / "ld2_env.json").read_text())) == {"python", "torch", "cuda", "numpy", "PIL", "transformers",
                                                                     "huggingface_hub"}


@NEEDS_CARNET
def test_T_START_is_the_first_statement_and_the_limits_come_from_the_config(tmp_path, monkeypatch):
    B2 = _b()
    tree = ast.parse(B2.cell("full"))
    assert ast.unparse(tree.body[0]) == "import time" and ast.unparse(tree.body[1]) == "T_START = time.monotonic()"
    lim = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
           and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("LIMITS", "TIMEOUTS")}
    rb = CFG["robustness"]
    assert lim["LIMITS"] == {k: rb[k] for k in ("deadline_s", "tail_reserve_s", "stop_margin_s", "eval_margin_s", "ew_grace_s")}
    assert lim["TIMEOUTS"] == rb["timeouts_s"]
    cfg_file = tmp_path / "cfg.json"
    cfg = json.loads(json.dumps(CFG))
    cfg["robustness"]["deadline_s"] = 1234
    cfg["robustness"]["timeouts_s"]["E"] = 5555
    cfg_file.write_text(json.dumps(cfg))
    monkeypatch.setattr(B2, "CONFIG", cfg_file)
    lim2 = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(B2.cell("full")).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("LIMITS", "TIMEOUTS")}
    assert lim2["LIMITS"]["deadline_s"] == 1234 and lim2["TIMEOUTS"]["E"] == 5555


# ---------- fumée : jamais de chemin mf_005x, pages de remplacement, référence factice ----------

@NEEDS_CARNET
def test_the_smoke_cell_passes_smoke_and_decoy_pages_to_EVAL_and_never_names_a_reserved_page(tmp_path, monkeypatch):
    B2 = _b()
    assert "mf_005" not in B2.cell("smoke") and "mf_005" not in _code("kaggle_ld2_smoke.ipynb")
    assert all(p in [f"mf_{i:04d}" for i in range(0, 50)] for p in B2.DECOY_PAGES)
    calls, caps = [], {}

    def fake_execute(out, steps, heartbeat_interval_s=60.0, cleanup=None):
        for st in steps:
            if "fn" in st:
                st["fn"]()
            else:
                argv = st["argv"]() if callable(st["argv"]) else st["argv"]
                calls.append((st["name"], [str(a) for a in argv]))
                caps[st["name"]] = st["timeout_s"]
        return {"status": "DONE", "step": None, "exit_code": 0, "durations_s": {}}

    inp, kit, mf = _mount_tree(tmp_path, "A")
    src = B2.cell("smoke", out=str(tmp_path / "out"), init_sha=hashlib.sha256(b"enc").hexdigest(),
                  dec_sha=hashlib.sha256(b"dec").hexdigest(), kit=[str(kit)], pages=[str(mf)],
                  ckpt=[str(inp / "babaorum-e2e5-lambda" / "logs" / "checkpoint_last.pt")],
                  ld1=[str(inp / "babaorum-ld1-run" / "ld1" / "ld1_step001871.pt")])
    real_spec = importlib.util.spec_from_file_location

    def spec_with_fake_loader(name, path, *a, **k):
        spec = real_spec(name, path, *a, **k)
        if name == "data2_common":
            spec.loader = types.SimpleNamespace(create_module=lambda s: None,
                                                exec_module=lambda m: setattr(m, "execute_pipeline", fake_execute))
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec_with_fake_loader)
    saved = list(sys.path)
    exec(compile(src, "ld2_smoke_cell", "exec"), {"__name__": "smoke"})
    sys.path[:] = saved
    sys.modules.pop("ld2_core", None)
    by = dict(calls)
    assert [c[0] for c in calls] == ["D", "pytest_heavy", "T", "TW", "E", "EW", "FREEZE", "CAL", "EVAL"]
    assert by["TW"][:2] == ["sh", "-c"] and by["EW"][:2] == ["sh", "-c"] and "ld2_TW_rc.txt" in by["TW"][2]
    assert not any("mf_005" in a for _n, argv in calls for a in argv)
    assert "--smoke" in by["EVAL"] and "--smoke" in by["CAL"] and by["EVAL"][by["EVAL"].index("--eval-pages") + 1:][:2] == ["mf_0000", "mf_0001"]
    assert "--reference-report" in by["EVAL"] and by["EVAL"][by["EVAL"].index("--reference-report") + 1].endswith("decoy_reference.json")
    ref = json.loads((tmp_path / "out" / "ld2_smoke" / "decoy_reference.json").read_text())
    assert [e["page_id"] for e in ref["eval"]] == ["mf_0000", "mf_0001"]
    assert "--smoke" not in by["T"] and "--pages" in by["T"] and by["T"][by["T"].index("--pages") + 1] == "mf_0000"
    assert "tests/test_ld2.py tests/test_ld2_heavy.py" in " ".join(by["pytest_heavy"]) and "-k" not in by["pytest_heavy"]
    assert "--device" in by["T"] and by["T"][by["T"].index("--device") + 1] == "cpu"
    assert caps["CAL"] == 5400 and caps["EVAL"] == 2 * caps["CAL"]                    # gel 7 : fumée CPU seulement
    cfg_t = CFG["robustness"]["timeouts_s"]
    full_t = {x.targets[0].id: ast.literal_eval(x.value) for x in ast.parse(_code("kaggle_ld2.ipynb")).body
              if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) and x.targets[0].id == "TIMEOUTS"}["TIMEOUTS"]
    assert cfg_t["CAL"] == 900 and full_t["CAL"] == 900                               # la config et le T4 gardent 900 s
    for ph in ("T", "TW"):                                                          # gel 5 (1) : 3 pas chronométrés en fumée
        words = shlex.split(by[ph][2]) if ph == "TW" else by[ph]
        assert words[words.index("--timing-steps") + 1] == "3", ph
    assert shlex.split(by["TW"][2])[:3] == ["timeout", "-s", "KILL"] and "--deadline-s" in shlex.split(by["TW"][2])


# ---------- gel 3 : points avec torch importé (aucun calcul de réseau) ----------

@pytest.fixture
def TD():
    pytest.importorskip("torch")
    import train_ld2
    return train_ld2


def _manifest(run, s_t=0, witness=None):
    run.mkdir(parents=True, exist_ok=True)
    (run / "ld2_manifest.json").write_text(json.dumps({"S2": 3, "S_t": s_t, "alphabet": ["<blank>", "a"],
                                                       "witness": witness or {"status": "NOT_TIMED"}, "init": {}}))


def test_gel3_1_TW_failure_gives_S_t_zero_and_a_NOT_RUN_witness_and_never_raises(TD, tmp_path, monkeypatch):
    run = tmp_path / "out" / "ld2"
    _manifest(run)
    monkeypatch.setattr(TD.TL, "check_cuda", lambda d: None)
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: {})
    monkeypatch.setattr(TD, "time_kind", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("out of memory (hôte)")))
    args = types.SimpleNamespace(device="cpu", root=str(tmp_path), pages=["mf_0000"], deadline_s=None)
    wit = TD.phase_TW(args, {"steps": CFG["steps"]}, run)
    man = json.loads((run / "ld2_manifest.json").read_text())
    assert man["S_t"] == 0 and wit["status"] == "TIMING_FAILED" and "out of memory" in wit["reason"]
    assert C2.witness_report(run)["status"] == "NOT_RUN"
    assert TD.phase_EW(types.SimpleNamespace(deadline_s=None), CFG, run)["status"] == "NOT_RUN"      # S_t = 0 : EW ne s'entraîne pas


def test_gel3_1_TW_success_writes_S_t_from_the_formula_and_S2_is_untouched(TD, tmp_path, monkeypatch):
    run = tmp_path / "out" / "ld2"
    _manifest(run)
    monkeypatch.setattr(TD.TL, "check_cuda", lambda d: None)
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: {})
    monkeypatch.setattr(TD, "time_kind", lambda *a, **k: {"s_per_step": 2.0, "batch_size": 4, "key_sources": {"k": "v"}})
    args = types.SimpleNamespace(device="cpu", root=str(tmp_path), pages=["mf_0000"], deadline_s=None)
    TD.phase_TW(args, CFG, run)
    man = json.loads((run / "ld2_manifest.json").read_text())
    assert man["S2"] == 3 and man["S_t"] == min(3, 4200 // 2) and man["witness"]["status"] == "TIMED"
    assert "T" in TD.PHASES and "TW" in TD.PHASES and TD.PHASES.index("T") < TD.PHASES.index("TW") < TD.PHASES.index("E")


def test_gel3_2_witness_report_reads_the_exit_code_files_and_reports_a_hard_stop_as_is(tmp_path):
    run = tmp_path / "out" / "ld2"
    ex = tmp_path / "out" / "exits"
    ex.mkdir(parents=True)
    _manifest(run, s_t=100, witness={"status": "TIMED"})
    (ex / "ld2_TW_rc.txt").write_text("0\n")
    (ex / "ld2_EW_rc.txt").write_text("137\n")
    rep = C2.witness_report(run)
    assert rep["status"] == "HARD_STOP" and rep["ew_rc"] == "137" and rep["tw_rc"] == "0" and rep["timing"] == "TIMED"
    (run / "ld2_witness_status.json").write_text(json.dumps({"status": "COMPLETE", "checkpoint": "x.pt"}))
    assert C2.witness_report(run)["status"] == "COMPLETE" and C2.witness_report(run)["ew_rc"] == "137"
    _manifest(run, s_t=0)
    assert C2.witness_report(run)["status"] == "NOT_RUN"


def test_gel3_2_FREEZE_records_the_witness_exit_codes_as_they_are(TD, tmp_path):
    run = tmp_path / "out" / "ld2"
    ex = tmp_path / "out" / "exits"
    ex.mkdir(parents=True)
    _manifest(run, s_t=100, witness={"status": "TIMED"})
    (ex / "ld2_EW_rc.txt").write_text("139\n")
    (run / "ld2_main_step000003.pt").write_bytes(b"ck")
    TD.phase_FREEZE(None, CFG, run)
    man = json.loads((run / "ld2_manifest.json").read_text())
    assert man["witness_status_at_freeze"]["status"] == "HARD_STOP" and man["witness_status_at_freeze"]["ew_rc"] == "139"


def test_gel3_5_FREEZE_writes_the_sha_of_train_ld2_ld2_core_and_the_config(TD, tmp_path):
    run = tmp_path / "out" / "ld2"
    _manifest(run)
    (run / "ld2_main_step000003.pt").write_bytes(b"ck")
    fz = TD.phase_FREEZE(None, CFG, run)
    for rel in ("scripts/train_ld2.py", "scripts/ld2_core.py", "config/ld2.json"):
        assert fz["frozen_files_sha256"][rel] == C2.sha256_file(ROOT / rel)
    assert set(fz["frozen_files_sha256"]) >= {"src/babaorum/experimental/line_detr.py", "scripts/train_line_detr.py"}


def test_gel3_3_a_witness_error_is_logged_in_the_page_row_and_a_main_error_stays_fatal(TD, monkeypatch):
    page = {"image": None}
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: {"blocks": [], "image": None})
    monkeypatch.setattr(TD, "zone_raws", lambda *a, **k: [])
    monkeypatch.setattr(TD.TL, "candidates_for_threshold", lambda raws, thr: [])
    monkeypatch.setattr(TD.TL, "pp_zones", lambda *a, **k: [])
    monkeypatch.setattr(TD, "score_lines", lambda page, kept, raws: {"n_pred": 0, "n_gt": 0, "precision": 0.0, "recall": 0.0})
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tuile"))                 # noqa: E731
    monkeypatch.setattr(TD, "witness_raw", boom)
    ctx = {"threshold": 0.6, "alphabet": [], "device": None, "main": object(), "witness": object(), "cache": None, "L0": None,
           "det_fn": None, "cfg": CFG}
    row = TD.evaluate_page("mf_0000", Path("."), ctx)
    assert row["witness"] == {"error": "RuntimeError: tuile"} and row["Z-PP-blocs"]["n_pred"] == 0
    monkeypatch.setattr(TD, "zone_raws", boom)
    with pytest.raises(RuntimeError):
        TD.evaluate_page("mf_0000", Path("."), ctx)


def test_gel3_3_the_cer_error_is_logged_without_losing_the_geometry_and_the_aggregate_skips_error_pages(TD):
    gt_line = {"id": "g0", "baseline": np.array([[0.0, 10.0], [300.0, 10.0]]), "neg": 12.0, "pos": 4.0, "text": "ab", "legibility": "readable"}
    page = {"gt_lines": [gt_line]}
    kept = [{"pts": np.array([[0.0, 10.0], [300.0, 10.0]]), "zone": 0, "query": 0, "score": 0.9, "h": 16.0}]
    res = TD.score_lines(page, kept, [(0, None, {"hyp": {0: 5}})])                       # hyp de type invalide : levenshtein échoue
    assert res["recall"] == 1.0 and res["precision"] == 1.0 and "error" in res["cer"]
    ok = {"page": "p1", "seconds": {"witness": 1.0}, "witness": dict(res, n_pred=1, n_gt=1)}
    bad = {"page": "p2", "seconds": {"witness": 1.0}, "witness": {"error": "RuntimeError: x"}}
    agg = TD.aggregate([ok, bad], "witness", 2)
    assert agg["pages"] == 1 and agg["pages_with_error"] == {"p2": "RuntimeError: x"} and agg["cer"]["pages_with_cer_error"] == 1
    assert TD.aggregate([bad], "witness", 1) == {"pages": 0, "pages_with_error": {"p2": "RuntimeError: x"}}


def test_gel3_7_the_report_limit_is_verbatim_and_written_in_the_report_code(TD):
    assert "down, l'encodeur du transformer et le décodeur viennent de LD-1" in C2.DECODER_PAIRING_LIMIT
    assert "C2.DECODER_PAIRING_LIMIT" in (ROOT / "scripts" / "train_ld2.py").read_text()
    if LOT.is_file():
        assert _norm(C2.DECODER_PAIRING_LIMIT) in _norm(LOT.read_text())


def test_gel4_A_the_hard_watchdog_is_armed_at_phase_start_and_fires_before_the_notebook_timeout(TD, tmp_path, monkeypatch):
    import time as _t
    run = tmp_path / "out" / "ld2"
    _manifest(run, s_t=5, witness={"status": "TIMED", "batch_size": 2})
    cfg = dict(CFG, robustness=dict(CFG["robustness"], ew_grace_s=0.4))          # watchdog à deadline + 0,2 s ; étape du carnet : deadline + 0,4 s
    monkeypatch.setattr(TD.TL, "check_cuda", lambda d: None)
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: _t.sleep(0.6) or {})   # chargement plus long que le délai de grâce
    monkeypatch.setattr(TD, "train_kind", lambda *a, **k: pytest.fail("aucun entraînement après un délai épuisé"))
    fired = []

    def on_fire():
        fired.append((_t.monotonic() - t0, json.loads((run / "ld2_witness_status.json").read_text())))

    args = types.SimpleNamespace(device="cpu", root=str(tmp_path), pages=["mf_0000"], deadline_s=0.1)
    t0 = _t.monotonic()
    st = TD.phase_EW(args, cfg, run, on_hard_fire=on_fire)
    assert len(fired) == 1 and fired[0][0] < 0.4 and fired[0][0] < 0.6            # avant la fin du chargement et avant le délai du carnet
    assert fired[0][1]["status"] == "NOT_REACHED" and "dur" in fired[0][1]["reason"]
    assert st["status"] == "NOT_REACHED" and st["steps_done"] == 0                  # délai doux <= 0 : pas d'entraînement


def test_gel4_A_TW_hard_watchdog_fires_during_a_slow_loading_and_leaves_S_t_zero(TD, tmp_path, monkeypatch):
    import time as _t
    run = tmp_path / "out" / "ld2"
    _manifest(run)
    cfg = dict(CFG, robustness=dict(CFG["robustness"], ew_grace_s=0.4))
    monkeypatch.setattr(TD.TL, "check_cuda", lambda d: None)
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: _t.sleep(0.6) or {})
    monkeypatch.setattr(TD, "time_kind", lambda *a, **k: pytest.fail("pas de chronométrage après un délai épuisé"))
    fired = []
    args = types.SimpleNamespace(device="cpu", root=str(tmp_path), pages=["mf_0000"], deadline_s=0.1)
    t0 = _t.monotonic()
    wit = TD.phase_TW(args, cfg, run, on_hard_fire=lambda: fired.append(_t.monotonic() - t0))
    assert len(fired) == 1 and fired[0] < 0.4
    man = json.loads((run / "ld2_manifest.json").read_text())
    assert man["S_t"] == 0 and wit["status"] == "TIMING_FAILED"


def test_gel4_A_with_time_left_after_loading_the_soft_deadline_is_reduced_by_the_loading_time(TD, tmp_path, monkeypatch):
    import time as _t
    run = tmp_path / "out" / "ld2"
    _manifest(run, s_t=5, witness={"status": "TIMED", "batch_size": 2})
    monkeypatch.setattr(TD.TL, "check_cuda", lambda d: None)
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: _t.sleep(0.2) or {})
    seen = {}
    monkeypatch.setattr(TD, "train_kind", lambda *a, **k: seen.update(k) or {"reached": True, "steps_done": 5})
    args = types.SimpleNamespace(device="cpu", root=str(tmp_path), pages=["mf_0000"], deadline_s=100.0)
    assert TD.phase_EW(args, CFG, run)["status"] == "COMPLETE"
    assert 99.0 < seen["deadline_s"] <= 99.8 and "on_start" not in seen


def test_gel4_B_the_main_aggregate_keeps_the_fixed_denominator_and_refuses_errors_or_missing_pages(TD):
    def row(p):
        return {"page": p, "seconds": {"Z-PP-blocs": 1.0}, "Z-PP-blocs": {"n_pred": 1, "n_gt": 1, "precision": 1.0, "recall": 1.0,
                "splits": 0, "merges": 0, "cer": {"edits": 0, "chars": 1, "lines": 1, "skipped": 0, "matched_readable_lines": 1,
                "matched_readable_chars": 1, "skipped_by_cause": {"degenerate_chord": 0, "band_outside_support": 0,
                                                                  "reader_exception": 0, "unknown": 0}}}}
    rows = [row(f"p{i}") for i in range(10)]
    assert TD.aggregate(rows, "Z-PP-blocs", 10)["pages"] == 10
    with pytest.raises(ValueError, match="attendu 10"):
        TD.aggregate(rows[:9], "Z-PP-blocs", 10)
    bad = rows[:9] + [{"page": "p9", "seconds": {"Z-PP-blocs": 1.0}, "Z-PP-blocs": {"error": "x"}}]
    with pytest.raises(ValueError, match="en erreur"):
        TD.aggregate(bad, "Z-PP-blocs", 10)
    wrows = [dict(r, witness=r["Z-PP-blocs"], seconds={"witness": 1.0}) for r in rows[:9]]
    wrows.append({"page": "p9", "seconds": {"witness": 1.0}, "witness": {"error": "x"}})
    agg = TD.aggregate(wrows, "witness", 10)
    assert agg["pages"] == 9 and agg["pages_with_error"] == {"p9": "x"}               # le témoin garde len(rows)


# ---------- gel 8 : B1 unités de la bande CTC, B2 lignes coupées, B3 EVAL non rejouable, M1 population du CER ----------

def _fake_last(neg, pos):
    import torch
    xs = torch.linspace(0.1, 0.9, 8)
    pts = torch.stack([xs, torch.full((8,), 0.5)], dim=-1)[None, None]            # [1, 1, 8, 2], droite horizontale
    return {"pts": pts, "neg": torch.full((1, 1), float(neg)), "pos": torch.full((1, 1), float(pos))}


def test_gel8_B1_the_ctc_band_gets_neg_pos_and_sigma_in_canvas_pixels(TD, monkeypatch):
    import torch
    seen = []
    monkeypatch.setattr(TD, "expected_ctc_loss", lambda feats, line, reader, labels, sigma_px: (
        seen.append((float(line["neg"]), float(line["pos"]), float(sigma_px))) or (torch.zeros(()), "ok")))
    cfg = TD.load_all()
    task = TD.Task("main", cfg, [], ["<blank>", "a"])
    for scale, want in ((0.5, (4.0, 2.0, 16.0)), (1.0, (8.0, 4.0, 32.0))):          # pQ : 8 / 4 / 32 passés tels quels avant le gel 8
        sample = {"scale": scale, "readable": [True], "texts": ["a"]}
        task.ctc_term(types.SimpleNamespace(reader=None), [sample], [[(0, 0)]], _fake_last(8.0, 4.0), torch.zeros(1, 64, 8, 8), 32.0,
                      np.random.default_rng(0), torch.device("cpu"))
        assert seen[-1] == want


def test_gel8_B1_the_eval_reading_converts_neg_pos_with_the_zone_scale_and_the_witness_keeps_scale_one(TD, monkeypatch):
    import torch
    seen = []
    monkeypatch.setattr(TD, "band_features", lambda f4, base, neg, pos: seen.append((float(neg), float(pos))) or None)

    class FakeModel:
        reader = None

        def __call__(self, canvas, mask):
            last = _fake_last(8.0, 4.0)
            return [dict(last, logit=torch.full((1, 1), 5.0), pts=last["pts"][0:1, 0:1].reshape(1, 1, 8, 2))], torch.zeros(1, 64, 8, 8)

    kw = dict(canvas=np.zeros((4, 4), np.float32), mask=np.ones((4, 4), np.float32), thr=0.6, alphabet=["<blank>"],
              wh=(1536, 512), device=torch.device("cpu"))
    out = TD.raw_outputs_read(FakeModel(), scale=0.5, **kw)
    assert seen[-1] == (4.0, 2.0) and out["hyp_none"] == {0: "band_outside_support"}
    TD.raw_outputs_read(FakeModel(), **kw)                                          # témoin : échelle 1
    assert seen[-1] == (8.0, 4.0)
    tile = TD.tile_sample({"image": np.ones((900, 900), np.float32), "lines": [{
        "id": "l", "baseline": [[100.0, 400.0], [500.0, 400.0]], "normal_negative_px": 12.0, "normal_positive_px": 4.0,
        "text": "a", "legibility": "readable"}]}, np.random.default_rng(0), 768, False, 512)
    assert tile is None or tile["scale"] == 1.0


def _zone_page():
    img = np.ones((400, 800), np.float32)
    line = lambda lid, x0: {"id": lid, "baseline": np.array([[x0, 100.0], [x0 + 400.0, 100.0]]), "neg": 24.0, "pos": 8.0,   # noqa: E731
                            "text": "ab", "legibility": "readable"}
    return {"image": img}, {"lines": [line("in", 100.0), line("cut", -24.0)]}


def test_gel8_B2_a_line_cut_by_the_crop_keeps_its_geometry_but_is_not_ctc_eligible(TD):
    """Marge : 8 px, celle de lines_in_tile (train_stage_s, défaut margin=8.0)."""
    page, block = _zone_page()
    s = TD.zone_sample_ctc(page, block, (50.0, 60.0, 550.0, 130.0))
    assert TD.CTC_MARGIN_PX == 8.0 and len(s["pts"]) == 2 and s["readable"] == [True, False]       # x = -24 : coupée, hors CTC
    assert s["scale"] == 1.0
    page2, block2 = _zone_page()
    block2["lines"] = block2["lines"][1:]                                                          # repro pQ : seule la ligne x = -24
    s2 = TD.zone_sample_ctc(page2, block2, (50.0, 60.0, 550.0, 130.0))
    import torch
    task = TD.Task("main", TD.load_all(), [], ["<blank>", "a", "b"])
    _l, info = task.ctc_term(types.SimpleNamespace(reader=None), [s2], [[(0, 0)]], _fake_last(8.0, 4.0), torch.zeros(1, 64, 8, 8), 32.0,
                             np.random.default_rng(0), torch.device("cpu"))
    assert info["ctc_included"] == 0 and info["ctc_excluded"] == 0                                # jamais candidate
    assert TD.ctc_inside(np.array([[8.0, 40.0], [100.0, 40.0]]), 24.0, 8.0, 200, 100)
    assert not TD.ctc_inside(np.array([[7.9, 40.0], [100.0, 40.0]]), 24.0, 8.0, 200, 100)         # bord gauche
    assert not TD.ctc_inside(np.array([[10.0, 31.0], [100.0, 31.0]]), 24.0, 8.0, 200, 100)        # bande au-dessus
    assert not TD.ctc_inside(np.array([[10.0, 40.0], [193.0, 40.0]]), 24.0, 8.0, 200, 100)        # bord droit


def test_gel8_B3_an_interrupted_eval_cannot_be_started_again_and_the_marker_is_never_deleted(TD, tmp_path, monkeypatch):
    run = tmp_path / "out" / "ld2"
    run.mkdir(parents=True)
    (run / "ld2_manifest.json").write_text(json.dumps({"freeze": {"written_before_EVAL": True, "checkpoint_sha256": "ab" * 32}}))
    ref = _ref_file(tmp_path, ["mf_0000", "mf_0001"])
    args = types.SimpleNamespace(smoke=True, eval_pages=["mf_0000", "mf_0001"], reference_report=str(ref))
    monkeypatch.setattr(TD, "eval_context", lambda *a, **k: {"witness_status": {}, "witness": None, "load_seconds": 0.0})
    calls = []

    def fake_page(pid, root, ctx):
        calls.append(pid)
        if len(calls) == 2:
            raise RuntimeError("interruption")
        return {"page": pid}

    monkeypatch.setattr(TD, "evaluate_page", fake_page)
    with pytest.raises(RuntimeError, match="interruption"):
        TD.phase_EVAL(args, CFG, None, run, tmp_path)
    marker = run / "ld2_eval_started.json"
    started = json.loads(marker.read_text())
    assert calls == ["mf_0000", "mf_0001"] and started["checkpoint_sha256"] == "ab" * 32 and "started_utc" in started
    with pytest.raises(SystemExit, match="décision explicite de pA"):
        TD.phase_EVAL(args, CFG, None, run, tmp_path)
    assert calls == ["mf_0000", "mf_0001"] and marker.is_file()                         # aucune page relue, marqueur conservé
    monkeypatch.setattr(TD, "eval_context", lambda *a, **k: pytest.fail("models must not be loaded again"))
    with pytest.raises(SystemExit, match="pA"):
        TD.phase_EVAL(args, CFG, None, run, tmp_path)


def test_gel8_M1_the_cer_reports_the_none_causes_the_matched_population_and_the_coverage(TD):
    gl = lambda i, x: {"id": f"g{i}", "baseline": np.array([[x, 10.0], [x + 300.0, 10.0]]), "neg": 12.0, "pos": 4.0,   # noqa: E731
                       "text": "ab", "legibility": "readable"}
    page = {"gt_lines": [gl(0, 0.0), gl(1, 400.0), gl(2, 800.0), gl(3, 1200.0)]}
    kept = [{"pts": np.array(g["baseline"]), "zone": 0, "query": i, "score": 0.9, "h": 16.0} for i, g in enumerate(page["gt_lines"])]
    raws = [(0, None, {"hyp": {0: "ab", 1: None, 2: None, 3: None},
                       "hyp_none": {1: "degenerate_chord", 2: "band_outside_support", 3: "reader_exception"}})]
    res = TD.score_lines(page, kept, raws)["cer"]
    assert res["matched_readable_lines"] == 4 and res["matched_readable_chars"] == 8 and res["lines"] == 1 and res["chars"] == 2
    assert res["skipped_by_cause"] == {"degenerate_chord": 1, "band_outside_support": 1, "reader_exception": 1, "unknown": 0}
    row = {"page": "p", "seconds": {"Z-GT": 1.0}, "Z-GT": dict(TD.score_lines(page, kept, raws))}
    agg = TD.aggregate([row], "Z-GT", 1)["cer"]
    assert agg["coverage_lines"] == 0.25 and agg["coverage_chars"] == 0.25 and agg["skipped_total"] == 3
    assert agg["skipped_by_cause"]["reader_exception"] == 1 and agg["scope"] == C2.CER_SCOPE


def test_gel9_ctc_cut_by_crop_counts_the_readable_lines_removed_by_ctc_inside_without_changing_the_sampling(TD):
    import torch
    page, block = _zone_page()
    s = TD.zone_sample_ctc(page, block, (50.0, 60.0, 550.0, 130.0))
    assert s["ctc_cut"] == 1 and s["readable"] == [True, False]
    page2, block2 = _zone_page()
    block2["lines"] = block2["lines"][1:]                                              # repro pQ : seule la ligne x = -24
    s2 = TD.zone_sample_ctc(page2, block2, (50.0, 60.0, 550.0, 130.0))
    assert s2["ctc_cut"] == 1 and s2["readable"] == [False] and len(s2["pts"]) == 1   # la géométrie reste supervisée
    block2["lines"][0]["legibility"] = "illegible"                                      # illisible : pas une ligne CTC retirée
    assert TD.zone_sample_ctc(page2, block2, (50.0, 60.0, 550.0, 130.0))["ctc_cut"] == 0

    class FakeModel:
        reader = None

        def __call__(self, canvas, mask):
            out = {"logit": torch.zeros(1, 32), "pts": torch.rand(1, 32, 8, 2), "neg": torch.ones(1, 32), "pos": torch.ones(1, 32)}
            return [dict(out), dict(out)], torch.zeros(1, 64, 8, 8)

    task = TD.Task("main", TD.load_all(), [], ["<blank>", "a", "b"])
    _loss, info = task.loss(FakeModel(), [s2], 32.0, np.random.default_rng(0), torch.device("cpu"))
    assert info["ctc_cut_by_crop"] == 1 and info["ctc_included"] == 0
    assert '"ctc_cut_by_crop": info["ctc_cut_by_crop"]' in (ROOT / "scripts" / "train_ld2.py").read_text()      # écrit au journal


# ---------- gel 10 : B2 pour les lignes inclinées, garde de corde, limite du témoin ----------

def test_gel10_a_pQ_inclined_repro_is_cut_and_counted(TD):
    import torch
    img = np.ones((400, 1100), np.float32)
    block = {"lines": [{"id": "i", "baseline": np.array([[45.0, 100.0], [55.0, 200.0]]), "neg": 4.0, "pos": 32.0,
                        "text": "ab", "legibility": "readable"}]}
    s = TD.zone_sample_ctc({"image": img}, block, (49.9, 50.0, 1000.0, 250.0))
    assert s["readable"] == [False] and s["ctc_cut"] == 1 and len(s["pts"]) == 1          # géométrie supervisée, CTC non
    task = TD.Task("main", TD.load_all(), [], ["<blank>", "a", "b"])
    _l, info = task.ctc_term(types.SimpleNamespace(reader=None), [s], [[(0, 0)]], _fake_last(8.0, 4.0), torch.zeros(1, 64, 8, 8), 32.0,
                             np.random.default_rng(0), torch.device("cpu"))
    assert info["ctc_included"] == 0


def test_gel10_b_ctc_inside_equals_band_features_support_on_random_inclined_lines(TD):
    """Convention : coordonnées de BORDS du canevas, centres = bords - 0,5, origine (1,5 ; 1,5), pas 4 (stage_s_reader.ORIGIN/STRIDE),
    carte de features du canevas entier 1536 x 512 (384 x 128). band_features accepte la bande ssi tous les points échantillonnés
    vérifient 1,5 <= x - 0,5 <= 4w - 2,5 (idem y) soit 2 <= x_bord <= 1536 - 2 : margin = 2, nw = 1536, nh = 512. Les cas à moins de
    1e-2 px de cette frontière sont écartés (float32 contre float64) : aucune autre différence."""
    import torch
    from babaorum.experimental.stage_s_reader import band_features
    feats = torch.zeros(1, 2, 128, 384)
    rng = np.random.default_rng(7)
    n_true = n_false = n_skipped = 0
    for _ in range(600):
        x0, y0 = rng.uniform(0, 1536), rng.uniform(0, 512)
        ang = np.deg2rad(rng.uniform(-80, 80))
        length = rng.uniform(40, 700)
        t = np.linspace(0, 1, 8)
        pts = np.stack([x0 + length * np.cos(ang) * t, y0 + length * np.sin(ang) * t], axis=-1)
        pts += rng.normal(0, 1.5, pts.shape)
        neg, pos = rng.uniform(0, 40), rng.uniform(0, 40)
        if neg + pos <= 0:
            continue
        ext = TD.band_extreme_points(pts, neg, pos)
        if ext is None:
            assert not TD.ctc_inside(pts, neg, pos, 1536, 512, 2.0)
            continue
        xy = np.concatenate([pts, *ext])
        slack = min(xy[:, 0].min() - 2.0, 1534.0 - xy[:, 0].max(), xy[:, 1].min() - 2.0, 510.0 - xy[:, 1].max())
        if abs(slack) < 1e-2:
            n_skipped += 1
            continue
        mine = TD.ctc_inside(pts, neg, pos, 1536, 512, 2.0)
        try:
            theirs = band_features(feats, torch.tensor(pts, dtype=torch.float32), neg, pos) is not None
        except ValueError:                                                              # corde non gauche-droite
            theirs = False
        assert mine == theirs, (pts.tolist(), neg, pos)
        n_true += mine
        n_false += not mine
    assert n_true > 50 and n_false > 50 and n_skipped < 30


def test_gel10_b_a_chord_not_left_to_right_or_a_zero_tangent_is_ineligible(TD):
    rl = np.array([[300.0, 40.0], [200.0, 40.0]])
    assert TD.band_extreme_points(rl, 4.0, 4.0) is None and not TD.ctc_inside(rl, 4.0, 4.0, 1000, 100)
    vertical = np.array([[100.0, 20.0], [100.0, 60.0]])                               # x_dernier == x_premier : non strict
    assert not TD.ctc_inside(vertical, 4.0, 4.0, 1000, 100)
    same = np.array([[100.0, 20.0]] * 4 + [[130.0, 20.0]] * 4)                         # tangente nulle sur un pas
    assert TD.band_extreme_points(same, 1.0, 1.0) is None                              # pas 0-1 identiques : différence nulle


def test_gel10_c_horizontal_lines_behave_as_in_gel9(TD):
    h = np.array([[10.0, 40.0], [100.0, 40.0]])
    assert TD.ctc_inside(h, 24.0, 8.0, 200, 100) and TD.ctc_inside(np.array([[8.0, 40.0], [192.0, 40.0]]), 24.0, 8.0, 200, 100)
    assert not TD.ctc_inside(np.array([[7.9, 40.0], [100.0, 40.0]]), 24.0, 8.0, 200, 100)
    assert not TD.ctc_inside(np.array([[10.0, 31.0], [100.0, 31.0]]), 24.0, 8.0, 200, 100)
    assert not TD.ctc_inside(np.array([[10.0, 40.0], [193.0, 40.0]]), 24.0, 8.0, 200, 100)
    lo, hi = TD.band_extreme_points(h, 24.0, 8.0)
    assert np.allclose(lo[:, 1], 16.0) and np.allclose(hi[:, 1], 48.0)                  # verticale : y - neg, y + pos


def test_gel10_3_a_predicted_chord_not_left_to_right_is_excluded_and_other_errors_propagate(TD, monkeypatch):
    import torch
    task = TD.Task("main", TD.load_all(), [], ["<blank>", "a"])
    sample = {"scale": 1.0, "readable": [True], "texts": ["a"]}
    last = _fake_last(8.0, 4.0)
    last["pts"] = last["pts"].flip(2)                                                     # x décroissant : corde droite -> gauche
    args = ([sample], [[(0, 0)]], last, torch.zeros(1, 64, 8, 8), 32.0, np.random.default_rng(0), torch.device("cpu"))
    _l, info = task.ctc_term(types.SimpleNamespace(reader=None), *args)
    assert info["ctc_included"] == 0 and info["ctc_excluded"] == 1 and info["ctc_status"] == {"excluded_chord_not_ltr": 1}
    for exc in (ValueError("autre erreur"), RuntimeError("boum")):
        monkeypatch.setattr(TD, "expected_ctc_loss", lambda *a, _e=exc, **k: (_ for _ in ()).throw(_e))
        with pytest.raises(type(exc), match=str(exc)):
            task.ctc_term(types.SimpleNamespace(reader=None), *args)
    assert '"ctc_chord_not_ltr"' in (ROOT / "scripts" / "train_ld2.py").read_text()          # écrit au journal


def test_gel10_2_the_witness_tile_limit_is_in_the_report_limits(TD):
    assert C2.TILE_SELECTION_LIMIT == ("témoin : sélection des lignes de tuile héritée d'E2E-5 (bande traitée comme verticale), "
                                       "non corrigée pour garder la définition de référence")
    assert "C2.TILE_SELECTION_LIMIT" in (ROOT / "scripts" / "train_ld2.py").read_text()
