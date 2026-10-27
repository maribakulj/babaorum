"""LD-1B, tests légers SANS torch (docs/END_TO_END_LD1B_LOT.txt §1-§3). Fonctions pures et structure du carnet."""
import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import ld1b_border_diag as D  # noqa: E402
import build_kaggle_ld1b as B  # noqa: E402

CFG = json.loads((ROOT / "config" / "ld1b_border.json").read_text())


def test_the_bins_have_the_boundaries_of_the_lot():
    assert D.bin_of(0.0) == "0-8" and D.bin_of(7.9) == "0-8"
    assert D.bin_of(8.0) == "8-16" and D.bin_of(15.9) == "8-16"
    assert D.bin_of(16.0) == "16-32" and D.bin_of(31.9) == "16-32"
    assert D.bin_of(32.0) == ">32" and D.bin_of(40.0) == ">32"
    assert D.bin_of(-3.0) == "0-8"                       # bande qui sort de la zone : compte en 0-8


def test_the_band_distance_is_the_smaller_of_top_and_bottom_distances():
    # baseline horizontale à y = 50, neg = pos = 5 : haut 45, bas 55 ; zone (0,40)-(100,70)
    assert D.band_distance([[0, 50], [100, 50]], 5, 5, (0, 40, 100, 70)) == pytest.approx(5.0)
    assert D.band_distance([[0, 50], [100, 50]], 5, 15, (0, 40, 100, 70)) == pytest.approx(5.0)   # haut 5, bas 5
    assert D.band_distance([[0, 50], [100, 50]], 1, 1, (0, 40, 100, 70)) == pytest.approx(9.0)


def test_page_counts_and_aggregate_are_unweighted_per_page():
    p1 = D.page_counts([("0-8", True), ("0-8", False)], [("0-8", True)])
    p2 = D.page_counts([("0-8", True), (">32", True), (">32", True), (">32", False)], [])
    agg = D.aggregate([p1, p2])
    assert agg["0-8"]["recall_mean"] == pytest.approx((0.5 + 1.0) / 2)      # moyenne par page, pas par ligne
    assert agg[">32"]["recall_mean"] == pytest.approx(2 / 3)
    assert agg["0-8"]["precision_mean"] == pytest.approx(1.0)
    assert agg["16-32"]["recall_mean"] is None and agg["16-32"]["pages_with_gt"] == 0


def test_the_rule_of_use_flags_exactly_at_the_gap_of_0_10():
    rule = CFG["rule_of_use"]
    at = {"0-8": {"recall_mean": 0.70}, ">32": {"recall_mean": 0.80}}
    assert D.rule_of_use(at, rule)["cited_as_track"] is True               # écart exactement 0,10
    above = {"0-8": {"recall_mean": 0.71}, ">32": {"recall_mean": 0.80}}
    assert D.rule_of_use(above, rule)["cited_as_track"] is False
    empty = {"0-8": {"recall_mean": None}, ">32": {"recall_mean": 0.8}}
    assert D.rule_of_use(empty, rule)["cited_as_track"] is None


def test_the_pages_are_exactly_held_out_mf_0045_to_0049_and_no_reserved_page():
    assert CFG["pages"] == [f"mf_{i:04d}" for i in range(45, 50)]
    assert not any(p.startswith("mf_005") and p not in ("mf_0045", "mf_0046", "mf_0047", "mf_0048", "mf_0049")
                   for p in CFG["pages"])


def test_the_pins_are_the_frozen_ld1_files_and_match_the_repo():
    assert D.check_pins(CFG)["line_detr_sha256"] == CFG["pins"]["line_detr_sha256"]
    assert D.check_pins(CFG)["train_line_detr_sha256"] == CFG["pins"]["train_line_detr_sha256"]


def test_the_checkpoint_sha_is_not_guessed_and_stays_unset_until_p6_fixes_it():
    assert CFG["checkpoint"]["sha256"] is None


def _code():
    nb = json.loads((ROOT / "scripts" / "stage_s_notebooks" / "ld1b" / "kaggle_ld1b.ipynb").read_text())
    return "".join(nb["cells"][1]["source"])


def test_the_carnet_is_cpu_with_internet_for_D_only_and_offline_for_the_diagnostic():
    meta = json.loads((ROOT / "scripts" / "stage_s_notebooks" / "ld1b" / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False and meta["enable_internet"] is True
    code = _code()
    assert 'lambda: dict(os.environ, HF_HUB_OFFLINE="1")' in code
    assert '"name": "D"' in code and '"name": "DIAG"' in code


def test_the_carnet_has_no_listing_and_names_no_reserved_page():
    code = _code()
    for bad in ("glob(", "listdir", "scandir", "iterdir", "os.walk", "rglob", "mf_005"):
        assert bad not in code, bad


def test_the_carnet_writes_its_exit_record_in_a_finally_and_resolves_inputs_explicitly():
    tree = ast.parse(_code())
    finals = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert any("ld1b_run_exit.json" in ast.unparse(f) for t in finals for f in t.finalbody)
    code = _code()
    assert "resolve_one(LD1_RUN_CANDIDATES" in code and "resolve_one(PAGES_CANDIDATES" in code


def test_the_kit_carries_the_script_the_config_and_the_frozen_ld1_modules():
    files = set(json.loads((ROOT / "scripts" / "stage_s_notebooks" / "ld1b" / "kit_files.json").read_text()))
    for rel in ("scripts/ld1b_border_diag.py", "config/ld1b_border.json", "src/babaorum/experimental/line_detr.py",
                "scripts/train_line_detr.py", "scripts/data2_common.py", "config/data2_pages.json"):
        assert rel in files, rel
    for rel in files:
        assert (ROOT / rel).exists(), rel


# ---- §4 du lot LD-1B : checkpoint final identifié par la sortie de LD-1 (S, unicité, sha en tête du rapport)

def _ld1_dir(tmp_path, steps=20, files=("ld1_step000020.pt",), report=True, manifest=True):
    d = tmp_path / "ld1"
    d.mkdir(parents=True)
    if manifest:
        (d / "ld1_timing_manifest.json").write_text(json.dumps({"steps": steps}))
    for name in files:
        (d / name).write_bytes(b"checkpoint:" + name.encode())
    if report:   # rapport de LD-1 = sortie de ld1-v (amendement 2), dossier voisin de ld1/
        ck = d / f"ld1_step{steps:06d}.pt" if isinstance(steps, int) else d / "x"
        rep = {"status": "COMPLETE", "checkpoint": ck.name, "checkpoint_sha256": D.sha256(ck) if ck.is_file() else "0"}
        rep.update(report if isinstance(report, dict) else {})
        _rep(d).parent.mkdir(parents=True, exist_ok=True)
        _rep(d).write_text(json.dumps(rep))
    return d


def _rep(d):
    return d.parent / "ld1v" / "ld1_report.json"


def test_the_checkpoint_path_is_the_unique_file_of_step_S_with_its_sha_and_the_two_ld1_shas(tmp_path):
    d = _ld1_dir(tmp_path)
    ident = D.resolve_checkpoint(d, _rep(d))
    assert ident["checkpoint_file"] == "ld1_step000020.pt" and ident["steps_S"] == 20
    assert ident["checkpoint_sha256"] == D.sha256(d / "ld1_step000020.pt")
    assert ident["ld1_timing_manifest_sha256"] == D.sha256(d / "ld1_timing_manifest.json")
    assert ident["ld1_report_sha256"] == D.sha256(_rep(d))


def test_a_missing_checkpoint_file_is_refused(tmp_path):
    d = _ld1_dir(tmp_path, files=("ld1_step000010.pt",))                  # mauvais pas présent, celui de S absent
    with pytest.raises(SystemExit):
        D.resolve_checkpoint(d, _rep(d))


def test_a_duplicate_checkpoint_for_step_S_is_refused(tmp_path):
    d = _ld1_dir(tmp_path, files=("ld1_step000020.pt", "ld1_step000020_copie.pt"))
    with pytest.raises(SystemExit):
        D.resolve_checkpoint(d, _rep(d))


def test_a_wrong_S_is_refused_when_the_file_names_another_step(tmp_path):
    d = _ld1_dir(tmp_path, steps=20, files=("ld1_step000010.pt",))
    with pytest.raises(SystemExit):
        D.resolve_checkpoint(d, _rep(d))


@pytest.mark.parametrize("bad", [0, "vingt", None, True])
def test_an_invalid_S_is_refused(tmp_path, bad):
    d = tmp_path / "ld1"
    d.mkdir()
    (d / "ld1_timing_manifest.json").write_text(json.dumps({"steps": bad}))
    _rep(d).parent.mkdir(); _rep(d).write_text("{}")
    with pytest.raises(SystemExit):
        D.resolve_checkpoint(d, _rep(d))


def test_a_missing_ld1_report_or_manifest_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        d = _ld1_dir(tmp_path / "a", report=False)
        D.resolve_checkpoint(d, _rep(d))
    with pytest.raises(SystemExit):
        d = _ld1_dir(tmp_path / "b", manifest=False)
        D.resolve_checkpoint(d, _rep(d))


def test_a_report_absent_is_refused_with_its_message(tmp_path):
    d = _ld1_dir(tmp_path, report=False)
    with pytest.raises(SystemExit, match="refusé : ld1_report.json absent"):
        D.resolve_checkpoint(d, _rep(d))


def test_a_report_naming_another_checkpoint_is_refused(tmp_path):
    d = _ld1_dir(tmp_path, report={"checkpoint": "ld1_step000010.pt"})
    with pytest.raises(SystemExit, match="refusé"):
        D.resolve_checkpoint(d, _rep(d))


def test_a_report_with_a_different_checkpoint_sha256_is_refused(tmp_path):
    d = _ld1_dir(tmp_path, report={"checkpoint_sha256": "ab" * 32})
    with pytest.raises(SystemExit, match="checkpoint_sha256"):
        D.resolve_checkpoint(d, _rep(d))
    d2 = _ld1_dir(tmp_path / "n")
    r = json.loads(_rep(d2).read_text()); del r["checkpoint_sha256"]; _rep(d2).write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="checkpoint_sha256"):
        D.resolve_checkpoint(d2, _rep(d2))


def test_a_good_report_returns_the_right_shas(tmp_path):
    d = _ld1_dir(tmp_path)
    ident = D.resolve_checkpoint(d, _rep(d))
    assert ident["checkpoint_sha256"] == json.loads(_rep(d).read_text())["checkpoint_sha256"] == D.sha256(d / "ld1_step000020.pt")
    assert ident["ld1_report_sha256"] == D.sha256(_rep(d))


def test_the_notebook_resolves_the_ld1v_report_and_the_kernel_sources_include_ld1_v():
    src = B.build_cell()
    for c in ("/kaggle/input/babaorum-ld1-v/ld1v/run", "/kaggle/input/notebooks/mabaku/babaorum-ld1-v/ld1v/run", "--ld1-report"):
        assert c in src
    assert "mabaku/babaorum-ld1-v" in B.kernel_metadata()["kernel_sources"]
    assert B.TIMEOUTS == {"D": 900, "DIAG": 10800}


def test_the_config_checkpoint_sha_stays_null_the_identity_comes_from_the_output():
    assert CFG["checkpoint"]["sha256"] is None


# ---- §5 du lot LD-1B (relecture) : règle Z-GT seule, arrondi, min/max, affectation, leurre niveau 3

def test_the_rule_is_for_z_gt_only_and_not_applicable_for_z_pp():
    rule = CFG["rule_of_use"]
    summary = {"0-8": {"recall_mean": 0.5}, ">32": {"recall_mean": 0.9}}
    assert D.rule_of_use(summary, rule, "Z-PP-blocs")["status"] == "non applicable"
    assert D.rule_of_use(summary, rule, "Z-GT")["cited_as_track"] is True


def test_the_gap_0_9_minus_0_8_is_rounded_and_reaches_the_threshold():
    rule = CFG["rule_of_use"]
    summary = {"0-8": {"recall_mean": 0.8}, ">32": {"recall_mean": 0.9}}
    r = D.rule_of_use(summary, rule, "Z-GT")
    assert r["gap_high_minus_low"] == 0.1 and r["cited_as_track"] is True      # 0,9 - 0,8 = 0,10000000000000009 sans arrondi


def test_the_band_distance_uses_min_y_minus_neg_and_max_y_plus_pos_for_a_sloped_line():
    # baseline inclinée : y de 40 à 60 ; neg = pos = 5 ; zone (0,40)-(100,70) : haut = 40-5 = 35 (-5 sous le bord), bas = 60+5 = 65 (5 au-dessus)
    assert D.band_distance([[0, 40], [100, 60]], 5, 5, (0, 40, 100, 70)) == pytest.approx(-5.0)
    assert D.bin_of(D.band_distance([[0, 40], [100, 60]], 5, 5, (0, 40, 100, 70))) == "0-8"


def test_the_config_states_the_assignment_and_the_effectifs_caveat():
    text = CFG["assignment"]["Z-PP-blocs"].lower()
    assert "plus grande part" in text and "première" in text
    assert "gt_hors_zone" in CFG["assignment"]["Z-GT"]
    assert "ne s'additionnent PAS" in CFG["effectifs_note"]


def test_a_spy_on_the_page_reads_never_opens_mf_0050_even_at_level_3(tmp_path, monkeypatch):
    import builtins
    import io
    import os
    from PIL import Image
    import build_kaggle_ld1 as L1
    base = tmp_path / "input" / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf"
    (base / "pages").mkdir(parents=True)
    (base / "pages" / "mf_0045.json").write_text("{}")
    (tmp_path / "input" / "x" / "y").mkdir(parents=True)
    decoy = tmp_path / "input" / "x" / "y" / "mf_0050.json"                # leurre au niveau 3
    decoy.write_text("{}")
    opened = []
    real_open, real_text, real_bytes, real_img = io.open, Path.read_text, Path.read_bytes, Image.open

    def spy_open(f, *a, **k):
        opened.append(os.fspath(f)); return real_open(f, *a, **k)

    monkeypatch.setattr(io, "open", spy_open); monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (opened.append(str(self)), real_text(self, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: (opened.append(str(self)), real_bytes(self, *a, **k))[1])
    monkeypatch.setattr(Image, "open", lambda fp, *a, **k: (opened.append(str(fp)), real_img(fp, *a, **k))[1])
    hit = L1.resolve_one([str(base)], "pages/mf_0045.json")
    monkeypatch.undo()
    assert hit == base and not any("mf_0050" in p for p in opened)
    assert CFG["pages"] == [f"mf_{i:04d}" for i in range(45, 50)]           # exactement 45 à 49
