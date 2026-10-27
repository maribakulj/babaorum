"""Carnet de jugement aveugle DATA-2 (docs/END_TO_END_DATA2_LOT.txt §23 (2)-(4)). Stdlib, sans torch, lançable en local.
Rien n'est exécuté sur les vraies entrées ici : seules des fixtures vides et la structure du carnet sont testées."""
import ast
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import build_kaggle_rg_judge as J  # noqa: E402

NOTE = Path.home() / "mathematicien" / "notes" / "2026-10-21-rg-v2-aveugle.md"
DATASET = ROOT / "scripts" / "stage_s_notebooks" / "rg_judge" / "dataset"
CARNET = ROOT / "scripts" / "stage_s_notebooks" / "rg_judge" / "carnet"


def _cell_code() -> str:
    nb = json.loads((CARNET / "notebook.ipynb").read_text())
    return "".join(nb["cells"][1]["source"])


# ---- §23 (2) : candidats explicites aux deux formats ----

def test_the_candidates_are_explicit_and_cover_both_formats():
    assert J.PREDICTIONS_CANDIDATES == [
        "/kaggle/input/mathematicien-rg-v2-mfn26/rg-v2-mfn-predictions.jsonl",
        "/kaggle/input/notebooks/mabaku/mathematicien-rg-v2-mfn26/rg-v2-mfn-predictions.jsonl"]
    assert len(J.GENERATION_ROOTS) == 4 and len(set(J.GENERATION_ROOTS)) == 4
    assert J.JUDGE_CANDIDATES == ["/kaggle/input/rg-v2-judge", "/kaggle/input/datasets/mabaku/rg-v2-judge"]
    assert any("notebooks/mabaku" in c for c in J.GENERATION_ROOTS) and any("gen_out" in c for c in J.GENERATION_ROOTS)


def test_the_cell_runs_with_fake_paths_and_blocks_without_a_nameerror(tmp_path, monkeypatch):
    code = _cell_code().replace("[\"/kaggle/input/rg-v2-judge\"", f"[\"{tmp_path}/none\"")
    code = code.replace('OUT = Path("/kaggle/working")', f'OUT = Path("{tmp_path}/out")')
    code = code.replace('LOGS = Path("/tmp/rg_judge_logs")', f'LOGS = Path("{tmp_path}/logs")')
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    ns = {}
    exec(compile(code, "rg_judge_cell", "exec"), ns)
    assert buf.getvalue().strip() == "BLOCKED_NOT_FOUND"          # montage absent : statut, pas de NameError


# ---- §23 (3) : commande identique à la note (mêmes options, même ordre) ----

def test_the_judge_command_has_the_note_options_in_the_same_order():
    if not NOTE.is_file():
        pytest.skip("note de pR absente (Mac seulement)")
    text = NOTE.read_text()
    block = text[text.index("PYTHONDONTWRITEBYTECODE=1 python /kaggle/input/rg-v2-judge/proto/rg_blind_judge.py"):]
    block = block[:block.index("```", 10)]
    options = re.findall(r"(--[a-z-]+)", block)
    assert options == J.JUDGE_OPTIONS
    argv = J.judge_argv("/J", "/F", "/P", "/M", "/I", "/R", "/O")
    assert [a for a in argv if a.startswith("--")] == J.JUDGE_OPTIONS


def test_the_judge_and_checker_pins_match_the_lot_and_the_note():
    assert J.SHA["predictions"] == "914443198d48d4444cb745ee76c95a37b2a88e47348dfc6c6c3b1ddce7b090d7"
    assert J.SHA["generation_manifest"] == "db41a16b6d6e1a1329671a383649b9703e33b9b4d8a1a8c0ea23ba26b93aa4a6"
    assert J.SHA["gt_index"] == "c13f6c1185ea61bc57386a1c49520d3b575ab3d57600b14e24898610830541e7"
    assert J.SHA["freeze"].startswith("7231382f") and J.SHA["judge"].startswith("2aba4c47")
    assert J.SHA["line_geometry"].startswith("b2a36dd9") and J.SHA["stage_s_data"].startswith("d61e9a8d")


# ---- §23 (4) : aucun contenu imprimé, aucun listage, stdlib seulement, une tentative, 3 600 s ----

def test_only_status_and_sha_are_printed():
    tree = ast.parse(_cell_code())
    allowed_names = {"status", "rc_check", "sum", "len", "report"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_emit":
            for arg in node.args:
                ok = (isinstance(arg, ast.Constant)
                      or (isinstance(arg, ast.Name) and arg.id in allowed_names)
                      or (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id in {"sum", "len"})
                      or (isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and arg.value.id == "report"
                          and isinstance(arg.slice, ast.Constant) and arg.slice.value == "scores_sha256"))
                assert ok, ast.dump(arg)[:120]


def test_no_directory_listing_and_stdlib_only_imports():
    code = _cell_code()
    for bad in ("glob", "listdir", "scandir", "iterdir", "os.walk", "rglob"):
        assert bad not in code, bad
    tree = ast.parse(code)
    stdlib = {"hashlib", "json", "os", "signal", "subprocess", "sys", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name in stdlib for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in stdlib, node.module


def test_one_attempt_with_the_3600_second_deadline_and_no_retry():
    assert J.TIMEOUT_S == 3600
    code = _cell_code()
    assert code.count("run(\"judge\"") == 1 and "retry" not in code


def test_the_carnet_is_offline_cpu_and_its_inputs_are_the_three_datasets():
    meta = json.loads((CARNET / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False and meta["enable_gpu"] is False
    assert meta["dataset_sources"] == ["mabaku/rg-v2-judge"]
    assert set(meta["kernel_sources"]) == {"mabaku/mathematicien-rg-v2-mfn26", "mabaku/babaorum-data2-generation"}


# ---- le dossier du dataset : cinq fichiers, sha de l'attestation ----

def test_the_dataset_has_exactly_the_five_files_with_the_attested_shas():
    files = sorted(str(p.relative_to(DATASET)) for p in DATASET.rglob("*") if p.is_file())
    assert files == ["dataset-metadata.json", "notes/2026-10-21-rg-v2-gel.json", "proto/check_rg_blind.py",
                     "proto/rg_blind_judge.py", "src/babaorum/experimental/stage_s_data.py",
                     "src/babaorum/line_geometry.py"]
    expected = {"proto/rg_blind_judge.py": J.SHA["judge"], "proto/check_rg_blind.py": J.SHA["checker"],
                "notes/2026-10-21-rg-v2-gel.json": J.SHA["freeze"],
                "src/babaorum/line_geometry.py": J.SHA["line_geometry"],
                "src/babaorum/experimental/stage_s_data.py": J.SHA["stage_s_data"]}
    for rel, sha in expected.items():
        assert hashlib.sha256((DATASET / rel).read_bytes()).hexdigest() == sha, rel


# ---- §23 (GEL 14) : aucun nom global lu n'est indéfini (même principe que DATA-2 test_no_notebook_cell_reads_an_undefined_name)

def test_no_name_read_in_the_judge_cell_is_undefined():
    import builtins
    tree = ast.parse(_cell_code())
    bound = {"__file__", "__name__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    reads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(n for n in reads if n not in bound and not hasattr(builtins, n))
    assert missing == [], missing


def test_the_timeouts_are_defined_in_the_cell_with_the_lot_values():
    code = _cell_code()
    assert "TIMEOUT_S = 3600" in code and "CHECK_TIMEOUT_S = 900" in code
