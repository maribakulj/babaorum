"""DATA-2 carnet de LIVRAISON (docs/END_TO_END_DATA2_LOT.txt §18 (2), §19, GEL 10). Local, sans torch, sans Kaggle.

(1) métadonnées : une seule kernel_source (les fragments), aucun dataset, internet coupé ;
(2) espions : la cellule ne lit que son entrée et n'écrit que sa sortie ;
(3) succès : stdout et stderr conformes (16 caractères au plus, sha, nombres, statuts listés), stderr vide ;
(4) §19 (1) : rien n'est livré sans exit_status DONE et statut DONE/COMPLETE ; payloads absents -> rien livré ;
(5) §19 (2) : JSON corrompu, fichier absent, BLOCKED_NOT_FOUND, BLOCKED_COPY : stdout = le statut seul, stderr vide ;
(6) §19 (3) : le texte de scripts/data2_text_guard.py est identique dans build_delivery_cell() et dans le .ipynb ;
(7) une clé texte bloque sans sortie de fragment ; la copie ne contient que les fichiers listés, sha identiques.
"""
import builtins
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import build_kaggle_delivery as D  # noqa: E402
import data2_text_guard as G  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "data1_shard0"
STATUSES = {"OK", "BLOCKED_UPSTREAM", "BLOCKED_TEXT_FIELD", "BLOCKED_NOT_FOUND", "BLOCKED_COPY", "ECHEC"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _conforming(token: str) -> bool:
    return len(token) <= 16 or bool(HEX64.match(token)) or token.isdigit() or token in STATUSES


def _payload(pid, arm):
    return {"page_id": pid, "bras": arm, "échelle": 0.64, "factor": 0.64, "native_wh": [5891, 8610],
            "scaled_wh": [3770, 5510], "stride": 4, "seuil": 0.5, "before": []}


def _make_input(root: Path, *, status="OK", exit_status="DONE", tamper=False, omit_payloads=False,
                corrupt_notebook=False, omit_manifest=False) -> Path:
    src = root / "in"
    (src / "fragments" / "B3").mkdir(parents=True)
    (src / "fragments" / "S2").mkdir(parents=True)
    raw = json.loads((FIX / "data1_manifest.json").read_text())
    clean, _ = G.sanitize_manifest(raw, "a" * 64)
    clean["status"] = status
    if not omit_manifest:
        (src / "data1_manifest.json").write_text(json.dumps(clean))
    template = G.template_from_source((ROOT / "scripts" / "data1_fragments.py").read_text())
    cfg_e2e4 = json.loads((ROOT / "config" / "e2e4_extraction.json").read_text())
    readme = G.expected_readme(clean, template, cfg_e2e4).replace("\x00DATE\x00", "2026-10-24 10:00")
    (src / "LISEZMOI.txt").write_text(readme, encoding="utf-8")
    mention = json.loads((ROOT / "config" / "data2_pages.json").read_text())["overlap_text_accepted_real"]["mention"]
    (src / "LISEZMOI_DATA2.txt").write_text(mention + "\n", encoding="utf-8")
    nb = "{not json" if corrupt_notebook else json.dumps({"exit_status": exit_status})
    (src / "data2_notebook_manifest.json").write_text(nb)
    if not omit_payloads:
        for arm in ("B3", "S2"):
            for pid in clean["pages_fragments"]:
                payload = _payload(pid, arm)
                if tamper and arm == "B3" and pid == clean["pages_fragments"][0]:
                    payload["text"] = "un mot"
                (src / "fragments" / arm / f"{pid}.json").write_text(json.dumps(payload))
    return src


def _run_cell(src: Path, out: Path, monkeypatch):
    cell = D.build_delivery_cell(candidates=[str(src)], out_dir=str(out))
    opened = []
    real_open = io.open

    def spy(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(compile(cell, "delivery_cell", "exec"), {"__name__": "delivery_cell"})
    finally:
        monkeypatch.undo()
    return out_buf.getvalue(), err_buf.getvalue(), opened


def _assert_only_status(stdout, stderr, status):
    assert stderr == "", stderr
    assert stdout.split() == [status], stdout


def test_metadata_one_source_no_dataset_no_internet():
    m = D.delivery_metadata()
    assert m["kernel_sources"] == ["mabaku/babaorum-data2-fragments"]
    assert m["dataset_sources"] == [] and m["competition_sources"] == [] and m["model_sources"] == []
    assert m["enable_internet"] is False
    on_disk = json.loads((ROOT / "scripts" / "stage_s_notebooks" / "delivery" / "kernel-metadata.json").read_text())
    assert on_disk == m


def test_the_cell_reads_only_its_input_and_writes_only_its_output(tmp_path, monkeypatch):
    src = _make_input(tmp_path)
    out = tmp_path / "out"
    _stdout, _stderr, opened = _run_cell(src, out, monkeypatch)
    data = [p for p in opened if p.endswith((".json", ".txt", ".py"))]
    assert data, "the spy saw no file access"
    for p in data:
        assert p.startswith(str(src)) or p.startswith(str(out)), p


def test_success_stdout_and_stderr_are_conforming(tmp_path, monkeypatch):
    stdout, stderr, _ = _run_cell(_make_input(tmp_path), tmp_path / "out", monkeypatch)
    assert stderr == ""
    tokens = stdout.split()
    assert "OK" in tokens
    bad = [t for t in tokens if not _conforming(t)]
    assert bad == [], bad


def test_a_nonconforming_string_is_flagged():
    assert not _conforming("une phrase bien trop longue pour un statut")
    assert _conforming("0" * 64) and _conforming("FICHIERS") and _conforming("BLOCKED_UPSTREAM")


def test_the_fragments_identifier_is_never_printed(tmp_path, monkeypatch):
    stdout, stderr, _ = _run_cell(_make_input(tmp_path), tmp_path / "out", monkeypatch)
    assert "babaorum-data2-fragments" not in stdout + stderr


def test_a_text_key_blocks_and_no_fragment_is_written(tmp_path, monkeypatch):
    src = _make_input(tmp_path, tamper=True)
    out = tmp_path / "out"
    stdout, stderr, _ = _run_cell(src, out, monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_TEXT_FIELD")
    assert not (out / "fragments").exists() and not (out / "data1_manifest.json").exists()


def test_success_copies_only_the_listed_files_with_the_source_sha(tmp_path, monkeypatch):
    src = _make_input(tmp_path)
    out = tmp_path / "out"
    _run_cell(src, out, monkeypatch)
    listed = {"data1_manifest.json", "LISEZMOI.txt", "LISEZMOI_DATA2.txt"} | {
        f"fragments/{arm}/mf_{i:04d}.json" for arm in ("B3", "S2") for i in range(17)}
    copied = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert copied == listed
    assert "data2_notebook_manifest.json" not in copied
    for rel in listed:
        assert hashlib.sha256((src / rel).read_bytes()).hexdigest() == hashlib.sha256((out / rel).read_bytes()).hexdigest()


# ---- §19 (1) : la livraison est conditionnée à l'amont

def test_failed_upstream_with_payloads_present_delivers_nothing(tmp_path, monkeypatch):
    src = _make_input(tmp_path, exit_status="FAILED")
    out = tmp_path / "out"
    stdout, stderr, _ = _run_cell(src, out, monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_UPSTREAM")
    assert not out.exists() or not any(out.iterdir())


def test_a_manifest_status_other_than_ok_delivers_nothing(tmp_path, monkeypatch):
    # §20 (2) : seul status == "OK" passe ; DONE, COMPLETE, SMOKE et FAILED sont bloqués
    for status in ("DONE", "COMPLETE", "SMOKE", "FAILED"):
        src = _make_input(tmp_path / status, status=status)
        out = tmp_path / status / "out"
        stdout, stderr, _ = _run_cell(src, out, monkeypatch)
        _assert_only_status(stdout, stderr, "BLOCKED_UPSTREAM")
        assert not out.exists() or not any(out.iterdir()), status


def test_the_smoke_manifest_is_never_delivered(tmp_path, monkeypatch):
    src = _make_input(tmp_path, status="SMOKE")
    stdout, stderr, _ = _run_cell(src, tmp_path / "out", monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_UPSTREAM")


def test_absent_payloads_deliver_nothing_with_an_explicit_status(tmp_path, monkeypatch):
    src = _make_input(tmp_path, omit_payloads=True)
    out = tmp_path / "out"
    stdout, stderr, _ = _run_cell(src, out, monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_UPSTREAM")
    assert not out.exists() or not any(out.iterdir())


# ---- §19 (2) : chemins d'échec, stdout = statut seul, stderr vide

def test_corrupt_json_gives_blocked_upstream_with_status_only(tmp_path, monkeypatch):
    stdout, stderr, _ = _run_cell(_make_input(tmp_path, corrupt_notebook=True), tmp_path / "out", monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_UPSTREAM")


def test_absent_manifest_gives_blocked_not_found_with_status_only(tmp_path, monkeypatch):
    src = _make_input(tmp_path, omit_manifest=True)
    stdout, stderr, _ = _run_cell(src, tmp_path / "out", monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_NOT_FOUND")


def test_blocked_copy_gives_status_only_when_the_output_cannot_be_written(tmp_path, monkeypatch):
    src = _make_input(tmp_path)
    blocker = tmp_path / "out_is_a_file"
    blocker.write_text("x")
    stdout, stderr, _ = _run_cell(src, blocker, monkeypatch)
    _assert_only_status(stdout, stderr, "BLOCKED_COPY")


# ---- §19 (3) : identité du garde-fou embarqué

def test_the_embedded_guard_text_is_identical_in_the_cell_and_in_the_notebook():
    guard = (ROOT / "scripts" / "data2_text_guard.py").read_text(encoding="utf-8")
    cell = D.build_delivery_cell()
    assert guard in cell
    nb = json.loads((ROOT / "scripts" / "stage_s_notebooks" / "delivery" / "kaggle_data2_delivery.ipynb").read_text())
    assert guard in "".join(nb["cells"][1]["source"])


# ---- §22 (2) : quatre candidats, exactement un contient data1_manifest.json, arbre appât aux deux formats ----

def test_the_delivery_has_four_explicit_candidates_in_both_mount_formats():
    c = D.DELIVERY_CANDIDATES
    assert len(c) == 4 and len(set(c)) == 4
    assert "/kaggle/input/babaorum-data2-fragments" in c and "/kaggle/input/babaorum-data2-fragments/frag_out" in c
    assert "/kaggle/input/notebooks/mabaku/babaorum-data2-fragments" in c
    assert "/kaggle/input/notebooks/mabaku/babaorum-data2-fragments/frag_out" in c


def _rebased(tmp_path):
    return [str(tmp_path / c.lstrip("/")) for c in D.DELIVERY_CANDIDATES]


def test_one_manifest_in_the_notebook_format_is_found_and_decoys_in_the_other_format_are_ignored(tmp_path, monkeypatch):
    real = tmp_path / "kaggle" / "input" / "notebooks" / "mabaku" / "babaorum-data2-fragments" / "frag_out"
    decoy = tmp_path / "kaggle" / "input" / "babaorum-data2-fragments"
    decoy.mkdir(parents=True)
    (decoy / "LISEZMOI.txt").write_text("appât")                 # sans data1_manifest.json : leurre
    _make_input(real.parent, status="OK")                       # le vrai carnet : frag_out sous le format notebooks
    src_real = real.parent / "in"
    (src_real).rename(real)                                    # place l'entrée sous frag_out
    cell = D.build_delivery_cell(candidates=_rebased(tmp_path), out_dir=str(tmp_path / "out"))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        exec(compile(cell, "delivery_cell", "exec"), {"__name__": "delivery_cell"})
    assert err.getvalue() == ""
    assert out.getvalue().split() != ["BLOCKED_NOT_FOUND"]
    assert "OK" in out.getvalue().split()


def test_two_manifests_in_two_formats_refuse_with_blocked_not_found(tmp_path):
    for base in (tmp_path / "kaggle/input/babaorum-data2-fragments", tmp_path / "kaggle/input/notebooks/mabaku/babaorum-data2-fragments"):
        base.mkdir(parents=True)
        (base / "data1_manifest.json").write_text("{}")
    cell = D.build_delivery_cell(candidates=_rebased(tmp_path), out_dir=str(tmp_path / "out"))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        exec(compile(cell, "delivery_cell", "exec"), {"__name__": "delivery_cell"})
    assert out.getvalue().split() == ["BLOCKED_NOT_FOUND"] and err.getvalue() == ""
