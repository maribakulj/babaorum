"""Carnet de JUGEMENT AVEUGLE DATA-2 (docs/END_TO_END_DATA2_LOT.txt §23, GEL 13 ; note pR ~/mathematicien/notes/2026-10-21-rg-v2-aveugle.md, l.40-138).

Stdlib seulement. CPU, internet coupé, une seule tentative, délai 3 600 s. Entrées : prédictions de pR (mabaku/mathematicien-rg-v2-mfn26),
VT (mabaku/babaorum-data2-generation : gen_out/mfn, manifeste, index), paquet du juge (mabaku/rg-v2-judge). Chaque chemin est choisi
parmi des candidats EXPLICITES, exactement un, sha contrôlé avant usage. Sortie : rg-v2-mfn-scores.json et un statut ; jamais de VT.
NE PAS LANCER avant la confirmation de pR, relayée par pA.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "scripts" / "stage_s_notebooks" / "rg_judge"

SHA = {
    "predictions": "914443198d48d4444cb745ee76c95a37b2a88e47348dfc6c6c3b1ddce7b090d7",
    "generation_manifest": "db41a16b6d6e1a1329671a383649b9703e33b9b4d8a1a8c0ea23ba26b93aa4a6",
    "gt_index": "c13f6c1185ea61bc57386a1c49520d3b575ab3d57600b14e24898610830541e7",
    "freeze": "7231382f921c7f9ad73be8590173438aa97f77ee0fd17d777ffa4f795aa09986",
    "judge": "2aba4c47950cd62b9e23b645d219368c1c089338209cf18bc3a910e87ef3b04c",
    "checker": "8993a6ce5cef01f464d6ab4011da8c4836fc40e6c1911c6a78c09b08d2c44c2a",
    "line_geometry": "b2a36dd9ec36d3edb083d0c47d9d9e9a05f8963610ecac8c962345fb8284c948",
    "stage_s_data": "d61e9a8da772fa647197257b78659de9ec360c9b6b975bf73546a668f513c6d6",
}
TIMEOUT_S = 3600
CHECK_TIMEOUT_S = 900
JUDGE_OPTIONS = ["--judge-root", "--freeze", "--freeze-sha", "--predictions", "--predictions-sha",
                 "--generation-manifest", "--generation-sha", "--gt-index", "--gt-index-sha", "--gt-root", "--output"]
PREDICTIONS_CANDIDATES = [
    "/kaggle/input/mathematicien-rg-v2-mfn26/rg-v2-mfn-predictions.jsonl",
    "/kaggle/input/notebooks/mabaku/mathematicien-rg-v2-mfn26/rg-v2-mfn-predictions.jsonl",
]
GENERATION_ROOTS = [  # quatre candidats : deux formats, avec et sans gen_out
    "/kaggle/input/babaorum-data2-generation/gen_out",
    "/kaggle/input/babaorum-data2-generation",
    "/kaggle/input/notebooks/mabaku/babaorum-data2-generation/gen_out",
    "/kaggle/input/notebooks/mabaku/babaorum-data2-generation",
]
JUDGE_CANDIDATES = ["/kaggle/input/rg-v2-judge", "/kaggle/input/datasets/mabaku/rg-v2-judge"]


CELL = r'''
import hashlib, json, os, signal, subprocess, sys
from pathlib import Path

OUT = Path("/kaggle/working")
LOGS = Path("/tmp/rg_judge_logs")
LOGS.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
SHA = __SHA__
TIMEOUT_S = __TIMEOUT__
CHECK_TIMEOUT_S = __CHECK_TIMEOUT__
PREDICTIONS_CANDIDATES = __PRED__
GENERATION_ROOTS = __GEN__
JUDGE_CANDIDATES = __JUDGE__

def _emit(*parts):
    print(*parts, flush=True)

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class MontageError(RuntimeError):
    """Montage absent ou ambigu : statut BLOCKED_NOT_FOUND, sans contenu."""


def one_file(candidates):
    hits = [Path(c) for c in candidates if Path(c).is_file()]
    if len(hits) != 1:
        raise MontageError("montage fichier : %d candidat(s)" % len(hits))
    return hits[0]

def one_dir(candidates, marker):
    hits = [Path(c) for c in candidates if (Path(c) / marker).is_file()]
    if len(hits) != 1:
        raise MontageError("montage dossier : %d candidat(s)" % len(hits))
    return hits[0]

def run(name, argv, timeout):
    with open(LOGS / (name + ".out"), "w") as fo, open(LOGS / (name + ".err"), "w") as fe:
        p = subprocess.Popen(argv, stdout=fo, stderr=fe, env=ENV, cwd="/tmp", start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "TIMEOUT"
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return rc

def main():
    JUDGE = one_dir(JUDGE_CANDIDATES, "proto/rg_blind_judge.py")
    PRED = one_file(PREDICTIONS_CANDIDATES)
    GEN = one_dir(GENERATION_ROOTS, "data2_generation_manifest.json")
    FREEZE = JUDGE / "notes" / "2026-10-21-rg-v2-gel.json"
    GEN_MANIFEST = GEN / "data2_generation_manifest.json"
    GT_INDEX = GEN / "rg-v2-mfn-gt-index.json"
    GT_ROOT = GEN / "mfn"
    checks = {"judge": sha256(JUDGE / "proto" / "rg_blind_judge.py") == SHA["judge"],
              "checker": sha256(JUDGE / "proto" / "check_rg_blind.py") == SHA["checker"],
              "freeze": sha256(FREEZE) == SHA["freeze"],
              "line_geometry": sha256(JUDGE / "src" / "babaorum" / "line_geometry.py") == SHA["line_geometry"],
              "stage_s_data": sha256(JUDGE / "src" / "babaorum" / "experimental" / "stage_s_data.py") == SHA["stage_s_data"],
              "predictions": sha256(PRED) == SHA["predictions"],
              "generation_manifest": sha256(GEN_MANIFEST) == SHA["generation_manifest"],
              "gt_index": sha256(GT_INDEX) == SHA["gt_index"]}
    if not all(checks.values()):
        _emit("BLOCKED_SHA")
        _emit("SHA_OK", sum(checks.values()), "SUR", len(checks))
        return
    rc_check = run("check", [sys.executable, str(JUDGE / "proto" / "check_rg_blind.py"), "--judge-root", str(JUDGE),
                             "--scratch-dir", "/tmp/rg-v2-blind-compatibility", "--stdlib-only"], CHECK_TIMEOUT_S)
    if rc_check != 0:
        _emit("BLOCKED_CHECK")
        return
    rc_judge = run("judge", [sys.executable, str(JUDGE / "proto" / "rg_blind_judge.py"),
                             "--judge-root", str(JUDGE), "--freeze", str(FREEZE), "--freeze-sha", SHA["freeze"],
                             "--predictions", str(PRED), "--predictions-sha", SHA["predictions"],
                             "--generation-manifest", str(GEN_MANIFEST), "--generation-sha", SHA["generation_manifest"],
                             "--gt-index", str(GT_INDEX), "--gt-index-sha", SHA["gt_index"],
                             "--gt-root", str(GT_ROOT), "--output", str(OUT / "rg-v2-mfn-scores.json")], TIMEOUT_S)
    scores = OUT / "rg-v2-mfn-scores.json"
    status = json.loads(scores.read_text()).get("status") if scores.is_file() else "absent"
    report = {"judge_exit": rc_judge, "status": status,
              "scores_sha256": sha256(scores) if scores.is_file() else None}
    (OUT / "rg-v2-judge-status.json").write_text(json.dumps(report, indent=2))
    _emit("JUGEMENT", status)
    _emit("SHA_SCORES", report["scores_sha256"])

try:
    main()
except MontageError:
    _emit("BLOCKED_NOT_FOUND")
except Exception:
    _emit("ECHEC")
'''


def judge_argv(judge: str, freeze: str, pred: str, gen_manifest: str, gt_index: str, gt_root: str, output: str):
    """La commande de la note (l.112-122), options dans le même ordre."""
    return ["--judge-root", judge, "--freeze", freeze, "--freeze-sha", SHA["freeze"], "--predictions", pred,
            "--predictions-sha", SHA["predictions"], "--generation-manifest", gen_manifest,
            "--generation-sha", SHA["generation_manifest"], "--gt-index", gt_index, "--gt-index-sha", SHA["gt_index"],
            "--gt-root", gt_root, "--output", output]


def build_cell() -> str:
    return (CELL.replace("__TIMEOUT__", str(TIMEOUT_S)).replace("__CHECK_TIMEOUT__", str(CHECK_TIMEOUT_S))
            .replace("__SHA__", repr(SHA)).replace("__PRED__", repr(PREDICTIONS_CANDIDATES))
            .replace("__GEN__", repr(GENERATION_ROOTS)).replace("__JUDGE__", repr(JUDGE_CANDIDATES)))


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# DATA-2 — JUGEMENT AVEUGLE (pR) : stdlib, CPU, internet coupé, une tentative, 3 600 s"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": build_cell()}]}


def kernel_metadata() -> dict:
    return {"id": "mabaku/babaorum-rg-v2-judge", "title": "babaorum-rg-v2-judge", "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_internet": False, "dataset_sources": ["mabaku/rg-v2-judge"], "competition_sources": [],
            "kernel_sources": ["mabaku/mathematicien-rg-v2-mfn26", "mabaku/babaorum-data2-generation"],
            "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))


if __name__ == "__main__":
    write(DIR / "carnet")
    print(DIR / "carnet")
