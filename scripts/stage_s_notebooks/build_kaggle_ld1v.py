"""Carnet Kaggle LD-1 « V seule » (docs/END_TO_END_LD1_LOT.txt §18). Aucun entraînement, aucune modification du code LD-1.

Réutilise build_kaggle_ld1 (résolveurs de montage, constantes, configuration) sans le modifier.
Étapes : copie (manifeste + ld1_step{S:06d}.pt, S == 1871), D (internet), sonde (V sur mf_0045, chronométrée), V complet.
"""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_ld1 as B  # noqa: E402

SLUG = "mabaku/babaorum-ld1-v"
RUN_SLUG = "mabaku/babaorum-ld1-run"
RUN_CANDIDATES = ["/kaggle/input/babaorum-ld1-run/ld1", "/kaggle/input/notebooks/mabaku/babaorum-ld1-run/ld1"]
EXPECTED_S = 1871
DEADLINE_S = 10800
MARGIN_S = 300      # réservé à kill_group, à la fermeture et à l'écriture des exits

HEAD = r'''
import time
T_START = time.monotonic()          # §16 (3) : pris au tout début de la cellule
import datetime, faulthandler, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
faulthandler.enable()

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')

print('ld1v start', _utc(), flush=True)
OUT = Path(__OUT__)
KIT_CANDIDATES = __KIT__
PAGES_CANDIDATES = __PAGES__
CKPT_CANDIDATES = __CKPT__
RUN_CANDIDATES = __RUNSRC__
INIT_SHA = __INIT_SHA__
EXPECTED_S = __EXPECTED_S__
DEADLINE_S = __DEADLINE__
D_CAP_S = __D_CAP__
MARGIN_S = __MARGIN__
'''

# Pas de listage ni de recherche récursive : candidats explicites ; le fichier de checkpoint est nommé d'après S.
RESOLVERS = inspect.getsource(B.resolve_one) + "\n" + inspect.getsource(B.resolve_file) + r'''

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

KIT = resolve_one(KIT_CANDIDATES, "config/ld1_line_detr.json")
PAGES = resolve_one(PAGES_CANDIDATES, "pages/mf_0000.json")
CKPT = resolve_file(CKPT_CANDIDATES)          # E2E-5 : exigé par build_model (init) même en V
RUN_SRC = resolve_one(RUN_CANDIDATES, "ld1_timing_manifest.json")
assert sha256(CKPT) == INIT_SHA, "checkpoint E2E-5 : sha différent"
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
import importlib.util as _u
_spec = _u.spec_from_file_location("data2_common", KIT / "scripts" / "data2_common.py")
C = _u.module_from_spec(_spec); _spec.loader.exec_module(C)
'''

BODY = r'''
BASE = OUT / "ld1v"
RUN = BASE / "run"                  # run de travail : manifeste + checkpoint, puis rapport de V
PROBE = OUT / "ld1v_probe" / "run"  # run de sonde jeté ensuite
EXITS = BASE / "exits"              # V et D écrivent leurs sorties dans <run>/../exits, hors de RUN
class Incomplete(Exception):
    pass

def remaining():
    return DEADLINE_S - (time.monotonic() - T_START)

def allowed(cap, reserve=0.0):
    return max(0.0, min(cap, remaining() - reserve))

def _argv(phase, extra=()):
    return [sys.executable, str(KIT / "scripts" / "train_line_detr.py"), "--phase", phase,
            "--root", str(PAGES), "--init-checkpoint", str(CKPT), "--device", "cpu",
            "--cache-dir", "/tmp/ld1_hf", *extra]

OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")

def checkpoint_inputs():
    """Manifeste unique, S == EXPECTED_S, fichier ld1_step{S:06d}.pt seul lu (aucun autre checkpoint)."""
    man = RUN_SRC / "ld1_timing_manifest.json"
    S = json.loads(man.read_text())["steps"]
    if S != EXPECTED_S:
        raise RuntimeError(f"manifeste : S = {S}, attendu {EXPECTED_S}")
    ck = RUN_SRC / f"ld1_step{S:06d}.pt"
    if not ck.is_file():
        raise RuntimeError(f"checkpoint {ck.name} absent")
    return man, ck, S

def s_copy():
    man, ck, S = checkpoint_inputs()
    print("CHECKPOINT", ck.name, "sha256", sha256(ck), flush=True)
    for dst in (RUN, PROBE):
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(man, dst / "ld1_timing_manifest.json")
        shutil.copyfile(ck, dst / ck.name)

PROBE_T0 = None

def probe_argv():
    """Sonde : étape ARGV (le minuteur faulthandler est annulé par execute_pipeline avant un sous-processus)."""
    global PROBE_T0
    t = allowed(DEADLINE_S, MARGIN_S)
    if t <= 0:
        raise Incomplete("échéance globale atteinte avant la sonde")
    PROBE_STEP["timeout_s"] = t
    PROBE_T0 = time.monotonic()
    return _argv("V", ["--out", str(PROBE), "--pages", "mf_0045"])

def s_estimate():
    dur = time.monotonic() - PROBE_T0
    est = dur * 5
    print(f"ESTIMATION_V_S {est:.0f}", flush=True)
    (OUT / "ld1v_probe_estimate.json").write_text(json.dumps(
        {"probe_s": dur, "estimate_v_s": est, "remaining_s": remaining(), "margin_s": MARGIN_S}, indent=1))
    rep = PROBE / "ld1_report.json"
    if rep.is_file():
        shutil.copyfile(rep, OUT / "ld1v_probe_report.json")
    shutil.rmtree(PROBE, ignore_errors=True)
    if est > (remaining() - MARGIN_S) * 0.9:
        raise Incomplete(f"budget insuffisant : estimation {est:.0f} s > 90 % du restant {remaining() - MARGIN_S:.0f} s")

def d_argv():
    t = allowed(D_CAP_S)
    if t <= 0:
        raise Incomplete("échéance globale atteinte avant D")
    D_STEP["timeout_s"] = t
    return _argv("D", ["--out", str(RUN)])

def v_argv():
    t = allowed(DEADLINE_S, MARGIN_S)
    if t <= 0:
        raise Incomplete("échéance globale atteinte avant V")
    V_STEP["timeout_s"] = t
    return _argv("V", ["--out", str(RUN)])

def ld1v_cleanup(rec):
    if rec["status"] == "TIMEOUT" or str(rec.get("exception") or "").startswith("Incomplete:"):
        rec["status"] = "INCOMPLETE"

D_STEP = {"name": "D", "argv": d_argv, "timeout_s": D_CAP_S}
PROBE_STEP = {"name": "probe", "argv": probe_argv, "timeout_s": DEADLINE_S, "env": OFFLINE}
V_STEP = {"name": "V", "argv": v_argv, "timeout_s": DEADLINE_S, "env": OFFLINE}

def run():
    steps = [{"name": "copy", "fn": s_copy}, D_STEP, PROBE_STEP,
             {"name": "estimate", "fn": s_estimate}, V_STEP]
    EXIT = None
    try:
        EXIT = C.execute_pipeline(OUT, steps, heartbeat_interval_s=60.0, cleanup=ld1v_cleanup)
    finally:
        for d in (RUN, PROBE):
            for f in d.glob("ld1_step*.pt"):
                f.unlink()
        (OUT / "ld1v_run_exit.json").write_text(json.dumps(EXIT if EXIT is not None else {"status": "FAILED_BEFORE_EXIT"},
                                                           indent=1, default=str))
    print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}) if EXIT else "none", flush=True)
    return EXIT

if globals().get("LD1V_NO_RUN") is None:
    run()
'''


def cell(out: str = "/kaggle/working", kit=None, pages=None, ckpt=None, run_src=None, init_sha=None) -> str:
    cfg = B._cfg()
    head = (HEAD.replace("__OUT__", repr(out)).replace("__KIT__", repr(kit or B.KIT_CANDIDATES))
            .replace("__PAGES__", repr(pages or B.PAGES_CANDIDATES)).replace("__CKPT__", repr(ckpt or B.CKPT_CANDIDATES))
            .replace("__RUNSRC__", repr(run_src or RUN_CANDIDATES))
            .replace("__INIT_SHA__", repr(init_sha or cfg["init"]["sha256_full"]))
            .replace("__EXPECTED_S__", repr(EXPECTED_S)).replace("__DEADLINE__", repr(DEADLINE_S))
            .replace("__MARGIN__", repr(MARGIN_S)).replace("__D_CAP__", repr(cfg["robustness"]["timeouts_s"]["D"])))
    return head + RESOLVERS + BODY


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-1 V seule — sonde mf_0045 puis V complet sur le checkpoint final (CPU, sans entraînement)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell()}]}


def kernel_metadata() -> dict:
    return {"id": SLUG, "title": "babaorum-ld1-v", "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [B.KIT_SLUG], "competition_sources": [],
            "kernel_sources": [RUN_SLUG, B.PAGES_SLUG, B.CKPT_SLUG], "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))
    (out_dir / "notebook.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld1v")
    print(Path(__file__).parent / "ld1v")
