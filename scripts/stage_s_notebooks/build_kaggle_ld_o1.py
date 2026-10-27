"""Carnet Kaggle LD-O1 (docs/END_TO_END_LD_O1_LOT.txt §5 ; modèle : build_kaggle_ld3). CPU, une passe, plafond 2 h (7 200 s). Non lancé.

Étapes (toutes en argv sauf la garde `pre`) : D (détecteur PP, internet) ; RUN1 = mf_0045 SEULE (fumée incluse dans la cellule : si elle
échoue, rien d'autre ne tourne) ; RUN4 = mf_0046..0049 ; REPORT (exactement les cinq pages). Sorties : petits JSON seulement dans
/kaggle/working/ld_o1 (ld_o1_report.json, ld_o1_page_<pid>.json). Aucune image.
Montages : pages d'entraînement du run 3 d'E2E-3 (chemins explicites mf_0045-0049) et sortie de LD-2 (checkpoint e5afd143… et bloc freeze).
Chaque étape argv a pour délai max(0, min(plafond, restant - marge d'arrêt)) calculé au lancement ; non positif -> INCOMPLETE sans lancer.
La marge d'arrêt (120 s) couvre le délai de grâce du kill d'execute_pipeline et le nettoyage final.
"""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_ld1 as B  # noqa: E402
import build_kaggle_ld3 as L3  # noqa: E402

ROOT = B.ROOT
CONFIG = ROOT / "config" / "ld_o1.json"
KIT_SLUG = "mabaku/babaorum-ld-o1-kit"
RUN_SLUG = "mabaku/babaorum-ld-o1"
KIT_CANDIDATES = ["/kaggle/input/babaorum-ld-o1-kit", "/kaggle/input/datasets/mabaku/babaorum-ld-o1-kit"]
LD2_CANDIDATES = L3.LD2_CANDIDATES
KIT_ROOTS = ["scripts/diag_ld_o1.py", "config/ld_o1.json", "tests/test_ld_o1.py", "scripts/eval_ld3.py", "scripts/data3_generate.py",
             "scripts/data2_generate.py", "scripts/data2_common.py", "scripts/ld1b_border_diag.py", "scripts/train_ld2.py",
             "scripts/ld2_core.py", "scripts/train_line_detr.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py",
             "scripts/ld0_zone_coverage.py", "src/babaorum/experimental/line_detr.py", "src/babaorum/experimental/stage_s_arms.py",
             "tests/resolve_probe.py"]
KIT_READS = ["config/ld_o1.json", "config/ld3.json", "config/ld2.json", "config/ld1_line_detr.json", "config/stage_s_smoke_v2.json",
             "config/data2_pages.json", "scripts/data2_common.py", "scripts/data2_generate.py", "scripts/data3_generate.py",
             "scripts/eval_ld3.py", "scripts/diag_ld_o1.py", "scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/ld1b_border_diag.py",
             "scripts/ld0_zone_coverage.py", "scripts/train_line_detr.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py"]
resolve_one, resolve_file = B.resolve_one, B.resolve_file


def kit_files() -> list:
    import glob
    vendor = sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / "scripts" / "ld0_vendor" / "**" / "*.py"), recursive=True))
    return sorted(set(B.import_closure(KIT_ROOTS)) | set(vendor) | set(KIT_READS) | {"scripts/stage_s_notebooks/KAGGLE_INSTALL.txt"})


def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


HEAD = r'''
import time
T_START = time.monotonic()          # pris au tout début de la cellule
import datetime, faulthandler, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
faulthandler.enable()

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')

print('ld_o1 start', _utc(), flush=True)
OUT = Path(__OUT__)
KIT_CANDIDATES = __KIT__
PAGES_CANDIDATES = __PAGES__
LD2_CANDIDATES = __LD2__
LD2_SHA = __LD2_SHA__
LD2_FILE = __LD2_FILE__
PAGE_IDS = __PAGE_IDS__
TIMEOUTS = __TIMEOUTS__
LIMITS = __LIMITS__
RUN_FILES = __RUNFILES__
'''

RESOLVERS = inspect.getsource(B.resolve_one) + "\n" + inspect.getsource(B.resolve_file) + r'''

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

KIT = resolve_one(KIT_CANDIDATES, "config/ld_o1.json")
PAGES = resolve_one(PAGES_CANDIDATES, "pages/mf_0000.json")
LD2DIR = resolve_one(LD2_CANDIDATES, "ld2_manifest.json")
assert sha256(LD2DIR / LD2_FILE) == LD2_SHA, "checkpoint LD-2 : sha différent"
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
import importlib.util as _u
_spec = _u.spec_from_file_location("data2_common", KIT / "scripts" / "data2_common.py")
C = _u.module_from_spec(_spec); _spec.loader.exec_module(C)
import ld2_core as C2
'''

BODY = r'''
RUN = OUT / "ld_o1"
EXITS = OUT / "exits"
class Incomplete(Exception):
    pass

def remaining():
    return LIMITS["deadline_s"] - (time.monotonic() - T_START)

def allowed(cap, reserve=0.0):
    """max(0, min(plafond, restant - réserve))."""
    return C2.allowed(cap, remaining(), reserve)

def step_argv(ph, build_argv, cap, reserve=0.0, **kw):
    def argv():
        t = allowed(cap, reserve)
        if t <= 0:
            raise Incomplete(f"échéance globale atteinte avant l'étape {ph}")
        st["timeout_s"] = t
        return build_argv()
    st = {"name": ph, "argv": argv, "timeout_s": cap, **kw}
    return st

def _argv(phase, extra=()):
    return [sys.executable, str(KIT / "scripts" / "diag_ld_o1.py"), "--phase", phase, "--root", str(PAGES), "--out", str(RUN),
            "--ld2-dir", str(LD2DIR), "--cache-dir", "/tmp/ld_o1_hf", *extra]

def guard_fn(name, fn):
    def run():
        if remaining() - LIMITS["stop_margin_s"] <= 0:
            raise Incomplete(f"échéance globale atteinte avant l'étape {name}")
        return fn()
    return {"name": name, "fn": run}

def ld_o1_cleanup(rec):
    """TIMEOUT, échéance ou échec d'étape : INCOMPLETE à l'étape, sauf si l'étape a écrit FAILED (refus, garde)."""
    if rec["status"] == "TIMEOUT" or str(rec.get("exception") or "").startswith("Incomplete:"):
        rec["status"] = "INCOMPLETE"
    elif rec["status"] == "FAILED" and rec["exit_code"] != "EXCEPTION":
        f = EXITS / f"ld_o1_run_exit_{rec['step']}.json"
        if not f.is_file() or json.loads(f.read_text()).get("status") != "FAILED":
            rec["status"] = "INCOMPLETE"

def s_pre():
    import torch
    (OUT / "ld_o1_env.json").write_text(json.dumps({"python": sys.version, "torch": torch.__version__}, indent=1))
    print("pre ok", _utc(), flush=True)

# TOUTE étape argv est bornée par l'échéance globale (7 200 s) moins la marge d'arrêt (120 s : grâce du kill d'execute_pipeline,
# SIGTERM puis SIGKILL après 10 s + 5 s d'attente, et nettoyage final).
M = LIMITS["stop_margin_s"]
OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")       # hors ligne sauf D
D_STEP = step_argv("D", lambda: _argv("D"), TIMEOUTS["D"], M)
RUN1 = step_argv("RUN1", lambda: _argv("RUN", ["--pages", PAGE_IDS[0]]), TIMEOUTS["RUN1"], M, env=OFFLINE)    # fumée : mf_0045 SEULE
RUN4 = step_argv("RUN4", lambda: _argv("RUN", ["--pages", *PAGE_IDS[1:]]), TIMEOUTS["RUN4"], M, env=OFFLINE)
REPORT = step_argv("REPORT", lambda: _argv("REPORT"), TIMEOUTS["REPORT"], M, env=OFFLINE)
STEPS = [guard_fn("pre", s_pre), D_STEP, RUN1, RUN4, REPORT]
EXIT = None
try:
    EXIT = C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=LIMITS["heartbeat_s"], cleanup=ld_o1_cleanup)
finally:
    _rec = dict(EXIT) if EXIT is not None else {"status": "FAILED_BEFORE_EXIT"}
    _rec["kit_sha256"] = {rel: sha256(KIT / rel) for rel in RUN_FILES if (KIT / rel).is_file()}
    (OUT / "ld_o1_run_exit.json").write_text(json.dumps(_rec, indent=1, default=str))
print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}) if EXIT else "none", flush=True)
if (RUN / "ld_o1_report.json").is_file():
    print("REPORT_SHA256", sha256(RUN / "ld_o1_report.json"), flush=True)
'''


def _limits() -> dict:
    r = _cfg()["robustness"]
    return {k: r[k] for k in ("deadline_s", "stop_margin_s", "heartbeat_s")}


def cell(out: str = "/kaggle/working", kit=None, pages=None, ld2=None, ld2_sha=None) -> str:
    cfg = _cfg()
    head = (HEAD.replace("__OUT__", repr(out)).replace("__KIT__", repr(kit or KIT_CANDIDATES))
            .replace("__PAGES__", repr(pages or B.PAGES_CANDIDATES)).replace("__LD2__", repr(ld2 or LD2_CANDIDATES))
            .replace("__LD2_SHA__", repr(ld2_sha or cfg["ld2"]["checkpoint_sha256"]))
            .replace("__LD2_FILE__", repr(cfg["ld2"]["checkpoint_file"])).replace("__PAGE_IDS__", repr(cfg["pages"]))
            .replace("__TIMEOUTS__", repr(cfg["robustness"]["timeouts_s"])).replace("__LIMITS__", repr(_limits()))
            .replace("__RUNFILES__", repr(cfg["run_identity_files"])))
    return head + RESOLVERS + BODY


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-O1 — diagnostic des lignes de bord manquées (CPU, mf_0045-0049, LD-2 gelé)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell()}]}


def kernel_metadata() -> dict:
    return {"id": RUN_SLUG, "title": "babaorum-ld-o1", "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [KIT_SLUG], "competition_sources": [],
            "kernel_sources": [L3.LD2_RUN_SLUG, B.PAGES_SLUG], "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))
    (out_dir / "notebook.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kit_files.json").write_text(json.dumps(kit_files(), indent=1))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld_o1")
    print(Path(__file__).parent / "ld_o1")
