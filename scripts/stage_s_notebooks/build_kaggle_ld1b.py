"""Carnet Kaggle LD-1B (docs/END_TO_END_LD1B_LOT.txt §3) : diagnostic par distance au bord, CPU.
Réutilise les résolveurs (resolve_one, resolve_file) et la fermeture des imports de build_kaggle_ld1 (gel 4) ;
étapes D (téléchargement, internet) puis DIAG (hors ligne), chacune en argv avec délai, ld1b_run_exit.json dans un finally.
Pas de mf_0050-0059 : les pages sont celles de config/ld1b_border.json (mf_0045-0049)."""
import inspect
import json
from pathlib import Path

import build_kaggle_ld1 as L1

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "ld1b_border.json"
KIT_ROOTS = ["scripts/ld1b_border_diag.py", "config/ld1b_border.json", "config/ld1_line_detr.json",
             "scripts/train_line_detr.py", "scripts/data2_common.py", "scripts/ld0_zone_coverage.py",
             "src/babaorum/experimental/line_detr.py", "config/data2_pages.json"]
KIT_CANDIDATES = ["/kaggle/input/babaorum-ld1b-kit", "/kaggle/input/datasets/mabaku/babaorum-ld1b-kit"]
LD1_RUN_CANDIDATES = ["/kaggle/input/babaorum-ld1-run/ld1", "/kaggle/input/notebooks/mabaku/babaorum-ld1-run/ld1"]
LD1V_CANDIDATES = ["/kaggle/input/babaorum-ld1-v/ld1v/run", "/kaggle/input/notebooks/mabaku/babaorum-ld1-v/ld1v/run"]
TIMEOUTS = {"D": 900, "DIAG": 10800}


def kit_files() -> list:
    vendor = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts" / "ld0_vendor").rglob("*.py"))
    return sorted(set(L1.import_closure(KIT_ROOTS)) | set(vendor) | set(KIT_ROOTS)
                  | {"scripts/stage_s_notebooks/KAGGLE_INSTALL.txt"})


BODY = r'''
import hashlib, json, os, subprocess, sys
from pathlib import Path
import datetime, faulthandler
faulthandler.enable()
OUT = Path("/kaggle/working")
KIT_CANDIDATES = __KIT__
LD1_RUN_CANDIDATES = __LD1__
LD1V_CANDIDATES = __LD1V__
PAGES_CANDIDATES = __PAGES__
TIMEOUTS = __TIMEOUTS__

''' + inspect.getsource(L1.resolve_one) + "\n" + inspect.getsource(L1.resolve_file) + r'''

KIT = resolve_one(KIT_CANDIDATES, "config/ld1b_border.json")
LD1_DIR = resolve_one(LD1_RUN_CANDIDATES, "ld1_timing_manifest.json")
LD1V_DIR = resolve_one(LD1V_CANDIDATES, "ld1_report.json")
PAGES = resolve_one(PAGES_CANDIDATES, "pages/mf_0000.json")
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
import importlib.util as _u
_spec = _u.spec_from_file_location("data2_common", KIT / "scripts" / "data2_common.py")
C = _u.module_from_spec(_spec); _spec.loader.exec_module(C)
CACHE = "/tmp/ld1b_hf"
DIAG_OUT = OUT
STEPS = [
    {"name": "D", "argv": [sys.executable, str(KIT / "scripts" / "train_line_detr.py"), "--phase", "D",
                            "--out", str(OUT / "ld1b_download"), "--cache-dir", CACHE], "timeout_s": TIMEOUTS["D"]},
    {"name": "DIAG", "argv": [sys.executable, str(KIT / "scripts" / "ld1b_border_diag.py"),
                              "--ld1-dir", str(LD1_DIR), "--ld1-report", str(LD1V_DIR / "ld1_report.json"), "--root", str(PAGES), "--out", str(DIAG_OUT),
                              "--cache-dir", CACHE], "timeout_s": TIMEOUTS["DIAG"],
     "env": (lambda: dict(os.environ, HF_HUB_OFFLINE="1"))},
]
EXIT = None
try:
    EXIT = C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=60.0)
finally:
    (OUT / "ld1b_run_exit.json").write_text(json.dumps(EXIT if EXIT is not None else {"status": "FAILED_BEFORE_EXIT"},
                                                      indent=1, default=str))
print("LD1B", EXIT["status"] if EXIT else "none", flush=True)
'''


def build_cell() -> str:
    return (BODY.replace("__KIT__", repr(KIT_CANDIDATES)).replace("__LD1__", repr(LD1_RUN_CANDIDATES))
            .replace("__LD1V__", repr(LD1V_CANDIDATES))
            .replace("__PAGES__", repr(L1.PAGES_CANDIDATES)).replace("__TIMEOUTS__", repr(TIMEOUTS)))


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-1B — diagnostic par distance au bord (CPU, mf_0045-0049, checkpoint final LD-1)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": build_cell()}]}


def kernel_metadata() -> dict:
    return {"id": "mabaku/babaorum-ld1b-border", "title": "babaorum-ld1b-border", "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_internet": True, "dataset_sources": ["mabaku/babaorum-ld1b-kit"], "competition_sources": [],
            "kernel_sources": ["mabaku/babaorum-ld1-run", "mabaku/babaorum-ld1-v", "mabaku/babaorum-e2e3-run3-fp-classes"], "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kaggle_ld1b.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))
    (out_dir / "kit_files.json").write_text(json.dumps(kit_files(), indent=1))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld1b")
    print(Path(__file__).parent / "ld1b")
