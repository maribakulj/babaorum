"""Carnets Kaggle LD-1 (docs/END_TO_END_LD1_LOT.txt §4, §7, §9, §11, §12 ; conception §8). pC, GEL 2 (code ; non lancé).

Carnets :
  kaggle_ld1.ipynb        : run complet, GPU (--device cuda, contrôlé à l'étape controls), étapes D, T, E, V en argv.
  kaggle_ld1_smoke.ipynb  : fumée Kaggle CPU : D, tests lourds (pytest en argv), T/E/V sur mf_0000, vrai checkpoint.
Montages : resolve_one (dossiers) et resolve_file (fichier) : exactement un candidat, aucun listage, aucune recherche récursive.
Pages : run 3 (slug mabaku/babaorum-e2e3-run3-fp-classes), chemin e2e3/mf. Checkpoint : mabaku/babaorum-e2e5-lambda, logs/checkpoint_last.pt.
Plafonds, configuration de robustesse et détecteur : lus dans config/ld1_line_detr.json (§12 (4), (12)).
"""
import ast
import glob
import inspect
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "ld1_line_detr.json"
KIT_ROOTS = ["src/babaorum/experimental/line_detr.py", "scripts/train_line_detr.py", "config/ld1_line_detr.json",
             "tests/test_line_detr.py", "tests/resolve_probe.py", "scripts/data2_common.py", "scripts/ld0_zone_coverage.py",
             "config/data2_pages.json"]
# §13 (1) : fichiers LUS par le code du kit (data2_common lit config/data2_pages.json ; écrit data2_run_exit.json)
KIT_READS = ["config/ld1_line_detr.json", "config/data2_pages.json", "scripts/data2_common.py",
             "scripts/train_line_detr.py", "scripts/ld0_zone_coverage.py"]
KIT_CANDIDATES = ["/kaggle/input/babaorum-ld1-kit", "/kaggle/input/datasets/mabaku/babaorum-ld1-kit"]
PAGES_CANDIDATES = ["/kaggle/input/babaorum-e2e3-run3-fp-classes/e2e3/mf",
                    "/kaggle/input/notebooks/mabaku/babaorum-e2e3-run3-fp-classes/e2e3/mf"]
CKPT_CANDIDATES = ["/kaggle/input/babaorum-e2e5-lambda/logs/checkpoint_last.pt",
                   "/kaggle/input/notebooks/mabaku/babaorum-e2e5-lambda/logs/checkpoint_last.pt"]
PAGES_SLUG = "mabaku/babaorum-e2e3-run3-fp-classes"
CKPT_SLUG = "mabaku/babaorum-e2e5-lambda"
KIT_SLUG = "mabaku/babaorum-ld1-kit"


def resolve_one(candidates, marker):
    """Exactement un candidat dont <candidat>/<marqueur> existe. Aucun listage."""
    hits = [Path(c) for c in candidates if (Path(c) / marker).is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"montage : {len(hits)} candidat(s) pour {marker}")
    return hits[0]


def resolve_file(candidates):
    """Exactement un candidat qui est un fichier (§12 (15))."""
    hits = [Path(c) for c in candidates if Path(c).is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"fichier : {len(hits)} candidat(s)")
    return hits[0]


def import_closure(roots) -> list:
    seen, todo = set(), list(roots)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = ROOT / rel
        if not path.is_file() or path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("babaorum"):
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names if a.name.startswith("babaorum")]
            for mod in mods:
                parts = mod.split(".")
                for k in range(1, len(parts) + 1):
                    for cand in (Path("src", *parts[:k]).with_suffix(".py"), Path("src", *parts[:k], "__init__.py")):
                        if (ROOT / cand).is_file():
                            todo.append(str(cand))
    return sorted(seen)


def kit_files() -> list:
    vendor = sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / "scripts" / "ld0_vendor" / "**" / "*.py"),
                                                                       recursive=True))
    return sorted(set(import_closure(KIT_ROOTS)) | set(vendor) | set(KIT_READS)
                  | {"scripts/stage_s_notebooks/KAGGLE_INSTALL.txt"})


def _cfg():
    return json.loads(CONFIG.read_text())


HEAD = r'''
import time
T_START = time.monotonic()          # §16 (3) : pris au tout début de la cellule
import datetime, faulthandler, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
faulthandler.enable()

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')

print('ld1 start', _utc(), flush=True)
OUT = Path("/kaggle/working")
KIT_CANDIDATES = __KIT__
PAGES_CANDIDATES = __PAGES__
CKPT_CANDIDATES = __CKPT__
INIT_SHA = __INIT_SHA__
TIMEOUTS = __TIMEOUTS__
LIMITS = __LIMITS__
'''

RESOLVERS = inspect.getsource(resolve_one) + "\n" + inspect.getsource(resolve_file) + r'''

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

KIT = resolve_one(KIT_CANDIDATES, "config/ld1_line_detr.json")
PAGES = resolve_one(PAGES_CANDIDATES, "pages/mf_0000.json")
CKPT = resolve_file(CKPT_CANDIDATES)
assert INIT_SHA != "NON FIXÉ", "sha complet du checkpoint non fixé"
assert sha256(CKPT) == INIT_SHA, "checkpoint : sha différent"
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
import importlib.util as _u
_spec = _u.spec_from_file_location("data2_common", KIT / "scripts" / "data2_common.py")
C = _u.module_from_spec(_spec); _spec.loader.exec_module(C)
'''


def _argv_builder(device: str) -> str:
    return r'''
def _argv(phase, extra=()):
    return [sys.executable, str(KIT / "scripts" / "train_line_detr.py"), "--phase", phase,
            "--root", str(PAGES), "--init-checkpoint", str(CKPT), "--device", "__DEVICE__",
            "--cache-dir", "/tmp/ld1_hf", *extra]
'''.replace("__DEVICE__", device)


FULL_BODY = r'''
RUN = OUT / "ld1"
EXITS = OUT / "exits"
class Incomplete(Exception):
    pass

def remaining():
    return LIMITS["deadline_s"] - (time.monotonic() - T_START)

def allowed(cap, reserve=0.0):
    """§16 (3) : max(0, min(plafond, restant - réserve))."""
    return max(0.0, min(cap, remaining() - reserve))

def step_argv(ph, build_argv, cap, reserve=0.0, **kw):
    """Étape argv dont le délai est calculé au lancement ; délai nul : INCOMPLETE sans lancer l'étape."""
    def argv():
        t = allowed(cap, reserve)
        if t <= 0:
            raise Incomplete(f"échéance globale atteinte avant l'étape {ph}")
        st["timeout_s"] = t
        return build_argv()
    st = {"name": ph, "argv": argv, "timeout_s": cap, **kw}
    return st

def s_budget():
    """§15 (4) : S recalculé sur le budget réellement disponible pour E, écrit au manifeste AVANT E."""
    mf = RUN / "ld1_timing_manifest.json"
    man = json.loads(mf.read_text())
    budget_e = allowed(TIMEOUTS["E"], LIMITS["v_reserve_s"] + LIMITS["stop_margin_s"])
    S = min(man["max_steps"], int((budget_e - LIMITS["e_reserve_s"]) // man["s_per_step"]))
    if budget_e <= 0 or S < 1:
        raise Incomplete("échéance globale : budget insuffisant pour l'étape E")
    man.update(steps=S, budget_s=budget_e, e_reserve_s=LIMITS["e_reserve_s"])
    mf.write_text(json.dumps(man, indent=1))

def ld1_cleanup(rec):
    """TIMEOUT, échéance ou échec d'étape : INCOMPLETE à l'étape, sauf si l'étape a écrit FAILED (device, garde)."""
    if rec["status"] == "TIMEOUT" or str(rec.get("exception") or "").startswith("Incomplete:"):
        rec["status"] = "INCOMPLETE"
    elif rec["status"] == "FAILED" and rec["exit_code"] != "EXCEPTION":
        f = EXITS / f"ld1_run_exit_{rec['step']}.json"
        if not f.is_file() or json.loads(f.read_text()).get("status") != "FAILED":
            rec["status"] = "INCOMPLETE"

def s_controls():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible : run complet refusé")
    from importlib.metadata import version as _v
    (OUT / "ld1_env.json").write_text(json.dumps({     # échec (PackageNotFoundError) si transformers ou huggingface_hub absents
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": _v("numpy"),
        "PIL": _v("pillow"), "transformers": _v("transformers"), "huggingface_hub": _v("huggingface_hub")}, indent=1))
    from babaorum.experimental.line_detr import LineDetr, load_enc_det_from_e2e5
    ck = torch.load(CKPT, map_location="cpu")
    load_enc_det_from_e2e5(LineDetr(), ck["model"], expected_step=2000, step=ck["step"])    # init_check
    print("controls ok", _utc(), flush=True)

STEPS = [{"name": "controls", "fn": s_controls}]
OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")       # §13 (3) : hors ligne sauf pendant D
RESERVES = {"D": 0.0, "T": 0.0, "E": LIMITS["v_reserve_s"] + LIMITS["stop_margin_s"], "V": 0.0}
STEPS += [step_argv(ph, (lambda ph=ph: _argv(ph, ["--out", str(RUN)])), TIMEOUTS[ph], RESERVES[ph],
                    **({} if ph == "D" else {"env": OFFLINE})) for ph in ("D", "T", "E", "V")]
STEPS.insert(3, {"name": "s_budget", "fn": s_budget})        # controls, D, T, s_budget, E, V
EXIT = None
try:
    EXIT = C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=60.0, cleanup=ld1_cleanup)
finally:
    (OUT / "ld1_run_exit.json").write_text(json.dumps(EXIT if EXIT is not None else {"status": "FAILED_BEFORE_EXIT"},
                                                      indent=1, default=str))
print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}) if EXIT else "none", flush=True)
'''

SMOKE_BODY = r'''
SMOKE = OUT / "ld1_smoke"
RUN = SMOKE / "ld1"
SMOKE.mkdir(parents=True, exist_ok=True)
HEAVY_ENV = dict(os.environ, LD1_HEAVY="1", LD1_PAGES_ROOT=str(PAGES), PYTHONPATH=str(KIT / "src") + os.pathsep + str(KIT / "scripts"))
HEAVY_ENV["HF_HUB_OFFLINE"] = "1"
def _smoke_argv(phase, extra=()):
    return [sys.executable, str(KIT / "scripts" / "train_line_detr.py"), "--phase", phase, "--root", str(PAGES),
            "--init-checkpoint", str(CKPT), "--device", "cpu", "--out", str(RUN), "--pages", "mf_0000",
            "--cache-dir", str(SMOKE / "hf"), *extra]
STEPS = [
    {"name": "D", "argv": [sys.executable, str(KIT / "scripts" / "train_line_detr.py"), "--phase", "D",
                            "--out", str(RUN), "--cache-dir", str(SMOKE / "hf")], "timeout_s": TIMEOUTS["D"]},
    # §13 (4) : TOUT tests/test_line_detr.py, sans -k ; rc écrit dans un fichier, sans bloquer la suite
    {"name": "pytest_heavy", "argv": ["sh", "-c", "cd \"" + str(KIT) + "\" && " + sys.executable + " -m pytest -q -rs "
                                      "tests/test_line_detr.py > \"" + str(SMOKE) + "/ld1_pytest.txt\" 2>&1; "
                                      "echo $? > \"" + str(SMOKE) + "/ld1_pytest_rc.txt\"; exit 0"],
     "env": HEAVY_ENV, "timeout_s": TIMEOUTS["pytest_heavy"]},
    {"name": "T", "argv": _smoke_argv("T", ["--timing-steps", "3", "--max-steps", "20"]), "timeout_s": TIMEOUTS["T"]},
    {"name": "E", "argv": _smoke_argv("E", ["--max-steps", "20"]), "timeout_s": TIMEOUTS["E"]},
    {"name": "V", "argv": _smoke_argv("V", ["--diagnostic"]), "timeout_s": TIMEOUTS["V"]},
]
SUMMARY = {"kind": "SMOKE_KAGGLE_CPU_VRAI_CHECKPOINT", "init_sha256": INIT_SHA}
EXIT = None
try:
    EXIT = C.execute_pipeline(SMOKE, STEPS, heartbeat_interval_s=60.0)
finally:
    for n in (20,):                              # la fumée n'écrit qu'un checkpoint (pas 20) : effacé par chemin
        (RUN / f"ld1_step{n:06d}.pt").unlink(missing_ok=True)      # la fumée ne rapporte que du JSON
    SUMMARY["pipeline"] = {k: EXIT[k] for k in ("status", "step", "exit_code", "durations_s")} if EXIT else None
    for name in ("ld1_pytest.txt", "ld1_pytest_rc.txt"):
        f = SMOKE / name
        if f.is_file():
            SUMMARY[name] = f.read_text()[-6000:]
    for name in ("ld1_timing_manifest.json", "ld1_report.json", "ld1_train_log.jsonl"):
        f = RUN / name
        if f.is_file():
            SUMMARY[name] = f.read_text()[-20000:] if name.endswith(".jsonl") else json.loads(f.read_text())
    (OUT / "ld1_smoke_summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False, default=str))
    shutil.rmtree(SMOKE / "hf", ignore_errors=True)
print("SMOKE", SUMMARY.get("pipeline"), flush=True)
'''


def _timeouts() -> dict:
    t = _cfg()["robustness"]["timeouts_s"]
    return {"D": t["D"], "T": t["T"], "E": t["E"], "V": t["V"], "pytest_heavy": t["pytest_heavy"]}


def _limits() -> dict:
    r = _cfg()["robustness"]
    return {k: r[k] for k in ("deadline_s", "v_reserve_s", "stop_margin_s", "e_reserve_s")}


def _cell(body: str, device: str) -> str:
    cfg = _cfg()
    head = (HEAD.replace("__KIT__", repr(KIT_CANDIDATES)).replace("__PAGES__", repr(PAGES_CANDIDATES))
            .replace("__CKPT__", repr(CKPT_CANDIDATES)).replace("__INIT_SHA__", repr(cfg["init"]["sha256_full"]))
            .replace("__TIMEOUTS__", repr(_timeouts())).replace("__LIMITS__", repr(_limits())))
    return head + RESOLVERS + _argv_builder(device) + body


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-1 — LINE-DETR : entraînement de contrôle Z-GT sans CTC (GEL 2, GPU)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": _cell(FULL_BODY, "cuda")}]}


def build_smoke_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-1 — FUMÉE Kaggle CPU : D, tests lourds (argv), T/E/V sur mf_0000, vrai checkpoint"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": _cell(SMOKE_BODY, "cpu")}]}


def kernel_metadata(slug: str, title: str, gpu: bool = False) -> dict:
    return {"id": slug, "title": title, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [KIT_SLUG], "competition_sources": [],
            "kernel_sources": [CKPT_SLUG, PAGES_SLUG], "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kaggle_ld1.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kaggle_ld1_smoke.ipynb").write_text(json.dumps(build_smoke_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kit_files.json").write_text(json.dumps(kit_files(), indent=1))
    smoke_dir = out_dir / "smoke_kernel"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata("mabaku/babaorum-ld1-smoke",
                                                                              "babaorum-ld1-smoke"), indent=2))
    (smoke_dir / "notebook.ipynb").write_text(json.dumps(build_smoke_notebook(), indent=1, ensure_ascii=False))
    full_dir = out_dir / "full_kernel"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata("mabaku/babaorum-ld1-run", "babaorum-ld1-run",
                                                                              gpu=True), indent=2))
    (full_dir / "notebook.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld1")
    print(Path(__file__).parent / "ld1")
