"""Carnets Kaggle LD-3 (docs/END_TO_END_LD3_PROPOSITION.txt §2-§8 ; modèle : build_kaggle_ld2). Non lancé.

Carnets :
  kaggle_ld3.ipynb        : run complet, GPU T4 : pip (bibliothèques du générateur, isolées), bundle v5 (unzip, clone, checkout,
                            contrôle des 51 fichiers), D (détecteur PP), controls, CAL (une page d'ENTRAÎNEMENT : estimation),
                            porte d'estimation, EVAL (génération des pages mf_0060..mf_0069 PUIS lecture, une fois).
                            Échéance globale 9 000 s depuis le début de la cellule.
  kaggle_ld3_smoke.ipynb  : fumée CPU : mêmes étapes, tests, CAL sur mf_0000, EVAL --smoke sur UNE page leurre (decoy_0000,
                            autre graine) avec une référence B3 factice : aucun nom mf_006x n'est construit.
Montages : resolve_one / resolve_file de build_kaggle_ld1 (candidats explicites, aucun listage).
  pages d'entraînement (CAL seulement ; recouvrement : mf_0000-0049 par chemins explicites) : run 3 d'E2E-3 ;
  checkpoint E2E-5 : mabaku/babaorum-e2e5-lambda, logs/checkpoint_last.pt ; sortie de LD-2 : mabaku/babaorum-ld2-run, ld2/ ;
  rapport e2e4 (sha des images d'entraînement) : mabaku/babaorum-e2e4-extraction ; bundle : dataset mabaku/babaorum-stage-s-bundle-v5.
"""
import glob
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_ld1 as B  # noqa: E402

ROOT = B.ROOT
CONFIG = ROOT / "config" / "ld3.json"
KIT_SLUG = "mabaku/babaorum-ld3-kit"
RUN_SLUG = "mabaku/babaorum-ld3-run"
SMOKE_SLUG = "mabaku/babaorum-ld3-smoke"
LD2_RUN_SLUG = "mabaku/babaorum-ld2-run"
E2E4_SLUG = "mabaku/babaorum-e2e4-extraction"
BUNDLE_SLUG = "mabaku/babaorum-stage-s-bundle-v5"
KIT_CANDIDATES = ["/kaggle/input/babaorum-ld3-kit", "/kaggle/input/datasets/mabaku/babaorum-ld3-kit"]
LD2_CANDIDATES = ["/kaggle/input/babaorum-ld2-run/ld2", "/kaggle/input/notebooks/mabaku/babaorum-ld2-run/ld2"]
E2E4_CANDIDATES = ["/kaggle/input/babaorum-e2e4-extraction/e2e4_extraction_report.json",
                   "/kaggle/input/notebooks/mabaku/babaorum-e2e4-extraction/e2e4_extraction_report.json"]
BUNDLE_CANDIDATES = ["/kaggle/input/babaorum-stage-s-bundle-v5/stage_s_bundle_v5.zip.bin",
                     "/kaggle/input/datasets/mabaku/babaorum-stage-s-bundle-v5/stage_s_bundle_v5.zip.bin"]
KIT_ROOTS = ["scripts/eval_ld3.py", "scripts/data3_generate.py", "config/ld3.json", "tests/test_ld3.py",
             "scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/ld1b_border_diag.py", "scripts/data2_generate.py",
             "scripts/data2_common.py", "scripts/ld0_zone_coverage.py", "scripts/train_line_detr.py", "scripts/train_stage_s.py",
             "scripts/train_stage_s_v2.py", "src/babaorum/experimental/line_detr.py", "src/babaorum/experimental/stage_s_arms.py",
             "tests/resolve_probe.py"]
KIT_READS = ["config/ld3.json", "config/ld2.json", "config/ld1_line_detr.json", "config/stage_s_smoke_v2.json",
             "config/data2_pages.json", "scripts/data2_common.py", "scripts/data2_generate.py", "scripts/data3_generate.py",
             "scripts/eval_ld3.py", "scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/ld1b_border_diag.py",
             "scripts/ld0_zone_coverage.py", "scripts/train_line_detr.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py"]
SMOKE_CAL_S = 5400                      # fumée CPU seulement (cf. fumée LD-2) ; le T4 garde le plafond de la config
SMOKE_EVAL_S = 2 * SMOKE_CAL_S
resolve_one, resolve_file = B.resolve_one, B.resolve_file


def kit_files() -> list:
    vendor = sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / "scripts" / "ld0_vendor" / "**" / "*.py"),
                                                                       recursive=True))
    return sorted(set(B.import_closure(KIT_ROOTS)) | set(vendor) | set(KIT_READS) | {"scripts/stage_s_notebooks/KAGGLE_INSTALL.txt"})


def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


HEAD = r'''
import time
T_START = time.monotonic()          # pris au tout début de la cellule
import datetime, faulthandler, hashlib, json, os, shlex, shutil, subprocess, sys, zipfile
from pathlib import Path
faulthandler.enable()

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')

print('ld3 start', _utc(), flush=True)
OUT = Path(__OUT__)
KIT_CANDIDATES = __KIT__
PAGES_CANDIDATES = __PAGES__
CKPT_CANDIDATES = __CKPT__
LD2_CANDIDATES = __LD2__
E2E4_CANDIDATES = __E2E4__
BUNDLE_CANDIDATES = __BUNDLE__
E2E5_SHA = __E2E5_SHA__
LD2_SHA = __LD2_SHA__
LD2_FILE = __LD2_FILE__
E2E4_SHA = __E2E4_SHA__
BUNDLE_SHA = __BUNDLE_SHA__
COMMIT = __COMMIT__
REQUIRED_PIP = __PIP__
TIMEOUTS = __TIMEOUTS__
LIMITS = __LIMITS__
RUN_FILES = __RUNFILES__
'''

RESOLVERS = inspect.getsource(B.resolve_one) + "\n" + inspect.getsource(B.resolve_file) + r'''

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

KIT = resolve_one(KIT_CANDIDATES, "config/ld3.json")
PAGES = resolve_one(PAGES_CANDIDATES, "pages/mf_0000.json")
CKPT = resolve_file(CKPT_CANDIDATES)
LD2DIR = resolve_one(LD2_CANDIDATES, "ld2_manifest.json")
E2E4 = resolve_file(E2E4_CANDIDATES)
BUNDLE_ZIP = resolve_file(BUNDLE_CANDIDATES)
assert sha256(CKPT) == E2E5_SHA, "checkpoint E2E-5 : sha différent"
assert sha256(LD2DIR / LD2_FILE) == LD2_SHA, "checkpoint LD-2 : sha différent"
assert sha256(E2E4) == E2E4_SHA, "rapport e2e4 : sha différent"
assert sha256(BUNDLE_ZIP) == BUNDLE_SHA, "bundle v5 : sha différent"
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
import importlib.util as _u
_spec = _u.spec_from_file_location("data2_common", KIT / "scripts" / "data2_common.py")
C = _u.module_from_spec(_spec); _spec.loader.exec_module(C)
import ld2_core as C2
'''


def _argv_builder(device: str) -> str:
    return r'''
BUNDLE_DIR = Path("/tmp/ld3_bundle")
SRC = Path("/tmp/ld3_heritage")
LIBS = Path("/tmp/ld3_genlib")
WORK = Path("/tmp/ld3_work")

def _argv(phase, extra=()):
    return [sys.executable, str(KIT / "scripts" / "eval_ld3.py"), "--phase", phase, "--root", str(PAGES),
            "--out", str(RUN), "--ld2-dir", str(LD2DIR), "--e2e5-checkpoint", str(CKPT), "--e2e4-report", str(E2E4),
            "--gen-src", str(SRC), "--work", str(WORK), "--extra-libs", str(LIBS), "--cal-out", str(CAL_OUT),
            "--device", "__DEVICE__", "--cache-dir", "/tmp/ld3_hf", *extra]

def d_mille_check():
    """Les 51 fichiers suivis du générateur cloné égalent bundle_manifest.json (comme DATA-2)."""
    manifest = json.loads((BUNDLE_DIR / "bundle_manifest.json").read_text())
    for rel, want in manifest["mille_feuilles_sha256"].items():
        assert sha256(SRC / rel[len("heritage-synth/"):]) == want, ("fichier du générateur différent", rel)
    print("générateur : fichiers contrôlés", len(manifest["mille_feuilles_sha256"]), flush=True)

GEN_CMDS = {
    "pip": [sys.executable, "-m", "pip", "install", "--quiet", "--timeout", "30", "--retries", "1", "--target", str(LIBS),
            *REQUIRED_PIP],
    "unzip": [sys.executable, "-c", "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])",
              str(BUNDLE_ZIP), str(BUNDLE_DIR)],
    "clone": ["git", "clone", "-q", str(BUNDLE_DIR / "heritage-synth.gitbundle"), str(SRC)],
    "checkout": ["git", "-C", str(SRC), "checkout", "-q", COMMIT],
}
GEN_CAPS = {"pip": TIMEOUTS["pip"], "unzip": TIMEOUTS["unzip"], "clone": TIMEOUTS["clone"], "checkout": TIMEOUTS["clone"]}
# Préparation à plafonds FIXES : utilisée telle quelle par la FUMÉE (CPU, aucune échéance globale de cellule à tenir). Le run
# complet reconstruit ces quatre étapes avec step_argv (plafond = restant - marge d'arrêt).
GEN_PREP = [*({"name": n, "argv": GEN_CMDS[n], "timeout_s": GEN_CAPS[n]} for n in ("pip", "unzip", "clone", "checkout")),
            {"name": "mille_check", "fn": d_mille_check}]
'''.replace("__DEVICE__", device)


FULL_BODY = r'''
RUN = OUT / "ld3"
CAL_OUT = OUT / "ld3_cal"
EXITS = OUT / "exits"
class Incomplete(Exception):
    pass
''' + r'''
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

def guard_fn(name, fn):
    """Étape fn : refusée (INCOMPLETE) si l'échéance globale moins la marge d'arrêt est atteinte, sans exécuter fn."""
    def run():
        if remaining() - LIMITS["stop_margin_s"] <= 0:
            raise Incomplete(f"échéance globale atteinte avant l'étape {name}")
        return fn()
    return {"name": name, "fn": run}

def s_estimate():
    """Porte avant EVAL : (chargements + n pages x durée mesurée par CAL sur une page d'ENTRAÎNEMENT) x facteur + génération.
    Au-delà du temps restant moins la marge : INCOMPLETE, avant que les pages soient générées ou lues."""
    cal = json.loads((CAL_OUT / "ld3_cal.json").read_text())
    avail = remaining() - LIMITS["eval_margin_s"]
    est = cal["estimate_s"] * LIMITS["estimate_factor"] + LIMITS["generation_estimate_s"]
    (OUT / "ld3_eval_estimate.json").write_text(json.dumps(
        {"estimate_s": est, "factor": LIMITS["estimate_factor"], "generation_estimate_s": LIMITS["generation_estimate_s"],
         "remaining_s": remaining(), "margin_s": LIMITS["eval_margin_s"], "available_s": avail, "cal": cal}, indent=1))
    print(f"ESTIMATION_EVAL_S {est:.0f} ; disponible {avail:.0f}", flush=True)
    if est > avail:
        raise Incomplete(f"budget insuffisant : estimation {est:.0f} s > {avail:.0f} s restants")

def eval_argv():
    t = allowed(LIMITS["deadline_s"], LIMITS["eval_margin_s"])
    if t <= 0:
        raise Incomplete("échéance globale atteinte avant EVAL")
    EVAL_STEP["timeout_s"] = t
    return _argv("EVAL")

def ld3_cleanup(rec):
    """TIMEOUT, échéance ou échec d'étape : INCOMPLETE à l'étape, sauf si l'étape a écrit FAILED (refus, garde)."""
    if rec["status"] == "TIMEOUT" or str(rec.get("exception") or "").startswith("Incomplete:"):
        rec["status"] = "INCOMPLETE"
    elif rec["status"] == "FAILED" and rec["exit_code"] != "EXCEPTION":
        f = EXITS / f"ld3_run_exit_{rec['step']}.json"
        if not f.is_file() or json.loads(f.read_text()).get("status") != "FAILED":
            rec["status"] = "INCOMPLETE"

def s_pre():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible : run complet refusé")
    from importlib.metadata import version as _v
    (OUT / "ld3_env.json").write_text(json.dumps({
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": _v("numpy"),
        "PIL": _v("pillow"), "transformers": _v("transformers"), "huggingface_hub": _v("huggingface_hub")}, indent=1))
    for d in (BUNDLE_DIR, SRC, LIBS, WORK):
        shutil.rmtree(d, ignore_errors=True)
    print("pre ok", _utc(), flush=True)

OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")       # hors ligne sauf pip et D
# TOUTE étape argv est bornée par l'échéance globale : délai = max(0, min(plafond, restant - marge d'arrêt)), calculé au
# lancement ; non positif -> Incomplete SANS lancer le sous-processus. La marge d'arrêt (stop_margin_s, 120 s) couvre le délai de
# grâce du kill de execute_pipeline (SIGTERM puis SIGKILL après 10 s + 5 s d'attente) et le nettoyage final (rmtree, écriture
# de ld3_run_exit.json). EVAL garde eval_margin_s (300 s).
MARGIN = LIMITS["stop_margin_s"]
GEN_FULL = [step_argv(n, (lambda n=n: GEN_CMDS[n]), GEN_CAPS[n], MARGIN) for n in ("pip", "unzip", "clone", "checkout")]
CONTROLS_STEP = step_argv("controls", lambda: _argv("controls"), TIMEOUTS["controls"], MARGIN, env=OFFLINE)
D_STEP = step_argv("D", lambda: _argv("D"), TIMEOUTS["D"], MARGIN)
CAL_STEP = step_argv("CAL", lambda: _argv("CAL"), TIMEOUTS["CAL"], MARGIN, env=OFFLINE)
EVAL_STEP = {"name": "EVAL", "argv": eval_argv, "timeout_s": LIMITS["deadline_s"], "env": OFFLINE}
STEPS = [guard_fn("pre", s_pre), *GEN_FULL, guard_fn("mille_check", lambda: d_mille_check()), D_STEP, CONTROLS_STEP, CAL_STEP,
         guard_fn("estimate", s_estimate), EVAL_STEP]
EXIT = None
try:
    EXIT = C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=LIMITS["heartbeat_s"], cleanup=ld3_cleanup)
finally:
    _rec = dict(EXIT) if EXIT is not None else {"status": "FAILED_BEFORE_EXIT"}
    _rec["kit_sha256"] = {rel: sha256(KIT / rel) for rel in RUN_FILES if (KIT / rel).is_file()}      # fichiers de LD-3 lus
    (OUT / "ld3_run_exit.json").write_text(json.dumps(_rec, indent=1, default=str))
    for d in (BUNDLE_DIR, SRC, LIBS, WORK):                          # aucune image générée ne reste dans la sortie
        shutil.rmtree(d, ignore_errors=True)
print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}) if EXIT else "none", flush=True)
if (RUN / "ld3_report.json").is_file():
    print("REPORT_SHA256", sha256(RUN / "ld3_report.json"), flush=True)
'''

SMOKE_BODY = r'''
SMOKE = OUT / "ld3_smoke"
RUN = SMOKE / "ld3"
CAL_OUT = SMOKE / "ld3_cal"
SMOKE.mkdir(parents=True, exist_ok=True)
# FUMÉE : plafonds FIXES de la config (CPU, aucune échéance globale de cellule) ; seul le run complet borne chaque étape par le restant.
HEAVY_ENV = dict(os.environ, PYTHONPATH=str(KIT / "src") + os.pathsep + str(KIT / "scripts"))
HEAVY_ENV["HF_HUB_OFFLINE"] = "1"
OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")
''' + r'''
def _smoke_extra(phase):
    return {"CAL": ["--pages", "mf_0000", "--smoke", "--eval-pages", "decoy_0000"],
            "EVAL": ["--smoke", "--decoy-reference"]}.get(phase, [])

STEPS = [
    *GEN_PREP,
    {"name": "D", "argv": _argv("D"), "timeout_s": TIMEOUTS["D"]},
    # tests/test_ld3.py complet, sans -k ; rc écrit dans un fichier, sans bloquer la suite
    {"name": "pytest", "argv": ["sh", "-c", "cd \"" + str(KIT) + "\" && " + sys.executable + " -m pytest -q -rs tests/test_ld3.py > \""
                                + str(SMOKE) + "/ld3_pytest.txt\" 2>&1; echo $? > \"" + str(SMOKE) + "/ld3_pytest_rc.txt\"; exit 0"],
     "env": HEAVY_ENV, "timeout_s": TIMEOUTS["CAL"]},
    {"name": "controls", "argv": _argv("controls"), "timeout_s": TIMEOUTS["controls"], "env": OFFLINE},
    {"name": "CAL", "argv": _argv("CAL", _smoke_extra("CAL")), "timeout_s": __CALS__, "env": OFFLINE},
    {"name": "EVAL", "argv": _argv("EVAL", _smoke_extra("EVAL")), "timeout_s": __EVALS__, "env": OFFLINE},
]
SUMMARY = {"kind": "SMOKE_KAGGLE_CPU_PAGE_LEURRE", "e2e5_sha256": E2E5_SHA, "ld2_sha256": LD2_SHA}
EXIT = None
try:
    EXIT = C.execute_pipeline(SMOKE, STEPS, heartbeat_interval_s=LIMITS["heartbeat_s"])
finally:
    SUMMARY["pipeline"] = {k: EXIT[k] for k in ("status", "step", "exit_code", "durations_s")} if EXIT else None
    for name in ("ld3_pytest.txt", "ld3_pytest_rc.txt"):
        f = SMOKE / name
        if f.is_file():
            SUMMARY[name] = f.read_text()[-6000:]
    for name in ("ld3_report.json", "ld3_edge_table.json", "data3_generation_manifest.json", "data3_overlap_report.json",
                 "ld3_eval_started.json"):
        f = RUN / name
        if f.is_file():
            SUMMARY[name] = json.loads(f.read_text())
    cal = CAL_OUT / "ld3_cal.json"
    if cal.is_file():
        SUMMARY["ld3_cal.json"] = json.loads(cal.read_text())
    (OUT / "ld3_smoke_summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False, default=str))
    for d in (BUNDLE_DIR, SRC, LIBS, WORK):
        shutil.rmtree(d, ignore_errors=True)
print("SMOKE", SUMMARY.get("pipeline"), flush=True)
'''


def _limits() -> dict:
    r = _cfg()["robustness"]
    return {k: r[k] for k in ("deadline_s", "stop_margin_s", "eval_margin_s", "estimate_factor", "generation_estimate_s", "heartbeat_s")}


def cell(kind: str, out: str = "/kaggle/working", kit=None, pages=None, ckpt=None, ld2=None, e2e4=None, bundle=None,
         e2e5_sha=None, ld2_sha=None, e2e4_sha=None, bundle_sha=None) -> str:
    cfg = _cfg()
    head = (HEAD.replace("__OUT__", repr(out)).replace("__KIT__", repr(kit or KIT_CANDIDATES))
            .replace("__PAGES__", repr(pages or B.PAGES_CANDIDATES)).replace("__CKPT__", repr(ckpt or B.CKPT_CANDIDATES))
            .replace("__LD2__", repr(ld2 or LD2_CANDIDATES)).replace("__E2E4__", repr(e2e4 or E2E4_CANDIDATES))
            .replace("__BUNDLE__", repr(bundle or BUNDLE_CANDIDATES))
            .replace("__E2E5_SHA__", repr(e2e5_sha or cfg["e2e5"]["checkpoint_sha256"]))
            .replace("__LD2_SHA__", repr(ld2_sha or cfg["ld2"]["checkpoint_sha256"]))
            .replace("__LD2_FILE__", repr(cfg["ld2"]["checkpoint_file"]))
            .replace("__E2E4_SHA__", repr(e2e4_sha or cfg["overlap"]["reference_report_sha256"]))
            .replace("__BUNDLE_SHA__", repr(bundle_sha or cfg["generator"]["bundle_sha256"]))
            .replace("__COMMIT__", repr(cfg["generator"]["commit"])).replace("__PIP__", repr(cfg["generator"]["required_pip"]))
            .replace("__RUNFILES__", repr(cfg["run_identity_files"])).replace("__TIMEOUTS__", repr(cfg["robustness"]["timeouts_s"])).replace("__LIMITS__", repr(_limits())))
    if kind == "full":
        return head + RESOLVERS + _argv_builder("cuda") + FULL_BODY
    return head + RESOLVERS + _argv_builder("cpu") + SMOKE_BODY.replace("__CALS__", repr(SMOKE_CAL_S)).replace("__EVALS__", repr(SMOKE_EVAL_S))


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-3 — marge du recadrage 16 px contre 32 px, inférence seule (GPU)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell("full")}]}


def build_smoke_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-3 — FUMÉE Kaggle CPU : une page leurre, référence B3 factice"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell("smoke")}]}


def kernel_metadata(slug: str, title: str, gpu: bool = False) -> dict:
    """Internet réglé par carnet : nécessaire à pip et à D ; hors de ces étapes, HF_HUB_OFFLINE=1."""
    return {"id": slug, "title": title, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [KIT_SLUG, BUNDLE_SLUG], "competition_sources": [],
            "kernel_sources": [LD2_RUN_SLUG, B.CKPT_SLUG, B.PAGES_SLUG, E2E4_SLUG], "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kaggle_ld3.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kaggle_ld3_smoke.ipynb").write_text(json.dumps(build_smoke_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kit_files.json").write_text(json.dumps(kit_files(), indent=1))
    for sub, slug, title, gpu, nb in (("smoke_kernel", SMOKE_SLUG, "babaorum-ld3-smoke", False, build_smoke_notebook()),
                                      ("full_kernel", RUN_SLUG, "babaorum-ld3-run", True, build_notebook())):
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(slug, title, gpu=gpu), indent=2))
        (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld3")
    print(Path(__file__).parent / "ld3")
