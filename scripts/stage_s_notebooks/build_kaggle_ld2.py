"""Carnets Kaggle LD-2 (docs/END_TO_END_LD2_LOT.txt §1-§7 ; LD-1 gels 4 à 7 pour les leçons de robustesse). Gel 1 ; non lancé.

Carnets :
  kaggle_ld2.ipynb        : run complet, GPU T4 (--device cuda, contrôlé à l'étape controls), étapes D, T, E, EW, FREEZE, CAL, EVAL
                            en argv (sous-processus, groupe, killpg) ; échéance globale 19 800 s depuis le début de la cellule.
  kaggle_ld2_smoke.ipynb  : fumée Kaggle CPU : D, tests (pytest en argv), T/E/EW/FREEZE/CAL sur mf_0000, EVAL sur des pages de
                            REMPLACEMENT (entraînement) avec une référence factice : aucun chemin d'évaluation n'est construit.
Montages : resolve_one (dossiers) et resolve_file (fichier) de build_kaggle_ld1 : exactement un candidat, aucun listage.
Pages : run 3 (mabaku/babaorum-e2e3-run3-fp-classes). Encodeur : mabaku/babaorum-e2e5-lambda, logs/checkpoint_last.pt. Décodeur :
mabaku/babaorum-ld1-run, ld1/ld1_step001871.pt. Référence B3 + λ : mabaku/babaorum-e2e5-lambda, e2e5_lambda_run_report.json.
Plafonds et réserves : lus dans config/ld2.json. L'EVAL n'a PAS de sonde sur une page d'évaluation : le temps d'une page est mesuré
par CAL sur une page d'entraînement, l'estimation est écrite, et si elle dépasse le temps restant le statut est INCOMPLETE avant
d'ouvrir une page d'évaluation.
"""
import glob
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_ld1 as B  # noqa: E402

ROOT = B.ROOT
CONFIG = ROOT / "config" / "ld2.json"
KIT_SLUG = "mabaku/babaorum-ld2-kit"
RUN_SLUG = "mabaku/babaorum-ld2-run"
SMOKE_SLUG = "mabaku/babaorum-ld2-smoke"
LD1_RUN_SLUG = "mabaku/babaorum-ld1-run"
KIT_CANDIDATES = ["/kaggle/input/babaorum-ld2-kit", "/kaggle/input/datasets/mabaku/babaorum-ld2-kit"]
LD1_CANDIDATES = ["/kaggle/input/babaorum-ld1-run/ld1/ld1_step001871.pt",
                  "/kaggle/input/notebooks/mabaku/babaorum-ld1-run/ld1/ld1_step001871.pt"]
REF_CANDIDATES = ["/kaggle/input/babaorum-e2e5-lambda/e2e5_lambda_run_report.json",
                  "/kaggle/input/notebooks/mabaku/babaorum-e2e5-lambda/e2e5_lambda_run_report.json"]
KIT_ROOTS = ["src/babaorum/experimental/line_detr.py", "src/babaorum/experimental/stage_s_arms.py", "scripts/train_line_detr.py",
             "scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py",
             "config/ld2.json", "tests/test_ld2.py", "tests/test_ld2_heavy.py", "tests/resolve_probe.py",
             "scripts/data2_common.py", "scripts/ld0_zone_coverage.py", "config/data2_pages.json"]
# fichiers LUS par le code du kit (data2_common lit config/data2_pages.json ; les configs de LD-1 et d'E2E-5 sont pointées par ld2.json)
KIT_READS = ["config/ld2.json", "config/ld1_line_detr.json", "config/stage_s_smoke_v2.json", "config/data2_pages.json",
             "scripts/data2_common.py", "scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/train_line_detr.py",
             "scripts/ld0_zone_coverage.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py"]
SMOKE_CAL_S = 5400                                 # fumée CPU seulement : une page dépasse 900 s (mesuré) ; la config et le T4 gardent 900 s
SMOKE_EVAL_S = 2 * SMOKE_CAL_S                     # EVAL de la fumée : 2 pages de remplacement
DECOY_PAGES = ["mf_0000", "mf_0001"]              # pages de remplacement de la fumée (pages d'ENTRAÎNEMENT)
resolve_one, resolve_file = B.resolve_one, B.resolve_file


def kit_files() -> list:
    vendor = sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / "scripts" / "ld0_vendor" / "**" / "*.py"),
                                                                       recursive=True))
    return sorted(set(B.import_closure(KIT_ROOTS)) | set(vendor) | set(KIT_READS)
                  | {"scripts/stage_s_notebooks/KAGGLE_INSTALL.txt", "config/ld1_line_detr.json", "config/stage_s_smoke_v2.json"})


def _cfg():
    return json.loads(CONFIG.read_text())


HEAD = r'''
import time
T_START = time.monotonic()          # pris au tout début de la cellule, avant la copie du kit
import datetime, faulthandler, hashlib, json, os, shlex, shutil, subprocess, sys
from pathlib import Path
faulthandler.enable()

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')

print('ld2 start', _utc(), flush=True)
OUT = Path(__OUT__)
KIT_CANDIDATES = __KIT__
PAGES_CANDIDATES = __PAGES__
CKPT_CANDIDATES = __CKPT__
LD1_CANDIDATES = __LD1__
REF_CANDIDATES = __REF__
REF_PREFIX = __REFP__
EST_FACTOR = __ESTF__
INIT_SHA = __INIT_SHA__
INIT_STEP = __INIT_STEP__
DEC_SHA = __DEC_SHA__
DEC_STEP = __DEC_STEP__
TIMEOUTS = __TIMEOUTS__
LIMITS = __LIMITS__
'''

RESOLVERS = inspect.getsource(B.resolve_one) + "\n" + inspect.getsource(B.resolve_file) + r'''

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

KIT = resolve_one(KIT_CANDIDATES, "config/ld2.json")
PAGES = resolve_one(PAGES_CANDIDATES, "pages/mf_0000.json")
CKPT = resolve_file(CKPT_CANDIDATES)
LD1CK = resolve_file(LD1_CANDIDATES)
assert sha256(CKPT) == INIT_SHA, "checkpoint de l'encodeur (E2E-5) : sha différent"
assert sha256(LD1CK) == DEC_SHA, "checkpoint du décodeur (LD-1) : sha différent"
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
import importlib.util as _u
_spec = _u.spec_from_file_location("data2_common", KIT / "scripts" / "data2_common.py")
C = _u.module_from_spec(_spec); _spec.loader.exec_module(C)
import ld2_core as C2
'''


def _argv_builder(device: str) -> str:
    return r'''
def _argv(phase, extra=()):
    return [sys.executable, str(KIT / "scripts" / "train_ld2.py"), "--phase", phase,
            "--root", str(PAGES), "--init-checkpoint", str(CKPT), "--decoder-checkpoint", str(LD1CK),
            "--device", "__DEVICE__", "--cache-dir", "/tmp/ld2_hf", "--out", str(RUN), *extra]

def wrapped(ph, argv, inner_s):
    """Étapes du témoin (TW, EW) : le code de sortie va dans <RUN>/../exits/ld2_<ph>_rc.txt et l'étape rend TOUJOURS 0 au carnet,
    même après un arrêt dur du processus (OOM de l'hôte, segfault). `timeout -s KILL inner_s` tue la commande AVANT le délai de
    l'étape du carnet (inner_s entre le chien de garde et ce délai), pour que sh survive, écrive 137 et rende 0 ; FREEZE, CAL
    et EVAL relisent ce fichier."""
    exits = RUN.parent / "exits"
    exits.mkdir(parents=True, exist_ok=True)
    rc = shlex.quote(str(exits / f"ld2_{ph}_rc.txt"))
    cmd = shlex.join(["timeout", "-s", "KILL", f"{inner_s:.1f}", *[str(a) for a in argv]])
    return ["sh", "-c", cmd + f"; echo $? > {rc}; exit 0"]
'''.replace("__DEVICE__", device)


FULL_BODY = r'''
REF = resolve_file(REF_CANDIDATES)
RUN = OUT / "ld2"
CAL_OUT = OUT / "ld2_cal"
EXITS = OUT / "exits"
TAIL = LIMITS["tail_reserve_s"] + LIMITS["stop_margin_s"]       # gardé pour FREEZE, CAL et EVAL
class Incomplete(Exception):
    pass

def remaining():
    return LIMITS["deadline_s"] - (time.monotonic() - T_START)

def allowed(cap, reserve=0.0):
    """max(0, min(plafond, restant - réserve))."""
    return C2.allowed(cap, remaining(), reserve)

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

def tw_argv():
    """Chronométrage du témoin : budget propre ; sans temps, il n'est PAS lancé (S_t reste 0, témoin NOT_RUN)."""
    t = allowed(TIMEOUTS["TW"], TAIL)
    if t < 60:
        TW_STEP["timeout_s"] = 60
        return [sys.executable, "-c", "pass"]
    TW_STEP["timeout_s"] = t + LIMITS["ew_grace_s"]
    return wrapped("TW", _argv("TW", ["--deadline-s", str(t)]), t + 0.75 * LIMITS["ew_grace_s"])

def ew_argv():
    """Témoin : budget propre ; s'il n'y a plus de temps, il n'est PAS lancé (état NOT_RUN) et le principal continue."""
    t = allowed(TIMEOUTS["EW"], TAIL)
    if t < 60:
        man = json.loads((RUN / "ld2_manifest.json").read_text())
        (RUN / "ld2_witness_status.json").write_text(json.dumps(
            {"status": "NOT_RUN", "reason": "échéance globale", "S2": man["S2"], "S_t": man["S_t"]}, indent=1))
        EW_STEP["timeout_s"] = 60
        return [sys.executable, "-c", "pass"]
    EW_STEP["timeout_s"] = t + LIMITS["ew_grace_s"]
    return wrapped("EW", _argv("EW", ["--deadline-s", str(t)]), t + 0.75 * LIMITS["ew_grace_s"])

def s_estimate():
    """Porte avant EVAL : estimation = chargements + n pages x durée mesurée par CAL sur une page d'ENTRAÎNEMENT. Si elle
    dépasse le temps restant moins la marge : INCOMPLETE, avant d'ouvrir une page d'évaluation."""
    cal = json.loads((CAL_OUT / "ld2_cal.json").read_text())
    avail = remaining() - LIMITS["eval_margin_s"]
    est = cal["estimate_s"] * EST_FACTOR                       # l'estimation de CAL (une page d'entraînement) est majorée
    (OUT / "ld2_eval_estimate.json").write_text(json.dumps(
        {"estimate_s": est, "factor": EST_FACTOR, "remaining_s": remaining(), "margin_s": LIMITS["eval_margin_s"], "available_s": avail,
         "cal": cal}, indent=1))
    print(f"ESTIMATION_EVAL_S {est:.0f} ; disponible {avail:.0f}", flush=True)
    if est > avail:
        raise Incomplete(f"budget insuffisant : estimation {est:.0f} s > {avail:.0f} s restants")

def eval_argv():
    t = allowed(LIMITS["deadline_s"], LIMITS["eval_margin_s"])
    if t <= 0:
        raise Incomplete("échéance globale atteinte avant EVAL")
    EVAL_STEP["timeout_s"] = t
    return _argv("EVAL", ["--reference-report", str(REF)])

def ld2_cleanup(rec):
    """TIMEOUT, échéance ou échec d'étape : INCOMPLETE à l'étape, sauf si l'étape a écrit FAILED (device, refus, garde)."""
    if rec["status"] == "TIMEOUT" or str(rec.get("exception") or "").startswith("Incomplete:"):
        rec["status"] = "INCOMPLETE"
    elif rec["status"] == "FAILED" and rec["exit_code"] != "EXCEPTION":
        f = EXITS / f"ld2_run_exit_{rec['step']}.json"
        if not f.is_file() or json.loads(f.read_text()).get("status") != "FAILED":
            rec["status"] = "INCOMPLETE"

def s_controls():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible : run complet refusé")
    assert sha256(REF).startswith(REF_PREFIX), "rapport de référence B3 + λ : sha différent"     # aucune page lue
    from importlib.metadata import version as _v
    (OUT / "ld2_env.json").write_text(json.dumps({     # échec (PackageNotFoundError) si transformers ou huggingface_hub absents
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": _v("numpy"),
        "PIL": _v("pillow"), "transformers": _v("transformers"), "huggingface_hub": _v("huggingface_hub")}, indent=1))
    from babaorum.experimental.line_detr import LineDetr, load_enc_det_from_e2e5
    ck = torch.load(CKPT, map_location="cpu")
    load_enc_det_from_e2e5(LineDetr(), ck["model"], expected_step=INIT_STEP, step=ck["step"])    # init_check de l'encodeur
    ck1 = torch.load(LD1CK, map_location="cpu")
    want = LineDetr().state_dict()                                                              # init_check du décodeur de LD-1
    assert ck1["step"] == DEC_STEP, "décodeur de LD-1 : pas différent"
    assert set(ck1["model"]) == set(want) and all(tuple(ck1["model"][k].shape) == tuple(v.shape) for k, v in want.items()), \
        "décodeur de LD-1 : clés ou formes différentes"
    print("controls ok", _utc(), flush=True)

OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")       # hors ligne sauf pendant D
D_STEP = step_argv("D", lambda: _argv("D"), TIMEOUTS["D"])
T_STEP = step_argv("T", lambda: _argv("T"), TIMEOUTS["T"], env=OFFLINE)
TW_STEP = {"name": "TW", "argv": tw_argv, "timeout_s": TIMEOUTS["TW"], "env": OFFLINE}
E_STEP = step_argv("E", lambda: _argv("E"), TIMEOUTS["E"], TAIL, env=OFFLINE)
EW_STEP = {"name": "EW", "argv": ew_argv, "timeout_s": TIMEOUTS["EW"], "env": OFFLINE}
FREEZE_STEP = step_argv("FREEZE", lambda: _argv("FREEZE"), TIMEOUTS["FREEZE"], env=OFFLINE)
CAL_STEP = step_argv("CAL", lambda: _argv("CAL", ["--cal-out", str(CAL_OUT)]), TIMEOUTS["CAL"], env=OFFLINE)
EVAL_STEP = {"name": "EVAL", "argv": eval_argv, "timeout_s": LIMITS["deadline_s"], "env": OFFLINE}
STEPS = [{"name": "controls", "fn": s_controls}, D_STEP, T_STEP, TW_STEP, E_STEP, EW_STEP, FREEZE_STEP, CAL_STEP,
         {"name": "estimate", "fn": s_estimate}, EVAL_STEP]
EXIT = None
try:
    EXIT = C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=60.0, cleanup=ld2_cleanup)
finally:
    (OUT / "ld2_run_exit.json").write_text(json.dumps(EXIT if EXIT is not None else {"status": "FAILED_BEFORE_EXIT"},
                                                      indent=1, default=str))
print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}) if EXIT else "none", flush=True)
if (RUN / "ld2_report.json").is_file():
    print("REPORT_SHA256", sha256(RUN / "ld2_report.json"), flush=True)
'''

SMOKE_BODY = r'''
SMOKE = OUT / "ld2_smoke"
RUN = SMOKE / "ld2"
CAL_OUT = SMOKE / "ld2_cal"
SMOKE.mkdir(parents=True, exist_ok=True)
DECOY_PAGES = __DECOY__
HEAVY_ENV = dict(os.environ, LD2_HEAVY="1", PYTHONPATH=str(KIT / "src") + os.pathsep + str(KIT / "scripts"))
HEAVY_ENV["HF_HUB_OFFLINE"] = "1"
OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")
DECOY_REF = SMOKE / "decoy_reference.json"

def s_decoy_reference():
    """Référence FACTICE pour les pages de remplacement : le bootstrap et le critère s'exécutent, sans aucune valeur réelle."""
    DECOY_REF.write_text(json.dumps({"eval": [{"page_id": p, "metrics": {"after": {"recall": 0.9, "precision": 0.7, "n_pred": 10}}}
                                              for p in DECOY_PAGES]}))

def _smoke_argv(phase, extra=()):
    return _argv(phase, ["--pages", "mf_0000", *extra]) if phase in ("T", "TW", "E", "EW") else _argv(phase, extra)

STEPS = [
    {"name": "D", "argv": [sys.executable, str(KIT / "scripts" / "train_ld2.py"), "--phase", "D", "--out", str(RUN),
                            "--cache-dir", "/tmp/ld2_hf"], "timeout_s": TIMEOUTS["D"]},
    # TOUS les tests des deux fichiers, sans -k ; rc écrit dans un fichier, sans bloquer la suite
    {"name": "pytest_heavy", "argv": ["sh", "-c", "cd \"" + str(KIT) + "\" && " + sys.executable + " -m pytest -q -rs "
                                      "tests/test_ld2.py tests/test_ld2_heavy.py > \"" + str(SMOKE) + "/ld2_pytest.txt\" 2>&1; "
                                      "echo $? > \"" + str(SMOKE) + "/ld2_pytest_rc.txt\"; exit 0"],
     "env": HEAVY_ENV, "timeout_s": TIMEOUTS["pytest_heavy"]},
    {"name": "decoy", "fn": s_decoy_reference},
    {"name": "T", "argv": _smoke_argv("T", ["--timing-steps", "3", "--max-steps", "20"]), "timeout_s": TIMEOUTS["T"], "env": OFFLINE},
    {"name": "TW", "argv": lambda: wrapped("TW", _smoke_argv("TW", ["--timing-steps", "3", "--deadline-s", "1200"]), 1200 + 0.75 * LIMITS["ew_grace_s"]), "timeout_s": TIMEOUTS["TW"],
     "env": OFFLINE},
    {"name": "E", "argv": _smoke_argv("E"), "timeout_s": TIMEOUTS["E"], "env": OFFLINE},
    {"name": "EW", "argv": lambda: wrapped("EW", _smoke_argv("EW", ["--deadline-s", "1200"]), 1200 + 0.75 * LIMITS["ew_grace_s"]), "timeout_s": TIMEOUTS["EW"],
     "env": OFFLINE},
    {"name": "FREEZE", "argv": _smoke_argv("FREEZE"), "timeout_s": TIMEOUTS["FREEZE"], "env": OFFLINE},
    {"name": "CAL", "argv": _smoke_argv("CAL", ["--pages", "mf_0000", "--smoke", "--eval-pages", *DECOY_PAGES,
                                                "--cal-out", str(CAL_OUT)]), "timeout_s": __CALS__, "env": OFFLINE},
    {"name": "EVAL", "argv": _smoke_argv("EVAL", ["--smoke", "--eval-pages", *DECOY_PAGES, "--reference-report", str(DECOY_REF)]),
     "timeout_s": __EVALS__, "env": OFFLINE},
]
SUMMARY = {"kind": "SMOKE_KAGGLE_CPU_VRAIS_CHECKPOINTS", "init_sha256": INIT_SHA, "decoder_sha256": DEC_SHA}
EXIT = None
try:
    EXIT = C.execute_pipeline(SMOKE, STEPS, heartbeat_interval_s=60.0)
finally:
    for f in RUN.glob("ld2_*_step*.pt"):                         # la fumée ne rapporte que du JSON
        f.unlink()
    SUMMARY["pipeline"] = {k: EXIT[k] for k in ("status", "step", "exit_code", "durations_s")} if EXIT else None
    for name in ("ld2_pytest.txt", "ld2_pytest_rc.txt"):
        f = SMOKE / name
        if f.is_file():
            SUMMARY[name] = f.read_text()[-6000:]
    for name in ("ld2_manifest.json", "ld2_witness_status.json", "ld2_report.json", "ld2_main_train_log.jsonl",
                 "ld2_witness_train_log.jsonl"):
        f = RUN / name
        if f.is_file():
            SUMMARY[name] = f.read_text()[-20000:] if name.endswith(".jsonl") else json.loads(f.read_text())
    cal = CAL_OUT / "ld2_cal.json"
    if cal.is_file():
        SUMMARY["ld2_cal.json"] = json.loads(cal.read_text())
    (OUT / "ld2_smoke_summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False, default=str))
    shutil.rmtree("/tmp/ld2_hf", ignore_errors=True)
print("SMOKE", SUMMARY.get("pipeline"), flush=True)
'''


def _timeouts() -> dict:
    t = _cfg()["robustness"]["timeouts_s"]
    return dict(t)


def _limits() -> dict:
    r = _cfg()["robustness"]
    return {k: r[k] for k in ("deadline_s", "tail_reserve_s", "stop_margin_s", "eval_margin_s", "ew_grace_s")}


def cell(kind: str, out: str = "/kaggle/working", kit=None, pages=None, ckpt=None, ld1=None, ref=None, init_sha=None,
         dec_sha=None) -> str:
    cfg = _cfg()
    head = (HEAD.replace("__OUT__", repr(out)).replace("__KIT__", repr(kit or KIT_CANDIDATES))
            .replace("__PAGES__", repr(pages or B.PAGES_CANDIDATES)).replace("__CKPT__", repr(ckpt or B.CKPT_CANDIDATES))
            .replace("__LD1__", repr(ld1 or LD1_CANDIDATES)).replace("__REF__", repr(ref or REF_CANDIDATES))
            .replace("__REFP__", repr(cfg["reference"]["sha256_prefix"]))
            .replace("__ESTF__", repr(cfg["eval"]["estimate_factor"]))
            .replace("__INIT_SHA__", repr(init_sha or cfg["init"]["encoder"]["sha256_full"]))
            .replace("__INIT_STEP__", repr(cfg["init"]["encoder"]["step"]))
            .replace("__DEC_SHA__", repr(dec_sha or cfg["init"]["decoder"]["sha256_full"]))
            .replace("__DEC_STEP__", repr(cfg["init"]["decoder"]["step"]))
            .replace("__TIMEOUTS__", repr(_timeouts())).replace("__LIMITS__", repr(_limits())))
    if kind == "full":
        return head + RESOLVERS + _argv_builder("cuda") + FULL_BODY
    return head + RESOLVERS + _argv_builder("cpu") + SMOKE_BODY.replace("__DECOY__", repr(DECOY_PAGES)).replace("__CALS__", repr(SMOKE_CAL_S)).replace("__EVALS__", repr(SMOKE_EVAL_S))


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-2 — LINE-DETR complet (géométrie + CTC, λ = 1/6) et évaluation unique pré-enregistrée (GPU)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell("full")}]}


def build_smoke_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# LD-2 — FUMÉE Kaggle CPU : tests, T/E/EW/FREEZE/CAL sur mf_0000, EVAL sur pages de remplacement"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell("smoke")}]}


def kernel_metadata(slug: str, title: str, gpu: bool = False) -> dict:
    """enable_internet vrai pour tout le carnet (réglage par carnet, pas par étape) ; hors D, HF_HUB_OFFLINE=1."""
    return {"id": slug, "title": title, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [KIT_SLUG], "competition_sources": [],
            "kernel_sources": [B.CKPT_SLUG, B.PAGES_SLUG, LD1_RUN_SLUG], "model_sources": []}


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kaggle_ld2.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kaggle_ld2_smoke.ipynb").write_text(json.dumps(build_smoke_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kit_files.json").write_text(json.dumps(kit_files(), indent=1))
    for sub, slug, title, gpu, nb in (("smoke_kernel", SMOKE_SLUG, "babaorum-ld2-smoke", False, build_smoke_notebook()),
                                      ("full_kernel", RUN_SLUG, "babaorum-ld2-run", True, build_notebook())):
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(slug, title, gpu=gpu), indent=2))
        (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld2")
    print(Path(__file__).parent / "ld2")
