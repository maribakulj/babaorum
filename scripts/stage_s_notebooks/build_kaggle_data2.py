"""Build the DATA-2 notebooks (pC; docs/END_TO_END_DATA2_LOT.txt §7, §11).

Three notebooks, one per carnet of the lot:
  kaggle_data2_generation.ipynb       : generation of mfn_0000..mfn_0019 (commit 23875ab, seed 20261021), rename,
                                        overlap check (mf_0000-0049 only), index. Output: /kaggle/working/gen_out.
  kaggle_data2_fragments.ipynb        : DATA-1 code (amended: pages and sha from the generation manifest), B3 and S2.
                                        Inputs: the generation output, the E2E-1 checkpoints, the bundle v5.
  kaggle_data2_smoke.ipynb            : fumée Kaggle CPU, 1 page, seed 1: generation then fragments on that page.

Robustness (§11 (1), (2), (3)): every step has a timeout from the config; subprocesses run in their own group
(execute_pipeline of scripts/data2_common.py, copied from LD-0 gel 6); stdout and stderr go to files; a heartbeat is
written every 60 s; data2_run_exit.json is written in a finally; the inputs are found by explicit white lists (no
recursive search, nothing listed inside e2e3/mf); faulthandler(600 s) around the imports of cell 1.
The kit is found at explicit candidate paths (is_file, exactly one). Nothing is pushed by this script.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "data2_pages.json"
SMOKE_CONFIG = ROOT / "config" / "data2_smoke.json"
BUNDLE_V5_SHA = "ca41c09272248c5096f2547fe738eee622db5af50414182537ac8e94e43367f5"  # sha256 complet, lu sur le zip (23/10)
BUNDLE_V5_DRIVE = Path("/Users/marcel/Library/CloudStorage/GoogleDrive-mrebg@hotmail.fr/Mon Drive/babaorum/stage_s/bundle/stage_s_bundle_v5.zip")
GEN_KIT = ["scripts/data2_generate.py", "scripts/data2_common.py", "config/data2_pages.json", "tests/test_data2_generate.py"]
FRAG_KIT = ["scripts/data1_fragments.py", "config/data1_fragments.json", "tests/test_data1_fragments.py",
            "scripts/data2_common.py", "scripts/data2_generate.py", "config/data2_pages.json",
            "tests/test_data2_generate.py", "scripts/data2_text_guard.py", "scripts/ld0_zone_coverage.py",
            "config/data2_smoke.json", "scripts/stage_s_notebooks/kaggle_data2_fragments.ipynb",
            "scripts/stage_s_notebooks/kaggle_data2_generation.ipynb", "scripts/stage_s_notebooks/kaggle_data2_smoke.ipynb",
            "tests/fixtures/data1_shard0/data1_manifest.json", "tests/fixtures/data1_shard0/LISEZMOI.txt",
            "tests/fixtures/data2_smoke_v3/data1_manifest.json",
            "scripts/stage_s_notebooks/build_kaggle_data1.py", "scripts/stage_s_notebooks/KAGGLE_INSTALL.txt",
            "scripts/stage_s_notebooks/build_kaggle_data2.py",
            "scripts/e2e3_fp_classes.py", "scripts/g_diag_mf.py", "scripts/e2e4_extraction.py",
            "config/e2e4_extraction.json"]
GENERATOR_REQUIRED = ["Pillow>=12.3,<13", "fonttools>=4.60,<5", "jsonschema>=4.25,<5", "lxml>=6,<7",
                      "numpy>=2.3,<3", "shapely>=2.1,<3"]
GENERATOR_MODULES = ["numpy", "PIL", "lxml", "fontTools", "jsonschema", "shapely"]
FRAG_MODULES = ["numpy", "torch", "PIL", "lxml"]


def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- cell 1: install (pip with timeout and retries, UTC prints, trace file, faulthandler) ----------

def install_cell(required: list, modules: list) -> str:
    lines = [
        "import importlib, os, subprocess, sys, time, datetime, faulthandler",
        "def _utc():",
        "    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')",
        "os.makedirs('/kaggle/working', exist_ok=True)",
        "_TRACE = open('/kaggle/working/data2_cell1_trace.txt', 'a')",
        "def _say(*parts):",
        "    line = ' '.join(str(p) for p in parts)",
        "    print(line, flush=True)",
        "    _TRACE.write(line + '\\n'); _TRACE.flush()",
        "_say('data2 install start', _utc())",
        "REQUIRED_OUTSIDE_IMAGE = " + json.dumps(required),
        "for pkg in REQUIRED_OUTSIDE_IMAGE:",
        "    _say('pip start', pkg, _utc())",
        "    _t0 = time.monotonic()",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--timeout', '30', '--retries', '1', pkg],",
        "                   check=True, timeout=900)",
        "    _say('pip end', pkg, _utc(), round(time.monotonic() - _t0, 1), 's')",
        "faulthandler.dump_traceback_later(600, exit=True, file=_TRACE)",
        "for mod in " + json.dumps(modules) + ":",
        "    _say('import module', mod, _utc())",
        "    importlib.import_module(mod)",
        "_say('imports ok', _utc())",
        "faulthandler.cancel_dump_traceback_later()",
    ]
    return "\n".join(lines) + "\n"


# ---------- cell 2 (shared head): kit by candidates, copied common code, discovery, pipeline ----------

HEAD = r'''import datetime, hashlib, importlib, importlib.util, json, os, platform, shutil, subprocess, sys, zipfile
from pathlib import Path

def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')

def _imp(name):
    print('import', name, _utc(), flush=True)
    return importlib.import_module(name)

print('data2 cell 2 start', _utc(), flush=True)
KIT_CANDIDATES = __KIT_CANDIDATES__
OUT = Path("/kaggle/working")
RAW = OUT / "gen_raw"
SRC = OUT / "heritage"
BUNDLE = OUT / "bundle"
GEN_OUT = OUT / "gen_out"
FRAG_OUT = OUT / "frag_out"
STATE = {}

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

E24_SHA_FULL = "__E24_SHA_FULL__"
BUNDLE_V5_SHA = "__BUNDLE_V5_SHA__"
OVERLAP_ACCEPT = __OVERLAP_ACCEPT__          # §14 (1)/(2): only OVERLAP_TEXT_FLAGGED, never OVERLAP_PAGE
OVERLAP_MENTION = __OVERLAP_MENTION__
def check_reference(path):
    """The E2E-4 v3 report: whole sha256 and alphabet size 74 are checked before any use."""
    assert sha256(path) == E24_SHA_FULL, "E2E-4 report: sha256 differs from the frozen value"
    assert json.loads(Path(path).read_text())["alphabet_size"] == 74, "E2E-4 report: alphabet size is not 74"

_hits = [Path(c) for c in KIT_CANDIDATES if Path(c).is_file()]        # explicit candidates only, no listing
assert len(_hits) == 1, ("kit found %d times among the candidates" % len(_hits), [str(h) for h in _hits])
KIT_DIR = _hits[0].parent.parent
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
C = _load("data2_common", KIT_DIR / "scripts" / "data2_common.py")     # copied LD-0 functions (provenance in header)
print("kit", KIT_DIR, _utc(), flush=True)
'''

# ---------- steps ----------

GEN_STEPS = r'''
def d_controls():
    for m in __GEN_MODULES__:
        _imp(m)
    print("versions", {m: getattr(sys.modules[m], "__version__", None) for m in __GEN_MODULES__}, flush=True)

def d_discover():
    assert not (OUT / "data2_generation_manifest.json").exists(), "the output must start empty"
    found = C.discover_inputs(Path("/kaggle/input"), relpaths=__GEN_RELPATHS__)
    STATE["found"] = found
    print("discovery roots (names only):", found["roots_used"], flush=True)
    assert sha256(found["bundle"]) == BUNDLE_V5_SHA, "bundle v5 sha differs"
    check_reference(found["reference"])

def d_bundle_clean():
    shutil.rmtree(BUNDLE, ignore_errors=True)
    BUNDLE.mkdir(parents=True)
    shutil.rmtree(RAW, ignore_errors=True)
    shutil.rmtree(SRC, ignore_errors=True)

def d_mille_check():
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    for rel, want in manifest["mille_feuilles_sha256"].items():
        got = sha256(SRC / rel[len("heritage-synth/"):])
        assert got == want, ("mille-feuilles file differs", rel)
    print("mille-feuilles: 51 files checked", flush=True)

def d_cleanup(rec):
    mpath = OUT / "data2_notebook_manifest.json"
    mani = json.loads(mpath.read_text()) if mpath.exists() else {}
    mani["exit_status"] = rec.get("status")
    mani["durations_s"] = rec.get("durations_s")
    mani["cleanup"] = {"raw_removed": str(RAW), "heritage_clone_removed": str(SRC), "bundle_removed": str(BUNDLE)}
    mpath.write_text(json.dumps(mani, indent=2))
    for d in (RAW, SRC, BUNDLE):
        shutil.rmtree(d, ignore_errors=True)

SMOKE_PAGES = __SMOKE_PAGES__
SMOKE_SEED = __SMOKE_SEED__
STEPS = [
    {"name": "controls", "fn": d_controls},
    {"name": "discover", "fn": d_discover},
    {"name": "bundle_clean", "fn": d_bundle_clean},
    {"name": "unzip", "argv": lambda: [sys.executable, "-c",
                                       "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])",
                                       str(STATE["found"]["bundle"]), str(BUNDLE)], "timeout_s": __T_UNZIP__},
    {"name": "clone", "argv": lambda: ["git", "clone", "-q", str(BUNDLE / "heritage-synth.gitbundle"), str(SRC)],
     "timeout_s": __T_CLONE__},
    {"name": "checkout", "argv": lambda: ["git", "-C", str(SRC), "checkout", "-q", "__COMMIT__"],
     "timeout_s": __T_CLONE__},
    {"name": "mille_check", "fn": d_mille_check},
    {"name": "generate", "argv": lambda: [sys.executable, str(KIT_DIR / "scripts" / "data2_generate.py"),
                                          "generate", "--src", str(SRC), "--raw", str(RAW)] +
     (["--pages", str(SMOKE_PAGES), "--seed", str(SMOKE_SEED)] if SMOKE_PAGES else []),
     "cwd": str(KIT_DIR), "timeout_s": __T_GENERATE__},
    {"name": "rename", "argv": lambda: [sys.executable, str(KIT_DIR / "scripts" / "data2_generate.py"),
                                        "rename", "--raw", str(RAW), "--out", str(GEN_OUT),
                                        "--pages", str(SMOKE_PAGES or 20)],
     "cwd": str(KIT_DIR), "timeout_s": __T_OVERLAP__},
    {"name": "overlap", "argv": lambda: [sys.executable, str(KIT_DIR / "scripts" / "data2_generate.py"),
                                         "overlap", "--out", str(GEN_OUT), "--mf-dir", str(STATE["found"]["run3_mf_dir"]),
                                         "--reference-report", str(STATE["found"]["reference"]),
                                         "--pages", str(SMOKE_PAGES or 20)],
     "cwd": str(KIT_DIR), "timeout_s": __T_OVERLAP__},
    {"name": "index", "argv": lambda: [sys.executable, str(KIT_DIR / "scripts" / "data2_generate.py"),
                                       "index", "--out", str(GEN_OUT), "--pages", str(SMOKE_PAGES or 20)],
     "cwd": str(KIT_DIR), "timeout_s": __T_INDEX__},
]
'''

FRAG_STEPS = r'''
GEN_MANIFEST = __GEN_MANIFEST__            # callable or path: the generation manifest (input or smoke output)

def d_controls():
    for m in __FRAG_MODULES__:
        _imp(m)

def d_discover():
    assert not (FRAG_OUT / "data1_manifest.json").exists(), "the output must start empty"
    found = C.discover_inputs(Path("/kaggle/input"), relpaths=__FRAG_RELPATHS__)
    STATE["found"] = found
    print("discovery roots (names only):", found["roots_used"], flush=True)
    assert sha256(found["bundle"]) == BUNDLE_V5_SHA, "bundle v5 sha differs"
    check_reference(found["reference"])

def d_bundle_clean():
    shutil.rmtree(BUNDLE, ignore_errors=True)
    BUNDLE.mkdir(parents=True)

def d_kit_overlay():
    for rel in __FRAG_KIT__:
        src = KIT_DIR / rel
        dst = BUNDLE / "babaorum" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def d_frag_check():
    gm = STATE["found"]["generation_manifest"] if "generation_manifest" in STATE["found"] else GEN_MANIFEST()
    manifest = json.loads(Path(gm).read_text())
    root = Path(gm).parent
    for key, want in manifest["files_sha256"].items():
        assert sha256(root / "mfn" / key) == want, ("generation file differs", key)
    STATE["gen_manifest"] = Path(gm)
    print("generation files checked:", len(manifest["files_sha256"]), flush=True)
    overlap_path = __OVERLAP_PATH__
    overlap = json.loads(Path(overlap_path).read_text())
    st = overlap.get("status")
    if st != "OK" and not (st == "OVERLAP_TEXT_FLAGGED" and OVERLAP_ACCEPT.get("accepted") is True):
        raise RuntimeError("BLOCKED_OVERLAP: overlap report status is %s ; no fragment is produced" % st)
    FRAG_OUT.mkdir(parents=True, exist_ok=True)
    if st != "OK":
        (FRAG_OUT / "LISEZMOI_DATA2.txt").write_text(OVERLAP_MENTION + "\n", encoding="utf-8")
    STATE["overlap_status"] = st
    print("overlap report:", st, flush=True)

def d_sanitize():
    G2 = _load("data2_text_guard", KIT_DIR / "scripts" / "data2_text_guard.py")
    STATE["sanitize"] = G2.sanitize_file(FRAG_OUT / "data1_manifest.json")   # §15 (1)(2): original removed
    print("sanitize:", STATE["sanitize"]["removed_keys"], "original sha consigned", flush=True)

def d_text_guard():
    G2 = _load("data2_text_guard", KIT_DIR / "scripts" / "data2_text_guard.py")
    manifest = json.loads((FRAG_OUT / "data1_manifest.json").read_text())
    payload_files = [FRAG_OUT / "fragments" / arm / (pid + ".json")          # explicit paths, no listing
                     for arm in ("B3", "S2") for pid in manifest["pages_fragments"]]
    payload_files = [p for p in payload_files if p.is_file()]
    problems = G2.check_all([json.loads(p.read_text()) for p in payload_files], manifest)
    template = G2.template_from_source((KIT_DIR / "scripts" / "data1_fragments.py").read_text())
    cfg_e2e4 = json.loads((KIT_DIR / "config" / "e2e4_extraction.json").read_text())
    if not G2.readme_matches((FRAG_OUT / "LISEZMOI.txt").read_text(encoding="utf-8"), manifest, template, cfg_e2e4):
        problems.append("LISEZMOI.txt: texte différent du texte attendu")
    accepted = STATE.get("overlap_status") not in (None, "OK")
    if accepted and not G2.exact_text_matches((FRAG_OUT / "LISEZMOI_DATA2.txt").read_text(encoding="utf-8"),
                                              OVERLAP_MENTION + "\n"):
        problems.append("LISEZMOI_DATA2.txt: texte différent de la mention")
    extra = G2.unexpected_output(FRAG_OUT, G2.expected_output(manifest, accepted))
    if extra:
        problems.append("sortie hors liste: " + ", ".join(extra[:5]))
    if problems:
        for p in payload_files:
            p.unlink()
        raise RuntimeError("BLOCKED_TEXT_FIELD: " + "; ".join(problems[:5]))
    print("text guard: %d payload(s), manifest, LISEZMOI files and output list checked" % len(payload_files), flush=True)

def d_cleanup(rec):
    mpath = FRAG_OUT / "data2_notebook_manifest.json"
    FRAG_OUT.mkdir(parents=True, exist_ok=True)
    mani = {"exit_status": rec.get("status"), "durations_s": rec.get("durations_s"),
            "cleanup": {"bundle_removed": str(BUNDLE)},
            "overlap_status": STATE.get("overlap_status"),
            "original_manifest_sha256": (STATE.get("sanitize") or {}).get("original_sha256"),
            "clean_manifest_sha256": (STATE.get("sanitize") or {}).get("clean_sha256"),
            "overlap_text_mention": OVERLAP_MENTION if STATE.get("overlap_status") not in (None, "OK") else None}
    mpath.write_text(json.dumps(mani, indent=2))
    shutil.rmtree(BUNDLE, ignore_errors=True)

ENV = dict(os.environ, PYTHONPATH=str(BUNDLE / "babaorum" / "src") + os.pathsep + str(BUNDLE / "babaorum" / "scripts"),
           DATA2_WITH_TORCH="1")          # §11 (3): the DATA-2 test file has one DATA-1 test that imports torch
STEPS = [
    {"name": "controls", "fn": d_controls},
    {"name": "discover", "fn": d_discover},
    {"name": "bundle_clean", "fn": d_bundle_clean},
    {"name": "unzip", "argv": lambda: [sys.executable, "-c",
                                       "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])",
                                       str(STATE["found"]["bundle"]), str(BUNDLE)], "timeout_s": __T_UNZIP__},
    {"name": "kit_overlay", "fn": d_kit_overlay},
    {"name": "generation_check", "fn": d_frag_check},
    {"name": "pytest", "argv": lambda: [sys.executable, "-m", "pytest", "-q", "tests/test_data1_fragments.py",
                                        "tests/test_data2_generate.py"],
     "cwd": str(BUNDLE / "babaorum"), "env": lambda: ENV, "timeout_s": __T_PYTEST__},
    {"name": "fragments", "argv": lambda: [sys.executable, "scripts/data1_fragments.py", "--confirmed-go",
                                           "--tmp", str(STATE["gen_manifest"].parent),
                                           "--calibration", str(STATE["found"]["checkpoint_b3"].parent.parent),
                                           "--generation-manifest", str(STATE["gen_manifest"]),
                                           "--out-dir", str(FRAG_OUT)] +
     (["--pages", ",".join(__SMOKE_PAGE_IDS__)] if __SMOKE_PAGE_IDS__ else []),
     "cwd": str(BUNDLE / "babaorum"), "env": lambda: ENV, "timeout_s": __T_FRAGMENTS__},
    {"name": "sanitize", "fn": d_sanitize},
    {"name": "text_guard", "fn": d_text_guard},
]
'''

PIPELINE_TAIL = r'''
EXIT = C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=60.0, cleanup=d_cleanup)
print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}), flush=True)
'''


def _fill(text: str, mapping: dict) -> str:
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


def _gen_block(cfg: dict, smoke: bool) -> str:
    t = cfg["timeouts_s"]
    return _fill(GEN_STEPS, {
        "__GEN_MODULES__": json.dumps(GENERATOR_MODULES),
        "__GEN_RELPATHS__": json.dumps(cfg["discovery"]["relpaths"]),
        "__SMOKE_PAGES__": str(1 if smoke else 0),
        "__SMOKE_SEED__": str(1 if smoke else cfg["generator"]["seed"]),
        "__T_UNZIP__": str(t["unzip"]), "__T_CLONE__": str(t["clone"]), "__T_GENERATE__": str(t["generate"]),
        "__T_OVERLAP__": str(t["overlap"]), "__T_INDEX__": str(t["index"]), "__COMMIT__": cfg["generator"]["commit"],
    })


def _frag_block(cfg: dict, smoke: bool, gen_manifest_expr: str, overlap_expr: str) -> str:
    t = cfg["timeouts_s"]
    rel = cfg["discovery"]["relpaths_smoke_fragments" if smoke else "relpaths_fragments"]
    block = _fill(FRAG_STEPS, {
        "__FRAG_MODULES__": json.dumps(FRAG_MODULES),
        "__FRAG_RELPATHS__": json.dumps(rel),
        "__FRAG_KIT__": json.dumps(FRAG_KIT),
        "__GEN_MANIFEST__": gen_manifest_expr,
        "__OVERLAP_PATH__": overlap_expr,
        "__SMOKE_PAGE_IDS__": json.dumps(["mfn_0000"] if smoke else []),
        "__T_UNZIP__": str(t["unzip"]), "__T_PYTEST__": str(t["pytest"]), "__T_FRAGMENTS__": str(t["fragments"]),
    })
    return block


def _suffix(block: str, names: list) -> str:
    """Renames the fragment identifiers so that the smoke carnet can hold both pipelines in one cell."""
    import re
    for n in names:
        block = re.sub(r"\b" + n + r"\b", n + "_f", block)
    return block


FRAG_NAMES = ["d_controls", "d_discover", "d_bundle_clean", "d_kit_overlay", "d_frag_check", "d_cleanup", "STEPS"]


def _accept_real(cfg: dict) -> dict:
    r = cfg["overlap_text_accepted_real"]
    return {"accepted": r["accepted"], "by": r["by"], "date": r["date"], "condition": r["condition"]}


def _head(cfg: dict, accept: dict, mention: str) -> str:
    return _fill(HEAD, {"__KIT_CANDIDATES__": json.dumps(cfg["discovery"]["kit_dataset_candidates"]),
                        "__E24_SHA_FULL__": cfg["page_reference"]["sha256_full"],
                        "__BUNDLE_V5_SHA__": BUNDLE_V5_SHA,
                        "__OVERLAP_ACCEPT__": repr(accept),
                        "__OVERLAP_MENTION__": json.dumps(mention, ensure_ascii=False)})


def gen_cell(smoke: bool) -> str:
    cfg = _cfg()
    head = _fill(HEAD, {"__KIT_CANDIDATES__": json.dumps(cfg["discovery"]["kit_dataset_candidates"]),
                        "__E24_SHA_FULL__": cfg["page_reference"]["sha256_full"],
                        "__BUNDLE_V5_SHA__": BUNDLE_V5_SHA,
                        "__OVERLAP_ACCEPT__": "{}", "__OVERLAP_MENTION__": '""'})
    return head + _gen_block(cfg, smoke) + PIPELINE_TAIL


def frag_cell() -> str:
    cfg = _cfg()
    head = _head(cfg, _accept_real(cfg), cfg["overlap_text_accepted_real"]["mention"])
    return head + _frag_block(cfg, False, "(lambda: STATE['found']['generation_manifest'])",
                              "STATE['found']['overlap_report']") + PIPELINE_TAIL


def smoke_cell() -> str:
    """One cell, one pipeline: generation of 1 page (seed 1), then the fragments on that page (smoke)."""
    cfg = _cfg()
    smoke_flag = json.loads(SMOKE_CONFIG.read_text())["overlap_text_flag_accepted_by_pA"]
    head = _head(cfg, {"accepted": smoke_flag["accepted"], "scope": "smoke"},
                 "FUMÉE : " + cfg["overlap_text_accepted_real"]["mention"])
    gen = _gen_block(cfg, True).replace("STEPS = [", "STEPS_GEN = [").replace('{"name": "', '{"name": "gen_')
    frag = _suffix(_frag_block(cfg, True, "(lambda: GEN_OUT / 'data2_generation_manifest.json')",
                               "GEN_OUT / 'data2_overlap_report.json'"), FRAG_NAMES)
    frag = frag.replace("STEPS_f = [", "STEPS_F = [").replace('{"name": "', '{"name": "frag_')
    tail = ("\nSTEPS = STEPS_GEN + STEPS_F\n" + PIPELINE_TAIL)
    return head + gen + frag + tail


def notebook(title: str, install: str, code: str) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": title},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": install},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
            ]}


def build_all() -> dict:
    return {
        "kaggle_data2_generation.ipynb": notebook(
            "# DATA-2 — génération de mfn_0000..0019 (commit 23875ab, graine 20261021), renommage, recouvrement, index",
            install_cell(GENERATOR_REQUIRED, GENERATOR_MODULES), gen_cell(smoke=False)),
        "kaggle_data2_fragments.ipynb": notebook(
            "# DATA-2 — fragments B3 et S2 (DATA-1 amendé, pages et sha du manifeste de génération)",
            install_cell([], FRAG_MODULES), frag_cell()),
        "kaggle_data2_smoke.ipynb": notebook(
            "# DATA-2 — FUMÉE Kaggle CPU : 1 page (graine 1) : génération puis fragments",
            install_cell(GENERATOR_REQUIRED, GENERATOR_MODULES), smoke_cell()),
    }


def main() -> None:
    out_dir = Path(__file__).parent
    for name, nb in build_all().items():
        (out_dir / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(out_dir / name)


if __name__ == "__main__":
    main()
