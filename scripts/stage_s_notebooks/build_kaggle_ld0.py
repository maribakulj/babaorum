"""Build kaggle_ld0_zone_coverage.ipynb (LD-0 on Kaggle CPU, pC, docs/END_TO_END_LD0_LOT.txt, amendements 1 à 4).

Inputs: bundle v6 (sha 91402c71), the LD-0 kit (scripts/ld0_zone_coverage.py, config, test, scripts/ld0_vendor/)
mounted from the kit dataset, the E2E-4 v3 report (page sha reference), and the E2E-3 run 3 output (e2e3/mf).
The notebook COPIES the 100 listed files (mf_0000..mf_0049) into /kaggle/working/ld0_pages and checks each sha.
§14 (amendment 4): every step has a timeout (config timeouts_s); each subprocess runs in its own process group,
stdout and stderr go to files; a heartbeat is written every 60 s; ld0_run_exit.json is written in a finally.
The execution cell only declares the steps; execute_pipeline (scripts/ld0_zone_coverage.py) runs them.
The first code cell is the common install cell of KAGGLE_INSTALL.txt: two lines are replaced (§12 (1)) and the pip
line gets timeout=900 (§14 (1)); KAGGLE_INSTALL.txt itself is not modified.
"""
import hashlib
import json
from pathlib import Path

BUNDLE_NAME = "stage_s_bundle_v6.zip"
BUNDLE_SHA = "91402c71b789f36d5c30c6dd1d5de400ec1ae6358bc327840877c26486862948"
BUNDLE_DRIVE = Path("/Users/marcel/Library/CloudStorage/GoogleDrive-mrebg@hotmail.fr/Mon Drive/babaorum/stage_s/bundle")
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "ld0_zone_coverage.json"
KIT_FILES = ["scripts/ld0_zone_coverage.py", "config/ld0_zone_coverage.json", "tests/test_ld0_zone_coverage.py"]
INSTALL_TXT = Path(__file__).with_name("KAGGLE_INSTALL.txt")
REQUIRED_LINE_NEW = ('REQUIRED_OUTSIDE_IMAGE = ["transformers==5.18.0", "huggingface_hub==1.33.0"]          '
                     '# LD-0 (§12 (1))')
FOR_MOD_NEW = "for mod in ('numpy', 'torch', 'PIL', 'lxml', 'transformers', 'huggingface_hub', 'torchvision', 'cv2'):"
PIP_LINE_OLD = "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)"
IMPORT_LINE_OLD = "import importlib, os, subprocess, sys"
IMPORT_DEFS = [          # §14 (8): UTC clock, printed with flush; the pip line below loses -q
    "import datetime, time, faulthandler, os",
    "os.environ.update({'HF_HUB_DOWNLOAD_TIMEOUT': '60', 'HF_HUB_ETAG_TIMEOUT': '30', 'HF_HUB_DISABLE_XET': '1'})   "
    "# §14 (10)-(11): before the first import of huggingface_hub",
    "def _utc():",
    "    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')",
    "os.makedirs('/kaggle/working', exist_ok=True)",
    "_TRACE = open('/kaggle/working/ld0_cell1_trace.txt', 'a')   # §14 (11): the same lines, with flush, in a file",
    "def _say(*parts):",
    "    line = ' '.join(str(p) for p in parts)",
    "    print(line, flush=True)",
    "    _TRACE.write(line + '\\n'); _TRACE.flush()",
    "_say('ld0 install start', _utc())",
]
PIP_BLOCK = [            # inside the loop: 4 spaces, as the common block after its dedent
    "    _say('pip start', pkg, _utc())",
    "    _t0 = time.monotonic()",
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--timeout', '30', '--retries', '1', pkg], check=True, timeout=900)   "
    "# LD-0 (§14 (1)): 900 s per package, no -q (§14 (8))",
    "    _say('pip end', pkg, _utc(), round(time.monotonic() - _t0, 1), 's')",
]
IMPORTS_DONE = "_say('imports ok', _utc())"
CANCEL_ARM = "faulthandler.cancel_dump_traceback_later()"
ARM_BEFORE_IMPORTS = "faulthandler.dump_traceback_later(600, exit=True, file=_TRACE)"   # §14 (11): 600 s
# tests that import this builder or compare with ~/axel: not run in the notebook (§10, §11 (3))
DESELECT = [
    "tests/test_ld0_zone_coverage.py::test_notebook_install_cell_is_the_common_block_except_the_declared_lines",
    "tests/test_ld0_zone_coverage.py::test_run_cell_prints_the_utc_hour_before_each_heavy_import",
    "tests/test_ld0_zone_coverage.py::test_notebook_in_the_repository_equals_build_output",
    "tests/test_ld0_zone_coverage.py::test_no_recursive_search_in_the_notebook_cell",
    "tests/test_ld0_zone_coverage.py::test_notebook_execution_cell_has_no_subprocess_call_and_no_capture",
    "tests/test_ld0_zone_coverage.py::test_hf_timeouts_are_set_before_the_first_import_of_huggingface_hub",
    "tests/test_ld0_zone_coverage.py::test_measure_runs_offline_and_download_runs_with_disable_xet",
    "tests/test_ld0_zone_coverage.py::test_faulthandler_is_armed_around_the_import_loop_of_cell_1",
    "tests/test_ld0_zone_coverage.py::test_cell_1_trace_file_receives_the_lines_with_flush",
    "tests/test_ld0_zone_coverage.py::test_notebook_kit_candidates_are_the_two_explicit_paths_of_the_config_and_exactly_one_must_exist",
    "tests/test_ld0_zone_coverage.py::test_factor_formula_equals_load_scaled_page_on_mf_0000",
    "tests/test_ld0_zone_coverage.py::test_axel_vendored_files_are_identical_to_commit_7b75303",
    "tests/test_ld0_zone_coverage.py::test_axel_union_of_members_is_the_same_as_zones_of",
]
REFERENCE_PREFIX = "714bc571"


def common_install_cell() -> str:
    """The common install cell, read from KAGGLE_INSTALL.txt (the file is the source of truth; not modified)."""
    lines = INSTALL_TXT.read_text().splitlines()
    start = lines.index("Cellule (première cellule de code) :") + 1
    end = lines.index("Lancement des scripts (PYTHONPATH vers les sources du bundle, jamais d'import depuis le dossier courant) :")
    block = [ln[4:] for ln in lines[start:end] if ln.startswith("    ")]
    return "\n".join(block) + "\n"


def install_cell() -> str:
    """The common cell with the declared lines replaced (§12 (1), §14 (1) and (8)); every other line is kept."""
    out = {"required": 0, "pip": 0, "formod": 0, "imports": 0, "done": 0}
    result = []
    for ln in common_install_cell().splitlines():
        stripped = ln.strip()
        if ln.lstrip().startswith("REQUIRED_OUTSIDE_IMAGE ="):
            result.append(REQUIRED_LINE_NEW)
            out["required"] += 1
        elif ln.lstrip().startswith("for mod in ("):
            result.append(ARM_BEFORE_IMPORTS)
            result.append(FOR_MOD_NEW)
            out["formod"] += 1
        elif stripped == PIP_LINE_OLD:
            result.extend(PIP_BLOCK)
            out["pip"] += 1
        elif stripped == IMPORT_LINE_OLD:
            result.append(ln)
            result.extend(IMPORT_DEFS)
            out["imports"] += 1
        elif "importlib.import_module(mod)" in ln:
            result.append("    _say('import module', mod, _utc())")   # §14 (11): UTC hour before each module
            result.append(ln)
            result.append(IMPORTS_DONE)
            result.append(CANCEL_ARM)
            out["done"] += 1
        else:
            result.append(ln)
    assert out == {"required": 1, "pip": 1, "formod": 1, "imports": 1, "done": 1}, out
    return "\n".join(result) + "\n"


def _config() -> dict:
    return json.loads(CONFIG.read_text())


CELL = r'''import os
os.environ.update(__HF_ENV__)          # §14 (10): the first action of cell 2, before any import of huggingface_hub
import datetime, hashlib, importlib, importlib.util, json, platform, shutil, sys, zipfile
from importlib.metadata import version
from pathlib import Path


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')


def _imp(name):
    """§14 (8): the UTC hour is printed (flush) before each heavy import."""
    print('import', name, _utc(), flush=True)
    return importlib.import_module(name)


print('ld0 cell 2 start', _utc(), flush=True)

BUNDLE_SHA = "__BUNDLE_SHA__"
KIT_SHA = __KIT_SHA__
KIT_DETECTOR_FILES = __DETECTOR_FILES__
VENDOR_SHA = __VENDOR_SHA__
PAGES = __PAGES__
REFERENCE_PREFIX = "__REFERENCE_PREFIX__"
DESELECT = __DESELECT__
TIMEOUTS = __TIMEOUTS__
KIT_CANDIDATES = __KIT_CANDIDATES__   # §14 (12): explicit paths, checked with is_file (no listing)
KIT_DIR = None
OUT = Path("/kaggle/working")
PAGES_ROOT = OUT / "ld0_pages"
BUNDLE = OUT / "bundle"
ROOT_LD0 = BUNDLE / "babaorum"
AXEL_RUNS = OUT / "axel_runs"          # empty folder: axel.layout.chains reads nothing from it
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT_LD0 / "src"), str(ROOT_LD0 / "scripts"),
                                                   str(ROOT_LD0 / "scripts" / "ld0_vendor")]),
           AXEL_RUNS=str(AXEL_RUNS))
ENV_DOWNLOAD = dict(ENV)
ENV_DOWNLOAD.update(__HF_ENV__)                        # §14 (11): HF_HUB_DISABLE_XET and the timeouts for the download
ENV_MEASURE = dict(ENV_DOWNLOAD, HF_HUB_OFFLINE="1")   # §14 (10): the measure reads the cache, without network
STATE = {}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# the script is loaded from the kit dataset at its explicit mount point (its top level imports numpy and PIL only)
print('ld0 loading the script', _utc(), flush=True)
try:
    _hits = [Path(c) for c in KIT_CANDIDATES if Path(c).is_file()]
    assert len(_hits) == 1, ("kit script found %d times among the candidates" % len(_hits), [str(h) for h in _hits])
    KIT_DIR = _hits[0].parent.parent
    print("kit script", _hits[0], _utc(), flush=True)
    _spec = importlib.util.spec_from_file_location("ld0_zone_coverage", _hits[0])
    L0 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(L0)
except BaseException as _exc:  # §14 (5): the exit record exists even when the script cannot be loaded
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ld0_run_exit.json").write_text(json.dumps({"status": "FAILED", "step": "load_script", "exit_code": "EXCEPTION",
                                                       "exception": f"{type(_exc).__name__}: {_exc}"}, indent=2))
    raise


def d_controls():
    torch = _imp("torch")
    torchvision = _imp("torchvision")
    cv2 = _imp("cv2")
    transformers = _imp("transformers")
    numpy = _imp("numpy")
    PIL = _imp("PIL")
    huggingface_hub = _imp("huggingface_hub")
    from torchvision.ops import nms
    assert shutil.which("ps"), "ps missing"
    print("torch", torch.__version__, "torchvision", torchvision.__version__, "cv2", cv2.__version__,
          "transformers", transformers.__version__, "huggingface_hub", huggingface_hub.__version__, flush=True)
    kept = nms(torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0]]), torch.tensor([0.9, 0.8]), 0.5)
    assert kept.tolist() == [0], ("torchvision.ops.nms check failed", kept.tolist())


def d_discover():
    assert not (OUT / "ld0_zone_coverage_report.json").exists(), "the output must start empty"
    assert (KIT_DIR / "scripts" / "ld0_zone_coverage.py").is_file(), "kit dataset not mounted at its explicit path"
    found = L0.discover_inputs(Path("/kaggle/input"))
    assert found["kit"] == KIT_DIR / "scripts" / "ld0_zone_coverage.py", found["kit"]
    STATE.update(found=found, zip=found["bundle"], ref=found["reference"], src=found["run3_mf_dir"])
    print("discovery roots (names only, down to level 3):", found["roots_used"], flush=True)
    assert sha256(STATE["zip"]) == BUNDLE_SHA, "bundle sha mismatch"


def d_bundle_clean():
    shutil.rmtree(BUNDLE, ignore_errors=True)
    BUNDLE.mkdir(parents=True)
    AXEL_RUNS.mkdir(parents=True, exist_ok=True)


def d_kit_overlay():
    kit_sha = {}
    for rel, want in KIT_SHA.items():
        src = KIT_DIR / rel
        kit_sha[rel] = sha256(src)
        assert kit_sha[rel] == want, ("kit sha mismatch", rel, kit_sha[rel])
        dst = ROOT_LD0 / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    vendor_sha = {}
    for rel, want in VENDOR_SHA.items():
        src = KIT_DIR / "scripts" / "ld0_vendor" / rel
        vendor_sha[rel] = sha256(src)
        assert vendor_sha[rel] == want, ("vendored sha mismatch", rel, vendor_sha[rel])
        dst = ROOT_LD0 / "scripts" / "ld0_vendor" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    STATE.update(kit_sha=kit_sha, vendor_sha=vendor_sha)
    manifest = {"bundle_sha256": BUNDLE_SHA, "kit_sha256": kit_sha, "vendor_sha256": vendor_sha,
                "reference_sha256": sha256(STATE["ref"]), "python": platform.python_version(),
                "timeouts_s": TIMEOUTS, "discovery_roots": STATE["found"]["roots_used"]}
    (OUT / "ld0_notebook_manifest.json").write_text(json.dumps(manifest, indent=2))


def d_pages():
    ref_sha = json.loads(STATE["ref"].read_text())["steps"]["page_cache"]["sha256"]
    assert sha256(STATE["ref"]).startswith(REFERENCE_PREFIX), "E2E-4 reference sha differs"
    (PAGES_ROOT / "pages").mkdir(parents=True, exist_ok=True)
    (PAGES_ROOT / "images").mkdir(parents=True, exist_ok=True)
    pages_sha, copied = {}, 0
    for pid in PAGES:                     # explicit paths built from the list in the config: no listing, no glob
        for key in (f"pages/{pid}.json", f"images/{pid}.png"):
            assert ref_sha.get(key) == sha256(STATE["src"] / key), ("page differs from the E2E-4 reference", key)
            shutil.copy2(STATE["src"] / key, PAGES_ROOT / key)
            pages_sha[key] = sha256(PAGES_ROOT / key)
            copied += 1
    assert copied == 2 * len(PAGES) == 100
    STATE["pages_sha"] = pages_sha
    print("copied", copied, "files", flush=True)


def d_cleanup(rec):
    """§12 (10): the sha of the bundle, the cache and the page copies go into the manifest, then the folders go."""
    mpath = OUT / "ld0_notebook_manifest.json"
    mani = json.loads(mpath.read_text()) if mpath.exists() else {}
    mani["cleanup"] = {"bundle_zip_sha256": BUNDLE_SHA, "bundle_extracted_removed": str(BUNDLE),
                       "hf_cache_removed": str(OUT / "hf_cache"),
                       "hf_cache_model_files_verified_by_script": json.loads(KIT_DETECTOR_FILES),
                       "ld0_pages_sha256": STATE.get("pages_sha", {}), "ld0_pages_removed": str(PAGES_ROOT)}
    mani["exit_status"] = rec.get("status")
    mpath.write_text(json.dumps(mani, indent=2))
    for d in (BUNDLE, OUT / "hf_cache", PAGES_ROOT):
        shutil.rmtree(d, ignore_errors=True)


DOWNLOAD_ARGV = lambda: [sys.executable, "scripts/ld0_zone_coverage.py", "--confirmed-go", "--download-only",
                         "--cache-dir", str(OUT / "hf_cache"), "--out-dir", str(OUT)]
PYTEST_ARGV = [sys.executable, "-m", "pytest", "-q", "tests/test_ld0_zone_coverage.py"] + \
    sum((["--deselect", d] for d in DESELECT), [])
MEASURE_ARGV = lambda: [sys.executable, "scripts/ld0_zone_coverage.py", "--confirmed-go",
                        "--root", str(PAGES_ROOT), "--cache-dir", str(OUT / "hf_cache"),
                        "--reference-report", str(STATE["ref"]), "--out-dir", str(OUT)]
AGGREGATE_ARGV = lambda: [sys.executable, "scripts/ld0_zone_coverage.py", "--confirmed-go",
                          "--aggregate-only", "--out-dir", str(OUT)]
STEPS = [
    {"name": "controls", "fn": d_controls},
    {"name": "discover", "fn": d_discover},
    {"name": "bundle_clean", "fn": d_bundle_clean},
    {"name": "unzip", "argv": lambda: [sys.executable, "-c",
                                       "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])",
                                       str(STATE["zip"]), str(BUNDLE)],
     "timeout_s": TIMEOUTS["unzip"]},
    {"name": "kit_overlay", "fn": d_kit_overlay},
    {"name": "pages_copy", "fn": d_pages},
    {"name": "download", "argv": DOWNLOAD_ARGV, "cwd": str(ROOT_LD0), "env": ENV_DOWNLOAD,
     "timeout_s": TIMEOUTS["download"], "retry": 1,
     "before_retry": lambda: L0.clean_partial_cache(OUT / "hf_cache", "__HF_REPO__")},
    {"name": "pytest", "argv": PYTEST_ARGV, "cwd": str(ROOT_LD0), "env": ENV, "timeout_s": TIMEOUTS["pytest"]},
    {"name": "measure", "argv": MEASURE_ARGV, "cwd": str(ROOT_LD0), "env": ENV_MEASURE,
     "timeout_s": TIMEOUTS["measure"],
     "after_timeout": {"name": "aggregate", "argv": AGGREGATE_ARGV, "cwd": str(ROOT_LD0), "env": ENV,
                       "timeout_s": TIMEOUTS["aggregate"]}},
]

EXIT = L0.execute_pipeline(OUT, STEPS, heartbeat_interval_s=float(__HEARTBEAT__), cleanup=d_cleanup)
print("EXIT", json.dumps({k: EXIT[k] for k in ("status", "step", "exit_code")}), flush=True)
'''


def build(kit_sha: dict, vendor_sha: dict, pages: list) -> dict:
    cfg = _config()
    code = (CELL.replace("__BUNDLE_SHA__", BUNDLE_SHA)
                .replace("__KIT_CANDIDATES__", json.dumps(cfg["discovery"]["kit_dataset_candidates"]))
                .replace("__DETECTOR_FILES__", json.dumps(json.dumps({k: v["sha256"] for k, v in
                                                                   cfg["detector"]["files"].items()})))
                .replace("__KIT_SHA__", json.dumps(kit_sha))
                .replace("__VENDOR_SHA__", json.dumps(vendor_sha))
                .replace("__PAGES__", json.dumps(pages))
                .replace("__REFERENCE_PREFIX__", REFERENCE_PREFIX)
                .replace("__DESELECT__", json.dumps(DESELECT))
                .replace("__TIMEOUTS__", json.dumps(cfg["timeouts_s"]))
                .replace("__HEARTBEAT__", str(cfg["heartbeat_interval_s"]))
                .replace("__HF_ENV__", json.dumps(cfg["hf_environment"] and {k: v for k, v in cfg["hf_environment"].items()
                                                                              if k.startswith("HF_")}))
                .replace("__HF_REPO__", cfg["detector"]["repo"]))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source":
                 "# babaorum LD-0 — couverture des lignes par les zones PP-DocLayoutV3 (Kaggle CPU, mf_0000-0049)\n"
                 "Lot : docs/END_TO_END_LD0_LOT.txt (amendements 1 à 4). Aucun GPU, aucune métrique de détection. "
                 "Entrées : bundle v6 (sha 91402c71), kit LD-0 (dans le dataset babaorum-ld0-kit), sortie E2E-4 v3 "
                 "(rapport, référence des sha), sortie E2E-3 run 3 (e2e3/mf). Les 100 fichiers listés sont copiés dans "
                 "/kaggle/working/ld0_pages et contrôlés. Chaque étape a un délai ; le statut et l'étape en cas d'arrêt "
                 "sont dans ld0_run_exit.json ; le battement est dans heartbeat.jsonl ; les journaux sont dans logs/."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": install_cell()},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
            ]}


def main() -> None:
    cfg = _config()
    pages = list(cfg["pages"]["list"])
    assert pages == [f"mf_{i:04d}" for i in range(50)]
    bundle = BUNDLE_DRIVE / BUNDLE_NAME
    assert bundle.exists(), f"bundle v6 missing: {bundle}"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == BUNDLE_SHA
    kit_sha = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in KIT_FILES}
    vendor_sha = dict(cfg["vendor"]["files"])
    nb = build(kit_sha, vendor_sha, pages)
    out = Path(__file__).with_name("kaggle_ld0_zone_coverage.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
