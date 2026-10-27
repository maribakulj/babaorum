"""Build kaggle_e2e5_lambda.ipynb (E2E-5 B3 with lambda = 1/6, Kaggle GPU T4, pC; docs/END_TO_END_E2E5_LOT.txt).

Thin wrapper on the model of build_kaggle_e2e4.py. Inputs: bundle v6 (stage_s_bundle_v6.zip, rebuilt because
stage_s_arms and train_stage_s_v2 changed), the E2E-5 kit (pilot, config, tests) and the E2E-3 run 3 output
(e2e3/mf and e2e3_fp_classes_run3.json, NOT regenerated).
Order: first cell = the common install cell of KAGGLE_INSTALL.txt (identical in every stage S notebook), then the
cheap controls (GPU, Python, ps, bundle sha, kit sha, tests), then the pilot with a hard stop at 38 min.
The notebook prints only the status, the sha of the report and the step-0 ratio.
"""
import hashlib
import json
from pathlib import Path

BUNDLE_NAME = "stage_s_bundle_v6.zip"
BUNDLE_DRIVE = Path("/Users/marcel/Library/CloudStorage/GoogleDrive-mrebg@hotmail.fr/Mon Drive/babaorum/stage_s/bundle")
KIT_FILES = ["scripts/e2e5_lambda.py", "config/e2e5_lambda.json", "tests/test_e2e5_lambda.py"]
ROOT = Path(__file__).resolve().parents[2]
HARD_STOP_MIN = 38

INSTALL_TXT = Path(__file__).with_name("KAGGLE_INSTALL.txt")


def install_cell() -> str:
    """The common install cell, read from KAGGLE_INSTALL.txt (the file is the source of truth)."""
    lines = INSTALL_TXT.read_text().splitlines()
    start = lines.index("Cellule (première cellule de code) :") + 1
    end = lines.index("Lancement des scripts (PYTHONPATH vers les sources du bundle, jamais d'import depuis le dossier courant) :")
    block = [ln[4:] for ln in lines[start:end] if ln.startswith("    ")]
    return "\n".join(block) + "\n"

CELL = r'''import glob, hashlib, json, os, platform, shutil, subprocess, sys, zipfile
from importlib.metadata import version
from pathlib import Path

BUNDLE_SHA = "__BUNDLE_SHA__"
KIT_SHA = __KIT_SHA__
HARD_STOP_S = __HARD_STOP__ * 60
OUT = Path("/kaggle/working")
REPORT = OUT / "e2e5_lambda_report.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


# 0. cheap controls: GPU, Python, ps, torch and numpy versions
import torch  # noqa: E402
import numpy  # noqa: E402
assert torch.cuda.is_available(), "no GPU: E2E-5 runs on Kaggle GPU T4 only"
assert shutil.which("ps"), "ps missing"
print("torch", torch.__version__, "numpy", numpy.__version__, "gpu", torch.cuda.get_device_name(0), flush=True)

# 1. bundle v6, checked then extracted
zip_path = one("/kaggle/input/**/stage_s_bundle_v6.zip.bin")  # .bin: Kaggle unzips .zip uploads
got = sha256(zip_path)
assert got == BUNDLE_SHA, ("bundle sha mismatch", got)
BUNDLE = OUT / "bundle"
shutil.rmtree(BUNDLE, ignore_errors=True)
BUNDLE.mkdir(parents=True)
zipfile.ZipFile(zip_path).extractall(BUNDLE)

# 2. E2E-5 kit overlaid on the babaorum tree of the bundle
kit_script = one("/kaggle/input/**/e2e5_lambda.py")
KIT = kit_script.parent.parent
kit_sha = {}
for rel, want in KIT_SHA.items():
    src = KIT / rel
    assert src.exists(), ("kit file missing", rel)
    kit_sha[rel] = sha256(src)
    assert kit_sha[rel] == want, ("kit sha mismatch", rel, kit_sha[rel])
    dst = BUNDLE / "babaorum" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
ROOT_E2E5 = BUNDLE / "babaorum"
PYTHONPATH = os.pathsep.join([str(ROOT_E2E5 / "src"), str(ROOT_E2E5 / "scripts")])

# 3. E2E-3 run 3 output: pages (not regenerated) and their report (sha reference)
mf_dirs = [Path(p) for p in glob.glob("/kaggle/input/**/e2e3/mf", recursive=True) if Path(p).is_dir()]
assert len(mf_dirs) == 1, ("e2e3/mf not found once", mf_dirs)
PAGES_TMP = mf_dirs[0].parent
RUN3 = one("/kaggle/input/**/e2e3_fp_classes_run3.json")

# 4. tests of the E2E-5 rules (cheap, before the pilot)
env = dict(os.environ, PYTHONPATH=PYTHONPATH)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_e2e5_lambda.py",
                "--deselect", "tests/test_e2e5_lambda.py::test_install_cell_of_the_builder_is_the_block_of_kaggle_install_txt",
                "--deselect", "tests/test_e2e5_lambda.py::test_notebook_first_code_cell_is_the_install_cell"],
               cwd=str(ROOT_E2E5), env=env, check=True, text=True)

# 5. manifest, then the pilot with a hard stop (report and summary only in /kaggle/working)
manifest = {"bundle_sha256": got, "kit_sha256": kit_sha, "run3_report_sha256": sha256(RUN3),
            "python": platform.python_version(), "torch": version("torch"), "numpy": version("numpy")}
(OUT / "e2e5_kaggle_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
try:
    proc = subprocess.run([sys.executable, "scripts/e2e5_lambda.py", "--confirmed-go", "--device", "cuda",
                           "--tmp", str(PAGES_TMP), "--run3-report", str(RUN3), "--out-dir", str(OUT)],
                          cwd=str(ROOT_E2E5), env=env, capture_output=True, text=True, timeout=HARD_STOP_S)
    print(proc.stdout.strip(), flush=True)
    rc = proc.returncode
except subprocess.TimeoutExpired:
    print(json.dumps({"status": "INCOMPLETE", "report_sha256": None, "step0_ratio": None,
                      "reason": "hard stop 38 min"}), flush=True)
    rc = "timeout"
(OUT / "e2e5_run_exit.json").write_text(json.dumps({"exit_code": rc}))
'''


def build(kit_sha: dict, bundle_sha: str) -> dict:
    code = (CELL.replace("__BUNDLE_SHA__", bundle_sha)
                .replace("__KIT_SHA__", json.dumps(kit_sha))
                .replace("__HARD_STOP__", str(HARD_STOP_MIN)))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source":
                 "# babaorum E2E-5 — B3 avec lambda = 1/6 sur le gradient CTC entrant dans enc_det (Kaggle GPU T4)\n"
                 "Lot : docs/END_TO_END_E2E5_LOT.txt (amendement 1 §6 et amendement 2 §7). Entrées : (1) bundle v6 ; "
                 "(2) kit E2E-5 ; (3) sortie E2E-3 run 3 (e2e3/mf et e2e3_fp_classes_run3.json, non régénérées). "
                 "Ne sont affichés que le statut, le sha du rapport et le rapport des normes du pas 0."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": install_cell()},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
            ]}


def main() -> None:
    bundle = BUNDLE_DRIVE / BUNDLE_NAME
    assert bundle.exists(), f"bundle v6 missing: {bundle}"
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    kit_sha = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in KIT_FILES}
    nb = build(kit_sha, bundle_sha)
    out = Path(__file__).with_name("kaggle_e2e5_lambda.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
