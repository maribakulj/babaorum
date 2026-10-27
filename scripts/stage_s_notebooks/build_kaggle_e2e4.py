"""Build kaggle_e2e4_extraction.ipynb (E2E-4 on Kaggle CPU, pC, lot docs/END_TO_END_E2E4_LOT.txt).

Thin wrapper on the model of build_kaggle_e2e3.py. Inputs: bundle v5 (sha ca41c092), the E2E-4 kit,
the E2E-1 output (checkpoints), and the E2E-3 run 3 output (the 60 pages e2e3/mf, NOT regenerated).
Order: cheap controls first (ps, Python, sources, checkpoints with a CPU load of B3, the tests), then the
run. Nothing is generated; the report goes to /kaggle/working.
"""
import hashlib
import json
from pathlib import Path

BUNDLE_SHA = "ca41c09272248c5096f2547fe738eee622db5af50414182537ac8e94e43367f5"
PINNED = "23875ab05425ce2aa07742bf314d02436c2f5642"
CKPT_PREFIX = {"B3": "738ca116bdfcef16", "B4": "85f96b00", "S2": "12fca335"}
KIT_FILES = ["scripts/e2e3_fp_classes.py", "scripts/g_diag_mf.py", "scripts/e2e4_extraction.py",
             "config/e2e4_extraction.json", "tests/test_e2e4_extraction.py"]
ROOT = Path(__file__).resolve().parents[2]

CELL = r'''import glob, hashlib, json, os, platform, re, shutil, subprocess, sys, zipfile
from importlib.metadata import version
from pathlib import Path

BUNDLE_SHA = "__BUNDLE_SHA__"
PINNED = "__PINNED__"
CKPT_PREFIX = __CKPT_PREFIX__
KIT_SHA = __KIT_SHA__
OUT = Path("/kaggle/working")
REPORT = OUT / "e2e4_extraction_report.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd))[:200], flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


# 0. cheap controls: Python 3.12, ps for the live-process guard, torch and numpy
assert sys.version_info >= (3, 12), ("needs Python >= 3.12", platform.python_version())
assert shutil.which("ps"), "ps missing: the live-process guard cannot run"
import numpy  # noqa: E402
import torch  # noqa: E402
print("torch", torch.__version__, "numpy", numpy.__version__, flush=True)

# 1. bundle v5, checked then extracted
zip_path = one("/kaggle/input/**/stage_s_bundle_v5.zip.bin")  # .bin: Kaggle unzips .zip uploads
got = sha256(zip_path)
assert got == BUNDLE_SHA, ("bundle sha mismatch", got)
BUNDLE = OUT / "bundle"
shutil.rmtree(BUNDLE, ignore_errors=True)
BUNDLE.mkdir(parents=True)
zipfile.ZipFile(zip_path).extractall(BUNDLE)

# 2. E2E-4 kit overlaid on the babaorum tree of the bundle
kit_script = one("/kaggle/input/**/e2e4_extraction.py")
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
ROOT_E2E4 = BUNDLE / "babaorum"

# 3. heritage-synth clone at the pinned commit (clean), dependencies only (no kraken)
CLONE = OUT / "heritage-synth"
shutil.rmtree(CLONE, ignore_errors=True)
run(["git", "clone", "-q", str(BUNDLE / "heritage-synth.gitbundle"), str(CLONE)])
run(["git", "-C", str(CLONE), "checkout", "-q", PINNED])
head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
assert head == PINNED, ("commit mismatch", head)
status = subprocess.run(["git", "-C", str(CLONE), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
assert status.strip() == "", "clone not clean"
block = re.search(r"dependencies = \[(.*?)\]", (CLONE / "pyproject.toml").read_text(), re.S).group(1)
run([sys.executable, "-m", "pip", "install", "--quiet", *re.findall(r'"([^"]+)"', block)])

# 4. E2E-1 checkpoints: located, sha prefix checked, B3 loaded on CPU
ckpt = {}
for arm, prefix in CKPT_PREFIX.items():
    ckpt[arm] = one(f"/kaggle/input/**/calibration/{arm}/checkpoint_last.pt")
    full = sha256(ckpt[arm])
    assert full.startswith(prefix), (arm, full[:16])
calib_dirs = {p.parent.parent for p in ckpt.values()}
assert len(calib_dirs) == 1, ("checkpoints from different folders", calib_dirs)
CALIB = calib_dirs.pop()
ck = torch.load(ckpt["B3"], map_location="cpu", weights_only=False)
assert "model" in ck, ("checkpoint without model key", list(ck)[:5])
del ck

# 5. E2E-3 run 3 pages: the folder that CONTAINS mf/ (mf/pages, mf/images); nothing is generated
mf_dirs = [Path(p) for p in glob.glob("/kaggle/input/**/e2e3/mf", recursive=True) if Path(p).is_dir()]
assert len(mf_dirs) == 1, ("e2e3/mf not found once", mf_dirs)
PAGES_TMP = mf_dirs[0].parent

# 6. tests of the E2E-4 rules (cheap, before the inference)
run([sys.executable, "-m", "pytest", "-q", "tests/test_e2e4_extraction.py"], cwd=str(ROOT_E2E4),
    env=dict(os.environ, PYTHONPATH=str(ROOT_E2E4 / "src")))

# 7. manifest and run (report written only in /kaggle/working)
manifest = {"bundle_sha256": got, "clone_commit": head, "kit_sha256": kit_sha,
            "checkpoint_sha256": {a: sha256(p) for a, p in ckpt.items()},
            "python": platform.python_version(), "torch": version("torch"), "numpy": version("numpy"),
            "pages_tmp": str(PAGES_TMP), "calibration_dir": str(CALIB)}
(OUT / "e2e4_kaggle_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
with open(OUT / "e2e4_run_stdout.log", "w") as log:
    proc = subprocess.Popen([sys.executable, "-u", "scripts/e2e4_extraction.py", "run",
                             "--clone", str(CLONE), "--confirmed-go",
                             "--tmp", str(PAGES_TMP), "--bundle-zip", str(zip_path),
                             "--calibration", str(CALIB), "--report", str(REPORT)],
                            cwd=str(ROOT_E2E4), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = proc.wait()
print("E2E-4 EXIT", rc, flush=True)
(OUT / "e2e4_run_exit.json").write_text(json.dumps({"exit_code": rc}))
if REPORT.exists():
    print("report sha256", sha256(REPORT), flush=True)
'''

def build(kit_sha: dict) -> dict:
    code = (CELL.replace("__BUNDLE_SHA__", BUNDLE_SHA)
                .replace("__PINNED__", PINNED)
                .replace("__CKPT_PREFIX__", json.dumps(CKPT_PREFIX))
                .replace("__KIT_SHA__", json.dumps(kit_sha)))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source":
                 "# babaorum E2E-4 — recoller les fragments (extraction + fusion, Kaggle CPU)\n"
                 "Lot : docs/END_TO_END_E2E4_LOT.txt (amendement 1 §7 inclus). Aucune génération, aucun GPU. "
                 "Entrées : (1) bundle v5 (sha ca41c092…) ; (2) kit E2E-4 ; (3) sortie E2E-1 (calibration/{B3,B4,S2}) ; "
                 "(4) sortie du carnet babaorum-e2e3-run3-fp-classes (e2e3/mf). "
                 "Sorties : /kaggle/working/e2e4_extraction_report.json, e2e4_run_stdout.log, e2e4_kaggle_manifest.json."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
            ]}


def main() -> None:
    kit_sha = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in KIT_FILES}
    nb = build(kit_sha)
    out = Path(__file__).with_name("kaggle_e2e4_extraction.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
