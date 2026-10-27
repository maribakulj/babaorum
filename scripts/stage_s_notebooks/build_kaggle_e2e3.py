"""Build kaggle_e2e3_fp_classes.ipynb (E2E-3 run 3 on Kaggle CPU, pC, after the GO of w2:p6, §13).

Thin wrapper, like build_kaggle_e2e1.py: no change to the bundle v5 nor to the pre-registration.
The notebook:
  1. finds stage_s_bundle_v5.zip.bin in /kaggle/input (sha ca41c092...), extracts it;
  2. finds the E2E-3 kit (scripts/e2e3_fp_classes.py, scripts/g_diag_mf.py, config, tests), checks its sha,
     and overlays it on the extracted babaorum tree;
  3. clones the pinned git bundle heritage-synth at 23875ab0 (clean tree) and installs its dependencies;
  4. checks ps, the E2E-1 checkpoints B3, B4, S2 (sha prefixes), torch and numpy, and loads B3 on CPU;
  5. runs the E2E-3 tests;
  6. only then generates the 60 pilot pages (seed 20261008) into /kaggle/working/e2e3/mf;
  7. runs e2e3_fp_classes.py run with the report in /kaggle/working/.
Each step raises on failure; the notebook never continues past a failed check.
"""
import hashlib
import json
from pathlib import Path

BUNDLE_SHA = "ca41c09272248c5096f2547fe738eee622db5af50414182537ac8e94e43367f5"
PINNED = "23875ab05425ce2aa07742bf314d02436c2f5642"
CKPT_PREFIX = {"B3": "738ca116bdfcef16", "B4": "85f96b00", "S2": "12fca335"}
KIT_SHA = {  # filled by the builder from the files on disk (see main below)
    "scripts/e2e3_fp_classes.py": None,
    "scripts/g_diag_mf.py": None,
    "config/e2e3_fp_classes.json": None,
    "tests/test_e2e3_fp_classes.py": None,
}
ROOT = Path(__file__).resolve().parents[2]

CELL = r'''import glob, hashlib, json, os, platform, re, shutil, subprocess, sys, zipfile
from importlib.metadata import version
from pathlib import Path

BUNDLE_SHA = "__BUNDLE_SHA__"
PINNED = "__PINNED__"
CKPT_PREFIX = __CKPT_PREFIX__
KIT_SHA = __KIT_SHA__
OUT = Path("/kaggle/working")
WORK = OUT / "e2e3"
REPORT = OUT / "e2e3_fp_classes_run3.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd))[:200], flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


# 0. Python 3.12 is required by mille-feuilles
assert sys.version_info >= (3, 12), ("mille-feuilles needs Python >= 3.12", platform.python_version())
# the live-process guard of the script needs ps; check it before the 60 pages are generated
assert shutil.which("ps"), "ps missing: the live-process guard cannot run"

# 1. bundle v5, checked then extracted
zip_path = one("/kaggle/input/**/stage_s_bundle_v5.zip.bin")  # .bin: Kaggle unzips .zip uploads
got = sha256(zip_path)
assert got == BUNDLE_SHA, ("bundle sha mismatch", got)
BUNDLE = OUT / "bundle"
shutil.rmtree(BUNDLE, ignore_errors=True)
BUNDLE.mkdir(parents=True)
zipfile.ZipFile(zip_path).extractall(BUNDLE)

# 2. E2E-3 kit overlaid on the babaorum tree of the bundle (scripts, config, tests)
kit_script = one("/kaggle/input/**/e2e3_fp_classes.py")
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
ROOT_E2E3 = BUNDLE / "babaorum"

# 3. heritage-synth from the pinned git bundle, clean tree, dependencies of its pyproject
CLONE = OUT / "heritage-synth"
shutil.rmtree(CLONE, ignore_errors=True)
run(["git", "clone", "-q", str(BUNDLE / "heritage-synth.gitbundle"), str(CLONE)])
run(["git", "-C", str(CLONE), "checkout", "-q", PINNED])
head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
assert head == PINNED, ("commit mismatch", head)
status = subprocess.run(["git", "-C", str(CLONE), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
assert status.strip() == "", "clone not clean"
block = re.search(r"dependencies = \[(.*?)\]", (CLONE / "pyproject.toml").read_text(), re.S).group(1)
deps = re.findall(r'"([^"]+)"', block)
run([sys.executable, "-m", "pip", "install", "--quiet", *deps])

# 5. E2E-1 checkpoints, located and checked by sha prefix (the full sha is recorded in the manifest)
ckpt = {}
for arm, prefix in CKPT_PREFIX.items():
    ckpt[arm] = one(f"/kaggle/input/**/calibration/{arm}/checkpoint_last.pt")
    full = sha256(ckpt[arm])
    assert full.startswith(prefix), (arm, full[:16])
calib_dirs = {p.parent.parent for p in ckpt.values()}
assert len(calib_dirs) == 1, ("checkpoints from different folders", calib_dirs)
CALIB = calib_dirs.pop()

import numpy
import torch
ck = torch.load(ckpt["B3"], map_location="cpu", weights_only=False)
assert "model" in ck, ("checkpoint without model key", list(ck)[:5])
del ck

# 6. tests of the E2E-3 rules (cheap, before generation)
run([sys.executable, "-m", "pytest", "-q", "tests/test_e2e3_fp_classes.py"], cwd=str(ROOT_E2E3),
    env=dict(os.environ, PYTHONPATH=str(ROOT_E2E3 / "src")))
# 4. the 60 pilot pages, generated only after every control above has passed
MF = WORK / "mf"
assert not MF.exists() or not any(MF.iterdir()), "mf not empty; refuse to overwrite"
WORK.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(CLONE / "src"))
gen = ("import sys; sys.argv=['mille-feuilles'] + sys.argv[1:]; "
       "from mille_feuilles.cli import main; raise SystemExit(main())")
run([sys.executable, "-c", gen, "generate", "--output", str(MF), "--pages", "60", "--seed", "20261008",
     "--width", "2680", "--height", "3698", "--dpi", "150", "--degradation", "mixed", "--jobs", "2"], env=env)
assert len(list((MF / "pages").glob("*.json"))) == 60 and len(list((MF / "images").glob("*.png"))) == 60

# 7. manifest and run (report written only in /kaggle/working)
manifest = {"bundle_sha256": got, "clone_commit": head, "kit_sha256": kit_sha,
            "checkpoint_sha256": {a: sha256(p) for a, p in ckpt.items()},
            "python": platform.python_version(), "torch": version("torch"), "numpy": version("numpy"),
            "calibration_dir": str(CALIB)}
(OUT / "e2e3_kaggle_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
with open(OUT / "e2e3_run3_stdout.log", "w") as log:
    proc = subprocess.Popen([sys.executable, "-u", "scripts/e2e3_fp_classes.py", "run",
                             "--clone", str(CLONE), "--confirmed-go",
                             "--tmp", str(WORK), "--bundle-zip", str(zip_path),
                             "--heritage-py", sys.executable, "--calibration", str(CALIB),
                             "--report", str(REPORT)],
                            cwd=str(ROOT_E2E3), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = proc.wait()
print("RUN 3 EXIT", rc, flush=True)
(OUT / "e2e3_run3_exit.json").write_text(json.dumps({"exit_code": rc}))
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
                 "# babaorum E2E-3 — run 3 sur Kaggle CPU (fp classes des checkpoints E2E-1)\n"
                 "Lot : docs/END_TO_END_E2E3_LOT.txt, §13 (run 3, Kaggle CPU, pas de GPU). "
                 "Entrées : (1) jeu contenant stage_s_bundle_v5.zip.bin (sha ca41c092…) ; "
                 "(2) jeu « e2e3-kit » (scripts, config, tests) ; (3) sortie E2E-1 contenant calibration/{B3,B4,S2}/checkpoint_last.pt. "
                 "Sorties : /kaggle/working/e2e3_fp_classes_run3.json, e2e3_run3_stdout.log, e2e3_kaggle_manifest.json. "
                 "Aucun GPU, Internet activé uniquement pour pip (dépendances heritage-synth uniquement ; kraken n'est pas installé)."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
            ]}


def main() -> None:
    kit_sha = {}
    for rel in KIT_SHA:
        kit_sha[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    nb = build(kit_sha)
    out = Path(__file__).with_name("kaggle_e2e3_fp_classes.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
