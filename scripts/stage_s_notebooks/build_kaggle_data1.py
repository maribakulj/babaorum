"""Build the three DATA-1 notebooks, one per shard (kaggle_data1_fragments_shard{0,1,2}.ipynb; pC).

Lot: docs/END_TO_END_DATA1_LOT.txt, amendment 1 (§5: --shard k/3, cap 480 min) and amendment 2 (§6: fusion removed,
no G, no e2e4_G_fixed.json). Thin wrapper on the model of build_kaggle_e2e4.py.
Inputs: bundle v5 (sha ca41c092, the bundle of E2E-4: the engine is unchanged), the E2E-4 kit (sha checked against
the frozen E2E-4 notebook) plus scripts/data1_fragments.py, its config and its test, the E2E-1 output (checkpoints
B3, S2), the E2E-4 v3 report (page sha reference), and the E2E-3 run 3 output (pages e2e3/mf, NOT regenerated).
Order in each notebook: the common install cell of KAGGLE_INSTALL.txt (first code cell), then the cheap controls,
the tests, the shard run (hard stop 510 min; the script's own cap is 480 min). Only the status and times are printed.
"""
import hashlib
import json
from pathlib import Path

BUNDLE_NAME = "stage_s_bundle_v5.zip"
BUNDLE_SHA = "ca41c09272248c5096f2547fe738eee622db5af50414182537ac8e94e43367f5"
BUNDLE_DRIVE = Path("/Users/marcel/Library/CloudStorage/GoogleDrive-mrebg@hotmail.fr/Mon Drive/babaorum/stage_s/bundle")
# frozen E2E-4 shas of the shared kit files (scripts/stage_s_notebooks/kaggle_e2e4_extraction.ipynb)
E2E4_KIT_SHA = {
    "scripts/e2e3_fp_classes.py": "fc96238085337b0cae2303fb717671a32ec0758048f9b252dd54c1e3a32f299b",
    "scripts/g_diag_mf.py": "aedd499722504468cb8c1764e174ae7137830b33b2043f3c8f47464d885d6f0a",
    "scripts/e2e4_extraction.py": "e874753604d7772709d13cbd997f8dcc61a5aad97cdf96055f908ccd5a24b386",
    "config/e2e4_extraction.json": "a2693cbc3447f5e66f76d3882cea3710b73dde6b08ff1ab15faac0cf464596e4",
}
KIT_FILES = ["scripts/e2e3_fp_classes.py", "scripts/g_diag_mf.py", "scripts/e2e4_extraction.py",
             "config/e2e4_extraction.json", "scripts/data1_fragments.py", "config/data1_fragments.json",
             "tests/test_data1_fragments.py"]
SHARDS = [0, 1, 2]
ROOT = Path(__file__).resolve().parents[2]
HARD_STOP_MIN = 510

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
SHARD = "__SHARD__"
HARD_STOP_S = __HARD_STOP__ * 60
OUT = Path("/kaggle/working")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


# 0. cheap controls: ps, torch and numpy versions (CPU run)
import torch  # noqa: E402
import numpy  # noqa: E402
assert shutil.which("ps"), "ps missing"
print("torch", torch.__version__, "numpy", numpy.__version__, "shard", SHARD, flush=True)

# 1. bundle v5, checked then extracted (the engine of E2E-4, unchanged)
zip_path = one("/kaggle/input/**/stage_s_bundle_v5.zip.bin")  # .bin: Kaggle unzips .zip uploads
got = sha256(zip_path)
assert got == BUNDLE_SHA, ("bundle sha mismatch", got)
BUNDLE = OUT / "bundle"
shutil.rmtree(BUNDLE, ignore_errors=True)
BUNDLE.mkdir(parents=True)
zipfile.ZipFile(zip_path).extractall(BUNDLE)

# 2. DATA-1 kit overlaid on the babaorum tree of the bundle; E2E-4 files must equal the frozen E2E-4 shas
kit_script = one("/kaggle/input/**/data1_fragments.py")
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
ROOT_D1 = BUNDLE / "babaorum"
PYTHONPATH = os.pathsep.join([str(ROOT_D1 / "src"), str(ROOT_D1 / "scripts")])

# 3. E2E-1 checkpoints (folder containing B3/ and S2/), E2E-4 v3 report (page sha reference), E2E-3 run 3 pages
ckpt = {arm: one(f"/kaggle/input/**/calibration/{arm}/checkpoint_last.pt") for arm in ("B3", "S2")}
assert sha256(ckpt["B3"]).startswith("738ca116bdfcef16") and sha256(ckpt["S2"]).startswith("12fca335")
CALIB = ckpt["B3"].parent.parent
assert CALIB == ckpt["S2"].parent.parent, "checkpoints from different folders"
E4_REPORT = one("/kaggle/input/**/e2e4_extraction_report.json")
assert sha256(E4_REPORT).startswith("714bc571"), "E2E-4 report sha differs from 714bc571"
mf_dirs = [Path(p) for p in glob.glob("/kaggle/input/**/e2e3/mf", recursive=True) if Path(p).is_dir()]
assert len(mf_dirs) == 1, ("e2e3/mf not found once", mf_dirs)
PAGES_TMP = mf_dirs[0].parent

# 4. tests of the DATA-1 rules (cheap, before the run); pytest is in the Kaggle image
env = dict(os.environ, PYTHONPATH=PYTHONPATH)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_data1_fragments.py",
                "--deselect", "tests/test_data1_fragments.py::test_install_cell_of_the_builder_is_the_block_of_kaggle_install_txt",
                "--deselect", "tests/test_data1_fragments.py::test_three_notebooks_each_with_its_shard_and_the_install_cell_first"],
               cwd=str(ROOT_D1), env=env, check=True, text=True)

# 5. manifest, then the shard run with a hard stop (outputs only in /kaggle/working)
manifest = {"bundle_sha256": got, "kit_sha256": kit_sha, "e2e4_report_sha256": sha256(E4_REPORT),
            "shard": SHARD, "python": platform.python_version(),
            "torch": version("torch"), "numpy": version("numpy")}
print(json.dumps(manifest, indent=2), flush=True)
try:
    proc = subprocess.run([sys.executable, "scripts/data1_fragments.py", "--confirmed-go",
                           "--tmp", str(PAGES_TMP), "--calibration", str(CALIB),
                           "--e2e4-report", str(E4_REPORT), "--shard", SHARD, "--out-dir", str(OUT)],
                          cwd=str(ROOT_D1), env=env, capture_output=True, text=True, timeout=HARD_STOP_S)
    print(proc.stdout.strip(), flush=True)
    rc = proc.returncode
except subprocess.TimeoutExpired:
    print(json.dumps({"status": "INCOMPLETE", "reason": "hard stop 510 min"}), flush=True)
    rc = "timeout"
(OUT / "data1_run_exit.json").write_text(json.dumps({"exit_code": rc}))
'''


def build(kit_sha: dict, shard: int) -> dict:
    code = (CELL.replace("__BUNDLE_SHA__", BUNDLE_SHA)
                .replace("__KIT_SHA__", json.dumps(kit_sha))
                .replace("__SHARD__", f"{shard}/3")
                .replace("__HARD_STOP__", str(HARD_STOP_MIN)))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source":
                 f"# babaorum DATA-1 — fragments B3 et S2 pour pR, morceau {shard} sur 3 (Kaggle CPU, données seulement)\n"
                 "Lot : docs/END_TO_END_DATA1_LOT.txt (amendements 1 et 2). Aucune métrique, aucune vérité, aucun GPU, "
                 "pas de fusion. "
                 "Entrées : (1) bundle v5 (sha ca41c092) ; (2) kit E2E-4 et fichiers DATA-1 ; (3) sortie E2E-1 "
                 "(calibration) ; (4) sortie E2E-4 v3 (e2e4_extraction_report.json, référence des sha) ; "
                 "(5) sortie du carnet babaorum-e2e3-run3-fp-classes (e2e3/mf). "
                 f"Sorties : /kaggle/working/fragments/<bras>/*.json du morceau {shard}, data1_manifest.json. "
                 "Seuls le statut et les temps sont affichés."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": install_cell()},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code},
            ]}


def main() -> None:
    for rel, want in E2E4_KIT_SHA.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == want, ("E2E-4 kit file changed since its freeze", rel, got)
    bundle = BUNDLE_DRIVE / BUNDLE_NAME
    assert bundle.exists(), f"bundle v5 missing: {bundle}"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == BUNDLE_SHA
    kit_sha = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in KIT_FILES}
    for k in SHARDS:
        out = Path(__file__).with_name(f"kaggle_data1_fragments_shard{k}.ipynb")
        out.write_text(json.dumps(build(kit_sha, k), indent=1, ensure_ascii=False))
        print(out)


if __name__ == "__main__":
    main()
