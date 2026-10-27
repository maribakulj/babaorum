"""Build kaggle_stage_s_calibration.ipynb (E2E-1 on Kaggle T4, p6, 15/10).

A thin wrapper around the UNCHANGED bundle v5 driver (babaorum/scripts/colab_stage_s_calibration.py):
the driver's Colab paths are recreated on the Kaggle VM; /content/drive/MyDrive/babaorum/stage_s/
calibration is a symlink to /kaggle/working/calibration, so the logs and the report land in the
Kaggle output. No Drive mount, no change to the bundle or the pre-registration.
"""
import json
from pathlib import Path

BUNDLE_SHA = "ca41c09272248c5096f2547fe738eee622db5af50414182537ac8e94e43367f5"
CELL = r'''import glob, hashlib, json, os, platform, shutil, subprocess, sys, zipfile
from pathlib import Path
BUNDLE_SHA = "%s"
zips = glob.glob("/kaggle/input/**/stage_s_bundle_v5.zip.bin", recursive=True)  # .bin: Kaggle unzips .zip uploads
assert len(zips) == 1, zips
z = Path(zips[0])
got = hashlib.sha256(z.read_bytes()).hexdigest()
assert got == BUNDLE_SHA, ("bundle sha mismatch", got)
dest = Path("/content/bundle")
shutil.rmtree(dest, ignore_errors=True)
dest.mkdir(parents=True)
zipfile.ZipFile(z).extractall(dest)
out = Path("/kaggle/working/calibration"); out.mkdir(parents=True, exist_ok=True)
link = Path("/content/drive/MyDrive/babaorum/stage_s/calibration")
link.parent.mkdir(parents=True, exist_ok=True)
if not link.exists():
    link.symlink_to(out, target_is_directory=True)
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
platform_info = {"platform": "Kaggle", "gpu": smi, "python": platform.python_version(),
                 "bundle_sha256": got, "wrapper": "kaggle_stage_s_calibration.ipynb (p6, E2E-1)",
                 "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
(out / "kaggle_platform.json").write_text(json.dumps(platform_info, indent=2))
print(platform_info, flush=True)
script = dest / "babaorum/scripts/colab_stage_s_calibration.py"
with open(out / "driver_stdout.log", "w") as log:
    proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=str(dest),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True); log.write(line); log.flush()
    rc = proc.wait()
print("DRIVER EXIT", rc, flush=True)
(out / "driver_exit.json").write_text(json.dumps({"exit_code": rc}))
''' % BUNDLE_SHA

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                 "# babaorum E2E-1 — calibration GPU de l'étape S (Kaggle T4)\n"
                 "Pré-enregistrement : config/stage_s_gpu_calibration.json (63ebbc86…) ; bundle v5 inchangé. "
                 "Entrée : jeu de données privé contenant stage_s_bundle_v5.zip.bin (même octets que le zip v5 ; extension .bin car Kaggle dézippe les .zip). Accélérateur T4, Internet activé."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CELL}]}
out = Path(__file__).with_name("kaggle_stage_s_calibration.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out)
