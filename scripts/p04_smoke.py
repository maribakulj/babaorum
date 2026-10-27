"""LOT-P04 local smoke of notebooks/kaggle_p04_blla.ipynb: 1 epoch on 2 training zones, CPU, inputs read only. The notebook is NOT modified.

Replaces /kaggle/input and /kaggle/working by local folders, sets P04_SMOKE=1, reuses the Mac kraken environment (no installation), HOME in the scratch
folder (stray writes land there), the heavy child goes through lourd.sh. Nothing is written under the dataset (copied read only) nor under ~/axel/runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "runs/p04_smoke"
UPLOAD = ROOT / "runs/p04_upload"   # the 9 MB pieces of the archive = the Kaggle input (v2 rule: the archive, never the repository)
IN, WORK, HOME = SMOKE / "input", SMOKE / "working", SMOKE / "home"
LOURD = str(Path.home() / "outils-seg/lourd.sh")
KR = Path.home() / "outils-seg/kraken-env"
NB = ROOT / "notebooks/kaggle_p04_blla.ipynb"
LISTING = "cd ~/axel && find runs -type f -exec stat -f '%m %z %N' {} + | sort -k3"
_run, _popen = subprocess.run, subprocess.Popen


VENV = os.environ.get("P04_VENV")   # fresh Kaggle-like venv: pip installs are REAL there (what the carnet installs), no Mac environment reused


def pip_noop(cmd, *a, **k):
    if VENV is None and isinstance(cmd, (list, tuple)) and "pip" in cmd and "install" in cmd:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return _run(cmd, *a, **k)


def popen_lourd(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)) and str(cmd[0]) == "ketos" and VENV is None:
        cmd = [LOURD, "/usr/bin/time", "-p", *cmd]
    return _popen(cmd, *a, **k)


def main() -> int:
    if SMOKE.exists():
        raise SystemExit(f"{SMOKE} exists: remove it first")
    before = _run(LISTING, shell=True, capture_output=True, text=True).stdout
    IN.mkdir(parents=True)
    for part in sorted(UPLOAD.glob("part_??")):
        shutil.copy2(part, IN / part.name)
    WORK.mkdir(parents=True); HOME.mkdir()
    os.system(f"chmod -R a-w '{SMOKE / 'input'}'")          # plays /kaggle/input: read only
    bindir = Path(VENV) / "bin" if VENV else KR / "bin"
    os.environ.update({"P04_SMOKE": "1", "HOME": str(HOME), "PATH": f"{bindir}{os.pathsep}{Path.home() / '.local/bin'}{os.pathsep}" + os.environ["PATH"],
                       "PYTHONDONTWRITEBYTECODE": "1"})
    subprocess.run, subprocess.Popen = pip_noop, popen_lourd
    nb = json.loads(NB.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__smoke__"}
    results, t_all = [], time.time()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]).replace("/kaggle/input", str(SMOKE / "input")).replace("/kaggle/working", str(WORK))
        t0 = time.time()
        try:
            exec(compile(src, f"cell{i}", "exec"), ns)
            results.append({"cell": i, "status": "ok", "seconds": round(time.time() - t0)})
            print(f"[smoke] cell {i} ok ({time.time() - t0:.0f} s)", flush=True)
        except BaseException as exc:  # noqa: BLE001
            results.append({"cell": i, "status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            traceback.print_exc()
            break
    subprocess.run, subprocess.Popen = _run, _popen
    after = _run(LISTING, shell=True, capture_output=True, text=True).stdout
    strays = [str(p.relative_to(HOME)) for p in HOME.rglob("*") if p.is_file()][:20]
    report = {"cells": results, "wall_seconds": round(time.time() - t_all), "axel_runs_listing_identical": before == after, "axel_runs_listing_diff": sorted(set(before.splitlines()) ^ set(after.splitlines()))[:20],
              "input_still_read_only_and_intact": _run(["diff", "-rq", str(UPLOAD), str(IN)], capture_output=True, text=True).stdout == "",
              "files_written_under_HOME_scratch": strays, "p04_report": json.loads((WORK / "p04_report.json").read_text()) if (WORK / "p04_report.json").exists() else None}
    (ROOT / "reports/p04_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("wall_seconds", "axel_runs_listing_identical", "input_still_read_only_and_intact")}, indent=1))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
