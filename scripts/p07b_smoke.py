"""LOT-P07b local smoke test of notebooks/kaggle_p07b.ipynb (E2E-4 lesson). The notebook is NOT modified.

Executes every code cell in one namespace after replacing /kaggle/input and /kaggle/working by local folders; reuses the Mac environments
through symlinks (so 'uv venv' never runs) and intercepts the 'pip install uv' call; the heavy children go through lourd.sh under /usr/bin/time
(compute seconds are separated from the queue wait); the session is cut to ONE page after the cell that reads the session list.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07b_smoke.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "runs/p07b_smoke"
INP, WORK = SMOKE / "input" / "babaorum-p07b", SMOKE / "working"
PARTS = ROOT / "runs/p07b_upload4"
LOURD = str(Path.home() / "outils-seg/lourd.sh")
NB = ROOT / "notebooks/kaggle_p07b.ipynb"
os.environ["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + os.environ["PATH"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def prepare():
    if SMOKE.exists():
        raise SystemExit(f"{SMOKE} exists: remove it first")
    INP.mkdir(parents=True)
    WORK.mkdir(parents=True)
    for p in sorted(PARTS.glob("part_*")):
        shutil.copy2(p, INP / p.name)
    os.system(f"chmod -R a-w '{SMOKE / 'input'}'")  # plays /kaggle/input: read only
    for name, target in (("layout-env", "~/outils-seg/layout-env"), ("kraken-env", "~/outils-seg/kraken-env"), ("axel-venv", "~/axel/.venv")):
        (WORK / name).symlink_to(Path(target).expanduser())


COMPUTE = []  # (label, real, user, sys)
_run = subprocess.run


def patched_run(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)) and len(cmd) >= 3 and list(cmd[1:4]) == ["-m", "pip", "install"] and "uv" in cmd:
        return subprocess.CompletedProcess(cmd, 0, "", "")  # no installation locally: the Mac environments are reused
    if isinstance(cmd, (list, tuple)) and str(cmd[0]).startswith((str(WORK / "layout-env"), str(WORK / "kraken-env"))):
        wrapped = [LOURD, "/usr/bin/time", "-p", *cmd]
        r = _run(wrapped, *a, **k)
        err = r.stderr or ""
        m = re.findall(r"^(real|user|sys)\s+([0-9.]+)\s*$", err, re.M)
        if len(m) >= 3:
            COMPUTE.append({"cmd": " ".join(str(x) for x in cmd[1:3])[-90:], **{k_: float(v) for k_, v in m[-3:]}})
            err = re.sub(r"^(real|user|sys)\s+[0-9.]+\s*$\n?", "", err, flags=re.M)
        return subprocess.CompletedProcess(cmd, r.returncode, r.stdout, err)
    return _run(cmd, *a, **k)


def main() -> int:
    runs_listing_cmd = "cd ~/axel && find runs -type f -exec stat -f '%m %z %N' {} + | sort -k3"
    before = subprocess.run(runs_listing_cmd, shell=True, capture_output=True, text=True).stdout
    prepare()
    subprocess.run = patched_run
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
            if "pages_all = json.loads" in src and "pages_all" in ns:
                full = len(ns["pages_all"])
                ns["pages_all"] = ns["pages_all"][:1]  # smoke session: control page + ONE page of session 1
                print(f"[smoke] session list cut from {full} to 1 page: {ns['pages_all'][0]['page']}", flush=True)
            results.append({"cell": i, "status": "ok", "seconds_wall": round(time.time() - t0)})
        except BaseException as exc:  # noqa: BLE001
            results.append({"cell": i, "status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:600]}", "seconds_wall": round(time.time() - t0)})
            traceback.print_exc()
            break
        print(f"[smoke] cell {i} ok ({time.time() - t0:.0f} s wall)", flush=True)
    subprocess.run = _run
    after = subprocess.run(runs_listing_cmd, shell=True, capture_output=True, text=True).stdout
    report = {"cells": results, "wall_seconds_total": round(time.time() - t_all), "compute_seconds_children": {
        "real": round(sum(c["real"] for c in COMPUTE)), "user": round(sum(c["user"] for c in COMPUTE)), "sys": round(sum(c["sys"] for c in COMPUTE)),
        "n_children": len(COMPUTE)}, "children": COMPUTE, "axel_runs_listing_identical": before == after, "axel_runs_files": len(before.splitlines()),
        "portability_report": json.loads((WORK / "portability_report.json").read_text()) if (WORK / "portability_report.json").exists() else None,
        "session_report": json.loads(next(WORK.glob("p07b_session*_report.json")).read_text()) if list(WORK.glob("p07b_session*_report.json")) else None}
    (ROOT / "reports/p07b_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("wall_seconds_total", "compute_seconds_children", "axel_runs_listing_identical")}, indent=1))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
