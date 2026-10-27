#!/usr/bin/env python3
"""Colab driver for the stage S GPU calibration (lot S-L6a). Code only; no pages are shipped.

Run on the Colab VM from the bundle directory. Layout expected inside the bundle:
  babaorum/     src/babaorum (closure modules), scripts/, config/
  heritage-synth/  pyproject.toml, src/, schemas/, assets/, profiles/, runs/pilot-v0.2/config.json

Steps (pre-registered in config/stage_s_gpu_calibration.json):
  a) install mille-feuilles dependencies read from its pyproject (no uv);
  b) generate 60 pages with the pilot command and profile, seed 20261008, into /content/mf;
  c) quick check: 60 pages, images present, sizes match the JSON;
  d) a 2-step CUDA preflight (added safety, not a measurement), then arms B3, B4, S2 in
     that order, each --steps 2000 --time-cap-min 35 --eval-every 500, logs on Drive;
  e) final JSON report with GPU name, peak memory, seconds per step and bundle hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2]
BABAORUM = BUNDLE / "babaorum"
HERITAGE = Path("/content/heritage-synth")  # clone of the pinned git bundle (step 0)
GITBUNDLE = BUNDLE / "heritage-synth.gitbundle"
PINNED_COMMIT = "23875ab05425ce2aa07742bf314d02436c2f5642"
PRE = json.loads((BABAORUM / "config/stage_s_gpu_calibration.json").read_text())

MF_OUT = Path("/content/mf")
PREFLIGHT_OUT = Path("/content/preflight")
DRIVE = Path("/content/drive/MyDrive/babaorum/stage_s/calibration")
TRAIN_IDS = [f"mf_{i:04d}" for i in range(0, 50)]
EVAL_IDS = [f"mf_{i:04d}" for i in range(50, 60)]
ARMS = ["B3", "B4", "S2"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceMismatch(RuntimeError):
    """The mille-feuilles clone is not the pinned commit, or its working tree is not clean."""


def step_0_source(log):
    """Clone the pinned git bundle into /content/heritage-synth and check commit and clean status."""
    if not HERITAGE.exists():
        run(["git", "clone", "-q", str(GITBUNDLE), str(HERITAGE)], log=log)
    run(["git", "-C", str(HERITAGE), "checkout", "-q", PINNED_COMMIT], log=log)
    head = run(["git", "-C", str(HERITAGE), "rev-parse", "HEAD"], log=log).strip()
    status = run(["git", "-C", str(HERITAGE), "status", "--porcelain"], log=log)
    if head != PINNED_COMMIT:
        raise SourceMismatch(f"HEAD {head} != pinned {PINNED_COMMIT}")
    if status.strip():
        raise SourceMismatch(f"working tree not clean: {status.strip()[:300]}")
    return {"head": head, "clean": True, "pinned": PINNED_COMMIT}


class WriteRefused(RuntimeError):
    """A VM or Drive write failed, or free space is below the declared floor."""


WRITE_FAILURE_PATTERN = re.compile(
    r"No space left|Errno 28|Read-only|Errno 30|Input/output error|Errno 5\b|Permission denied|Errno 13")


def check_writable(folder: Path, min_free_gib: float = 1.0) -> dict:
    """Create the folder, check free space, write and read back a probe file, then remove it."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        free_gib = shutil.disk_usage(folder).free / 2 ** 30
        if free_gib < min_free_gib:
            raise WriteRefused(f"{folder}: free space {free_gib:.2f} GiB < {min_free_gib} GiB")
        probe = folder / ".write_probe"
        payload = os.urandom(4096)
        probe.write_bytes(payload)
        if probe.read_bytes() != payload:
            raise WriteRefused(f"{folder}: probe read-back mismatch")
        probe.unlink()
        return {"folder": str(folder), "free_gib": round(free_gib, 2), "probe": "ok"}
    except WriteRefused:
        raise
    except OSError as exc:
        raise WriteRefused(f"{folder}: write failed: {exc}") from exc


def classify_failure(message: str) -> bool:
    """True when a failed subprocess reports a storage error rather than a method error."""
    return bool(WRITE_FAILURE_PATTERN.search(message))


def load_bundle_manifest() -> dict:
    manifest = BUNDLE / "bundle_manifest.json"
    if not manifest.exists():
        raise RuntimeError("bundle_manifest.json missing: rebuild the bundle with build_stage_s_bundle.py")
    return json.loads(manifest.read_text())


def verify_mille_feuilles(manifest: dict, heritage_root: Path = None, bundle_root: Path = None) -> dict:
    """sha256 of every mille-feuilles file in the clone, compared with the bundle manifest.
    Roots default to the Colab layout (/content/heritage-synth and the bundle folder)."""
    heritage_root = Path(heritage_root) if heritage_root is not None else HERITAGE
    bundle_root = Path(bundle_root) if bundle_root is not None else BUNDLE
    expected = manifest["mille_feuilles_sha256"]
    observed = {}
    for rel in expected:
        if rel.startswith("heritage-synth/"):
            observed[rel] = sha256(heritage_root / rel[len("heritage-synth/"):])
        else:
            observed[rel] = sha256(bundle_root / rel)
    mismatch = [rel for rel in expected if observed[rel] != expected[rel]]
    if mismatch:
        raise RuntimeError(f"mille-feuilles files differ from the bundle manifest: {mismatch[:5]}")
    return {"files": len(observed), "matches_manifest": True, "sha256": observed}


class ArmKilled(RuntimeError):
    """The arm subprocess exceeded its hard timeout (cap + kill margin) and was killed."""


def run(cmd, *, env=None, cwd=None, log=None, timeout=None):
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        seconds = time.monotonic() - t0
        if log is not None:
            log.append({"cmd": " ".join(map(str, cmd))[:300], "killed_after_s": round(seconds, 1),
                        "timeout_s": timeout})
        raise ArmKilled(f"killed at hard timeout {timeout} s: {' '.join(map(str, cmd))[:200]}") from exc
    seconds = time.monotonic() - t0
    if log is not None:
        log.append({"cmd": " ".join(map(str, cmd))[:300], "returncode": proc.returncode,
                    "seconds": round(seconds, 1), "stdout_tail": proc.stdout[-800:],
                    "stderr_tail": proc.stderr[-800:]})
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))[:200]}\n"
                           f"{proc.stderr[-1500:]}")
    return proc.stdout


def step_a_install(log):
    if sys.version_info < (3, 12):
        raise RuntimeError(f"mille-feuilles needs Python >= 3.12; this VM has {platform.python_version()}")
    pyproject = (HERITAGE / "pyproject.toml").read_text()
    block = re.search(r"dependencies = \[(.*?)\]", pyproject, re.S).group(1)
    deps = re.findall(r'"([^"]+)"', block)
    run([sys.executable, "-m", "pip", "install", "--quiet", *deps], log=log)
    return deps


def mf_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERITAGE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def mille_feuilles(args_list, log):
    code = ("import sys; sys.argv=['mille-feuilles'] + sys.argv[1:]; "
            "from mille_feuilles.cli import main; raise SystemExit(main())")
    return run([sys.executable, "-c", code, *args_list], env=mf_env(), log=log)


def step_b_generate(log):
    if MF_OUT.exists() and any(MF_OUT.iterdir()):
        raise RuntimeError(f"{MF_OUT} is not empty; refuse to overwrite")
    mille_feuilles(["--help"], log)
    mille_feuilles(["generate", "--help"], log)
    mille_feuilles(["generate", "--output", str(MF_OUT), "--pages", "60", "--seed", "20261008",
                    "--width", "2680", "--height", "3698", "--dpi", "150",
                    "--degradation", "mixed", "--jobs", "2"], log)


def step_c_check(log):
    import numpy as np
    from PIL import Image
    pages = sorted((MF_OUT / "pages").glob("*.json"))
    images = sorted((MF_OUT / "images").glob("*.png"))
    if len(pages) != 60 or len(images) != 60:
        raise RuntimeError(f"expected 60 pages and 60 images, found {len(pages)} / {len(images)}")
    ids = {p.stem for p in pages}
    missing = [i for i in TRAIN_IDS + EVAL_IDS if i not in ids]
    if missing:
        raise RuntimeError(f"missing expected ids: {missing[:5]}")
    checked = 0
    for pid in TRAIN_IDS[:3] + EVAL_IDS[:3]:
        meta = json.loads((MF_OUT / "pages" / f"{pid}.json").read_text())["image"]
        with Image.open(MF_OUT / "images" / f"{pid}.png") as im:
            if (im.size[0], im.size[1]) != (meta["width"], meta["height"]):
                raise RuntimeError(f"{pid}: image size {im.size} != JSON {meta['width']}x{meta['height']}")
        checked += 1
    return {"pages": len(pages), "images": len(images), "size_checked": checked,
            "np": np.__version__}


def arm_command(arm, steps_out, log_dir, pages_root, steps, cap, eval_every, train, evals):
    return [sys.executable, str(BABAORUM / "scripts/train_stage_s_v2.py"), "--arm", arm,
            "--output", str(steps_out), "--device", "cuda", "--steps", str(steps),
            "--time-cap-min", str(cap), "--eval-every", str(eval_every),
            "--train-pages", *train, "--eval-pages", *evals,
            "--root", str(pages_root), "--log-dir", str(log_dir),
            "--config", str(BABAORUM / "config/stage_s_smoke_v2.json")]


def babaorum_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BABAORUM / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def guard_parameters():
    """Cap per arm and the global guard, read from the pre-registration (not hard-coded)."""
    schedule = PRE["schedule"]
    guard = schedule["global_guard"]
    margin = float(re.search(r"remaining_of_\d+ - (\d+(?:\.\d+)?)", guard).group(1))
    min_cap = float(re.search(r"cap < (\d+(?:\.\d+)?)", guard).group(1))
    kill_margin = float(re.search(r"cap \+ (\d+(?:\.\d+)?) min", schedule["cap_scope"]).group(1))
    return {"cap_max_min": float(schedule["time_cap_per_arm_minutes"]), "margin_min": margin,
            "min_cap_min": min_cap, "total_min": float(PRE["max_wall_clock_hours"]) * 60.0,
            "kill_margin_min": kill_margin}


def arm_cap_minutes(elapsed_min: float, params: dict):
    """cap = min(cap_max, remaining - margin); None (arm skipped) when cap < min_cap."""
    remaining = params["total_min"] - elapsed_min
    cap = min(params["cap_max_min"], remaining - params["margin_min"])
    if cap < params["min_cap_min"]:
        return None, round(cap, 2)
    return round(cap, 2), round(cap, 2)


def step_d_arms(log, started, clock=time.monotonic, runner=None):
    runner = runner or run
    params = guard_parameters()
    check_writable(PREFLIGHT_OUT.parent)
    kill_s = params["kill_margin_min"] * 60.0
    preflight = runner(arm_command("B4", PREFLIGHT_OUT / "report.json", PREFLIGHT_OUT, MF_OUT,
                                   2, 5, 0, TRAIN_IDS[:2], EVAL_IDS[:1]), env=babaorum_env(), log=log,
                       timeout=5 * 60.0 + kill_s)
    results = {"guard": params,
               "preflight": json.loads(preflight.strip().splitlines()[-1]) if preflight.strip() else None}
    reports = {}
    for arm in ARMS:
        elapsed_min = (clock() - started) / 60.0
        cap, cap_raw = arm_cap_minutes(elapsed_min, params)
        if cap is None:
            results[arm] = {"status": "SKIPPED_GLOBAL_GUARD", "elapsed_min": round(elapsed_min, 2),
                            "cap_raw_min": cap_raw, "min_cap_min": params["min_cap_min"]}
            continue
        out = DRIVE / arm / "report.json"
        results.setdefault("write_checks", {})[arm] = check_writable(DRIVE / arm)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            runner(arm_command(arm, out, DRIVE / arm, MF_OUT, PRE["schedule"]["steps_per_arm"],
                               cap, PRE["schedule"]["eval_every_steps"], TRAIN_IDS, EVAL_IDS),
                   env=babaorum_env(), log=log, timeout=(cap + kill_s / 60.0) * 60.0)
        except ArmKilled as exc:
            # hard kill: no report JSON from the arm; journal and checkpoints on Drive remain
            results[arm] = {"status": "KILLED_HARD_TIMEOUT", "elapsed_before_min": round(elapsed_min, 2),
                            "cap_min": cap, "error": str(exc)[-300:]}
            continue
        reports[arm] = json.loads(out.read_text())
        results[arm] = {"status": "RAN", "elapsed_before_min": round(elapsed_min, 2), "cap_min": cap}
    return results, reports


def bundle_hashes():
    """sha256 of the babaorum bundle files and of the pinned mille-feuilles files in the clone."""
    out = {}
    for path in sorted(BABAORUM.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            out[str(path.relative_to(BUNDLE))] = sha256(path)
    out["heritage-synth/pyproject.toml (clone)"] = sha256(HERITAGE / "pyproject.toml")
    out["heritage-synth/runs/pilot-v0.2/config.json (bundle)"] = sha256(BUNDLE / "heritage-synth/runs/pilot-v0.2/config.json")
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


EVAL_PAGES_REQUIRED = [f"mf_{i:04d}" for i in range(50, 60)]


def launch_criterion(reports) -> dict:
    """Pre-registered gate: B3 or B4 only (S2 does not count); mean over the 10 eval pages of
    recall and precision (after refinement) at the last completed evaluation; a page with no
    prediction counts as precision 0; both means must reach 0.5.
    All 10 pages mf_0050..mf_0059 must be present in the retained evaluation; otherwise the arm
    is CRITERION_NOT_EVALUABLE, and so is the gate (never PASS)."""
    out = {}
    for arm in ("B3", "B4"):
        rep = reports.get(arm)
        if rep is None:
            out[arm] = {"status": "NOT_RUN"}
            continue
        done = [t for t in rep["timeline"] if t.get("evals")]
        if not done:
            out[arm] = {"status": "CRITERION_NOT_EVALUABLE", "reason": "no completed evaluation"}
            continue
        last = done[-1]
        present = {e.get("page_id") for e in last["evals"]}
        missing = [pid for pid in EVAL_PAGES_REQUIRED if pid not in present]
        if missing:
            out[arm] = {"status": "CRITERION_NOT_EVALUABLE", "step": last["step"],
                        "missing_pages": missing, "pages": len(present)}
            continue
        recalls, precisions = [], []
        for e in last["evals"]:
            if e.get("page_id") not in EVAL_PAGES_REQUIRED:
                continue
            m = e["metrics"]["after"]
            recalls.append(float(m["recall"]) if m["recall"] is not None else 0.0)
            precisions.append(0.0 if not m.get("n_pred") else
                              (float(m["precision"]) if m["precision"] is not None else 0.0))
        mr, mp = sum(recalls) / len(recalls), sum(precisions) / len(precisions)
        out[arm] = {"status": "EVALUATED", "step": last["step"], "pages": len(recalls),
                    "mean_recall_after": mr, "mean_precision_after": mp,
                    "meets_criterion": mr >= 0.5 and mp >= 0.5, "final_eval": "final" in last}
    if any(out[a].get("meets_criterion") for a in ("B3", "B4")):
        gate = "PASS"
    elif any(out[a].get("status") in ("CRITERION_NOT_EVALUABLE", "NOT_RUN") for a in ("B3", "B4")):
        gate = "CRITERION_NOT_EVALUABLE"
    else:
        gate = "FAIL_STAGE_S_NOT_LAUNCHED"
    return {"arms": out, "launch_gate": gate}


def summarize(reports):
    summary = {}
    for arm, rep in reports.items():
        steps = rep["steps"]
        summary[arm] = {
            "status": rep["status"], "steps_completed": rep["steps_completed"],
            "seconds_per_step_median_100_to_end": rep["seconds_per_step_median_100_to_end"],
            "gpu_peak_mb": rep["gpu_peak_mb"], "elapsed_seconds": rep["elapsed_seconds"],
            "L_geo_first_last": [steps[0]["L_geo"], steps[-1]["L_geo"]] if steps else None,
            "timeline": [{"step": t["step"],
                          "before_recall_mean": _mean([e["metrics"]["before"]["recall"] for e in t["evals"]]),
                          "after_recall_mean": _mean([e["metrics"]["after"]["recall"] for e in t["evals"]]),
                          "after_precision_mean": _mean([e["metrics"]["after"]["precision"]
                                                         for e in t["evals"]
                                                         if e["metrics"]["after"]["precision"] is not None]),
                          "after_mean_normalized_distance": _mean(
                              [e["metrics"]["after"]["mean_normalized_distance"] for e in t["evals"]
                               if e["metrics"]["after"]["mean_normalized_distance"] is not None])}
                         for t in rep["timeline"]],
        }
    return summary


def write_report(report: dict) -> None:
    text = json.dumps(report, indent=1, allow_nan=False) + "\n"
    try:
        DRIVE.mkdir(parents=True, exist_ok=True)
        (DRIVE / "calibration_report.json").write_text(text)
    except OSError as exc:  # the Drive itself may be the failure: keep the report on stdout
        report["status"] = "STOPPED_WRITE_REFUSED" if report.get("status") == "COMPLETED" else report.get("status")
        report["error"] = f"final report not written to Drive: {exc}"
        print(json.dumps(report, indent=1, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-generate", action="store_true", help="reuse /content/mf if already checked")
    args = parser.parse_args(argv)
    started = time.monotonic()
    log = []
    report = {"schema_version": "stage_s_gpu_calibration_report_v1",
              "pre_registration": "config/stage_s_gpu_calibration.json",
              "account": PRE["account"], "python": platform.python_version()}
    try:
        report["source"] = step_0_source(log)
        manifest = load_bundle_manifest()
        report["source_identity"] = {"heritage_synth_git_commit": manifest["heritage_synth_git_commit"],
                                     "heritage_synth_git_clean": manifest["heritage_synth_git_clean"],
                                     "bundle_built_utc": manifest.get("built_utc")}
        report["mille_feuilles_sha256"] = verify_mille_feuilles(manifest)
        report["write_checks"] = {"vm": check_writable(MF_OUT.parent), "drive": check_writable(DRIVE)}
        report["dependencies"] = step_a_install(log)
        if not args.skip_generate:
            step_b_generate(log)
        report["data_check"] = step_c_check(log)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device: pick a GPU runtime (or the Kaggle T4 fallback)")
        report["gpu"] = {"name": torch.cuda.get_device_name(0), "torch": torch.__version__}
        arms_meta, reports = step_d_arms(log, started)
        report["arms"] = arms_meta
        report["results"] = summarize(reports)
        report["launch_criterion"] = launch_criterion(reports)
        report["gpu_peak_mb_max"] = max((r["gpu_peak_mb"] or 0.0) for r in reports.values()) if reports else None
        report["status"] = "COMPLETED"
    except SourceMismatch as exc:
        report["status"] = "STOPPED_SOURCE_MISMATCH"
        report["error"] = str(exc)[-2000:]
    except WriteRefused as exc:
        report["status"] = "STOPPED_WRITE_REFUSED"
        report["error"] = str(exc)[-2000:]
    except Exception as exc:  # report the failure instead of hiding it
        report["status"] = "STOPPED_WRITE_REFUSED" if classify_failure(str(exc)) else "FAILED"
        report["error"] = str(exc)[-2000:]
    finally:
        report["elapsed_minutes"] = round((time.monotonic() - started) / 60.0, 1)
        report["commands"] = log
        try:
            report["bundle_sha256"] = bundle_hashes()
        except Exception as exc:  # hashing must never prevent the report from being written
            report["bundle_sha256"] = None
            report["bundle_hash_error"] = f"{type(exc).__name__}: {exc}"[-500:]
        write_report(report)
        print(json.dumps({"status": report["status"], "elapsed_minutes": report["elapsed_minutes"]}))


if __name__ == "__main__":
    main()
