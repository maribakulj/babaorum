"""DATA-2 common code: functions COPIED from scripts/ld0_zone_coverage.py (LD-0, frozen at gel 6).

Provenance (§11 (4) of docs/END_TO_END_DATA2_LOT.txt): source scripts/ld0_zone_coverage.py, sha256 4fae48f8616a6177819bfb2a6a636d140117681c536407216d92076a4da05913.
The copied text is verbatim, except for the three names listed here, which are changed for DATA-2:
  - CONFIG points to config/data2_pages.json (same keys: pages.list, timeouts_s, discovery.relpaths);
  - execute_pipeline writes data2_run_exit.json (not ld0_run_exit.json);
  - the import of faulthandler is added (execute_pipeline uses it; it was missing in the first copy, gel 1 B1).
  - discover_inputs takes one optional argument, relpaths (default: the config list), so that each carnet
    gives its own white list; nothing else in that function changed.
  - heartbeat_errors.txt and the other file names are unchanged.
The LD-0 script is not modified.
"""
from __future__ import annotations

import datetime
import faulthandler
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "data2_pages.json"
FORBIDDEN_SUBPATHS = ("e2e3/mf/pages", "e2e3/mf/images")

def _child_dirs(d: Path) -> list:
    """Sub-directories of d, listed by name. §14 (12): never listed: a directory that is or contains e2e3/mf (checked
    with is_dir on an explicit path, not by listing) and anything under e2e3/mf."""
    if d.name == "mf" and d.parent.name == "e2e3":
        return []
    if d.name.startswith("mf"):              # page-like folders (mf_0050 decoys included): never listed
        return []
    if (d / "e2e3" / "mf").is_dir():
        return []
    return sorted(c for c in d.iterdir() if c.is_dir())


def discover_inputs(input_root: Path = Path("/kaggle/input"), relpaths: dict = None) -> dict:
    """§13 (1) and §14 (12): lists /kaggle/input and its directories of levels 1, 2 and 3 (names only, never inside
    e2e3/mf), then checks the EXPLICIT relative paths of the white list in the config under each root: each must be
    found exactly once. Returns the found paths and the roots used, as names relative to input_root."""
    cfg = json.loads(CONFIG.read_text())
    relpaths = relpaths if relpaths is not None else cfg["discovery"]["relpaths"]
    for name, rel in relpaths.items():
        if rel.startswith("/") or any(bad in rel for bad in FORBIDDEN_SUBPATHS):
            raise ValueError(f"white list entry refused: {name} = {rel}")
    roots = [input_root]
    level1 = _child_dirs(input_root)
    level2 = []
    for d in level1:
        level2 += _child_dirs(d)
    level3 = []
    for d in level2:
        level3 += _child_dirs(d)
    roots += level1 + level2 + level3
    roots = list(dict.fromkeys(roots))
    found, root_of = {}, {}
    for name, rel in relpaths.items():
        hits = [(r, r / rel) for r in roots if (r / rel).exists()]
        if len(hits) != 1:
            raise RuntimeError(f"discovery: {name} ({rel}) found {len(hits)} times, expected 1: "
                               f"{[str(h[1]) for h in hits]}")
        root_of[name], found[name] = hits[0]
    names = sorted({"." if r == input_root else str(r.relative_to(input_root)) for r in root_of.values()})
    found["roots_used"] = names                # §14 (12): folder names only, down to level 3
    return found


def find_kit_script(candidates) -> Path:
    """§14 (12): the kit script at one of the explicit candidate paths (is_file only, no listing); exactly one."""
    hits = [Path(c) for c in candidates if Path(c).is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"kit script found {len(hits)} times among the candidates, expected 1: {[str(h) for h in hits]}")
    return hits[0]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _kill_group(proc: subprocess.Popen, grace_s: float) -> None:
    """§14 (2): the whole group (the child and its descendants) gets SIGTERM, then SIGKILL after grace_s."""
    pgid = proc.pid                       # start_new_session=True: the group id is the child pid
    _signal_group(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + grace_s
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _group_alive(pgid):
        _signal_group(pgid, signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _tail(path: Path, n: int = 200) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_step(name: str, argv: list, timeout_s: float, logs_dir: Path, cwd=None, env=None,
             kill_grace_s: float = 10.0) -> dict:
    """§14 (1)(2)(3): one subprocess in its own group, stdout and stderr in files, no pipe, no subprocess.run timeout."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path, err_path = logs_dir / f"{name}.stdout", logs_dir / f"{name}.stderr"
    t0 = time.monotonic()
    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        proc = subprocess.Popen([str(a) for a in argv], cwd=cwd, env=env, stdout=out_f, stderr=err_f,
                                start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc, kill_grace_s)
        else:
            _signal_group(proc.pid, signal.SIGKILL)       # leftover descendants of a finished step
    rc = "TIMEOUT" if timed_out else proc.returncode
    return {"step": name, "rc": rc, "seconds": round(time.monotonic() - t0, 3),
            "stdout": str(out_path), "stderr": str(err_path), "stderr_tail": _tail(err_path)}


class Heartbeat:
    """§14 (4): a daemon thread writes heartbeat.jsonl every interval seconds."""

    def __init__(self, out: Path, interval_s: float = 60.0):
        self.out = Path(out)
        self.interval = interval_s
        self.cfg = json.loads(CONFIG.read_text())
        self.step = "setup"
        self.step_t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    def set_step(self, name: str) -> None:
        self.step, self.step_t0 = name, time.monotonic()

    def beat(self) -> dict:
        n = sum(1 for pid in self.cfg["pages"]["list"] if (self.out / "measures" / f"{pid}.json").exists())
        line = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "step": self.step, "seconds_in_step": round(time.monotonic() - self.step_t0, 1),
                "measures_json": n}
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "heartbeat.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        return line

    def _loop(self) -> None:
        """§14 (11): the loop catches and logs its own errors; it never ends the pipeline."""
        while not self._stop.wait(self.interval):
            try:
                self.beat()
            except Exception as exc:  # noqa: BLE001
                try:
                    with open(self.out / "heartbeat_errors.txt", "a") as fh:
                        fh.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {exc!r}\n")
                except OSError:
                    pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ld0-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def execute_pipeline(out: Path, steps: list, heartbeat_interval_s: float = 60.0, cleanup=None) -> dict:
    """Runs the steps in order. A step is {"name", "fn"} (in process) or {"name", "argv", "timeout_s", "cwd", "env",
    "retry", "before_retry", "after_timeout"} (subprocess). argv/env may be callables evaluated at run time.
    §14 (10): a retry (once) runs after before_retry (partial cache removed); a TIMEOUT or a failure stops the list,
    except that the step's "after_timeout" step runs after a TIMEOUT. The cause goes into the exit record.
    §14 (11): faulthandler dumps the stacks after 600 s in process steps (re-armed per step, cancelled around
    subprocess steps); the heartbeat is built inside the try; the exit record is written in the finally."""
    out = Path(out)
    rec = {"status": "DONE", "step": None, "exit_code": 0, "durations_s": {}, "attempts": {}, "cause": None,
           "exception": None}
    logs = out / "logs"
    hb = None
    fh = None
    current = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        fh = open(out / "ld0_faulthandler.txt", "a")
        faulthandler.dump_traceback_later(600, exit=True, file=fh)
        hb = Heartbeat(out, heartbeat_interval_s)
        hb.start()
        for st in steps:
            current = st["name"]
            hb.set_step(current)
            if "fn" in st:
                faulthandler.dump_traceback_later(600, exit=True, file=fh)
                t0 = time.monotonic()
                st["fn"]()
                rec["durations_s"][current] = round(time.monotonic() - t0, 3)
                continue
            faulthandler.cancel_dump_traceback_later()
            tries = 1 + int(st.get("retry", 0))
            r = None
            for attempt in range(1, tries + 1):
                if attempt > 1 and st.get("before_retry"):
                    st["before_retry"]()
                argv = st["argv"]() if callable(st["argv"]) else st["argv"]
                env = st["env"]() if callable(st.get("env")) else st.get("env")
                r = run_step(current, argv, st["timeout_s"], logs, cwd=st.get("cwd"), env=env)
                rec["attempts"][current] = attempt
                rec["durations_s"][current] = r["seconds"]
                print(f"--- {current} attempt {attempt}: rc={r['rc']} in {r['seconds']} s ; "
                      f"last lines of its stderr:", flush=True)
                print(r["stderr_tail"], flush=True)
                if r["rc"] == 0:
                    break
            if r["rc"] == "TIMEOUT":
                rec.update(status="TIMEOUT", step=current, exit_code="TIMEOUT",
                           cause={"step": current, "rc": "TIMEOUT", "stderr_tail": r["stderr_tail"]})
                if st.get("after_timeout"):
                    nxt = st["after_timeout"]
                    hb.set_step(nxt["name"])
                    nargv = nxt["argv"]() if callable(nxt["argv"]) else nxt["argv"]
                    r2 = run_step(nxt["name"], nargv, nxt["timeout_s"], logs, cwd=nxt.get("cwd"),
                                  env=nxt["env"]() if callable(nxt.get("env")) else nxt.get("env"))
                    rec["durations_s"][nxt["name"]] = r2["seconds"]
                    print(f"--- {nxt['name']}: rc={r2['rc']} ; last lines of its stderr:", flush=True)
                    print(r2["stderr_tail"], flush=True)
                break
            if r["rc"] != 0:
                rec.update(status="FAILED", step=current, exit_code=r["rc"],
                           cause={"step": current, "rc": r["rc"], "stderr_tail": r["stderr_tail"]})
                break
    except BaseException as exc:  # noqa: BLE001 — the exit record must be written whatever happened
        rec.update(status="FAILED", step=current, exit_code="EXCEPTION",
                   exception=f"{type(exc).__name__}: {exc}",
                   cause={"step": current, "exception": f"{type(exc).__name__}: {exc}"})
        print(traceback.format_exc(), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        if hb is not None:
            hb.stop()
        if cleanup is not None:
            try:
                cleanup(rec)
            except Exception as exc:  # noqa: BLE001
                rec["cleanup_exception"] = f"{type(exc).__name__}: {exc}"
        try:
            rec["timeouts_s_from_config"] = json.loads(CONFIG.read_text()).get("timeouts_s")
            out.mkdir(parents=True, exist_ok=True)
            (out / "data2_run_exit.json").write_text(json.dumps(rec, indent=2))
        finally:
            if fh is not None:
                fh.close()
    return rec
