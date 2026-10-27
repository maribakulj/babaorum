"""CPU checks of the calibration driver guards and the v2 final-evaluation rule, with simulated time.

No Colab, no GPU, no pages. The fake runners advance a simulated clock by the time each
arm is allowed to use, so the 2-hour guard, the hard kill at cap + 3 min and the
final-evaluation rule are exercised without running anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import colab_stage_s_calibration as drv  # noqa: E402
import train_stage_s_v2 as trainer  # noqa: E402


def test_guard_parameters_are_read_from_the_pre_registration():
    params = drv.guard_parameters()
    assert params == {"cap_max_min": 30.0, "margin_min": 5.0, "min_cap_min": 10.0,
                      "total_min": 120.0, "kill_margin_min": 3.0}


@pytest.mark.parametrize("elapsed, expected_cap", [(0, 30.0), (85, 30.0), (100, 15.0)])
def test_cap_is_min_of_per_arm_cap_and_remaining_minus_margin(elapsed, expected_cap):
    cap, _ = drv.arm_cap_minutes(elapsed, drv.guard_parameters())
    assert cap == expected_cap


def test_arm_is_skipped_below_minimum_cap():
    assert drv.arm_cap_minutes(108.0, drv.guard_parameters())[0] is None  # remaining 12 -> cap 7 < 10


def _fake_clock(state):
    return lambda: state["now"] * 60.0


def _write_report(out: Path, arm: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": arm, "status": "SIM", "steps": [], "timeline": [], "eval": [],
                               "gpu_peak_mb": None, "steps_completed": 0,
                               "seconds_per_step_median_100_to_end": None, "elapsed_seconds": 0.0}))


def test_all_arms_fit_and_each_gets_hard_timeout_cap_plus_three(tmp_path, monkeypatch):
    state = {"now": 0.0}
    timeouts = {}

    def fake_runner(cmd, *, env=None, log=None, timeout=None):
        if "--output" not in cmd:
            return ""
        arm = cmd[cmd.index("--arm") + 1]
        cap = float(cmd[cmd.index("--time-cap-min") + 1])
        timeouts[arm] = timeout
        state["now"] += 2.0 if arm == "B4" and cap == 5.0 else cap
        _write_report(Path(cmd[cmd.index("--output") + 1]), arm)
        return ""

    monkeypatch.setattr(drv, "DRIVE", tmp_path / "drive")
    monkeypatch.setattr(drv, "PREFLIGHT_OUT", tmp_path / "preflight")
    results, _ = drv.step_d_arms([], 0.0, clock=_fake_clock(state), runner=fake_runner)
    assert [results[a]["status"] for a in ("B3", "B4", "S2")] == ["RAN", "RAN", "RAN"]
    assert timeouts == {"B3": 33 * 60.0, "B4": 33 * 60.0, "S2": 33 * 60.0}
    assert state["now"] <= 120.0


def test_hard_kill_is_recorded_and_next_arm_still_gets_a_guarded_cap(tmp_path, monkeypatch):
    state = {"now": 0.0}

    def fake_runner(cmd, *, env=None, log=None, timeout=None):
        if "--output" not in cmd:
            return ""
        arm = cmd[cmd.index("--arm") + 1]
        if arm == "B3":
            state["now"] += timeout / 60.0  # the process ran until the hard kill
            raise drv.ArmKilled("killed at hard timeout")
        cap = float(cmd[cmd.index("--time-cap-min") + 1])
        state["now"] += cap
        _write_report(Path(cmd[cmd.index("--output") + 1]), arm)
        return ""

    monkeypatch.setattr(drv, "DRIVE", tmp_path / "drive")
    monkeypatch.setattr(drv, "PREFLIGHT_OUT", tmp_path / "preflight")
    results, reports = drv.step_d_arms([], 0.0, clock=_fake_clock(state), runner=fake_runner)
    # preflight 5 min, then B3 runs to its hard kill at cap 30 + 3 = 33 min: elapsed 5 + 33 = 38
    assert results["B3"]["status"] == "KILLED_HARD_TIMEOUT"
    assert results["B3"]["cap_min"] == 30.0
    assert "B3" not in reports
    assert results["B4"]["status"] == "RAN" and results["B4"]["cap_min"] == 30.0
    assert results["B4"]["elapsed_before_min"] == pytest.approx(38.0)
    assert results["S2"]["status"] == "RAN" and results["S2"]["elapsed_before_min"] == pytest.approx(68.0)
    assert state["now"] == pytest.approx(98.0)
    assert state["now"] <= 120.0


def test_late_start_shrinks_caps_and_skips_arms_below_minimum(tmp_path, monkeypatch):
    state = {"now": 82.0}

    def fake_runner(cmd, *, env=None, log=None, timeout=None):
        if "--output" not in cmd:
            return ""
        cap = float(cmd[cmd.index("--time-cap-min") + 1])
        state["now"] += cap
        _write_report(Path(cmd[cmd.index("--output") + 1]), "x")
        return ""

    monkeypatch.setattr(drv, "DRIVE", tmp_path / "drive")
    monkeypatch.setattr(drv, "PREFLIGHT_OUT", tmp_path / "preflight")
    results, _ = drv.step_d_arms([], 0.0, clock=_fake_clock(state), runner=fake_runner)
    # 82 min used + 5 min preflight cap = 87 min before B3: cap = min(30, 120 - 87 - 5) = 28
    assert results["B3"]["status"] == "RAN" and results["B3"]["cap_min"] == 28.0
    assert results["B4"]["status"] == "SKIPPED_GLOBAL_GUARD"
    assert results["S2"]["status"] == "SKIPPED_GLOBAL_GUARD"


def test_run_passes_timeout_to_subprocess_and_maps_expiry_to_arm_killed(monkeypatch):
    import subprocess

    def fake_subprocess_run(cmd, **kwargs):
        assert kwargs["timeout"] == 180.0
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(drv.subprocess, "run", fake_subprocess_run)
    with pytest.raises(drv.ArmKilled):
        drv.run(["python", "x"], timeout=180.0)


def test_storage_errors_are_classified_as_write_refusals():
    assert drv.classify_failure("OSError: [Errno 28] No space left on device")
    assert drv.classify_failure("Read-only file system")
    assert not drv.classify_failure("ValueError: baseline chord runs right-to-left")


def test_check_writable_passes_and_refuses_below_floor(tmp_path):
    ok = drv.check_writable(tmp_path / "w", min_free_gib=1e-9)
    assert ok["probe"] == "ok"
    with pytest.raises(drv.WriteRefused):
        drv.check_writable(tmp_path / "w2", min_free_gib=1e9)


def test_mille_feuilles_hashes_must_match_the_manifest(tmp_path, monkeypatch):
    src = tmp_path / "heritage-synth/src/mille_feuilles/x.py"
    src.parent.mkdir(parents=True)
    src.write_text("print('a')\n")
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    # explicit roots: a temporary clone-like tree, no global state, no skip
    clone = tmp_path / "heritage-synth"
    ok = drv.verify_mille_feuilles({"mille_feuilles_sha256": {"heritage-synth/src/mille_feuilles/x.py": good}},
                                   heritage_root=clone, bundle_root=tmp_path)
    assert ok["matches_manifest"] is True
    with pytest.raises(RuntimeError, match="differ from the bundle manifest"):
        drv.verify_mille_feuilles({"mille_feuilles_sha256": {"heritage-synth/src/mille_feuilles/x.py": "0" * 64}},
                                  heritage_root=clone, bundle_root=tmp_path)


def test_cap_scope_is_read_from_the_pre_registration():
    scope = trainer.load_cap_scope()
    assert scope == {"final_eval_min_ratio": 1.5, "kill_margin_min": 3.0}


@pytest.mark.parametrize("remaining, last, expected", [
    (10.0, 4.0, True),    # 10 >= 1.5 x 4 = 6
    (5.0, 4.0, False),    # 5 < 6
    (6.0, 4.0, True),     # boundary
    (0.5, None, True),    # no periodic evaluation yet: any positive remaining time
    (0.0, None, False),
])
def test_final_eval_runs_only_if_remaining_covers_ratio_times_last_periodic(remaining, last, expected):
    assert trainer.final_eval_allowed(remaining * 60.0, None if last is None else last * 60.0, 1.5) is expected


def test_final_status_word_reports_budget_skips_and_caps():
    class Dev:
        type = "cuda"

    assert trainer.final_status_word("FINAL_EVAL_SKIPPED_BUDGET", False, Dev, 30.0) == "FINAL_EVAL_SKIPPED_BUDGET"
    assert trainer.final_status_word("RUN", True, Dev, 30.0) == "STOPPED_TIME_CAP"
    assert trainer.final_status_word("RUN", False, Dev, 30.0) == "COMPLETED"


def test_real_subprocess_is_killed_at_its_hard_timeout():
    """Real kill, not simulated: a 30-s child is stopped after 0.5 s and reported as ArmKilled."""
    import time

    t0 = time.monotonic()
    with pytest.raises(drv.ArmKilled):
        drv.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert time.monotonic() - t0 < 10.0
