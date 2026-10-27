"""CPU checks of the source gate (pinned git bundle), the launch criterion (10 pages required)
and the always-written final report. Temporary repos and fake reports only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import colab_stage_s_calibration as drv  # noqa: E402

PAGES = [f"mf_{i:04d}" for i in range(50, 60)]


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def bundle(tmp_path):
    src = tmp_path / "src-repo"
    src.mkdir()
    _git(["init", "-q"], src)
    _git(["config", "user.email", "t@example.org"], src)
    _git(["config", "user.name", "t"], src)
    (src / "pyproject.toml").write_text("[project]\nname='x'\n")
    _git(["add", "."], src)
    _git(["commit", "-q", "-m", "one"], src)
    head = _git(["rev-parse", "HEAD"], src)
    gb = tmp_path / "pinned.gitbundle"
    _git(["bundle", "create", str(gb), "HEAD"], src)
    return gb, head, tmp_path


def test_source_gate_passes_on_the_pinned_commit(bundle, monkeypatch):
    gb, head, tmp = bundle
    monkeypatch.setattr(drv, "GITBUNDLE", gb)
    monkeypatch.setattr(drv, "HERITAGE", tmp / "content" / "heritage-synth")
    monkeypatch.setattr(drv, "PINNED_COMMIT", head)
    info = drv.step_0_source([])
    assert info["head"] == head and info["clean"] is True
    assert (tmp / "content" / "heritage-synth" / "pyproject.toml").exists()


def test_source_gate_refuses_a_different_commit(bundle, monkeypatch):
    gb, head, tmp = bundle
    monkeypatch.setattr(drv, "GITBUNDLE", gb)
    monkeypatch.setattr(drv, "HERITAGE", tmp / "content2" / "heritage-synth")
    monkeypatch.setattr(drv, "PINNED_COMMIT", "0" * 40)
    with pytest.raises((drv.SourceMismatch, subprocess.CalledProcessError, RuntimeError)):
        drv.step_0_source([])


def _eval(page_id, recall, precision, n_pred=10):
    return {"page_id": page_id,
            "metrics": {"after": {"recall": recall, "precision": precision, "n_pred": n_pred}}}


def _report(steps_evals):
    timeline = [{"step": s, "evals": evs, **({"final": True} if fin else {})} for s, evs, fin in steps_evals]
    return {"timeline": timeline, "steps": []}


def _evals(recall, precision, pages=PAGES):
    return [_eval(p, recall, precision) for p in pages]


def test_launch_gate_uses_b3_or_b4_only_and_mean_over_pages():
    bad = _evals(0.2, 0.2)
    good = _evals(0.6, 0.6)
    gate = drv.launch_criterion({"B3": _report([(1000, bad, False)]),
                                 "B4": _report([(1000, bad, False)]),
                                 "S2": _report([(1000, good, False)])})
    assert gate["launch_gate"] == "FAIL_STAGE_S_NOT_LAUNCHED"
    gate = drv.launch_criterion({"B3": _report([(1000, bad, False)]),
                                 "B4": _report([(1000, good, True)])})
    assert gate["launch_gate"] == "PASS"
    assert gate["arms"]["B4"]["step"] == 1000 and gate["arms"]["B4"]["final_eval"] is True


def test_nine_pages_is_not_evaluable_never_pass():
    good9 = _evals(0.9, 0.9, pages=PAGES[:9])
    gate = drv.launch_criterion({"B3": _report([(1000, good9, True)]), "B4": _report([(1000, good9, True)])})
    assert gate["launch_gate"] == "CRITERION_NOT_EVALUABLE"
    assert gate["arms"]["B3"]["status"] == "CRITERION_NOT_EVALUABLE"
    assert gate["arms"]["B3"]["missing_pages"] == ["mf_0059"]


def test_no_prediction_counts_as_precision_zero():
    evals = _evals(0.9, 0.9)
    evals[-1] = _eval(PAGES[-1], 0.9, None, n_pred=0)
    gate = drv.launch_criterion({"B3": _report([(1000, evals, False)])})
    arm = gate["arms"]["B3"]
    assert arm["mean_precision_after"] == pytest.approx((9 * 0.9 + 0.0) / 10)  # 0.81
    assert arm["meets_criterion"] is True  # recall 0.9 and precision 0.81 both reach 0.5


def test_report_is_written_even_when_bundle_hashing_raises(tmp_path, monkeypatch):
    def broken_hashes():
        raise OSError("hashing failed on purpose")

    monkeypatch.setattr(drv, "bundle_hashes", broken_hashes)
    monkeypatch.setattr(drv, "DRIVE", tmp_path / "drive")
    monkeypatch.setattr(drv, "GITBUNDLE", tmp_path / "missing.gitbundle")  # forces an early failure
    monkeypatch.setattr(drv, "HERITAGE", tmp_path / "content" / "heritage-synth")
    drv.main(argv=[])
    written = json.loads((tmp_path / "drive" / "calibration_report.json").read_text())
    assert written["bundle_sha256"] is None
    assert "hashing failed on purpose" in written["bundle_hash_error"]
    assert written["status"] in ("STOPPED_SOURCE_MISMATCH", "FAILED", "STOPPED_WRITE_REFUSED")
