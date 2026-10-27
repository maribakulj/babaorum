"""Two smoke steps on one pilot page per arm; checks finite losses and a valid report (CPU)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PILOT = Path("~/heritage-synth/runs/pilot-v0.2").expanduser()
CONFIG = ROOT / "config/stage_s_smoke.json"

if not (PILOT / "pages/mf_0000.json").exists() or not CONFIG.exists():
    pytest.skip("pilot pages or stage_s_smoke config not available", allow_module_level=True)

from train_stage_s import run  # noqa: E402


@pytest.mark.parametrize("arm", ["S2", "B3", "B4"])
def test_two_smoke_steps_run_for_each_arm(arm, tmp_path):
    cfg = json.loads(CONFIG.read_text())
    out = tmp_path / f"stage_s_smoke_{arm}.json"
    report = run(cfg, arm, out, train_pages=["mf_0000"], eval_pages=["mf_0090"], steps=2)

    assert report["steps_completed"] == 2
    for row in report["steps"]:
        for key in ("L_det", "L_geo", "L_ctc", "L_total"):
            assert math.isfinite(row[key]), (arm, key, row[key])
        assert row["seconds"] > 0
    assert report["status"] in ("SMOKE_ONLY_NO_CLAIM", "STOPPED_TIME_LIMIT_SMOKE_ONLY")
    assert report["arm"] == arm
    assert set(report["provenance"]["module_sha256"]) == {
        "stage_s_data", "stage_s_targets", "stage_s_model", "stage_s_reader", "stage_s_arms"}

    saved = json.loads(out.read_text())
    assert saved["steps_completed"] == 2
    assert len(saved["eval"]) == 1
