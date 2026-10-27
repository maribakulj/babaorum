"""CLI checks for the v2 trainer: options, journal, checkpoint, timeline.

Fast by default: a tiny synthetic page (256 px) written in a temporary folder, reduced model
(channels 8, hidden 16), 128-px tiles, 2 steps, evaluation every step. The original
pilot-page version runs only with BABAORUM_SLOW_TESTS=1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CONFIG = ROOT / "config/stage_s_smoke_v2.json"

import train_stage_s_v2 as trainer  # noqa: E402

PILOT = Path("~/heritage-synth/runs/pilot-v0.2").expanduser()
SLOW = os.environ.get("BABAORUM_SLOW_TESTS") == "1"


def _write_synthetic_root(root: Path):
    (root / "pages").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    img = np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8)
    for pid in ("mf_0000", "mf_0090"):
        png = root / "images" / f"{pid}.png"
        Image.fromarray(img, mode="L").save(png)
        meta = {"sha256": hashlib.sha256(png.read_bytes()).hexdigest(), "height": 256, "width": 256}
        lines = []
        for k, (y, text) in enumerate([(100, "AB"), (160, "CD")]):
            lines.append({"id": f"{pid}_l{k}", "block_id": f"{pid}_b0", "text": text,
                          "legibility": "readable", "baseline": [[20.0, float(y)], [230.0, float(y)]],
                          "polygon": [[20.0, y - 8.0], [230.0, y - 8.0], [230.0, y + 8.0], [20.0, y + 8.0]]})
        (root / "pages" / f"{pid}.json").write_text(json.dumps(
            {"page_id": pid, "image": meta, "lines": lines}))
    return root


def _fast_config(tmp_path: Path) -> Path:
    cfg = json.loads(CONFIG.read_text())
    cfg["model"]["channels"] = 8
    cfg["model"]["hidden"] = 16
    cfg["tiles"]["size_px"] = 128
    cfg["steps"] = 2
    path = tmp_path / "fast_config.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.mark.parametrize("arm", ["B3", "S2"])
def test_fast_two_steps_eval_every_step_writes_report_journal_and_checkpoint(arm, tmp_path):
    root = _write_synthetic_root(tmp_path / "pages_root")
    out = tmp_path / f"report_{arm}.json"
    log_dir = tmp_path / "logs" / arm
    report = trainer.main([
        "--config", str(_fast_config(tmp_path)), "--arm", arm, "--output", str(out), "--device", "cpu",
        "--steps", "2", "--eval-every", "1", "--eval-pages", "mf_0090", "--train-pages", "mf_0000",
        "--root", str(root), "--log-dir", str(log_dir),
    ])
    assert report["steps_completed"] == 2
    for row in report["steps"]:
        for key in ("L_det", "L_geo", "L_ctc", "L_total"):
            assert math.isfinite(row[key]), (arm, key, row[key])
    assert [t["step"] for t in report["timeline"]] == [1, 2]
    assert report["timeline"][-1].get("final") is True
    assert (log_dir / "checkpoint_last.pt").exists()
    lines = (log_dir / "journal.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["step"] == 2
    saved = json.loads(out.read_text())
    assert saved["device"] == "cpu" and saved["gpu_peak_mb"] is None
    assert saved["provenance"]["config_sha256"] is not None


@pytest.mark.skipif(not SLOW, reason="slow pilot-page check; set BABAORUM_SLOW_TESTS=1")
def test_slow_pilot_three_steps_eval_every_two(tmp_path):
    if not (PILOT / "pages/mf_0000.json").exists():
        pytest.skip("pilot pages not available")
    out = tmp_path / "report.json"
    log_dir = tmp_path / "logs"
    report = trainer.main([
        "--arm", "B3", "--output", str(out), "--device", "cpu", "--steps", "3",
        "--eval-every", "2", "--eval-pages", "mf_0090", "--train-pages", "mf_0000",
        "--root", str(PILOT), "--log-dir", str(log_dir),
    ])
    assert report["steps_completed"] == 3
    assert [t["step"] for t in report["timeline"]] == [2, 3]
    assert (log_dir / "checkpoint_last.pt").exists()
