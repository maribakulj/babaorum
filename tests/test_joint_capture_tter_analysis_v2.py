"""Checks for the T4-ter v2 reanalysis output (synthetic, no OCR, no data)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports/joint_capture_tter_analysis_v2.json"


def load():
    return json.loads(ANALYSIS.read_text())


def test_recorded_input_hashes_match_current_files():
    recorded = load()["input_sha256"]
    assert set(recorded) == {
        "config/joint_capture_tter.json",
        "reports/joint_capture_tter.json",
        "scripts/probe_joint_capture_tter.py",
    }
    for path, digest in recorded.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_status_and_constants_are_declared():
    out = load()
    assert out["schema_version"] == "joint_capture_tter_analysis_v2"
    assert out["status"] == "REANALYSIS_NO_NEW_OPTIMIZATION"
    assert out["constants"] == {"plateau_tolerance_L": 1e-3, "background_tolerance_L": 1e-6}


def test_y_star_lies_inside_its_plateau():
    for key, stride in load()["strides"].items():
        assert stride["y_star_in_plateau"] is True, key
        assert stride["plateau"]["lo_px"] <= stride["y_star_table"] <= stride["plateau"]["hi_px"], key


def test_plateau_is_contained_in_support():
    for key, stride in load()["strides"].items():
        assert stride["plateau_in_support"] is True, key
        support = stride["support"]
        assert support is not None, key
        assert support["lo_px"] <= stride["plateau"]["lo_px"], key
        assert support["hi_px"] >= stride["plateau"]["hi_px"], key
