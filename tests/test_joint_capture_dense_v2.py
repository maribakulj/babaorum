"""Checks for the dense corrective re-analysis helpers (no OCR, no data)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_joint_capture_dense_v2 import Tabulated, contiguous_interval  # noqa: E402


def make_table(fn, lo=-50.0, hi=50.0, step=0.005):
    tab = Tabulated.__new__(Tabulated)
    tab.y = np.arange(lo, hi + 1e-12, step)
    tab.l = fn(tab.y)
    return tab


def test_expected_value_and_derivative_match_gaussian_closed_form():
    tab = make_table(lambda y: y * y)  # E[(mu+s z)^2] = mu^2 + s^2, d/dmu = 2 mu
    value, derivative = tab.expected(1.7, 2.0)
    assert value == pytest.approx(1.7 ** 2 + 4.0, rel=1e-6)
    assert derivative == pytest.approx(2 * 1.7, rel=1e-6)


def test_expected_smooths_a_kink_without_node_artifacts():
    tab = make_table(np.abs)  # E|mu + s z| derivative = erf(mu / (s sqrt 2))
    _, derivative = tab.expected(0.3, 1.0)
    assert derivative == pytest.approx(math.erf(0.3 / math.sqrt(2)), abs=1e-4)


def test_support_outside_tabulation_is_refused():
    tab = make_table(lambda y: y, lo=-1.0, hi=1.0)
    with pytest.raises(ValueError, match="tabulated range"):
        tab.expected(0.0, 1.0)


def test_contiguous_interval_reports_edges_and_censoring():
    offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert contiguous_interval(offsets, [True, True, True, False, True]) == {
        "contains_target": True, "left_h": -1.0, "right_h": 0.0,
        "left_censored": True, "right_censored": False}
    assert contiguous_interval(offsets, [True, True, False, True, True])["contains_target"] is False


def test_archived_report_matches_current_sources():
    report = json.loads((ROOT / "reports/joint_capture_dense_v2.json").read_text())
    import hashlib
    assert report["config_sha256"] == hashlib.sha256(
        (ROOT / "config/joint_capture_dense_v2.json").read_bytes()).hexdigest()
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["ocr_performance_claim"] is False
