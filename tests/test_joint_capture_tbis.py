"""Checks for the T4-bis capture-radius probe (synthetic, no OCR, no data)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_joint_capture_tbis import Table, radius  # noqa: E402


def quadratic_table():
    table = Table.__new__(Table)
    table.y = np.arange(-60.0, 60.0 + 1e-12, 0.005)
    table.l = table.y ** 2
    table.dl = np.gradient(table.l, table.y)
    table.y_star = 0.0
    return table


def test_gaussian_value_and_both_derivatives_match_closed_form():
    e, d_mu, d_sigma = quadratic_table().gaussian(1.5, 2.0)  # E = mu^2 + s^2
    assert e == pytest.approx(1.5 ** 2 + 4.0, rel=1e-6)
    assert d_mu == pytest.approx(3.0, rel=1e-6)
    assert d_sigma == pytest.approx(4.0, rel=1e-5)


def test_support_exit_is_an_error_not_a_clamp():
    with pytest.raises(ValueError, match="tabulated range"):
        quadratic_table().gaussian(55.0, 2.0)


def test_radius_is_symmetric_contiguous_and_flags_censoring():
    offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert radius(offsets, [False, True, True, True, True]) == {
        "lower_h": 0.5, "upper_h": 1.0, "lower_censored": False, "upper_censored": True}
    assert radius(offsets, [True, True, False, True, True]) == {"lower_h": None, "upper_h": None}


def test_archived_report_provenance_and_scope():
    report = json.loads((ROOT / "reports/joint_capture_tbis.json").read_text())
    assert report["config_sha256"] == hashlib.sha256(
        (ROOT / "config/joint_capture_tbis.json").read_bytes()).hexdigest()
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["ocr_performance_claim"] is False
    assert all(math.isfinite(v["y_star"]) for v in report["results"].values())
