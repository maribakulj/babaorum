import json
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from babaorum.experimental.joint_geometry import FeatureFrame
from scripts.probe_joint_capture_profile import ctc_at_y


ROOT = Path(__file__).resolve().parents[1]


def test_predeclared_capture_profile_reports_point_and_smoothed_directional_ranges(tmp_path):
    output = tmp_path / "capture-profile.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/probe_joint_capture_profile.py"),
            "--config",
            str(ROOT / "config/joint_capture_profile.json"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())

    assert report["result"] == "PROFILE_COMPUTED_SYNTHETIC_ONLY"
    assert all(report["checks"].values())
    assert len(report["profiles"]) == 17
    intervals = report["contiguous_directional_intervals_from_target"]
    assert intervals["point_ctc"]["right_offset_h"] == 0.0
    assert intervals["point_ctc"]["left_edge_censored"] is True
    assert intervals["gaussian_0.5h_ctc"]["right_offset_h"] == 1.25
    assert intervals["gaussian_0.5h_ctc_plus_kl"]["right_offset_h"] == 0.5

    config = json.loads((ROOT / "config/joint_capture_profile.json").read_text())
    epsilon = config["directional_gradient_epsilon"]
    evidence_y = config["evidence_center_y"]

    def points_toward(y, gradient):
        if y < evidence_y:
            return gradient < -epsilon
        if y > evidence_y:
            return gradient > epsilon
        return abs(gradient) <= epsilon

    cases = {"point_ctc": lambda row: (row["point_dloss_dy"], row["point_gradient_toward_evidence"])}
    for i, sigma in enumerate(config["gaussian_sigma_fractions_of_height"]):
        cases[f"gaussian_{sigma:g}h_ctc"] = lambda row, j=i: (
            row["gaussian"][j]["d_expected_ctc_dmu"],
            row["gaussian"][j]["ctc_gradient_toward_evidence"],
        )
        cases[f"gaussian_{sigma:g}h_ctc_plus_kl"] = lambda row, j=i: (
            row["gaussian"][j]["d_ctc_plus_kl_dmu"],
            row["gaussian"][j]["total_gradient_toward_evidence"],
        )

    for name, extract in cases.items():
        valid = []
        for row in report["profiles"]:
            gradient, stored_flag = extract(row)
            expected_flag = points_toward(row["baseline_y"], gradient)
            assert stored_flag is expected_flag
            valid.append(expected_flag)
        zero_idx = [row["offset_heights_from_target"] for row in report["profiles"]].index(0.0)
        if valid[zero_idx]:
            left = right = zero_idx
            while left > 0 and valid[left - 1]:
                left -= 1
            while right + 1 < len(valid) and valid[right + 1]:
                right += 1
            expected_interval = (
                report["profiles"][left]["offset_heights_from_target"],
                report["profiles"][right]["offset_heights_from_target"],
            )
        else:
            expected_interval = (0.0, 0.0)
        observed = intervals[name]
        assert (observed["left_offset_h"], observed["right_offset_h"]) == expected_interval
        assert observed["left_edge_censored"] is (expected_interval[0] == -2.0)
        assert observed["right_edge_censored"] is (expected_interval[1] == 2.0)


def test_capture_profile_preserves_config_and_source_hashes_on_support_error(tmp_path):
    config = ROOT / "config/joint_capture_profile_support_fail.json"
    output = tmp_path / "capture-profile-error.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/probe_joint_capture_profile.py"),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())

    assert result.returncode == 1
    assert report["result"] == "ERROR"
    assert report["error"]["message"] == "sampled line band extends outside original page pixels"
    assert report["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert "scripts/probe_joint_capture_profile.py" in report["source_sha256"]


def test_point_and_soft_band_gradients_match_central_finite_differences():
    config = json.loads((ROOT / "config/joint_capture_profile.json").read_text())
    dtype = torch.float64
    width, page_height = config["image_size_wh"]
    target_y, evidence_y = config["target_baseline_y"], config["evidence_center_y"]
    yy, xx = torch.meshgrid(
        torch.arange(page_height, dtype=dtype),
        torch.arange(width, dtype=dtype), indexing="ij",
    )
    field = torch.stack((
        torch.zeros_like(xx),
        5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - evidence_y).square(),
        5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - evidence_y).square(),
    ))[None]
    frame = FeatureFrame((width, page_height), (0.0, 0.0), (1.0, 1.0))
    x = torch.tensor([2.0, 13.0], dtype=dtype)
    height = torch.tensor([config["height_px"]], dtype=dtype)
    epsilon = 1e-5

    def central_difference(fn, value):
        return (fn(value + epsilon) - fn(value - epsilon)) / (2 * epsilon)

    for offset_h in (0.0, 0.13, 0.2, 0.3, 0.37, 0.5):
        y0 = target_y + offset_h * config["height_px"]
        y = torch.tensor(y0, dtype=dtype, requires_grad=True)
        loss = ctc_at_y(y, field, x, height, frame, config)
        auto = torch.autograd.grad(loss, y)[0]
        finite = central_difference(
            lambda value: ctc_at_y(value, field, x, height, frame, config), y0
        )
        assert torch.allclose(auto, finite, rtol=2e-6, atol=2e-7)

    nodes_np, weights_np = np.polynomial.hermite.hermgauss(config["quadrature_order"])
    nodes = torch.tensor(nodes_np * math.sqrt(2), dtype=dtype)
    weights = torch.tensor(weights_np / math.sqrt(math.pi), dtype=dtype)
    sigma = 0.5 * config["height_px"]
    tau = config["prior_sigma_fraction_of_height"] * config["height_px"]
    mu0 = target_y + 0.5 * config["height_px"]

    def expected_plus_kl(mu):
        expected = torch.stack([
            ctc_at_y(mu + sigma * node, field, x, height, frame, config)
            for node in nodes
        ]).dot(weights)
        kl = math.log(tau / sigma) + (
            sigma**2 + (mu - target_y) ** 2
        ) / (2 * tau**2) - 0.5
        return expected + kl

    mu = torch.tensor(mu0, dtype=dtype, requires_grad=True)
    objective = expected_plus_kl(mu)
    auto = torch.autograd.grad(objective, mu)[0]
    finite = central_difference(expected_plus_kl, mu0)
    assert torch.allclose(auto, finite, rtol=3e-6, atol=3e-7)


def test_bilinear_grid_kink_changes_gradient_magnitude_but_not_direction():
    config = json.loads((ROOT / "config/joint_capture_profile.json").read_text())
    dtype = torch.float64
    width, page_height = config["image_size_wh"]
    yy, xx = torch.meshgrid(
        torch.arange(page_height, dtype=dtype),
        torch.arange(width, dtype=dtype), indexing="ij",
    )
    evidence_y = config["evidence_center_y"]
    field = torch.stack((
        torch.zeros_like(xx),
        5.0 - 0.15 * (xx - 4.0).square() - 0.3 * (yy - evidence_y).square(),
        5.0 - 0.15 * (xx - 11.0).square() - 0.3 * (yy - evidence_y).square(),
    ))[None]
    frame = FeatureFrame((width, page_height), (0.0, 0.0), (1.0, 1.0))
    x = torch.tensor([2.0, 13.0], dtype=dtype)
    height = torch.tensor([config["height_px"]], dtype=dtype)
    y0 = config["target_baseline_y"] + 0.25 * config["height_px"]
    y = torch.tensor(y0, dtype=dtype, requires_grad=True)
    loss = ctc_at_y(y, field, x, height, frame, config)
    auto = torch.autograd.grad(loss, y)[0]
    epsilon = 1e-5
    finite = (
        ctc_at_y(torch.tensor(y0 + epsilon, dtype=dtype), field, x, height, frame, config)
        - ctc_at_y(torch.tensor(y0 - epsilon, dtype=dtype), field, x, height, frame, config)
    ) / (2 * epsilon)

    assert torch.sign(auto) == torch.sign(finite)
    assert abs(float(auto - finite)) > 0.02
