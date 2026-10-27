import json
import hashlib
from pathlib import Path
import subprocess
import sys

import torch

from scripts.probe_joint_capture_trajectories import kl_normal
from scripts.probe_joint_variational_band import gaussian_kl


ROOT = Path(__file__).resolve().parents[1]


def test_exploratory_capture_trajectories_emit_all_declared_starts_and_finite_paths(tmp_path):
    output = tmp_path / "capture-trajectories.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/probe_joint_capture_trajectories.py"),
            "--config",
            str(ROOT / "config/joint_capture_trajectories.json"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    def reject_nonstandard_constant(value):
        raise AssertionError(f"non-standard JSON numeric constant: {value}")

    report = json.loads(output.read_text(), parse_constant=reject_nonstandard_constant)

    assert report["result"] == "COMPUTED_EXPLORATORY_TRAJECTORIES"
    config_path = ROOT / "config/joint_capture_trajectories.json"
    assert report["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    for relative_path, expected_hash in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash
    archived = json.loads((ROOT / "reports/joint_capture_trajectories.json").read_text())
    assert report["result"] == archived["result"]
    assert report["config_sha256"] == archived["config_sha256"]
    assert report["source_sha256"] == archived["source_sha256"]
    runtime_fields = ("python_version", "platform", "torch_version")
    if all(report[key] == archived[key] for key in runtime_fields):
        assert report["trajectories"] == archived["trajectories"]
    assert all(report["checks"].values())
    assert len(report["trajectories"]) == 6
    for item in report["trajectories"].values():
        assert item["all_values_finite"]
        assert [row["step"] for row in item["trajectories"]["point_ctc"]] == [0, 1, 5, 10, 20, 40]
        assert [row["step"] for row in item["trajectories"]["gaussian_expected_ctc_plus_kl"]] == [0, 1, 5, 10, 20, 40]


def test_trajectory_kl_matches_t4_formula_and_pulls_mean_toward_ground_truth():
    mu = torch.tensor(52.75, dtype=torch.float64, requires_grad=True)
    sigma = 1.5
    target_y = 50.0
    prior_sigma = 1.5

    trajectory_value = kl_normal(mu, sigma, target_y, prior_sigma)
    t4_value = gaussian_kl(
        mu, torch.tensor(sigma, dtype=torch.float64), target_y, prior_sigma
    )
    gradient = torch.autograd.grad(trajectory_value, mu)[0]

    torch.testing.assert_close(trajectory_value, t4_value, atol=1e-14, rtol=0)
    torch.testing.assert_close(
        gradient,
        torch.tensor((52.75 - target_y) / prior_sigma**2, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
