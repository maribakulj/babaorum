"""Sign test: a refinement offset delta = -d must cancel a normal shift d of a GT baseline (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from babaorum.experimental.joint_curves import curve_geometry_loss
from babaorum.experimental.stage_s_model import HeadD, HeadG, refine_lines, resample_k

K = 8
HEIGHT = 20.0
GT = [[40.0, 100.0], [300.0, 110.0]]


def _head_g_with_constant_delta(delta):
    """HeadG whose offsets are the constant `delta` and whose distance corrections are zero."""
    torch.manual_seed(0)
    hg = HeadG(64, K).double()
    with torch.no_grad():
        for p in hg.parameters():
            p.zero_()
        hg.mlp[-1].bias[:K] = float(delta)
    return hg


def _shift_normal(baseline, d):
    b = np.asarray(baseline, dtype=np.float64)
    t = b[-1] - b[0]
    t = t / np.linalg.norm(t)
    n = np.array([-t[1], t[0]])
    return (b + d * n[None, :]).tolist()


def _refined_and_loss(input_polyline, delta):
    torch.manual_seed(0)
    features = torch.randn(1, 64, 128, 128, dtype=torch.float64)
    hd = HeadD(64).double()
    hg = _head_g_with_constant_delta(delta)
    out = refine_lines(features, [input_polyline], hg, head_d=hd, K=K)
    gt = torch.as_tensor(resample_k(GT, K), dtype=torch.float64)
    loss = curve_geometry_loss(out["baseline"][0], gt, target_height_px=HEIGHT, spacing_px=4.0)
    return loss


@pytest.mark.parametrize("d", [3.0, -4.0])
def test_delta_minus_d_cancels_the_normal_shift(d):
    loss = _refined_and_loss(_shift_normal(GT, d), delta=-d)
    assert float(loss.total.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(loss.curve.detach()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("d", [3.0, -4.0])
def test_delta_plus_d_doubles_the_normal_error(d):
    loss = _refined_and_loss(_shift_normal(GT, d), delta=+d)
    expected_curve = 2.0 * abs(d) / HEIGHT
    assert float(loss.curve) == pytest.approx(expected_curve, rel=1e-3)
