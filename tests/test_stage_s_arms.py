"""Gradient routing and parameter counts for the stage S arms (synthetic 256x256 tile, CPU)."""

from __future__ import annotations

import torch

from babaorum.experimental.stage_s_arms import StageSModel

ALPHABET = 10
POLY = [[[30.0, 80.0], [220.0, 85.0]], [[30.0, 160.0], [220.0, 165.0]]]
LABELS = [[2, 4, 2], [3, 5]]
SIGMA = 0.0


def _inputs(seed=0):
    torch.manual_seed(seed)
    return torch.rand(1, 1, 256, 256)


def _build(arm, seed=0):
    torch.manual_seed(seed)
    return StageSModel(arm, ALPHABET)


def _total(out):
    included = [x for x in out["ctc_losses"] if x is not None]
    assert included, "no line included; the check would be vacuous"
    return sum(included)


def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum())
    return total


def test_b4_gradient_reaches_head_g():
    model = _build("B4")
    out = model.forward_lines(_inputs(), POLY, LABELS, sigma_px=SIGMA)
    _total(out).backward()
    assert _grad_norm(model.head_g) > 0


def test_b3_blocks_gradient_to_head_g_but_reaches_shared_encoder():
    model = _build("B3")
    out = model.forward_lines(_inputs(), POLY, LABELS, sigma_px=SIGMA)
    _total(out).backward()
    assert _grad_norm(model.head_g) == 0
    assert _grad_norm(model.enc_det) > 0


def test_s2_blocks_gradient_to_det_branch_and_reaches_read_encoder():
    model = _build("S2")
    out = model.forward_lines(_inputs(), POLY, LABELS, sigma_px=SIGMA)
    _total(out).backward()
    assert _grad_norm(model.head_g) == 0
    assert _grad_norm(model.enc_det) == 0
    assert _grad_norm(model.enc_read) > 0


def test_b3_and_b4_have_identical_forward_geometry_from_same_seed():
    x = _inputs()
    out3 = _build("B3").forward_lines(x, POLY, LABELS, sigma_px=SIGMA)
    out4 = _build("B4").forward_lines(x, POLY, LABELS, sigma_px=SIGMA)
    assert torch.allclose(out3["refined"]["baseline"], out4["refined"]["baseline"], atol=1e-6)


def test_parameter_counts_per_arm_are_reported():
    counts = {arm: sum(p.numel() for p in _build(arm).parameters()) for arm in ("S2", "B3", "B4")}
    print(f"\nparameters: {counts}")
    assert counts["S2"] > counts["B3"]
    assert counts["B3"] == counts["B4"]
