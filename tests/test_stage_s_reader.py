"""Checks for the stage S reader and expected CTC loss (untrained, CPU, random tensors)."""

from __future__ import annotations

import torch

from babaorum.experimental.stage_s_reader import (
    Reader, band_features, expected_ctc_loss, offset_nodes, _ctc_nll,
)

ALPHABET = 10
LABELS = [3, 5, 3, 7]
CHANNELS, COARSE_H, COARSE_W = 64, 128, 128   # tile 512 x 512 at stride 4


def _features(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, CHANNELS, COARSE_H, COARSE_W)


def _line(x0=40.0, x1=300.0, y0=100.0, y1=110.0, *, requires_grad=False):
    base = torch.tensor([[x0, y0], [x1, y1]], dtype=torch.float32, requires_grad=requires_grad)
    return {"baseline": base, "neg": 12.0, "pos": 6.0}


def _reader(seed=1):
    torch.manual_seed(seed)
    return Reader(CHANNELS * 8, ALPHABET)


def test_zero_sigma_equals_point_ctc():
    feats, reader, line = _features(), _reader(), _line()
    loss, status = expected_ctc_loss(feats, line, reader, LABELS, sigma_px=0.0)
    band = band_features(feats, line["baseline"], line["neg"], line["pos"])
    expected = _ctc_nll(reader(band), LABELS)
    assert status == "ok"
    assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-6)


def test_offset_weights_sum_to_one_and_are_symmetric():
    for sigma in (0.5, 2.0, 6.0):
        u, w = offset_nodes(sigma, 9)
        assert len(u) == len(w) == 9
        assert abs(float(w.sum()) - 1.0) < 1e-12
        assert abs(float(u[0]) + 3 * sigma) < 1e-12 and abs(float(u[-1]) - 3 * sigma) < 1e-12
        assert abs(float(w[0]) - float(w[-1])) < 1e-15


def test_line_outside_feature_support_is_excluded_with_status():
    feats, reader = _features(), _reader()
    off_page = _line(x0=490.0, x1=530.0)   # right end beyond the 512-px tile
    loss, status = expected_ctc_loss(feats, off_page, reader, LABELS, sigma_px=0.0)
    assert loss is None
    assert status == "excluded_outside_support"


def test_infeasible_target_length_is_excluded_with_status():
    feats, reader = _features(), _reader()
    short = _line(x0=40.0, x1=100.0)      # ~60 px -> T = 15 frames
    labels = [1] * 20                     # 20 + 19 repeats = 39 > 15
    loss, status = expected_ctc_loss(feats, short, reader, labels, sigma_px=0.0)
    assert loss is None
    assert status == "excluded_infeasible_length"


def test_expected_loss_gradients_reach_reader_and_baseline():
    feats, reader = _features(), _reader()
    line = _line(requires_grad=True)
    loss, status = expected_ctc_loss(feats, line, reader, LABELS, sigma_px=2.0)
    assert status == "ok"
    loss.backward()
    assert reader.out.weight.grad is not None
    assert float(reader.out.weight.grad.abs().sum()) > 0
    assert line["baseline"].grad is not None
    assert float(line["baseline"].grad.abs().sum()) > 0
