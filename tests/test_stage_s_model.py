"""Checks for the stage S model (untrained, CPU) on random tensors and a 512x512 tile of mf_0000."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from babaorum.experimental.stage_s_model import (
    Encoder, HeadD, HeadG, feature_indices, refine_lines, resample_k,
)
from babaorum.experimental.joint_geometry import FeatureFrame
from babaorum.experimental.joint_reader_bridge import page_edges_to_feature_indices

K = 8
CHANNELS = 64
DATA = Path("/Users/marcel/heritage-synth/runs/accept-layout-v2/reproduced")
PAGE_JSON = DATA / "pages/mf_0000.json"
IMAGE = DATA / "images/mf_0000.png"
POLY = [[[40.0, 100.0], [300.0, 110.0]], [[60.0, 200.0], [260.0, 205.0]]]


def _models(dtype=torch.float32):
    torch.manual_seed(0)
    enc = Encoder(CHANNELS).to(dtype)
    hd = HeadD(CHANNELS).to(dtype)
    hg = HeadG(CHANNELS, K).to(dtype)
    return enc, hd, hg


def test_shapes_on_mf_0000_tile():
    if not (PAGE_JSON.exists() and IMAGE.exists()):
        pytest.skip(f"canonical page not found under {DATA}")
    from PIL import Image
    tile = np.asarray(Image.open(IMAGE), dtype=np.float32)[:512, :512] / 255.0
    image = torch.as_tensor(tile)[None, None]
    enc, hd, hg = _models()
    features = enc(image)
    assert tuple(features.shape) == (1, CHANNELS, 128, 128)
    logit, neg, pos = hd(features)
    assert tuple(logit.shape) == tuple(neg.shape) == tuple(pos.shape) == (1, 1, 128, 128)
    out = refine_lines(features, POLY, hg, head_d=hd, K=K)
    assert tuple(out["baseline"].shape) == (2, K, 2)
    assert tuple(out["neg"].shape) == tuple(out["pos"].shape) == (2,)
    assert tuple(out["delta"].shape) == (2, K)


def test_head_g_receives_gradient_from_refined_baseline():
    enc, hd, hg = _models()
    features = torch.randn(1, CHANNELS, 128, 128)
    out = refine_lines(features, POLY, hg, head_d=hd, K=K)
    out["baseline"].sum().backward()
    grad = hg.mlp[-1].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_zero_offsets_return_resampled_polyline():
    enc, hd, hg = _models(torch.float64)
    with torch.no_grad():
        for p in hg.parameters():
            p.zero_()
    features = torch.randn(1, CHANNELS, 128, 128, dtype=torch.float64)
    out = refine_lines(features, POLY, hg, head_d=hd.double(), K=K)
    expected = np.stack([resample_k(p, K) for p in POLY])
    assert float(out["delta"].detach().abs().max()) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(out["baseline"].detach().numpy(), expected, atol=1e-6)


def test_index_conversion_matches_bridge_and_cell_centres_map_to_cells():
    frame = FeatureFrame(original_size_wh=(512, 512), origin_xy=(1.5, 1.5), stride_xy=(4.0, 4.0))
    points = torch.tensor([[[10.0, 20.0], [100.5, 36.0], [509.0, 500.0]]], dtype=torch.float64)
    mine, mine_valid = feature_indices(points, frame, (128, 128))
    ref, ref_valid = page_edges_to_feature_indices(points, frame, (128, 128))
    assert torch.equal(mine, ref) and torch.equal(mine_valid, ref_valid)
    cells = torch.tensor([[[3, 7], [0, 127], [60, 2]]], dtype=torch.float64)
    centres = (cells + 0.5) * 4.0
    idx, valid = feature_indices(centres, frame, (128, 128))
    assert bool(valid.all())
    assert torch.allclose(idx, cells, atol=1e-9)
