"""Stage S model (step S-L3): encoder, head D (cell map) and geometry refinement head G.

Untrained, CPU, synthetic-probe scope. Feature cell c has its centre at page edge
coordinate (c + 0.5) * 4, which is FeatureFrame(origin=(1.5, 1.5), stride=(4, 4))
in pixel-centre coordinates, as in joint_geometry.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from babaorum.experimental.joint_geometry import FeatureFrame
from babaorum.experimental.joint_reader_bridge import page_edges_to_feature_indices

STRIDE = 4
ORIGIN = (1.5, 1.5)


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.relu(x + self.body(x))


class Encoder(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, stride=2, padding=1), nn.GroupNorm(8, channels), nn.ReLU(inplace=True),
            _ResBlock(channels), _ResBlock(channels),
        )

    def forward(self, image):
        return self.net(image)


class HeadD(nn.Module):
    """1x1 conv to logit(baseline), neg_px, pos_px (softplus on the two distances)."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(channels, 3, 1)

    def forward(self, features):
        out = self.conv(features)
        return out[:, 0:1], F.softplus(out[:, 1:2]), F.softplus(out[:, 2:3])


class HeadG(nn.Module):
    """MLP from K sampled feature vectors (channels*K inputs) to K offsets + 2 distance corrections."""

    def __init__(self, channels: int = 64, K: int = 8, hidden: int = 128):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(nn.Linear(channels * K, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, K + 2))

    def forward(self, x):
        return self.mlp(x)


def resample_k(points_xy, K: int) -> np.ndarray:
    """K points at equal arc length along a polyline, both endpoints included."""
    pts = np.asarray(points_xy, dtype=np.float64)
    seg = np.hypot(*(pts[1:] - pts[:-1]).T)
    if np.any(seg == 0):
        raise ValueError("polyline contains a zero-length segment")
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    targets = np.linspace(0.0, cum[-1], K)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack((x, y), axis=-1)


def feature_indices(points_edge_xy: torch.Tensor, frame: FeatureFrame, feature_hw):
    """Thin wrapper over page_edges_to_feature_indices (bords -> centres handled there)."""
    return page_edges_to_feature_indices(points_edge_xy, frame, feature_hw)


def _sample(maps: torch.Tensor, feature_xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of [1,C,h,w] maps at [N,K,2] feature indices -> [N,C,K], zero off-support."""
    _, channels, h, w = maps.shape
    n, k, _ = feature_xy.shape
    ix = feature_xy[..., 0].clamp(0.0, w - 1.0)
    iy = feature_xy[..., 1].clamp(0.0, h - 1.0)
    gx = (2.0 * ix + 1.0) / w - 1.0
    gy = (2.0 * iy + 1.0) / h - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(1, n * k, 1, 2)
    sampled = F.grid_sample(maps, grid, mode="bilinear", align_corners=False)
    sampled = sampled.reshape(channels, n, k).permute(1, 0, 2)
    return sampled * valid[:, None, :].to(sampled.dtype)


def refine_lines(features, polylines_page_edges, head_g: HeadG, *, head_d: HeadD, K: int = 8,
                 page_size_wh=None) -> dict:
    """Refine each polyline with the geometry head; all coordinates in continuous page edges."""
    _, channels, h, w = features.shape
    if page_size_wh is None:
        page_size_wh = (w * STRIDE, h * STRIDE)
    frame = FeatureFrame(original_size_wh=tuple(page_size_wh), origin_xy=ORIGIN,
                         stride_xy=(float(STRIDE), float(STRIDE)))
    dtype = features.dtype

    pts = np.stack([resample_k(p, K) for p in polylines_page_edges])          # N,K,2
    pts_t = torch.as_tensor(pts, dtype=dtype, device=features.device)
    chord = pts_t[:, -1] - pts_t[:, 0]
    t = chord / torch.linalg.norm(chord, dim=-1, keepdim=True)
    normal = torch.stack((-t[:, 1], t[:, 0]), dim=-1)                         # N,2

    feature_xy, valid = feature_indices(pts_t, frame, (h, w))
    out_of_support = int((~valid).sum())

    sampled = _sample(features, feature_xy, valid)                            # N,C,K
    _, d_neg_map, d_pos_map = head_d(features)
    neg_at = _sample(d_neg_map, feature_xy, valid)[:, 0]                      # N,K
    pos_at = _sample(d_pos_map, feature_xy, valid)[:, 0]
    count = valid.sum(dim=1).clamp(min=1).to(dtype)
    base_neg = (neg_at * valid).sum(dim=1) / count
    base_pos = (pos_at * valid).sum(dim=1) / count

    out = head_g(sampled.reshape(sampled.shape[0], channels * K))             # N,K+2
    delta = out[:, :K]
    baseline = pts_t + delta[..., None] * normal[:, None, :]
    neg = F.softplus(base_neg + out[:, K])
    pos = F.softplus(base_pos + out[:, K + 1])
    return {"baseline": baseline, "neg": neg, "pos": pos, "delta": delta,
            "out_of_support": out_of_support}
