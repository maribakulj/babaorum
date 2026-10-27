"""Stage S reader (step S-L4a): band features, a small CTC reader, expected CTC loss.

CPU, untrained, synthetic-probe scope. Baselines are continuous page-edge
coordinates [K,2]; lines are dicts {"baseline": tensor [K,2], "neg": px, "pos": px}.
Exclusions are returned as explicit statuses, never as zero losses or exceptions.
"""

from __future__ import annotations


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from babaorum.experimental.joint_geometry import FeatureFrame, sample_baseline_bands
from babaorum.experimental.joint_reader_bridge import contract_band_to_sampler_geometry

STRIDE = 4
ORIGIN = (1.5, 1.5)


def _offset_baseline(base: torch.Tensor, offset_px: float) -> torch.Tensor:
    """Rigid shift of the baseline along the chord normal n = (-t_y, t_x)."""
    chord = base[-1] - base[0]
    t = chord / torch.linalg.norm(chord)
    n = torch.stack((-t[1], t[0]))
    return base + offset_px * n[None, :]


def band_features(features, baseline_edges, neg, pos, *, rows: int = 8,
                  px_per_frame: float = 4.0, offset_px: float = 0.0):
    """[T, C*rows] band features, or None when the band leaves the feature support."""
    _, channels, h, w = features.shape
    base = baseline_edges
    arc = float(torch.linalg.norm(base[1:] - base[:-1], dim=-1).detach().sum())
    steps = max(4, round(arc / px_per_frame))
    shifted = _offset_baseline(base, offset_px) if offset_px != 0.0 else base

    dtype = base.dtype
    neg_t = torch.as_tensor(neg, dtype=dtype).reshape(1)
    pos_t = torch.as_tensor(pos, dtype=dtype).reshape(1)
    center, total_h, fraction = contract_band_to_sampler_geometry(shifted[None], neg_t, pos_t)
    frame = FeatureFrame(original_size_wh=(w * STRIDE, h * STRIDE), origin_xy=ORIGIN,
                         stride_xy=(float(STRIDE), float(STRIDE)))
    try:
        band = sample_baseline_bands(features, center, total_h, frame,
                                     output_size=(rows, steps), baseline_fraction=fraction)
    except ValueError:
        return None
    # band.features is [1, C, rows, T] -> [T, C*rows]
    return band.features[0].permute(2, 0, 1).reshape(steps, channels * rows)


class Reader(nn.Module):
    def __init__(self, in_dim: int, alphabet_size: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, hidden, 3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, alphabet_size)

    def forward(self, x):
        """[T, in_dim] -> logits [T, A]."""
        h = F.relu(self.conv(x.t()[None]))
        h, _ = self.lstm(h.transpose(1, 2))
        return self.out(h)[0]


def offset_nodes(sigma_px: float, nodes: int = 9):
    """Offsets and renormalised trapezoidal weights of N(0, sigma^2) on [-3s, 3s]."""
    u = np.linspace(-3.0 * sigma_px, 3.0 * sigma_px, nodes)
    step = u[1] - u[0]
    tw = step * np.exp(-u * u / (2.0 * sigma_px * sigma_px))
    tw[0] *= 0.5
    tw[-1] *= 0.5
    return u, tw / tw.sum()


def _ctc_nll(logits: torch.Tensor, labels) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=-1)[:, None, :]
    device = logits.device
    target = torch.as_tensor(labels, dtype=torch.long, device=device)
    return F.ctc_loss(log_probs, target, torch.tensor([logits.shape[0]], device=device),
                      torch.tensor([len(labels)], device=device), blank=0, reduction="sum",
                      zero_infinity=False)


def _feasible(steps: int, labels) -> bool:
    repeats = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    return steps >= len(labels) + repeats


def expected_ctc_loss(features, line: dict, reader: Reader, labels, *, sigma_px: float,
                      nodes: int = 9):
    """(loss or None, status). sigma_px == 0 is the point CTC at offset 0."""
    if sigma_px == 0:
        offsets, weights = [0.0], [1.0]
    else:
        u, w = offset_nodes(sigma_px, nodes)
        offsets, weights = [float(x) for x in u], [float(x) for x in w]

    bands = []
    for u_j in offsets:
        band = band_features(features, line["baseline"], line["neg"], line["pos"], offset_px=u_j)
        if band is None:
            return None, "excluded_outside_support"
        bands.append(band)
    steps = bands[0].shape[0]
    if not _feasible(steps, labels):
        return None, "excluded_infeasible_length"

    loss = sum(w_j * _ctc_nll(reader(band), labels) for w_j, band in zip(weights, bands))
    return loss, "ok"
