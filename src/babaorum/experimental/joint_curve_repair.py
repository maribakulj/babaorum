"""Explicit supervised repair of degenerate predicted polylines.

This conditional surrogate assigns fixed arc-length anchors to control indices
only while a prediction contains a segment within an explicit numerical cutoff.
It must not replace the
parameterization-invariant geometry loss for valid curves. No jitter, inferred
transcription or silent prediction deduplication is used. Repair does not make a
query eligible for CTC until its next geometry validation succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .joint_curves import _finite_computed, _height, _number, _points, _segments
from .joint_geometry import require_torch, torch


@dataclass
class DegenerateCurveRepairLoss:
    total: Any
    anchors: Any
    endpoints: Any
    height: Any
    target_anchors_xy: Any
    reason: str


def repair_degenerate_curve_loss(
    predicted_xy: Any,
    target_xy: Any,
    *,
    target_height_px: Any,
    predicted_height_px: Any = None,
    orientation: str = "ltr",
    min_segment_length_px: float = 0.0,
) -> DegenerateCurveRepairLoss:
    """Repair predictions containing a segment at or below a numerical cutoff.

    Ground truth is detached, exact consecutive duplicate vertices are removed,
    and its ordered endpoints must agree with ltr/rtl/ttb/btt. The prediction is
    never deduplicated. A prediction above the cutoff is rejected to prevent an
    unnoticed switch from curve-set supervision to control-index supervision.

    For K predicted controls, exactly K GT anchors are evaluated at arc-length
    fractions j/(K-1) on the ORIGINAL GT segments. This is not interpolation of
    an intermediate sampling grid. The anchor term is the mean Euclidean error
    of every control; the endpoint term pairs start/start and end/end. Optional
    height error is absolute. All three are divided by detached GT height and
    added with weight one, so endpoints also participate in the anchor term.

    ``min_segment_length_px`` must be finite and nonnegative. Its default zero
    repairs exactly collapsed segments; a positive value defines an explicit
    numerical admissibility domain, not a physical GT threshold. Extremely
    dense collinear insertion below this cutoff can change repair activation,
    so invariance to arbitrary densification is not claimed for that branch.

    The segment-cutoff condition is discrete. This surrogate deliberately fixes
    control-point placement only on that invalid branch; it is not a smooth
    extension of the valid-curve loss or a geometric-integrity certificate.
    Finite gradients reach predicted controls and optional predicted height;
    no gradient reaches GT coordinates, GT height or the returned GT anchors.
    """
    require_torch()
    _points(predicted_xy, "predicted_xy")
    _points(target_xy, "target_xy")
    if predicted_xy.dtype != target_xy.dtype or predicted_xy.device != target_xy.device:
        raise ValueError("predicted and target curves must share a dtype and device")
    if orientation not in ("ltr", "rtl", "ttb", "btt"):
        raise ValueError("orientation must be ltr, rtl, ttb or btt; unknown is unsupported")
    cutoff = _number(min_segment_length_px, "min_segment_length_px", positive=False)

    predicted_lengths = torch.linalg.vector_norm(predicted_xy[1:] - predicted_xy[:-1], dim=-1)
    _finite_computed(predicted_lengths, "predicted segment lengths")
    if not bool((predicted_lengths <= cutoff).any()):
        raise ValueError("repair requires a degenerate prediction with a segment at or below min_segment_length_px")

    target = target_xy.detach()
    keep = torch.cat((
        torch.ones(1, dtype=torch.bool, device=target.device),
        (target[1:] != target[:-1]).any(dim=-1),
    ))
    target = target[keep]
    if target.shape[0] < 2:
        raise ValueError("target curve must have positive length after deduplication")
    axis = 0 if orientation in ("ltr", "rtl") else 1
    sign = 1 if orientation in ("ltr", "ttb") else -1
    if bool(sign * (target[-1, axis] - target[0, axis]) <= 0):
        raise ValueError("target endpoint direction disagrees with orientation")

    differences, lengths = _segments(target, "target_xy")
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(dim=0)))
    _finite_computed(cumulative, "target cumulative arc length")
    fractions = torch.linspace(
        0, 1, len(predicted_xy), dtype=target.dtype, device=target.device,
    )
    positions = fractions * cumulative[-1]
    indices = (torch.searchsorted(cumulative, positions, right=True) - 1).clamp(
        min=0, max=len(lengths) - 1,
    )
    alpha = (positions - cumulative[indices]) / lengths[indices]
    target_anchors = target[indices] + alpha[:, None] * differences[indices]
    target_anchors = torch.cat((target[:1], target_anchors[1:-1], target[-1:]), dim=0)
    _finite_computed(target_anchors, "target repair anchors")

    target_height = _height(target_height_px, predicted_xy, "target_height_px", detach=True)
    anchors = torch.linalg.vector_norm(predicted_xy - target_anchors, dim=-1).mean() / target_height
    endpoint_residual = torch.stack((predicted_xy[0] - target[0], predicted_xy[-1] - target[-1]))
    endpoints = torch.linalg.vector_norm(endpoint_residual, dim=-1).mean() / target_height
    if predicted_height_px is None:
        height = predicted_xy.new_zeros(())
    else:
        predicted_height = _height(predicted_height_px, predicted_xy, "predicted_height_px", detach=False)
        height = (predicted_height - target_height).abs() / target_height
    total = anchors + endpoints + height
    _finite_computed(total, "degenerate curve repair loss")
    return DegenerateCurveRepairLoss(
        total=total, anchors=anchors, endpoints=endpoints, height=height,
        target_anchors_xy=target_anchors,
        reason=("near_degenerate_predicted_segment" if cutoff > 0 else "zero_length_predicted_segment"),
    )
