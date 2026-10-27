"""Tiny differentiable polyline operators; no OCR performance is implied.

Coordinates are original encoded-page pixel centers, as in ``joint_geometry``.
These primitives handle one line at a time so matched targets may have distinct
numbers of vertices. Batched callers can iterate over the matched line pairs.

Arc-length interpolation and segment distances are differentiable only within
a fixed choice of sample count, interpolation segment and nearest segment.
Those choices are discrete. No soft minimum or detached interpolation weights
are used. Finite-difference checks must avoid their transition boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from .joint_geometry import require_torch

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class CurveSamplingCapError(ValueError):
    """Valid sampling arguments would require more than the declared budget."""


class CurveNumericalError(ValueError):
    """A geometry calculation overflowed or produced a nonfinite result.

    Invalid arguments, nonfinite input coordinates and malformed GT keep their
    ordinary ValueError contract. Callers can exclude numerical computation
    failures without matching exception text or swallowing input mistakes.
    """


@dataclass
class ArcSamples:
    points_xy: Any  # M,2; original pixel coordinates, including both endpoints
    length_px: Any  # scalar tensor; retains prediction gradients
    segment_index: Any  # M integer indices into the input polyline's segments


@dataclass
class CurveGeometryLoss:
    total: Any
    curve: Any
    endpoints: Any
    height: Any
    pred_to_target: Any
    target_to_pred: Any


def _finite(tensor: Any, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _finite_computed(tensor: Any, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise CurveNumericalError(f"{name} must contain only finite values")


def _points(points_xy: Any, name: str, *, minimum: int = 2) -> None:
    if (
        not isinstance(points_xy, torch.Tensor)
        or points_xy.ndim != 2
        or points_xy.shape[0] < minimum
        or points_xy.shape[1] != 2
    ):
        raise ValueError(f"{name} must have shape [K>={minimum},2]")
    if points_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    _finite(points_xy, name)


def _segments(points_xy: Any, name: str) -> tuple[Any, Any]:
    differences = points_xy[1:] - points_xy[:-1]
    lengths = torch.linalg.vector_norm(differences, dim=-1)
    _finite_computed(lengths, f"{name} segment lengths")
    if bool((lengths <= 0).any()):
        raise ValueError(f"{name} must not contain zero-length consecutive segments")
    return differences, lengths


def _number(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def _height(value: Any, reference: Any, name: str, *, detach: bool) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one height")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must share the prediction dtype and device")
        result = value.reshape(())
        if detach:
            result = result.detach()
    else:
        result = reference.new_tensor(_number(value, name, positive=True))
    _finite(result, name)
    if bool(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def resample_arclength(
    points_xy: Any,
    *,
    spacing_px: float,
    max_samples: int = 256,
) -> ArcSamples:
    """Interpolate a nondegenerate polyline at uniform arc-length positions.

    ``M = ceil(length / spacing_px) + 1``, with both endpoints included. Near an
    integer, the ratio is snapped within ``8 * dtype_eps * max(1, abs(ratio))``
    to prevent roundoff from adding a point after collinear densification. Thus
    the maximum spacing can exceed the request only by this roundoff tolerance.
    ``max_samples`` is a rejection limit, never an implicit coarser spacing.

    The sample count and searchsorted indices are detached discrete decisions;
    segment lengths, cumulative lengths and interpolation fractions retain
    autograd. Consecutive coincident predicted vertices are rejected. For static
    GT, ``curve_geometry_loss`` first removes exact consecutive duplicates.
    """
    require_torch()
    _points(points_xy, "points_xy")
    spacing = _number(spacing_px, "spacing_px", positive=True)
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
        raise ValueError("max_samples must be an integer >= 2")
    differences, lengths = _segments(points_xy, "points_xy")
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(dim=0)))
    total_length = cumulative[-1]
    _finite_computed(total_length, "polyline length")
    ratio = float(total_length.detach()) / spacing
    if not math.isfinite(ratio):
        raise CurveSamplingCapError("spacing_px requires more than max_samples samples")
    tolerance = 8 * torch.finfo(points_xy.dtype).eps * max(1.0, abs(ratio))
    nearest_integer = round(ratio)
    if abs(ratio - nearest_integer) <= tolerance:
        ratio = float(nearest_integer)
    count = max(2, math.ceil(ratio) + 1)
    if count > max_samples:
        raise CurveSamplingCapError(f"polyline requires {count} samples, exceeding max_samples={max_samples}")

    fractions = torch.linspace(0, 1, count, dtype=points_xy.dtype, device=points_xy.device)
    distances = fractions * total_length
    segment_index = (
        torch.searchsorted(cumulative.detach(), distances.detach(), right=True) - 1
    ).clamp(min=0, max=lengths.numel() - 1)
    alpha = (distances - cumulative[segment_index]) / lengths[segment_index]
    samples = points_xy[segment_index] + alpha[:, None] * differences[segment_index]
    # Exact endpoints also avoid length-dependent roundoff at the last vertex.
    samples = torch.cat((points_xy[:1], samples[1:-1], points_xy[-1:]), dim=0)
    _finite_computed(samples, "arc-length samples")
    return ArcSamples(samples, total_length, segment_index)


def point_to_polyline_distances(points_xy: Any, polyline_xy: Any) -> Any:
    """Euclidean distance from each point to the nearest bounded segment.

    The interior projection measures normal distance; endpoint projection also
    captures extension beyond the curve. Original segments are used, so corners
    missed by a finite sampling grid are not replaced by shortcut segments.
    Ties choose the first nearest segment. Projection bounds and changes of the
    nearest segment are nonsmooth. The norm has its zero subgradient at zero.
    """
    require_torch()
    _points(points_xy, "points_xy", minimum=1)
    _points(polyline_xy, "polyline_xy")
    if points_xy.dtype != polyline_xy.dtype or points_xy.device != polyline_xy.device:
        raise ValueError("points and polyline must share a dtype and device")
    differences, lengths = _segments(polyline_xy, "polyline_xy")
    tangents = differences / lengths[:, None]
    offsets = points_xy[:, None, :] - polyline_xy[None, :-1, :]
    along = (offsets * tangents[None, :, :]).sum(dim=-1)
    bounded = torch.minimum(along.clamp_min(0), lengths[None, :])
    projection = polyline_xy[None, :-1, :] + bounded[..., None] * tangents[None, :, :]
    residual = points_xy[:, None, :] - projection
    distances = torch.linalg.vector_norm(residual, dim=-1).min(dim=1).values
    _finite_computed(distances, "point-to-polyline distances")
    return distances


def curve_geometry_loss(
    predicted_xy: Any,
    target_xy: Any,
    *,
    target_height_px: Any,
    predicted_height_px: Any = None,
    spacing_px: float,
    max_samples: int = 256,
    curve_weight: float = 1.0,
    endpoint_weight: float = 1.0,
    height_weight: float = 1.0,
    orientation: str = "ltr",
) -> CurveGeometryLoss:
    """Matched variable-K curve supervision, normalized by detached GT height.

    Curve loss averages both directed mean point-to-original-segment distances.
    Endpoints are ordered start/start and end/end; reversing the prediction is
    penalized even when both curves cover the same point set. ``orientation``
    checks the GT endpoint direction on x (ltr/rtl) or y (ttb/btt); unknown is
    refused. Local backtracking and band folds require separate validity checks.

    Heights and endpoint distances use absolute Euclidean error divided by GT
    height, without Smooth-L1. A normally translated straight line therefore has
    ``curve == abs(translation) / target_height``. The total also includes the
    endpoint term. Missing predicted height gives a zero height component.

    The fixed pixel spacing and cap apply to both curves. To preserve unit/scale
    invariance, scale spacing together with coordinates and heights. This finite
    quadrature is not a Hausdorff bound or a global curve-integrity certificate.
    GT coordinates and GT height are always detached; exact adjacent GT vertex
    duplicates are removed, and zero-length targets are rejected.
    """
    require_torch()
    _points(predicted_xy, "predicted_xy")
    _points(target_xy, "target_xy")
    if predicted_xy.dtype != target_xy.dtype or predicted_xy.device != target_xy.device:
        raise ValueError("predicted and target curves must share a dtype and device")
    weights = (
        _number(curve_weight, "curve_weight", positive=False),
        _number(endpoint_weight, "endpoint_weight", positive=False),
        _number(height_weight, "height_weight", positive=False),
    )
    if orientation not in ("ltr", "rtl", "ttb", "btt"):
        raise ValueError("orientation must be ltr, rtl, ttb or btt; unknown is unsupported")

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

    target_height = _height(target_height_px, predicted_xy, "target_height_px", detach=True)
    predicted = resample_arclength(predicted_xy, spacing_px=spacing_px, max_samples=max_samples)
    reference = resample_arclength(target, spacing_px=spacing_px, max_samples=max_samples)
    pred_to_target = point_to_polyline_distances(predicted.points_xy, target).mean() / target_height
    target_to_pred = point_to_polyline_distances(reference.points_xy, predicted_xy).mean() / target_height
    curve = (pred_to_target + target_to_pred) * 0.5
    endpoint_residual = torch.stack((predicted_xy[0] - target[0], predicted_xy[-1] - target[-1]))
    endpoints = torch.linalg.vector_norm(endpoint_residual, dim=-1).mean() / target_height
    if predicted_height_px is None:
        height = predicted_xy.new_zeros(())
    else:
        predicted_height = _height(predicted_height_px, predicted_xy, "predicted_height_px", detach=False)
        height = (predicted_height - target_height).abs() / target_height
    total = weights[0] * curve + weights[1] * endpoints + weights[2] * height
    _finite_computed(total, "curve geometry loss")
    return CurveGeometryLoss(total, curve, endpoints, height, pred_to_target, target_to_pred)
