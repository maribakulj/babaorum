"""Geometry-only bridge primitives for the proposed line reader interfaces.

This module deliberately stops before a real encoder/reader call. Page graph
coordinates use continuous pixel edges; the experimental feature sampler uses
pixel centers and an explicit FeatureFrame. Angular M1 perturbations follow
the chord-normal convention agreed with reader w2:p9. No OCR claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .joint_geometry import FeatureFrame, require_torch, torch

M1_MAX_EXCLUSION_FRACTION = 0.20
M1_SIGNED_ANGLE_DELTAS = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class AngularPerturbation:
    """One rigid angular perturbation, or a counted infeasible request."""

    status: str  # included | excluded_infeasible_chord | excluded_angle_limit | excluded_degenerate | excluded_wrong_orientation
    baseline_xy: Any | None
    theta_radians: Any | None
    theta_degrees: Any | None
    chord_length_px: Any
    start_chord_normal_displacement_px: Any | None
    end_chord_normal_displacement_px: Any | None
    chord_tangential_slip_px: Any | None
    interior_local_normal_min_h: Any | None
    interior_local_normal_max_h: Any | None


@dataclass(frozen=True)
class DeltaSupport:
    """Predeclared support summary for one angular-offset magnitude."""

    delta: float
    attempted: int
    included: int
    excluded: int
    exclusion_fraction: float
    status: str  # supported | under_supported
    exclusion_counts: tuple[tuple[str, int], ...]


def summarize_delta_support(
    attempts: Iterable[tuple[float, str]],
) -> tuple[DeltaSupport, ...]:
    """Aggregate M1 angular statuses by signed ``delta`` deterministically.

    The cutoff is strict: exactly 20% excluded remains supported; greater
    than 20% is ``under_supported`` and must not receive an M1 verdict. Every
    attempted perturbation contributes to its denominator, including
    infeasible, angle-limited, degenerate, or wrongly oriented cases.
    """
    cutoff = M1_MAX_EXCLUSION_FRACTION
    known_statuses = {
        "included", "excluded_infeasible_chord", "excluded_angle_limit",
        "excluded_degenerate", "excluded_wrong_orientation",
    }
    grouped: dict[float, list[str]] = {}
    for delta, status in attempts:
        value = float(delta)
        if not math.isfinite(value):
            raise ValueError("delta values must be finite")
        if value not in M1_SIGNED_ANGLE_DELTAS:
            raise ValueError("delta must be an exact member of the predeclared M1 grid")
        if status not in known_statuses:
            raise ValueError(f"unknown angular status: {status}")
        grouped.setdefault(value, []).append(status)
    if not grouped:
        raise ValueError("at least one angular attempt is required")

    summaries = []
    for delta in sorted(grouped):
        statuses = grouped[delta]
        excluded_counts: dict[str, int] = {}
        for status in statuses:
            if status != "included":
                excluded_counts[status] = excluded_counts.get(status, 0) + 1
        excluded = sum(excluded_counts.values())
        fraction = excluded / len(statuses)
        summaries.append(DeltaSupport(
            delta=delta,
            attempted=len(statuses),
            included=len(statuses) - excluded,
            excluded=excluded,
            exclusion_fraction=fraction,
            status="under_supported" if fraction > cutoff else "supported",
            exclusion_counts=tuple(sorted(excluded_counts.items())),
        ))
    return tuple(summaries)


def page_edges_to_feature_indices(
    page_edge_xy: Any,
    frame: FeatureFrame,
    feature_size_hw: tuple[int, int],
) -> tuple[Any, Any]:
    """Map page-edge coordinates to feature indices without clamp.

    Returns ``(feature_xy, valid)`` with the input leading dimensions. Validity
    requires both the source pixel center and the feature-map center to lie in
    support. Out-of-support coordinates are retained unmodified and marked
    false; callers must not use them for sampling or silently clamp them.
    """
    require_torch()
    frame.validate()
    if not isinstance(page_edge_xy, torch.Tensor) or page_edge_xy.ndim < 1 or page_edge_xy.shape[-1] != 2:
        raise ValueError("page_edge_xy must be a tensor with final dimension 2")
    if page_edge_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError("page_edge_xy must use float32 or float64")
    if not bool(torch.isfinite(page_edge_xy).all()):
        raise ValueError("page_edge_xy must be finite")
    if (not isinstance(feature_size_hw, (tuple, list)) or len(feature_size_hw) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in feature_size_hw)):
        raise ValueError("feature_size_hw must contain integer height,width >=2")

    height_f, width_f = feature_size_hw
    width_page, height_page = frame.original_size_wh
    center_xy = page_edge_xy - page_edge_xy.new_tensor((0.5, 0.5))
    origin_xy = page_edge_xy.new_tensor(frame.origin_xy)
    stride_xy = page_edge_xy.new_tensor(frame.stride_xy)
    feature_xy = (center_xy - origin_xy) / stride_xy
    valid = (
        (center_xy[..., 0] >= 0.0)
        & (center_xy[..., 0] <= width_page - 1.0)
        & (center_xy[..., 1] >= 0.0)
        & (center_xy[..., 1] <= height_page - 1.0)
        & (feature_xy[..., 0] >= 0.0)
        & (feature_xy[..., 0] <= width_f - 1.0)
        & (feature_xy[..., 1] >= 0.0)
        & (feature_xy[..., 1] <= height_f - 1.0)
    )
    return feature_xy, valid


def contract_band_to_sampler_geometry(
    baseline_edge_xy: Any,
    normal_negative_px: float | Any,
    normal_positive_px: float | Any,
) -> tuple[Any, Any, Any]:
    """Convert a page-contract LTR band to center-frame sampler geometry.

    ``baseline_edge_xy`` is [N,K>=2,2] in continuous page-edge coordinates;
    the two normal distances are scalars or [N] tensors in pixels. The return
    is ``(baseline_center_xy, total_height_px, baseline_fraction)``. Only point
    coordinates shift by -0.5; distances are unchanged. Tensor distances keep
    their autograd paths through both total height and the per-line fraction.
    This helper does not sample pixels or clamp unsupported bands; the sampler
    still performs its strict page/feature-support check. Orientation support
    is LTR by endpoint chord and no control points are reordered.
    """
    require_torch()
    if not isinstance(baseline_edge_xy, torch.Tensor) or baseline_edge_xy.ndim != 3:
        raise ValueError("baseline_edge_xy must be a tensor [N,K>=2,2]")
    if baseline_edge_xy.shape[0] < 1 or baseline_edge_xy.shape[1] < 2 or baseline_edge_xy.shape[2] != 2:
        raise ValueError("baseline_edge_xy must have shape [N,K>=2,2]")
    if baseline_edge_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError("baseline_edge_xy must use float32 or float64")
    if not bool(torch.isfinite(baseline_edge_xy).all()):
        raise ValueError("baseline_edge_xy must be finite")
    if bool((baseline_edge_xy[:, -1, 0] <= baseline_edge_xy[:, 0, 0]).any()):
        raise ValueError("only left-to-right endpoint chords are supported")
    n = baseline_edge_xy.shape[0]
    converted = []
    for value, name in ((normal_negative_px, "normal_negative_px"),
                        (normal_positive_px, "normal_positive_px")):
        try:
            tensor = torch.as_tensor(value, device=baseline_edge_xy.device, dtype=baseline_edge_xy.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite scalar or [N] values") from exc
        if tensor.ndim == 0:
            tensor = tensor.expand(n)
        if tensor.shape != (n,) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite scalar or [N] values")
        if bool((tensor < 0).any()):
            raise ValueError(f"{name} must be nonnegative")
        converted.append(tensor)
    negative, positive = converted
    height = negative + positive
    if bool((height <= 0).any()):
        raise ValueError("the total band extension must be positive")
    fraction = negative / height
    baseline_center_xy = baseline_edge_xy - baseline_edge_xy.new_tensor((0.5, 0.5))
    return baseline_center_xy, height, fraction


def rotate_baseline_about_chord(
    baseline_xy: Any,
    delta: float | Any,
    height_px: float,
    *,
    max_angle_deg: float,
) -> AngularPerturbation:
    """Rotate an ordered polyline about its chord midpoint for M1.

    ``delta`` is dimensionless and signed. For a chord of length C, the rigid
    rotation angle is ``asin(2 * delta * height_px / C)``. The caller must
    supply a predeclared ``max_angle_deg`` strictly between 0 and 90; requests
    above that limit return a counted exclusion, including the 90-degree
    equality case ``2*abs(delta)*h == C``. Thus the endpoint
    displacements projected on the chord normal are exactly ``-delta*h`` at
    the start and ``+delta*h`` at the end. Positive theta uses the standard
    rotation matrix in image coordinates (x right, y down). Requests with
    ``2*abs(delta)*h > C`` return an explicit exclusion, never a clamped angle.

    For curved polylines, the same rigid rotation is applied to all vertices.
    The chord-normal endpoint displacements are exact; interior local-normal
    displacements are reported as min/max relative to centered secant tangents.
    A zero chord or undefined interior tangent returns ``excluded_degenerate``.
    """
    require_torch()
    if not isinstance(baseline_xy, torch.Tensor) or baseline_xy.ndim != 2 or baseline_xy.shape[1] != 2 or baseline_xy.shape[0] < 2:
        raise ValueError("baseline_xy must have shape [K>=2,2]")
    if baseline_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError("baseline_xy must use float32 or float64")
    if not bool(torch.isfinite(baseline_xy).all()):
        raise ValueError("baseline_xy must be finite")
    height = float(height_px)
    if not math.isfinite(height) or height <= 0:
        raise ValueError("height_px must be finite and positive")
    angle_limit = float(max_angle_deg)
    if not math.isfinite(angle_limit) or not 0.0 < angle_limit < 90.0:
        raise ValueError("max_angle_deg must be finite and strictly between 0 and 90")
    delta_t = torch.as_tensor(delta, dtype=baseline_xy.dtype, device=baseline_xy.device)
    if delta_t.numel() != 1 or not bool(torch.isfinite(delta_t).all()):
        raise ValueError("delta must be a finite scalar")

    chord = baseline_xy[-1] - baseline_xy[0]
    chord_length = torch.linalg.vector_norm(chord)
    if not bool(torch.isfinite(chord_length)) or float(chord_length.detach()) <= 0:
        return AngularPerturbation(
            "excluded_degenerate", None, None, None, chord_length,
            None, None, None, None, None,
        )
    if float(chord[0].detach()) <= 0:
        return AngularPerturbation(
            "excluded_wrong_orientation", None, None, None, chord_length,
            None, None, None, None, None,
        )
    half_sine = 2.0 * delta_t * height / chord_length
    if abs(float(half_sine.detach())) > 1.0:
        return AngularPerturbation(
            "excluded_infeasible_chord", None, None, None, chord_length,
            None, None, None, None, None,
        )
    if abs(float(half_sine.detach())) > math.sin(math.radians(angle_limit)):
        return AngularPerturbation(
            "excluded_angle_limit", None, None, None, chord_length,
            None, None, None, None, None,
        )

    theta = torch.asin(half_sine)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    midpoint = 0.5 * (baseline_xy[0] + baseline_xy[-1])
    relative = baseline_xy - midpoint
    rotated_relative = torch.stack((
        cosine * relative[:, 0] - sine * relative[:, 1],
        sine * relative[:, 0] + cosine * relative[:, 1],
    ), dim=-1)
    rotated = midpoint + rotated_relative

    chord_unit = chord / chord_length
    chord_normal = torch.stack((-chord_unit[1], chord_unit[0]))
    displacement = rotated - baseline_xy
    start_shift = torch.dot(displacement[0], chord_normal)
    end_shift = torch.dot(displacement[-1], chord_normal)
    tangential_slip = 0.5 * chord_length * (1.0 - cosine)

    local_min = local_max = None
    if baseline_xy.shape[0] > 2:
        tangent = baseline_xy[2:] - baseline_xy[:-2]
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
        if bool((tangent_norm <= torch.finfo(baseline_xy.dtype).eps).any()):
            return AngularPerturbation(
                "excluded_degenerate", None, None, None, chord_length,
                None, None, None, None, None,
            )
        tangent = tangent / tangent_norm[:, None]
        local_normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
        interior_shift_h = (displacement[1:-1] * local_normal).sum(dim=-1) / height
        local_min, local_max = interior_shift_h.min(), interior_shift_h.max()

    return AngularPerturbation(
        "included", rotated, theta, theta * (180.0 / math.pi), chord_length,
        start_shift, end_shift, tangential_slip, local_min, local_max,
    )
