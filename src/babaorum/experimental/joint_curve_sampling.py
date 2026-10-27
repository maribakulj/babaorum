"""Arc-length bands with explicit validity for tiny CPU geometry probes.

Coordinates remain raw original-image pixel centers. Boundary penalties use
those raw coordinates; clamping happens only inside ``sample_curved_band``.
A line with any invalid sample, reversed cell, local fold, sharp turn, wrong
orientation or self-intersecting contour is inadmissible for CTC. No rejected
geometry is silently repaired. This is a mechanism experiment, not OCR.

Normals are (-t_y,t_x) in image coordinates with y down. Arc interpolation is
piecewise differentiable; segment selection, sample count and all validity
decisions are discrete. Local fold penalties do not promise to repair global
intersections, whose detection is a separate bounded CPU check.
The epsilon hinge is a repair penalty, not a barrier guaranteeing valid bands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .joint_curves import CurveNumericalError, resample_arclength
from .joint_geometry import FeatureFrame, require_torch, torch, torch_f


@dataclass
class CurveBandGrid:
    pixel_xy: Any  # [S,M,2], RAW original-page coordinates
    center_xy: Any  # [M,2], arc-length-resampled baseline
    sample_valid: Any  # [S,M], page and feature-center support
    cell_valid: Any  # [S-1,M-1], support and local nonfold criteria
    line_valid: Any  # scalar bool tensor; required before CTC
    boundary_loss: Any  # scalar, squared error normalized by fixed penalty scale
    fold_loss: Any  # scalar, dimensionless local penalties
    orientation_valid: Any  # scalar bool tensor
    frame: FeatureFrame
    feature_size_hw: tuple[int, int]
    penalty_scale_px: float  # fixed caller-provided scale, never predicted height
    contour_valid: Any  # scalar bool tensor, discrete global contour check
    sharp_turn_valid: Any  # scalar bool tensor; original angles must be <90 deg
    tangent_valid: Any  # scalar bool tensor
    projection_valid: Any  # [S,M-1]
    jacobian_valid: Any  # [S-1,M-1], both triangles positive


@dataclass
class CurveBandSample:
    features: Any  # [1,C,S,M]; invalid lines are entirely zero
    grid: CurveBandGrid

    def as_ctc_sequence(self, *, require_valid: bool = True) -> Any:
        """Return [M,1,C*S], refusing invalid lines instead of hiding targets."""
        require_torch()
        if require_valid and not bool(self.grid.line_valid):
            raise ValueError("invalid curved band is inadmissible for CTC")
        n, channels, rows, steps = self.features.shape
        return self.features.permute(3, 0, 1, 2).reshape(steps, n, channels * rows)


@dataclass
class CurveCTCLoss:
    loss: Any
    status: str  # included | excluded_invalid_geometry | excluded_infeasible_length
    included_lines: int
    excluded_lines: int


def _support_bounds(frame: FeatureFrame, feature_size_hw: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
    frame.validate()
    if (not isinstance(feature_size_hw, (tuple, list)) or len(feature_size_hw) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in feature_size_hw)):
        raise ValueError("feature_size_hw must contain integer height,width >=2")
    feature_h, feature_w = feature_size_hw
    page_w, page_h = frame.original_size_wh
    origin_x, origin_y = frame.origin_xy
    stride_x, stride_y = frame.stride_xy
    lower = (max(0.0, origin_x), max(0.0, origin_y))
    upper = (min(page_w - 1.0, origin_x + (feature_w - 1) * stride_x),
             min(page_h - 1.0, origin_y + (feature_h - 1) * stride_y))
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError("feature-center support must overlap the original page in both dimensions")
    return lower, upper


def _cross(a: Any, b: Any) -> Any:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _simple_contour(pixel_xy: Any) -> bool:
    """Detect all intersections/touches between nonadjacent perimeter edges.

    This uses detached CPU double coordinates and includes the end caps. It is
    a discrete rejection test, never a differentiable correction. At most 572
    perimeter vertices are possible under this module's fixed size caps. Its
    O(V**2) cost is intended only for this bounded CPU engineering probe.
    """
    contour = torch.cat((pixel_xy[0], pixel_xy[1:, -1],
                         pixel_xy[-1, :-1].flip(0), pixel_xy[1:-1, 0].flip(0)))
    points = contour.detach().cpu().double().tolist()
    count = len(points)
    scale = max(1.0, max(abs(value) for point in points for value in point))
    tolerance = 1e-10 * scale
    cross_tolerance = tolerance * scale

    def orientation(a: list[float], b: list[float], c: list[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def contains(a: list[float], b: list[float], p: list[float]) -> bool:
        return all(min(a[k], b[k]) - tolerance <= p[k] <= max(a[k], b[k]) + tolerance for k in (0, 1))

    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue  # the closing pair is adjacent
            c, d = points[j], points[(j + 1) % count]
            if any(max(a[k], b[k]) + tolerance < min(c[k], d[k])
                   or max(c[k], d[k]) + tolerance < min(a[k], b[k]) for k in (0, 1)):
                continue
            o1, o2 = orientation(a, b, c), orientation(a, b, d)
            o3, o4 = orientation(c, d, a), orientation(c, d, b)
            straddle1 = (o1 > cross_tolerance and o2 < -cross_tolerance) or (o2 > cross_tolerance and o1 < -cross_tolerance)
            straddle2 = (o3 > cross_tolerance and o4 < -cross_tolerance) or (o4 > cross_tolerance and o3 < -cross_tolerance)
            if straddle1 and straddle2:
                return False
            if ((abs(o1) <= cross_tolerance and contains(a, b, c))
                    or (abs(o2) <= cross_tolerance and contains(a, b, d))
                    or (abs(o3) <= cross_tolerance and contains(c, d, a))
                    or (abs(o4) <= cross_tolerance and contains(c, d, b))):
                return False
    return True


def curved_band_grid(
    points_xy: Any,
    height_px: Any,
    frame: FeatureFrame,
    feature_size_hw: tuple[int, int],
    *,
    band_rows: int,
    spacing_px: float,
    baseline_fraction: float,
    orientation: str = "ltr",
    max_samples: int = 256,
    penalty_scale_px: float = 1.0,
) -> CurveBandGrid:
    """Build a raw band and masks; invalid geometry remains available for loss.

    Height must be a positive scalar fp32/fp64 tensor sharing points' dtype and
    device. ``baseline_fraction`` gives the fraction along the negative normal
    (above a horizontal ltr baseline). Original polyline turns of >=90 degrees
    are rejected, with a small floating-point tolerance at exactly 90 degrees.
    Query orientation checks the signed endpoint displacement on its named
    axis. Unknown orientation is refused. This CPU probe caps rows at 32 and
    resampled points at 256; exceeding a cap raises rather than undersampling.

    Boundary loss is mean squared hinge distance to page/support divided by
    ``penalty_scale_px**2`` (default: one pixel). A caller may supply a fixed
    configuration scale or GT height; it must never be predicted height.
    Fold loss sums equally weighted mean hinges for normalized row projection,
    each triangle Jacobian, and tangent length / fixed penalty scale, plus the
    maximum original-turn-cosine hinge (invariant to collinear densification).
    Projection denominator length**2 and Jacobian denominator
    length*height/(rows-1) remain differentiable. The positive hinge margin is
    8*dtype epsilon; clamping denominators there only handles degeneracies.
    """
    require_torch()
    lower_values, upper_values = _support_bounds(frame, feature_size_hw)
    if orientation not in ("ltr", "rtl", "ttb", "btt"):
        raise ValueError("orientation must be ltr, rtl, ttb or btt; unknown is unsupported")
    if not isinstance(band_rows, int) or isinstance(band_rows, bool) or not 2 <= band_rows <= 32:
        raise ValueError("band_rows must be an integer in [2,32] for this bounded CPU probe")
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or not 2 <= max_samples <= 256:
        raise ValueError("max_samples must be an integer in [2,256]")
    if not isinstance(points_xy, torch.Tensor) or points_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError("points_xy must be a float32/float64 tensor")
    if points_xy.device.type != "cpu":
        raise ValueError("this bounded curved-band probe supports CPU only")
    if not isinstance(height_px, torch.Tensor) or height_px.ndim != 0:
        raise ValueError("height_px must be a scalar tensor")
    if height_px.dtype != points_xy.dtype or height_px.device != points_xy.device:
        raise ValueError("height_px and points_xy must share float32/float64 dtype and device")
    if not bool(torch.isfinite(height_px)) or not bool(height_px > 0):
        raise ValueError("height_px must be finite and positive")
    if not math.isfinite(baseline_fraction) or not 0 <= baseline_fraction <= 1:
        raise ValueError("baseline_fraction must be finite and in [0,1]")
    if isinstance(penalty_scale_px, bool) or not isinstance(penalty_scale_px, (int, float)) or not math.isfinite(penalty_scale_px) or penalty_scale_px <= 0:
        raise ValueError("penalty_scale_px must be a fixed finite positive number")
    arc = resample_arclength(points_xy, spacing_px=spacing_px, max_samples=max_samples)
    center = arc.points_xy
    tangent = torch.cat((center[1:2] - center[:1], center[2:] - center[:-2], center[-1:] - center[-2:-1]))
    tangent_length = torch.linalg.vector_norm(tangent, dim=-1)
    tiny = torch.finfo(points_xy.dtype).eps * 8
    tangent_valid = (tangent_length > tiny).all()
    unit_tangent = tangent / tangent_length.clamp_min(tiny)[:, None]
    normal = torch.stack((-unit_tangent[:, 1], unit_tangent[:, 0]), dim=-1)
    fractions = torch.linspace(-baseline_fraction, 1 - baseline_fraction, band_rows,
                               dtype=points_xy.dtype, device=points_xy.device)
    pixels = center[None] + height_px * fractions[:, None, None] * normal[None]
    if not bool(torch.isfinite(pixels).all()):
        raise CurveNumericalError("raw curved-band coordinates must be finite; geometry overflowed")
    lower = pixels.new_tensor(lower_values)
    upper = pixels.new_tensor(upper_values)
    sample_valid = ((pixels >= lower) & (pixels <= upper)).all(dim=-1)
    boundary_error = torch.relu(lower - pixels) + torch.relu(pixels - upper)
    boundary_loss = (boundary_error / penalty_scale_px).square().mean()

    central_segments = center[1:] - center[:-1]
    central_length = torch.linalg.vector_norm(central_segments, dim=-1)
    offset_segments = pixels[:, 1:] - pixels[:, :-1]
    normalized_projection = (offset_segments * central_segments[None]).sum(-1) / central_length.square().clamp_min(tiny * tiny)
    projection_valid = normalized_projection > 0
    a, b, c, d = pixels[:-1, :-1], pixels[:-1, 1:], pixels[1:, :-1], pixels[1:, 1:]
    area_scale = (central_length * height_px / (band_rows - 1)).clamp_min(tiny * tiny)
    jacobian_1 = _cross(b - a, c - a) / area_scale[None]
    jacobian_2 = _cross(d - b, c - b) / area_scale[None]
    jacobian_valid = (jacobian_1 > 0) & (jacobian_2 > 0)
    cell_valid = (sample_valid[:-1, :-1] & sample_valid[:-1, 1:]
                  & sample_valid[1:, :-1] & sample_valid[1:, 1:]
                  & jacobian_valid & projection_valid[:-1] & projection_valid[1:])

    original_segments = points_xy[1:] - points_xy[:-1]
    original_unit = original_segments / torch.linalg.vector_norm(original_segments, dim=-1)[:, None]
    turn_cosines = (original_unit[1:] * original_unit[:-1]).sum(-1)
    sharp_turn_valid = (turn_cosines > tiny).all()
    turn_penalty = torch.relu(tiny - turn_cosines).max() if len(turn_cosines) else points_xy.sum() * 0
    fold_loss = (torch.relu(tiny - normalized_projection).mean()
                 + torch.relu(tiny - jacobian_1).mean() + torch.relu(tiny - jacobian_2).mean()
                 + turn_penalty + torch.relu(tiny - tangent_length / penalty_scale_px).mean())
    for name, value in (("boundary_loss", boundary_loss), ("fold_loss", fold_loss),
                        ("normalized_projection", normalized_projection),
                        ("jacobian_1", jacobian_1), ("jacobian_2", jacobian_2)):
        if not bool(torch.isfinite(value).all()):
            raise CurveNumericalError(f"{name} must be finite; curved-band geometry overflowed")
    displacement = center[-1] - center[0]
    axis, sign = {"ltr": (0, 1), "rtl": (0, -1), "ttb": (1, 1), "btt": (1, -1)}[orientation]
    orientation_valid = displacement[axis] * sign > 0
    contour_valid = torch.tensor(_simple_contour(pixels), dtype=torch.bool, device=pixels.device)
    line_valid = (sample_valid.all() & cell_valid.all() & tangent_valid
                  & sharp_turn_valid & orientation_valid & contour_valid)
    return CurveBandGrid(
        pixel_xy=pixels, center_xy=center, sample_valid=sample_valid, cell_valid=cell_valid,
        line_valid=line_valid, boundary_loss=boundary_loss, fold_loss=fold_loss,
        orientation_valid=orientation_valid, frame=frame, feature_size_hw=tuple(feature_size_hw),
        penalty_scale_px=float(penalty_scale_px),
        contour_valid=contour_valid, sharp_turn_valid=sharp_turn_valid, tangent_valid=tangent_valid,
        projection_valid=projection_valid, jacobian_valid=jacobian_valid,
    )


def sample_curved_band(
    feature_map: Any,
    grid: CurveBandGrid,
    frame: FeatureFrame,
    *,
    detach_geometry: bool = False,
) -> CurveBandSample:
    """Sample safely, mask invalid lines, and preserve raw penalty gradients.

    Detachment affects only sampling coordinates. The returned grid keeps its
    original differentiable boundary/fold losses for supervised repair.
    """
    require_torch()
    if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4 or feature_map.shape[0] != 1 or feature_map.shape[1] < 1:
        raise ValueError("feature_map must have shape [1,C>=1,H,W]")
    if feature_map.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(feature_map).all()):
        raise ValueError("feature_map must contain finite float32/float64 values")
    if feature_map.device.type != "cpu":
        raise ValueError("this bounded curved-band probe supports CPU only")
    if frame != grid.frame or tuple(feature_map.shape[-2:]) != grid.feature_size_hw:
        raise ValueError("feature_map dimensions and frame must match the grid's declared support")
    if feature_map.dtype != grid.pixel_xy.dtype or feature_map.device != grid.pixel_xy.device:
        raise ValueError("features and raw grid must share dtype and device")
    lower_values, upper_values = _support_bounds(frame, grid.feature_size_hw)
    pixels = grid.pixel_xy.detach() if detach_geometry else grid.pixel_xy
    # This clamp is only a sampling safety mechanism. The raw grid, support
    # masks and boundary_loss were computed before any clamp.
    safe = torch.maximum(torch.minimum(pixels, pixels.new_tensor(upper_values)), pixels.new_tensor(lower_values))
    feature_h, feature_w = grid.feature_size_hw
    ix = (safe[..., 0] - frame.origin_xy[0]) / frame.stride_xy[0]
    iy = (safe[..., 1] - frame.origin_xy[1]) / frame.stride_xy[1]
    normalized = torch.stack(((2 * ix + 1) / feature_w - 1, (2 * iy + 1) / feature_h - 1), -1)
    sampled = torch_f.grid_sample(feature_map, normalized[None], mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
    effective_valid = grid.sample_valid & grid.line_valid
    sampled = sampled * effective_valid[None, None].to(sampled.dtype)
    return CurveBandSample(features=sampled, grid=grid)


def masked_curve_ctc_loss(logits: Any, sample_or_grid: Any, target_ids: Any) -> CurveCTCLoss:
    """Include one valid line or explicitly count geometry/length exclusions.

    An invalid line contributes a differentiable zero, without replacing its
    transcription by an empty target. Boundary/fold losses remain on the grid
    and must be included separately by an eventual training objective. Counts
    must accompany any future aggregate so excluding lines cannot look like
    improved OCR. This single-line helper does not evaluate recognition.
    Arc-length sampling makes available time steps depend on predicted length:
    an infeasible label alignment is therefore excluded and counted rather than
    stopping the iteration. Invalid target formats/blank/alphabet still raise.
    """
    require_torch()
    grid = sample_or_grid.grid if isinstance(sample_or_grid, CurveBandSample) else sample_or_grid
    if not isinstance(grid, CurveBandGrid):
        raise ValueError("sample_or_grid must be CurveBandSample or CurveBandGrid")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[1] != 1 or logits.shape[2] < 2:
        raise ValueError("logits must have shape [M,1,A>=2]")
    if logits.shape[0] != grid.pixel_xy.shape[1]:
        raise ValueError("CTC time steps must match the sampled arc-length grid")
    if logits.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(logits).all()):
        raise ValueError("CTC logits must contain finite float32/float64 values")
    if logits.device.type != "cpu":
        raise ValueError("this curved-band CTC helper supports CPU only")
    if isinstance(target_ids, (list, tuple)) and not target_ids:
        labels = torch.empty(0, dtype=torch.long, device=logits.device)
    else:
        try:
            labels = torch.as_tensor(target_ids, device=logits.device)
        except (TypeError, ValueError) as exc:
            raise ValueError("CTC targets must be a one-dimensional integer sequence") from exc
    if labels.ndim != 1 or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("CTC targets must be a one-dimensional integer sequence")
    if bool(((labels <= 0) | (labels >= logits.shape[-1])).any()):
        raise ValueError("CTC targets must exclude blank 0 and lie within the alphabet")
    if not bool(grid.line_valid):
        return CurveCTCLoss(logits.reshape(-1)[0] * 0, "excluded_invalid_geometry", 0, 1)
    minimum_steps = len(labels) + int((labels[1:] == labels[:-1]).sum().item())
    if minimum_steps > logits.shape[0]:
        return CurveCTCLoss(logits.reshape(-1)[0] * 0, "excluded_infeasible_length", 0, 1)
    loss = torch_f.ctc_loss(
        logits.log_softmax(-1), labels.long().detach(),
        torch.tensor([logits.shape[0]], dtype=torch.long), torch.tensor([len(labels)], dtype=torch.long),
        blank=0, reduction="mean", zero_infinity=False,
    )
    return CurveCTCLoss(loss, "included", 1, 0)
