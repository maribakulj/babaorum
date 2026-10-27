"""Differentiable sampling of text-line feature bands in page coordinates.

This is an engineering probe, not a page recognizer. Baseline control points,
heights, and ``FeatureFrame`` all use the original encoded scan's pixel frame:
origin at the upper-left, x rightward, y downward, before EXIF orientation.
The feature map's center at index (0, 0) is ``origin_xy`` in that frame; each
feature step spans ``stride_xy`` original pixels.

PyTorch is optional for the Babaorum package as a whole. Import this module
without torch and call ``require_torch`` only at the experimental API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:  # pragma: no cover - exercised only in torch-free envs
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FeatureFrame:
    """Mapping from feature-map centers to original encoded-page pixels."""

    original_size_wh: tuple[int, int]
    origin_xy: tuple[float, float]
    stride_xy: tuple[float, float]

    def validate(self) -> None:
        width, height = self.original_size_wh
        if width <= 0 or height <= 0:
            raise ValueError("original page width and height must be positive")
        if len(self.origin_xy) != 2 or len(self.stride_xy) != 2:
            raise ValueError("origin_xy and stride_xy must each contain x,y")
        if not all(math.isfinite(v) for v in (*self.origin_xy, *self.stride_xy)):
            raise ValueError("feature frame values must be finite")
        if self.stride_xy[0] <= 0 or self.stride_xy[1] <= 0:
            raise ValueError("feature stride must be positive")


@dataclass
class BandSample:
    """Sampled feature bands and the corresponding original-page coordinates."""

    features: Any  # N,C,S,T
    pixel_xy: Any  # N,S,T,2 in original encoded-page pixels

    def as_ctc_sequence(self) -> Any:
        """Flatten band rows into channels; return T,N,C*S for a linear CTC head."""
        require_torch()
        n, channels, band_rows, steps = self.features.shape
        return self.features.permute(3, 0, 1, 2).reshape(steps, n, channels * band_rows)


@dataclass
class GeometryLoss:
    total: Any
    baseline: Any
    height: Any


def require_torch() -> None:
    if torch is None or torch_f is None:
        raise RuntimeError(
            "This experimental sampler requires PyTorch; install it only in a "
            "dedicated optional environment after project approval."
        )


def _finite(tensor: Any, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def sample_baseline_bands(
    feature_map: Any,
    baseline_xy: Any,
    height_px: Any,
    frame: FeatureFrame,
    *,
    output_size: tuple[int, int],
    detach_geometry: bool = False,
    baseline_fraction: float | Any = 0.5,
) -> BandSample:
    """Bilinearly sample N line bands along parametric baselines.

    Args:
        feature_map: float tensor [N,C,Hf,Wf].
        baseline_xy: float tensor [N,K,2], K>=2 control points, in original
            encoded-page pixels. Points are uniformly spaced in the curve's
            parameter, not necessarily in x or arc length.
        height_px: positive full band height per line, [N] or [N,1], in the
            same original-page pixel frame.
        frame: explicit feature-to-original-pixel mapping.
        output_size: (band_rows, sequence_steps), both >=2.
        detach_geometry: stop sampling-loss gradients at baseline/height while
            preserving encoder gradients. Supervised geometry loss is separate.
        baseline_fraction: fraction of the full band height above the baseline
            for a left-to-right baseline. A scalar applies to every line; a
            tensor [N] allows a different differentiable fraction per line.
            The compatible default, 0.5, centers the band; 0.75 reserves three
            quarters above and one quarter below. Reversing the control-point
            order reverses the band normal too.

    Samples outside either the original image or the available feature-map
    centers are rejected. This avoids silent border extrapolation at tile edges.
    This strict experimental policy requires in-bounds predicted geometry; it is
    not a production strategy for handling arbitrary detector predictions.
    Nonzero sampled tangents do not rule out self-intersections or folded curved
    bands; callers requiring geometric integrity must constrain or validate them.
    The returned grid remains differentiable with respect to control points and
    height and tensor-valued ``baseline_fraction`` when
    ``detach_geometry=False``.
    """
    require_torch()
    frame.validate()
    if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4:
        raise ValueError("feature_map must be a torch tensor [N,C,H,W]")
    if feature_map.dtype not in (torch.float32, torch.float64):
        raise ValueError("feature_map and geometry must use float32 or float64")
    if not isinstance(baseline_xy, torch.Tensor) or baseline_xy.ndim != 3:
        raise ValueError("baseline_xy must be a tensor [N,K,2]")
    if not isinstance(height_px, torch.Tensor):
        raise ValueError("height_px must be a tensor [N] or [N,1]")
    n, channels, feature_h, feature_w = feature_map.shape
    if n < 1 or channels < 1 or feature_h < 2 or feature_w < 2:
        raise ValueError("feature_map dimensions must be positive with H,W >= 2")
    if baseline_xy.shape[0] != n or baseline_xy.shape[1] < 2 or baseline_xy.shape[2] != 2:
        raise ValueError("baseline_xy must have shape [N,K>=2,2] matching feature_map")
    rows, steps = output_size
    if rows < 2 or steps < 2:
        raise ValueError("output_size band_rows and sequence_steps must be >= 2")
    if height_px.numel() != n or height_px.ndim not in (1, 2):
        raise ValueError("height_px must have one value per line")
    if height_px.ndim == 2 and height_px.shape != (n, 1):
        raise ValueError("two-dimensional height_px must have shape [N,1]")
    if feature_map.device != baseline_xy.device or feature_map.device != height_px.device:
        raise ValueError("feature_map, baseline_xy, and height_px must share a device")
    if feature_map.dtype != baseline_xy.dtype or feature_map.dtype != height_px.dtype:
        raise ValueError("feature_map, baseline_xy, and height_px must share a dtype")
    try:
        fractions_above = torch.as_tensor(
            baseline_fraction, device=baseline_xy.device, dtype=baseline_xy.dtype
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline_fraction must be a finite scalar or [N] tensor") from exc
    if fractions_above.ndim == 0:
        fractions_above = fractions_above.expand(n)
    if fractions_above.shape != (n,):
        raise ValueError("baseline_fraction tensor must be scalar or have shape [N]")

    _finite(feature_map, "feature_map")
    _finite(baseline_xy, "baseline_xy")
    _finite(height_px, "height_px")
    _finite(fractions_above, "baseline_fraction")
    if bool(((fractions_above < 0.0) | (fractions_above > 1.0)).any()):
        raise ValueError("baseline_fraction must be finite and in [0,1]")
    heights = height_px.reshape(n)
    if bool((heights <= 0).any()):
        raise ValueError("all line heights must be positive")

    controls = baseline_xy.detach() if detach_geometry else baseline_xy
    heights = heights.detach() if detach_geometry else heights
    fractions_above = fractions_above.detach() if detach_geometry else fractions_above
    k = controls.shape[1]
    param = torch.linspace(0, k - 1, steps, device=controls.device, dtype=controls.dtype)
    lo = param.floor().long()
    hi = (lo + 1).clamp(max=k - 1)
    alpha = (param - lo.to(param.dtype)).view(1, steps, 1)
    center = controls[:, lo, :] * (1.0 - alpha) + controls[:, hi, :] * alpha

    tangent = torch.empty_like(center)
    tangent[:, 0] = center[:, 1] - center[:, 0]
    tangent[:, -1] = center[:, -1] - center[:, -2]
    if steps > 2:
        tangent[:, 1:-1] = center[:, 2:] - center[:, :-2]
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    _finite(tangent_norm, "baseline tangent norms")
    if bool((tangent_norm <= torch.finfo(tangent_norm.dtype).eps).any()):
        raise ValueError("baseline must have a nonzero tangent at every sampled step")
    tangent = tangent / tangent_norm
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)

    row_fractions = torch.linspace(0.0, 1.0, rows, device=controls.device, dtype=controls.dtype)
    fractions = row_fractions[None, :] - fractions_above[:, None]
    offsets = heights[:, None, None, None] * fractions[:, :, None, None]
    pixel_xy = center[:, None, :, :] + offsets * normal[:, None, :, :]
    _finite(pixel_xy, "sampled line band coordinates")

    page_w, page_h = frame.original_size_wh
    x, y = pixel_xy[..., 0], pixel_xy[..., 1]
    if bool(((x < 0) | (x > page_w - 1) | (y < 0) | (y > page_h - 1)).any()):
        raise ValueError("sampled line band extends outside original page pixels")

    origin_x, origin_y = frame.origin_xy
    stride_x, stride_y = frame.stride_xy
    ix = (x - origin_x) / stride_x
    iy = (y - origin_y) / stride_y
    if bool(((ix < 0) | (ix > feature_w - 1) | (iy < 0) | (iy > feature_h - 1)).any()):
        raise ValueError("sampled line band extends outside feature-map support")
    # grid_sample with align_corners=False: integer feature indices are pixel centers.
    gx = (2.0 * ix + 1.0) / feature_w - 1.0
    gy = (2.0 * iy + 1.0) / feature_h - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(n, rows * steps, 1, 2)
    sampled = torch_f.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(n, channels, rows, steps)
    return BandSample(features=sampled, pixel_xy=pixel_xy)


def geometry_loss(
    predicted_baseline_xy: Any,
    predicted_height_px: Any,
    target_baseline_xy: Any,
    target_height_px: Any,
    *,
    coordinate_scale_px: float = 1.0,
    baseline_weight: float = 1.0,
    height_weight: float = 1.0,
) -> GeometryLoss:
    """Smooth-L1 supervision for matched line baselines and full heights.

    ``coordinate_scale_px`` can be page diagonal or another fixed scale. This
    loss is intended for geometrically matched lines with corresponding control
    points; it does not do assignment or arc-length reparameterization. Both
    predicted and target heights must be positive. Inputs use float32 or float64.
    """
    require_torch()
    if not math.isfinite(coordinate_scale_px) or coordinate_scale_px <= 0:
        raise ValueError("coordinate_scale_px must be finite and positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in (baseline_weight, height_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    tensors = (predicted_baseline_xy, predicted_height_px, target_baseline_xy, target_height_px)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("geometry loss inputs must be torch tensors")
    if (
        predicted_baseline_xy.shape != target_baseline_xy.shape
        or predicted_baseline_xy.ndim != 3
        or predicted_baseline_xy.shape[0] < 1
        or predicted_baseline_xy.shape[1] < 2
        or predicted_baseline_xy.shape[2] != 2
    ):
        raise ValueError("predicted and target baselines must share shape [N>=1,K>=2,2]")
    n = predicted_baseline_xy.shape[0]
    if any(t.shape not in ((n,), (n, 1)) for t in (predicted_height_px, target_height_px)):
        raise ValueError("predicted and target heights must each have shape [N] or [N,1]")
    pred_h = predicted_height_px.reshape(-1)
    target_h = target_height_px.reshape(-1)
    if any(t.device != predicted_baseline_xy.device for t in tensors):
        raise ValueError("all geometry loss tensors must share a device")
    if any(t.dtype != predicted_baseline_xy.dtype for t in tensors):
        raise ValueError("all geometry loss tensors must share a dtype")
    if predicted_baseline_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError("geometry loss tensors must use float32 or float64")
    for name, tensor in zip(("predicted_baseline_xy", "predicted_height_px", "target_baseline_xy", "target_height_px"), tensors):
        _finite(tensor, name)
    if bool((pred_h <= 0).any()) or bool((target_h <= 0).any()):
        raise ValueError("predicted and target line heights must be positive")
    scale = float(coordinate_scale_px)
    baseline = torch_f.smooth_l1_loss(
        predicted_baseline_xy / scale, target_baseline_xy / scale, reduction="mean"
    )
    height = torch_f.smooth_l1_loss(pred_h / scale, target_h / scale, reduction="mean")
    total = baseline_weight * baseline + height_weight * height
    _finite(baseline, "normalized baseline loss")
    _finite(height, "normalized height loss")
    _finite(total, "weighted geometry loss")
    return GeometryLoss(total=total, baseline=baseline, height=height)
