"""Tiny, untrained page-structure mechanism for bounded CPU experiments.

One grayscale page is encoded once. A few learned queries, conditioned only on
global mean features, predict horizontal line bands, presence and directed
relations. Sampled shared features feed a CTC head. This is deliberately not an
architecture for dense newspapers and provides no OCR-performance evidence.

The geometry is in original encoded-image pixels, before EXIF orientation.
``baseline_fraction`` is mandatory: .75 places 75% of the band above its
physical baseline. Only left-to-right horizontal, two-point lines are supported.
Matching and export are discrete; matching uses geometry alone and never text.
PyTorch remains optional outside this experimental API.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import re
from typing import Any

from .joint_geometry import (
    FeatureFrame,
    geometry_loss,
    require_torch,
    sample_baseline_bands,
    torch,
    torch_f,
)


@dataclass
class PagePrediction:
    feature_map: Any  # [1,C,H,W], shared across line queries
    baseline_xy: Any  # [Q,2,2], original pixels, left-to-right horizontal
    height_px: Any  # [Q], full band heights in original pixels
    presence_logits: Any  # [Q]
    order_logits: Any  # [Q,Q], source -> target, diagonal ignored
    ctc_logits: Any  # [T,Q,A], blank token 0
    pixel_xy: Any  # [Q,S,T,2], sampled original-pixel coordinates
    frame: FeatureFrame
    baseline_fraction: float


@dataclass
class PageTargets:
    baseline_xy: Any  # [N,2,2], N <= Q
    height_px: Any  # [N]
    texts: Any  # N one-dimensional integer sequences; no blank tokens
    order: Any = None  # [N,N], -1 unknown, 0 absent, 1 directed edge


_Module = torch.nn.Module if torch is not None else object


class TinyJointPage(_Module):
    """A small integration probe, with at most six line queries per page.

    The global pooled conditioning is intentionally weak: it demonstrates the
    wiring of the losses and gradients, not spatial detection capability.
    ``detach_geometry`` stops only the CTC sampling path; geometry supervision,
    encoder gradients, presence and relation heads remain active.
    This CPU probe accepts fp32/fp64; mixed precision is not implemented.
    """

    def __init__(
        self,
        *,
        baseline_fraction: float,
        num_queries: int = 3,
        channels: int = 8,
        alphabet_size: int = 4,
        band_rows: int = 3,
        sequence_steps: int = 12,
    ) -> None:
        require_torch()
        super().__init__()
        values = (num_queries, channels, alphabet_size, band_rows, sequence_steps)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise ValueError("model dimensions must be integers")
        if not 1 <= num_queries <= 6:
            raise ValueError("num_queries must be between 1 and 6 for exhaustive matching")
        if channels < 1 or alphabet_size < 2 or band_rows < 2 or sequence_steps < 2:
            raise ValueError("channels >= 1, alphabet_size/band_rows/sequence_steps >= 2 required")
        if not math.isfinite(baseline_fraction) or not 0 <= baseline_fraction <= 1:
            raise ValueError("baseline_fraction must be finite and in [0,1]")
        self.num_queries = num_queries
        self.channels = channels
        self.alphabet_size = alphabet_size
        self.band_rows = band_rows
        self.sequence_steps = sequence_steps
        self.baseline_fraction = float(baseline_fraction)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, 3, padding=1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.Tanh(),
        )
        self.queries = torch.nn.Parameter(torch.randn(num_queries, channels) * 0.1)
        self.geometry_head = torch.nn.Linear(channels, 4)
        self.presence_head = torch.nn.Linear(channels, 1)
        self.order_head = torch.nn.Sequential(
            torch.nn.Linear(2 * channels, channels),
            torch.nn.Tanh(),
            torch.nn.Linear(channels, 1),
        )
        self.ctc_head = torch.nn.Linear(channels * band_rows, alphabet_size)

    def forward(self, image: Any, *, detach_geometry: bool = False) -> PagePrediction:
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[:2] != (1, 1):
            raise ValueError("image must have shape [1,1,H,W]: one grayscale page")
        if image.dtype not in (torch.float32, torch.float64):
            raise ValueError("this CPU probe requires float32 or float64 images; mixed precision is unsupported")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("image must contain finite floating-point values")
        page_h, page_w = image.shape[-2:]
        if min(page_h, page_w) < 4:
            raise ValueError("page width and height must be at least 4 pixels")
        feature_map = self.encoder(image)
        query_features = torch.tanh(self.queries + feature_map.mean(dim=(2, 3)))
        raw = self.geometry_head(query_features)
        # Never perform pixel-coordinate arithmetic in bf16/fp16.
        geometry_dtype = torch.float64 if raw.dtype == torch.float64 else torch.float32
        fractions = raw.to(dtype=geometry_dtype).sigmoid()
        border, min_width, min_height = 0.125, 0.25, 0.5
        x_start = border + fractions[:, 0] * (page_w - 1 - 2 * border - min_width)
        x_end = x_start + min_width + fractions[:, 1] * (
            page_w - 1 - border - x_start - min_width
        )
        height = min_height + fractions[:, 3] * (0.4 * (page_h - 1) - min_height)
        y = border + self.baseline_fraction * height + fractions[:, 2] * (
            page_h - 1 - 2 * border - height
        )
        baseline = torch.stack((torch.stack((x_start, y), -1), torch.stack((x_end, y), -1)), 1)
        frame = FeatureFrame((page_w, page_h), (0.0, 0.0), (1.0, 1.0))
        bands = sample_baseline_bands(
            feature_map.expand(self.num_queries, -1, -1, -1),
            baseline,
            height,
            frame,
            output_size=(self.band_rows, self.sequence_steps),
            baseline_fraction=self.baseline_fraction,
            detach_geometry=detach_geometry,
        )
        source = query_features[:, None, :].expand(-1, self.num_queries, -1)
        target = query_features[None, :, :].expand(self.num_queries, -1, -1)
        return PagePrediction(
            feature_map=feature_map,
            baseline_xy=baseline,
            height_px=height,
            presence_logits=self.presence_head(query_features).squeeze(-1),
            order_logits=self.order_head(torch.cat((source, target), -1)).squeeze(-1),
            ctc_logits=self.ctc_head(bands.as_ctc_sequence()),
            pixel_xy=bands.pixel_xy,
            frame=frame,
            baseline_fraction=self.baseline_fraction,
        )


def _check_prediction(prediction: PagePrediction) -> int:
    require_torch()
    prediction.frame.validate()
    baseline = prediction.baseline_xy
    if not isinstance(baseline, torch.Tensor) or baseline.ndim != 3 or baseline.shape[1:] != (2, 2):
        raise ValueError("prediction baseline_xy must have shape [Q,2,2]")
    q = len(baseline)
    if not 1 <= q <= 6 or baseline.dtype not in (torch.float32, torch.float64):
        raise ValueError("prediction requires 1..6 queries and fp32/fp64 geometry")
    expected = ((prediction.height_px, (q,)), (prediction.presence_logits, (q,)),
                (prediction.order_logits, (q, q)))
    for tensor, shape in expected:
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
            raise ValueError("prediction heights/presence/order have inconsistent shapes")
    if prediction.height_px.dtype != baseline.dtype:
        raise ValueError("prediction baseline and height must share fp32/fp64 dtype")
    logits = prediction.ctc_logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[1] != q:
        raise ValueError("prediction ctc_logits must have shape [T,Q,A]")
    if logits.shape[0] < 2 or logits.shape[2] < 2:
        raise ValueError("CTC requires T >= 2 and at least blank plus one label")
    tensors = (baseline, prediction.height_px, prediction.presence_logits, prediction.order_logits, logits)
    if any(t.device != baseline.device or not t.is_floating_point() for t in tensors):
        raise ValueError("prediction tensors must share a device and have floating dtypes")
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("prediction tensors must contain only finite values")
    if bool((prediction.height_px <= 0).any()):
        raise ValueError("prediction heights must be positive")
    if bool((baseline[:, 1, 0] <= baseline[:, 0, 0]).any()) or not torch.equal(baseline[:, 0, 1], baseline[:, 1, 1]):
        raise ValueError("only left-to-right horizontal two-point baselines are supported")
    if not math.isfinite(prediction.baseline_fraction) or not 0 <= prediction.baseline_fraction <= 1:
        raise ValueError("baseline_fraction must be finite and in [0,1]")
    _check_page_extents(baseline, prediction.height_px, prediction, "prediction")
    return q


def _check_page_extents(baseline: Any, height: Any, prediction: PagePrediction, name: str) -> None:
    width, page_height = prediction.frame.original_size_wh
    fraction = prediction.baseline_fraction
    if bool(((baseline[..., 0] < 0) | (baseline[..., 0] > width - 1)
             | (baseline[..., 1] < 0) | (baseline[..., 1] > page_height - 1)).any()):
        raise ValueError(f"{name} baselines must be within original page pixels")
    top = baseline[:, 0, 1] - fraction * height
    bottom = baseline[:, 0, 1] + (1 - fraction) * height
    if bool(((top < 0) | (bottom > page_height - 1)).any()):
        raise ValueError(f"{name} sampling bands must be within original page pixels")


def _target_geometry(prediction: PagePrediction, targets: PageTargets) -> tuple[Any, Any]:
    q = _check_prediction(prediction)
    reference = prediction.baseline_xy
    baseline = torch.as_tensor(targets.baseline_xy, device=reference.device, dtype=reference.dtype).detach()
    height = torch.as_tensor(targets.height_px, device=reference.device, dtype=reference.dtype).detach()
    if baseline.ndim != 3 or baseline.shape[1:] != (2, 2) or height.shape != (len(baseline),):
        raise ValueError("target geometry must have shapes [N,2,2] and [N]")
    if len(baseline) > q:
        raise ValueError("more target lines than queries: this probe cannot silently omit targets")
    if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(height).all()):
        raise ValueError("target geometry must contain finite values")
    if bool((height <= 0).any()):
        raise ValueError("target heights must be positive")
    if bool((baseline[:, 1, 0] <= baseline[:, 0, 0]).any()) or not torch.equal(baseline[:, 0, 1], baseline[:, 1, 1]):
        raise ValueError("targets must be left-to-right horizontal two-point baselines")
    _check_page_extents(baseline, height, prediction, "target")
    return baseline, height


def match_lines(prediction: PagePrediction, targets: PageTargets) -> tuple[int, ...]:
    """Return one unique query index per target, using detached geometry only.

    Exact exhaustive assignment is capped at six queries (at most 720
    permutations). Cost is mean absolute endpoint error plus height error,
    normalized by the fixed page diagonal. Ties use lexicographic query order;
    equal geometries cannot provide a text-independent identity guarantee.
    """
    baseline, height = _target_geometry(prediction, targets)
    if len(baseline) == 0:
        return ()
    diagonal = math.hypot(*prediction.frame.original_size_wh)
    cost = (
        (prediction.baseline_xy.detach()[:, None] - baseline[None]).abs().mean(dim=(-1, -2))
        + (prediction.height_px.detach()[:, None] - height[None]).abs()
    ) / diagonal
    detached_cost = cost.cpu().tolist()
    return min(
        permutations(range(len(prediction.baseline_xy)), len(baseline)),
        key=lambda assignment: sum(detached_cost[query][target] for target, query in enumerate(assignment)),
    )


def _text_targets(prediction: PagePrediction, targets: PageTargets, n: int) -> list[Any]:
    if not isinstance(targets.texts, (tuple, list)) or len(targets.texts) != n:
        raise ValueError("texts must be a list/tuple with one integer sequence per target line")
    result = []
    steps, _, alphabet = prediction.ctc_logits.shape
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for text in targets.texts:
        if isinstance(text, (list, tuple)) and not text:
            labels = torch.empty(0, dtype=torch.long, device=prediction.ctc_logits.device)
        else:
            try:
                labels = torch.as_tensor(text, device=prediction.ctc_logits.device)
            except (TypeError, ValueError) as exc:
                raise ValueError("CTC targets must be integer label sequences") from exc
        if labels.ndim != 1 or labels.dtype not in integer_dtypes:
            raise ValueError("CTC target labels must be one-dimensional integers")
        if bool(((labels <= 0) | (labels >= alphabet)).any()):
            raise ValueError("CTC labels must exclude blank 0 and lie within the alphabet")
        minimum_steps = len(labels) + int((labels[1:] == labels[:-1]).sum().item())
        if minimum_steps > steps:
            raise ValueError("CTC target is infeasible: T must cover labels plus adjacent repeats")
        result.append(labels.long().detach())
    return result


def page_losses(prediction: PagePrediction, targets: PageTargets) -> dict[str, Any]:
    """Four equally weighted mechanism losses, with unknown relations masked.

    CTC and geometry supervise matched queries only. Every unmatched query is a
    no-object target for presence. Order rows/columns are explicitly remapped
    from target identity to query identity. Diagonal edges are not supervised.
    Empty pages have differentiable zero CTC/geometry/order losses.
    """
    baseline, height = _target_geometry(prediction, targets)
    assignment = match_lines(prediction, targets)
    n, q = len(baseline), len(prediction.baseline_xy)
    texts = _text_targets(prediction, targets, n)
    indices = torch.tensor(assignment, device=prediction.baseline_xy.device, dtype=torch.long)
    presence_targets = torch.zeros_like(prediction.presence_logits)
    presence_targets[indices] = 1
    presence = torch_f.binary_cross_entropy_with_logits(prediction.presence_logits, presence_targets)
    if n:
        geometry = geometry_loss(
            prediction.baseline_xy[indices], prediction.height_px[indices], baseline, height,
            coordinate_scale_px=math.hypot(*prediction.frame.original_size_wh),
        ).total
        logits = prediction.ctc_logits[:, indices]
        if logits.dtype not in (torch.float32, torch.float64):
            logits = logits.float()
        ctc = torch_f.ctc_loss(
            logits.log_softmax(-1), torch.cat(texts),
            torch.full((n,), logits.shape[0], dtype=torch.long),
            torch.tensor([len(text) for text in texts], dtype=torch.long),
            blank=0, reduction="mean", zero_infinity=False,
        )
    else:
        geometry = prediction.baseline_xy.sum() * 0 + prediction.height_px.sum() * 0
        ctc = prediction.ctc_logits.sum() * 0
    order = prediction.order_logits.sum() * 0
    if targets.order is not None:
        relation = torch.as_tensor(targets.order, device=prediction.order_logits.device).detach()
        if relation.shape != (n, n) or not bool(((relation == -1) | (relation == 0) | (relation == 1)).all()):
            raise ValueError("target order must have shape [N,N], with values -1,0,1")
        if bool((relation.diagonal() == 1).any()):
            raise ValueError("target order cannot contain self-edges")
        known = (relation >= 0) & ~torch.eye(n, dtype=torch.bool, device=relation.device)
        if bool(known.any()):
            matched_logits = prediction.order_logits[indices[:, None], indices[None, :]]
            order = torch_f.binary_cross_entropy_with_logits(
                matched_logits[known], relation[known].to(matched_logits.dtype)
            )
    return {"ctc": ctc, "geometry": geometry, "presence": presence, "order": order,
            "total": ctc + geometry + presence + order}


def _greedy_ctc(labels: list[int]) -> list[int]:
    tokens, previous = [], None
    for label in labels:
        if label != 0 and label != previous:
            tokens.append(label)
        previous = label
    return tokens


def _order_diagnostic(ids: list[str], edges: list[dict[str, Any]]) -> dict[str, Any]:
    outgoing = {line_id: [] for line_id in ids}
    degree = {line_id: 0 for line_id in ids}
    for edge in edges:
        outgoing[edge["source"]].append(edge["target"])
        degree[edge["target"]] += 1
    available = sorted(line_id for line_id in ids if degree[line_id] == 0)
    ordered, ambiguous = [], False
    while available:
        ambiguous |= len(available) > 1
        current = available.pop(0)
        ordered.append(current)
        for successor in outgoing[current]:
            degree[successor] -= 1
            if degree[successor] == 0:
                available.append(successor)
        available.sort()
    cyclic = len(ordered) != len(ids)
    status = "empty" if not ids else "cycle" if cyclic else "ambiguous" if ambiguous else "unique"
    return {
        "edges": edges, "status": status,
        "topological_order": None if cyclic or ambiguous else ordered,
        "candidate_topological_order": None if cyclic else ordered,
    }


def export_page(
    prediction: PagePrediction,
    *,
    image_sha256: str,
    page_key: str,
    document_key: str,
    presence_threshold: float = 0.5,
    order_threshold: float = 0.5,
) -> dict[str, Any]:
    """Export a discrete diagnostic graph, using predictions without any GT.

    CTC output consists of token IDs because this probe has no text alphabet.
    Band polygons describe sampling support, not word/character localization.
    Thresholded cycles are reported as such; no order or missing edge is
    silently repaired. This is neither ALTO/PAGE nor a pipeline contract export.
    """
    q = _check_prediction(prediction)
    if not isinstance(image_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", image_sha256) is None:
        raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
    if not isinstance(page_key, str) or not page_key.strip() or not isinstance(document_key, str) or not document_key.strip():
        raise ValueError("page_key and document_key must be explicit nonempty identities")
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in (presence_threshold, order_threshold)):
        raise ValueError("presence/order thresholds must lie in [0,1]")
    pixels = prediction.pixel_xy
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[0] != q or pixels.shape[1] < 2 or pixels.shape[2] != prediction.ctc_logits.shape[0] or pixels.shape[3] != 2:
        raise ValueError("pixel_xy must have shape [Q,S>=2,T,2] matching CTC")
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("sampled coordinates must be finite")
    if pixels.dtype != prediction.baseline_xy.dtype or pixels.device != prediction.baseline_xy.device:
        raise ValueError("sampled coordinates and baselines must share dtype and device")
    width, height = prediction.frame.original_size_wh
    if bool(((pixels[..., 0] < 0) | (pixels[..., 0] > width - 1)
             | (pixels[..., 1] < 0) | (pixels[..., 1] > height - 1)).any()):
        raise ValueError("exported sampling polygons must lie within original page pixels")
    # A replaced or stale pixel_xy must not export a different band from the
    # supplied baseline. These equations are specific to horizontal K=2 lines.
    parameter = torch.linspace(0, 1, pixels.shape[2], dtype=pixels.dtype, device=pixels.device)
    center = (prediction.baseline_xy[:, 0, None] * (1 - parameter[None, :, None])
              + prediction.baseline_xy[:, 1, None] * parameter[None, :, None])
    fractions = torch.linspace(-prediction.baseline_fraction, 1 - prediction.baseline_fraction,
                               pixels.shape[1], dtype=pixels.dtype, device=pixels.device)
    expected = center[:, None].expand(-1, pixels.shape[1], -1, -1).clone()
    expected[..., 1] += prediction.height_px[:, None, None] * fractions[None, :, None]
    tolerance = 32 * torch.finfo(pixels.dtype).eps * max(width, height)
    if not torch.allclose(pixels, expected, rtol=0, atol=tolerance):
        raise ValueError("sampled coordinates must match the predicted baseline, height and fraction")
    presence = prediction.presence_logits.detach().sigmoid().cpu().tolist()
    selected = [query for query in range(q) if presence[query] >= presence_threshold]
    labels = prediction.ctc_logits.detach().argmax(-1).cpu()
    baselines = prediction.baseline_xy.detach().cpu().tolist()
    heights = prediction.height_px.detach().cpu().tolist()
    pixels = pixels.detach().cpu()
    lines = []
    for query in selected:
        polygon = torch.cat((pixels[query, 0], pixels[query, -1].flip(0)), dim=0)
        lines.append({
            "id": f"q{query}", "query_index": query,
            "presence_probability": presence[query],
            "baseline_xy": baselines[query], "height_px": heights[query],
            "polygon_xy": polygon.tolist(),
            "ctc_token_ids": _greedy_ctc(labels[:, query].tolist()),
            "word_boxes": None,
        })
    probability = prediction.order_logits.detach().sigmoid().cpu().tolist()
    edges = [
        {"source": f"q{source}", "target": f"q{target}", "probability": probability[source][target]}
        for source in selected for target in selected
        if source != target and probability[source][target] >= order_threshold
    ]
    return {
        "schema_version": "babaorum.joint_page.diagnostic.v1",
        "status": "UNTRAINED_SYNTHETIC_MECHANISM_ONLY",
        "image_sha256": image_sha256.lower(), "page_key": page_key, "document_key": document_key,
        "coordinate_frame": {"name": "original_encoded_pixels", "width": width, "height": height,
                             "coordinate_convention": "pixel_centers_0_width_minus_1_height_minus_1",
                             "origin": "upper_left", "x_direction": "right", "y_direction": "down",
                             "before_exif_orientation": True},
        "baseline_fraction": prediction.baseline_fraction,
        "presence_threshold": presence_threshold, "order_threshold": order_threshold,
        "lines": lines, "order": _order_diagnostic([line["id"] for line in lines], edges),
        "word_localization_available": False,
    }
