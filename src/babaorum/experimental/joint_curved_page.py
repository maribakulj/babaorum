"""Tiny shared-feature page integration for curved-line engineering probes.

This CPU-only, untrained model conditions a few learned queries on global mean
CNN features. It is not a dense-page OCR architecture. Geometry uses original
encoded-image pixel centers, with ltr query orientation and explicit band
fraction. Fault-injection overrides exist only for synthetic diagnostics.

Sampling exclusions never remove GT lines from supervision or accounting. CTC
uses a fixed matched-line denominator and a declared constant charge for each
excluded line. Neither that charge nor the total loss is an OCR metric or a
valid checkpoint-selection criterion. Geometry repair is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import re
from typing import Any

from .joint_curve_repair import repair_degenerate_curve_loss
from .joint_curve_sampling import curved_band_grid, masked_curve_ctc_loss, sample_curved_band
from .joint_curves import CurveNumericalError, CurveSamplingCapError, curve_geometry_loss
from .joint_geometry import FeatureFrame, require_torch, torch, torch_f


@dataclass
class CurvedQueryPrediction:
    status: str
    grid: Any = None
    ctc_logits: Any = None  # [M,1,A], variable M per query
    failure_reason: str | None = None
    required_steps: int | None = None


@dataclass
class CurvedPagePrediction:
    feature_map: Any
    baseline_xy: Any  # [Q,K,2]
    height_px: Any  # [Q]
    presence_logits: Any  # [Q]
    order_logits: Any  # [Q,Q]
    queries: tuple[CurvedQueryPrediction, ...]
    frame: FeatureFrame
    baseline_fraction: float
    spacing_px: float
    max_samples: int
    penalty_scale_px: float
    alphabet_size: int
    min_segment_length_px: float
    orientation: str = "ltr"


@dataclass
class CurvedPageTargets:
    baseline_xy: Any  # list/tuple of [K_i,2] tensors, variable K_i
    height_px: Any  # [N]
    texts: Any  # N integer label sequences, blank 0 forbidden
    order: Any = None  # [N,N], -1 unknown, 0 absent, 1 directed relation


@dataclass
class CurvedPageLosses:
    losses: dict[str, Any]
    counts: dict[str, int]
    matching: tuple[int, ...]  # one query index per GT line
    line_statuses: tuple[str, ...]  # includes unmatched query statuses
    geometry_statuses: tuple[str, ...]
    diagnostics: dict[str, Any]


_Module = torch.nn.Module if torch is not None else object


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a fixed finite positive number")
    return float(value)


def _required_steps(length: Any, spacing: float) -> int:
    ratio = float(length.detach()) / spacing
    if not math.isfinite(ratio):
        return 2**63 - 1
    nearest = round(ratio)
    tolerance = 8 * torch.finfo(length.dtype).eps * max(1.0, abs(ratio))
    if abs(ratio - nearest) <= tolerance:
        ratio = float(nearest)
    return max(2, math.ceil(ratio) + 1)


class TinyCurvedJointPage(_Module):
    """A bounded wiring probe with one shared feature map and <=6 queries.

    Every normal x increment is >=.125 pixels, including at saturated logits.
    Positive softplus weights distribute any extra width without sorting. The
    integration layer classifies predicted segments <=1e-4 pixels as degenerate
    before invoking the strict primitive sampler. Normal predictions cannot
    reach that cutoff; overrides can test the recovery path explicitly.
    """

    def __init__(
        self, *, baseline_fraction: float, num_queries: int = 3,
        control_points: int = 4, channels: int = 8, alphabet_size: int = 4,
        band_rows: int = 3, spacing_px: float = 0.75, max_samples: int = 256,
        penalty_scale_px: float = 4.0,
    ) -> None:
        require_torch()
        super().__init__()
        dimensions = (num_queries, control_points, channels, alphabet_size, band_rows, max_samples)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in dimensions):
            raise ValueError("model dimensions must be integers")
        if not 1 <= num_queries <= 6 or not 2 <= control_points <= 8:
            raise ValueError("this CPU probe requires 1..6 queries and 2..8 control points")
        if channels < 1 or alphabet_size < 2 or not 2 <= band_rows <= 32 or not 2 <= max_samples <= 256:
            raise ValueError("invalid channel/alphabet/band/sample dimensions")
        if not isinstance(baseline_fraction, (int, float)) or not math.isfinite(baseline_fraction) or not 0 <= baseline_fraction <= 1:
            raise ValueError("baseline_fraction must lie in [0,1]")
        self.num_queries, self.control_points = num_queries, control_points
        self.channels, self.alphabet_size, self.band_rows = channels, alphabet_size, band_rows
        self.baseline_fraction = float(baseline_fraction)
        self.spacing_px = _positive(spacing_px, "spacing_px")
        self.max_samples = max_samples
        self.penalty_scale_px = _positive(penalty_scale_px, "penalty_scale_px")
        self.min_increment_px = 0.125
        self.min_segment_length_px = 1e-4
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, 3, padding=1), torch.nn.Tanh(),
            torch.nn.Conv2d(channels, channels, 3, padding=1), torch.nn.Tanh(),
        )
        self.queries = torch.nn.Parameter(torch.randn(num_queries, channels) * 0.1)
        self.geometry_head = torch.nn.Linear(channels, 2 * control_points + 3)
        self.presence_head = torch.nn.Linear(channels, 1)
        self.order_head = torch.nn.Sequential(torch.nn.Linear(2 * channels, channels),
                                             torch.nn.Tanh(), torch.nn.Linear(channels, 1))
        self.ctc_head = torch.nn.Linear(channels * band_rows, alphabet_size)

    def forward(
        self, image: Any, *, detach_geometry: bool = False,
        baseline_override: Any = None, height_override: Any = None,
    ) -> CurvedPagePrediction:
        """Predict normally or inject explicit synthetic geometry faults.

        Overrides bypass the learned geometry head for the supplied quantity.
        They exercise recovery/accounting paths, not detector performance or a
        production prediction API. Finite out-of-page, collapsed and excessive
        length overrides remain visible as classified query failures.
        """
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[:2] != (1, 1):
            raise ValueError("image must be one grayscale page [1,1,H,W]")
        if image.dtype not in (torch.float32, torch.float64) or image.device.type != "cpu":
            raise ValueError("this probe requires CPU float32/float64 images")
        if not bool(torch.isfinite(image).all()) or min(image.shape[-2:]) < 4:
            raise ValueError("image must be finite with height and width >=4")
        page_h, page_w = image.shape[-2:]
        features = self.encoder(image)
        context = torch.tanh(self.queries + features.mean(dim=(2, 3)))
        raw = self.geometry_head(context)
        if raw.dtype not in (torch.float32, torch.float64):
            raise ValueError("mixed-precision autocast is unsupported for this geometry probe")
        border = 0.125
        height = 0.25 + raw[:, 3].sigmoid() * (0.2 * min(page_h - 1, page_w - 1) - 0.25)
        margin = border + max(self.baseline_fraction, 1 - self.baseline_fraction) * height
        minimum_width = (self.control_points - 1) * self.min_increment_px
        x_start = margin + raw[:, 0].sigmoid() * (page_w - 1 - 2 * margin - minimum_width)
        width = minimum_width + raw[:, 1].sigmoid() * (page_w - 1 - margin - x_start - minimum_width)
        weights = torch_f.softplus(raw[:, 4 + self.control_points:]) + 1e-3
        increments = self.min_increment_px + (width - minimum_width).clamp_min(0)[:, None] * weights / weights.sum(-1, keepdim=True)
        positions = torch.cat((increments.new_zeros((self.num_queries, 1)), increments.cumsum(-1)), dim=-1)
        x = x_start[:, None] + positions
        amplitude = 0.1 * height
        y_center = border + self.baseline_fraction * height + amplitude + raw[:, 2].sigmoid() * (
            page_h - 1 - 2 * border - height - 2 * amplitude
        )
        y = y_center[:, None] + amplitude[:, None] * raw[:, 4:4 + self.control_points].tanh()
        baseline = torch.stack((x, y), dim=-1)
        for override, shape, name in ((baseline_override, (self.num_queries, self.control_points, 2), "baseline_override"),
                                      (height_override, (self.num_queries,), "height_override")):
            if override is not None:
                if (not isinstance(override, torch.Tensor) or override.shape != shape
                        or override.dtype != image.dtype or override.device != image.device
                        or not bool(torch.isfinite(override).all())):
                    raise ValueError(f"{name} must be a finite CPU tensor of shape {shape} and image dtype")
        if baseline_override is not None:
            baseline = baseline_override
        if height_override is not None:
            height = height_override
        frame = FeatureFrame((page_w, page_h), (0.0, 0.0), (1.0, 1.0))
        results = []
        for query in range(self.num_queries):
            points, h = baseline[query], height[query]
            lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
            if bool((lengths <= self.min_segment_length_px).any()):
                results.append(CurvedQueryPrediction("degenerate_curve", failure_reason="near_degenerate_predicted_segment"))
                continue
            if not bool(torch.isfinite(lengths).all()) or not bool(torch.isfinite(lengths.sum())):
                results.append(CurvedQueryPrediction("numerical_geometry_failure", failure_reason="nonfinite_segment_length"))
                continue
            if bool(h <= 0):
                results.append(CurvedQueryPrediction("invalid_height", failure_reason="nonpositive_predicted_height"))
                continue
            required = _required_steps(lengths.sum(), self.spacing_px)
            if required > self.max_samples:
                results.append(CurvedQueryPrediction("sampling_cap_exceeded", required_steps=required))
                continue
            try:
                grid = curved_band_grid(
                    points, h, frame, (page_h, page_w), band_rows=self.band_rows,
                    spacing_px=self.spacing_px, baseline_fraction=self.baseline_fraction,
                    orientation="ltr", max_samples=self.max_samples, penalty_scale_px=self.penalty_scale_px,
                )
            except CurveSamplingCapError as exc:
                results.append(CurvedQueryPrediction("sampling_cap_exceeded", failure_reason=str(exc), required_steps=required))
                continue
            except CurveNumericalError as exc:
                results.append(CurvedQueryPrediction("numerical_geometry_failure", failure_reason=str(exc)))
                continue
            sampled = sample_curved_band(features, grid, frame, detach_geometry=detach_geometry)
            logits = self.ctc_head(sampled.as_ctc_sequence(require_valid=False))
            results.append(CurvedQueryPrediction("sampled" if bool(grid.line_valid) else "invalid_geometry",
                                                grid=grid, ctc_logits=logits, required_steps=required))
        source = context[:, None].expand(-1, self.num_queries, -1)
        target = context[None].expand(self.num_queries, -1, -1)
        return CurvedPagePrediction(
            feature_map=features, baseline_xy=baseline, height_px=height,
            presence_logits=self.presence_head(context).squeeze(-1),
            order_logits=self.order_head(torch.cat((source, target), dim=-1)).squeeze(-1),
            queries=tuple(results), frame=frame, baseline_fraction=self.baseline_fraction,
            spacing_px=self.spacing_px, max_samples=self.max_samples, penalty_scale_px=self.penalty_scale_px,
            alphabet_size=self.alphabet_size, min_segment_length_px=self.min_segment_length_px,
        )


def _target_data(prediction: CurvedPagePrediction, targets: CurvedPageTargets) -> tuple[list[Any], Any]:
    require_torch()
    prediction.frame.validate()
    reference = prediction.baseline_xy
    if not isinstance(reference, torch.Tensor) or reference.ndim != 3 or reference.shape[-1] != 2 or reference.shape[1] < 2:
        raise ValueError("predicted curves must have shape [Q,K>=2,2]")
    q = len(reference)
    if not 1 <= q <= 6 or len(prediction.queries) != q or reference.dtype not in (torch.float32, torch.float64) or reference.device.type != "cpu":
        raise ValueError("prediction must have 1..6 CPU float32/float64 queries")
    for value, shape in ((prediction.height_px, (q,)), (prediction.presence_logits, (q,)), (prediction.order_logits, (q, q))):
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError("prediction tensors must have consistent shapes, device and dtype")
    if any(not bool(torch.isfinite(value).all()) for value in (reference, prediction.height_px, prediction.presence_logits, prediction.order_logits)):
        raise ValueError("prediction tensors must be finite")
    if not isinstance(targets.baseline_xy, (list, tuple)) or len(targets.baseline_xy) > q:
        raise ValueError("target baselines must be a list/tuple with no more lines than queries")
    n = len(targets.baseline_xy)
    heights = torch.as_tensor(targets.height_px, dtype=reference.dtype, device=reference.device).detach()
    if heights.shape != (n,) or not bool(torch.isfinite(heights).all()) or bool((heights <= 0).any()):
        raise ValueError("GT heights must be finite positive values of shape [N]")
    curves = []
    width, height = prediction.frame.original_size_wh
    for supplied in targets.baseline_xy:
        curve = torch.as_tensor(supplied, dtype=reference.dtype, device=reference.device).detach()
        if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2 or not bool(torch.isfinite(curve).all()):
            raise ValueError("every GT curve must have shape [K_i>=2,2] with finite points")
        if bool(curve[-1, 0] <= curve[0, 0]):
            raise ValueError("GT endpoints must follow the declared ltr orientation")
        if bool(((curve[:, 0] < 0) | (curve[:, 0] > width - 1) | (curve[:, 1] < 0) | (curve[:, 1] > height - 1)).any()):
            raise ValueError("GT baseline coordinates must be within original page pixels")
        curves.append(curve)
    return curves, heights


def _match(prediction: CurvedPagePrediction, curves: list[Any], heights: Any) -> tuple[int, ...]:
    if not curves:
        return ()
    predicted = prediction.baseline_xy.detach().double()
    pred_height = prediction.height_px.detach().double()
    costs = []
    for target, h in zip(curves, heights):
        target = target.double()
        endpoint = torch.linalg.vector_norm(torch.stack((predicted[:, 0] - target[0], predicted[:, -1] - target[-1]), dim=1), dim=-1).mean(-1)
        cost = (endpoint + (pred_height - h.double()).abs()) / h.double()
        costs.append(torch.nan_to_num(cost, nan=1e100, posinf=1e100, neginf=1e100).clamp(max=1e100).tolist())
    return min(permutations(range(len(predicted)), len(curves)),
               key=lambda assignment: sum(costs[target][query] for target, query in enumerate(assignment)))


def match_curved_lines(prediction: CurvedPagePrediction, targets: CurvedPageTargets) -> tuple[int, ...]:
    """Exact detached endpoint+height matching, including invalid predictions.

    Cost is mean ordered endpoint distance / GT height plus absolute height
    error / GT height. It ignores text, sampling validity and vertex count.
    This intentionally simple diagnostic rule is not the pipeline's curved
    matching contract. Ties use lexicographic query order.
    """
    curves, heights = _target_data(prediction, targets)
    return _match(prediction, curves, heights)


def _labels(targets: CurvedPageTargets, n: int, alphabet_size: int, reference: Any) -> list[Any]:
    if not isinstance(targets.texts, (tuple, list)) or len(targets.texts) != n:
        raise ValueError("texts must contain one integer sequence per GT curve")
    result = []
    for supplied in targets.texts:
        if isinstance(supplied, (list, tuple)) and not supplied:
            labels = torch.empty(0, dtype=torch.long, device=reference.device)
        else:
            try:
                labels = torch.as_tensor(supplied, device=reference.device)
            except (ValueError, TypeError) as exc:
                raise ValueError("GT text must be a one-dimensional integer sequence") from exc
        if labels.ndim != 1 or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("GT text must be a one-dimensional integer sequence")
        if bool(((labels <= 0) | (labels >= alphabet_size)).any()):
            raise ValueError("GT labels must exclude blank 0 and lie within the alphabet")
        result.append(labels.long().detach())
    return result


def curved_page_losses(prediction: CurvedPagePrediction, targets: CurvedPageTargets) -> CurvedPageLosses:
    """Account for every GT line, including sampled or geometrically excluded.

    CTC signal, ordinary geometry, explicit anchor repair, boundary and fold
    sums all divide by the fixed number of matched GT lines. Presence covers
    all queries. Relation supervision follows the matched target identities.
    Regular geometry uses GT-height/2 spacing and an independent 256-point cap;
    a remaining cap exclusion is counted and receives an ordered-endpoint plus
    height fallback in ``repair``. That fallback has no resampling and no
    gradient to interior vertices. It preserves a limited recovery signal,
    without certifying the curve's interior or promising eventual inclusion.
    Nonfinite computed geometry is explicitly counted without a fake repair.

    Each CTC exclusion adds max_samples*log(alphabet_size), divided by matched
    count. This is a fixed uniform-logit reference upper bound for a feasible
    target, not an upper bound for arbitrary model logits. Counts and observed
    signal must accompany this synthetic objective; do not select checkpoints
    or claim OCR improvements using its total.
    """
    curves, heights = _target_data(prediction, targets)
    assignment = _match(prediction, curves, heights)
    n, q = len(curves), len(prediction.queries)
    labels = _labels(targets, n, prediction.alphabet_size, prediction.baseline_xy)
    denominator = max(n, 1)
    zero_geometry = prediction.baseline_xy.reshape(-1)[0] * 0 + prediction.height_px[0] * 0
    ctc_signal = prediction.feature_map.sum() * 0
    geometry, repair, boundary, fold = (zero_geometry for _ in range(4))
    line_statuses = ["unmatched:" + query.status for query in prediction.queries]
    geometry_statuses = ["unmatched"] * q
    counts = {name: 0 for name in (
        "included_ctc", "excluded_ctc", "excluded_invalid_geometry", "excluded_degenerate_curve",
        "excluded_sampling_cap_exceeded", "excluded_infeasible_length", "excluded_invalid_height",
        "excluded_numerical_geometry_failure", "regular_geometry_lines", "repair_geometry_lines", "repair_cap_geometry_lines",
        "excluded_geometry_sampling_cap", "excluded_geometry_numerical", "supported_target_tokens", "excluded_target_tokens",
    )}
    counts.update(num_queries=q, ground_truth_lines=n, matched=n, unmatched_queries=q - n,
                  ctc_denominator=n, geometry_denominator=n, ground_truth_tokens=sum(len(text) for text in labels))
    spacings = []
    for target_index, query_index in enumerate(assignment):
        query = prediction.queries[query_index]
        points, h = prediction.baseline_xy[query_index], prediction.height_px[query_index]
        target_curve, target_h, text = curves[target_index], heights[target_index], labels[target_index]
        spacing = float(target_h) * 0.5
        spacings.append(spacing)
        height_error = (h - target_h).abs() / target_h
        lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
        if bool((lengths <= prediction.min_segment_length_px).any()):
            try:
                repaired = repair_degenerate_curve_loss(
                    points, target_curve, target_height_px=target_h,
                    orientation=prediction.orientation, min_segment_length_px=prediction.min_segment_length_px,
                )
                candidate = repaired.total + height_error
                if not bool(torch.isfinite(candidate)):
                    raise CurveNumericalError("repair geometry total must be finite")
                repair = repair + candidate
                counts["repair_geometry_lines"] += 1
                geometry_statuses[query_index] = "repair_degenerate"
            except CurveNumericalError:
                counts["excluded_geometry_numerical"] += 1
                geometry_statuses[query_index] = "excluded_numerical"
        else:
            try:
                regular = curve_geometry_loss(
                    points, target_curve, target_height_px=target_h,
                    spacing_px=spacing, max_samples=256, orientation=prediction.orientation,
                )
                candidate = regular.total + height_error
                if not bool(torch.isfinite(candidate)):
                    raise CurveNumericalError("regular geometry total must be finite")
                geometry = geometry + candidate
                counts["regular_geometry_lines"] += 1
                geometry_statuses[query_index] = "regular"
            except CurveSamplingCapError:
                counts["excluded_geometry_sampling_cap"] += 1
                # No resampling: retain a declared endpoint/height signal when
                # both reader and regular geometry sample budgets are exceeded.
                # Interior vertices remain unsupervised by this narrow fallback.
                endpoints = torch.stack((points[0] - target_curve[0], points[-1] - target_curve[-1]))
                candidate = torch.linalg.vector_norm(endpoints, dim=-1).mean() / target_h + height_error
                if bool(torch.isfinite(candidate)):
                    repair = repair + candidate
                    counts["repair_cap_geometry_lines"] += 1
                    geometry_statuses[query_index] = "repair_cap"
                else:
                    counts["excluded_geometry_numerical"] += 1
                    geometry_statuses[query_index] = "excluded_numerical"
            except CurveNumericalError:
                counts["excluded_geometry_numerical"] += 1
                geometry_statuses[query_index] = "excluded_numerical"
        if query.grid is not None:
            boundary = boundary + query.grid.boundary_loss
            fold = fold + query.grid.fold_loss
        if query.ctc_logits is not None and query.grid is not None:
            text_loss = masked_curve_ctc_loss(query.ctc_logits, query.grid, text)
            ctc_signal = ctc_signal + text_loss.loss
            status = text_loss.status
        else:
            status = "excluded_" + query.status
        line_statuses[query_index] = status
        if status == "included":
            counts["included_ctc"] += 1
            counts["supported_target_tokens"] += len(text)
        else:
            counts["excluded_ctc"] += 1
            counts["excluded_target_tokens"] += len(text)
            if status not in counts:
                raise ValueError(f"unclassified query exclusion: {status}")
            counts[status] += 1
    ctc_signal = ctc_signal / denominator
    charge_per_exclusion = prediction.max_samples * math.log(prediction.alphabet_size)
    exclusion_charge = charge_per_exclusion * counts["excluded_ctc"] / denominator
    ctc = ctc_signal + exclusion_charge
    geometry, repair, boundary, fold = (loss / denominator for loss in (geometry, repair, boundary, fold))
    indices = torch.tensor(assignment, dtype=torch.long, device=prediction.baseline_xy.device)
    presence_target = torch.zeros_like(prediction.presence_logits)
    presence_target[indices] = 1
    presence = torch_f.binary_cross_entropy_with_logits(prediction.presence_logits, presence_target)
    order = prediction.order_logits.sum() * 0
    if targets.order is not None:
        relation = torch.as_tensor(targets.order, device=prediction.order_logits.device).detach()
        if relation.shape != (n, n) or not bool(((relation == -1) | (relation == 0) | (relation == 1)).all()):
            raise ValueError("GT order must have shape [N,N] and entries -1,0,1")
        if bool((relation.diagonal() == 1).any()):
            raise ValueError("GT order cannot contain self edges")
        known = (relation >= 0) & ~torch.eye(n, dtype=torch.bool, device=relation.device)
        if bool(known.any()):
            matched = prediction.order_logits[indices[:, None], indices[None, :]]
            order = torch_f.binary_cross_entropy_with_logits(matched[known], relation[known].to(matched.dtype))
    losses = {"ctc": ctc, "geometry": geometry, "repair": repair, "presence": presence,
              "order": order, "boundary": boundary, "fold": fold}
    losses["total"] = sum(losses.values())
    included_mean = None if not counts["included_ctc"] else float(ctc_signal.detach()) * denominator / counts["included_ctc"]
    diagnostics = {
        "ctc_signal": ctc_signal, "ctc_exclusion_charge": exclusion_charge,
        "ctc_charge_per_exclusion": charge_per_exclusion,
        "ctc_included_mean": included_mean, "ctc_charge_rule": "max_samples_log_alphabet",
        "geometry_spacing_px_by_target": spacings, "geometry_max_samples": 256,
        "geometry_spacing_rule": "target_height_divided_by_two",
        "geometry_cap_repair_rule": "ordered_endpoint_mean_plus_height_over_target_height",
        "geometry_cap_repair_supervises_interior_vertices": False,
        "matching_rule": "ordered_endpoints_plus_height_over_target_height",
        "checkpoint_selection_allowed": False,
    }
    return CurvedPageLosses(losses, counts, assignment, tuple(line_statuses), tuple(geometry_statuses), diagnostics)


def export_curved_page(
    prediction: CurvedPagePrediction, *, image_sha256: str, page_key: str, document_key: str,
) -> dict[str, Any]:
    """Export only predictions and statuses, without consulting any GT."""
    require_torch()
    if not isinstance(image_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", image_sha256) is None:
        raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
    if not isinstance(page_key, str) or not page_key.strip() or not isinstance(document_key, str) or not document_key.strip():
        raise ValueError("page_key and document_key must be nonempty")
    lines = []
    for index, query in enumerate(prediction.queries):
        tokens = None
        if query.ctc_logits is not None and query.grid is not None and bool(query.grid.line_valid):
            tokens, previous = [], None
            for label in query.ctc_logits.detach().argmax(-1)[:, 0].tolist():
                if label != 0 and label != previous:
                    tokens.append(label)
                previous = label
        lines.append({
            "query_index": index, "status": query.status, "failure_reason": query.failure_reason,
            "baseline_xy": prediction.baseline_xy[index].detach().tolist(),
            "height_px": float(prediction.height_px[index].detach()),
            "presence_probability": float(prediction.presence_logits[index].detach().sigmoid()),
            "required_steps": query.required_steps,
            "line_valid": None if query.grid is None else bool(query.grid.line_valid),
            "ctc_token_ids": tokens, "word_boxes": None,
        })
    width, height = prediction.frame.original_size_wh
    return {
        "schema_version": "babaorum.joint_curved_page.diagnostic.v1",
        "status": "UNTRAINED_SYNTHETIC_MECHANISM_ONLY",
        "image_sha256": image_sha256.lower(), "page_key": page_key, "document_key": document_key,
        "coordinate_frame": {"name": "original_encoded_pixels", "width": width, "height": height,
                             "coordinate_convention": "pixel_centers_0_width_minus_1_height_minus_1",
                             "before_exif_orientation": True},
        "baseline_fraction": prediction.baseline_fraction, "orientation": prediction.orientation,
        "queries": lines, "word_localization_available": False,
        "order_probabilities": prediction.order_logits.detach().sigmoid().tolist(),
        "order_status": "unresolved_logits_only",
    }
