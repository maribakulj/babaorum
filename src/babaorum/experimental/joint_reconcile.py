"""Provenance-preserving geometric candidate graph for synthetic probes.

This module does not select, merge, suppress, or score OCR text. It accepts
already emitted line proposals and reports geometric relationships only. All
coordinates use continuous pixel-edge coordinates in the page frame. Local
coordinates are inverted through an explicit crop/scale/padding transform;
rotation and mirroring are deliberately unsupported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal


Source = Literal["global_page", "local_block"]
GeometryStatus = Literal["available", "error", "missing"]


@dataclass(frozen=True)
class CropTransform:
    """Actual clipped crop and its placement in a padded, isotropically scaled canvas.

    ``actual_rect_page`` is [left, top, right, bottom] in continuous page-edge
    coordinates, after clipping to the page. Padding order is left, top, right,
    bottom. Local baseline coordinates are measured in the padded canvas.
    """

    actual_rect_page: tuple[float, float, float, float]
    scale: float
    padding_ltrb: tuple[float, float, float, float]
    rotation_degrees: float = 0.0
    mirrored: bool = False

    def validate(self) -> None:
        rect = self.actual_rect_page
        pad = self.padding_ltrb
        if len(rect) != 4 or len(pad) != 4:
            raise ValueError("crop rectangle and padding must each have four values")
        if any(not math.isfinite(v) for v in (*rect, *pad, self.scale, self.rotation_degrees)):
            raise ValueError("crop transform values must be finite")
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0 or self.scale <= 0 or any(v < 0 for v in pad):
            raise ValueError("crop rectangle, scale, or padding is invalid")
        if self.rotation_degrees != 0 or self.mirrored:
            raise ValueError("crop transforms do not support rotation or mirroring")

    def local_to_page(self, point_xy: tuple[float, float]) -> tuple[float, float]:
        self.validate()
        x, y = point_xy
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("local point must be finite")
        x0, y0, _, _ = self.actual_rect_page
        left, top, _, _ = self.padding_ltrb
        return x0 + (x - left) / self.scale, y0 + (y - top) / self.scale

    def content_rect_local(self) -> tuple[float, float, float, float]:
        self.validate()
        x0, y0, x1, y1 = self.actual_rect_page
        left, top, right, bottom = self.padding_ltrb
        return left, top, left + (x1 - x0) * self.scale, top + (y1 - y0) * self.scale


@dataclass(frozen=True)
class LineProposal:
    proposal_id: str
    source: Source
    page_key: str
    document_key: str | None
    page_size_wh: tuple[float, float]
    page_sha256: str
    query_id: str
    input_sha256: str
    checkpoint_sha256: str | None
    geometry_status: GeometryStatus
    baseline_xy: tuple[tuple[float, float], ...] | None
    # Band extents use the proposal's own input frame (local crop pixels for
    # local_block, page pixels for global_page).
    band_normal_negative_px: float | None
    band_normal_positive_px: float | None
    orientation: str
    text_status: str
    text: str | None
    token_ids: tuple[int, ...] | None
    ctc_status: str
    block_id: str | None = None
    crop_transform: CropTransform | None = None


@dataclass(frozen=True)
class ReconcileConfig:
    # Deliberately fixed synthetic-probe scale; not a pipeline/evaluation rule.
    normalization_height_px: float = 20.0
    distance_tolerance_heights: float = 0.5
    endpoint_tolerance_heights: float = 1.0
    duplicate_coverage: float = 0.75
    multi_local_group_coverage: float = 0.75
    multi_local_group_requires_each_partial: bool = True
    sample_step_heights: float = 0.125
    crop_edge_epsilon_heights: float = 0.05
    truncation_extension_heights: float = 1.0

    def validate(self) -> None:
        positive = (self.normalization_height_px, self.distance_tolerance_heights,
                    self.endpoint_tolerance_heights, self.sample_step_heights,
                    self.truncation_extension_heights)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("reconciliation scales and tolerances must be positive and finite")
        if (not 0 < self.duplicate_coverage <= 1 or not math.isfinite(self.duplicate_coverage)
                or not 0 < self.multi_local_group_coverage <= 1
                or not math.isfinite(self.multi_local_group_coverage)
                or not math.isfinite(self.crop_edge_epsilon_heights)
                or self.crop_edge_epsilon_heights < 0):
            raise ValueError("coverage or crop-edge epsilon is invalid")
        if type(self.multi_local_group_requires_each_partial) is not bool:
            raise ValueError("multi_local_group_requires_each_partial must be boolean")


def _validate_proposal(p: LineProposal) -> None:
    if not isinstance(p, LineProposal):
        raise ValueError("all inputs must be LineProposal instances")
    if not all(isinstance(x, str) and x for x in (
        p.proposal_id, p.page_key, p.query_id, p.input_sha256, p.page_sha256
    )):
        raise ValueError("proposal, page, query and input identities must be nonempty strings")
    if (len(p.page_size_wh) != 2 or any(not math.isfinite(v) or v <= 0 for v in p.page_size_wh)):
        raise ValueError("page_size_wh must contain finite positive width and height")
    if p.document_key is not None and (not isinstance(p.document_key, str) or not p.document_key):
        raise ValueError("document_key must be null or a nonempty string")
    if p.source not in ("global_page", "local_block"):
        raise ValueError("unsupported proposal source")
    if p.source == "local_block":
        if not p.block_id or p.crop_transform is None:
            raise ValueError("local proposals require block identity and crop transform")
        p.crop_transform.validate()
        x0, y0, x1, y1 = p.crop_transform.actual_rect_page
        width, height = p.page_size_wh
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError("actual clipped crop rectangle must lie inside page bounds")
    elif p.block_id is not None or p.crop_transform is not None:
        raise ValueError("global proposals cannot carry local crop metadata")
    if p.geometry_status not in ("available", "error", "missing"):
        raise ValueError("unsupported geometry status")
    if p.geometry_status == "available":
        if p.baseline_xy is None or len(p.baseline_xy) < 2:
            raise ValueError("available geometry requires at least two baseline points")
        if any(len(xy) != 2 or any(not math.isfinite(v) for v in xy) for xy in p.baseline_xy):
            raise ValueError("baseline coordinates must be finite pairs")
    elif p.baseline_xy is not None:
        raise ValueError("unavailable geometry must not carry a baseline")
    if p.orientation not in ("ltr", "rtl", "ttb", "btt", "unknown"):
        raise ValueError("unsupported orientation")
    if p.text_status not in ("available", "missing", "error") or p.ctc_status not in (
        "feasible", "empty", "unavailable", "error"
    ):
        raise ValueError("text or inference-time CTC status is invalid")
    if p.text_status == "available" and p.text is None:
        raise ValueError("available text must preserve a string, including empty text")
    if p.text_status != "available" and p.text is not None:
        raise ValueError("unavailable text must not carry a value")
    for name, value, nullable in (("input_sha256", p.input_sha256, False),
                                  ("page_sha256", p.page_sha256, False),
                                  ("checkpoint_sha256", p.checkpoint_sha256, True)):
        if value is None and nullable:
            continue
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be {'null or ' if nullable else ''}lowercase SHA-256 hex")
    extents = (p.band_normal_negative_px, p.band_normal_positive_px)
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in extents):
        raise ValueError("band normal extents must be finite and nonnegative")
    if (extents[0] is None) != (extents[1] is None):
        raise ValueError("both band normal extents must be supplied together")


def _resample(points: tuple[tuple[float, float], ...], step: float) -> list[tuple[float, float]]:
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    if any(length == 0 for length in lengths):
        raise ValueError("baseline contains a zero-length segment")
    total = sum(lengths)
    count = max(2, math.ceil(total / step) + 1)
    targets = [total * i / (count - 1) for i in range(count)]
    out, segment, prefix = [], 0, 0.0
    for target in targets:
        while segment < len(lengths) - 1 and target > prefix + lengths[segment]:
            prefix += lengths[segment]
            segment += 1
        ratio = (target - prefix) / lengths[segment]
        a, b = points[segment], points[segment + 1]
        out.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
    return out


def _distance_to_polyline(point: tuple[float, float], points: tuple[tuple[float, float], ...]) -> float:
    best = math.inf
    px, py = point
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom == 0:
            continue
        t = min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _coverage(source, target, tolerance: float, step: float) -> float:
    points = _resample(source, step)
    return sum(_distance_to_polyline(p, target) <= tolerance for p in points) / len(points)


def _page_baseline(p: LineProposal) -> tuple[tuple[float, float], ...] | None:
    if p.geometry_status != "available":
        return None
    if p.source == "global_page":
        return p.baseline_xy
    if p.crop_transform is None or p.baseline_xy is None:
        raise ValueError("internal invariant failed: available local geometry lacks its transform")
    return tuple(p.crop_transform.local_to_page(point) for point in p.baseline_xy)


def _nonzero_segments(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...] | None:
    cleaned = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    return tuple(cleaned) if len(cleaned) >= 2 else None


def _point_segment_distance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.dist(p, a)
    t = min(1.0, max(0.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_distance(a, b, c, d) -> float:
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    def on_segment(x, y, p):
        return (min(x[0], y[0]) - 1e-12 <= p[0] <= max(x[0], y[0]) + 1e-12
                and min(x[1], y[1]) - 1e-12 <= p[1] <= max(x[1], y[1]) + 1e-12)
    intersects = ((o1 * o2 < 0 and o3 * o4 < 0)
                  or (o1 == 0 and on_segment(a, b, c))
                  or (o2 == 0 and on_segment(a, b, d))
                  or (o3 == 0 and on_segment(c, d, a))
                  or (o4 == 0 and on_segment(c, d, b)))
    if intersects:
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def _crop_flags(p: LineProposal, page_points, config: ReconcileConfig) -> dict:
    if p.source != "local_block" or p.crop_transform is None or p.baseline_xy is None:
        width, height = p.page_size_wh
        outside_page = bool(page_points) and any(
            x < 0 or x > width or y < 0 or y > height for x, y in page_points
        )
        return {"touches_crop_content_edge": False,
                "has_points_in_padding_or_outside_content": False,
                "has_points_outside_page": outside_page,
                "may_be_truncated_by_crop_band": False,
                "touches_internal_crop_edge": False}
    crop = p.crop_transform
    x0, y0, x1, y1 = crop.content_rect_local()
    eps = config.crop_edge_epsilon_heights * config.normalization_height_px * crop.scale
    extension = config.truncation_extension_heights * config.normalization_height_px * crop.scale
    if p.band_normal_negative_px is not None:
        extension = max(p.band_normal_negative_px, p.band_normal_positive_px)
    touches = any(
        (x0 <= x <= x1 and y0 <= y <= y1 and min(x - x0, x1 - x, y - y0, y1 - y) <= eps)
        for x, y in p.baseline_xy
    )
    all_boundaries = [((x0, y0), (x0, y1)), ((x1, y0), (x1, y1)),
                      ((x0, y0), (x1, y0)), ((x0, y1), (x1, y1))]
    page_width, page_height = p.page_size_wh
    actual_x0, actual_y0, actual_x1, actual_y1 = crop.actual_rect_page
    internal_boundaries = []
    if actual_x0 > 0:
        internal_boundaries.append(((x0, y0), (x0, y1)))
    if actual_x1 < page_width:
        internal_boundaries.append(((x1, y0), (x1, y1)))
    if actual_y0 > 0:
        internal_boundaries.append(((x0, y0), (x1, y0)))
    if actual_y1 < page_height:
        internal_boundaries.append(((x0, y1), (x1, y1)))
    min_boundary_distance = math.inf
    min_any_boundary_distance = math.inf
    for (ax, ay), (bx, by) in zip(p.baseline_xy, p.baseline_xy[1:]):
        for c, d in all_boundaries:
            min_any_boundary_distance = min(
                min_any_boundary_distance, _segments_distance((ax, ay), (bx, by), c, d)
            )
        for c, d in internal_boundaries:
            min_boundary_distance = min(min_boundary_distance, _segments_distance((ax, ay), (bx, by), c, d))
    touches = touches or min_any_boundary_distance <= eps
    touches_internal = min_boundary_distance <= eps
    outside = any(x < x0 or x > x1 or y < y0 or y > y1 for x, y in p.baseline_xy)
    width, height = p.page_size_wh
    outside_page = bool(page_points) and any(
        x < 0 or x > width or y < 0 or y > height for x, y in page_points
    )
    return {"touches_crop_content_edge": touches,
            "has_points_in_padding_or_outside_content": outside,
            "has_points_outside_page": outside_page,
            "may_be_truncated_by_crop_band": min_boundary_distance <= extension,
            "touches_internal_crop_edge": touches_internal}


def build_reconciliation_graph(proposals: tuple[LineProposal, ...] | list[LineProposal],
                               config: ReconcileConfig = ReconcileConfig()) -> dict:
    """Return all proposals and deterministic geometric cross-source candidates.

    Every global↔local and local↔local pair on a page gets evaluated, including
    pairs with unavailable geometry (which receive an explicit no-edge reason).
    Edge labels are candidates only. Text, token ids, CTC statuses and sources
    are copied side by side and never reconciled or ranked.
    """
    config.validate()
    if not isinstance(proposals, (tuple, list)):
        raise ValueError("proposals must be a list or tuple")
    for p in proposals:
        _validate_proposal(p)
    ids = [p.proposal_id for p in proposals]
    if len(ids) != len(set(ids)):
        raise ValueError("proposal_id must be unique across both passes")
    query_keys = [(p.source, p.input_sha256, p.query_id) for p in proposals]
    if len(query_keys) != len(set(query_keys)):
        raise ValueError("query_id must be unique within each source/input crop")
    frame_by_page = {}
    for p in proposals:
        frame = (p.document_key, p.page_size_wh, p.page_sha256)
        if p.page_key in frame_by_page and frame_by_page[p.page_key] != frame:
            raise ValueError("proposals for one page must share document identity and page frame")
        frame_by_page[p.page_key] = frame
    ordered = sorted(proposals, key=lambda p: (p.page_key, p.source, p.block_id or "", p.proposal_id))
    nodes = []
    for p in ordered:
        page_points = _page_baseline(p)
        nodes.append({
            "proposal": asdict(p), "baseline_page_xy": page_points,
            "crop_flags": _crop_flags(p, page_points, config),
        })
    node_by_id = {node["proposal"]["proposal_id"]: node for node in nodes}
    edges = []
    partial_members: dict[str, list[str]] = {}
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if left.page_key != right.page_key or (left.source == right.source == "global_page"):
                continue
            a, b = _page_baseline(left), _page_baseline(right)
            edge = {"source_ids": [left.proposal_id, right.proposal_id], "labels": [],
                    "status": "no_candidate", "normalization_height_px": config.normalization_height_px,
                    "rule": {"distance_tolerance_heights": config.distance_tolerance_heights,
                             "endpoint_tolerance_heights": config.endpoint_tolerance_heights,
                             "duplicate_coverage": config.duplicate_coverage,
                             "sample_step_heights": config.sample_step_heights}}
            if a is None or b is None:
                edge["reason"] = "geometry_unavailable"
                edges.append(edge)
                continue
            a_clean, b_clean = _nonzero_segments(a), _nonzero_segments(b)
            if a_clean is None or b_clean is None:
                edge["reason"] = "geometry_degenerate"
                edges.append(edge)
                continue
            a, b = a_clean, b_clean
            h = config.normalization_height_px
            tol = config.distance_tolerance_heights * h
            step = config.sample_step_heights * h
            ca, cb = _coverage(a, b, tol, step), _coverage(b, a, tol, step)
            # Ordered endpoints: compare direct and reversed pairing, retaining
            # the lower cost while separately flagging orientation disagreement.
            direct = math.dist(a[0], b[0]) + math.dist(a[-1], b[-1])
            reverse = math.dist(a[0], b[-1]) + math.dist(a[-1], b[0])
            endpoints_compatible = min(direct, reverse) <= 2 * config.endpoint_tolerance_heights * h
            orient_conflict = (left.orientation != "unknown" and right.orientation != "unknown"
                               and left.orientation != right.orientation)
            reversed_geometry = reverse < direct
            if ca >= config.duplicate_coverage and cb >= config.duplicate_coverage and endpoints_compatible:
                edge["labels"].append("duplicate_candidate")
            elif max(ca, cb) >= config.duplicate_coverage:
                edge["labels"].append("partial_overlap_candidate")
            geometric_candidate = bool(edge["labels"])
            if (left.source == "global_page" and right.source == "local_block"
                    and cb >= config.duplicate_coverage
                    and (not config.multi_local_group_requires_each_partial
                         or ca < config.multi_local_group_coverage)):
                partial_members.setdefault(left.proposal_id, []).append(right.proposal_id)
            if geometric_candidate and orient_conflict:
                edge["labels"].append("direction_conflict")
            if geometric_candidate and reversed_geometry:
                edge["labels"].append("reversed_geometry")
            lf, rf = node_by_id[left.proposal_id]["crop_flags"], node_by_id[right.proposal_id]["crop_flags"]
            if geometric_candidate and (lf["may_be_truncated_by_crop_band"] or rf["may_be_truncated_by_crop_band"]):
                edge["labels"].append("crop_truncation_possible")
            if geometric_candidate and (lf["has_points_in_padding_or_outside_content"]
                                        or rf["has_points_in_padding_or_outside_content"]):
                edge["labels"].append("padding_or_outside_content")
            edge["coverage_left_on_right"] = ca
            edge["coverage_right_on_left"] = cb
            edge["endpoint_distance_heights"] = min(direct, reverse) / h
            edge["status"] = "candidate" if geometric_candidate else "no_candidate"
            edges.append(edge)
    by_id = {p.proposal_id: p for p in ordered}
    groups = []
    for global_id, local_ids in sorted(partial_members.items()):
        unique_locals = sorted(set(local_ids))
        blocks = {by_id[item].block_id for item in unique_locals}
        if len(unique_locals) < 2 or len(blocks) < 2:
            continue
        global_points = _nonzero_segments(_page_baseline(by_id[global_id]))
        if global_points is None:
            raise ValueError("internal invariant failed: candidate group has degenerate global geometry")
        sample_step = config.sample_step_heights * config.normalization_height_px
        tol = config.distance_tolerance_heights * config.normalization_height_px
        samples = _resample(global_points, sample_step)
        local_lines = [_nonzero_segments(_page_baseline(by_id[item])) for item in unique_locals]
        union_mask = [
            any(_distance_to_polyline(point, local_line) <= tol for local_line in local_lines)
            for point in samples
        ]
        union_coverage = sum(union_mask) / len(samples)
        individual_coverages = {
            item: _coverage(global_points, line, tol, sample_step)
            for item, line in zip(unique_locals, local_lines)
        }
        best_single_coverage = max(individual_coverages.values())
        if (union_coverage >= config.multi_local_group_coverage
                and (not config.multi_local_group_requires_each_partial
                     or all(value < config.multi_local_group_coverage
                            for value in individual_coverages.values()))):
            member_edges = []
            for local_id in unique_locals:
                member = next((edge for edge in edges
                               if edge["source_ids"] == [global_id, local_id]), None)
                if member is None:
                    raise ValueError("internal invariant failed: candidate group member edge is missing")
                member_edges.append({"source_ids": member["source_ids"], "labels": member["labels"]})
            groups.append({
                "global_proposal_id": global_id,
                "local_proposal_ids": unique_locals,
                "block_ids": sorted(blocks),
                "global_arc_coverage_union": union_coverage,
                "individual_global_arc_coverage": individual_coverages,
                "increment_over_best_single_coverage": union_coverage - best_single_coverage,
                "redundant_coverage_sum_minus_union": sum(individual_coverages.values()) - union_coverage,
                "member_edges": member_edges,
                "status": "multi_local_coverage_candidate",
                "selection": None,
            })
    return {
        "schema_version": "babaorum.experimental_reconciliation.v1",
        "scope": "SYNTHETIC_GEOMETRIC_CANDIDATES_ONLY",
        "selection": None,
        "config": asdict(config),
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "limitations": [
            "No proposal is selected, merged, suppressed, or treated as ground truth.",
            "Candidate thresholds and the fixed normalization height are synthetic-probe settings, not evaluation rules.",
            "Text and CTC outputs remain side by side; no text-span matching or raw CTC score comparison is performed.",
            "Crop edge and padding flags are diagnostics; proposals are never clamped or dropped.",
            "Pair generation is quadratic and is not a production-scale reconciler.",
        ],
    }


def graph_sha256(graph: dict) -> str:
    payload = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
