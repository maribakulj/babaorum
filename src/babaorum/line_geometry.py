"""Versioned, geometry-only matching for OCR baselines.

This is a new metric (``line_geometry.v1``), not a reproduction of E27/D-211.
It uses symmetric Euclidean polyline distance and one-to-one assignment. For a
pair, both lines use equal-arc-length midpoint cells no longer than
min(4 px, GT band height / 4), with at least 16 samples each. Candidate edges
require both directional coverages to reach 0.75 within half the explicit GT
band height. A line needing more than 65,536 samples is rejected. Assignment
maximizes cardinality, then minimizes summed symmetric mean distance normalized
by GT height; the assignment runs in exact integer arithmetic (float costs
scaled by one power of two), per connected component, and among optimal
matchings returns the one with the lexicographically least sorted
(gt_id, pred_id) list. Inputs and edge construction are sorted by stable IDs, making
exact-cost ties repeatable.
"""

from __future__ import annotations

import bisect
import heapq
import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence


METRIC_ID = "babaorum.line_geometry.v1"
SENSITIVITY_POLICY_ID = "babaorum.line_geometry.sensitivity_1h.v1"
GeometryStatus = Literal["available", "missing", "error"]
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_SOURCES = {"predicted", "transferred", "ground_truth", "interpolated"}


class LineGeometryError(ValueError):
    """Invalid line geometry input."""


Point = tuple[float, float]
MAX_SAMPLES_PER_LINE = 65_536


@dataclass(frozen=True)
class GroundTruthLine:
    id: str
    baseline: Sequence[Sequence[float]] | None
    height_px: float | None
    baseline_source: str | None
    geometry_status: GeometryStatus = "available"


@dataclass(frozen=True)
class PredictedLine:
    id: str
    baseline: Sequence[Sequence[float]] | None
    baseline_source: str | None
    geometry_status: GeometryStatus = "available"


@dataclass(frozen=True)
class LineMatch:
    gt_id: str
    pred_id: str
    gt_to_pred_coverage: float
    pred_to_gt_coverage: float
    mean_symmetric_distance_px: float
    normalized_symmetric_distance: float
    orientation_dot_sign: int | None


@dataclass(frozen=True)
class LineGeometryResult:
    metric_id: str
    status: Literal["evaluated", "non_evaluable"]
    precision: float | None
    recall: float | None
    f1: float | None
    n_gt: int
    gt_evaluable_count: int
    gt_non_evaluable_count: int
    n_pred: int
    pred_geometry_unavailable_count: int
    matched_count: int
    orientation_disagreement_count: int
    orientation_unresolved_count: int
    # Counts below are diagnostic when any GT line is non-evaluable.
    unmatched_evaluable_gt_count: int
    unmatched_pred_count: int | None
    matches: tuple[LineMatch, ...]
    # Sensitivity and topology-like counts are diagnostics, not primary scores.
    sensitivity_policy_id: str | None
    sensitivity_1h_matched_count: int | None
    gt_gained_only_at_1h_count: int | None
    split_candidate_gt_count: int | None
    split_candidate_pred_count: int | None
    merge_candidate_pred_count: int | None
    merge_candidate_gt_count: int | None


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int
    pair: tuple[int, int] | None = None


def _fail(message: str) -> None:
    raise LineGeometryError(message)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(f"{label} must be representable as a finite number")
    if not math.isfinite(result):
        _fail(f"{label} must be a finite number")
    return result


def _validate_line_id(value: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _fail("line id must match [A-Za-z_][A-Za-z0-9_.-]*")


def _polyline(value: Sequence[Sequence[float]] | None, label: str) -> tuple[Point, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        _fail(f"{label} must have at least two points")
    points: list[Point] = []
    for i, point in enumerate(value):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            _fail(f"{label}[{i}] must be [x,y]")
        points.append((_finite(point[0], f"{label}[{i}].x"), _finite(point[1], f"{label}[{i}].y")))
    if not any(a != b for a, b in zip(points, points[1:])):
        _fail(f"{label} must not be degenerate")
    return tuple(points)


def _check_source(source: str | None, status: GeometryStatus, label: str) -> None:
    if status == "available":
        if not isinstance(source, str) or source not in _SOURCES:
            _fail(f"{label} must declare a recognized baseline source")
    elif source is not None:
        _fail(f"{label} must be null when geometry is {status}")


def _validated_gt(lines: Sequence[GroundTruthLine]) -> tuple[list[tuple[GroundTruthLine, tuple[Point, ...], float]], list[GroundTruthLine]]:
    if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
        _fail("ground_truth must be a sequence")
    seen: set[str] = set()
    evaluable = []
    incomplete = []
    for line in lines:
        if not isinstance(line, GroundTruthLine):
            _fail("ground_truth entries must be GroundTruthLine")
        _validate_line_id(line.id)
        if line.id in seen:
            _fail(f"duplicate ground-truth id {line.id!r}")
        seen.add(line.id)
        if not isinstance(line.geometry_status, str) or line.geometry_status not in {"available", "missing", "error"}:
            _fail(f"ground_truth {line.id}: invalid geometry_status")
        if line.geometry_status == "available":
            points = _polyline(line.baseline, f"ground_truth[{line.id}].baseline")
            if points is None:
                _fail(f"ground_truth[{line.id}] available geometry requires a baseline")
            height = _finite(line.height_px, f"ground_truth[{line.id}].height_px")
            if height <= 0:
                _fail(f"ground_truth[{line.id}].height_px must be positive")
            _check_source(line.baseline_source, line.geometry_status, f"ground_truth[{line.id}].baseline_source")
            if line.baseline_source != "ground_truth":
                _fail(f"ground_truth[{line.id}].baseline_source must be ground_truth")
            evaluable.append((line, points, height))
        else:
            if line.baseline is not None or line.height_px is not None:
                _fail(f"ground_truth[{line.id}] non-available geometry requires null baseline and height")
            _check_source(line.baseline_source, line.geometry_status, f"ground_truth[{line.id}].baseline_source")
            incomplete.append(line)
    return evaluable, incomplete


def _validated_pred(lines: Sequence[PredictedLine]) -> tuple[list[tuple[PredictedLine, tuple[Point, ...]]], int]:
    if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
        _fail("predictions must be a sequence")
    seen: set[str] = set()
    available = []
    unavailable = 0
    for line in lines:
        if not isinstance(line, PredictedLine):
            _fail("prediction entries must be PredictedLine")
        _validate_line_id(line.id)
        if line.id in seen:
            _fail(f"duplicate prediction id {line.id!r}")
        seen.add(line.id)
        if not isinstance(line.geometry_status, str) or line.geometry_status not in {"available", "missing", "error"}:
            _fail(f"prediction {line.id}: invalid geometry_status")
        if line.geometry_status == "available":
            points = _polyline(line.baseline, f"prediction[{line.id}].baseline")
            if points is None:
                _fail(f"prediction[{line.id}] available geometry requires a baseline")
            _check_source(line.baseline_source, line.geometry_status, f"prediction[{line.id}].baseline_source")
            if line.baseline_source == "ground_truth":
                _fail(f"prediction[{line.id}] cannot claim ground_truth baseline provenance")
            available.append((line, points))
        else:
            if line.baseline is not None:
                _fail(f"prediction[{line.id}] non-available geometry requires null baseline")
            _check_source(line.baseline_source, line.geometry_status, f"prediction[{line.id}].baseline_source")
            unavailable += 1
    return available, unavailable


def _length(points: tuple[Point, ...]) -> float:
    try:
        length = math.fsum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    except (OverflowError, ValueError) as exc:
        raise LineGeometryError("baseline arc length is not representable") from exc
    if not math.isfinite(length):
        _fail("baseline arc length is not finite")
    return length


def _sample(points: tuple[Point, ...], max_spacing: float) -> tuple[Point, ...]:
    """Sample arc-length cell midpoints, with 16 minimum and a hard size bound."""
    if not math.isfinite(max_spacing) or max_spacing <= 0:
        _fail("sampling cell size must be finite and positive")
    total = _length(points)
    if total <= 0:
        _fail("baseline arc length must be positive")
    cells = total / max_spacing
    if not math.isfinite(cells) or cells > MAX_SAMPLES_PER_LINE:
        _fail(f"baseline exceeds sampling limit of {MAX_SAMPLES_PER_LINE} cells per line")
    n = max(16, math.ceil(cells))
    if n > MAX_SAMPLES_PER_LINE:
        _fail(f"baseline requires {n} samples; limit is {MAX_SAMPLES_PER_LINE} per line")
    targets = [(i + 0.5) * total / n for i in range(n)]
    result: list[Point] = []
    segment = 0
    traversed = 0.0
    for target in targets:
        while segment < len(points) - 2:
            length = math.dist(points[segment], points[segment + 1])
            if traversed + length >= target:
                break
            traversed += length
            segment += 1
        a, b = points[segment], points[segment + 1]
        length = math.dist(a, b)
        if length == 0:
            result.append(a)
        else:
            ratio = (target - traversed) / length
            result.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
    return tuple(result)


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if not math.isfinite(dx) or not math.isfinite(dy):
        _fail("baseline segment difference is not representable")
    segment_length = math.hypot(dx, dy)
    if not math.isfinite(segment_length):
        _fail("baseline segment length is not finite")
    if segment_length == 0:
        delta = (p[0] - a[0], p[1] - a[1])
        if not all(math.isfinite(value) for value in delta):
            _fail("point-to-baseline distance is not representable")
        distance = math.hypot(*delta)
        if not math.isfinite(distance):
            _fail("point-to-baseline distance is not finite")
        return distance
    delta_x, delta_y = p[0] - a[0], p[1] - a[1]
    if not math.isfinite(delta_x) or not math.isfinite(delta_y):
        _fail("point-to-baseline offset is not representable")
    along = delta_x * (dx / segment_length) + delta_y * (dy / segment_length)
    if not math.isfinite(along):
        _fail("point-to-baseline projection is not finite")
    ratio = max(0.0, min(1.0, along / segment_length))
    closest = (a[0] * (1.0 - ratio) + b[0] * ratio,
               a[1] * (1.0 - ratio) + b[1] * ratio)
    delta = (p[0] - closest[0], p[1] - closest[1])
    if not all(math.isfinite(value) for value in delta):
        _fail("point-to-baseline distance is not representable")
    distance = math.hypot(*delta)
    if not math.isfinite(distance):
        _fail("point-to-baseline distance is not finite")
    return distance


def _point_polyline_distance(p: Point, line: tuple[Point, ...]) -> float:
    return min(_point_segment_distance(p, a, b) for a, b in zip(line, line[1:]))


# Below this magnitude no difference, product, hypot or projection in
# _point_segment_distance can overflow, so its finiteness checks can never
# fire. The fast path below then performs the SAME float operations in the
# same order, on per-segment values precomputed once: results are bit-identical.
_SAFE_COORDINATE = 1e150
_SegmentRow = tuple[float, float, float, float, float, float, float]


def _segment_table(points: tuple[Point, ...]) -> tuple[_SegmentRow, ...] | None:
    """Per-segment (ax, ay, bx, by, length, dx/length, dy/length), or None if unsafe."""
    if any(abs(x) > _SAFE_COORDINATE or abs(y) > _SAFE_COORDINATE for x, y in points):
        return None
    rows = []
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            rows.append((a[0], a[1], b[0], b[1], 0.0, 0.0, 0.0))
        else:
            rows.append((a[0], a[1], b[0], b[1], length, dx / length, dy / length))
    return tuple(rows)


def _point_table_distance(p: Point, table: tuple[_SegmentRow, ...]) -> float:
    """Fast path of _point_polyline_distance for coordinates within _SAFE_COORDINATE."""
    px, py = p
    best = None
    hypot = math.hypot
    for ax, ay, bx, by, length, ux, uy in table:
        if length == 0:
            distance = hypot(px - ax, py - ay)
        else:
            along = (px - ax) * ux + (py - ay) * uy
            ratio = max(0.0, min(1.0, along / length))
            distance = hypot(px - (ax * (1.0 - ratio) + bx * ratio),
                             py - (ay * (1.0 - ratio) + by * ratio))
        if best is None or distance < best:
            best = distance
    return best


def _bounds(points: tuple[Point, ...]) -> tuple[float, float, float, float]:
    return (min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points))


def _bounds_distance_lower_bound(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    *,
    height: float,
) -> float:
    """Conservative lower bound for polyline distance, or zero if uncertain.

    A pair farther than one GT height cannot contribute to either the primary
    (0.5 h) match or its 1 h sensitivity, nor to the primary split/merge
    adjacency counters. The rounding guard deliberately retains borderline
    pairs for the exact calculation.
    """
    dx = max(0.0, first[0] - second[2], second[0] - first[2])
    dy = max(0.0, first[1] - second[3], second[1] - first[3])
    if not math.isfinite(dx) or not math.isfinite(dy):
        return 0.0  # Preserve the exact path's controlled overflow behavior.
    distance = math.hypot(dx, dy)
    if not math.isfinite(distance):
        return 0.0
    coordinate_scale = max(1.0, *(abs(value) for value in (*first, *second)), abs(height))
    guard = 8.0 * math.ulp(coordinate_scale)
    if not math.isfinite(guard):
        return 0.0
    lower = math.nextafter(distance, -math.inf) - guard
    return max(0.0, lower)


_WINDOW_SAFE_SCALE = 1e12


class _PredictionWindow:
    """Vertical index over prediction bounds: a conservative pre-filter only.

    A pair is skipped here only if the vertical gap between bounding boxes
    exceeds height + 1 px while every coordinate and height is at most 1e12.
    Then _bounds_distance_lower_bound(...) ≥ gap − ulp − 8·ulp(1e12) > height,
    so the exact test would skip the pair anyway: the evaluated pairs, and
    their order (ascending prediction index), are unchanged. Otherwise every
    prediction is returned and the exact test alone decides.
    """

    def __init__(self, pred_bounds: list[tuple[float, float, float, float]], gt_scale: float) -> None:
        self.count = len(pred_bounds)
        scale = max([gt_scale, *(abs(v) for bounds in pred_bounds for v in bounds)], default=0.0)
        self.safe = scale <= _WINDOW_SAFE_SCALE
        order = sorted(range(self.count), key=lambda j: (pred_bounds[j][1], j))
        self.y0 = [pred_bounds[j][1] for j in order]
        self.order = order
        self.bounds = pred_bounds
        self.max_span = max((b[3] - b[1] for b in pred_bounds), default=0.0)

    def nearby(self, gt_bounds: tuple[float, float, float, float], height: float) -> list[int] | range:
        if not self.safe or height > _WINDOW_SAFE_SCALE:
            return range(self.count)
        low = gt_bounds[1] - height - 1.0
        high = gt_bounds[3] + height + 1.0
        start = bisect.bisect_left(self.y0, low - self.max_span)
        stop = bisect.bisect_right(self.y0, high)
        return sorted(j for j in self.order[start:stop] if self.bounds[j][3] >= low)


def _pair_distances(
    gt: tuple[Point, ...], pred: tuple[Point, ...],
    gt_samples: tuple[Point, ...], pred_samples: tuple[Point, ...],
    gt_table: tuple[_SegmentRow, ...] | None = None,
    pred_table: tuple[_SegmentRow, ...] | None = None,
) -> tuple[list[float], list[float]]:
    safe = (gt_table is not None and pred_table is not None
            and all(abs(x) <= _SAFE_COORDINATE and abs(y) <= _SAFE_COORDINATE
                    for samples in (gt_samples, pred_samples) for x, y in samples))
    if safe:
        return ([_point_table_distance(p, pred_table) for p in gt_samples],
                [_point_table_distance(p, gt_table) for p in pred_samples])
    return ([_point_polyline_distance(p, pred) for p in gt_samples],
            [_point_polyline_distance(p, gt) for p in pred_samples])


def _pair_measure(
    gt: tuple[Point, ...], pred: tuple[Point, ...],
    gt_samples: tuple[Point, ...], pred_samples: tuple[Point, ...],
    height: float, tolerance_fraction: float,
    *, distances: tuple[list[float], list[float]] | None = None,
) -> tuple[float, float, float, float]:
    """Coverages, mean distance and normalized distance for one pair.

    ``distances`` lets the primary pass reuse its sample distances for the
    1 h diagnostic coverage: the values are identical, only recomputation is
    skipped.
    """
    if distances is None:
        distances = _pair_distances(gt, pred, gt_samples, pred_samples)
    gt_distances, pred_distances = distances
    tolerance = tolerance_fraction * height
    cover_gt = sum(distance <= tolerance for distance in gt_distances) / len(gt_distances)
    cover_pred = sum(distance <= tolerance for distance in pred_distances) / len(pred_distances)
    try:
        gt_mean = math.fsum(distance / len(gt_distances) for distance in gt_distances)
        pred_mean = math.fsum(distance / len(pred_distances) for distance in pred_distances)
    except (OverflowError, ValueError) as exc:
        raise LineGeometryError("mean baseline distance is not finite") from exc
    mean_distance = gt_mean / 2.0 + pred_mean / 2.0
    normalized = mean_distance / height
    if not math.isfinite(mean_distance) or not math.isfinite(normalized):
        _fail("normalized baseline distance is not finite")
    return cover_gt, cover_pred, mean_distance, normalized


def _add_edge(graph: list[list[_Edge]], source: int, target: int, capacity: int, cost: int, pair: tuple[int, int] | None = None) -> None:
    forward = _Edge(target, len(graph[target]), capacity, cost, pair)
    reverse = _Edge(source, len(graph[source]), 0, -cost, None)
    graph[source].append(forward)
    graph[target].append(reverse)


def _validated_candidates(
    n_gt: int, n_pred: int, candidates: Sequence[tuple[int, int, float]],
) -> list[tuple[int, int, int, int]]:
    """Return sorted (i, j, numerator, power-of-two denominator) exact costs.

    Every finite binary float (subnormals included) is a dyadic rational, so
    ``as_integer_ratio`` is exact. An int too large for a finite float is
    refused like a non-finite float.
    """
    for count, label in ((n_gt, "n_gt"), (n_pred, "n_pred")):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail(f"matching {label} must be a nonnegative integer")
    rows: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in candidates:
        if not isinstance(item, tuple) or len(item) != 3:
            _fail("matching candidates must be (gt, pred, cost) tuples")
        i, j, cost = item
        for index in (i, j):
            if isinstance(index, bool) or not isinstance(index, int):
                _fail("matching candidate indices must be integers (not bool or float)")
        if not (0 <= i < n_gt and 0 <= j < n_pred) or (i, j) in seen:
            _fail("matching candidates must be unique in-range (gt, pred) pairs")
        seen.add((i, j))
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            _fail("matching candidate cost must be a finite nonnegative number")
        try:
            finite = math.isfinite(float(cost))
        except OverflowError:
            finite = False
        if not finite or cost < 0:
            _fail("matching candidate cost must be a finite nonnegative number")
        numerator, denominator = cost.as_integer_ratio()
        rows.append((i, j, numerator, denominator))
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def _successive_shortest_paths(
    n_gt: int, n_pred: int, candidates: list[tuple[int, int, int]],
) -> set[tuple[int, int]]:
    """Exact integer successive shortest augmenting paths (Dijkstra + potentials).

    Integer costs keep reduced costs exactly nonnegative, so Dijkstra
    terminates; float reduced costs could make a zero-cost residual cycle
    slightly negative and loop forever (G1 B1). A bounded relaxation budget
    raises ``LineGeometryError`` instead of looping if an invariant broke.
    """
    source, gt0, pred0, sink = 0, 1, 1 + n_gt, 1 + n_gt + n_pred
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for i in range(n_gt):
        _add_edge(graph, source, gt0 + i, 1, 0)
    for j in range(n_pred):
        _add_edge(graph, pred0 + j, sink, 1, 0)
    pair_edges: dict[tuple[int, int], _Edge] = {}
    for i, j, cost in candidates:  # already sorted by (i, j)
        _add_edge(graph, gt0 + i, pred0 + j, 1, cost, (i, j))
        pair_edges[(i, j)] = graph[gt0 + i][-1]

    node_count = len(graph)
    arc_count = sum(len(edges) for edges in graph)
    max_augmentations = min(n_gt, n_pred)
    # With exact nonnegative reduced costs each node is settled once per pass,
    # so each residual arc is relaxed at most once per pass.
    relaxation_budget = (max_augmentations + 1) * (arc_count + node_count)
    relaxations = 0
    augmentations = 0
    potentials = [0] * node_count
    while True:
        dist: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        dist[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > dist[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.cap == 0:
                    continue
                reduced = edge.cost + potentials[node] - potentials[edge.to]
                if reduced < 0:
                    _fail("matching invariant violated: negative exact reduced cost")
                candidate = distance + reduced
                current = dist[edge.to]
                if current is None or candidate < current:
                    relaxations += 1
                    if relaxations > relaxation_budget:
                        _fail("matching exceeded its bounded relaxation budget")
                    dist[edge.to] = candidate
                    previous[edge.to] = (node, edge_index)
                    heapq.heappush(queue, (candidate, edge.to))
        if previous[sink] is None:
            break
        augmentations += 1
        if augmentations > max_augmentations:
            _fail("matching exceeded its maximum number of augmentations")
        for node, distance in enumerate(dist):
            if distance is not None:
                potentials[node] += distance
        node = sink
        while node != source:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge.cap -= 1
            graph[node][edge.rev].cap += 1
            node = parent
    return {pair for pair, edge in pair_edges.items() if edge.cap == 0}


def _scaled_integer_costs(rows: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int]]:
    """Multiply every exact cost by one common 2**k: order and ties are unchanged."""
    scale = max((denominator for *_, denominator in rows), default=1)
    return [(i, j, numerator * (scale // denominator)) for i, j, numerator, denominator in rows]


def _lexicographic_tie_break(costs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Perturb integer costs so the optimum is unique and lexicographically least.

    Edges sorted by (gt, pred) get rank r = 0..R-1. The perturbed cost is
    cost·2^(R+1) + 2^R − 2^(R−1−r). Every maximum matching has the same
    cardinality, so the constant 2^R cancels, any true cost difference (≥ 1)
    outweighs the bonus sum (< 2^R), and distinct sets have distinct bonus
    sums. The unique optimum is the maximum-cardinality, exact-minimum-cost
    matching whose sorted pair list is lexicographically least, i.e. the one
    holding the smallest pair of any symmetric difference.
    """
    size = len(costs)
    return [(i, j, (cost << (size + 1)) + (1 << size) - (1 << (size - 1 - rank)))
            for rank, (i, j, cost) in enumerate(costs)]


def _components(costs: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Connected components of the candidate bipartite graph, in sorted edge order."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j, _ in costs:
        a, b = find(("g", i)), find(("p", j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    for row in costs:
        groups.setdefault(find(("g", row[0])), []).append(row)
    return list(groups.values())


def _solve_tie_broken(n_gt: int, n_pred: int, costs: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    """Solve one graph (whole or component) under the lexicographic tie rule."""
    gt_ids = sorted({i for i, _, _ in costs})
    pred_ids = sorted({j for _, j, _ in costs})
    gt_local = {value: index for index, value in enumerate(gt_ids)}
    pred_local = {value: index for index, value in enumerate(pred_ids)}
    local = [(gt_local[i], pred_local[j], cost) for i, j, cost in costs]  # order kept
    selected = _successive_shortest_paths(len(gt_ids), len(pred_ids), _lexicographic_tie_break(local))
    return {(gt_ids[a], pred_ids[b]) for a, b in selected}


def _minimum_cost_maximum_matching(
    n_gt: int,
    n_pred: int,
    candidates: list[tuple[int, int, float]],
    *,
    _decompose: bool = True,
) -> set[tuple[int, int]]:
    """Maximum-cardinality, exact minimum-cost matching with a global tie rule.

    Costs become exact integers (one common power-of-two scale), then each
    connected component of the candidate graph is solved independently. Among
    optimal matchings the result is the unique one whose sorted (gt, pred)
    index list is lexicographically least; inputs are sorted by stable IDs, so
    this is the least list of (gt_id, pred_id) pairs. That rule does not depend
    on the decomposition: ``_decompose=False`` returns the same set.
    """
    costs = _scaled_integer_costs(_validated_candidates(n_gt, n_pred, candidates))
    if not costs:
        return set()
    if not _decompose:
        return _solve_tie_broken(n_gt, n_pred, costs)
    selected: set[tuple[int, int]] = set()
    for component in _components(costs):
        selected |= _solve_tie_broken(n_gt, n_pred, component)
    return selected


def _orientation_dot_sign(gt: tuple[Point, ...], pred: tuple[Point, ...]) -> int | None:
    """Sign of directed endpoint dot product; not part of geometric distance.

    Normalize vectors before the dot product to avoid overflow. A zero endpoint
    vector has no direction; a zero dot product gives no agreement sign either.
    This is a global diagnostic, not proof of consistent local curve normals.
    """
    vectors = []
    for points in (gt, pred):
        dx, dy = points[-1][0] - points[0][0], points[-1][1] - points[0][1]
        if not math.isfinite(dx) or not math.isfinite(dy):
            _fail("orientation endpoint vector is not representable")
        scale = max(abs(dx), abs(dy))
        if scale == 0:
            return None
        vectors.append((dx / scale, dy / scale))
    dot = math.fsum((vectors[0][0] * vectors[1][0], vectors[0][1] * vectors[1][1]))
    return 1 if dot > 0 else -1 if dot < 0 else 0


def _evaluate_line_geometry_policy(
    ground_truth: Sequence[GroundTruthLine],
    predictions: Sequence[PredictedLine],
    *,
    max_sample_spacing_px: float = 4.0,
    coverage_threshold: float = 0.75,
    tolerance_height_fraction: float = 0.5,
    _metric_id: str = METRIC_ID,
) -> LineGeometryResult:
    """Evaluate baseline geometry without consulting any transcription text.

    Main precision/recall/F1 are null if any GT line lacks an available baseline
    or explicit positive GT band height. The evaluable subset remains available
    for diagnostics, never as the page's principal score. The 1.0-height pass
    and permissive split/merge adjacency counts are sensitivity diagnostics;
    gained-only-at-1h is a bounded geometry-threshold sensitivity count, not a
    claim that a line was truly omitted. They never replace the v1 0.5-height
    metrics or assign split/merge classes.
    """
    spacing = _finite(max_sample_spacing_px, "max_sample_spacing_px")
    threshold = _finite(coverage_threshold, "coverage_threshold")
    tolerance_fraction = _finite(tolerance_height_fraction, "tolerance_height_fraction")
    if spacing <= 0 or not 0 <= threshold <= 1 or tolerance_fraction <= 0:
        _fail("sampling spacing and tolerance must be positive; coverage threshold must be in [0,1]")
    if spacing != 4.0 or threshold != 0.75 or tolerance_fraction not in {0.5, 1.0}:
        _fail("line geometry fixes spacing=4px and coverage=0.75; tolerance is 0.5h or the separate 1.0h sensitivity policy")

    gt_lines, incomplete_gt = _validated_gt(ground_truth)
    pred_lines, pred_unavailable = _validated_pred(predictions)
    gt_lines.sort(key=lambda row: row[0].id)
    pred_lines.sort(key=lambda row: row[0].id)
    pred_bounds = [_bounds(points) for _, points in pred_lines]
    window = _PredictionWindow(
        pred_bounds,
        max([abs(v) for _, points, height in gt_lines for point in points for v in (*point, height)], default=0.0),
    )
    gt_tables = {i: _segment_table(points) for i, (_, points, _) in enumerate(gt_lines)}
    pred_tables = [_segment_table(points) for _, points in pred_lines]
    gt_samples = {
        i: _sample(points, min(4.0, height / 4.0))
        for i, (_, points, height) in enumerate(gt_lines)
    }
    measurements: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    candidates: list[tuple[int, int, float]] = []
    relaxed_candidates: list[tuple[int, int, float]] = []
    split_gt_neighbors: dict[int, set[int]] = {i: set() for i in range(len(gt_lines))}
    merge_pred_neighbors: dict[int, set[int]] = {j: set() for j in range(len(pred_lines))}
    gt_by_spacing: dict[float, list[int]] = {}
    for i, (_, _, height) in enumerate(gt_lines):
        spacing = min(4.0, height / 4.0)
        gt_by_spacing.setdefault(spacing, []).append(i)
    for spacing, gt_indices in sorted(gt_by_spacing.items()):
        # Keep prediction samples only for this height group, avoiding a
        # page-sized cache over every GT-height × prediction combination.
        pred_samples = [_sample(points, spacing) for _, points in pred_lines]
        for i in gt_indices:
            gt_line, gt_points, height = gt_lines[i]
            gt_bounds = _bounds(gt_points)
            for j in window.nearby(gt_bounds, height):
                pred_line, pred_points = pred_lines[j]
                if _bounds_distance_lower_bound(gt_bounds, pred_bounds[j], height=height) > height:
                    continue
                gt_line_samples = gt_samples[i]
                pred_line_samples = pred_samples[j]
                distances = _pair_distances(gt_points, pred_points, gt_line_samples, pred_line_samples,
                                            gt_tables[i], pred_tables[j])
                cover_gt, cover_pred, mean_px, normalized = _pair_measure(
                    gt_points, pred_points, gt_line_samples, pred_line_samples,
                    height, tolerance_fraction, distances=distances,
                )
                measurements[(i, j)] = (cover_gt, cover_pred, mean_px, normalized)
                if cover_gt >= threshold and cover_pred >= threshold:
                    candidates.append((i, j, normalized))
                if tolerance_fraction == 0.5:
                    relaxed_cover_gt, relaxed_cover_pred, _, _ = _pair_measure(
                        gt_points, pred_points, gt_line_samples, pred_line_samples, height, 1.0,
                        distances=distances,
                    )
                    if relaxed_cover_gt >= threshold and relaxed_cover_pred >= threshold:
                        relaxed_candidates.append((i, j, normalized))
                # D-552-like cut/fusion indicators are deliberately diagnostic
                # only; their permissive OR rule does not define a line match.
                if tolerance_fraction == 0.5 and (cover_gt >= 0.25 or cover_pred >= 0.50):
                    split_gt_neighbors[i].add(j)
                    merge_pred_neighbors[j].add(i)
    selected = _minimum_cost_maximum_matching(len(gt_lines), len(pred_lines), candidates)
    selected_1h = (
        _minimum_cost_maximum_matching(len(gt_lines), len(pred_lines), relaxed_candidates)
        if tolerance_fraction == 0.5 else set()
    )
    matches = tuple(
        LineMatch(
            gt_id=gt_lines[i][0].id,
            pred_id=pred_lines[j][0].id,
            gt_to_pred_coverage=measurements[(i, j)][0],
            pred_to_gt_coverage=measurements[(i, j)][1],
            mean_symmetric_distance_px=measurements[(i, j)][2],
            normalized_symmetric_distance=measurements[(i, j)][3],
            orientation_dot_sign=_orientation_dot_sign(gt_lines[i][1], pred_lines[j][1]),
        )
        for i, j in sorted(selected, key=lambda pair: (gt_lines[pair[0]][0].id, pred_lines[pair[1]][0].id))
    )
    unmatched_gt = len(gt_lines) - len(selected)
    matched_gt = {i for i, _ in selected}
    relaxed_gt = {i for i, _ in selected_1h}
    split_gt = {i for i, neighbors in split_gt_neighbors.items() if len(neighbors) >= 2}
    split_pred = {j for i in split_gt for j in split_gt_neighbors[i]}
    merge_pred = {j for j, neighbors in merge_pred_neighbors.items() if len(neighbors) >= 2}
    merge_gt = {i for j in merge_pred for i in merge_pred_neighbors[j]}
    unmatched_pred = len(predictions) - len(selected) if not incomplete_gt else None
    status: Literal["evaluated", "non_evaluable"] = "non_evaluable" if incomplete_gt else "evaluated"
    precision = recall = f1 = None
    if not incomplete_gt:
        if predictions:
            precision = len(selected) / len(predictions)
        if gt_lines:
            recall = len(selected) / len(gt_lines)
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        elif precision == 0 and recall == 0:
            f1 = 0.0
    return LineGeometryResult(
        metric_id=_metric_id,
        status=status,
        precision=precision,
        recall=recall,
        f1=f1,
        n_gt=len(ground_truth),
        gt_evaluable_count=len(gt_lines),
        gt_non_evaluable_count=len(incomplete_gt),
        n_pred=len(predictions),
        pred_geometry_unavailable_count=pred_unavailable,
        matched_count=len(selected),
        orientation_disagreement_count=sum(m.orientation_dot_sign == -1 for m in matches),
        orientation_unresolved_count=sum(m.orientation_dot_sign in (None, 0) for m in matches),
        unmatched_evaluable_gt_count=unmatched_gt,
        unmatched_pred_count=unmatched_pred,
        matches=matches,
        sensitivity_policy_id=SENSITIVITY_POLICY_ID if tolerance_fraction == 0.5 else None,
        sensitivity_1h_matched_count=len(selected_1h) if tolerance_fraction == 0.5 else None,
        gt_gained_only_at_1h_count=len(relaxed_gt - matched_gt) if tolerance_fraction == 0.5 else None,
        split_candidate_gt_count=len(split_gt) if tolerance_fraction == 0.5 else None,
        split_candidate_pred_count=len(split_pred) if tolerance_fraction == 0.5 else None,
        merge_candidate_pred_count=len(merge_pred) if tolerance_fraction == 0.5 else None,
        merge_candidate_gt_count=len(merge_gt) if tolerance_fraction == 0.5 else None,
    )


def evaluate_line_geometry(
    ground_truth: Sequence[GroundTruthLine],
    predictions: Sequence[PredictedLine],
    *,
    max_sample_spacing_px: float = 4.0,
    coverage_threshold: float = 0.75,
) -> LineGeometryResult:
    """Run the primary ``line_geometry.v1`` metric at tolerance 0.5×GT height."""
    return _evaluate_line_geometry_policy(
        ground_truth,
        predictions,
        max_sample_spacing_px=max_sample_spacing_px,
        coverage_threshold=coverage_threshold,
        tolerance_height_fraction=0.5,
        _metric_id=METRIC_ID,
    )


def evaluate_line_geometry_sensitivity(
    ground_truth: Sequence[GroundTruthLine],
    predictions: Sequence[PredictedLine],
) -> LineGeometryResult:
    """Run the separately versioned 1.0×GT-height sensitivity pass.

    Returns its own matches and distances. It is not the primary v1 result and
    must not replace or select the 0.5× result based on observed outcomes.
    """
    return _evaluate_line_geometry_policy(
        ground_truth,
        predictions,
        tolerance_height_fraction=1.0,
        _metric_id=SENSITIVITY_POLICY_ID,
    )
