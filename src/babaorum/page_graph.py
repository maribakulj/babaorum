"""Strict, dependency-free contract for page-level OCR pipeline outputs.

This is a page/run graph, not the synthetic renderer's fully labelled
ground-truth schema. Missing word geometry is represented explicitly; it is
never inferred from a line transcript. Without the optional ``graph_role`` the
validator is structural only and does not police provenance: any declared
source, including ``ground_truth``, is accepted. With
``graph_role="prediction"``, no geometry, band or relation source may be
``ground_truth`` or ``transferred`` (the latter does not say where it was
transferred from, so it could carry ground truth).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "babaorum.page_graph.v1"
COORDINATE_FRAME = "original_encoded_pixels_top_left_xy"
COORDINATE_CONVENTION = "continuous_pixel_edges_0_width_height"
NORMAL_CONVENTION = "n=(-ty,tx);positive_90deg_y_down"
GEOMETRY_SOURCES = {"predicted", "transferred", "ground_truth", "interpolated"}
GEOMETRY_STATES = {"available", "missing", "error"}
TEXT_STATES = {"available", "missing", "error"}
NODE_STATES = {"ok", "partial", "error"}
GRAPH_ROLES = {"prediction", "ground_truth", "gt_assisted"}
COVERAGE_STATES = {"complete", "partial", "unavailable", "error", "unknown"}
RUN_STATES = {"ok", "partial", "error"}
BLOCK_CATEGORIES = {
    "titre", "texte", "legende", "annonce", "tableau", "illustration", "separateur", "autre"
}
ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class PageGraphError(ValueError):
    """Invalid graph, with a JSON-like path in the message."""


def _fail(path: str, message: str) -> None:
    raise PageGraphError(f"{path}: {message}")


def _object(value: Any, path: str, required: set[str], optional: set[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        _fail(path, f"missing fields {sorted(missing)}")
    if extra:
        _fail(path, f"unknown fields {sorted(extra)}")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        _fail(path, "must be a nonempty string" if nonempty else "must be a string")
    return value


def _enum(value: Any, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"must be one of {sorted(choices)}")
    return value


def _id(value: Any, path: str) -> str:
    text = _string(value, path)
    if not ID_RE.fullmatch(text):
        _fail(path, "must match [A-Za-z_][A-Za-z0-9_.-]*")
    return text


def _count(value: Any, path: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, "must be a nonnegative integer" + (" or null" if nullable else ""))
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(path, "must be representable as a finite number")
    if not math.isfinite(result):
        _fail(path, "must be a finite number")
    return result


def _point(value: Any, path: str, width: int, height: int) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        _fail(path, "must be [x,y]")
    x = _number(value[0], f"{path}[0]")
    y = _number(value[1], f"{path}[1]")
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        _fail(path, f"point lies outside [0,{width}] x [0,{height}]")
    return x, y


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-10
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def _segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    eps = 1e-10
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return (
        (abs(o1) <= eps and _on_segment(a, b, c))
        or (abs(o2) <= eps and _on_segment(a, b, d))
        or (abs(o3) <= eps and _on_segment(c, d, a))
        or (abs(o4) <= eps and _on_segment(c, d, b))
    )


def _polygon(value: Any, path: str, width: int, height: int) -> None:
    if not isinstance(value, list) or len(value) < 3:
        _fail(path, "polygon must contain at least three vertices")
    points = [_point(p, f"{path}[{i}]", width, height) for i, p in enumerate(value)]
    if len(set(points)) < 3:
        _fail(path, "polygon must have at least three distinct vertices")
    if points[0] == points[-1]:
        _fail(path, "do not repeat the first vertex at the end")
    if any(points[i] == points[(i + 1) % len(points)] for i in range(len(points))):
        _fail(path, "consecutive vertices must differ")
    twice_area = sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )
    if abs(twice_area) <= 1e-10:
        _fail(path, "polygon must have positive area")
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent edges share a valid endpoint; all other intersections invalidate simplicity.
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            if _segments_intersect(a, b, points[j], points[(j + 1) % n]):
                _fail(path, "polygon edges must not self-intersect or touch")


def _polyline(value: Any, path: str, width: int, height: int) -> None:
    if not isinstance(value, list) or len(value) < 2:
        _fail(path, "polyline must contain at least two points")
    points = [_point(p, f"{path}[{i}]", width, height) for i, p in enumerate(value)]
    if not any(a != b for a, b in zip(points, points[1:])):
        _fail(path, "polyline must not be degenerate")


def _geometry(value: Any, path: str, width: int, height: int, *, kind: str) -> None:
    obj = _object(value, path, {"status", "points", "source"})
    status = obj["status"]
    _enum(status, GEOMETRY_STATES, f"{path}.status")
    points, source = obj["points"], obj["source"]
    if status == "available":
        _enum(source, GEOMETRY_SOURCES, f"{path}.source")
        if kind == "polygon":
            _polygon(points, f"{path}.points", width, height)
        else:
            _polyline(points, f"{path}.points", width, height)
    elif points is not None or source is not None:
        _fail(path, "missing/error geometry must have points=null and source=null")


def _text(value: Any, path: str) -> None:
    obj = _object(value, path, {"status", "value"})
    status = _enum(obj["status"], TEXT_STATES, f"{path}.status")
    if status == "available":
        _string(obj["value"], f"{path}.value", nonempty=False)
    elif obj["value"] is not None:
        _fail(path, "missing/error text must have value=null")


def _baseline_band(value: Any, path: str) -> None:
    obj = _object(value, path, {"status", "normal_negative_px", "normal_positive_px", "source"})
    _enum(obj["status"], GEOMETRY_STATES, f"{path}.status")
    if obj["status"] == "available":
        for key in ("normal_negative_px", "normal_positive_px"):
            distance = _number(obj[key], f"{path}.{key}")
            if distance < 0:
                _fail(f"{path}.{key}", "distance from baseline must be nonnegative")
        _enum(obj["source"], GEOMETRY_SOURCES, f"{path}.source")
        if obj["normal_negative_px"] + obj["normal_positive_px"] <= 0:
            _fail(path, "available band must have positive total extent")
    elif any(obj[key] is not None for key in ("normal_negative_px", "normal_positive_px", "source")):
        _fail(path, "missing/error band must have null extents and source")


def _coverage(value: Any, path: str) -> None:
    obj = _object(value, path, {"status", "expected_count", "available_count", "error_count"})
    status = obj["status"]
    _enum(status, COVERAGE_STATES, f"{path}.status")
    expected = _count(obj["expected_count"], f"{path}.expected_count", nullable=True)
    available = _count(obj["available_count"], f"{path}.available_count")
    errors = _count(obj["error_count"], f"{path}.error_count")
    if expected is not None and available + errors > expected:
        _fail(path, "available_count + error_count cannot exceed expected_count")
    if status == "complete" and (expected is None or available != expected or errors != 0):
        _fail(path, "complete requires a known denominator, all units available, and zero errors")
    if status == "partial" and expected is None:
        _fail(path, "partial requires a known denominator")
    if status == "unavailable" and (available != 0 or errors != 0):
        _fail(path, "unavailable requires zero available and failed units")
    if status == "error" and errors == 0:
        _fail(path, "error requires at least one failed unit")
    if status == "unknown" and (expected is not None or errors != 0):
        _fail(path, "unknown requires expected_count=null and error_count=0")


def _check_node_counts(coverage: Mapping[str, Any], items: list[Mapping[str, Any]], path: str) -> None:
    available = sum(item["status"] in {"ok", "partial"} for item in items)
    errors = sum(item["status"] == "error" for item in items)
    if coverage["available_count"] != available or coverage["error_count"] != errors:
        _fail(path, f"counts must match node statuses (available={available}, error={errors})")


def _check_text_counts(coverage: Mapping[str, Any], texts: list[Mapping[str, Any]], path: str) -> None:
    available = sum(item["status"] == "available" for item in texts)
    errors = sum(item["status"] == "error" for item in texts)
    if coverage["available_count"] != available or coverage["error_count"] != errors:
        _fail(path, f"counts must match transcription statuses (available={available}, error={errors})")


def _check_partial_coverage(coverage: Mapping[str, Any], items: list[Mapping[str, Any]], path: str) -> None:
    if coverage["status"] == "complete" and any(item["status"] == "partial" for item in items):
        _fail(path, "complete coverage cannot contain partial nodes")
    if (coverage["status"] == "partial" and coverage["available_count"] == coverage["expected_count"]
            and coverage["error_count"] == 0 and not any(item["status"] == "partial" for item in items)):
        _fail(path, "partial coverage requires missing/failed units or partial nodes")


def _check_relation_count(coverage: Mapping[str, Any], count: int, path: str) -> None:
    if coverage["available_count"] != count:
        _fail(path, f"available_count must equal emitted order edges ({count})")


PREDICTION_FORBIDDEN_SOURCES = {"ground_truth", "transferred"}


def _check_prediction_provenance(root: Mapping[str, Any]) -> None:
    """A prediction graph may not claim ground-truth or unqualified transferred provenance.

    ``transferred`` names no origin; it may have been transferred from ground
    truth (e.g. GT-assisted alignment), so a prediction graph fails closed.
    """
    fields = {"blocks": ("polygon",), "lines": ("polygon", "baseline", "baseline_band"), "words": ("polygon",)}
    for collection, keys in fields.items():
        for i, node in enumerate(root[collection]):
            for key in keys:
                source = node[key]["source"]
                if source in PREDICTION_FORBIDDEN_SOURCES:
                    _fail(f"$.{collection}[{i}].{key}.source", f"a prediction graph cannot claim {source} provenance")
    for i, relation in enumerate(root["relations"]):
        if relation["source"] in PREDICTION_FORBIDDEN_SOURCES:
            _fail(f"$.relations[{i}].source", f"a prediction graph cannot claim {relation['source']} provenance")


def validate_page_graph(graph: Mapping[str, Any]) -> None:
    """Validate a pipeline page graph; raise ``PageGraphError`` at first defect.

    Coordinates use continuous pixel-edge bounds ``[0,width] x [0,height]``.
    Node array order carries no reading-order semantics; only ``precedes`` edges
    do. Partial order is allowed, but cycles and cross-type order edges are not.
    The optional ``graph_role`` (prediction, ground_truth, gt_assisted) is
    checked only for ``prediction``: no ground_truth or transferred source.
    """
    root = _object(
        graph,
        "$",
        {"schema_version", "identity", "coordinate_frame", "coordinate_convention", "normal_convention", "image", "provenance", "run_status", "coverage",
         "articles", "blocks", "lines", "words", "relations"},
        {"graph_role"},
    )
    if root["schema_version"] != SCHEMA_VERSION:
        _fail("$.schema_version", f"must be {SCHEMA_VERSION!r}")
    identity = _object(root["identity"], "$.identity", {"page_key", "document_key", "title_key"})
    _string(identity["page_key"], "$.identity.page_key")
    for key in ("document_key", "title_key"):
        if identity[key] is not None:
            _string(identity[key], f"$.identity.{key}")
    graph_role = None
    if "graph_role" in root:
        graph_role = _enum(root["graph_role"], GRAPH_ROLES, "$.graph_role")
    if root["coordinate_frame"] != COORDINATE_FRAME:
        _fail("$.coordinate_frame", f"must be {COORDINATE_FRAME!r}")
    if root["coordinate_convention"] != COORDINATE_CONVENTION:
        _fail("$.coordinate_convention", f"must be {COORDINATE_CONVENTION!r}")
    if root["normal_convention"] != NORMAL_CONVENTION:
        _fail("$.normal_convention", f"must be {NORMAL_CONVENTION!r}")
    image = _object(root["image"], "$.image", {"sha256", "width", "height"})
    if not isinstance(image["sha256"], str) or not SHA256_RE.fullmatch(image["sha256"]):
        _fail("$.image.sha256", "must be a lowercase 64-character SHA-256")
    width = _count(image["width"], "$.image.width")
    height = _count(image["height"], "$.image.height")
    if width == 0 or height == 0:
        _fail("$.image", "width and height must be positive")
    provenance = _object(root["provenance"], "$.provenance", {"run_id", "components"})
    _string(provenance["run_id"], "$.provenance.run_id")
    components = provenance["components"]
    if not isinstance(components, list) or not components:
        _fail("$.provenance.components", "must be a nonempty array")
    for i, component in enumerate(components):
        path = f"$.provenance.components[{i}]"
        obj = _object(component, path, {"name", "code_revision", "config_sha256", "checkpoint_sha256", "license"})
        for key in ("name", "code_revision"):
            _string(obj[key], f"{path}.{key}")
        for key in ("config_sha256", "checkpoint_sha256"):
            digest = obj[key]
            if digest is not None and (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
                _fail(f"{path}.{key}", "must be null or a lowercase 64-character SHA-256")
        if obj["license"] is not None:
            _string(obj["license"], f"{path}.license")
    run_status = _enum(root["run_status"], RUN_STATES, "$.run_status")

    coverage = _object(
        root["coverage"], "$.coverage",
        {"blocks", "lines", "words", "transcription", "reading_order"},
    )
    for stage, record in coverage.items():
        _coverage(record, f"$.coverage.{stage}")

    node_map: dict[str, str] = {}
    nodes: dict[str, list[Mapping[str, Any]]] = {}
    specs = {
        "articles": {"common": {"id", "status"}, "kind": "article"},
        "blocks": {"common": {"id", "status", "category", "polygon"}, "kind": "block"},
        "lines": {"common": {"id", "status", "polygon", "baseline", "baseline_semantics", "orientation", "baseline_band", "text"}, "kind": "line"},
        "words": {"common": {"id", "status", "polygon", "text"}, "kind": "word"},
    }
    has_component_error = False
    has_partial_node = False
    for collection, spec in specs.items():
        items = root[collection]
        if not isinstance(items, list):
            _fail(f"$.{collection}", "must be an array")
        nodes[spec["kind"]] = items
        for i, item in enumerate(items):
            path = f"$.{collection}[{i}]"
            obj = _object(item, path, spec["common"])
            ident = _id(obj["id"], f"{path}.id")
            if ident in node_map:
                _fail(f"{path}.id", f"duplicate global node id {ident!r}")
            node_map[ident] = spec["kind"]
            node_status = _enum(obj["status"], NODE_STATES, f"{path}.status")
            has_partial_node = has_partial_node or node_status == "partial"
            if spec["kind"] in {"block", "line", "word"}:
                _geometry(obj["polygon"], f"{path}.polygon", width, height, kind="polygon")
                field_errors = obj["polygon"]["status"] == "error"
            if spec["kind"] == "line":
                _geometry(obj["baseline"], f"{path}.baseline", width, height, kind="polyline")
                _baseline_band(obj["baseline_band"], f"{path}.baseline_band")
                if obj["baseline_semantics"] != "writing_line_foot_of_body":
                    _fail(f"{path}.baseline_semantics", "must identify the typographic writing line at the foot of the body")
                orientation = _enum(obj["orientation"], {"ltr", "rtl", "ttb", "btt", "unknown"}, f"{path}.orientation")
                baseline = obj["baseline"]
                if obj["baseline_band"]["status"] == "available" and baseline["status"] != "available":
                    _fail(f"{path}.baseline_band", "available band requires an available baseline")
                if baseline["status"] == "available":
                    first, last = baseline["points"][0], baseline["points"][-1]
                    if orientation == "ltr" and last[0] <= first[0]:
                        _fail(f"{path}.baseline.points", "ltr orientation conflicts with endpoint direction")
                    if orientation == "rtl" and last[0] >= first[0]:
                        _fail(f"{path}.baseline.points", "rtl orientation conflicts with endpoint direction")
                    if orientation == "ttb" and last[1] <= first[1]:
                        _fail(f"{path}.baseline.points", "ttb orientation conflicts with endpoint direction")
                    if orientation == "btt" and last[1] >= first[1]:
                        _fail(f"{path}.baseline.points", "btt orientation conflicts with endpoint direction")
                _text(obj["text"], f"{path}.text")
                field_errors = field_errors or baseline["status"] == "error" or obj["baseline_band"]["status"] == "error" or obj["text"]["status"] == "error"
            elif spec["kind"] == "word":
                _text(obj["text"], f"{path}.text")
                field_errors = field_errors or obj["text"]["status"] == "error"
            else:
                field_errors = field_errors if spec["kind"] in {"block", "word"} else False
            if field_errors and node_status == "ok":
                _fail(f"{path}.status", "must be partial or error when a geometry/text field is in error")
            has_component_error = has_component_error or field_errors or node_status == "error"

    for collection, stage in (("blocks", "blocks"), ("lines", "lines"), ("words", "words")):
        _check_node_counts(coverage[stage], root[collection], f"$.coverage.{stage}")
        _check_partial_coverage(coverage[stage], root[collection], f"$.coverage.{stage}")
    text_units = [node["text"] for collection in ("lines", "words") for node in root[collection]]
    _check_text_counts(coverage["transcription"], text_units, "$.coverage.transcription")
    _check_partial_coverage(coverage["transcription"], text_units, "$.coverage.transcription")

    coverage_errors = any(
        record["error_count"] > 0 or record["status"] in {"partial", "error"}
        for record in coverage.values()
    )
    if run_status == "ok" and (has_component_error or has_partial_node or coverage_errors):
        _fail("$.run_status", "ok is inconsistent with partial or failed node, text, geometry, or coverage")

    relations = root["relations"]
    if not isinstance(relations, list):
        _fail("$.relations", "must be an array")
    parents: dict[str, str] = {}
    precedes: dict[str, list[str]] = defaultdict(list)
    seen_edges: set[tuple[str, str, str]] = set()
    order_count = 0
    allowed_parent = {"article": "block", "block": "line", "line": "word"}
    for i, relation in enumerate(relations):
        path = f"$.relations[{i}]"
        edge = _object(
            relation, path,
            {"type", "source_id", "target_id", "source", "confidence"},
        )
        kind = _enum(edge["type"], {"contains", "precedes"}, f"{path}.type")
        source_id, target_id = edge["source_id"], edge["target_id"]
        _id(source_id, f"{path}.source_id")
        _id(target_id, f"{path}.target_id")
        if source_id not in node_map or target_id not in node_map:
            _fail(path, "relation endpoints must reference existing node IDs")
        if source_id == target_id:
            _fail(path, "self-relations are invalid")
        source_type, target_type = node_map[source_id], node_map[target_id]
        _enum(edge["source"], GEOMETRY_SOURCES, f"{path}.source")
        confidence = _number(edge["confidence"], f"{path}.confidence")
        if not 0.0 <= confidence <= 1.0:
            _fail(f"{path}.confidence", "must be in [0,1]")
        key = (kind, source_id, target_id)
        if key in seen_edges:
            _fail(path, "duplicate relation")
        seen_edges.add(key)
        if kind == "contains":
            if allowed_parent.get(source_type) != target_type:
                _fail(path, "contains must point article→block, block→line, or line→word")
            if target_id in parents:
                _fail(path, f"node {target_id!r} has more than one parent")
            parents[target_id] = source_id
        else:
            if source_type != target_type:
                _fail(path, "precedes relations must connect nodes of the same type")
            precedes[source_id].append(target_id)
            order_count += 1

    if graph_role == "prediction":
        _check_prediction_provenance(root)
    _check_relation_count(coverage["reading_order"], order_count, "$.coverage.reading_order")
    _check_partial_coverage(coverage["reading_order"], [], "$.coverage.reading_order")

    # Project order edges to known ancestors. A word-order edge can contradict
    # an explicit line-order edge even though each level is acyclic by itself.
    def ancestor(node_id: str, wanted_type: str) -> str | None:
        current: str | None = node_id
        while current is not None:
            if node_map[current] == wanted_type:
                return current
            current = parents.get(current)
        return None

    projected: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source_id, targets in precedes.items():
        for target_id in targets:
            for node_type in ("article", "block", "line", "word"):
                source_ancestor = ancestor(source_id, node_type)
                target_ancestor = ancestor(target_id, node_type)
                if source_ancestor and target_ancestor and source_ancestor != target_ancestor:
                    projected[node_type].add((source_ancestor, target_ancestor))

    for node_type, node_ids in nodes.items():
        edges = projected[node_type]
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {item["id"]: 0 for item in node_ids}
        for source_id, target_id in edges:
            adjacency[source_id].append(target_id)
            indegree[target_id] += 1
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for target_id in adjacency.get(node_id, ()):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    queue.append(target_id)
        if visited != len(node_ids):
            _fail("$.relations", f"precedes order conflicts across known {node_type} ancestors or contains a cycle")


def make_geometry(points: list[list[float]] | None, source: str | None = None, *, error: bool = False) -> dict[str, Any]:
    """Build the explicit geometry envelope; validation remains separate."""
    if error:
        if points is not None or source is not None:
            raise ValueError("error geometry cannot carry points or source")
        return {"status": "error", "points": None, "source": None}
    if points is None:
        if source is not None:
            raise ValueError("missing geometry cannot carry a source")
        return {"status": "missing", "points": None, "source": None}
    return {"status": "available", "points": points, "source": source}


def make_text(value: str | None, *, error: bool = False) -> dict[str, Any]:
    """Represent transcription availability without substituting empty text."""
    if error:
        if value is not None:
            raise ValueError("error text cannot carry a value")
        return {"status": "error", "value": None}
    if value is None:
        return {"status": "missing", "value": None}
    return {"status": "available", "value": value}
