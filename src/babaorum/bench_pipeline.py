"""Pipeline bench v1: reproduction controls and the page-graph input adapter.

This module is stdlib-only and never imports axel. The CLI (scripts/bench_pipeline.py)
recomputes the published numbers with axel's own functions, read-only, and hands
the measured values to ``check_controls``. Everything here is labelled
"dev exposé (D-600)": a dev comparison point, never a test result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from babaorum.page_graph import (
    COORDINATE_CONVENTION,
    COORDINATE_FRAME,
    NORMAL_CONVENTION,
    SCHEMA_VERSION,
    PageGraphError,
    _geometry,
    make_geometry,
    make_text,
    validate_page_graph,
)

SCHEMA = "babaorum.bench_pipeline.v1"
LABEL = "dev exposé (D-600)"
ROLE_SOURCE = {"prediction": "predicted", "ground_truth": "ground_truth"}
#: detector/chain block label -> page_graph category. All three labels are text containers by construction
#: (PAGE TextRegion, ALTO TextBlock, PP-DocLayoutV3 column zone). "texte" means a text-bearing region, not a
#: finer typing: titles inside a zone are not distinguished. Any other or unknown label stays "autre".
CATEGORY_BY_LABEL = {"TextRegion": "texte", "TextBlock": "texte", "zone": "texte"}
FORBIDDEN_IN_PREDICTION = {"ground_truth", "transferred"}


class BenchError(ValueError):
    pass


class ReproductionError(BenchError):
    pass


# --- configuration and reproduction controls -------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise BenchError("bench config schema_version mismatch")
    if config.get("label") != LABEL:
        raise BenchError(f"bench config must be labelled {LABEL!r}")
    tolerance = config.get("tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 < tolerance <= 1e-6:
        raise BenchError("bench config tolerance must be in (0, 1e-6]")
    controls = config.get("controls")
    if not isinstance(controls, list) or not controls:
        raise BenchError("bench config has no reproduction controls")
    ids = [c.get("id") for c in controls]
    if len(set(ids)) != len(ids):
        raise BenchError("duplicate control id")
    return config


def _check_axel_path(axel_root: Path, relative: str, forbidden: Sequence[str]) -> Path:
    if any(fragment in relative for fragment in forbidden):
        raise BenchError(f"forbidden source for a dev bench: {relative}")
    root = axel_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or "runs" not in path.relative_to(root).parts[:1]:
        raise BenchError(f"control source must be under axel/runs: {relative}")
    return path


def read_expected(axel_root: str | Path, spec: Mapping[str, Any], forbidden: Sequence[str]) -> float:
    path = _check_axel_path(Path(axel_root), spec["file"], forbidden)
    node: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in spec["path"]:
        if not isinstance(node, dict) or key not in node:
            raise BenchError(f"published value not found: {spec['file']} {spec['path']}")
        node = node[key]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise BenchError(f"published value is not a number: {spec['file']} {spec['path']}")
    return float(node)


def check_controls(measured: Mapping[str, float], config: Mapping[str, Any],
                   axel_root: str | Path) -> list[dict[str, Any]]:
    """One result per control: exact, deviation, missing, or published_mismatch."""
    tolerance = float(config["tolerance"])
    forbidden = config.get("forbidden_path_fragments", [])
    results = []
    for control in config["controls"]:
        expected = read_expected(axel_root, control["expected"], forbidden)
        value = measured.get(control["id"])
        row = {"id": control["id"], "chain": control["chain"], "expected": expected,
               "measured": value, "published_pct": control["published_pct"],
               "source": control["expected"]["file"], "tolerance": tolerance}
        if round(expected * 100, 2) != control["published_pct"]:
            row.update(status="published_mismatch", diff=None)
        elif value is None:
            row.update(status="missing", diff=None)
        else:
            diff = abs(float(value) - expected)
            row.update(status="exact" if diff <= tolerance else "deviation", diff=diff)
        results.append(row)
    return results


def require_exact(results: Sequence[Mapping[str, Any]]) -> None:
    bad = [f"{r['id']}: {r['status']}" for r in results if r["status"] != "exact"]
    if bad:
        raise ReproductionError("reproduction controls not exact: " + "; ".join(bad))


def empty_metric_slots(config: Mapping[str, Any]) -> dict[str, Any]:
    """Placeholders for the metrics that wait for review: value null, status explicit."""
    slots = {}
    for name, spec in config.get("metric_slots", {}).items():
        body = {"status": spec} if isinstance(spec, str) else dict(spec)
        slots[name] = {"value": None, **body}
    return slots


# --- performance: memoize polygon validation (pure function) ----------------------

_ORIGINAL_POLYGON = None


def install_polygon_cache() -> None:
    """Skip re-validating a polygon already accepted at the same image size.

    ``page_graph._polygon`` is a pure check (O(v^2) edge tests); the chains of one
    page share most polygons, so the same check ran dozens of times. Only successes
    are cached: an invalid polygon still raises every time. page_graph.py is not
    modified; the wrapper lives in this process only. Idempotent.
    """
    global _ORIGINAL_POLYGON
    from babaorum import page_graph
    if _ORIGINAL_POLYGON is not None:
        return
    original = _ORIGINAL_POLYGON = page_graph._polygon
    accepted: set = set()

    def cached(value, path, width, height):
        try:
            key = (tuple(tuple(p) for p in value), width, height)
            hash(key)
        except TypeError:
            key = None
        if key is not None and key in accepted:
            return None
        original(value, path, width, height)
        if key is not None:
            accepted.add(key)
        return None

    page_graph._polygon = cached


def uninstall_polygon_cache() -> None:
    global _ORIGINAL_POLYGON
    if _ORIGINAL_POLYGON is not None:
        from babaorum import page_graph
        page_graph._polygon = _ORIGINAL_POLYGON
        _ORIGINAL_POLYGON = None


# --- adapter: chain units -> page_graph.v1 ----------------------------------------

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _points(raw: Any) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in raw]


def _drop_closing_vertex(points: list[list[float]]) -> tuple[list[list[float]], bool]:
    if len(points) > 3 and points[0] == points[-1]:
        return points[:-1], True
    return points, False


def _geometry_or_error(points: list[list[float]] | None, kind: str, source: str,
                       width: int, height: int, path: str) -> dict[str, Any]:
    if not points:
        return make_geometry(None)
    candidate = make_geometry(points, source)
    try:
        _geometry(candidate, path, width, height, kind=kind)
    except PageGraphError:
        return make_geometry(None, error=True)
    return candidate


def _orientation(baseline: dict[str, Any]) -> str:
    if baseline["status"] != "available":
        return "unknown"
    first, last = baseline["points"][0], baseline["points"][-1]
    if last[0] > first[0]:
        return "ltr"
    if last[0] < first[0]:
        return "rtl"
    return "unknown"


def _bbox_polygon(polygons: list[list[list[float]]]) -> list[list[float]] | None:
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    if not xs:
        return None
    return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]


def _coverage(status: str, expected: int | None, available: int, errors: int = 0) -> dict[str, Any]:
    return {"status": status, "expected_count": expected, "available_count": available, "error_count": errors}


def build_graph(units: Sequence[Mapping[str, Any]], *, role: str, identity: Mapping[str, Any],
                image: Mapping[str, Any], provenance: Mapping[str, Any],
                block_labels: Mapping[str, str] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    """Convert ordered text units into a validated page_graph.v1 and report what was lost.

    Each unit: id, text, polygon, baseline, optional block (id) and optional source.
    A prediction graph refuses units that carry ground-truth provenance. Invalid
    geometry becomes explicit ``error`` geometry on a ``partial`` node; nothing is
    clipped or repaired except removing a repeated closing vertex, which is counted.
    """
    if role not in ROLE_SOURCE:
        raise BenchError(f"unknown graph role: {role}")
    source = ROLE_SOURCE[role]
    width, height = int(image["width"]), int(image["height"])
    stats = {"units": len(units), "invalid_polygons": 0, "invalid_baselines": 0,
             "closing_vertex_dropped": 0, "partial_lines": 0}
    lines, block_members = [], {}
    for index, unit in enumerate(units):
        declared = unit.get("source")
        if role == "prediction" and declared in FORBIDDEN_IN_PREDICTION:
            raise BenchError(f"a {declared} unit cannot enter a prediction graph: {unit.get('id')}")
        polygon_points, dropped = _drop_closing_vertex(_points(unit["polygon"]))
        stats["closing_vertex_dropped"] += int(dropped)
        baseline_points = _points(unit["baseline"])
        polygon = _geometry_or_error(polygon_points, "polygon", source, width, height, f"$.lines[{index}].polygon")
        baseline = _geometry_or_error(baseline_points, "polyline", source, width, height, f"$.lines[{index}].baseline")
        invalid_polygon, invalid_baseline = polygon["status"] == "error", baseline["status"] == "error"
        stats["invalid_polygons"] += int(invalid_polygon)
        stats["invalid_baselines"] += int(invalid_baseline)
        partial = invalid_polygon or invalid_baseline
        stats["partial_lines"] += int(partial)
        node_id = str(unit["id"])
        lines.append({
            "id": node_id, "status": "partial" if partial else "ok", "polygon": polygon, "baseline": baseline,
            "baseline_semantics": "writing_line_foot_of_body", "orientation": _orientation(baseline),
            "baseline_band": {"status": "missing", "normal_negative_px": None, "normal_positive_px": None, "source": None},
            "text": make_text(str(unit["text"])),
        })
        block = unit.get("block")
        if block is not None:
            block_members.setdefault(str(block), []).append((node_id, polygon))
    blocks, relations = [], []
    for block_id, members in block_members.items():
        node_id = "block_" + re.sub(r"[^A-Za-z0-9_.-]", "_", block_id)
        boxes = [m[1]["points"] for m in members if m[1]["status"] == "available"]
        geometry = _geometry_or_error(_bbox_polygon(boxes), "polygon", source, width, height, f"$.blocks[{node_id}].polygon")
        blocks.append({"id": node_id, "status": "ok" if geometry["status"] == "available" else "partial",
                       "category": CATEGORY_BY_LABEL.get((block_labels or {}).get(block_id, ""), "autre"),
                       "polygon": geometry})
        relations += [{"type": "contains", "source_id": node_id, "target_id": m[0],
                       "source": source, "confidence": 1.0} for m in members]
    order = [{"type": "precedes", "source_id": a["id"], "target_id": b["id"], "source": source, "confidence": 1.0}
             for a, b in zip(lines, lines[1:])]
    relations += order
    n = len(lines)
    lines_cov = _coverage("complete" if not stats["partial_lines"] else "partial", n, n)
    if blocks:
        partial_blocks = sum(b["status"] == "partial" for b in blocks)
        blocks_cov = _coverage("complete" if not partial_blocks else "partial", len(blocks), len(blocks))
    else:
        blocks_cov = _coverage("unavailable", None, 0)
    clean = lines_cov["status"] == "complete" and blocks_cov["status"] in {"complete", "unavailable"}
    graph: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "graph_role": role,
        "identity": dict(identity), "coordinate_frame": COORDINATE_FRAME,
        "coordinate_convention": COORDINATE_CONVENTION, "normal_convention": NORMAL_CONVENTION,
        "image": dict(image), "provenance": dict(provenance),
        "run_status": "ok" if clean else "partial",
        "coverage": {"blocks": blocks_cov, "lines": lines_cov, "words": _coverage("unavailable", None, 0),
                     "transcription": _coverage("complete", n, n),
                     "reading_order": _coverage("complete", len(order), len(order))},
        "articles": [], "blocks": blocks, "lines": lines, "words": [], "relations": relations,
    }
    validate_page_graph(graph)
    return graph, stats


# --- P-01c: words on a valid line graph ---------------------------------------------------

def add_words(graph: dict[str, Any], words_by_line: Mapping[str, Sequence[tuple[str, Sequence[Sequence[float]] | None]]]) -> dict[str, int]:
    """Attach word nodes to the lines of a validated graph, in place, and re-validate it.

    ``words_by_line[line_id]`` lists (word text, polygon points or None) in reading order. A word without
    polygon is written with a ``missing`` geometry (never an imputed one). The line text is replaced by the
    words joined with one space, so that words and line agree to the character (whitespace runs and borders
    are collapsed; the count of such lines is returned, the original line text stays in the bench metrics).
    """
    source = ROLE_SOURCE["prediction"] if graph.get("graph_role", "prediction") == "prediction" else ROLE_SOURCE["ground_truth"]
    words, extra, order = [], [], []
    stats = {"words": 0, "words_without_box": 0, "lines_whitespace_normalized": 0, "lines_blank_source_kept": 0, "lines_without_words": 0}
    for line in graph["lines"]:
        items = list(words_by_line.get(line["id"], ()))
        joined = " ".join(text for text, _ in items)
        if line["text"]["status"] == "available" and line["text"]["value"] != joined:
            if joined == "" and line["text"]["value"] != "":
                stats["lines_blank_source_kept"] += 1  # whitespace-only source: kept as is, never hidden as an empty line
            else:
                stats["lines_whitespace_normalized"] += 1
                line["text"] = make_text(joined)
        if not items:
            stats["lines_without_words"] += 1
        previous = None
        for k, (text, polygon) in enumerate(items):
            node_id = f"{line['id']}_w{k}"
            geometry = make_geometry([[float(x), float(y)] for x, y in polygon], source) if polygon else make_geometry(None)
            words.append({"id": node_id, "status": "ok", "polygon": geometry, "text": make_text(text)})
            extra.append({"type": "contains", "source_id": line["id"], "target_id": node_id, "source": source, "confidence": 1.0})
            if previous is not None:
                order.append({"type": "precedes", "source_id": previous, "target_id": node_id, "source": source, "confidence": 1.0})
            previous = node_id
            stats["words"] += 1
            stats["words_without_box"] += polygon is None
    graph["words"] = words
    graph["relations"] = [*graph["relations"], *extra, *order]
    precedes = sum(r["type"] == "precedes" for r in graph["relations"])
    n_lines = len(graph["lines"])
    graph["coverage"]["words"] = _coverage("complete" if words else "unavailable", len(words) if words else None, len(words))
    graph["coverage"]["transcription"] = _coverage("complete", n_lines + len(words), n_lines + len(words))
    graph["coverage"]["reading_order"] = _coverage("complete", precedes, precedes)
    validate_page_graph(graph)
    return stats

