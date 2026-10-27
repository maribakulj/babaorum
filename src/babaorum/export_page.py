"""Auditable PAGE XML and ALTO projections of page_graph.v1.

The graph JSON is always returned as a required sidecar, hashed over canonical
UTF-8 JSON bytes. XML validity does not establish semantic fidelity or OCR truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping
import xml.etree.ElementTree as ET

from babaorum.page_graph import PageGraphError, _segments_intersect, validate_page_graph

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
PAGE_XSD_SHA256 = "5d7da5af5f5e06d3b9cd1e78b407ffca1862f78ad9823ed89c302fb6409932d5"
ALTO_XSD_SHA256 = "5564fd29d2dd090d8102b8a0aa081906afd677cd5ecc632312e56f21ea14702b"
XLINK_XSD_SHA256 = "f1f5bb6003165cdd8f6c1fcc32f8fd1f965e1681010f3b9806d9460bcffa8a3c"
DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

ET.register_namespace("", PAGE_NS)


class ExportError(ValueError):
    """Invalid export request or non-representable graph in strict mode."""


@dataclass(frozen=True)
class ExportLoss:
    path: str
    code: str
    severity: Literal["info", "warning", "error"]
    explanation: str
    preserved_in_sidecar: bool = True


@dataclass(frozen=True)
class ExportResult:
    format: Literal["pagexml-2019-07-15", "alto-4.4"]
    xml: str | None
    sidecar_json: str
    sidecar_sha256: str
    schema_sha256: str
    losses: tuple[ExportLoss, ...]
    accepted: bool
    quantization: str
    max_coordinate_error_px: float


def _canonical_graph(graph: Mapping[str, Any]) -> tuple[str, str]:
    try:
        payload = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (UnicodeEncodeError, TypeError, ValueError) as exc:
        raise ExportError(f"page graph cannot be encoded as canonical UTF-8 sidecar: {exc}") from exc
    return payload, digest


def _check_xml10_string(value: str, path: str) -> None:
    for index, char in enumerate(value):
        code = ord(char)
        allowed = code in (0x9, 0xA, 0xD) or 0x20 <= code <= 0xD7FF or 0xE000 <= code <= 0xFFFD or 0x10000 <= code <= 0x10FFFF
        if not allowed:
            raise ExportError(f"XML 1.0-illegal character U+{code:04X} at {path}[{index}]")
        # XML normalizes literal carriage returns during parsing. Reject them
        # rather than silently changing a transcription or filename.
        if code == 0x0D:
            raise ExportError(f"XML carriage return would be normalized at {path}[{index}]")


def _check_export_strings(graph: Mapping[str, Any], image_filename: str) -> None:
    _check_xml10_string(image_filename, "image_filename")
    for collection in ("lines", "words"):
        for node in graph[collection]:
            text = node["text"]
            if text["status"] == "available":
                _check_xml10_string(text["value"], f"{collection}/{node['id']}/text")


def _loss(losses: list[ExportLoss], path: str, code: str, explanation: str,
          *, severity: Literal["info", "warning", "error"] = "warning") -> None:
    losses.append(ExportLoss(path, code, severity, explanation, True))


def _report_exclusive_edges(graph: Mapping[str, Any], losses: list[ExportLoss]) -> None:
    width, height = graph["image"]["width"], graph["image"]["height"]
    for collection in ("blocks", "lines", "words"):
        for node in graph[collection]:
            shapes = [("polygon", node.get("polygon", {}))]
            if collection == "lines":
                shapes.append(("baseline", node.get("baseline", {})))
            for name, shape in shapes:
                if shape.get("status") != "available":
                    continue
                points = shape["points"]
                path = f"{collection}/{node['id']}/{name}"
                if any(x == width for x, _ in points):
                    _loss(losses, path, "exclusive_right_edge",
                          "coordinate x=W is retained although the source frame uses the continuous edge [0,W]")
                if any(y == height for _, y in points):
                    _loss(losses, path, "exclusive_bottom_edge",
                          "coordinate y=H is retained although the source frame uses the continuous edge [0,H]")


def _reject_or_record(mode: str, losses: list[ExportLoss], path: str, code: str,
                      explanation: str, *, always_reject: bool = False) -> None:
    _loss(losses, path, code, explanation, severity="error" if mode == "strict" or always_reject else "warning")
    if mode == "strict" or always_reject:
        raise _ProjectionRejected


class _ProjectionRejected(Exception):
    pass


def _schema_path(format: str, schema_dir: str | Path | None) -> Path:
    root = Path(schema_dir) if schema_dir is not None else Path(os.environ.get("HERITAGE_SYNTH_SCHEMA_DIR", DEFAULT_SCHEMA_DIR))
    name = "page-2019-07-15.xsd" if format == "page" else "alto-4-4.xsd"
    path = root / name
    if not path.is_file():
        raise ExportError(f"local pinned schema not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = PAGE_XSD_SHA256 if format == "page" else ALTO_XSD_SHA256
    if digest != expected:
        raise ExportError(f"schema hash mismatch for {path}: {digest}")
    if format == "alto":
        dep = root / "xlink.xsd"
        if not dep.is_file() or hashlib.sha256(dep.read_bytes()).hexdigest() != XLINK_XSD_SHA256:
            raise ExportError(f"missing or mismatched pinned ALTO dependency: {dep}")
    return path


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    # Python's repr is the shortest decimal that round-trips to the same
    # binary64 value; fixed 15 significant digits would silently lose bits.
    return repr(value)


def _point_list(points: list[list[float | int]], *, page_integer: bool,
                quantization: str, state: dict[str, float], path: str,
                geometry_kind: Literal["polygon", "polyline"] | None = None) -> str:
    out: list[str] = []
    projected: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(points):
        vals: list[str] = []
        for axis, raw in (("x", x), ("y", y)):
            v = float(raw)
            if not math.isfinite(v):
                raise ExportError(f"{path}[{i}].{axis} is not finite")
            if page_integer and not v.is_integer():
                if quantization != "nearest_pixel":
                    raise ExportError(f"PAGE coordinates require integer pixels; explicitly request quantization='nearest_pixel' ({path}[{i}])")
                rounded = int(math.floor(v + 0.5))
                state["max_error"] = max(state["max_error"], abs(v - rounded))
                state.setdefault("quantized", []).append(f"{path}[{i}].{axis}")
                v = float(rounded)
            vals.append(_fmt(int(v) if page_integer else v))
        out.append(",".join(vals))
        projected.append((float(vals[0]), float(vals[1])))
    if page_integer and geometry_kind == "polygon":
        if len(set(projected)) < 3 or any(projected[i] == projected[(i + 1) % len(projected)] for i in range(len(projected))):
            raise ExportError(f"PAGE quantization collapses polygon geometry ({path})")
        twice_area = sum(projected[i][0] * projected[(i + 1) % len(projected)][1]
                         - projected[(i + 1) % len(projected)][0] * projected[i][1]
                         for i in range(len(projected)))
        if not math.isfinite(twice_area) or abs(twice_area) < 1e-12:
            raise ExportError(f"PAGE quantization collapses polygon area ({path})")
        # m4: rounding can make a simple polygon self-intersect or touch itself
        # even when its area stays positive; refuse rather than emit it.
        n = len(projected)
        for i in range(n):
            a, b = projected[i], projected[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # adjacent through the closing edge
                if _segments_intersect(a, b, projected[j], projected[(j + 1) % n]):
                    raise ExportError(f"PAGE quantization makes polygon non-simple ({path})")
    if page_integer and geometry_kind == "polyline":
        if len(set(projected)) < 2 or any(a == b for a, b in zip(projected, projected[1:])):
            raise ExportError(f"PAGE quantization collapses baseline geometry ({path})")
    return " ".join(out)


def _idmap(nodes: list[Mapping[str, Any]]) -> dict[str, str]:
    # page_graph IDs already obey XML ID syntax; prefix keeps IDs distinct across
    # node kinds and avoids clashes with IDs generated for groups.
    return {node["id"]: f"n_{node['id']}" for node in nodes}


def _parent_map(graph: Mapping[str, Any]) -> dict[str, str]:
    return {r["target_id"]: r["source_id"] for r in graph["relations"] if r["type"] == "contains"}


def _ancestor(node_id: str, kind_by_id: Mapping[str, str], parent: Mapping[str, str], wanted: str) -> str | None:
    cur: str | None = node_id
    while cur is not None:
        if kind_by_id[cur] == wanted:
            return cur
        cur = parent.get(cur)
    return None


def _ordered_siblings(graph: Mapping[str, Any], child_ids: list[str], kind_by_id: Mapping[str, str],
                      parent: Mapping[str, str], mode: str, losses: list[ExportLoss], path: str) -> list[str]:
    if len(child_ids) < 2:
        return child_ids
    # Project any explicit order edge down to the immediate sibling level.
    edges: set[tuple[str, str]] = set()
    for rel in graph["relations"]:
        if rel["type"] != "precedes":
            continue
        a = _ancestor(rel["source_id"], kind_by_id, parent, kind_by_id[child_ids[0]])
        b = _ancestor(rel["target_id"], kind_by_id, parent, kind_by_id[child_ids[0]])
        if a in child_ids and b in child_ids and a != b:
            edges.add((a, b))
    adjacency = {x: set() for x in child_ids}
    indegree = {x: 0 for x in child_ids}
    for a, b in edges:
        if b not in adjacency[a]:
            adjacency[a].add(b)
            indegree[b] += 1
    order: list[str] = []
    remaining = set(child_ids)
    while remaining:
        zero = sorted(x for x in remaining if indegree[x] == 0)
        if not zero:
            raise ExportError(f"cyclic sibling precedence at {path}")
        if len(zero) > 1:
            _reject_or_record(mode, losses, path, "ambiguous_order",
                              "sibling order is not uniquely determined by explicit precedes relations")
            # Best-effort deterministic output only; report the added tie break.
            # Continue Kahn's algorithm from a deterministic zero-indegree
            # choice. Sorting all remaining IDs could violate known arcs.
            cur = zero[0]
            order.append(cur)
            remaining.remove(cur)
            for nxt in adjacency[cur]:
                indegree[nxt] -= 1
            continue
        cur = zero[0]
        order.append(cur)
        remaining.remove(cur)
        for nxt in adjacency[cur]:
            indegree[nxt] -= 1
    return order


def _collect(graph: Mapping[str, Any], mode: str, losses: list[ExportLoss], fmt: str):
    kind_by_id: dict[str, str] = {}
    for key, kind in (("articles", "article"), ("blocks", "block"), ("lines", "line"), ("words", "word")):
        for n in graph[key]:
            kind_by_id[n["id"]] = kind
    parent = _parent_map(graph)
    blocks = {n["id"]: n for n in graph["blocks"]}
    lines = {n["id"]: n for n in graph["lines"]}
    words = {n["id"]: n for n in graph["words"]}
    # Empty blocks are retained as geometry regions only in PAGE; ALTO needs
    # string content to form text blocks in this constrained profile.
    supported_categories = {"texte", "titre", "legende", "annonce"}
    for b in blocks.values():
        if b["category"] not in supported_categories:
            _reject_or_record(mode, losses, f"blocks/{b['id']}", "unsupported_block_category",
                              f"category {b['category']!r} is not mapped to this text export profile")
    for a in graph["articles"]:
        _reject_or_record(mode, losses, f"articles/{a['id']}", "article_layer_omitted",
                          "article grouping has no lossless mapping in this export profile")
    for node_id, node in [*blocks.items(), *lines.items(), *words.items()]:
        if node["status"] != "ok":
            _reject_or_record(mode, losses, f"nodes/{node_id}", "node_not_ok",
                              "node is partial or failed and cannot be represented as a complete export unit")
        if node["polygon"]["status"] != "available":
            if fmt == "alto" and node_id in words and node["polygon"]["status"] == "missing":
                # ALTO 4.4 String coordinates (HPOS, VPOS, WIDTH, HEIGHT) are optional: a word whose geometry is
                # unavailable is written without them, never with an imputed box. A failed geometry still rejects.
                _loss(losses, f"words/{node_id}/polygon", "word_geometry_unavailable",
                      "word geometry is unavailable: the String is written without HPOS/VPOS/WIDTH/HEIGHT and without Shape",
                      severity="info")
                continue
            _reject_or_record(mode, losses, f"nodes/{node_id}/polygon", "geometry_unavailable",
                              "required contour is missing or failed")
    if fmt == "page":
        for n in lines.values():
            if n["text"]["status"] != "available":
                _reject_or_record(mode, losses, f"lines/{n['id']}/text", "text_unavailable",
                                  "PAGE line TextEquiv requires available transcription in this profile")
    else:
        omitted_empty: set[str] = set()
        for n in words.values():
            if n["text"]["status"] != "available":
                _reject_or_record(mode, losses, f"words/{n['id']}/text", "text_unavailable",
                                  "ALTO String requires available word text")
        for line in lines.values():
            line_words = [w for w, node in words.items() if parent.get(w) == line["id"]
                          and node["status"] == "ok" and node["polygon"]["status"] in {"available", "missing"}
                          and node["text"]["status"] == "available"]
            if not line_words:
                if line["text"]["status"] == "available" and line["text"]["value"].strip() == "" and not any(
                        parent.get(w) == line["id"] for w in words):
                    # S1/S5: a line whose source text is empty (empty_line_omitted) or made only of whitespace
                    # (blank_line_omitted) carries nothing a String could hold (an empty CONTENT is refused on purpose).
                    # It is omitted from the ALTO only; PAGE and the graph sidecar keep it. A line with at least one
                    # non-blank character enters neither code and is never omitted.
                    code = "empty_line_omitted" if line["text"]["value"] == "" else "blank_line_omitted"
                    omitted_empty.add(line["id"])
                    _loss(losses, f"lines/{line['id']}", code,
                          "line with empty source text and no word is omitted from the ALTO; it stays in the PAGE and in the graph sidecar"
                          if code == "empty_line_omitted" else
                          "line whose source text is only whitespace, with no word, is omitted from the ALTO; it stays in the PAGE and in the graph sidecar",
                          severity="info")
                    continue
                _reject_or_record(mode, losses, f"lines/{line['id']}", "words_required_for_alto",
                                  "ALTO line transcription is represented only by actual word nodes")
            if line["text"]["status"] != "available":
                _reject_or_record(mode, losses, f"lines/{line['id']}/text", "line_text_unavailable",
                                  "exact line text is required to check ALTO word serialization")
                if mode == "best_effort":
                    continue
            ordered = _ordered_siblings(graph, line_words, kind_by_id, parent, mode, losses, f"lines/{line['id']}/words")
            joined = " ".join(words[w]["text"]["value"] for w in ordered)
            if joined != line["text"]["value"]:
                _reject_or_record(mode, losses, f"lines/{line['id']}/text", "word_join_mismatch",
                                  "single-space-joined available word strings are not byte-for-byte equal to line text")
    if fmt == "alto":
        # flagged copies: they stay in the ordering of their block (so the order of their neighbours is still
        # determined) and are skipped when the XML is written
        lines = {k: ({**v, "_omit": True} if k in omitted_empty else v) for k, v in lines.items()}
    # Required containment avoids losing parent grouping and prevents implicit
    # row/order based attachment.
    for lid in lines:
        if parent.get(lid) not in blocks:
            _reject_or_record(mode, losses, f"lines/{lid}", "line_parent_missing", "line has no containing block")
    for wid in words:
        if parent.get(wid) not in lines:
            _reject_or_record(mode, losses, f"words/{wid}", "word_parent_missing", "word has no containing line")
    if mode == "best_effort":
        original_lines, original_words = set(lines), set(words)
        blocks = {k: v for k, v in blocks.items()
                  if v["status"] == "ok" and v["polygon"]["status"] == "available"
                  and v["category"] in supported_categories and parent.get(k) not in blocks}
        lines = {k: v for k, v in lines.items()
                 if v["status"] == "ok" and v["polygon"]["status"] == "available"
                 and v["text"]["status"] == "available" and parent.get(k) in blocks}
        words = {k: v for k, v in words.items()
                 if v["status"] == "ok"
                 and v["polygon"]["status"] in ({"available"} if fmt == "page" else {"available", "missing"})
                 and (fmt == "page" or v["text"]["status"] == "available")
                 and parent.get(k) in lines}
        if fmt == "alto":
            for line_id in list(lines):
                if lines[line_id].get("_omit"):
                    continue
                child = [wid for wid in words if parent.get(wid) == line_id]
                child = _ordered_siblings(graph, child, kind_by_id, parent, mode, losses, f"lines/{line_id}/words") if child else []
                if not child or " ".join(words[w]["text"]["value"] for w in child) != lines[line_id]["text"]["value"]:
                    _loss(losses, f"lines/{line_id}", "line_not_exported", "line cannot be represented exactly by available word String nodes")
                    lines.pop(line_id)
            words = {k: v for k, v in words.items() if parent.get(k) in lines}
        for line_id in original_lines - set(lines):
            if parent.get(line_id) not in blocks:
                _loss(losses, f"lines/{line_id}", "ancestor_omitted", "line was not exported because its containing block is not representable")
        for word_id in original_words - set(words):
            if parent.get(word_id) not in lines:
                _loss(losses, f"words/{word_id}", "ancestor_omitted", "word was not exported because its containing line is not representable")
    return blocks, lines, words, kind_by_id, parent


def _base_result(fmt: str, graph: Mapping[str, Any], xml: str | None, losses: list[ExportLoss],
                 quantization: str, max_error: float, schema_dir: str | Path | None,
                 *, accepted: bool) -> ExportResult:
    sidecar, side_hash = _canonical_graph(graph)
    schema_file = _schema_path("page" if fmt.startswith("page") else "alto", schema_dir)
    digest = PAGE_XSD_SHA256 if fmt.startswith("page") else ALTO_XSD_SHA256
    return ExportResult(fmt, xml, sidecar, side_hash, digest, tuple(losses), accepted, quantization, max_error)


def export_pagexml(graph: Mapping[str, Any], *, image_filename: str,
                   mode: Literal["strict", "best_effort"] = "strict",
                   quantization: Literal["reject", "nearest_pixel"] = "reject",
                   schema_dir: str | Path | None = None) -> ExportResult:
    """Project to PAGE 2019-07-15; graph sidecar is always returned and hashed."""
    validate_page_graph(graph)
    if mode not in {"strict", "best_effort"}:
        raise ExportError("mode must be 'strict' or 'best_effort'")
    if quantization not in {"reject", "nearest_pixel"}:
        raise ExportError("quantization must be 'reject' or 'nearest_pixel'")
    if not isinstance(image_filename, str) or not image_filename.strip():
        raise ExportError("image_filename must be supplied explicitly")
    _check_export_strings(graph, image_filename)
    losses: list[ExportLoss] = []
    for path, code, desc in (
        ("identity", "identity_in_sidecar", "graph identities are preserved in the required graph sidecar"),
        ("image/sha256", "image_hash_in_sidecar", "image hash is preserved in the required graph sidecar"),
        ("provenance", "provenance_in_sidecar", "run and component provenance are preserved in the required graph sidecar"),
        ("coverage", "coverage_in_sidecar", "coverage and error statuses are preserved in the required graph sidecar"),
        ("geometry/source", "geometry_source_in_sidecar", "geometry source labels (predicted/transferred/ground_truth/interpolated) are preserved only in the graph sidecar"),
        ("geometry/coordinate_frame", "coordinate_frame_in_sidecar", "continuous pixel-edge coordinate convention is preserved in the graph sidecar"),
        ("lines/baseline_band", "baseline_band_in_sidecar", "baseline-band extents and their source are preserved only in the graph sidecar"),
        ("lines/baseline_semantics", "baseline_semantics_in_sidecar", "baseline semantics are preserved in the graph sidecar"),
        ("geometry/normal_convention", "normal_convention_in_sidecar", "normal convention is preserved in the graph sidecar"),
        ("relations/confidence", "relation_confidence_in_sidecar", "relation confidence values are preserved only in the graph sidecar"),
        ("relations/source", "relation_source_in_sidecar", "relation provenance is preserved in the graph sidecar"),
        ("provenance/run_status", "run_status_in_sidecar", "run status is preserved in the graph sidecar"),
    ):
        _loss(losses, path, code, desc, severity="info")
    state = {"max_error": 0.0, "quantized": []}
    try:
        blocks, lines, words, kind_by_id, parent = _collect(graph, mode, losses, "page")
        root = ET.Element(f"{{{PAGE_NS}}}PcGts")
        metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
        ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = "Babaorum page_graph exporter"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        ET.SubElement(metadata, f"{{{PAGE_NS}}}Created").text = now
        ET.SubElement(metadata, f"{{{PAGE_NS}}}LastChange").text = now
        page = ET.SubElement(root, f"{{{PAGE_NS}}}Page", {
            "imageFilename": image_filename,
            "imageWidth": str(graph["image"]["width"]),
            "imageHeight": str(graph["image"]["height"]),
        })
        bid = _idmap(list(blocks.values()))
        lid = _idmap(list(lines.values()))
        wid = _idmap(list(words.values()))
        roots = [b for b in blocks if parent.get(b) not in blocks]
        root_order = _ordered_siblings(graph, roots, kind_by_id, parent, mode, losses, "blocks/order") if roots else []
        # ReadingOrder is mandatory semantically only where order constraints exist;
        # for multiple regions it records the uniquely proven total order.
        if len(root_order) > 1:
            ro = ET.SubElement(page, f"{{{PAGE_NS}}}ReadingOrder")
            group = ET.SubElement(ro, f"{{{PAGE_NS}}}OrderedGroup", {"id": "ro_0"})
            for ix, b in enumerate(root_order):
                ET.SubElement(group, f"{{{PAGE_NS}}}RegionRefIndexed", {"index": str(ix), "regionRef": bid[b]})
        for b_id in root_order:
            b = blocks[b_id]
            region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": bid[b_id]})
            ET.SubElement(region, f"{{{PAGE_NS}}}Coords", {"points": _point_list(b["polygon"]["points"], page_integer=True, quantization=quantization, state=state, path=f"blocks/{b_id}/polygon", geometry_kind="polygon")})
            child_lines = [x for x in lines if parent.get(x) == b_id]
            ordered_lines = _ordered_siblings(graph, child_lines, kind_by_id, parent, mode, losses, f"blocks/{b_id}/lines") if child_lines else []
            for idx, line_id in enumerate(ordered_lines):
                line = lines[line_id]
                attrs = {"id": lid[line_id], "index": str(idx)}
                direction = {"ltr": "left-to-right", "rtl": "right-to-left",
                             "ttb": "top-to-bottom", "btt": "bottom-to-top"}
                if line["orientation"] in direction:
                    attrs["readingDirection"] = direction[line["orientation"]]
                line_el = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", attrs)
                ET.SubElement(line_el, f"{{{PAGE_NS}}}Coords", {"points": _point_list(line["polygon"]["points"], page_integer=True, quantization=quantization, state=state, path=f"lines/{line_id}/polygon", geometry_kind="polygon")})
                if line["baseline"]["status"] == "available":
                    ET.SubElement(line_el, f"{{{PAGE_NS}}}Baseline", {"points": _point_list(line["baseline"]["points"], page_integer=True, quantization=quantization, state=state, path=f"lines/{line_id}/baseline", geometry_kind="polyline")})
                else:
                    _loss(losses, f"lines/{line_id}/baseline", "baseline_omitted", "no baseline available; no geometry was fabricated")
                child_words = [x for x in words if parent.get(x) == line_id]
                ordered_words = _ordered_siblings(graph, child_words, kind_by_id, parent, mode, losses, f"lines/{line_id}/words") if child_words else []
                for word_id in ordered_words:
                    word = words[word_id]
                    we = ET.SubElement(line_el, f"{{{PAGE_NS}}}Word", {"id": wid[word_id]})
                    ET.SubElement(we, f"{{{PAGE_NS}}}Coords", {"points": _point_list(word["polygon"]["points"], page_integer=True, quantization=quantization, state=state, path=f"words/{word_id}/polygon", geometry_kind="polygon")})
                    if word["text"]["status"] == "available":
                        te = ET.SubElement(we, f"{{{PAGE_NS}}}TextEquiv")
                        ET.SubElement(te, f"{{{PAGE_NS}}}Unicode").text = word["text"]["value"]
                    else:
                        _loss(losses, f"words/{word_id}/text", "word_text_omitted", "word geometry is retained but no word transcription is available")
                te = ET.SubElement(line_el, f"{{{PAGE_NS}}}TextEquiv")
                ET.SubElement(te, f"{{{PAGE_NS}}}Unicode").text = line["text"]["value"]
        # Record any graph nodes not represented in XML in best-effort mode.
        represented_blocks = set(root_order)
        represented_lines = {x for b in root_order for x in lines if parent.get(x) == b}
        represented_words = {x for x in words if parent.get(x) in represented_lines}
        for collection, represented in ((blocks, represented_blocks), (lines, represented_lines), (words, represented_words)):
            for node_id in collection.keys() - represented:
                if any(loss.path == f"{collection_name}/{node_id}" and loss.code == "ancestor_omitted"
                       for loss in losses for collection_name in ("lines", "words")):
                    continue
                _reject_or_record(mode, losses, f"nodes/{node_id}", "node_omitted", "node could not be attached to a representable XML parent")
        for path in state["quantized"]:
            _loss(losses, path, "coordinate_quantized",
                  "PAGE integer coordinate rounded to nearest pixel; maximum error is reported separately")
        _report_exclusive_edges(graph, losses)
        for path, code, desc in (
            ("relations/source", "relation_source_in_sidecar", "relation provenance is preserved in the graph sidecar"),
            ("geometry/normal_convention", "normal_convention_in_sidecar", "normal convention is preserved in the graph sidecar"),
            ("lines/baseline_semantics", "baseline_semantics_in_sidecar", "baseline semantics are preserved in the graph sidecar"),
            ("provenance/run_status", "run_status_in_sidecar", "run status is preserved in the graph sidecar"),
        ):
            _loss(losses, path, code, desc, severity="info")
        _loss(losses, "blocks/category", "category_coarsened", "supported text block categories are serialized as PAGE TextRegion; exact category remains in sidecar", severity="info")
        xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
        result = _base_result("pagexml-2019-07-15", graph, xml, losses, quantization, state["max_error"], schema_dir, accepted=True)
        return result
    except _ProjectionRejected:
        return _base_result("pagexml-2019-07-15", graph, None, losses, quantization, state["max_error"], schema_dir, accepted=False)


def _alto_shape(parent: ET.Element, points: list[list[float | int]]) -> None:
    shape = ET.SubElement(parent, f"{{{ALTO_NS}}}Shape")
    poly = ET.SubElement(shape, f"{{{ALTO_NS}}}Polygon")
    poly.set("POINTS", _point_list(points, page_integer=False, quantization="reject", state={"max_error": 0.0}, path="ALTO/Shape"))


def _float32_exact(value: float) -> bool:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0] == value
    except OverflowError:
        return False


def _report_alto_float32(graph: Mapping[str, Any], blocks, lines, words,
                         losses: list[ExportLoss]) -> None:
    """m5: ALTO 4.4 types HPOS/VPOS/WIDTH/HEIGHT as xsd:float (32-bit)."""
    inexact = []
    for collection, nodes in (("blocks", blocks), ("lines", lines), ("words", words)):
        for node_id, node in nodes.items():
            box = _bbox(node["polygon"]["points"])
            if not all(_float32_exact(float(value)) for value in box.values()):
                inexact.append(f"{collection}/{node_id}/polygon")
    explanation = ("ALTO 4.4 types HPOS, VPOS, WIDTH and HEIGHT as xsd:float (IEEE 32-bit); "
                   "values are written as shortest binary64 decimals and a conforming float32 reader may round them; "
                   "exact coordinates remain in the graph sidecar")
    _loss(losses, "alto/bbox_attributes", "alto_bbox_xsd_float32",
          explanation + (f"; not float32-exact: {len(inexact)} node(s)" if inexact else "; all emitted values are float32-exact"),
          severity="warning" if inexact else "info")
    for path in inexact:
        _loss(losses, path, "alto_bbox_float32_rounding",
              "HPOS/VPOS/WIDTH/HEIGHT of this node are not exactly representable as xsd:float")


def _bbox(points: list[list[float | int]]) -> dict[str, str]:
    xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
    return {"HPOS": _fmt(min(xs)), "VPOS": _fmt(min(ys)), "WIDTH": _fmt(max(xs) - min(xs)), "HEIGHT": _fmt(max(ys) - min(ys))}


def export_alto(graph: Mapping[str, Any], *, image_filename: str, physical_img_nr: int | float,
                mode: Literal["strict", "best_effort"] = "strict",
                schema_dir: str | Path | None = None) -> ExportResult:
    """Project actual word nodes to ALTO 4.4; never synthesizes String units."""
    validate_page_graph(graph)
    if mode not in {"strict", "best_effort"}:
        raise ExportError("mode must be 'strict' or 'best_effort'")
    if not isinstance(image_filename, str) or not image_filename.strip():
        raise ExportError("image_filename must be supplied explicitly")
    _check_export_strings(graph, image_filename)
    if type(physical_img_nr) is not int or not 1 <= physical_img_nr <= 2_147_483_647:
        raise ExportError("physical_img_nr must be an explicitly supplied positive 32-bit integer")
    if any(node["text"]["status"] == "available" and node["text"]["value"] == ""
           for node in graph["words"]):
        raise ExportError("ALTO cannot encode an available word with empty CONTENT")
    losses: list[ExportLoss] = []
    for path, code, desc in (
        ("identity", "identity_in_sidecar", "graph identities are preserved in the required graph sidecar"),
        ("image/sha256", "image_hash_in_sidecar", "image hash is preserved in the required graph sidecar"),
        ("provenance", "provenance_in_sidecar", "run and component provenance are preserved in the required graph sidecar"),
        ("coverage", "coverage_in_sidecar", "coverage and error statuses are preserved in the required graph sidecar"),
        ("geometry/source", "geometry_source_in_sidecar", "geometry source labels (predicted/transferred/ground_truth/interpolated) are preserved only in the graph sidecar"),
        ("geometry/coordinate_frame", "coordinate_frame_in_sidecar", "continuous pixel-edge coordinate convention is preserved in the graph sidecar"),
        ("lines/baseline_band", "baseline_band_in_sidecar", "baseline-band extents and their source are preserved only in the graph sidecar"),
        ("lines/baseline_semantics", "baseline_semantics_in_sidecar", "baseline semantics are preserved in the graph sidecar"),
        ("geometry/normal_convention", "normal_convention_in_sidecar", "normal convention is preserved in the graph sidecar"),
        ("relations/confidence", "relation_confidence_in_sidecar", "relation confidence values are preserved only in the graph sidecar"),
        ("relations/source", "relation_source_in_sidecar", "relation provenance is preserved in the graph sidecar"),
        ("provenance/run_status", "run_status_in_sidecar", "run status is preserved in the graph sidecar"),
    ):
        _loss(losses, path, code, desc, severity="info")
    try:
        blocks, lines, words, kind_by_id, parent = _collect(graph, mode, losses, "alto")
        _report_exclusive_edges(graph, losses)
        root = ET.Element(f"{{{ALTO_NS}}}alto", {"SCHEMAVERSION": "4.4"})
        desc = ET.SubElement(root, f"{{{ALTO_NS}}}Description")
        ET.SubElement(desc, f"{{{ALTO_NS}}}MeasurementUnit").text = "pixel"
        source = ET.SubElement(desc, f"{{{ALTO_NS}}}sourceImageInformation")
        ET.SubElement(source, f"{{{ALTO_NS}}}fileName").text = image_filename
        layout = ET.SubElement(root, f"{{{ALTO_NS}}}Layout")
        page = ET.SubElement(layout, f"{{{ALTO_NS}}}Page", {
            "ID": "page_0", "PHYSICAL_IMG_NR": _fmt(physical_img_nr),
            "WIDTH": str(graph["image"]["width"]), "HEIGHT": str(graph["image"]["height"]),
        })
        print_space = ET.SubElement(page, f"{{{ALTO_NS}}}PrintSpace", {
            "ID": "printspace_0", "HPOS": "0", "VPOS": "0",
            "WIDTH": str(graph["image"]["width"]), "HEIGHT": str(graph["image"]["height"]),
        })
        bid = _idmap(list(blocks.values()))
        lid = _idmap(list(lines.values()))
        wid = _idmap(list(words.values()))
        roots = [b for b in blocks if parent.get(b) not in blocks]
        root_order = _ordered_siblings(graph, roots, kind_by_id, parent, mode, losses, "blocks/order") if roots else []
        for block_id in root_order:
            block = blocks[block_id]
            be = ET.SubElement(print_space, f"{{{ALTO_NS}}}TextBlock", {"ID": bid[block_id], **_bbox(block["polygon"]["points"])})
            _alto_shape(be, block["polygon"]["points"])
            child_lines = [x for x in lines if parent.get(x) == block_id]
            ordered_lines = _ordered_siblings(graph, child_lines, kind_by_id, parent, mode, losses, f"blocks/{block_id}/lines") if child_lines else []
            for line_id in ordered_lines:
                line = lines[line_id]
                if line.get("_omit"):
                    continue
                attrs = {"ID": lid[line_id], **_bbox(line["polygon"]["points"])}
                if line["baseline"]["status"] == "available":
                    attrs["BASELINE"] = _point_list(line["baseline"]["points"], page_integer=False, quantization="reject", state={"max_error": 0.0}, path=f"lines/{line_id}/baseline")
                else:
                    _loss(losses, f"lines/{line_id}/baseline", "baseline_omitted", "no baseline available; no geometry was fabricated")
                if line["orientation"] != "unknown":
                    attrs["BASEDIRECTION"] = line["orientation"]
                le = ET.SubElement(be, f"{{{ALTO_NS}}}TextLine", attrs)
                _alto_shape(le, line["polygon"]["points"])
                child_words = [x for x in words if parent.get(x) == line_id]
                ordered_words = _ordered_siblings(graph, child_words, kind_by_id, parent, mode, losses, f"lines/{line_id}/words") if child_words else []
                for ix, word_id in enumerate(ordered_words):
                    word = words[word_id]
                    string_attrs = {"ID": wid[word_id], "CONTENT": word["text"]["value"]}
                    if word["polygon"]["status"] == "available":
                        string_attrs.update(_bbox(word["polygon"]["points"]))
                    se = ET.SubElement(le, f"{{{ALTO_NS}}}String", string_attrs)
                    if word["polygon"]["status"] == "available":
                        _alto_shape(se, word["polygon"]["points"])
                    if ix + 1 < len(ordered_words):
                        ET.SubElement(le, f"{{{ALTO_NS}}}SP")
        exported_lines = {x: lines[x] for x in lines if parent.get(x) in root_order and not lines[x].get("_omit")}
        _report_alto_float32(graph, {x: blocks[x] for x in root_order}, exported_lines,
                             {x: words[x] for x in words if parent.get(x) in exported_lines
                              and words[x]["polygon"]["status"] == "available"},
                             losses)
        _loss(losses, "blocks/category", "category_coarsened", "supported text block categories are serialized as ALTO TextBlock; exact category remains in sidecar", severity="info")
        xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
        return _base_result("alto-4.4", graph, xml, losses, "none", 0.0, schema_dir, accepted=True)
    except _ProjectionRejected:
        return _base_result("alto-4.4", graph, None, losses, "none", 0.0, schema_dir, accepted=False)


def validate_export_xsd(result: ExportResult, *, schema_dir: str | Path | None = None) -> None:
    """Validate generated XML with lxml and the locally pinned XSD; never fetches schemas."""
    if not result.accepted or result.xml is None:
        raise ExportError("cannot XSD-validate a rejected export")
    try:
        from lxml import etree
    except ImportError as exc:  # optional runtime validation dependency
        raise ExportError("lxml is required for XSD validation but is not installed") from exc
    fmt = "page" if result.format.startswith("page") else "alto"
    xsd_path = _schema_path(fmt, schema_dir)
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    if fmt == "alto":
        local_xlink = xsd_path.parent / "xlink.xsd"

        class _LocalXLinkResolver(etree.Resolver):
            def resolve(self, url, pubid, context):
                if url.endswith("/xlink.xsd"):
                    return self.resolve_filename(str(local_xlink), context)
                if Path(url).resolve() == xsd_path.resolve():
                    return self.resolve_filename(str(xsd_path), context)
                raise ExportError(f"unexpected external schema reference: {url}")

        parser.resolvers.add(_LocalXLinkResolver())
    schema = etree.XMLSchema(etree.parse(str(xsd_path), parser))
    doc = etree.fromstring(result.xml.encode("utf-8"))
    if not schema.validate(doc):
        raise ExportError(f"XSD validation failed: {schema.error_log.last_error}")
