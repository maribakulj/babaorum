"""LOT-P02: pool records of a page -> LineProposal for the S1-fix reconciler. No truth, no text and no CTC value is used to choose.

Input: the ``zone`` records of ``pool/<page>.jsonl`` that ``select_blocks`` kept (exactly the set ``detector_chain`` walks before ``_covered``).
Each one becomes a ``local_block`` proposal whose baseline is expressed in the LOCAL frame of its zone crop (the real crop: clipped rectangle,
isotropic scale, padding), so that ``local_to_page`` gives back the page baseline. Text and confidence are copied, never used.

``proposal_id`` is geometric, so that ties are broken the same way in every run (kraken uuids are random):
    ``<zone>:<9999999 - 10 x baseline length>:<x0, 7 digits>:<y0, 7 digits>``  (longest baseline first, then position).
On a key collision, the whole geometry breaks the tie (every baseline point, then every polygon point, as integers), and the suffix
``:NNNN`` is the 1-based position in that sorted order. An exact duplicate of geometry is broken by the persistent SOURCE rank of its line
(its index in ``lines/<det>/<page>.json``, read in file order before any permutation, see ``source_rank_from_lines``), else by the line id.
Nothing depends on the order in which the candidates are received (v15, v16).
"""
from __future__ import annotations

import hashlib
import math
from statistics import median
from typing import Any, Mapping, Sequence

from babaorum.experimental.joint_reconcile import CropTransform, LineProposal


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0


def line_height(polygon: Sequence[Sequence[float]], baseline: Sequence[Sequence[float]]) -> float:
    """axel.layout.geom.line_height, copied: polygon area over the x extent of the baseline (bounding-box height as a fallback)."""
    if polygon and baseline and len(baseline) >= 2:
        xs = [p[0] for p in baseline]
        length = max(xs) - min(xs)
        if length > 0:
            h = polygon_area(polygon) / length
            if h > 0:
                return h
    if polygon:
        ys = [p[1] for p in polygon]
        return max(ys) - min(ys)
    return 0.0


def baseline_length(baseline: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(baseline, baseline[1:]))


def geometric_id(row: Mapping[str, Any]) -> str:
    bl = row["baseline"]
    x0, y0 = int(min(p[0] for p in bl)), int(min(p[1] for p in bl))
    return f"{row['zone_id']}:{9999999 - int(round(baseline_length(bl) * 10)):07d}:{x0:07d}:{y0:07d}"


def crop_transform_of(zone_crop: Mapping[str, Any]) -> CropTransform:
    scale = float(zone_crop["facteur_echelle"])
    rect = (float(zone_crop["x0"]), float(zone_crop["y0"]), float(zone_crop["x1"]), float(zone_crop["y1"]))
    bottom = (float(zone_crop["canvas_h"]) - (rect[3] - rect[1])) * scale if zone_crop.get("complete_de_blanc") else 0.0
    return CropTransform(rect, scale, (0.0, 0.0, 0.0, max(bottom, 0.0)))


def to_local(page_points: Sequence[Sequence[float]], ct: CropTransform) -> tuple[tuple[float, float], ...]:
    x0, y0 = ct.actual_rect_page[0], ct.actual_rect_page[1]
    left, top = ct.padding_ltrb[0], ct.padding_ltrb[1]
    return tuple(((x - x0) * ct.scale + left, (y - y0) * ct.scale + top) for x, y in page_points)


def full_geometry_key(r: Mapping[str, Any]) -> tuple[int, ...]:
    """The whole geometry, as integers: every baseline point, then every polygon point (v15)."""
    return tuple(int(round(v)) for p in r["baseline"] for v in p) + tuple(int(round(v)) for p in r["polygon"] for v in p)


def source_rank_from_lines(path) -> dict[str, int]:
    """{line_id: global index} read from ``lines/<det>/<page>.json`` IN FILE ORDER (block after block, line after line), before any
    permutation of the candidates (v16). This is the persistent source rank that ``dedup`` carries as ``_src``."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rank: dict[str, int] = {}
    index = 0
    for block in data["blocks"]:
        for line in block["lines"]:
            rank.setdefault(str(line["id"]), index)
            index += 1
    return rank


def source_key(r: Mapping[str, Any]) -> tuple[int, int, str]:
    """Integers compared, not strings (v16): the source rank when the candidate carries one (``_src``), else the line id as the fallback."""
    src = r.get("_src")
    return (0, int(src), "") if src is not None else (1, 0, str(r.get("line_id")))


def rows_to_proposals(rows: Sequence[Mapping[str, Any]], page: Mapping[str, Any]):
    """``page``: page_key, document_key, width, height, image_sha256. Returns (proposals, row_of_proposal_id, h, skipped).

    v15: the result does not depend on the order of ``rows``. A key collision is broken by the WHOLE geometry (baseline, then polygon, as
    integers); an exact duplicate of geometry is broken by the persistent source rank, else by the line id. The suffix is the 1-based position
    in that sorted order, never a rank taken from the order of reception.
    """
    kept = [dict(r) for r in rows if r.get("origin") == "zone" and r.get("zone_selected_by_select_blocks")]
    usable, skipped = [], []
    for r in kept:
        if not r.get("baseline") or len(r["baseline"]) < 2 or not r.get("zone_crop"):
            skipped.append({"line_id": r.get("line_id"), "why": "no baseline or no crop"})
            continue
        usable.append(r)
    if not usable:
        return [], {}, 0.0, skipped
    for r in usable:
        r["_pid"] = geometric_id(r)
    groups: dict[str, list[dict]] = {}
    for r in usable:
        groups.setdefault(r["_pid"], []).append(r)
    for pid, members in groups.items():
        if len(members) > 1:
            members.sort(key=lambda r: (full_geometry_key(r), source_key(r), str(r["line_id"])))
            for position, r in enumerate(members, start=1):
                r["_pid"] = f"{pid}:{position:04d}"
    if len({r["_pid"] for r in usable}) != len(usable):
        raise ValueError("proposal ids are not unique even with the blla rank")
    h = median(line_height(r["polygon"], r["baseline"]) for r in usable)
    props, row_of = [], {}
    for r in usable:
        ct = crop_transform_of(r["zone_crop"])
        local = [to_local([p], ct)[0] for p in r["baseline"]]
        dd = [local[0]]
        for q in local[1:]:
            if q != dd[-1]:
                dd.append(q)
        if len(dd) < 2:
            skipped.append({"line_id": r["line_id"], "why": "baseline of zero length"})
            continue
        rect = ct.actual_rect_page
        props.append(LineProposal(
            proposal_id=r["_pid"], source="local_block", page_key=page["page_key"], document_key=page.get("document_key"),
            page_size_wh=(float(page["width"]), float(page["height"])), page_sha256=page["image_sha256"], query_id=str(r["line_id"]),
            input_sha256=hashlib.sha256(f"{page['image_sha256']}{rect}{ct.scale}".encode()).hexdigest(), checkpoint_sha256=None,
            geometry_status="available", baseline_xy=tuple(dd), band_normal_negative_px=None, band_normal_positive_px=None,
            orientation="ltr", text_status="available", text=r.get("text") or "", token_ids=None, ctc_status="feasible",
            block_id=str(r["zone_id"]), crop_transform=ct))
        row_of[r["_pid"]] = r
    return props, row_of, float(h), skipped
