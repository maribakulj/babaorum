"""LOT-P02: exact, index-based replacements for the two quadratic loops of the S1-fix reconciler. S1-fix itself is untouched.

``build_graph_fast``    == joint_reconcile.build_reconciliation_graph, except that it only evaluates the pairs of proposals whose baselines
                           lie within the distance tolerance (spatial grid + exact polyline distance). Every other pair carries no label
                           (a label needs a coverage >= 0.75, hence a point within tolerance) and ``solve`` ignores unlabeled edges, so the
                           omitted edges change nothing. The pair evaluation is the original code, copied line for line.
``build_actions_fast``  == joint_reconcile_selection.build_actions (S1-fix2, b5d2a2e0): a sample joins the FIRST support (smallest index) whose seed is
                           within the merge distance AND whose other members all overlap the incoming proposal (``_overlaps``, the solver's own
                           function); the seeds within the merge distance are found through a grid and visited by increasing index.
Identity (graph nodes, labelled edges, groups, supports, actions, emitted set) is checked against the originals by the tests and by
scripts/p02_identity_check.py. Only local_block proposals are supported (the lot has no global pass): a global proposal raises.
"""
from __future__ import annotations

import math
from dataclasses import asdict

from babaorum.experimental.joint_reconcile import (LineProposal, ReconcileConfig, _crop_flags, _nonzero_segments, _page_baseline,
                                                   _coverage, _resample, _segments_distance, _validate_proposal)
from babaorum.experimental.joint_reconcile_selection import _overlaps  # S1-fix2: the true-overlap relation, used as it is

#: float noise only: the grid never drops a pair that the original could label
_EPS_REL = 1e-6


def _validate(proposals, config):
    config.validate()
    if not isinstance(proposals, (tuple, list)):
        raise ValueError("proposals must be a list or tuple")
    for p in proposals:
        _validate_proposal(p)
        if p.source != "local_block":
            raise ValueError("p02_fast supports local_block proposals only")
    ids = [p.proposal_id for p in proposals]
    if len(ids) != len(set(ids)):
        raise ValueError("proposal_id must be unique across both passes")
    keys = [(p.source, p.input_sha256, p.query_id) for p in proposals]
    if len(keys) != len(set(keys)):
        raise ValueError("query_id must be unique within each source/input crop")
    frame_by_page = {}
    for p in proposals:
        frame = (p.document_key, p.page_size_wh, p.page_sha256)
        if p.page_key in frame_by_page and frame_by_page[p.page_key] != frame:
            raise ValueError("proposals for one page must share document identity and page frame")
        frame_by_page[p.page_key] = frame


def _candidate_pairs(ordered, cleaned, reach):
    """Index pairs (i < j, same page) whose cleaned polylines are within ``reach``; found with a grid, checked by exact distance."""
    boxes = {}
    for i, poly in cleaned.items():
        xs, ys = [q[0] for q in poly], [q[1] for q in poly]
        boxes[i] = (min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach)
    cell = max(4.0 * reach, 1e-9)
    grid: dict[tuple[int, int], list[int]] = {}
    for i, b in boxes.items():
        for gx in range(math.floor(b[0] / cell), math.floor(b[2] / cell) + 1):
            for gy in range(math.floor(b[1] / cell), math.floor(b[3] / cell) + 1):
                grid.setdefault((gx, gy), []).append(i)
    seen, out = set(), []
    for members in grid.values():
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                i, j = (members[a], members[c]) if members[a] < members[c] else (members[c], members[a])
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if ordered[i].page_key != ordered[j].page_key:
                    continue
                bi, bj = boxes[i], boxes[j]
                if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                    continue
                pi, pj = cleaned[i], cleaned[j]
                d = min(_segments_distance(a1, a2, b1, b2) for a1, a2 in zip(pi, pi[1:]) for b1, b2 in zip(pj, pj[1:]))
                if d <= reach:
                    out.append((i, j))
    return sorted(out)


def build_graph_fast(proposals, config: ReconcileConfig = ReconcileConfig()) -> dict:
    """Same graph as build_reconciliation_graph, restricted to the pairs that can carry a label (see the module docstring)."""
    _validate(proposals, config)
    ordered = sorted(proposals, key=lambda p: (p.page_key, p.source, p.block_id or "", p.proposal_id))
    nodes, page_pts, cleaned = [], [], {}
    for idx, p in enumerate(ordered):
        pts = _page_baseline(p)
        page_pts.append(pts)
        nodes.append({"proposal": asdict(p), "baseline_page_xy": pts, "crop_flags": _crop_flags(p, pts, config)})
        clean = _nonzero_segments(pts) if pts is not None else None
        if clean is not None:
            cleaned[idx] = clean
    node_by_id = {n["proposal"]["proposal_id"]: n for n in nodes}
    h = config.normalization_height_px
    tol = config.distance_tolerance_heights * h
    step = config.sample_step_heights * h
    reach = tol * (1.0 + _EPS_REL) + 1e-9
    edges = []
    for i, j in _candidate_pairs(ordered, cleaned, reach):
        left, right = ordered[i], ordered[j]
        a, b = cleaned[i], cleaned[j]
        # ---- the original pair evaluation (joint_reconcile.build_reconciliation_graph), unchanged ----
        edge = {"source_ids": [left.proposal_id, right.proposal_id], "labels": [],
                "status": "no_candidate", "normalization_height_px": config.normalization_height_px,
                "rule": {"distance_tolerance_heights": config.distance_tolerance_heights,
                         "endpoint_tolerance_heights": config.endpoint_tolerance_heights,
                         "duplicate_coverage": config.duplicate_coverage,
                         "sample_step_heights": config.sample_step_heights}}
        ca, cb = _coverage(a, b, tol, step), _coverage(b, a, tol, step)
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
    return {"schema_version": "babaorum.experimental_reconciliation.v1", "scope": "SYNTHETIC_GEOMETRIC_CANDIDATES_ONLY", "selection": None,
            "config": asdict(config), "nodes": nodes, "edges": edges, "groups": [],
            "limitations": ["p02_fast: unlabeled pairs farther than the distance tolerance are not enumerated (lossless for solve)."]}


def build_actions_fast(graph: dict, *, step_heights: float = 0.125, merge_heights: float = 0.5) -> dict:
    """Same result as joint_reconcile_selection.build_actions; the seed search uses a grid (smallest index among neighbouring seeds)."""
    h = graph["config"]["normalization_height_px"]
    step, merge = step_heights * h, merge_heights * h
    ineligible, degenerate = [], []
    samples = []
    for node in sorted(graph["nodes"], key=lambda n: n["proposal"]["proposal_id"]):
        pid = node["proposal"]["proposal_id"]
        page_points = node["baseline_page_xy"]
        if page_points is None:
            ineligible.append(pid)
            continue
        segments = _nonzero_segments(tuple(page_points))
        if segments is None:
            degenerate.append(pid)
            continue
        for index, point in enumerate(_resample(segments, step)):
            samples.append((pid, index, point))
    geometry = {}
    for node in graph["nodes"]:
        pid = node["proposal"]["proposal_id"]
        pts = node["baseline_page_xy"]
        if pts is None:
            continue
        seg = _nonzero_segments(tuple(pts))
        if seg is not None:
            geometry[pid] = (seg, _resample(seg, step))
    overlap_cache: dict[tuple[str, str], bool] = {}

    def overlaps(p: str, q: str) -> bool:
        if (p, q) not in overlap_cache:
            overlap_cache[(p, q)] = _overlaps(geometry[p][0], geometry[p][1], geometry[q][0], h)
        return overlap_cache[(p, q)]

    cell = max(merge, 1e-12)
    seeds: list[tuple[float, float]] = []
    seed_grid: dict[tuple[int, int], list[int]] = {}
    supports: dict[str, set[str]] = {}
    support_of_proposal: dict[str, set[str]] = {}
    for pid, _, point in samples:
        gx, gy = math.floor(point[0] / cell), math.floor(point[1] / cell)
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in seed_grid.get((gx + dx, gy + dy), ()):
                    seed = seeds[k]
                    if ((point[0] - seed[0]) ** 2 + (point[1] - seed[1]) ** 2) ** 0.5 <= merge:
                        near.append(k)
        sid_index = None
        for k in sorted(near):                       # increasing index, as in the full scan
            members = supports.get(f"s{k:04d}", set())
            if all(overlaps(pid, q) for q in sorted(members) if q != pid):
                sid_index = k
                break
        if sid_index is None:
            seeds.append(point)
            sid_index = len(seeds) - 1
            seed_grid.setdefault((gx, gy), []).append(sid_index)
        sid = f"s{sid_index:04d}"
        supports.setdefault(sid, set()).add(pid)
        support_of_proposal.setdefault(pid, set()).add(sid)
    actions = []
    for node in sorted(graph["nodes"], key=lambda n: n["proposal"]["proposal_id"]):
        pid = node["proposal"]["proposal_id"]
        covered = sorted(support_of_proposal.get(pid, set()))
        actions.append({"id": f"single:{pid}", "kind": "single", "P": [pid], "O": [pid], "S": covered})
    for group in graph["groups"]:
        raise ValueError("p02_fast supports no global/local groups")
    actions.sort(key=lambda a: a["id"])
    return {"normalization_height_px": h, "step_heights": step_heights, "merge_heights": merge_heights,
            "supports": {sid: sorted(members) for sid, members in sorted(supports.items())},
            "actions": actions, "ineligible": sorted(ineligible), "degenerate": sorted(degenerate)}
