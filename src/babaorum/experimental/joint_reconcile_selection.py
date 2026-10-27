"""Selection-stage inputs for joint reconciliation: actions and supports (step B1).

Builds, from a graph returned by joint_reconcile.build_reconciliation_graph,
the candidate actions and the discrete geometric supports they cover. No solver,
no ground truth, no text and no CTC score is used. Synthetic CPU probe only.
"""

from __future__ import annotations

import math
import statistics

from babaorum.experimental.joint_reconcile import _distance_to_polyline, _nonzero_segments, _resample

# S1-fix2 (docs/END_TO_END_S1FIX2_LOT.txt §1, §6, §7, §8). tau is a fraction of the shorter length,
# fixed before any reading; it is not tuned on P-02. Transversal threshold: 0.5 h_norm.
REC_TAU = 0.5
REC_T_HEIGHTS = 0.5


def _oriented(points) -> list:
    """Points from the first to the last one in the increasing-x direction (§1)."""
    pts = [(float(x), float(y)) for x, y in points]
    if pts[-1][0] < pts[0][0]:
        pts.reverse()
    return pts


def _unit(v):
    n = math.hypot(v[0], v[1])
    return None if n == 0 else (v[0] / n, v[1] / n)


def _overlap_direction(a_segments, b_segments):
    """Mean of the two unit chord vectors, normalised; if that sum is zero, the direction of A (§1).
    None when a baseline has a zero chord (closed loop): no overlap is then claimed."""
    a, b = _oriented(a_segments), _oriented(b_segments)
    ua = _unit((a[-1][0] - a[0][0], a[-1][1] - a[0][1]))
    ub = _unit((b[-1][0] - b[0][0], b[-1][1] - b[0][1]))
    if ua is None or ub is None:
        return None
    return _unit((ua[0] + ub[0], ua[1] + ub[1])) or ua


def _overlaps(p_segments, p_samples, q_segments, h: float, tau: float = REC_TAU) -> bool:
    """§1/§6: does the incoming proposal p overlap the member q? Both baselines in page coordinates.
    o = longitudinal overlap on d; t = median, over the samples of p whose projection lies in the common
    interval, of their distance to the polyline of q (+inf if none). Overlap iff o > tau*min(L_p, L_q)
    and t <= 0.5 h. Contacts and light encroachments (o <= tau*min) are not overlaps."""
    d = _overlap_direction(p_segments, q_segments)
    if d is None:
        return False

    def extent(points):
        proj = [x * d[0] + y * d[1] for x, y in points]
        return min(proj), max(proj)

    p0, p1 = extent(p_segments)
    q0, q1 = extent(q_segments)
    lo, hi = max(p0, q0), min(p1, q1)
    o = hi - lo
    if not (o > tau * min(p1 - p0, q1 - q0)):
        return False
    dists = sorted(_distance_to_polyline(pt, tuple(q_segments)) for pt in p_samples
                   if lo <= pt[0] * d[0] + pt[1] * d[1] <= hi)
    if not dists:
        return False
    return statistics.median(dists) <= REC_T_HEIGHTS * h


def build_actions(graph: dict, *, step_heights: float = 0.125, merge_heights: float = 0.5,
                  _legacy_grouping: bool = False) -> dict:
    """_legacy_grouping=True restores the pre-S1-fix2 grouping (test use only, to reproduce the defect)."""
    h = graph["config"]["normalization_height_px"]
    step, merge = step_heights * h, merge_heights * h

    ineligible, degenerate = [], []
    samples = []  # (proposal_id, index, (x, y)) over all eligible proposals, sorted
    geometry = {}  # pid -> (segments, resampled samples)
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
        resampled = _resample(segments, step)
        geometry[pid] = (segments, resampled)
        for index, point in enumerate(resampled):
            samples.append((pid, index, point))

    # Relation of S1-fix2, computed once per ordered pair (incoming p, member q), before any use.
    overlap_cache: dict[tuple[str, str], bool] = {}

    def overlaps(p: str, q: str) -> bool:
        if (p, q) not in overlap_cache:
            overlap_cache[(p, q)] = _overlaps(geometry[p][0], geometry[p][1], geometry[q][0], h)
        return overlap_cache[(p, q)]

    # Non-transitive seed grouping: each sample joins the FIRST existing support whose
    # seed (its first point) lies within merge distance AND whose other members all overlap it
    # (S1-fix2); otherwise it tries the next support, and if none fits it seeds a new one.
    # Samples are visited in (proposal_id, index) order, so numbering is deterministic.
    seeds: list[tuple[float, float]] = []
    supports: dict[str, set[str]] = {}
    support_of_proposal: dict[str, set[str]] = {}
    for pid, _, point in samples:
        sid = None
        for k, seed in enumerate(seeds):
            if ((point[0] - seed[0]) ** 2 + (point[1] - seed[1]) ** 2) ** 0.5 <= merge:
                members = supports.get(f"s{k:04d}", set())
                if _legacy_grouping or all(overlaps(pid, q) for q in sorted(members) if q != pid):
                    sid = f"s{k:04d}"
                    break
        if sid is None:
            seeds.append(point)
            sid = f"s{len(seeds) - 1:04d}"
        supports.setdefault(sid, set()).add(pid)
        support_of_proposal.setdefault(pid, set()).add(sid)

    actions = []
    for node in sorted(graph["nodes"], key=lambda n: n["proposal"]["proposal_id"]):
        pid = node["proposal"]["proposal_id"]
        covered = sorted(support_of_proposal.get(pid, set()))
        actions.append({"id": f"single:{pid}", "kind": "single", "P": [pid], "O": [pid], "S": covered})
    for group in graph["groups"]:
        global_id = group["global_proposal_id"]
        locals_ = sorted(group["local_proposal_ids"])
        covered = sorted(set().union(*(support_of_proposal.get(lp, set()) for lp in locals_)))
        actions.append({"id": f"group:{global_id}", "kind": "group",
                        "P": sorted([global_id, *locals_]), "O": locals_, "S": covered})
    actions.sort(key=lambda a: a["id"])

    return {
        "normalization_height_px": h,
        "step_heights": step_heights,
        "merge_heights": merge_heights,
        "supports": {sid: sorted(members) for sid, members in sorted(supports.items())},
        "actions": actions,
        "ineligible": sorted(ineligible),
        "degenerate": sorted(degenerate),
    }


# Test weight sets of docs/END_TO_END_RECONCILE_SELECTION.txt §7 (spec v3, sha 03de8d9f…).
# Missing keys count as 0. Not tuned on data.
WEIGHTS = {
    "W1": {"w_cov": 1.0, "kappa": 10.0},
    "W2": {"w_cov": 1.0, "kappa": 10.0, "w_glob": 0.1},
    "W3": {"w_cov": 1.0, "kappa": 0.5, "w_trunc": 10.0},
    "W4": {"w_cov": 1.0, "kappa": 2.0, "w_trunc": 0.5, "w_pad": 0.5, "w_red": 0.5,
           "w_ov": 0.5, "w_glob": 0.0},
}

_TIE_TOL = 1e-12
_CONFLICT_LABELS = ("reversed_geometry", "direction_conflict")
_MAX_TIE_COMBINATIONS = 100000
_INTERACTION_LABELS = frozenset(_CONFLICT_LABELS) | {"partial_overlap_candidate"}


def _components(actions: list[dict], edge_labels: dict) -> list[list[int]]:
    """Connected components over every interaction that can constrain a joint choice:
    a shared proposal (P), a shared support (S), or a graph edge between outputs (O) of two actions
    carrying a hard conflict label or a partial-overlap label. The partition does not depend on
    the soft weights, so a conflict is never left outside its component."""
    parent = list(range(len(actions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    owner: dict[tuple[str, str], int] = {}
    for i, act in enumerate(actions):
        for key in [("P", p) for p in act["P"]] + [("S", s) for s in act["S"]]:
            if key in owner:
                union(i, owner[key])
            else:
                owner[key] = i

    actions_by_output: dict[str, list[int]] = {}
    for i, act in enumerate(actions):
        for pid in act["O"]:
            actions_by_output.setdefault(pid, []).append(i)
    for pair, labels in edge_labels.items():
        if not (labels & _INTERACTION_LABELS):
            continue
        ends = sorted(pair)
        if len(ends) == 1:
            ends = ends * 2
        for a in actions_by_output.get(ends[0], []):
            for b in actions_by_output.get(ends[1], []):
                union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(actions)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda members: members[0])


def _assert_no_hard_conflict(chosen_actions: list[dict], edge_labels: dict) -> None:
    """Global guard: no two chosen actions may carry a hard conflict between their outputs."""
    for i, a in enumerate(chosen_actions):
        for b in chosen_actions[i + 1:]:
            for x in a["O"]:
                for y in b["O"]:
                    if edge_labels.get(frozenset((x, y)), set()) & set(_CONFLICT_LABELS):
                        raise RuntimeError(f"selection contains a hard conflict: {a['id']} / {b['id']}")


def solve(built: dict, graph: dict, weights: dict, *, max_actions: int = 18) -> dict:
    """Exact enumeration per connected component (CPU, pure Python).

    Tie-break: among feasible selections, the lowest cost (equal within _TIE_TOL); among those,
    the smallest sorted tuple of chosen action IDs. Components larger than max_actions are
    abstained and reported, not solved.
    """
    from itertools import product

    actions = built["actions"]
    nodes = {n["proposal"]["proposal_id"]: n for n in graph["nodes"]}
    group_red = {g["global_proposal_id"]: g["redundant_coverage_sum_minus_union"] for g in graph["groups"]}
    w = {k: float(weights.get(k, 0.0)) for k in
         ("w_cov", "kappa", "w_trunc", "w_pad", "w_red", "w_ov", "w_glob")}

    edge_labels: dict[frozenset, set[str]] = {}
    for e in graph["edges"]:
        edge_labels[frozenset(e["source_ids"])] = set(e["labels"])

    def action_cost(act):
        cost = -w["w_cov"] * len(act["S"])
        for pid in act["O"]:
            flags = nodes[pid]["crop_flags"]
            cost += w["w_trunc"] * bool(flags["may_be_truncated_by_crop_band"])
            cost += w["w_pad"] * bool(flags["has_points_in_padding_or_outside_content"])
            cost += w["w_glob"] * (nodes[pid]["proposal"]["source"] == "global_page")
        if act["kind"] == "group":
            global_id = act["id"].split(":", 1)[1]
            cost += w["w_red"] * group_red[global_id]
        return cost

    def pair_flags(a, b):
        """(hard conflict, partial overlap) between the outputs of two actions."""
        conflict = partial = False
        for x in a["O"]:
            for y in b["O"]:
                labels = edge_labels.get(frozenset((x, y)), set())
                conflict = conflict or any(lbl in labels for lbl in _CONFLICT_LABELS)
                partial = partial or "partial_overlap_candidate" in labels
        return conflict, partial

    comps = _components(actions, edge_labels)
    sizes, abstained = [], []
    comp_infos = []  # per solved component: (members, comp_supports, optima) with optima = [(ids, x)]
    for members in comps:
        comp = [actions[i] for i in members]
        sizes.append(len(comp))
        comp_supports = sorted({s for a in comp for s in a["S"]})
        if len(comp) > max_actions:
            abstained.append({"status": "abstained_component_too_large",
                              "actions": [a["id"] for a in comp],
                              "proposals": sorted({p for a in comp for p in a["P"]})})
            continue
        costs = [action_cost(a) for a in comp]
        pairs = {(i, j): pair_flags(comp[i], comp[j]) for i in range(len(comp))
                 for j in range(i + 1, len(comp))}
        candidates = []  # (cost, ids, x)
        for x in product((0, 1), repeat=len(comp)):
            pick = [i for i, v in enumerate(x) if v]
            seen_p, seen_s, ok = set(), set(), True
            for i in pick:
                if seen_p & set(comp[i]["P"]) or seen_s & set(comp[i]["S"]):
                    ok = False
                    break
                seen_p |= set(comp[i]["P"])
                seen_s |= set(comp[i]["S"])
            if not ok:
                continue
            if any(pairs[(pick[a_i], pick[b_i])][0]
                   for a_i in range(len(pick)) for b_i in range(a_i + 1, len(pick))):
                continue
            cost = sum(costs[i] for i in pick)
            cost += w["kappa"] * len(set(comp_supports) - seen_s)
            cost += w["w_ov"] * sum(1 for a_i in range(len(pick)) for b_i in range(a_i + 1, len(pick))
                                    if pairs[(pick[a_i], pick[b_i])][1])
            candidates.append((cost, tuple(sorted(comp[i]["id"] for i in pick)), x))
        best_cost = min(c for c, _, _ in candidates)
        optima = [(ids, x) for c, ids, x in candidates if abs(c - best_cost) <= _TIE_TOL]
        comp_infos.append((members, comp_supports, optima))

    # Components are independent: the global optimum is the union of per-component optima.
    # Ties are broken on the global sorted tuple of chosen IDs (not per component).
    combos = 1
    for _, _, optima in comp_infos:
        combos *= len(optima)
    if combos > _MAX_TIE_COMBINATIONS:
        raise RuntimeError(f"tie set too large for the global ID tie-break ({combos} combinations)")
    best_union, best_pick = None, None
    for choice in product(*[range(len(optima)) for _, _, optima in comp_infos]):
        union = tuple(sorted(i for k, (_, _, optima) in zip(choice, comp_infos) for i in optima[k][0]))
        if best_union is None or union < best_union:
            best_union, best_pick = union, choice

    chosen_all, uncovered = [], []
    for k, (members, comp_supports, optima) in zip(best_pick, comp_infos):
        ids, x = optima[k]
        comp = [actions[i] for i in members]
        chosen_now = [comp[i] for i, v in enumerate(x) if v]
        chosen_all.extend(chosen_now)
        covered = {s for a in chosen_now for s in a["S"]}
        uncovered.extend(s for s in comp_supports if s not in covered)
    for ab in abstained:
        uncovered.extend(sorted({s for i in range(len(actions)) if actions[i]["id"] in ab["actions"]
                                 for s in actions[i]["S"]}))

    _assert_no_hard_conflict(chosen_all, edge_labels)
    total_actions = len(actions)
    abstained_actions = sum(len(a["actions"]) for a in abstained)
    return {
        "chosen_actions": sorted(a["id"] for a in chosen_all),
        "uncovered_supports": sorted(uncovered),
        "component_sizes": sizes,
        "abstained_components": abstained,
        "abstained_action_count": abstained_actions,
        "abstention_rate": (abstained_actions / total_actions) if total_actions else 0.0,
        "tie_rule": "lowest cost (equal within _TIE_TOL), then smallest sorted tuple of chosen action IDs",
        "weights": weights,
    }


STATUSES = ("emitted", "consumed_not_emitted", "abstained_component_too_large",
            "ineligible_no_geometry", "degenerate", "not_selected")


def statuses(built: dict, graph: dict, solution: dict) -> dict:
    """Exactly one status per proposal of the graph, plus counts per status.

    Precedence follows the spec order: emitted, consumed_not_emitted, abstained,
    ineligible, degenerate, otherwise not_selected.
    """
    chosen = set(solution["chosen_actions"])
    by_id = {a["id"]: a for a in built["actions"]}
    emitted, consumed = set(), set()
    for aid in chosen:
        emitted.update(by_id[aid]["O"])
        consumed.update(by_id[aid]["P"])
    abstained = {pid for comp in solution["abstained_components"] for pid in comp["proposals"]}
    ineligible, degenerate = set(built["ineligible"]), set(built["degenerate"])

    result = {}
    for node in graph["nodes"]:
        pid = node["proposal"]["proposal_id"]
        if pid in emitted:
            status = "emitted"
        elif pid in consumed:
            status = "consumed_not_emitted"
        elif pid in abstained:
            status = "abstained_component_too_large"
        elif pid in ineligible:
            status = "ineligible_no_geometry"
        elif pid in degenerate:
            status = "degenerate"
        else:
            status = "not_selected"
        result[pid] = status
    counts = {s: sum(1 for v in result.values() if v == s) for s in STATUSES}
    return {"statuses": result, "counts": counts}
