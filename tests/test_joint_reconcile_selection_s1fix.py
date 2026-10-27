"""S1-fix checks for joint_reconcile_selection.solve (synthetic, CPU, no data).

- counterexample: two actions in hard conflict with no shared proposal or support are in
  different selection components in the old code; the new solver must not choose both;
- comparison with an independent monolithic enumeration over random instances (fixed seed) in
  which every instance has a hard conflict and a partial overlap between actions sharing no P or S;
- invariance under permutation of actions, nodes and edges: same chosen IDs;
- global guard against a hard conflict between chosen actions.
"""

from __future__ import annotations

import random
from itertools import combinations, product

import pytest

from babaorum.experimental.joint_reconcile_selection import (
    _assert_no_hard_conflict, _TIE_TOL, solve,
)


def _action(aid, P, S, O=None):
    return {"id": aid, "kind": "single", "P": sorted(P), "O": sorted(O if O is not None else P), "S": sorted(S)}


def _graph(proposal_ids, edges, flags=None):
    nodes = [{"proposal": {"proposal_id": pid, "source": "local_block"},
              "crop_flags": (flags or {}).get(pid, {"may_be_truncated_by_crop_band": False,
                                                   "has_points_in_padding_or_outside_content": False})}
             for pid in proposal_ids]
    return {"nodes": nodes, "edges": [{"source_ids": list(e), "labels": list(lbl)} for e, lbl in edges],
            "groups": [], "config": {"normalization_height_px": 20.0}}


def _oracle(built, graph, weights):
    """Independent monolithic enumeration of all subsets; same feasibility and cost; ID tie-break."""
    actions = sorted(built["actions"], key=lambda a: a["id"])
    labels = {frozenset(e["source_ids"]): set(e["labels"]) for e in graph["edges"]}
    node = {n["proposal"]["proposal_id"]: n for n in graph["nodes"]}
    w = {k: float(weights.get(k, 0.0)) for k in ("w_cov", "kappa", "w_trunc", "w_pad", "w_ov", "w_glob")}
    supports = {s for a in actions for s in a["S"]}
    best = None
    for bits in product((0, 1), repeat=len(actions)):
        chosen = [a for a, b in zip(actions, bits) if b]
        ps = [p for a in chosen for p in a["P"]]
        ss = [s for a in chosen for s in a["S"]]
        if len(ps) != len(set(ps)) or len(ss) != len(set(ss)):
            continue
        cost, feasible = 0.0, True
        for a, b in combinations(chosen, 2):
            ls = set()
            for x in a["O"]:
                for y in b["O"]:
                    ls |= labels.get(frozenset((x, y)), set())
            if ls & {"reversed_geometry", "direction_conflict"}:
                feasible = False
                break
            if "partial_overlap_candidate" in ls:
                cost += w["w_ov"]
        if not feasible:
            continue
        cost += w["kappa"] * len(supports - set(ss))
        for a in chosen:
            cost -= w["w_cov"] * len(a["S"])
            for pid in a["O"]:
                f = node[pid]["crop_flags"]
                cost += w["w_trunc"] * bool(f["may_be_truncated_by_crop_band"])
                cost += w["w_pad"] * bool(f["has_points_in_padding_or_outside_content"])
                cost += w["w_glob"] * (node[pid]["proposal"]["source"] == "global_page")
        ids = tuple(sorted(a["id"] for a in chosen))
        if best is None or cost < best[0] - _TIE_TOL or (abs(cost - best[0]) <= _TIE_TOL and ids < best[1]):
            best = (cost, ids)
    return best


def test_counterexample_hard_conflict_across_components_is_not_both_chosen():
    # B and C have no shared proposal or support, so the old per-component solver chose both.
    built = {"actions": [_action("single:B", ["pB"], ["sB"]), _action("single:C", ["pC"], ["sC"])],
             "ineligible": [], "degenerate": []}
    graph = _graph(["pB", "pC"], [(("pB", "pC"), ("direction_conflict",))])
    result = solve(built, graph, {"w_cov": 1, "kappa": 10})
    assert len(result["chosen_actions"]) == 1
    assert result["chosen_actions"] == ["single:B"]  # equal cost: smallest sorted ID tuple wins
    assert result["component_sizes"] == [2]


def test_counterexample_with_covering_global_action_keeps_the_optimum():
    built = {"actions": [_action("single:A", ["pA"], ["sB", "sC"]), _action("single:B", ["pB"], ["sB"]),
                         _action("single:C", ["pC"], ["sC"])], "ineligible": [], "degenerate": []}
    graph = _graph(["pA", "pB", "pC"], [(("pB", "pC"), ("direction_conflict",))])
    weights = {"w_cov": 1, "kappa": 10}
    result = solve(built, graph, weights)
    assert result["chosen_actions"] == ["single:A"]
    assert _oracle(built, graph, weights) == (-2.0, ("single:A",))


def test_partial_overlap_between_disjoint_actions_joins_components_even_with_zero_kappa():
    built = {"actions": [_action("single:p0", ["p0"], ["s0"]), _action("single:p1", ["p1"], ["s1"])],
             "ineligible": [], "degenerate": []}
    graph = _graph(["p0", "p1"], [(("p0", "p1"), ("partial_overlap_candidate",))])
    result = solve(built, graph, {"w_cov": 1, "kappa": 0, "w_ov": 3})
    assert result["component_sizes"] == [2] and len(result["chosen_actions"]) == 1


def _random_instance(rng, n):
    """n actions on distinct proposals; at least one hard conflict and one partial overlap between
    two actions that share no proposal and no support; w_ov > 0 (required by the protocol)."""
    pids = [f"p{i}" for i in range(n)]
    for _ in range(200):
        supports = {i: set(f"s{rng.randrange(2 * n)}" for _ in range(rng.randint(1, 3))) for i in range(n)}
        disjoint = [(i, j) for i, j in combinations(range(n), 2) if not (supports[i] & supports[j])]
        if len(disjoint) >= 2:
            break
    else:
        supports = {i: {f"s{i}"} for i in range(n)}
        disjoint = list(combinations(range(n), 2))
    rng.shuffle(disjoint)
    if len(disjoint) >= 2:
        conflict_pair, partial_pair = disjoint[0], disjoint[1]
        edges = [((pids[conflict_pair[0]], pids[conflict_pair[1]]), ("direction_conflict",)),
                 ((pids[partial_pair[0]], pids[partial_pair[1]]), ("partial_overlap_candidate",))]
        used = {conflict_pair, partial_pair}
    else:  # a single disjoint pair: both labels on the same edge
        pair = disjoint[0]
        edges = [((pids[pair[0]], pids[pair[1]]), ("direction_conflict", "partial_overlap_candidate"))]
        used = {pair}
    for i, j in combinations(range(n), 2):
        if (i, j) in used or rng.random() > 0.15:
            continue
        edges.append(((pids[i], pids[j]), ("partial_overlap_candidate",)))
    built = {"actions": [_action(f"single:{pids[i]}", [pids[i]], sorted(supports[i])) for i in range(n)],
             "ineligible": [], "degenerate": []}
    weights = {"w_cov": rng.randint(1, 3), "kappa": rng.randint(0, 5), "w_trunc": 0,
               "w_pad": 0, "w_ov": rng.randint(1, 4), "w_glob": 0}
    return built, _graph(pids, edges), weights


def test_matches_monolithic_oracle_on_200_random_instances_with_chosen_ids():
    rng = random.Random(20261015)
    cases = 0
    for _ in range(200):
        n = rng.randint(2, 10)
        built, graph, weights = _random_instance(rng, n)
        result = solve(built, graph, weights)
        _, oracle_ids = _oracle(built, graph, weights)
        chosen = tuple(result["chosen_actions"])
        labels = {frozenset(e["source_ids"]): set(e["labels"]) for e in graph["edges"]}
        chosen_actions = [a for a in built["actions"] if a["id"] in chosen]
        _assert_no_hard_conflict(chosen_actions, labels)  # feasible
        # cost of the solver's selection, computed with the oracle's cost function
        sub = {"actions": chosen_actions, "ineligible": [], "degenerate": []}
        sel_ids = _oracle(sub, graph, weights)[1] if chosen_actions else ()
        assert chosen == oracle_ids
        assert sel_ids == chosen
        cases += 1
    assert cases == 200


def test_result_is_invariant_under_permutation_of_actions_nodes_and_edges():
    rng = random.Random(7)
    for _ in range(25):
        n = rng.randint(3, 8)
        built, graph, weights = _random_instance(rng, n)
        base = solve(built, graph, weights)["chosen_actions"]
        shuffled_built = {"actions": list(reversed(built["actions"])), "ineligible": [], "degenerate": []}
        shuffled_graph = dict(graph, nodes=list(reversed(graph["nodes"])), edges=list(reversed(graph["edges"])))
        assert solve(shuffled_built, shuffled_graph, weights)["chosen_actions"] == base


def test_global_guard_raises_on_a_hard_conflict_between_chosen_actions():
    a = _action("single:x", ["px"], ["sx"])
    b = _action("single:y", ["py"], ["sy"])
    graph = _graph(["px", "py"], [(("px", "py"), ("reversed_geometry",))])
    labels = {frozenset(e["source_ids"]): set(e["labels"]) for e in graph["edges"]}
    with pytest.raises(RuntimeError, match="hard conflict"):
        _assert_no_hard_conflict([a, b], labels)


def test_abstention_rate_is_reported_and_max_actions_is_unchanged():
    actions = [_action(f"single:c{i:02d}", [f"c{i}"], [f"s{i}", f"s{i + 1}"]) for i in range(19)]
    built = {"actions": actions, "ineligible": [], "degenerate": []}
    graph = _graph([f"c{i}" for i in range(19)], [((f"c{i}", f"c{i + 1}"), ("partial_overlap_candidate",))
                                                   for i in range(18)])
    result = solve(built, graph, {"w_cov": 1, "kappa": 10, "w_ov": 1})
    assert result["component_sizes"] == [19]
    assert result["abstained_action_count"] == 19
    assert result["abstention_rate"] == pytest.approx(1.0)
    assert result["chosen_actions"] == []
