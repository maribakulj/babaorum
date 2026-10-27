"""S1-fix2 toys (docs/END_TO_END_S1FIX2_LOT.txt §2, §7, §8, §9). Synthetic, CPU, no data.

Two proposals share a support only if every other member overlaps the newcomer (§1, §6). Hard conflicts
(reversed_geometry, direction_conflict) are never allowed; a duplicate or partial-overlap label is admitted
and is NOT a hard conflict (pK). C2 is unchanged: a real duplicate is kept out by the shared support alone.
"""

from __future__ import annotations

import hashlib

from babaorum.experimental.joint_reconcile import LineProposal, ReconcileConfig, build_reconciliation_graph
from babaorum.experimental.joint_reconcile_selection import WEIGHTS, build_actions, solve

SHA = hashlib.sha256(b"synthetic").hexdigest()
H = 32.0
CFG = ReconcileConfig(normalization_height_px=H)
HARD = {"reversed_geometry", "direction_conflict"}


def prop(pid, source, points, *, block=None, crop=None):
    return LineProposal(
        proposal_id=pid, source=source, page_key="fixture:p", document_key="fixture:d",
        page_size_wh=(600.0, 100.0), page_sha256=SHA, query_id=pid, input_sha256=SHA,
        checkpoint_sha256=None, geometry_status="available" if points is not None else "missing",
        baseline_xy=tuple(tuple(xy) for xy in points) if points is not None else None,
        band_normal_negative_px=None, band_normal_positive_px=None,
        orientation="ltr", text_status="available", text="", token_ids=None,
        ctc_status="empty", block_id=block, crop_transform=crop,
    )


def line(pid, pts):
    return prop(pid, "global_page", pts)


def graph_of(props):
    return build_reconciliation_graph(props, CFG)


def shared(built, a, b):
    return [sid for sid, members in built["supports"].items() if a in members and b in members]


def labels_between(graph, a, b):
    out = set()
    for e in graph["edges"]:
        if set(e["source_ids"]) == {a, b}:
            out |= set(e["labels"])
    return out


def emitted(graph, built, weights="W1"):
    sol = solve(built, graph, WEIGHTS[weights])
    return set(sol["chosen_actions"])


def seg(x0, x1, y):
    return [(x0, y), (x1, y)]


# ---- (a) bout à bout, écart de 10 px, graphe construit par build_reconciliation_graph (§7)

def test_a_end_to_end_gap_10px_no_labeled_edge_legacy_shares_and_new_emits_both():
    # A starts at x = 1 (not 0): with the seed grid of A from x = 0 the last seed is at x = 192, 18 px
    # from B, so the old grouping could not share a support. The gap of 10 px is unchanged.
    A, B = line("g:A", seg(1, 200, 20)), line("g:B", seg(210, 400, 20))
    g = graph_of([A, B])
    # (1) aucune arête étiquetée entre A et B
    assert labels_between(g, "g:A", "g:B") == set()
    # (2) ancien regroupement : partage d'un support, et solve n'émet qu'une ligne
    old = build_actions(g, _legacy_grouping=True)
    assert shared(old, "g:A", "g:B")
    assert len({a for a in emitted(g, old) if a in {"single:g:A", "single:g:B"}}) == 1
    # (3) S1-fix2 : aucun support partagé, les deux lignes émises
    new = build_actions(g)
    assert shared(new, "g:A", "g:B") == []
    assert {"single:g:A", "single:g:B"} <= emitted(g, new)


def test_a_both_solvers_emit_both_with_no_shared_support():
    A, B = line("g:A", seg(0, 200, 20)), line("g:B", seg(210, 400, 20))
    g = graph_of([A, B])
    built = build_actions(g)
    for w in ("W1", "W4"):
        assert {"single:g:A", "single:g:B"} <= emitted(g, built, w)


# ---- (a') contact léger : empiètement de 6 px ; (a'') limite : 0,2 h = 6,4 px (pA)

def test_a_prime_light_encroachment_6px_emits_both():
    A, B = line("g:A", seg(0, 200, 20)), line("g:B", seg(194, 400, 20))
    g = graph_of([A, B])
    built = build_actions(g)
    assert shared(built, "g:A", "g:B") == []
    assert {"single:g:A", "single:g:B"} <= emitted(g, built)


def test_a_double_prime_limit_contact_0_2h_emits_both():
    A, B = line("g:A", seg(0, 200, 20)), line("g:B", seg(193.6, 400, 20))
    g = graph_of([A, B])
    built = build_actions(g)
    assert shared(built, "g:A", "g:B") == []
    assert {"single:g:A", "single:g:B"} <= emitted(g, built)


# ---- (b) vrai doublon décalé de 5 px ; étiquettes dures interdites, duplicate admis (pK, §8)

def test_b_true_shifted_duplicate_one_line_by_shared_support_only():
    A, B = line("g:A", seg(0, 300, 20)), line("g:B", seg(60, 340, 25))
    g = graph_of([A, B])
    built = build_actions(g)
    assert shared(built, "g:A", "g:B")
    assert not (labels_between(g, "g:A", "g:B") & HARD)
    chosen = emitted(g, built)
    assert len({"single:g:A", "single:g:B"} & chosen) == 1


# ---- (b') doublon court contenu (0,43 h)

def test_b_prime_short_contained_duplicate_one_line():
    A, B = line("g:A", seg(0, 300, 20)), line("g:B", seg(100, 114, 20))
    g = graph_of([A, B])
    built = build_actions(g)
    assert shared(built, "g:A", "g:B")
    assert len({"single:g:A", "single:g:B"} & emitted(g, built)) == 1


# ---- (c) rangées voisines, 32 px d'écart vertical

def test_c_neighbouring_rows_32px_apart_share_nothing():
    A, B = line("g:A", seg(0, 200, 20)), line("g:B", seg(0, 200, 52))
    g = graph_of([A, B])
    built = build_actions(g)
    assert shared(built, "g:A", "g:B") == []
    assert {"single:g:A", "single:g:B"} <= emitted(g, built)


# ---- (d) gouttière : 15 px horizontal, 3 px vertical

def test_d_gutter_no_shared_support_and_both_emitted():
    A, B = line("g:A", seg(0, 200, 20)), line("g:B", seg(215, 400, 23))
    g = graph_of([A, B])
    built = build_actions(g)
    assert shared(built, "g:A", "g:B") == []
    assert {"single:g:A", "single:g:B"} <= emitted(g, built)


# ---- (f) déterminisme : répétition et permutation des nœuds

def test_f_deterministic_and_invariant_to_node_order():
    props = [line("g:A", seg(0, 300, 20)), line("g:B", seg(60, 340, 25)),
             line("g:C", seg(0, 200, 52)), line("g:D", seg(210, 400, 20))]
    g = graph_of(props)
    first, again = build_actions(g), build_actions(g)
    assert first["supports"] == again["supports"]
    shuffled = dict(g, nodes=list(reversed(g["nodes"])))
    other = build_actions(shuffled)
    assert other["supports"] == first["supports"]
    assert emitted(g, first) == emitted(shuffled, other)


# ---- §9(ii) : ancien décor (19 fragments de 12 px, pas de 10 px) en test nouveau

def test_s9_old_decor_is_19_singletons_with_s1fix2_and_one_component_with_legacy():
    props = [line(f"g:c{i:02d}", seg(10 * i, 10 * i + 12, 20)) for i in range(19)]
    g = graph_of(props)
    new = build_actions(g)
    res_new = solve(new, g, WEIGHTS["W1"], max_actions=18)
    assert res_new["component_sizes"] == [1] * 19
    assert res_new["abstained_components"] == []
    assert len(res_new["chosen_actions"]) == 19
    old = build_actions(g, _legacy_grouping=True)
    res_old = solve(old, g, WEIGHTS["W1"], max_actions=18)
    assert res_old["component_sizes"] == [19]
    assert res_old["abstained_components"] != []
