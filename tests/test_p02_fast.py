"""LOT-P02: the index-based replacements are IDENTICAL to the S1-fix originals (graph, labelled edges, supports, actions, emitted set)."""
from __future__ import annotations

import pytest

from babaorum.experimental import joint_reconcile_selection as S
from babaorum.experimental.joint_reconcile import ReconcileConfig, build_reconciliation_graph
from babaorum.p02_fast import build_actions_fast, build_graph_fast
from p02_support import proposal, synthetic_page

W1 = {"w_cov": 1.0, "kappa": 10.0}


def labelled(edges):
    return [e for e in edges if e["labels"]]


def check_identity(props, h):
    cfg = ReconcileConfig(normalization_height_px=h)
    g0, g1 = build_reconciliation_graph(props, cfg), build_graph_fast(props, cfg)
    assert g0["nodes"] == g1["nodes"] and g0["groups"] == g1["groups"] == [] and g0["config"] == g1["config"]
    assert labelled(g0["edges"]) == labelled(g1["edges"])                       # every labelled edge, identical (labels, coverages, order)
    # no_candidate edges (written by the original for every pair) are NOT compared (pJ): only edges carrying at least one label are
    b0, b1 = S.build_actions(g0), build_actions_fast(g1)
    assert b0 == b1                                                              # same supports, same actions
    s0, s1 = S.solve(b0, g0, W1), S.solve(b1, g1, W1)
    assert s0 == s1                                                              # same emitted set, same uncovered supports, same components
    return s0


@pytest.mark.parametrize("seed", range(12))
def test_fast_pipeline_equals_the_originals_on_synthetic_pages(seed):
    props = synthetic_page(seed)
    sol = check_identity(props, 20.0)
    assert sol["chosen_actions"]  # something is emitted


def test_fast_pipeline_equals_the_originals_at_the_reach_limits():
    """Parallel lines at exactly 0.5 h and just around it, and fragments abutting at 0.5 h: the grid must not lose a pair or a seed."""
    h = 20.0
    props = []
    for k, gap in enumerate((0.49, 0.5, 0.5000001, 0.51, 0.75, 0.99, 1.0, 1.01)):
        y = 100.0 + k * 200.0
        props.append(proposal(f"a{k:02d}", [(100.0, y), (400.0, y)]))
        props.append(proposal(f"b{k:02d}", [(100.0, y + gap * h), (400.0, y + gap * h)], block="z1"))
    props.append(proposal("c-left", [(1000.0, 100.0), (1300.0, 100.0)]))
    props.append(proposal("c-right", [(1300.0 + 0.5 * h, 100.0), (1600.0, 100.0)], block="z1"))   # abutting at exactly 0.5 h
    check_identity(props, h)


def test_fast_pipeline_is_independent_of_the_input_order():
    props = synthetic_page(3)
    first = check_identity(props, 20.0)
    second = check_identity(list(reversed(props)), 20.0)
    assert first["chosen_actions"] == second["chosen_actions"]


def test_global_proposals_are_refused():
    from dataclasses import replace

    p = replace(proposal("g", [(0.0, 0.0), (100.0, 0.0)]), source="global_page", block_id=None, crop_transform=None)
    with pytest.raises(ValueError):
        build_graph_fast([p], ReconcileConfig(normalization_height_px=20.0))


def test_the_accelerated_grouping_follows_the_s1fix2_rule_not_the_legacy_one():
    """On pages where the true-overlap rule changes the supports, the fast version must equal the NEW original and differ from the legacy grouping."""
    found_difference = False
    for seed in range(12):
        props = synthetic_page(seed)
        cfg = ReconcileConfig(normalization_height_px=20.0)
        g = build_graph_fast(props, cfg)
        new, legacy, fast = S.build_actions(g), S.build_actions(g, _legacy_grouping=True), build_actions_fast(g)
        assert fast == new
        found_difference = found_difference or new["supports"] != legacy["supports"]
    assert found_difference, "no synthetic page exercises the overlap test: the identity check would be blind to it"
