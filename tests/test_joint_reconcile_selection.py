import hashlib

from babaorum.experimental.joint_reconcile import (
    CropTransform, LineProposal, build_reconciliation_graph,
)
from babaorum.experimental.joint_reconcile_selection import WEIGHTS, build_actions, solve, statuses


SHA = hashlib.sha256(b"synthetic").hexdigest()


def proposal(pid, source, points, *, block=None, crop=None):
    return LineProposal(
        proposal_id=pid, source=source, page_key="fixture:p", document_key="fixture:d",
        page_size_wh=(200.0, 100.0), page_sha256=SHA, query_id=pid, input_sha256=SHA,
        checkpoint_sha256=None, geometry_status="available" if points is not None else "missing",
        baseline_xy=tuple(tuple(xy) for xy in points) if points is not None else None,
        band_normal_negative_px=None, band_normal_positive_px=None,
        orientation="ltr", text_status="available", text="", token_ids=None,
        ctc_status="empty", block_id=block, crop_transform=crop,
    )


def local(pid, block, points):
    crop = CropTransform((0.0, 0.0, 200.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    return proposal(pid, "local_block", points, block=block, crop=crop)


def action(actions, aid):
    return next(a for a in actions["actions"] if a["id"] == aid)


def test_identical_duplicates_share_all_supports_and_get_two_single_actions():
    a = proposal("g:q0", "global_page", [(10, 20), (90, 20)])
    b = local("l:block1:q0", "block1", [(10, 20), (90, 20)])
    actions = build_actions(build_reconciliation_graph([a, b]))
    assert actions["supports"]
    for members in actions["supports"].values():
        assert members == ["g:q0", "l:block1:q0"]
    singles = [x for x in actions["actions"] if x["kind"] == "single"]
    assert [x["id"] for x in singles] == ["single:g:q0", "single:l:block1:q0"]
    assert action(actions, "single:g:q0")["S"] == action(actions, "single:l:block1:q0")["S"]
    assert not [x for x in actions["actions"] if x["kind"] == "group"]


def test_group_action_consumes_global_and_locals_and_emits_only_locals():
    global_line = proposal("g:long", "global_page", [(10, 20), (60, 35), (110, 20)])
    left = local("l:a", "a", [(10, 20), (60, 35)])
    right = local("l:b", "b", [(60, 35), (110, 20)])
    actions = build_actions(build_reconciliation_graph([global_line, left, right]))
    group = action(actions, "group:g:long")
    assert group["P"] == ["g:long", "l:a", "l:b"]
    assert group["O"] == ["l:a", "l:b"]
    expected_union = sorted(sid for sid, members in actions["supports"].items()
                            if "l:a" in members or "l:b" in members)
    assert group["S"] == expected_union
    assert len(group["S"]) > 0


def test_global_crossing_two_columns_keeps_local_supports_disjoint():
    wide = proposal("g:wide", "global_page", [(10, 20), (100, 20), (190, 20)])
    left = local("l:a", "a", [(10, 20), (90, 20)])
    right = local("l:b", "b", [(110, 20), (190, 20)])
    actions = build_actions(build_reconciliation_graph([wide, left, right]))
    sup = actions["supports"]
    a_sids = {sid for sid, members in sup.items() if "l:a" in members}
    b_sids = {sid for sid, members in sup.items() if "l:b" in members}
    g_sids = {sid for sid, members in sup.items() if "g:wide" in members}
    assert a_sids and b_sids
    assert a_sids.isdisjoint(b_sids)
    assert g_sids >= (a_sids | b_sids)


def test_output_is_invariant_to_node_and_group_order():
    wide = proposal("g:wide", "global_page", [(10, 20), (100, 20), (190, 20)])
    left = local("l:a", "a", [(10, 20), (90, 20)])
    right = local("l:b", "b", [(110, 20), (190, 20)])
    graph = build_reconciliation_graph([wide, left, right])
    shuffled = dict(graph, nodes=list(reversed(graph["nodes"])),
                    groups=list(reversed(graph["groups"])))
    assert build_actions(shuffled) == build_actions(graph)


def test_w1_keeps_exactly_one_of_two_identical_duplicates():
    a = proposal("g:q0", "global_page", [(10, 20), (90, 20)])
    b = local("l:block1:q0", "block1", [(10, 20), (90, 20)])
    graph = build_reconciliation_graph([a, b])
    result = solve(build_actions(graph), graph, WEIGHTS["W1"])
    assert len(result["chosen_actions"]) == 1
    assert result["uncovered_supports"] == []


def test_w2_prefers_local_group_over_global_single_at_equal_coverage():
    global_line = proposal("g:long", "global_page", [(10, 20), (60, 35), (110, 20)])
    left = local("l:a", "a", [(10, 20), (60, 35)])
    right = local("l:b", "b", [(60, 35), (110, 20)])
    graph = build_reconciliation_graph([global_line, left, right])
    result = solve(build_actions(graph), graph, WEIGHTS["W2"])
    assert result["chosen_actions"] == ["group:g:long"]


def test_oversized_component_is_abstained_not_solved():
    props = [proposal(f"g:c{i:02d}", "global_page", [(2 * i, 20), (2 * i + 100, 20)])
             for i in range(19)]
    graph = build_reconciliation_graph(props)
    result = solve(build_actions(graph), graph, WEIGHTS["W1"], max_actions=18)
    assert result["component_sizes"] == [19]
    assert result["chosen_actions"] == []
    (abstained,) = result["abstained_components"]
    assert abstained["status"] == "abstained_component_too_large"
    assert len(abstained["proposals"]) == 19


def test_solution_is_invariant_to_node_order_on_group_fixture():
    global_line = proposal("g:long", "global_page", [(10, 20), (60, 35), (110, 20)])
    left = local("l:a", "a", [(10, 20), (60, 35)])
    right = local("l:b", "b", [(60, 35), (110, 20)])
    graph = build_reconciliation_graph([global_line, left, right])
    shuffled = dict(graph, nodes=list(reversed(graph["nodes"])),
                    groups=list(reversed(graph["groups"])))
    base = solve(build_actions(graph), graph, WEIGHTS["W2"])
    again = solve(build_actions(shuffled), shuffled, WEIGHTS["W2"])
    assert again["chosen_actions"] == base["chosen_actions"]


def _run(props, weights):
    graph = build_reconciliation_graph(props)
    built = build_actions(graph)
    return graph, statuses(built, graph, solve(built, graph, weights))


def test_every_proposal_gets_exactly_one_status_and_counts_add_up():
    good = proposal("g:ok", "global_page", [(10, 20), (90, 20)])
    dup = local("l:block1:ok", "block1", [(10, 20), (90, 20)])
    missing = local("l:block2:x", "block2", None)
    global_line = proposal("g:long", "global_page", [(10, 20), (60, 35), (110, 20)])
    left = local("l:a", "a", [(10, 20), (60, 35)])
    right = local("l:b", "b", [(60, 35), (110, 20)])
    for weights in WEIGHTS.values():
        graph, res = _run([good, dup, missing, global_line, left, right], weights)
        assert set(res["statuses"]) == {n["proposal"]["proposal_id"] for n in graph["nodes"]}
        assert sum(res["counts"].values()) == len(graph["nodes"])
        assert all(v in res["counts"] for v in res["statuses"].values())


def test_w2_group_marks_global_consumed_and_locals_emitted():
    global_line = proposal("g:long", "global_page", [(10, 20), (60, 35), (110, 20)])
    left = local("l:a", "a", [(10, 20), (60, 35)])
    right = local("l:b", "b", [(60, 35), (110, 20)])
    _, res = _run([global_line, left, right], WEIGHTS["W2"])
    assert res["statuses"]["g:long"] == "consumed_not_emitted"
    assert res["statuses"]["l:a"] == "emitted"
    assert res["statuses"]["l:b"] == "emitted"


def test_w3_prefers_untruncated_global_over_truncated_local_duplicate():
    glob = proposal("g:t", "global_page", [(10, 20), (99, 20)])
    crop = CropTransform((0.0, 0.0, 100.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    trunc = proposal("l:t", "local_block", [(10, 20), (99, 20)], block="b", crop=crop)
    _, res = _run([glob, trunc], WEIGHTS["W3"])
    assert res["statuses"]["g:t"] == "emitted"
    assert res["statuses"]["l:t"] == "not_selected"


def test_missing_geometry_is_ineligible_with_empty_support():
    good = proposal("g:ok", "global_page", [(10, 20), (90, 20)])
    crop = CropTransform((0.0, 0.0, 200.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    missing = proposal("l:block2:q1", "local_block", None, block="block2", crop=crop)
    actions = build_actions(build_reconciliation_graph([good, missing]))
    assert actions["ineligible"] == ["l:block2:q1"]
    assert action(actions, "single:l:block2:q1")["S"] == []
    assert all("l:block2:q1" not in members for members in actions["supports"].values())
