from dataclasses import replace
import hashlib

import pytest

from babaorum.experimental.joint_reconcile import (
    CropTransform, LineProposal, ReconcileConfig,
    build_reconciliation_graph, graph_sha256,
)


SHA = hashlib.sha256(b"synthetic").hexdigest()


def proposal(pid, source, points, *, block=None, crop=None, orientation="ltr",
             text_status="available", text="", ctc_status="empty", query=None):
    return LineProposal(
        proposal_id=pid, source=source, page_key="fixture:p", document_key="fixture:d",
        page_size_wh=(200.0, 100.0), page_sha256=SHA, query_id=query or pid, input_sha256=SHA,
        checkpoint_sha256=None, geometry_status="available" if points is not None else "missing",
        baseline_xy=tuple(tuple(xy) for xy in points) if points is not None else None,
        band_normal_negative_px=None, band_normal_positive_px=None,
        orientation=orientation, text_status=text_status, text=text,
        token_ids=() if ctc_status == "empty" else None, ctc_status=ctc_status,
        block_id=block, crop_transform=crop,
    )


def local(pid, block, points, *, crop=None, **kwargs):
    transform = crop or CropTransform((0.0, 0.0, 200.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    return proposal(pid, "local_block", points, block=block, crop=transform, **kwargs)


def edge(graph, left, right):
    return next(e for e in graph["edges"] if e["source_ids"] == [left, right])


def test_exact_duplicate_candidates_keep_both_texts_and_empty_ctc_distinct():
    a = proposal("g:q0", "global_page", [(10, 20), (90, 20)], text="", ctc_status="empty")
    b = local("l:block1:q0", "block1", [(10, 20), (90, 20)], text="ABC", ctc_status="feasible")
    graph = build_reconciliation_graph([b, a])
    candidate = edge(graph, "g:q0", "l:block1:q0")
    assert candidate["labels"] == ["duplicate_candidate"]
    assert graph["selection"] is None
    nodes = {x["proposal"]["proposal_id"]: x for x in graph["nodes"]}
    assert nodes["g:q0"]["proposal"]["text"] == ""
    assert nodes["g:q0"]["proposal"]["ctc_status"] == "empty"
    assert nodes["l:block1:q0"]["proposal"]["text"] == "ABC"
    assert nodes["l:block1:q0"]["proposal"]["ctc_status"] == "feasible"


def test_global_line_can_have_partial_edges_to_multiple_locals_without_selection():
    global_line = proposal("g:long", "global_page", [(10, 20), (60, 35), (110, 20)])
    left = local("l:a", "a", [(10, 20), (60, 35)])
    right = local("l:b", "b", [(60, 35), (110, 20)])
    graph = build_reconciliation_graph([global_line, right, left])
    assert len(graph["nodes"]) == 3
    assert edge(graph, "g:long", "l:a")["labels"] == ["partial_overlap_candidate"]
    assert edge(graph, "g:long", "l:b")["labels"] == ["partial_overlap_candidate"]
    assert edge(graph, "l:a", "l:b")["status"] == "no_candidate"
    assert len(graph["groups"]) == 1
    assert graph["groups"][0]["global_proposal_id"] == "g:long"
    assert graph["groups"][0]["local_proposal_ids"] == ["l:a", "l:b"]
    assert graph["groups"][0]["block_ids"] == ["a", "b"]
    assert graph["groups"][0]["global_arc_coverage_union"] == pytest.approx(1.0, abs=1e-12)
    assert graph["groups"][0]["increment_over_best_single_coverage"] > 0.3
    individual = graph["groups"][0]["individual_global_arc_coverage"]
    union = graph["groups"][0]["global_arc_coverage_union"]
    assert graph["groups"][0]["redundant_coverage_sum_minus_union"] == pytest.approx(
        sum(individual.values()) - union
    )
    assert len(graph["groups"][0]["member_edges"]) == 2
    assert all("partial_overlap_candidate" in item["labels"]
               for item in graph["groups"][0]["member_edges"])
    assert graph["groups"][0]["status"] == "multi_local_coverage_candidate"
    assert graph["groups"][0]["selection"] is None
    densified_global = replace(global_line, baseline_xy=(
        (10, 20), (15, 21.5), (55, 33.5), (60, 35), (65, 33.5), (105, 21.5), (110, 20)
    ))
    densified = build_reconciliation_graph([densified_global, right, left])
    assert edge(densified, "g:long", "l:a")["coverage_left_on_right"] == pytest.approx(
        edge(graph, "g:long", "l:a")["coverage_left_on_right"], abs=1e-12
    )
    base_edge = edge(graph, "g:long", "l:a")
    dense_edge = edge(densified, "g:long", "l:a")
    assert dense_edge["coverage_right_on_left"] == pytest.approx(base_edge["coverage_right_on_left"], abs=1e-12)
    assert dense_edge["endpoint_distance_heights"] == pytest.approx(base_edge["endpoint_distance_heights"], abs=1e-12)
    base_group, dense_group = graph["groups"][0], densified["groups"][0]
    for key in ("increment_over_best_single_coverage", "redundant_coverage_sum_minus_union"):
        assert dense_group[key] == pytest.approx(base_group[key], abs=1e-12)
    for proposal_id, coverage in base_group["individual_global_arc_coverage"].items():
        assert dense_group["individual_global_arc_coverage"][proposal_id] == pytest.approx(coverage, abs=1e-12)
    assert dense_group["member_edges"] == base_group["member_edges"]


def test_candidate_geometry_is_invariant_to_page_scale_when_config_tracks_scale():
    base_crop = CropTransform((50.0, 10.0, 150.0, 110.0), 1.5, (4.0, 6.0, 2.0, 3.0))
    global_base = replace(proposal("g:scale", "global_page", [(55, 50), (120, 50)]),
                          page_size_wh=(200.0, 120.0))
    local_base = replace(local("l:scale", "b", [(11.5, 66.0), (109.0, 66.0)], crop=base_crop),
                         page_size_wh=(200.0, 120.0))
    first = build_reconciliation_graph([global_base, local_base], ReconcileConfig(normalization_height_px=20))

    scaled_crop = CropTransform((100.0, 20.0, 300.0, 220.0), 1.5, (8.0, 12.0, 4.0, 6.0))
    global_scaled = replace(global_base, page_size_wh=(400.0, 240.0), baseline_xy=((110, 100), (240, 100)),
                            input_sha256=hashlib.sha256(b"scaled page model input").hexdigest())
    local_scaled = replace(local_base, page_size_wh=(400.0, 240.0),
                           baseline_xy=((23.0, 132.0), (218.0, 132.0)), crop_transform=scaled_crop,
                           input_sha256=hashlib.sha256(b"scaled crop model input").hexdigest())
    second = build_reconciliation_graph(
        [global_scaled, local_scaled], ReconcileConfig(normalization_height_px=40)
    )
    first_edge = edge(first, "g:scale", "l:scale")
    second_edge = edge(second, "g:scale", "l:scale")
    assert first_edge["labels"] == second_edge["labels"]
    assert first_edge["coverage_left_on_right"] == pytest.approx(second_edge["coverage_left_on_right"])
    assert first_edge["coverage_right_on_left"] == pytest.approx(second_edge["coverage_right_on_left"])
    assert first_edge["endpoint_distance_heights"] == pytest.approx(second_edge["endpoint_distance_heights"])
    first_local = next(n["proposal"] for n in first["nodes"] if n["proposal"]["proposal_id"] == "l:scale")
    second_local = next(n["proposal"] for n in second["nodes"] if n["proposal"]["proposal_id"] == "l:scale")
    assert first_local["page_sha256"] == second_local["page_sha256"]
    assert first_local["input_sha256"] != second_local["input_sha256"]
    first_flags = next(n["crop_flags"] for n in first["nodes"]
                       if n["proposal"]["proposal_id"] == "l:scale")
    second_flags = next(n["crop_flags"] for n in second["nodes"]
                        if n["proposal"]["proposal_id"] == "l:scale")
    assert first_flags == second_flags
    assert first_flags["may_be_truncated_by_crop_band"] is True
    assert first_flags["touches_internal_crop_edge"] is False


def test_crop_scale_factor_scales_epsilon_and_band_extension_on_same_page():
    rect = (50.0, 10.0, 150.0, 110.0)
    pad = (4.0, 6.0, 2.0, 3.0)
    crop_15 = CropTransform(rect, 1.5, pad)
    crop_3 = CropTransform(rect, 3.0, pad)

    def local_coordinates(page_points, scale):
        return tuple((pad[0] + (x - rect[0]) * scale,
                      pad[1] + (y - rect[1]) * scale) for x, y in page_points)

    near = ((50.5, 50.0), (80.0, 50.0))  # 0.5 page px: epsilon must scale
    mid = ((60.0, 70.0), (120.0, 70.0))  # 10 page px: extension omission changes result at scale 3
    proposals = [
        replace(proposal("g:near", "global_page", near), page_size_wh=(200.0, 120.0)),
        replace(proposal("g:mid", "global_page", mid), page_size_wh=(200.0, 120.0)),
        replace(local("l:near15", "near", local_coordinates(near, 1.5), crop=crop_15),
                page_size_wh=(200.0, 120.0), input_sha256=hashlib.sha256(b"near crop scale 1.5").hexdigest()),
        replace(local("l:near3", "near", local_coordinates(near, 3.0), crop=crop_3),
                page_size_wh=(200.0, 120.0), input_sha256=hashlib.sha256(b"near crop scale 3").hexdigest()),
        replace(local("l:mid15", "mid", local_coordinates(mid, 1.5), crop=crop_15),
                page_size_wh=(200.0, 120.0), input_sha256=hashlib.sha256(b"mid crop scale 1.5").hexdigest()),
        replace(local("l:mid3", "mid", local_coordinates(mid, 3.0), crop=crop_3),
                page_size_wh=(200.0, 120.0), input_sha256=hashlib.sha256(b"mid crop scale 3").hexdigest()),
    ]
    graph = build_reconciliation_graph(proposals, ReconcileConfig(normalization_height_px=20))
    nodes = {node["proposal"]["proposal_id"]: node for node in graph["nodes"]}
    assert nodes["l:near15"]["crop_flags"] == nodes["l:near3"]["crop_flags"]
    assert nodes["l:near15"]["crop_flags"]["touches_internal_crop_edge"] is True
    assert nodes["l:mid15"]["crop_flags"] == nodes["l:mid3"]["crop_flags"]
    assert nodes["l:mid15"]["crop_flags"]["touches_internal_crop_edge"] is False
    assert nodes["l:mid15"]["crop_flags"]["may_be_truncated_by_crop_band"] is True
    assert nodes["l:mid3"]["crop_flags"]["may_be_truncated_by_crop_band"] is True
    for local_id, global_id in (("l:near15", "g:near"), ("l:near3", "g:near"),
                                ("l:mid15", "g:mid"), ("l:mid3", "g:mid")):
        candidate = edge(graph, global_id, local_id)
        assert candidate["labels"] == ["duplicate_candidate", "crop_truncation_possible"]
        assert candidate["endpoint_distance_heights"] == pytest.approx(0.0)


def test_explicit_crop_band_extents_scale_with_crop_on_same_page():
    rect = (50.0, 10.0, 150.0, 110.0)
    padding = (4.0, 6.0, 2.0, 3.0)
    page_points = ((60.0, 25.0), (120.0, 25.0))
    page_hash = hashlib.sha256(b"explicit-band-scale-page").hexdigest()
    global_line = replace(
        proposal("g:explicit-band", "global_page", page_points),
        page_size_wh=(200.0, 120.0), page_sha256=page_hash,
    )
    locals_ = []
    for scale, negative, positive in ((1.5, 30.0, 10.0), (3.0, 60.0, 20.0)):
        crop = CropTransform(rect, scale, padding)
        points = tuple((padding[0] + (x - rect[0]) * scale,
                        padding[1] + (y - rect[1]) * scale)
                       for x, y in page_points)
        locals_.append(replace(
            local(f"l:explicit-band:{scale:g}", "b", points, crop=crop),
            page_size_wh=(200.0, 120.0), page_sha256=page_hash,
            input_sha256=hashlib.sha256(f"explicit-band-crop-{scale:g}".encode()).hexdigest(),
            band_normal_negative_px=negative, band_normal_positive_px=positive,
        ))

    graph = build_reconciliation_graph([global_line, *locals_], ReconcileConfig())
    nodes = {node["proposal"]["proposal_id"]: node for node in graph["nodes"]}
    flags_15 = nodes["l:explicit-band:1.5"]["crop_flags"]
    flags_3 = nodes["l:explicit-band:3"]["crop_flags"]
    assert flags_15 == flags_3
    assert flags_15["may_be_truncated_by_crop_band"] is True
    assert flags_15["touches_internal_crop_edge"] is False
    assert nodes["l:explicit-band:3"]["proposal"]["band_normal_negative_px"] == pytest.approx(
        2 * nodes["l:explicit-band:1.5"]["proposal"]["band_normal_negative_px"]
    )
    edge_15 = edge(graph, "g:explicit-band", "l:explicit-band:1.5")
    edge_3 = edge(graph, "g:explicit-band", "l:explicit-band:3")
    assert edge_15["labels"] == edge_3["labels"]
    assert "crop_truncation_possible" in edge_15["labels"]


def test_local_local_duplicates_are_compared_and_direction_conflicts_are_flagged():
    a = local("l:a:q", "a", [(20, 30), (80, 30)])
    b = local("l:b:q", "b", [(80, 30), (20, 30)], orientation="rtl")
    graph = build_reconciliation_graph([a, b])
    result = edge(graph, "l:a:q", "l:b:q")
    assert "duplicate_candidate" in result["labels"]
    assert "direction_conflict" in result["labels"]


def test_crop_is_clipped_frame_inverted_once_and_padding_or_truncation_retained():
    transform = CropTransform(
        actual_rect_page=(0.0, 0.0, 20.0, 20.0), scale=2.0,
        padding_ltrb=(5.0, 7.0, 3.0, 3.0),
    )
    line = local("l:edge", "b", [(5, 20), (25, 20), (45, 20), (55, 20)], crop=transform)
    graph = build_reconciliation_graph([line])
    node = graph["nodes"][0]
    assert node["baseline_page_xy"] == ((0.0, 6.5), (10.0, 6.5), (20.0, 6.5), (25.0, 6.5))
    assert node["crop_flags"]["touches_crop_content_edge"] is True
    assert node["crop_flags"]["has_points_in_padding_or_outside_content"] is True
    assert node["proposal"]["crop_transform"]["actual_rect_page"] == (0.0, 0.0, 20.0, 20.0)


def test_actual_crop_must_be_clipped_and_rotation_or_mirror_is_rejected():
    outside = CropTransform((-1.0, 0.0, 20.0, 20.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="inside page bounds"):
        build_reconciliation_graph([local("l:oob", "b", [(0, 1), (5, 1)], crop=outside)])
    rotated = CropTransform((0.0, 0.0, 20.0, 20.0), 1.0, (0.0, 0.0, 0.0, 0.0), 90.0)
    with pytest.raises(ValueError, match="rotation or mirroring"):
        build_reconciliation_graph([local("l:rot", "b", [(0, 1), (5, 1)], crop=rotated)])


def test_segment_crossing_crop_edge_between_vertices_is_flagged():
    transform = CropTransform((0.0, 0.0, 20.0, 20.0), 1.0, (5.0, 5.0, 5.0, 5.0))
    line = local("l:cross", "b", [(2, 15), (28, 15)], crop=transform)
    node = build_reconciliation_graph([line])["nodes"][0]
    assert node["crop_flags"]["touches_crop_content_edge"] is True
    assert node["crop_flags"]["has_points_in_padding_or_outside_content"] is True


def test_same_query_id_is_allowed_across_crops_but_not_within_one_input():
    a = local("l:a", "a", [(1, 10), (10, 10)], query="q0")
    different_crop = replace(local("l:b", "b", [(1, 12), (10, 12)], query="q0"),
                             input_sha256=hashlib.sha256(b"another crop").hexdigest())
    assert len(build_reconciliation_graph([a, different_crop])["nodes"]) == 2
    with pytest.raises(ValueError, match="unique within each source/input crop"):
        build_reconciliation_graph([a, replace(a, proposal_id="l:a2")])


def test_repeated_consecutive_vertices_do_not_abort_page_graph():
    a = proposal("g:repeat", "global_page", [(10, 20), (10, 20), (90, 20)])
    b = local("l:repeat", "b", [(10, 20), (90, 20)])
    graph = build_reconciliation_graph([a, b])
    assert "duplicate_candidate" in edge(graph, "g:repeat", "l:repeat")["labels"]
    node = next(x for x in graph["nodes"] if x["proposal"]["proposal_id"] == "g:repeat")
    assert len(node["proposal"]["baseline_xy"]) == 3


def test_reversed_geometry_is_flagged_even_if_orientation_labels_match():
    a = proposal("g:forward", "global_page", [(10, 20), (90, 20)])
    b = local("l:reverse", "b", [(90, 20), (10, 20)])
    labels = edge(build_reconciliation_graph([a, b]), "g:forward", "l:reverse")["labels"]
    assert "duplicate_candidate" in labels
    assert "reversed_geometry" in labels


def test_band_extent_flags_vertical_crop_truncation_but_page_edge_does_not():
    crop = CropTransform((0.0, 20.0, 100.0, 80.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    near_top = replace(local("l:near-top", "b", [(10, 35), (90, 35)], crop=crop),
                       band_normal_negative_px=15.0, band_normal_positive_px=5.0)
    graph = build_reconciliation_graph([near_top])
    flags = graph["nodes"][0]["crop_flags"]
    assert flags["may_be_truncated_by_crop_band"] is True

    full_page = CropTransform((0.0, 0.0, 200.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    at_page_edge = local("l:page-edge", "page", [(10, 0), (190, 0)], crop=full_page)
    page_flags = build_reconciliation_graph([at_page_edge])["nodes"][0]["crop_flags"]
    assert page_flags["touches_crop_content_edge"] is True
    assert page_flags["touches_internal_crop_edge"] is False
    assert page_flags["may_be_truncated_by_crop_band"] is False


def test_page_source_hash_must_match_across_global_and_local_passes():
    global_line = proposal("g:hash", "global_page", [(10, 30), (90, 30)])
    local_line = replace(local("l:hash", "b", [(10, 30), (90, 30)]),
                         page_sha256=hashlib.sha256(b"different page").hexdigest())
    with pytest.raises(ValueError, match="share document identity and page frame"):
        build_reconciliation_graph([global_line, local_line])


def test_crop_truncation_diagnostic_does_not_create_unrelated_pair_candidate():
    crop = CropTransform((0.0, 0.0, 100.0, 100.0), 1.0, (0.0, 0.0, 0.0, 0.0))
    global_line = proposal("g:far", "global_page", [(10, 70), (50, 70)])
    local_line = local("l:edge-far", "b", [(85, 20), (95, 20)], crop=crop)
    graph = build_reconciliation_graph([global_line, local_line])
    candidate = edge(graph, "g:far", "l:edge-far")
    assert candidate["status"] == "no_candidate"
    assert candidate["labels"] == []
    node = next(item for item in graph["nodes"] if item["proposal"]["proposal_id"] == "l:edge-far")
    assert node["crop_flags"]["may_be_truncated_by_crop_band"] is True


def test_group_rule_requires_each_local_to_be_partial_when_configured():
    global_line = proposal("g:full", "global_page", [(10, 20), (110, 20)])
    full_local = local("l:full", "a", [(10, 20), (110, 20)])
    partial_local = local("l:partial", "b", [(10, 20), (90, 20)])
    graph = build_reconciliation_graph([global_line, full_local, partial_local])
    assert graph["groups"] == []
    assert graph["config"]["multi_local_group_requires_each_partial"] is True


def test_proposal_ids_and_page_frames_are_validated():
    a = proposal("same", "global_page", [(0, 2), (5, 2)])
    b = local("same", "b", [(0, 2), (5, 2)])
    with pytest.raises(ValueError, match="unique"):
        build_reconciliation_graph([a, b])
    b = replace(local("l:b", "b", [(0, 2), (5, 2)]), page_size_wh=(201.0, 100.0))
    with pytest.raises(ValueError, match="share document identity and page frame"):
        build_reconciliation_graph([a, b])


def test_graph_is_permutation_invariant_and_uses_declared_height_thresholds():
    items = [
        proposal("g:a", "global_page", [(10, 40), (80, 40)]),
        local("l:b", "b", [(10, 40), (80, 49)]),
        local("l:c", "c", [(10, 70), (80, 70)]),
    ]
    config = ReconcileConfig(normalization_height_px=18.0, distance_tolerance_heights=0.5)
    first = build_reconciliation_graph(items, config)
    second = build_reconciliation_graph(list(reversed(items)), config)
    assert graph_sha256(first) == graph_sha256(second)
    assert first["config"]["normalization_height_px"] == 18.0
    assert edge(first, "g:a", "l:b")["normalization_height_px"] == 18.0
    assert edge(first, "g:a", "l:c")["status"] == "no_candidate"


def test_unavailable_geometry_and_ctc_error_remain_explicit_nodes():
    invalid = replace(local("l:invalid", "b", [(2, 2), (8, 2)]),
                      geometry_status="error", baseline_xy=None,
                      text_status="error", text=None, token_ids=None,
                      ctc_status="error")
    graph = build_reconciliation_graph([invalid])
    node = graph["nodes"][0]["proposal"]
    assert node["geometry_status"] == "error" and node["baseline_xy"] is None
    assert node["text_status"] == "error" and node["text"] is None
    assert node["ctc_status"] == "error"


def test_gt_dependent_infeasible_length_status_is_forbidden_on_inference_proposals():
    invalid = replace(local("l:bad-state", "b", [(2, 2), (8, 2)]),
                      ctc_status="excluded_infeasible_length", text_status="error", text=None)
    with pytest.raises(ValueError, match="inference-time CTC status is invalid"):
        build_reconciliation_graph([invalid])
