from __future__ import annotations

from copy import deepcopy

import pytest

from babaorum.page_graph import (
    COORDINATE_FRAME,
    COORDINATE_CONVENTION,
    NORMAL_CONVENTION,
    SCHEMA_VERSION,
    PageGraphError,
    make_geometry,
    make_text,
    validate_page_graph,
)


def geom_polygon(x0, y0, x1, y1, source="predicted"):
    return make_geometry([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], source)


def geom_line(x0, y, x1):
    return make_geometry([[x0, y], [x1, y]], "predicted")


def coverage(status, expected, available, errors=0):
    return {
        "status": status,
        "expected_count": expected,
        "available_count": available,
        "error_count": errors,
    }


def relation(kind, source_id, target_id, confidence=1.0):
    return {
        "type": kind,
        "source_id": source_id,
        "target_id": target_id,
        "source": "predicted",
        "confidence": confidence,
    }


def valid_graph():
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "page_key": "gallica:7635415/f2",
            "document_key": "gallica:7635415",
            "title_key": "le-journal-exemple",
        },
        "coordinate_frame": COORDINATE_FRAME,
        "coordinate_convention": COORDINATE_CONVENTION,
        "normal_convention": NORMAL_CONVENTION,
        "image": {"sha256": "a" * 64, "width": 100, "height": 60},
        "provenance": {
            "run_id": "run-001",
            "components": [{
                "name": "reader",
                "code_revision": "abc123",
                "config_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "license": "BSD-3-Clause",
            }],
        },
        "run_status": "ok",
        "coverage": {
            "blocks": coverage("complete", 1, 1),
            "lines": coverage("complete", 2, 2),
            "words": coverage("complete", 2, 2),
            "transcription": coverage("complete", 4, 4),
            "reading_order": coverage("complete", 2, 2),
        },
        "articles": [{"id": "A1", "status": "ok"}],
        "blocks": [{
            "id": "B1", "status": "ok", "category": "texte",
            "polygon": geom_polygon(5, 5, 90, 50),
        }],
        "lines": [
            {"id": "L1", "status": "ok", "polygon": geom_polygon(8, 8, 60, 20),
             "baseline": geom_line(8, 18, 60), "baseline_semantics": "writing_line_foot_of_body",
             "orientation": "ltr", "baseline_band": {"status": "missing", "normal_negative_px": None, "normal_positive_px": None, "source": None},
             "text": make_text("première ligne")},
            {"id": "L2", "status": "ok", "polygon": geom_polygon(8, 25, 60, 37),
             "baseline": make_geometry([[8, 35], [34, 35.5], [60, 35]], "predicted"),
             "baseline_semantics": "writing_line_foot_of_body",
             "orientation": "unknown", "baseline_band": {"status": "available", "normal_negative_px": 8.0, "normal_positive_px": 2.0, "source": "predicted"},
             "text": make_text("deuxième ligne")},
        ],
        "words": [
            {"id": "W1", "status": "ok", "polygon": geom_polygon(8, 9, 25, 19), "text": make_text("première")},
            {"id": "W2", "status": "ok", "polygon": geom_polygon(8, 26, 28, 36), "text": make_text("deuxième")},
        ],
        "relations": [
            relation("contains", "A1", "B1"),
            relation("contains", "B1", "L1"),
            relation("contains", "B1", "L2"),
            relation("contains", "L1", "W1"),
            relation("contains", "L2", "W2"),
            relation("precedes", "L1", "L2"),
            relation("precedes", "W1", "W2"),
        ],
    }


def test_valid_graph_and_explicit_partial_order():
    graph = valid_graph()
    validate_page_graph(graph)
    # No array-order or geometry heuristic is needed to read the explicit edges.
    graph["relations"].pop()
    graph["coverage"]["reading_order"] = coverage("partial", 2, 1)
    graph["run_status"] = "partial"
    validate_page_graph(graph)


def test_missing_word_geometry_is_valid_and_not_synthesized():
    graph = valid_graph()
    graph["words"][0]["polygon"] = make_geometry(None)
    assert graph["words"][0]["polygon"] == {
        "status": "missing", "points": None, "source": None
    }
    validate_page_graph(graph)


def test_unavailable_word_segmentation_can_be_empty():
    graph = valid_graph()
    graph["words"] = []
    graph["relations"] = [edge for edge in graph["relations"] if edge["target_id"] not in {"W1", "W2"}]
    graph["coverage"]["words"] = coverage("unavailable", None, 0)
    graph["coverage"]["transcription"] = coverage("complete", 2, 2)
    graph["coverage"]["reading_order"] = coverage("complete", 1, 1)
    graph["relations"] = [edge for edge in graph["relations"] if edge["type"] != "precedes" or edge["source_id"].startswith("L")]
    validate_page_graph(graph)


@pytest.mark.parametrize("mutation,match", [
    (lambda g: g["image"].update(width=0), "positive"),
    (lambda g: g["image"].update(sha256="bad"), "SHA-256"),
    (lambda g: g["identity"].update(page_key=""), "nonempty"),
    (lambda g: g["blocks"][0]["polygon"]["points"].__setitem__(1, [101, 5]), "outside"),
    (lambda g: g["blocks"][0]["polygon"]["points"].__setitem__(1, [float("nan"), 5]), "finite"),
    (lambda g: g["lines"][0]["baseline"].update(points=[[10, 10]]), "at least two"),
    (lambda g: g["coverage"]["lines"].update(status="unknown", expected_count=None, available_count=1), "node statuses"),
])
def test_invalid_identity_geometry_and_coverage_are_rejected(mutation, match):
    graph = deepcopy(valid_graph())
    mutation(graph)
    with pytest.raises(PageGraphError, match=match):
        validate_page_graph(graph)


def test_duplicate_global_ids_and_multiple_parents_are_rejected():
    graph = deepcopy(valid_graph())
    graph["articles"].append({"id": "L1", "status": "ok"})
    with pytest.raises(PageGraphError, match="duplicate global"):
        validate_page_graph(graph)

    graph = deepcopy(valid_graph())
    graph["articles"].append({"id": "A2", "status": "ok"})
    graph["relations"].append(relation("contains", "A2", "B1"))
    with pytest.raises(PageGraphError, match="more than one parent"):
        validate_page_graph(graph)


def test_order_cycles_and_wrong_typed_edges_are_rejected():
    graph = deepcopy(valid_graph())
    graph["relations"].append(relation("precedes", "L2", "L1"))
    graph["coverage"]["reading_order"] = coverage("complete", 3, 3)
    with pytest.raises(PageGraphError, match="cycle"):
        validate_page_graph(graph)

    graph = deepcopy(valid_graph())
    graph["relations"].append(relation("precedes", "L1", "W1"))
    graph["coverage"]["reading_order"] = coverage("partial", 4, 3)
    graph["run_status"] = "partial"
    with pytest.raises(PageGraphError, match="same type"):
        validate_page_graph(graph)


def test_polygon_self_intersection_and_text_error_are_rejected():
    graph = deepcopy(valid_graph())
    graph["blocks"][0]["polygon"]["points"] = [[5, 5], [80, 40], [10, 40], [80, 5], [50, 25]]
    with pytest.raises(PageGraphError, match="self-intersect"):
        validate_page_graph(graph)

    graph = deepcopy(valid_graph())
    graph["lines"][0]["text"] = {"status": "error", "value": "oops"}
    with pytest.raises(PageGraphError, match="value=null"):
        validate_page_graph(graph)


def test_error_units_are_counted_and_run_status_is_coherent():
    graph = deepcopy(valid_graph())
    graph["words"][1] = {
        "id": "W2", "status": "error", "polygon": make_geometry(None, error=True),
        "text": make_text(None, error=True),
    }
    graph["coverage"]["words"] = coverage("partial", 2, 1, 1)
    graph["coverage"]["transcription"] = coverage("partial", 4, 3, 1)
    graph["run_status"] = "partial"
    validate_page_graph(graph)

    graph["run_status"] = "ok"
    with pytest.raises(PageGraphError, match="run_status"):
        validate_page_graph(graph)


def test_geometry_error_requires_non_ok_run_and_partial_node():
    graph = deepcopy(valid_graph())
    graph["blocks"][0]["status"] = "partial"
    graph["blocks"][0]["polygon"] = make_geometry(None, error=True)
    graph["coverage"]["blocks"] = coverage("partial", 1, 1)
    with pytest.raises(PageGraphError, match="run_status"):
        validate_page_graph(graph)
    graph["run_status"] = "partial"
    validate_page_graph(graph)


def test_cross_level_reading_order_contradiction_is_rejected():
    graph = deepcopy(valid_graph())
    graph["relations"].append(relation("precedes", "W2", "W1"))
    graph["coverage"]["reading_order"] = coverage("complete", 3, 3)
    with pytest.raises(PageGraphError, match="known line ancestors"):
        validate_page_graph(graph)


@pytest.mark.parametrize("mutation,match", [
    (lambda g: g["blocks"][0].update(status=[]), "status"),
    (lambda g: g["blocks"][0]["polygon"].update(source={"bad": 1}), "source"),
    (lambda g: g["identity"].update(page_key="  "), "nonempty"),
    (lambda g: g["provenance"].update(run_id="  "), "nonempty"),
    (lambda g: g["blocks"][0]["polygon"]["points"].__setitem__(1, [10**10000, 5]), "representable"),
    (lambda g: g["lines"][0].update(orientation="ltr", baseline=geom_line(60, 18, 8)), "endpoint direction"),
    (lambda g: g["lines"][1]["baseline_band"].update(normal_negative_px=0.0, normal_positive_px=0.0), "positive total extent"),
])
def test_hostile_values_and_baseline_semantics_are_rejected(mutation, match):
    graph = deepcopy(valid_graph())
    mutation(graph)
    with pytest.raises(PageGraphError, match=match):
        validate_page_graph(graph)


def test_baseline_band_and_text_status_are_type_safe():
    graph = deepcopy(valid_graph())
    graph["lines"][0]["baseline"] = make_geometry(None)
    graph["lines"][0]["baseline_band"] = {
        "status": "available", "normal_negative_px": 1.0,
        "normal_positive_px": 1.0, "source": "predicted",
    }
    with pytest.raises(PageGraphError, match="requires an available baseline"):
        validate_page_graph(graph)

    graph = deepcopy(valid_graph())
    graph["lines"][0]["text"]["status"] = []
    with pytest.raises(PageGraphError, match="status"):
        validate_page_graph(graph)


def test_blank_text_is_preserved_as_an_available_transcription():
    graph = deepcopy(valid_graph())
    graph["lines"][0]["text"] = make_text("")
    validate_page_graph(graph)


# --- G1 I1 (2026-10-08): optional graph_role; prediction graphs fail closed ---

GT_SOURCE_MUTATIONS = [
    ("block polygon", lambda g: g["blocks"][0]["polygon"].update(source="ground_truth")),
    ("line polygon", lambda g: g["lines"][0]["polygon"].update(source="ground_truth")),
    ("line baseline", lambda g: g["lines"][0]["baseline"].update(source="ground_truth")),
    ("baseline band", lambda g: g["lines"][1]["baseline_band"].update(source="ground_truth")),
    ("word polygon", lambda g: g["words"][0]["polygon"].update(source="ground_truth")),
    ("relation", lambda g: g["relations"][0].update(source="ground_truth")),
]


@pytest.mark.parametrize("label,mutation", GT_SOURCE_MUTATIONS)
def test_prediction_role_refuses_any_ground_truth_provenance(label, mutation):
    graph = deepcopy(valid_graph())
    graph["graph_role"] = "prediction"
    validate_page_graph(graph)
    mutation(graph)
    with pytest.raises(PageGraphError, match="prediction graph cannot claim ground_truth"):
        validate_page_graph(graph)


@pytest.mark.parametrize("label,mutation", GT_SOURCE_MUTATIONS)
def test_prediction_role_refuses_unqualified_transferred_provenance(label, mutation):
    # N3: "transferred" names no origin and may come from ground truth.
    def to_transferred(graph):
        mutation(graph)
        for collection in ("blocks", "lines", "words"):
            for node in graph[collection]:
                for key in ("polygon", "baseline", "baseline_band"):
                    if key in node and node[key]["source"] == "ground_truth":
                        node[key]["source"] = "transferred"
        for relation in graph["relations"]:
            if relation["source"] == "ground_truth":
                relation["source"] = "transferred"
    graph = deepcopy(valid_graph())
    to_transferred(graph)
    validate_page_graph(graph)  # still accepted without a role
    graph["graph_role"] = "prediction"
    with pytest.raises(PageGraphError, match="prediction graph cannot claim transferred"):
        validate_page_graph(graph)
    graph["graph_role"] = "gt_assisted"
    validate_page_graph(graph)


@pytest.mark.parametrize("label,mutation", GT_SOURCE_MUTATIONS)
@pytest.mark.parametrize("role", [None, "ground_truth", "gt_assisted"])
def test_ground_truth_provenance_is_only_structural_without_prediction_role(label, mutation, role):
    # Backward compatible: without a declared role (or with a GT/GT-assisted
    # role) the validator does not police provenance; callers that score
    # predictions must declare graph_role="prediction" or check sources.
    graph = deepcopy(valid_graph())
    if role is not None:
        graph["graph_role"] = role
    mutation(graph)
    validate_page_graph(graph)


@pytest.mark.parametrize("bad", ["predicted", "", None, 1, ["prediction"]])
def test_graph_role_is_a_closed_enum(bad):
    graph = deepcopy(valid_graph())
    graph["graph_role"] = bad
    with pytest.raises(PageGraphError, match="graph_role"):
        validate_page_graph(graph)


# --- G1 m3: normal side for every orientation, including rtl --------------

def _declared_normal(first, last):
    # NORMAL_CONVENTION "n=(-ty,tx);positive_90deg_y_down", t from first to last.
    tx, ty = last[0] - first[0], last[1] - first[1]
    return (-ty, tx)


@pytest.mark.parametrize("orientation,points,positive_side", [
    ("ltr", [[8, 18], [60, 18]], (0, 1)),    # positive side downwards (y grows)
    ("rtl", [[60, 18], [8, 18]], (0, -1)),   # positive side upwards (y decreases)
    ("ttb", [[30, 8], [30, 50]], (-1, 0)),   # positive side to the left
    ("btt", [[30, 50], [30, 8]], (1, 0)),    # positive side to the right
])
def test_normal_side_per_orientation_matches_contract(orientation, points, positive_side):
    assert NORMAL_CONVENTION == "n=(-ty,tx);positive_90deg_y_down"
    nx, ny = _declared_normal(points[0], points[-1])
    scale = max(abs(nx), abs(ny))
    assert (nx / scale, ny / scale) == positive_side
    graph = deepcopy(valid_graph())
    graph["lines"][1]["orientation"] = orientation
    graph["lines"][1]["baseline"] = make_geometry(points, "predicted")
    validate_page_graph(graph)  # band stays attached to the declared normal
