"""ALTO 4.4: a String may have no coordinates when the word geometry is unavailable; empty source lines are omitted.

LOT-P01c v4 (S1, S2). The PAGE profile is unchanged: a PAGE Word needs Coords.
"""
from __future__ import annotations

from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import pytest

from babaorum.bench_metrics import texts_from_alto
from babaorum.export_page import ALTO_NS, ExportError, export_alto, export_pagexml, validate_export_xsd
from babaorum.page_graph import make_geometry, make_text, validate_page_graph
from test_export_page import SCHEMAS, exportable_graph
from test_page_graph import coverage, geom_line, geom_polygon, relation

POSITION = {"HPOS", "VPOS", "WIDTH", "HEIGHT"}


def graph_with_unavailable_word():
    graph = exportable_graph()
    graph["words"][0]["polygon"] = make_geometry(None)  # W1: geometry unavailable
    validate_page_graph(graph)
    return graph


def export(graph, **kw):
    return export_alto(graph, image_filename="scan.tif", physical_img_nr=1, schema_dir=SCHEMAS, **kw)


def strings(result):
    root = ET.fromstring(result.xml)
    return {s.attrib["CONTENT"]: s for s in root.iter(f"{{{ALTO_NS}}}String")}


def test_strict_alto_with_a_string_without_coordinates_is_accepted_and_valid_against_the_xsd():
    result = export(graph_with_unavailable_word(), mode="strict")
    assert result.accepted and result.xml
    validate_export_xsd(result, schema_dir=SCHEMAS)
    info = [loss for loss in result.losses if loss.code == "word_geometry_unavailable"]
    assert len(info) == 1 and info[0].severity == "info" and info[0].path == "words/W1/polygon"
    assert not any(loss.severity == "error" for loss in result.losses)


def test_a_string_without_geometry_has_no_position_attributes_and_no_shape_but_others_keep_theirs():
    result = export(graph_with_unavailable_word(), mode="strict")
    found = strings(result)
    bare, boxed = found["première"], found["deuxième"]
    assert POSITION.isdisjoint(bare.attrib) and bare.find(f"{{{ALTO_NS}}}Shape") is None
    assert POSITION <= set(boxed.attrib) and boxed.find(f"{{{ALTO_NS}}}Shape") is not None


def test_text_round_trips_through_an_alto_with_unavailable_word_geometry():
    graph = graph_with_unavailable_word()
    result = export(graph, mode="strict")
    assert texts_from_alto(result.xml) == [line["text"]["value"] for line in graph["lines"]]
    assert json.loads(result.sidecar_json) == graph  # the companion graph is untouched


def test_a_failed_word_geometry_is_still_refused_and_pagexml_still_needs_coords():
    graph = exportable_graph()
    graph["words"][0]["polygon"] = make_geometry(None, error=True)
    graph["words"][0]["status"] = "partial"
    graph["coverage"]["words"] = coverage("partial", 2, 2)
    graph["run_status"] = "partial"
    validate_page_graph(graph)
    refused = export(graph, mode="strict")
    assert refused.accepted is False and any(loss.severity == "error" for loss in refused.losses)
    page_graph = graph_with_unavailable_word()
    refused = export_pagexml(page_graph, image_filename="scan.tif", mode="strict", quantization="nearest_pixel", schema_dir=SCHEMAS)
    assert refused.accepted is False and any(loss.code == "geometry_unavailable" for loss in refused.losses)


def test_best_effort_also_keeps_a_word_with_unavailable_geometry():
    result = export(graph_with_unavailable_word(), mode="best_effort")
    assert result.accepted and "première" in strings(result)


def graph_with_empty_line_between_two_lines():
    graph = exportable_graph()
    graph["lines"].insert(1, {
        "id": "Le", "status": "ok", "polygon": geom_polygon(8, 21, 60, 24), "baseline": geom_line(8, 23, 60),
        "baseline_semantics": "writing_line_foot_of_body", "orientation": "ltr",
        "baseline_band": {"status": "missing", "normal_negative_px": None, "normal_positive_px": None, "source": None},
        "text": make_text("")})
    graph["relations"] += [relation("contains", "B1", "Le"), relation("precedes", "L1", "Le"), relation("precedes", "Le", "L2")]
    graph["coverage"]["lines"] = coverage("complete", 3, 3)
    graph["coverage"]["transcription"] = coverage("complete", 5, 5)
    graph["coverage"]["reading_order"] = coverage("complete", 4, 4)
    validate_page_graph(graph)
    return graph


def test_an_empty_source_line_is_omitted_from_the_alto_only_and_declared():
    graph = graph_with_empty_line_between_two_lines()
    result = export(graph, mode="strict")
    assert result.accepted
    validate_export_xsd(result, schema_dir=SCHEMAS)
    omitted = [loss for loss in result.losses if loss.code == "empty_line_omitted"]
    assert len(omitted) == 1 and omitted[0].severity == "info" and omitted[0].path == "lines/Le"
    assert texts_from_alto(result.xml) == ["première", "deuxième"]  # the neighbours keep their order
    assert "Le" in {line["id"] for line in json.loads(result.sidecar_json)["lines"]}  # the companion graph keeps it
    page = export_pagexml(graph, image_filename="scan.tif", mode="strict", quantization="nearest_pixel", schema_dir=SCHEMAS)
    assert page.accepted and page.xml.count("<ns0:TextLine") + page.xml.count("<TextLine") == 3


def test_a_line_with_text_but_no_word_is_never_silently_omitted():
    graph = exportable_graph()
    graph["words"] = [w for w in graph["words"] if w["id"] != "W2"]
    graph["relations"] = [r for r in graph["relations"] if "W2" not in (r["source_id"], r["target_id"])]
    graph["coverage"]["words"] = coverage("complete", 1, 1)
    graph["coverage"]["transcription"] = coverage("complete", 3, 3)
    graph["coverage"]["reading_order"] = coverage("complete", 1, 1)
    validate_page_graph(graph)
    result = export(graph, mode="strict")
    assert result.accepted is False and any(loss.code == "words_required_for_alto" for loss in result.losses)


def with_middle_line_text(text):
    graph = graph_with_empty_line_between_two_lines()
    graph["lines"][1]["text"] = make_text(text)
    validate_page_graph(graph)
    return graph


def test_a_whitespace_only_line_is_omitted_under_its_own_code_apart_from_empty_lines():
    result = export(with_middle_line_text("   "), mode="strict")
    assert result.accepted
    validate_export_xsd(result, schema_dir=SCHEMAS)
    codes = [loss.code for loss in result.losses if loss.code in {"empty_line_omitted", "blank_line_omitted"}]
    assert codes == ["blank_line_omitted"]  # never confused with a literally empty line
    assert texts_from_alto(result.xml) == ["première", "deuxième"]
    assert next(l for l in json.loads(result.sidecar_json)["lines"] if l["id"] == "Le")["text"]["value"] == "   "


def test_a_literally_empty_line_keeps_the_empty_code_and_counts_do_not_mix():
    graph = with_middle_line_text("")
    result = export(graph, mode="strict")
    assert [loss.code for loss in result.losses if loss.code in {"empty_line_omitted", "blank_line_omitted"}] == ["empty_line_omitted"]


def test_a_line_with_one_non_blank_character_is_in_neither_code_and_is_refused_without_words():
    result = export(with_middle_line_text(" a "), mode="strict")
    assert result.accepted is False
    assert any(loss.code == "words_required_for_alto" for loss in result.losses)
    assert not any(loss.code in {"empty_line_omitted", "blank_line_omitted"} for loss in result.losses)
