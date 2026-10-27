from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from babaorum.export_page import (
    ALTO_NS,
    PAGE_NS,
    ExportError,
    export_alto,
    export_pagexml,
    validate_export_xsd,
    _ordered_siblings,
)
from test_page_graph import valid_graph


SCHEMAS = Path(__file__).parents[1] / "src" / "babaorum" / "schemas"


def exportable_graph():
    graph = deepcopy(valid_graph())
    graph["articles"] = []
    graph["relations"] = [r for r in graph["relations"] if r["source_id"] != "A1"]
    graph["lines"][0]["text"]["value"] = "première"
    graph["lines"][1]["text"]["value"] = "deuxième"
    graph["coverage"]["transcription"]["available_count"] = 4
    return graph


def test_pagexml_xsd_and_semantic_roundtrip_preserve_line_text_and_geometry():
    graph = exportable_graph()
    result = export_pagexml(
        graph, image_filename="scan-02.tif", quantization="nearest_pixel", schema_dir=SCHEMAS
    )
    assert result.accepted and result.xml
    assert result.sidecar_sha256 == hashlib.sha256(result.sidecar_json.encode()).hexdigest()
    assert json.loads(result.sidecar_json) == graph
    assert result.max_coordinate_error_px == pytest.approx(0.5)
    validate_export_xsd(result, schema_dir=SCHEMAS)

    root = ET.fromstring(result.xml)
    page = root.find(f"{{{PAGE_NS}}}Page")
    assert page is not None
    assert page.attrib["imageFilename"] == "scan-02.tif"
    lines = page.findall(f".//{{{PAGE_NS}}}TextLine")
    assert [line.findtext(f"{{{PAGE_NS}}}TextEquiv/{{{PAGE_NS}}}Unicode") for line in lines] == ["première", "deuxième"]
    assert lines[0].find(f"{{{PAGE_NS}}}Baseline").attrib["points"] == "8,18 60,18"
    # Word units come only from graph words. TextEquiv line remains a line-level unit.
    assert len(page.findall(f".//{{{PAGE_NS}}}Word")) == 2


def test_pagexml_fractional_coordinates_rejected_without_explicit_quantization():
    graph = exportable_graph()
    with pytest.raises(ExportError, match="nearest_pixel"):
        export_pagexml(graph, image_filename="scan.tif", schema_dir=SCHEMAS)
    quantized = export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel", schema_dir=SCHEMAS)
    assert quantized.accepted
    assert quantized.max_coordinate_error_px == pytest.approx(0.5)


def test_alto_xsd_and_semantic_roundtrip_use_real_word_nodes_only():
    graph = exportable_graph()
    result = export_alto(graph, image_filename="scan-02.tif", physical_img_nr=2, schema_dir=SCHEMAS)
    assert result.accepted and result.xml
    validate_export_xsd(result, schema_dir=SCHEMAS)
    root = ET.fromstring(result.xml)
    page = root.find(f".//{{{ALTO_NS}}}Page")
    assert page is not None
    assert page.attrib["PHYSICAL_IMG_NR"] == "2"
    strings = root.findall(f".//{{{ALTO_NS}}}String")
    assert [x.attrib["CONTENT"] for x in strings] == ["première", "deuxième"]
    assert [x.attrib["ID"] for x in strings] == ["n_W1", "n_W2"]
    assert len(root.findall(f".//{{{ALTO_NS}}}Polygon")) == 5  # block, line, word shapes
    assert len(root.findall(f".//{{{ALTO_NS}}}SP")) == 0  # one word per line


def test_alto_refuses_line_only_transcription_and_word_join_mismatch():
    graph = exportable_graph()
    graph["words"] = []
    graph["coverage"]["words"] = {"status": "unavailable", "expected_count": 0, "available_count": 0, "error_count": 0}
    graph["relations"] = [r for r in graph["relations"] if r["source_id"] not in {"W1", "W2"} and r["target_id"] not in {"W1", "W2"}]
    graph["coverage"]["transcription"] = {"status": "complete", "expected_count": 2, "available_count": 2, "error_count": 0}
    graph["coverage"]["reading_order"] = {"status": "complete", "expected_count": 1, "available_count": 1, "error_count": 0}
    refused = export_alto(graph, image_filename="scan.tif", physical_img_nr=2, schema_dir=SCHEMAS)
    assert not refused.accepted
    assert any(loss.code == "words_required_for_alto" for loss in refused.losses)

    graph = exportable_graph()
    graph["lines"][0]["text"]["value"] = "première,"
    mismatch = export_alto(graph, image_filename="scan.tif", physical_img_nr=2, schema_dir=SCHEMAS)
    assert not mismatch.accepted
    assert any(loss.code == "word_join_mismatch" for loss in mismatch.losses)


def test_best_effort_reports_omitted_articulation_and_keeps_sidecar():
    graph = deepcopy(valid_graph())
    result = export_pagexml(graph, image_filename="scan.tif", mode="best_effort", quantization="nearest_pixel", schema_dir=SCHEMAS)
    assert result.accepted and result.xml
    assert any(loss.code == "article_layer_omitted" and loss.preserved_in_sidecar for loss in result.losses)
    assert json.loads(result.sidecar_json)["articles"] == graph["articles"]


def test_alto_requires_explicit_physical_page_number():
    with pytest.raises(ExportError, match="physical_img_nr"):
        export_alto(exportable_graph(), image_filename="scan.tif", physical_img_nr=float("nan"), schema_dir=SCHEMAS)
    for invalid in (0, -1, 1.5, 10**100):
        with pytest.raises(ExportError, match="positive 32-bit integer"):
            export_alto(exportable_graph(), image_filename="scan.tif", physical_img_nr=invalid, schema_dir=SCHEMAS)


@pytest.mark.parametrize("bad_text", ["cr\rhere", "line\r\nbreak"])
def test_xml_carriage_returns_are_refused_in_text_and_filenames(bad_text):
    graph = exportable_graph()
    graph["lines"][0]["text"]["value"] = bad_text
    with pytest.raises(ExportError, match="carriage return"):
        export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel")
    with pytest.raises(ExportError, match="carriage return"):
        export_alto(graph, image_filename="scan.tif", physical_img_nr=1)
    graph = exportable_graph()
    with pytest.raises(ExportError, match="carriage return"):
        export_alto(graph, image_filename=bad_text, physical_img_nr=1)


def test_page_quantization_and_exclusive_image_edge_are_reported():
    graph = exportable_graph()
    result = export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel")
    assert result.max_coordinate_error_px == pytest.approx(0.5)
    assert any(loss.code == "coordinate_quantized" for loss in result.losses)
    graph["blocks"][0]["polygon"]["points"][1][0] = graph["image"]["width"]
    graph["blocks"][0]["polygon"]["points"][2][0] = graph["image"]["width"]
    result = export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel")
    assert any(loss.code == "exclusive_right_edge" for loss in result.losses)


def test_alto_refuses_available_empty_word_text():
    graph = exportable_graph()
    graph["words"][0]["text"]["value"] = ""
    with pytest.raises(ExportError, match="empty CONTENT"):
        export_alto(graph, image_filename="scan.tif", physical_img_nr=1)


def test_strict_rejects_ambiguous_sibling_order_but_best_effort_records_it():
    graph = exportable_graph()
    graph["relations"] = [r for r in graph["relations"] if r["type"] != "precedes"]
    graph["coverage"]["reading_order"] = {"status": "unavailable", "expected_count": 0, "available_count": 0, "error_count": 0}
    strict = export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel", schema_dir=SCHEMAS)
    assert not strict.accepted
    assert any(loss.code == "ambiguous_order" for loss in strict.losses)
    fallback = export_pagexml(graph, image_filename="scan.tif", mode="best_effort", quantization="nearest_pixel", schema_dir=SCHEMAS)
    assert fallback.accepted
    assert any(loss.code == "ambiguous_order" for loss in fallback.losses)


def test_best_effort_tie_break_preserves_known_precedence():
    graph = {
        "relations": [{"type": "precedes", "source_id": "Z", "target_id": "A"}],
    }
    kinds = {"Z": "line", "A": "line", "B": "line"}
    parents = {}
    losses = []
    order = _ordered_siblings(graph, ["Z", "A", "B"], kinds, parents, "best_effort", losses, "lines/order")
    assert order.index("Z") < order.index("A")
    assert any(loss.code == "ambiguous_order" for loss in losses)


def test_alto_best_effort_uses_ordered_words_for_exact_join_check():
    graph = exportable_graph()
    graph["lines"][0]["text"]["value"] = "x première"
    graph["words"].append({
        "id": "W3", "status": "ok",
        "polygon": {"status": "available", "points": [[30, 9], [38, 9], [38, 19], [30, 19]], "source": "predicted"},
        "text": {"status": "available", "value": "x"},
    })
    graph["relations"].append({"type": "contains", "source_id": "L1", "target_id": "W3", "source": "predicted", "confidence": 1.0})
    graph["relations"] = [r for r in graph["relations"] if not (r["type"] == "precedes" and r["source_id"] == "W1")]
    graph["relations"].append({"type": "precedes", "source_id": "W3", "target_id": "W1", "source": "predicted", "confidence": 1.0})
    graph["coverage"]["words"].update(expected_count=3, available_count=3)
    graph["coverage"]["transcription"].update(expected_count=5, available_count=5)
    result = export_alto(graph, image_filename="scan.tif", physical_img_nr=1, mode="best_effort")
    assert result.accepted
    root = ET.fromstring(result.xml)
    line = root.find(f".//{{{ALTO_NS}}}TextLine[@ID='n_L1']")
    assert [node.attrib["CONTENT"] for node in line.findall(f"{{{ALTO_NS}}}String")] == ["x", "première"]
    assert not any(loss.code == "line_not_exported" for loss in result.losses)


def test_float64_coordinate_round_trips_and_loss_report_names_sidecar_only_fields():
    graph = exportable_graph()
    exact = float("8.123456789012345")
    graph["words"][0]["polygon"]["points"][0][0] = exact
    result = export_alto(graph, image_filename="scan.tif", physical_img_nr=1)
    assert result.accepted
    root = ET.fromstring(result.xml)
    points = root.find(f".//{{{ALTO_NS}}}String[@ID='n_W1']/{{{ALTO_NS}}}Shape/{{{ALTO_NS}}}Polygon").attrib["POINTS"]
    assert float(points.split()[0].split(",")[0]) == exact
    codes = {loss.code for loss in result.losses}
    assert {"geometry_source_in_sidecar", "baseline_band_in_sidecar", "relation_confidence_in_sidecar"} <= codes


def test_page_quantization_refuses_collapsed_polygon():
    graph = exportable_graph()
    graph["blocks"][0]["polygon"]["points"] = [[5.1, 5.1], [5.4, 5.1], [5.4, 5.4], [5.1, 5.4]]
    with pytest.raises(ExportError, match="collapses polygon"):
        export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel")


@pytest.mark.parametrize("bad_text", ["before\x01after", "\ud800"])
def test_xml_illegal_text_is_refused_without_replacement_in_both_formats(bad_text):
    graph = exportable_graph()
    graph["lines"][0]["text"]["value"] = bad_text
    with pytest.raises(ExportError, match="XML 1.0-illegal"):
        export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel")
    with pytest.raises(ExportError, match="XML 1.0-illegal"):
        export_alto(graph, image_filename="scan.tif", physical_img_nr=1)


def test_xml_illegal_context_name_and_sidecar_surrogate_fail_as_export_errors():
    graph = exportable_graph()
    with pytest.raises(ExportError, match="XML 1.0-illegal"):
        export_pagexml(graph, image_filename="scan\x00.tif", quantization="nearest_pixel")
    graph["identity"]["title_key"] = "broken\ud800"
    with pytest.raises(ExportError, match="canonical UTF-8 sidecar"):
        export_alto(graph, image_filename="scan.tif", physical_img_nr=1)


def test_best_effort_reports_each_descendant_omitted_with_ancestor():
    graph = exportable_graph()
    graph["blocks"][0]["category"] = "tableau"
    result = export_pagexml(graph, image_filename="scan.tif", mode="best_effort", quantization="nearest_pixel")
    assert result.accepted
    ancestor_losses = {(loss.path, loss.code) for loss in result.losses if loss.code == "ancestor_omitted"}
    assert ("lines/L1", "ancestor_omitted") in ancestor_losses
    assert ("lines/L2", "ancestor_omitted") in ancestor_losses
    assert ("words/W1", "ancestor_omitted") in ancestor_losses
    assert ("words/W2", "ancestor_omitted") in ancestor_losses


# --- G1 m4 / m5 (2026-10-08) -------------------------------------------------

def test_page_quantization_refuses_polygon_that_becomes_non_simple():
    # Simple and valid before rounding: the notch tip (45,10.4) stays 0.4 px
    # above the bottom edge y=10. Rounding puts it ON that edge (positive area
    # is preserved), which PAGE would then carry as a self-touching contour.
    graph = exportable_graph()
    graph["blocks"][0]["polygon"]["points"] = [
        [10, 10], [80, 10], [80, 40], [46, 40], [45, 10.4], [44, 40], [10, 40]
    ]
    from babaorum.page_graph import validate_page_graph
    validate_page_graph(graph)
    with pytest.raises(ExportError, match="non-simple"):
        export_pagexml(graph, image_filename="scan.tif", quantization="nearest_pixel")


def test_alto_reports_xsd_float32_bbox_attributes():
    graph = exportable_graph()
    exact = export_alto(graph, image_filename="scan.tif", physical_img_nr=1, schema_dir=SCHEMAS)
    assert exact.accepted
    notes = [loss for loss in exact.losses if loss.code == "alto_bbox_xsd_float32"]
    assert len(notes) == 1 and notes[0].severity == "info"
    assert "xsd:float" in notes[0].explanation and "HPOS, VPOS, WIDTH and HEIGHT" in notes[0].explanation
    assert not any(loss.code == "alto_bbox_float32_rounding" for loss in exact.losses)

    graph["words"][0]["polygon"]["points"][0] = [8.1, 9]  # HPOS 8.1 is not float32-exact
    graph["words"][0]["polygon"]["points"][3] = [8.1, 19]
    inexact = export_alto(graph, image_filename="scan.tif", physical_img_nr=1, schema_dir=SCHEMAS)
    assert inexact.accepted
    validate_export_xsd(inexact, schema_dir=SCHEMAS)
    notes = [loss for loss in inexact.losses if loss.code == "alto_bbox_xsd_float32"]
    assert len(notes) == 1 and notes[0].severity == "warning"
    assert [loss.path for loss in inexact.losses if loss.code == "alto_bbox_float32_rounding"] == ["words/W1/polygon"]
