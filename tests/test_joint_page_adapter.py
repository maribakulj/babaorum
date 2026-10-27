"""Synthetic contract integration; image binding and OCR quality remain separate."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import io
import json

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from babaorum.experimental.joint_curved_page import TinyCurvedJointPage  # noqa: E402
from babaorum.experimental.joint_page_adapter import (  # noqa: E402
    adapt_curved_page,
    geometry_predictions_from_graph,
)
from babaorum.line_geometry import GroundTruthLine, evaluate_line_geometry  # noqa: E402
from babaorum.page_graph import validate_page_graph  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def png_and_tensor(size=(64, 32)):
    width, height = size
    pixels = bytes((x + 3 * y) % 256 for y in range(height) for x in range(width))
    image = Image.frombytes("L", size, pixels)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    tensor = torch.tensor(list(pixels), dtype=torch.float32).reshape(1, 1, height, width) / 255
    return stream.getvalue(), tensor


def controls():
    return torch.tensor([[[x, y] for x in (4.0, 12.0, 20.0, 32.0)] for y in (8.0, 16.0, 24.0)])


def prediction(baseline=None, max_samples=30):
    png, image = png_and_tensor()
    torch.manual_seed(17)
    model = TinyCurvedJointPage(baseline_fraction=0.75, spacing_px=1.5, max_samples=max_samples)
    pred = model(
        image, baseline_override=controls() if baseline is None else baseline,
        height_override=torch.full((3,), 4.0),
    )
    return png, replace(pred, presence_logits=torch.full((3,), 10.0))


IDENTITY = dict(
    page_key="fixture:adapter/page1", document_key=None, title_key=None,
    run_id="synthetic_adapter_test",
    components=[dict(name="tiny_curved_joint", code_revision="synthetic-test", config_sha256=None,
                     checkpoint_sha256=None, license=None)],
    presence_threshold=0.5,
)
ALPHABET = ("", "a", "b", "c")


def adapt(pred, png, **kwargs):
    options = {**IDENTITY, "alphabet": ALPHABET, **kwargs}
    return adapt_curved_page(pred, png_bytes=png, **options)


def force_blank_logits(pred):
    queries = []
    for query in pred.queries:
        if query.ctc_logits is None:
            queries.append(query)
            continue
        logits = torch.full_like(query.ctc_logits, -10.0)
        logits[..., 0] = 10.0
        queries.append(replace(query, ctc_logits=logits))
    return replace(pred, queries=tuple(queries))


def test_graph_roundtrip_translates_centers_once_and_preserves_lengths_and_band_extents():
    png, pred = prediction()
    result = adapt(pred, png)
    graph = result.graph
    validate_page_graph(graph)
    roundtrip = json.loads(json.dumps(graph, allow_nan=False))
    validate_page_graph(roundtrip)
    json.dumps(result.diagnostics, allow_nan=False)
    assert graph == roundtrip
    assert graph["image"] == {"sha256": hashlib.sha256(png).hexdigest(), "width": 64, "height": 32}
    assert graph["identity"]["document_key"] is None and graph["identity"]["title_key"] is None
    assert graph["coordinate_convention"] == "continuous_pixel_edges_0_width_height"
    for index, line in enumerate(graph["lines"]):
        converted = torch.tensor(line["baseline"]["points"])
        torch.testing.assert_close(converted, pred.baseline_xy[index] + 0.5)
        torch.testing.assert_close(converted[1:] - converted[:-1], pred.baseline_xy[index, 1:] - pred.baseline_xy[index, :-1])
        assert line["baseline"]["source"] == "predicted"
        assert line["baseline_band"] == {
            "status": "available", "normal_negative_px": 3.0, "normal_positive_px": 1.0, "source": "predicted"
        }
        assert line["polygon"] == {"status": "missing", "points": None, "source": None}
        assert line["status"] == "partial"
    assert graph["words"] == graph["blocks"] == graph["articles"] == []
    assert graph["coverage"]["lines"]["status"] == "unknown"
    assert graph["coverage"]["lines"]["expected_count"] is None
    assert "diagnostics" not in graph
    assert result.diagnostics["prediction_input_content_verified"] is False
    assert result.diagnostics["image_binding"] == "caller_declared_prediction_link"


def test_collapsed_selected_query_stays_in_metric_denominator():
    baseline = controls()
    baseline[1] = torch.tensor([18.0, 16.0])
    png, pred = prediction(baseline)
    result = adapt(pred, png)
    graph = result.graph
    validate_page_graph(graph)
    assert len(graph["lines"]) == 3
    assert graph["lines"][1]["status"] == "error"
    assert graph["lines"][1]["baseline"] == {"status": "error", "points": None, "source": None}
    assert graph["coverage"]["lines"]["status"] == "error"
    assert graph["coverage"]["lines"]["error_count"] == 1
    assert graph["coverage"]["lines"]["expected_count"] is None
    predictions = geometry_predictions_from_graph(graph)
    assert len(predictions) == 3
    assert predictions[1].geometry_status == "error" and predictions[1].baseline is None
    truth = [GroundTruthLine(f"gt{index}", (line + 0.5).tolist(), 4.0, "ground_truth") for index, line in enumerate(controls())]
    measured = evaluate_line_geometry(truth, predictions)
    assert measured.n_pred == 3 and measured.matched_count == 2
    assert measured.pred_geometry_unavailable_count == 1
    assert measured.precision == pytest.approx(2 / 3)


def test_sampling_cap_preserves_evaluable_baseline_without_inventing_a_band():
    baseline = controls()
    baseline[0, :, 0] = torch.tensor([4.0, 22.0, 40.0, 60.0])
    png, pred = prediction(baseline)
    assert pred.queries[0].status == "sampling_cap_exceeded"
    graph = adapt(pred, png).graph
    line = graph["lines"][0]
    assert line["baseline"]["status"] == "available"
    assert line["baseline_band"]["status"] == "missing"
    assert line["text"] == {"status": "error", "value": None}
    assert line["status"] == "partial"
    assert geometry_predictions_from_graph(graph)[0].geometry_status == "available"


def test_support_failure_preserves_baseline_but_marks_band_and_text_as_errors():
    baseline = controls()
    baseline[0, :, 1] = 1.0
    png, pred = prediction(baseline)
    assert pred.queries[0].status == "invalid_geometry"
    graph = adapt(pred, png).graph
    line = graph["lines"][0]
    assert line["baseline"]["status"] == "available"
    assert line["baseline_band"]["status"] == "error"
    assert line["text"] == {"status": "error", "value": None}
    assert geometry_predictions_from_graph(graph)[0].geometry_status == "available"


def test_invalid_height_cannot_reuse_old_valid_grid_to_claim_available_text():
    png, pred = prediction()
    height = pred.height_px.clone()
    height[0] = 0.0
    assert pred.queries[0].grid.line_valid
    graph = adapt(replace(pred, height_px=height), png).graph
    line = graph["lines"][0]
    assert line["baseline"]["status"] == "available"
    assert line["baseline_band"]["status"] == "error"
    assert line["text"] == {"status": "error", "value": None}
    assert line["status"] == "partial"


def test_missing_alphabet_is_distinct_from_available_empty_decoding():
    png, pred = prediction()
    pred = force_blank_logits(pred)
    missing = adapt(pred, png, alphabet=None).graph
    blank = adapt(pred, png).graph
    assert all(line["text"] == {"status": "missing", "value": None} for line in missing["lines"])
    assert all(line["text"] == {"status": "available", "value": ""} for line in blank["lines"])
    assert missing["coverage"]["transcription"]["status"] == "unavailable"
    assert blank["coverage"]["transcription"]["available_count"] == 3
    assert blank["coverage"]["transcription"]["expected_count"] is None


def test_short_intrinsically_valid_query_is_decoded_without_consulting_any_target():
    baseline = controls()
    baseline[0, :, 0] = torch.tensor([4.0, 4.06, 4.12, 4.2])
    png, pred = prediction(baseline)
    assert pred.queries[0].ctc_logits.shape[0] == 2
    graph = adapt(force_blank_logits(pred), png).graph
    assert graph["lines"][0]["text"] == {"status": "available", "value": ""}


def test_explicit_alphabet_preserves_ctc_repeats_separated_by_blank():
    png, pred = prediction()
    pred = force_blank_logits(pred)
    logits = pred.queries[0].ctc_logits.clone()
    for step, label in enumerate([1, 1, 0, 1, 2]):
        logits[step, 0] = -10.0
        logits[step, 0, label] = 10.0
    queries = (replace(pred.queries[0], ctc_logits=logits), *pred.queries[1:])
    graph = adapt(replace(pred, queries=queries), png).graph
    assert graph["lines"][0]["text"] == {"status": "available", "value": "aab"}
    assert graph["words"] == []


def test_presence_selection_keeps_stable_query_ids_and_an_unknown_denominator():
    png, pred = prediction()
    pred = replace(pred, presence_logits=torch.tensor([-10.0, 0.0, 10.0]))
    selected = adapt(pred, png, presence_threshold=0.5).graph
    narrower = adapt(pred, png, presence_threshold=0.9).graph
    assert [line["id"] for line in selected["lines"]] == ["line_q001", "line_q002"]
    assert [line["id"] for line in narrower["lines"]] == ["line_q002"]
    assert selected["coverage"]["lines"]["expected_count"] is None
    none = adapt(replace(pred, presence_logits=torch.full((3,), -10.0)), png).graph
    assert none["lines"] == [] and geometry_predictions_from_graph(none) == ()
    assert none["coverage"]["lines"]["status"] == "unknown"
    assert none["coverage"]["lines"]["available_count"] == 0


def test_cyclic_order_logits_remain_diagnostic_and_do_not_sort_lines_by_geometry():
    baseline = controls()[[2, 0, 1]]
    png, pred = prediction(baseline)
    order = torch.full((3, 3), -10.0)
    order[0, 1] = order[1, 2] = order[2, 0] = 10.0
    pred = replace(pred, order_logits=order)
    result = adapt(pred, png)
    assert result.graph["relations"] == []
    assert [line["id"] for line in result.graph["lines"]] == ["line_q000", "line_q001", "line_q002"]
    assert [line["baseline"]["points"][0][1] for line in result.graph["lines"]] == [24.5, 8.5, 16.5]
    assert result.diagnostics["order_logits"] == order.tolist()


def test_metric_conversion_rejects_ground_truth_provenance_in_prediction_graph():
    png, pred = prediction()
    graph = copy.deepcopy(adapt(pred, png).graph)
    graph["lines"][0]["baseline"]["source"] = "ground_truth"
    validate_page_graph(graph)  # Generic graph permits it, prediction conversion must not.
    with pytest.raises(ValueError):
        geometry_predictions_from_graph(graph)


def test_baseline_bounds_apply_after_translation_without_clamping():
    baseline = controls()
    baseline[0, :, 0] = torch.tensor([-0.5, 10.0, 20.0, 63.5])
    png, pred = prediction(baseline, max_samples=256)
    line = adapt(pred, png).graph["lines"][0]
    assert line["baseline"]["status"] == "available"
    assert line["baseline"]["points"][0][0] == 0
    assert line["baseline"]["points"][-1][0] == 64
    baseline[0, 0, 0] = -0.6
    png, outside = prediction(baseline, max_samples=256)
    rejected = adapt(outside, png).graph["lines"][0]
    assert rejected["baseline"] == {"status": "error", "points": None, "source": None}


@pytest.mark.parametrize("kind", ["not_png", "header_only", "wrong_dimensions"])
def test_adapter_verifies_real_png_content_and_declared_dimensions(kind):
    png, pred = prediction()
    if kind == "not_png":
        invalid = b"this is not a PNG"
    elif kind == "header_only":
        invalid = png[:33]
    else:
        invalid, _ = png_and_tensor((65, 32))
    with pytest.raises(ValueError):
        adapt(pred, invalid)


@pytest.mark.parametrize("alphabet", [("", "a", "b"), ("blank", "a", "b", "c"), ("", "a", None, "c")])
def test_invalid_alphabet_is_rejected(alphabet):
    png, pred = prediction()
    with pytest.raises(ValueError):
        adapt(pred, png, alphabet=alphabet)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_presence_threshold_is_rejected(threshold):
    png, pred = prediction()
    with pytest.raises(ValueError):
        adapt(pred, png, presence_threshold=threshold)
