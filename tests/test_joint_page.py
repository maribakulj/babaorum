"""Small synthetic CPU checks; these establish mechanisms, not OCR performance."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_page import (  # noqa: E402
    PageTargets,
    TinyJointPage,
    export_page,
    match_lines,
    page_losses,
)


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(**kwargs):
    torch.manual_seed(17)
    options = dict(
        num_queries=3,
        channels=8,
        alphabet_size=4,
        band_rows=3,
        sequence_steps=12,
        baseline_fraction=0.75,
    )
    options.update(kwargs)
    return TinyJointPage(**options)


def synthetic_image():
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 24), torch.linspace(0, 1, 32), indexing="ij"
    )
    return (xx + yy.square() + 0.3 * torch.sin(7 * xx + 3 * yy))[None, None]


def targets_for(pred, indices=(2, 0), order=None):
    return PageTargets(
        baseline_xy=pred.baseline_xy[list(indices)].detach().clone(),
        height_px=pred.height_px[list(indices)].detach().clone(),
        texts=[[1, 2], [2, 1]][: len(indices)],
        order=order,
    )


def controlled_prediction():
    pred = make_model()(synthetic_image())
    baseline = torch.tensor(
        [[[3.0, 6.0], [28.0, 6.0]],
         [[3.0, 12.0], [28.0, 12.0]],
         [[3.0, 18.0], [28.0, 18.0]]],
        requires_grad=True,
    )
    steps = pred.ctc_logits.shape[0]
    xx = torch.linspace(3.0, 28.0, steps)[None, None].expand(3, 3, -1)
    yy = baseline[:, :1, 1, None] + torch.tensor([-3.0, -1.0, 1.0])[None, :, None]
    pixel_xy = torch.stack((xx, yy.expand(-1, -1, steps)), -1)
    return replace(
        pred,
        baseline_xy=baseline,
        height_px=torch.full((3,), 4.0, requires_grad=True),
        presence_logits=torch.zeros(3, requires_grad=True),
        order_logits=torch.zeros(3, 3, requires_grad=True),
        pixel_xy=pixel_xy,
    )


@pytest.mark.parametrize("detach_geometry", [False, True])
def test_ctc_geometry_ablation_preserves_shared_encoder_gradient(detach_geometry):
    model = make_model()
    pred = model(synthetic_image(), detach_geometry=detach_geometry)
    pred.baseline_xy.retain_grad()
    pred.height_px.retain_grad()
    pred.feature_map.retain_grad()
    targets = targets_for(pred)
    assignment = match_lines(pred, targets)
    unmatched = sorted(set(range(3)).difference(assignment))
    page_losses(pred, targets)["ctc"].backward()

    assert pred.feature_map.grad is not None
    assert torch.isfinite(pred.feature_map.grad).all()
    assert pred.feature_map.grad.abs().sum().item() > 0
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert encoder_grads
    assert all(torch.isfinite(gradient).all() for gradient in encoder_grads)
    assert sum(gradient.abs().sum().item() for gradient in encoder_grads) > 0
    for coordinates in (pred.baseline_xy, pred.height_px):
        if detach_geometry:
            assert coordinates.grad is None or torch.count_nonzero(coordinates.grad) == 0
        else:
            assert coordinates.grad is not None
            assert torch.isfinite(coordinates.grad).all()
            assert coordinates.grad.abs().sum().item() > 0
            # CTC alone has no direct coordinate signal for an unassigned query.
            assert torch.count_nonzero(coordinates.grad[unmatched]) == 0


def test_assignment_and_losses_are_invariant_to_target_permutation():
    pred = controlled_prediction()
    order = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    targets = targets_for(pred, order=order)
    permutation = torch.tensor([1, 0])
    permuted = PageTargets(
        baseline_xy=targets.baseline_xy[permutation],
        height_px=targets.height_px[permutation],
        texts=[targets.texts[index] for index in permutation.tolist()],
        order=order[permutation][:, permutation],
    )

    assert match_lines(pred, targets) == (2, 0)
    assert match_lines(pred, permuted) == (0, 2)
    losses = page_losses(pred, targets)
    permuted_losses = page_losses(pred, permuted)
    for name in ("ctc", "geometry", "presence", "order", "total"):
        torch.testing.assert_close(losses[name], permuted_losses[name])


def test_order_supervision_follows_assignment_not_query_index():
    pred = controlled_prediction()
    targets = targets_for(pred, order=torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    page_losses(pred, targets)["order"].backward()
    gradient = pred.order_logits.grad
    assert gradient is not None
    assert gradient[2, 0] < 0  # GT first line is query 2, before query 0.
    assert gradient[0, 2] > 0
    assert torch.count_nonzero(gradient) == 2


def test_unknown_order_relations_are_masked_instead_of_negative_labels():
    pred = controlled_prediction()
    targets = targets_for(pred, order=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
    page_losses(pred, targets)["order"].backward()
    gradient = pred.order_logits.grad
    assert gradient is not None
    assert gradient[2, 0] < 0
    assert gradient[0, 2] == 0
    assert torch.count_nonzero(gradient) == 1


def test_presence_trains_matched_and_unmatched_queries():
    pred = controlled_prediction()
    page_losses(pred, targets_for(pred))["presence"].backward()
    gradient = pred.presence_logits.grad
    assert gradient is not None
    assert gradient[0] < 0 and gradient[2] < 0
    assert gradient[1] > 0  # The unused slot receives a no-object label.


def test_empty_page_has_finite_loss_and_trains_all_queries_as_no_object():
    pred = controlled_prediction()
    targets = PageTargets(torch.empty(0, 2, 2), torch.empty(0), [])
    assert match_lines(pred, targets) == ()
    losses = page_losses(pred, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    for name in ("ctc", "geometry", "order"):
        assert losses[name].item() == 0
        assert losses[name].requires_grad
    assert losses["presence"].item() > 0
    losses["total"].backward()
    assert pred.presence_logits.grad is not None
    assert (pred.presence_logits.grad > 0).all()


def test_repeated_ctc_target_requires_a_blank_timestep_and_is_not_hidden():
    pred = make_model(sequence_steps=2)(synthetic_image())
    targets = PageTargets(
        pred.baseline_xy[:1].detach(), pred.height_px[:1].detach(), [[1, 1]]
    )
    with pytest.raises(ValueError):
        page_losses(pred, targets)


@pytest.mark.parametrize("text", [[0], [4], [-1], [1.5]])
def test_ctc_rejects_blank_out_of_alphabet_and_noninteger_targets(text):
    pred = make_model()(synthetic_image())
    targets = PageTargets(
        pred.baseline_xy[:1].detach(), pred.height_px[:1].detach(), [text]
    )
    with pytest.raises(ValueError):
        page_losses(pred, targets)


def test_more_target_lines_than_queries_is_an_explicit_error():
    pred = make_model(num_queries=1)(synthetic_image())
    targets = PageTargets(
        pred.baseline_xy.detach().repeat(2, 1, 1),
        pred.height_px.detach().repeat(2),
        [[1], [2]],
    )
    with pytest.raises(ValueError):
        match_lines(pred, targets)


def test_baseline_fraction_requires_an_explicit_convention():
    with pytest.raises(TypeError):
        TinyJointPage()


@pytest.mark.parametrize("fraction", [0.25, 0.75])
def test_requested_baseline_fraction_sets_asymmetric_band(fraction):
    pred = make_model(baseline_fraction=fraction)(synthetic_image())
    assert pred.baseline_xy.shape == (3, 2, 2)
    assert pred.pixel_xy.shape == (3, 3, 12, 2)
    # At the first baseline endpoint, project row displacements on the normal.
    tangent = pred.baseline_xy[:, 1] - pred.baseline_xy[:, 0]
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
    displacements = pred.pixel_xy[:, :, 0] - pred.baseline_xy[:, None, 0]
    measured = (displacements * normal[:, None]).sum(-1) / pred.height_px[:, None]
    expected = torch.linspace(-fraction, 1 - fraction, 3).expand(3, -1)
    torch.testing.assert_close(measured, expected, atol=2e-6, rtol=2e-6)


EXPORT_IDENTITY = dict(
    image_sha256="a" * 64,
    page_key="fixture:joint_page/page1",
    document_key="fixture:joint_page",
)


def export_prediction_with_edges(edges):
    pred = controlled_prediction()
    order = torch.full((3, 3), -12.0)
    for source, target in edges:
        order[source, target] = 12.0
    logits = torch.full_like(pred.ctc_logits, -12.0)
    logits[:, :, 0] = 12.0
    logits[1:3, :, 0] = -12.0
    logits[1:3, :, 1] = 12.0
    logits[5:7, :, 0] = -12.0
    logits[5:7, :, 2] = 12.0
    return replace(
        pred,
        presence_logits=torch.full((3,), 12.0),
        order_logits=order,
        ctc_logits=logits,
    )


def test_export_is_deterministic_and_keeps_original_pixels_without_gt_or_words():
    pred = export_prediction_with_edges([(2, 0), (0, 1), (2, 1)])
    result = export_page(pred, **EXPORT_IDENTITY)
    assert result == export_page(pred, **EXPORT_IDENTITY)
    json.dumps(result, allow_nan=False)
    assert result["image_sha256"] == EXPORT_IDENTITY["image_sha256"]
    assert result["page_key"] == EXPORT_IDENTITY["page_key"]
    assert result["document_key"] == EXPORT_IDENTITY["document_key"]
    assert result["coordinate_frame"]["width"] == 32
    assert result["coordinate_frame"]["height"] == 24
    assert result["coordinate_frame"]["coordinate_convention"] == (
        "pixel_centers_0_width_minus_1_height_minus_1"
    )
    assert result["word_localization_available"] is False
    assert result["baseline_fraction"] == 0.75
    assert len(result["lines"]) == 3
    for line in result["lines"]:
        query = line["query_index"]
        assert line["baseline_xy"] == pred.baseline_xy[query].tolist()
        assert line["height_px"] == pred.height_px[query].item()
        assert line["ctc_token_ids"] == [1, 2]
        assert line["word_boxes"] is None
        polygon = torch.tensor(line["polygon_xy"])
        assert polygon[:, 0].min().item() == 3.0
        assert polygon[:, 0].max().item() == 28.0
        assert polygon[:, 1].min().item() == pred.baseline_xy[query, 0, 1].item() - 3.0
        assert polygon[:, 1].max().item() == pred.baseline_xy[query, 0, 1].item() + 1.0
    assert result["order"]["status"] == "unique"
    assert result["order"]["topological_order"] == ["q2", "q0", "q1"]


def test_export_detects_cycle_instead_of_inventing_reading_order():
    pred = export_prediction_with_edges([(0, 1), (1, 2), (2, 0)])
    result = export_page(pred, **EXPORT_IDENTITY)
    assert result["order"]["status"] == "cycle"
    assert result["order"]["topological_order"] is None


def test_export_reports_ambiguous_order_and_valid_empty_prediction():
    pred = export_prediction_with_edges([])
    result = export_page(pred, **EXPORT_IDENTITY)
    assert result["order"]["status"] == "ambiguous"
    assert result["order"]["topological_order"] is None
    assert result["order"]["candidate_topological_order"] == ["q0", "q1", "q2"]
    empty = export_page(
        replace(pred, presence_logits=torch.full((3,), -12.0)), **EXPORT_IDENTITY
    )
    assert empty["lines"] == []
    assert empty["order"]["status"] == "empty"
    assert empty["order"]["edges"] == []


def test_export_rejects_outside_baseline_even_when_old_sampling_grid_is_inside():
    pred = controlled_prediction()
    baseline = pred.baseline_xy.detach().clone()
    baseline[0, 0, 0] = -1.0
    assert (pred.pixel_xy[..., 0] >= 0).all()
    assert (pred.pixel_xy[..., 0] < 32).all()
    with pytest.raises(ValueError, match="prediction baselines"):
        export_page(replace(pred, baseline_xy=baseline), **EXPORT_IDENTITY)


def test_losses_reject_target_band_outside_page_despite_inside_baseline():
    pred = controlled_prediction()
    targets = PageTargets(
        baseline_xy=torch.tensor([[[3.0, 1.0], [28.0, 1.0]]]),
        height_px=torch.tensor([4.0]),
        texts=[[1]],
    )
    # The baseline y=1 is inside, but the required 75%-above band starts at y=-2.
    with pytest.raises(ValueError, match="target sampling bands"):
        page_losses(pred, targets)


def test_export_rejects_stale_grid_even_when_both_geometries_are_inside():
    pred = controlled_prediction()
    baseline = pred.baseline_xy.detach().clone()
    baseline[..., 1] += 1.0
    with pytest.raises(ValueError, match="sampled coordinates must match"):
        export_page(replace(pred, baseline_xy=baseline), **EXPORT_IDENTITY)


def test_export_filters_order_edges_to_selected_queries():
    pred = export_prediction_with_edges([(0, 1), (1, 2), (2, 0)])
    pred = replace(pred, presence_logits=torch.tensor([12.0, -12.0, 12.0]))
    result = export_page(pred, **EXPORT_IDENTITY)
    assert {line["id"] for line in result["lines"]} == {"q0", "q2"}
    assert [(edge["source"], edge["target"]) for edge in result["order"]["edges"]] == [
        ("q2", "q0")
    ]
    assert result["order"]["status"] == "unique"
    assert result["order"]["topological_order"] == ["q2", "q0"]


@pytest.mark.parametrize("bad_hash", ["", "a" * 63, "g" * 64, "a" * 65])
def test_export_rejects_malformed_image_hash(bad_hash):
    identity = {**EXPORT_IDENTITY, "image_sha256": bad_hash}
    with pytest.raises(ValueError):
        export_page(controlled_prediction(), **identity)


@pytest.mark.parametrize("key", ["page_key", "document_key"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_export_rejects_missing_identity(key, value):
    identity = {**EXPORT_IDENTITY, key: value}
    with pytest.raises(ValueError):
        export_page(controlled_prediction(), **identity)


@pytest.mark.parametrize("name", ["presence_threshold", "order_threshold"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_export_rejects_invalid_probability_thresholds(name, value):
    with pytest.raises(ValueError):
        export_page(controlled_prediction(), **EXPORT_IDENTITY, **{name: value})
