"""Synthetic page integration checks; no OCR or corpus-performance claims."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_curve_sampling import masked_curve_ctc_loss  # noqa: E402
from babaorum.experimental.joint_curved_page import (  # noqa: E402
    CurvedPageTargets,
    TinyCurvedJointPage,
    curved_page_losses,
    match_curved_lines,
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
        baseline_fraction=0.75, num_queries=3, control_points=4,
        channels=8, alphabet_size=4, band_rows=3, spacing_px=0.75,
    )
    options.update(kwargs)
    return TinyCurvedJointPage(**options)


def synthetic_image():
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 64), indexing="ij")
    return (xx + yy.square() + 0.3 * torch.sin(7 * xx + 3 * yy))[None, None]


def controls():
    return torch.tensor([[[x, y] for x in (4.0, 12.0, 20.0, 32.0)] for y in (8.0, 16.0, 24.0)])


def target_page(order=None):
    return CurvedPageTargets(
        baseline_xy=[
            torch.tensor([[4.0, 8.0], [32.0, 8.0]]),
            torch.tensor([[4.0, 16.0], [11.0, 16.0], [32.0, 16.0]]),
            torch.tensor([[4.0, 24.0], [5.0, 24.0], [9.0, 24.0], [24.0, 24.0], [32.0, 24.0]]),
        ],
        height_px=torch.full((3,), 4.0),
        texts=[[1, 2], [2, 1], [1, 3]],
        order=order,
    )


def forward_controlled(model=None, baseline=None, detach_geometry=False):
    model = make_model() if model is None else model
    baseline = controls() if baseline is None else baseline
    pred = model(
        synthetic_image(), baseline_override=baseline,
        height_override=torch.full((3,), 4.0, requires_grad=True),
        detach_geometry=detach_geometry,
    )
    return model, pred


def test_collapsed_query_is_counted_and_repaired_while_other_queries_keep_ctc():
    baseline = controls()
    baseline[1] = torch.tensor([18.0, 16.0])
    baseline.requires_grad_()
    model, pred = forward_controlled(baseline=baseline)
    pred.presence_logits.retain_grad()
    targets = target_page()
    result = curved_page_losses(pred, targets)
    assert result.matching == (0, 1, 2)
    assert result.line_statuses == ("included", "excluded_degenerate_curve", "included")
    for name, expected in (
        ("matched", 3), ("included_ctc", 2), ("excluded_ctc", 1),
        ("excluded_degenerate_curve", 1), ("regular_geometry_lines", 2),
        ("repair_geometry_lines", 1), ("ctc_denominator", 3),
    ):
        assert result.counts[name] == expected
    assert result.counts["ground_truth_tokens"] == 6
    assert result.counts["supported_target_tokens"] == 4
    assert result.counts["excluded_target_tokens"] == 2
    expected_ctc_sum = sum(
        masked_curve_ctc_loss(pred.queries[index].ctc_logits, pred.queries[index].grid, targets.texts[index]).loss
        for index in (0, 2)
    )
    torch.testing.assert_close(result.diagnostics["ctc_signal"], expected_ctc_sum / 3)
    assert result.diagnostics["ctc_exclusion_charge"] == pytest.approx(256 * math.log(4) / 3)
    assert result.diagnostics["ctc_included_mean"] == pytest.approx((expected_ctc_sum / 2).item())
    assert result.diagnostics["ctc_charge_rule"] == "max_samples_log_alphabet"
    torch.testing.assert_close(
        result.losses["ctc"], result.diagnostics["ctc_signal"] + result.diagnostics["ctc_exclusion_charge"]
    )
    assert not torch.allclose(result.diagnostics["ctc_signal"], expected_ctc_sum / 2)
    assert result.losses["repair"].item() > 0
    assert all(torch.isfinite(value) for value in result.losses.values())
    result.losses["total"].backward()
    assert pred.presence_logits.grad is not None and pred.presence_logits.grad[1] < 0
    assert baseline.grad is not None
    repaired = baseline.detach()[1] - 0.1 * baseline.grad[1]
    assert (torch.linalg.vector_norm(repaired[1:] - repaired[:-1], dim=-1) > 0).all()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.ctc_head.parameters())


def test_explicit_near_collapse_cutoff_routes_tiny_nonzero_segments_to_repair():
    baseline = controls()
    baseline[1, :, 0] = torch.tensor([5.0 + index * 1e-6 for index in range(4)])
    assert (torch.linalg.vector_norm(baseline[1, 1:] - baseline[1, :-1], dim=-1) > 0).all()
    baseline.requires_grad_()
    model, pred = forward_controlled(baseline=baseline)
    assert model.min_segment_length_px == 1e-4
    assert pred.queries[1].status == "degenerate_curve"
    assert pred.queries[1].failure_reason == "near_degenerate_predicted_segment"
    result = curved_page_losses(pred, target_page())
    assert result.counts["included_ctc"] == 2
    assert result.counts["excluded_degenerate_curve"] == 1
    assert result.counts["repair_geometry_lines"] == 1
    result.losses["repair"].backward()
    assert baseline.grad is not None and baseline.grad[1].abs().sum() > 0


def test_normal_model_parameterization_keeps_declared_minimum_control_increment():
    model = make_model()
    pred = model(synthetic_image())
    # This is the synthetic model's declared constraint, not a corpus criterion.
    assert model.min_increment_px == 0.125
    increments = pred.baseline_xy[:, 1:, 0] - pred.baseline_xy[:, :-1, 0]
    assert (increments >= 0.125 - 1e-5).all()


@pytest.mark.parametrize(
    "fault,expected_status,count_key",
    [
        ("outside", "excluded_invalid_geometry", "excluded_invalid_geometry"),
        ("cap", "excluded_sampling_cap_exceeded", "excluded_sampling_cap_exceeded"),
        ("short", "excluded_infeasible_length", "excluded_infeasible_length"),
    ],
)
def test_one_unusable_query_does_not_abort_other_lines(fault, expected_status, count_key):
    baseline = controls()
    targets = target_page()
    if fault == "outside":
        baseline[0, :, 1] = 1.0
    elif fault == "cap":
        baseline[0, :, 0] = torch.tensor([4.0, 22.0, 40.0, 60.0])
    else:
        baseline[0, :, 0] = torch.tensor([4.0, 4.06, 4.12, 4.2])
        targets.texts[0] = [1, 1]
    baseline.requires_grad_()
    _, pred = forward_controlled(make_model(max_samples=48), baseline=baseline)
    result = curved_page_losses(pred, targets)
    assert result.counts["matched"] == 3
    assert result.counts["included_ctc"] == 2
    assert result.counts["excluded_ctc"] == 1
    assert result.counts[count_key] == 1
    assert result.line_statuses[0] == expected_status
    assert all(torch.isfinite(value) for value in result.losses.values())
    assert result.losses["ctc"].item() > 0
    result.losses["total"].backward()
    assert baseline.grad is not None and baseline.grad[0].abs().sum() > 0


def double_cap_case(reverse=False):
    baseline = controls()
    baseline[0, :, 0] = torch.tensor([4.0, 204.0, 404.0, 604.0])
    if reverse:
        baseline[0] = baseline[0].flip(0)
    baseline.requires_grad_()
    _, pred = forward_controlled(baseline=baseline)
    targets = target_page()
    # Equal GT heights keep the intended assignment independent of the fault.
    targets.height_px = torch.full((3,), 2.0)
    return baseline, pred, curved_page_losses(pred, targets)


def test_query_exceeding_both_sampling_caps_retains_endpoint_and_height_repair():
    baseline, pred, result = double_cap_case()
    assert result.matching == (0, 1, 2)
    assert result.line_statuses == ("excluded_sampling_cap_exceeded", "included", "included")
    assert result.geometry_statuses == ("repair_cap", "regular", "regular")
    for name, expected in (
        ("included_ctc", 2), ("excluded_ctc", 1), ("excluded_sampling_cap_exceeded", 1),
        ("excluded_geometry_sampling_cap", 1), ("repair_cap_geometry_lines", 1),
        ("regular_geometry_lines", 2), ("matched", 3),
    ):
        assert result.counts[name] == expected
    assert result.losses["repair"].item() > 0
    assert result.diagnostics["ctc_signal"].item() > 0
    assert all(torch.isfinite(value) for value in result.losses.values())
    result.losses["repair"].backward()
    assert baseline.grad is not None and baseline.grad[0, -1, 0] > 0
    assert pred.height_px.grad is not None and pred.height_px.grad[0] > 0
    assert torch.count_nonzero(baseline.grad[0, 1:-1]) == 0  # Declared endpoint-only fallback.
    before = torch.linalg.vector_norm(baseline.detach()[0, 1:] - baseline.detach()[0, :-1], dim=-1).sum()
    repaired = baseline.detach()[0] - 0.1 * baseline.grad[0]
    after = torch.linalg.vector_norm(repaired[1:] - repaired[:-1], dim=-1).sum()
    assert after < before


def test_cap_repair_keeps_endpoint_order_instead_of_matching_reversed_endpoints():
    _, _, forward = double_cap_case()
    _, _, reversed_curve = double_cap_case(reverse=True)
    assert reversed_curve.counts["repair_cap_geometry_lines"] == 1
    assert reversed_curve.losses["repair"] > forward.losses["repair"]


@pytest.mark.parametrize("detach_geometry", [False, True])
def test_page_ctc_detaches_coordinates_but_preserves_encoder_and_reader(detach_geometry):
    model = make_model()
    pred = model(synthetic_image(), detach_geometry=detach_geometry)
    baseline = pred.baseline_xy
    baseline.retain_grad()
    pred.height_px.retain_grad()
    pred.feature_map.retain_grad()
    result = curved_page_losses(pred, target_page())
    assert result.counts["included_ctc"] == 3
    assert result.counts["repair_geometry_lines"] == 0
    assert result.losses["repair"].item() == 0
    result.losses["ctc"].backward()
    for parameter in (baseline, pred.height_px):
        if detach_geometry:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        else:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert pred.feature_map.grad is not None and pred.feature_map.grad.abs().sum() > 0
    for module in (model.encoder, model.ctc_head):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and sum(gradient.abs().sum().item() for gradient in gradients) > 0
    geometry_gradients = [parameter.grad for parameter in model.geometry_head.parameters() if parameter.grad is not None]
    if detach_geometry:
        assert not geometry_gradients or all(torch.count_nonzero(gradient) == 0 for gradient in geometry_gradients)
    else:
        assert geometry_gradients and sum(gradient.abs().sum().item() for gradient in geometry_gradients) > 0


def test_variable_target_vertices_and_permuted_order_preserve_matching_and_loss():
    baseline = controls()
    baseline[1, :, 1] = torch.tensor([16.0, 15.0, 15.0, 16.0])
    _, pred = forward_controlled(baseline=baseline)
    order = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    targets = target_page(order=order)
    targets.baseline_xy[1] = torch.tensor(
        [[4.0, 16.0], [8.0, 15.5], [12.0, 15.0], [20.0, 15.0], [32.0, 16.0]]
    )
    permutation = [2, 0, 1]
    permuted = CurvedPageTargets(
        baseline_xy=[targets.baseline_xy[index] for index in permutation],
        height_px=targets.height_px[permutation],
        texts=[targets.texts[index] for index in permutation],
        order=order[permutation][:, permutation],
    )
    assert match_curved_lines(pred, targets) == (0, 1, 2)
    assert match_curved_lines(pred, permuted) == (2, 0, 1)
    first, second = curved_page_losses(pred, targets), curved_page_losses(pred, permuted)
    assert first.counts == second.counts
    for name in first.losses:
        torch.testing.assert_close(first.losses[name], second.losses[name])


def test_empty_page_trains_no_object_without_ctc_or_geometry_support():
    _, pred = forward_controlled()
    pred.presence_logits.retain_grad()
    result = curved_page_losses(pred, CurvedPageTargets([], torch.empty(0), []))
    assert result.matching == ()
    assert result.counts["matched"] == 0
    assert result.counts["unmatched_queries"] == 3
    assert result.counts["included_ctc"] == 0 and result.counts["excluded_ctc"] == 0
    assert all(status.startswith("unmatched:") for status in result.line_statuses)
    for name in ("ctc", "geometry", "repair", "order"):
        assert result.losses[name].item() == 0
    result.losses["total"].backward()
    assert pred.presence_logits.grad is not None and (pred.presence_logits.grad > 0).all()


@pytest.mark.parametrize("malformation", ["zero_length", "blank_label", "nonfinite"])
def test_malformed_ground_truth_is_rejected_even_with_collapsed_prediction(malformation):
    baseline = controls()
    baseline[1] = torch.tensor([18.0, 16.0])
    _, pred = forward_controlled(baseline=baseline)
    targets = target_page()
    if malformation == "zero_length":
        targets.baseline_xy[1] = torch.tensor([[18.0, 16.0], [18.0, 16.0]])
    elif malformation == "blank_label":
        targets.texts[1] = [0]
    else:
        targets.baseline_xy[1][0, 0] = float("nan")
    with pytest.raises(ValueError):
        curved_page_losses(pred, targets)
