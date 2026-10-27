"""Analytic CPU checks for curve geometry, without OCR data or checkpoints."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_curves import (  # noqa: E402
    CurveNumericalError,
    CurveSamplingCapError,
    curve_geometry_loss,
    point_to_polyline_distances,
    resample_arclength,
)


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def xy(values, **kwargs):
    return torch.tensor(values, dtype=torch.float64, **kwargs)


def loss(predicted, target, **kwargs):
    options = dict(target_height_px=4.0, spacing_px=1.3)
    options.update(kwargs)
    return curve_geometry_loss(predicted, target, **options)


def coarse_and_dense():
    coarse = xy([[2, 10], [12, 8], [25, 12]])
    dense = xy([[2, 10], [2.5, 9.9], [3, 9.8], [5, 9.4], [11, 8.2], [12, 8], [25, 12]])
    return coarse, dense


def test_arc_sample_count_and_locations_follow_physical_length():
    result = resample_arclength(xy([[0, 0], [10, 0]]), spacing_px=3.0)
    assert result.points_xy.shape == (math.ceil(10 / 3) + 1, 2)
    assert result.length_px.item() == 10.0
    torch.testing.assert_close(result.points_xy[:, 0], torch.linspace(0, 10, 5, dtype=torch.float64))
    assert torch.count_nonzero(result.points_xy[:, 1]) == 0
    assert result.segment_index.shape == (5,)


def test_nonuniform_densification_preserves_arc_samples():
    coarse, dense = coarse_and_dense()
    first = resample_arclength(coarse, spacing_px=1.3)
    second = resample_arclength(dense, spacing_px=1.3)
    torch.testing.assert_close(first.length_px, second.length_px, atol=1e-10, rtol=0)
    torch.testing.assert_close(first.points_xy, second.points_xy, atol=1e-10, rtol=0)


def test_nonuniform_densification_of_either_argument_preserves_loss():
    coarse, dense = coarse_and_dense()
    displacement = xy([0.3, 1.7])
    reference = loss(coarse + displacement, coarse)
    alternatives = (loss(dense + displacement, coarse), loss(coarse + displacement, dense))
    for result in alternatives:
        for name in ("total", "curve", "endpoints", "pred_to_target", "target_to_pred"):
            torch.testing.assert_close(getattr(result, name), getattr(reference, name), atol=1e-10, rtol=0)


def test_tangential_sliding_on_same_polyline_preserves_samples_and_loss():
    first = xy([[0, 0], [0.5, 0], [1, 0], [9, 0], [10, 0]])
    second = xy([[0, 0], [4, 0], [7, 0], [9.5, 0], [10, 0]])
    target = xy([[0, 2], [10, 2]])
    torch.testing.assert_close(
        resample_arclength(first, spacing_px=1.3).points_xy,
        resample_arclength(second, spacing_px=1.3).points_xy,
        atol=1e-10, rtol=0,
    )
    torch.testing.assert_close(loss(first, target).total, loss(second, target).total, atol=1e-10, rtol=0)


def test_normal_translation_is_normalized_by_target_line_height():
    target = xy([[0, 0], [10, 0]])
    result = loss(target + xy([0, 2]), target)
    for component in (result.curve, result.pred_to_target, result.target_to_pred, result.endpoints):
        torch.testing.assert_close(component, xy(0.5), atol=1e-12, rtol=0)
    torch.testing.assert_close(result.total, xy(1.0), atol=1e-12, rtol=0)


def test_reversal_is_detected_by_ordered_endpoints():
    target = xy([[0, 0], [10, 0]])
    result = loss(target.flip(0), target, target_height_px=2.0)
    assert result.curve.item() == pytest.approx(0.0, abs=1e-12)
    assert result.endpoints.item() == pytest.approx(5.0, abs=1e-12)


def test_partial_coverage_is_detected_in_target_to_prediction_direction():
    result = loss(xy([[0, 0], [4, 0]]), xy([[0, 0], [10, 0]]))
    assert result.pred_to_target.item() == pytest.approx(0.0, abs=1e-12)
    assert result.target_to_pred.item() > 0
    assert result.curve.item() > 0


def test_normalized_loss_is_invariant_to_pixel_unit_scale():
    predicted = xy([[1.2, 2.5], [8.4, 3.9]])
    target = xy([[-1.3, 0.2], [10.7, 1.1]])
    first = loss(predicted, target, predicted_height_px=3.5, spacing_px=1.13)
    scale = 7.0
    scaled = loss(
        predicted * scale, target * scale,
        target_height_px=4.0 * scale, predicted_height_px=3.5 * scale,
        spacing_px=1.13 * scale,
    )
    for name in ("total", "curve", "endpoints", "height", "pred_to_target", "target_to_pred"):
        torch.testing.assert_close(getattr(first, name), getattr(scaled, name), atol=1e-10, rtol=0)


def test_point_to_segment_distance_uses_clamped_projection():
    points = xy([[-2, 3], [4, 3], [12, 3]])
    result = point_to_polyline_distances(points, xy([[0, 0], [10, 0]]))
    torch.testing.assert_close(result, xy([math.sqrt(13), 3, math.sqrt(13)]), atol=1e-12, rtol=0)


def test_duplicate_target_vertices_are_deduplicated_and_targets_detached():
    predicted = xy([[0, 2], [10, 2]], requires_grad=True)
    predicted_height = xy(3.0, requires_grad=True)
    target = xy([[0, 0], [0, 0], [3, 0], [3, 0], [10, 0]], requires_grad=True)
    target_height = xy(4.0, requires_grad=True)
    result = loss(predicted, target, target_height_px=target_height, predicted_height_px=predicted_height)
    result.total.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert predicted_height.grad is not None and predicted_height.grad.abs() > 0
    assert target.grad is None
    assert target_height.grad is None
    assert result.curve.item() == pytest.approx(0.5, abs=1e-12)


def test_duplicate_prediction_vertices_are_rejected_instead_of_hidden():
    predicted = xy([[0, 0], [0, 0], [10, 0]])
    with pytest.raises(ValueError):
        resample_arclength(predicted, spacing_px=1.3)
    with pytest.raises(ValueError):
        loss(predicted, xy([[0, 2], [10, 2]]))


def test_sampling_cap_refuses_underresolved_line():
    with pytest.raises(CurveSamplingCapError):
        resample_arclength(xy([[0, 0], [10, 0]]), spacing_px=1.0, max_samples=5)
    with pytest.raises(CurveSamplingCapError):
        loss(xy([[0, 2], [10, 2]]), xy([[0, 0], [10, 0]]), spacing_px=1.0, max_samples=5)


def test_computed_numerical_overflow_is_distinct_from_malformed_nonfinite_input():
    with pytest.raises(CurveNumericalError):
        resample_arclength(xy([[0, 0], [1e200, 0]]), spacing_px=1.0)
    with pytest.raises(ValueError) as failure:
        resample_arclength(xy([[0, 0], [float("nan"), 0]]), spacing_px=1.0)
    assert not isinstance(failure.value, CurveNumericalError)


def test_sample_count_threshold_has_small_loss_jump_on_stated_nearly_straight_fixture():
    below = xy([[0, 1], [10 - 1e-6, 1]])
    above = xy([[0, 1], [10 + 1e-6, 1]])
    target = xy([[0, 0], [4, 0.01], [10, 0]])
    assert len(resample_arclength(below, spacing_px=1.0).points_xy) == 11
    assert len(resample_arclength(above, spacing_px=1.0).points_xy) == 12
    first = loss(below, target, spacing_px=1.0).total
    second = loss(above, target, spacing_px=1.0).total
    assert torch.isfinite(first) and torch.isfinite(second)
    # This is a fixture bound, not a global continuity claim for discrete M.
    assert abs(second.item() - first.item()) < 1e-3


def test_zero_length_target_and_unknown_orientation_are_explicit_errors():
    predicted = xy([[0, 2], [10, 2]])
    with pytest.raises(ValueError):
        loss(predicted, xy([[0, 0], [0, 0]]))
    with pytest.raises(ValueError):
        loss(predicted, xy([[0, 0], [10, 0]]), orientation="unknown")


def test_curve_loss_gradcheck_away_from_projection_and_sample_count_boundaries():
    # Generic oblique segments avoid zero distances, projection endpoints,
    # equal nearest-segment costs, vertices and integer length/spacing ratios.
    predicted = xy([[1.2, 2.5], [8.4, 3.9]], requires_grad=True)
    predicted_height = xy(3.5, requires_grad=True)
    target = xy([[-1.3, 0.2], [10.7, 1.1]])

    def objective(points, height):
        return loss(points, target, predicted_height_px=height, spacing_px=1.13).total

    assert torch.autograd.gradcheck(objective, (predicted, predicted_height), eps=1e-6, atol=1e-5, rtol=1e-4)
