"""Small CPU checks for curved-band sampling and explicit geometry exclusion."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_curve_sampling import (  # noqa: E402
    curved_band_grid,
    masked_curve_ctc_loss,
    sample_curved_band,
)
from babaorum.experimental.joint_curves import curve_geometry_loss  # noqa: E402
from babaorum.experimental.joint_geometry import FeatureFrame  # noqa: E402


FRAME = FeatureFrame((64, 64), (0.0, 0.0), (1.0, 1.0))


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def xy(values, **kwargs):
    return torch.tensor(values, dtype=torch.float64, **kwargs)


def grid_for(points, height=None, **kwargs):
    options = dict(band_rows=3, spacing_px=1.7, baseline_fraction=0.75)
    options.update(kwargs)
    return curved_band_grid(points, xy(4.0) if height is None else height, FRAME, (64, 64), **options)


def linear_features(frame=FRAME, size=(64, 64)):
    yy, xx = torch.meshgrid(
        torch.arange(size[0], dtype=torch.float64),
        torch.arange(size[1], dtype=torch.float64), indexing="ij",
    )
    xx = frame.origin_xy[0] + frame.stride_xy[0] * xx
    yy = frame.origin_xy[1] + frame.stride_xy[1] * yy
    return torch.stack((xx + 2 * yy, 0.5 * xx - yy))[None]


def class_features():
    yy, xx = torch.meshgrid(
        torch.arange(64, dtype=torch.float64), torch.arange(64, dtype=torch.float64), indexing="ij"
    )
    first = 4 - 0.01 * (xx - 17).square() - 0.03 * (yy - 20).square()
    second = 4 - 0.01 * (xx - 39).square() - 0.03 * (yy - 20).square()
    return torch.stack((torch.zeros_like(xx), first, second))[None].requires_grad_()


def class_logits(sample):
    return sample.features.mean(dim=2).permute(2, 0, 1).contiguous()


def test_nonuniform_densification_preserves_grid_and_sampled_features():
    coarse = xy([[5, 20], [25, 16], [48, 22]])
    dense = xy([[5, 20], [6, 19.8], [7, 19.6], [9, 19.2], [24, 16.2], [25, 16], [48, 22]])
    first, second = grid_for(coarse), grid_for(dense)
    assert first.line_valid and second.line_valid
    torch.testing.assert_close(first.pixel_xy, second.pixel_xy, atol=1e-10, rtol=0)
    features = linear_features()
    torch.testing.assert_close(
        sample_curved_band(features, first, FRAME).features,
        sample_curved_band(features, second, FRAME).features,
        atol=1e-10, rtol=0,
    )


def test_tangential_control_point_sliding_preserves_grid_and_features():
    first = grid_for(xy([[5, 20], [6, 20], [8, 20], [45, 20], [48, 20]]))
    second = grid_for(xy([[5, 20], [20, 20], [35, 20], [46, 20], [48, 20]]))
    assert first.line_valid and second.line_valid
    torch.testing.assert_close(first.pixel_xy, second.pixel_xy, atol=1e-10, rtol=0)
    torch.testing.assert_close(
        sample_curved_band(linear_features(), first, FRAME).features,
        sample_curved_band(linear_features(), second, FRAME).features,
        atol=1e-10, rtol=0,
    )


def test_linear_field_sampling_is_exact_with_nontrivial_stride_and_origin():
    frame = FeatureFrame((64, 64), (3.0, 5.0), (2.0, 3.0))
    size = (16, 20)
    grid = curved_band_grid(
        xy([[7, 23], [35, 23]]), xy(4.0), frame, size,
        band_rows=3, spacing_px=1.7, baseline_fraction=0.75,
    )
    assert grid.line_valid
    sample = sample_curved_band(linear_features(frame, size), grid, frame)
    xx, yy = grid.pixel_xy.unbind(-1)
    expected = torch.stack((xx + 2 * yy, 0.5 * xx - yy))[None]
    torch.testing.assert_close(sample.features, expected, atol=1e-10, rtol=0)
    torch.testing.assert_close(grid.pixel_xy[:, 0, 1], xy([20, 22, 24]), atol=1e-12, rtol=0)


def test_outside_band_keeps_raw_coordinates_and_live_boundary_repair_gradient():
    points = xy([[8, 1], [40, 1]], requires_grad=True)
    height = xy(8.0, requires_grad=True)
    grid = grid_for(points, height)
    assert grid.pixel_xy[..., 1].min() < 0
    assert not grid.sample_valid.all()
    assert not grid.line_valid
    assert grid.boundary_loss.item() > 0
    sample = sample_curved_band(linear_features(), grid, FRAME)
    assert torch.isfinite(sample.features).all()
    assert torch.count_nonzero(sample.features) == 0
    with pytest.raises(ValueError, match="invalid"):
        sample.as_ctc_sequence()
    grid.boundary_loss.backward()
    assert points.grad is not None and points.grad[:, 1].sum() < 0
    assert height.grad is not None and height.grad > 0
    # Gradient descent moves the baseline downward and decreases the band height.


def test_feature_support_mask_can_reject_coordinates_inside_original_page():
    frame = FeatureFrame((64, 64), (3.0, 5.0), (2.0, 3.0))
    grid = curved_band_grid(
        xy([[7, 6], [35, 6]]), xy(4.0), frame, (16, 20),
        band_rows=3, spacing_px=1.7, baseline_fraction=0.75,
    )
    assert (grid.pixel_xy >= 0).all() and (grid.pixel_xy <= 63).all()
    assert not grid.sample_valid.all()
    assert not grid.line_valid
    assert grid.boundary_loss.item() > 0


@pytest.mark.parametrize("radius,height,valid", [(12.0, 6.0, True), (2.0, 8.0, False)])
def test_offset_arc_accepts_large_radius_and_detects_fold_below_offset(radius, height, valid):
    theta = torch.linspace(-2.6, -0.54, 41, dtype=torch.float64)
    points = xy([32, 32]) + radius * torch.stack((theta.cos(), theta.sin()), -1)
    grid = grid_for(points, xy(height), spacing_px=0.15 if radius < 3 else 0.7, baseline_fraction=0.5)
    assert bool(grid.line_valid) is valid
    if valid:
        assert grid.cell_valid.all()
        assert grid.fold_loss.item() == pytest.approx(0.0, abs=1e-12)
    else:
        assert not grid.projection_valid.all()
        assert not grid.cell_valid.all()
        assert grid.fold_loss.item() > 0


def test_right_angle_corner_is_explicitly_marked_invalid():
    grid = grid_for(xy([[10, 20], [20, 20], [20, 30]]))
    assert not grid.sharp_turn_valid
    assert not grid.line_valid
    with pytest.raises(ValueError, match="invalid"):
        sample_curved_band(linear_features(), grid, FRAME).as_ctc_sequence()


def test_local_backtracking_is_invalid_despite_correct_endpoint_orientation():
    grid = grid_for(xy([[5, 20], [13, 20], [7, 20], [15, 20]]))
    assert grid.orientation_valid
    assert grid.boundary_loss.item() == 0
    assert not grid.sharp_turn_valid
    assert not grid.line_valid


def test_densification_cannot_dilute_sharp_turn_penalty():
    coarse = grid_for(xy([[10, 20], [20, 20], [15, 28]]))
    dense = grid_for(xy([[10, 20], [11, 20], [12, 20], [18, 20], [20, 20], [15, 28]]))
    assert not coarse.sharp_turn_valid and not dense.sharp_turn_valid
    torch.testing.assert_close(coarse.pixel_xy, dense.pixel_xy, atol=1e-10, rtol=0)
    torch.testing.assert_close(coarse.fold_loss, dense.fold_loss, atol=1e-10, rtol=0)


def test_vertical_orientation_uses_declared_tangent_normal_convention():
    points = xy([[20, 10], [20, 30]])
    grid = grid_for(points, orientation="ttb")
    assert grid.line_valid
    torch.testing.assert_close(grid.pixel_xy[:, 0], xy([[23, 10], [21, 10], [19, 10]]))
    wrong_orientation = grid_for(points, orientation="btt")
    assert not wrong_orientation.orientation_valid
    assert not wrong_orientation.line_valid


def test_global_self_intersection_is_detected_even_with_valid_local_cells():
    parameter = torch.linspace(-0.4, 5.9, 101, dtype=torch.float64)
    points = torch.stack((32 + 15 * parameter.sin(), 32 + 10 * (2 * parameter).sin()), -1)
    grid = grid_for(points, xy(0.5), spacing_px=0.8, baseline_fraction=0.5)
    assert grid.sample_valid.all()
    assert grid.sharp_turn_valid
    assert grid.cell_valid.all()
    assert not grid.contour_valid
    assert not grid.line_valid


@pytest.mark.parametrize("detach_geometry", [False, True])
def test_ctc_sampling_gradient_can_detach_geometry_while_preserving_features(detach_geometry):
    points = xy([[8, 25], [48, 25]], requires_grad=True)
    height = xy(6.0, requires_grad=True)
    feature = class_features()
    grid = grid_for(points, height)
    assert grid.line_valid
    sample = sample_curved_band(feature, grid, FRAME, detach_geometry=detach_geometry)
    assert sample.as_ctc_sequence().shape[0] == len(grid.center_xy)
    result = masked_curve_ctc_loss(class_logits(sample), sample, [1, 2])
    assert result.included_lines == 1 and result.excluded_lines == 0
    result.loss.backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    for parameter in (points, height):
        if detach_geometry:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        else:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_invalid_ctc_line_is_counted_without_blocking_valid_line_or_geometry_repair():
    good_points = xy([[8, 25], [48, 25]], requires_grad=True)
    bad_points = xy([[8, 1], [48, 1]], requires_grad=True)
    good_grid = grid_for(good_points, xy(6.0))
    bad_grid = grid_for(bad_points, xy(8.0))
    feature = class_features()
    good_sample = sample_curved_band(feature, good_grid, FRAME)
    bad_sample = sample_curved_band(feature, bad_grid, FRAME, detach_geometry=True)
    included = masked_curve_ctc_loss(class_logits(good_sample), good_sample, [1, 2])
    excluded = masked_curve_ctc_loss(class_logits(bad_sample), bad_sample, [1, 2])
    assert included.included_lines + excluded.included_lines == 1
    assert included.excluded_lines + excluded.excluded_lines == 1
    assert excluded.status == "excluded_invalid_geometry"
    assert excluded.loss.item() == 0 and excluded.loss.requires_grad
    assert included.loss.item() > 0
    objective = included.loss + excluded.loss + bad_grid.boundary_loss
    objective.backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert good_points.grad is not None and good_points.grad.abs().sum() > 0
    assert bad_points.grad is not None and bad_points.grad[:, 1].sum() < 0


def test_excluded_line_still_rejects_blank_target_instead_of_hiding_bad_labels():
    grid = grid_for(xy([[8, 1], [48, 1]]), xy(8.0))
    sample = sample_curved_band(class_features(), grid, FRAME)
    with pytest.raises(ValueError):
        masked_curve_ctc_loss(class_logits(sample), sample, [0])
    # A well-formed target remains excluded even if this invalid grid is too short.
    result = masked_curve_ctc_loss(class_logits(sample), sample, [1] * 100)
    assert result.excluded_lines == 1 and result.loss.item() == 0


def test_short_predicted_line_excludes_infeasible_ctc_without_blocking_other_lines():
    points = [
        xy([[8, 25], [48, 25]], requires_grad=True),
        xy([[9, 31], [44, 31]], requires_grad=True),
        xy([[12, 20], [13, 20]], requires_grad=True),
    ]
    feature = class_features()
    grids = [grid_for(line) for line in points]
    assert all(grid.line_valid for grid in grids)
    logits = [class_logits(sample_curved_band(feature, grid, FRAME)) for grid in grids]
    assert logits[2].shape[0] == 2
    logits[2].retain_grad()
    targets = [[1, 2], [2, 1], [1, 1]]
    results = [masked_curve_ctc_loss(scores, grid, target) for scores, grid, target in zip(logits, grids, targets)]
    assert sum(result.included_lines for result in results) == 2
    assert sum(result.excluded_lines for result in results) == 1
    assert results[2].status == "excluded_infeasible_length"
    assert results[2].loss.item() == 0 and results[2].loss.requires_grad
    assert all(result.loss.item() > 0 for result in results[:2])

    # The excluded short query still receives independent geometry supervision.
    geometry = curve_geometry_loss(
        points[2], xy([[12, 18], [16, 18]]), target_height_px=4.0, spacing_px=1.7
    ).total
    ctc = sum(result.loss for result in results)
    assert ctc.requires_grad and torch.isfinite(ctc)
    (ctc + geometry).backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert all(line.grad is not None and line.grad.abs().sum() > 0 for line in points[:2])
    assert logits[2].grad is not None and torch.count_nonzero(logits[2].grad) == 0
    assert points[2].grad is not None and points[2].grad[:, 1].sum() > 0


def test_finite_height_that_overflows_geometry_penalties_is_rejected():
    with pytest.raises(ValueError):
        grid_for(xy([[8, 25], [48, 25]]), xy(1e308))


def test_boundary_penalty_requires_positive_external_scale():
    with pytest.raises(ValueError):
        grid_for(xy([[8, 25], [48, 25]]), penalty_scale_px=0)


def test_sampling_gradcheck_at_generic_float64_coordinates():
    points = xy([[5.2, 20.3], [19.4, 23.1]], requires_grad=True)
    height = xy(4.2, requires_grad=True)
    feature = linear_features()

    def objective(controls, band_height):
        grid = grid_for(controls, band_height, spacing_px=2.3)
        assert grid.line_valid
        return sample_curved_band(feature, grid, FRAME).features

    assert torch.autograd.gradcheck(objective, (points, height), eps=1e-6, atol=1e-5, rtol=1e-4)
