from __future__ import annotations

import math

import pytest
torch = pytest.importorskip("torch")

from babaorum.experimental.joint_geometry import FeatureFrame, sample_baseline_bands
from babaorum.experimental.joint_reader_bridge import (
    contract_band_to_sampler_geometry,
    page_edges_to_feature_indices,
    rotate_baseline_about_chord,
    summarize_delta_support,
)


def test_page_edges_convert_to_pixel_centers_before_feature_frame():
    frame = FeatureFrame(
        original_size_wh=(100, 80), origin_xy=(10.0, 20.0), stride_xy=(4.0, 5.0)
    )
    edge_xy = torch.tensor([[18.5, 35.5]], dtype=torch.float64)
    feature_xy, valid = page_edges_to_feature_indices(edge_xy, frame, (10, 20))

    torch.testing.assert_close(feature_xy, torch.tensor([[2.0, 3.0]], dtype=torch.float64))
    assert valid.tolist() == [True]


def test_contract_band_extents_map_to_sampler_without_losing_gradients():
    edges = torch.tensor([[[2.5, 10.5], [17.5, 10.5]]], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([6.0], dtype=torch.float64, requires_grad=True)
    positive = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    baseline, height, fraction = contract_band_to_sampler_geometry(edges, negative, positive)

    torch.testing.assert_close(baseline, torch.tensor([[[2.0, 10.0], [17.0, 10.0]]], dtype=torch.float64))
    torch.testing.assert_close(height, torch.tensor([8.0], dtype=torch.float64))
    torch.testing.assert_close(fraction, torch.tensor([0.75], dtype=torch.float64))
    yy = torch.arange(24, dtype=torch.float64).view(1, 1, 24, 1).expand(1, 1, 24, 24)
    sample = sample_baseline_bands(
        yy, baseline, height, FeatureFrame((24, 24), (0.0, 0.0), (1.0, 1.0)),
        output_size=(2, 4), baseline_fraction=fraction,
    )
    torch.testing.assert_close(
        sample.pixel_xy[0, :, :, 1],
        torch.tensor([[4.0] * 4, [12.0] * 4], dtype=torch.float64),
    )
    sample.features.mean().backward()
    torch.testing.assert_close(negative.grad, torch.tensor([-0.5], dtype=torch.float64))
    torch.testing.assert_close(positive.grad, torch.tensor([0.5], dtype=torch.float64))
    torch.testing.assert_close(edges.grad[..., 1], torch.full((1, 2), 0.5, dtype=torch.float64))


def test_contract_band_rejects_zero_width_negative_width_and_reversed_chord():
    baseline = torch.tensor([[[1.0, 2.0], [8.0, 2.0]]])
    with pytest.raises(ValueError, match="total band extension must be positive"):
        contract_band_to_sampler_geometry(baseline, 0.0, 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        contract_band_to_sampler_geometry(baseline, -1.0, 2.0)
    with pytest.raises(ValueError, match="left-to-right"):
        contract_band_to_sampler_geometry(baseline.flip(1), 1.0, 1.0)


def test_detached_contract_band_stops_gradient_through_both_extents():
    edges = torch.tensor([[[2.5, 10.5], [17.5, 10.5]]], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([6.0], dtype=torch.float64, requires_grad=True)
    positive = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    baseline, height, fraction = contract_band_to_sampler_geometry(edges, negative, positive)
    yy = torch.arange(24, dtype=torch.float64).view(1, 1, 24, 1).expand(1, 1, 24, 24).requires_grad_()
    sample = sample_baseline_bands(
        yy, baseline, height, FeatureFrame((24, 24), (0.0, 0.0), (1.0, 1.0)),
        output_size=(2, 4), baseline_fraction=fraction, detach_geometry=True,
    )
    sample.features.sum().backward()

    assert yy.grad is not None
    assert edges.grad is None
    assert negative.grad is None
    assert positive.grad is None


def test_feature_mapping_keeps_out_of_support_coordinates_and_masks_them():
    frame = FeatureFrame(
        original_size_wh=(100, 80), origin_xy=(10.0, 20.0), stride_xy=(4.0, 5.0)
    )
    edge_xy = torch.tensor([[0.0, 0.5], [18.5, 80.5]], dtype=torch.float32)
    feature_xy, valid = page_edges_to_feature_indices(edge_xy, frame, (10, 20))

    assert valid.tolist() == [False, False]
    assert feature_xy[0, 0] < 0  # no clamp at the page border
    assert feature_xy[1, 1] > 9  # no clamp at the feature-map border


def test_feature_mapping_rejects_bf16_and_nonfinite_coordinates():
    frame = FeatureFrame(
        original_size_wh=(100, 80), origin_xy=(0.0, 0.0), stride_xy=(4.0, 4.0)
    )
    with pytest.raises(ValueError, match="float32 or float64"):
        page_edges_to_feature_indices(torch.zeros((1, 2), dtype=torch.bfloat16), frame, (20, 25))
    with pytest.raises(ValueError, match="finite"):
        page_edges_to_feature_indices(torch.tensor([[float("nan"), 1.0]]), frame, (20, 25))


@pytest.mark.parametrize("delta", [-0.5, 0.5])
def test_straight_rotation_has_exact_signed_chord_normal_displacement(delta):
    baseline = torch.tensor([[10.0, 20.0], [110.0, 20.0]], dtype=torch.float64)
    result = rotate_baseline_about_chord(baseline, delta, height_px=10.0, max_angle_deg=30.0)

    assert result.status == "included"
    assert math.isclose(float(result.start_chord_normal_displacement_px), -delta * 10.0)
    assert math.isclose(float(result.end_chord_normal_displacement_px), delta * 10.0)
    assert math.isclose(float(result.theta_radians), math.asin(2 * delta * 10 / 100))
    assert math.isclose(float(result.chord_tangential_slip_px), 50 * (1 - math.cos(float(result.theta_radians))))
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.baseline_xy[-1] - result.baseline_xy[0]),
        torch.tensor(100.0, dtype=torch.float64),
    )
    assert result.interior_local_normal_min_h is None
    assert result.interior_local_normal_max_h is None


def test_angular_perturbation_is_differentiable_in_delta():
    baseline = torch.tensor([[0.0, 0.0], [75.0, 0.0]], dtype=torch.float64)
    delta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    result = rotate_baseline_about_chord(baseline, delta, height_px=8.0, max_angle_deg=30.0)

    assert result.status == "included"
    result.end_chord_normal_displacement_px.backward()
    torch.testing.assert_close(delta.grad, torch.tensor(8.0, dtype=torch.float64))


def test_curved_angular_coordinates_pass_float64_gradcheck():
    baseline = torch.tensor(
        [[0.0, 0.0], [3.0, 1.0], [7.0, 1.0], [10.0, 0.0]], dtype=torch.float64
    )
    delta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

    def transformed_points(value):
        result = rotate_baseline_about_chord(
            baseline, value, height_px=1.0, max_angle_deg=30.0
        )
        assert result.status == "included"
        return result.baseline_xy

    assert torch.autograd.gradcheck(
        transformed_points, (delta,), eps=1e-6, atol=1e-6, rtol=1e-5
    )


def test_infeasible_angular_perturbation_is_countable_not_clamped():
    baseline = torch.tensor([[0.0, 0.0], [15.0, 0.0]], dtype=torch.float64)
    result = rotate_baseline_about_chord(baseline, delta=1.0, height_px=10.0, max_angle_deg=30.0)

    assert result.status == "excluded_infeasible_chord"
    assert result.baseline_xy is None
    assert result.theta_radians is None
    assert math.isclose(float(result.chord_length_px), 15.0)


def test_curved_rotation_reports_interior_local_normal_extrema():
    baseline = torch.tensor(
        [[0.0, 0.0], [3.0, 1.0], [7.0, 1.0], [10.0, 0.0]], dtype=torch.float64
    )
    result = rotate_baseline_about_chord(baseline, delta=0.25, height_px=2.0, max_angle_deg=30.0)

    assert result.status == "included"
    assert math.isclose(float(result.start_chord_normal_displacement_px), -0.5)
    assert math.isclose(float(result.end_chord_normal_displacement_px), 0.5)
    assert result.interior_local_normal_min_h is not None
    assert result.interior_local_normal_max_h is not None
    assert result.interior_local_normal_min_h < result.interior_local_normal_max_h


def test_curved_rotation_preserves_all_pairwise_vertex_distances():
    baseline = torch.tensor(
        [[0.0, 0.0], [3.0, 1.0], [7.0, 1.0], [10.0, 0.0]], dtype=torch.float64
    )
    result = rotate_baseline_about_chord(
        baseline, delta=0.25, height_px=2.0, max_angle_deg=30.0
    )

    assert result.status == "included"
    before = torch.cdist(baseline, baseline)
    after = torch.cdist(result.baseline_xy, result.baseline_xy)
    torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)


def test_wrong_direction_and_undefined_interior_tangent_are_excluded():
    reversed_line = torch.tensor([[10.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    collapsed_secant = torch.tensor(
        [[0.0, 0.0], [5.0, 0.0], [0.0, 0.0], [10.0, 0.0]], dtype=torch.float64
    )

    assert rotate_baseline_about_chord(reversed_line, 0.1, 2.0, max_angle_deg=30.0).status == "excluded_wrong_orientation"
    assert rotate_baseline_about_chord(collapsed_secant, 0.1, 2.0, max_angle_deg=30.0).status == "excluded_degenerate"


def test_ninety_degree_boundary_is_excluded_before_asin_gradient():
    baseline = torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=torch.float64)
    delta = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)  # 2δh=C
    result = rotate_baseline_about_chord(baseline, delta, height_px=10.0, max_angle_deg=30.0)

    assert result.status == "excluded_angle_limit"
    assert result.theta_radians is None
    assert result.end_chord_normal_displacement_px is None


def test_gradient_remains_finite_near_predeclared_angle_limit():
    baseline = torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=torch.float64)
    limit = 30.0
    delta_at_limit = 100.0 * math.sin(math.radians(limit)) / 20.0
    delta = torch.tensor(delta_at_limit * (1.0 - 1e-6), dtype=torch.float64, requires_grad=True)
    result = rotate_baseline_about_chord(
        baseline, delta, height_px=10.0, max_angle_deg=limit
    )

    assert result.status == "included"
    result.end_chord_normal_displacement_px.backward()
    assert torch.isfinite(delta.grad)
    torch.testing.assert_close(delta.grad, torch.tensor(10.0, dtype=torch.float64), rtol=1e-9, atol=1e-9)


def test_angle_limit_is_required_and_validated():
    baseline = torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=torch.float64)
    with pytest.raises(TypeError):
        rotate_baseline_about_chord(baseline, 0.1, 10.0)
    with pytest.raises(ValueError, match="between 0 and 90"):
        rotate_baseline_about_chord(baseline, 0.1, 10.0, max_angle_deg=90.0)


def test_reader_m1_design_cap_is_fifteen_degrees():
    baseline = torch.tensor([[0.0, 0.0], [75.0, 0.0]], dtype=torch.float64)
    # For delta=1 and h=10, C/h=7.5 exceeds the 15-degree cutoff ratio
    # 2/sin(15°)≈7.73, so this perturbation is counted as unsupported by angle.
    excluded = rotate_baseline_about_chord(
        baseline, 1.0, height_px=10.0, max_angle_deg=15.0
    )
    assert excluded.status == "excluded_angle_limit"
    # C/h=8 is just outside the exclusion region and remains included.
    long_baseline = torch.tensor([[0.0, 0.0], [80.0, 0.0]], dtype=torch.float64)
    included = rotate_baseline_about_chord(
        long_baseline, 1.0, height_px=10.0, max_angle_deg=15.0
    )
    assert included.status == "included"


def test_delta_support_uses_strict_twenty_percent_cutoff_and_counts_reasons():
    summary = summarize_delta_support([
        (-0.5, "included"), (-0.5, "included"), (-0.5, "included"),
        (-0.5, "excluded_angle_limit"),
        (0.5, "included"), (0.5, "included"), (0.5, "included"),
        (0.5, "included"), (0.5, "excluded_angle_limit"),
        (1.0, "included"), (1.0, "included"), (1.0, "included"),
        (1.0, "excluded_angle_limit"),
    ])

    assert [row.delta for row in summary] == [-0.5, 0.5, 1.0]
    negative, at_cutoff, above_cutoff = summary
    assert negative.status == "under_supported"  # signed offsets are not pooled
    assert negative.exclusion_fraction == 0.25
    assert (at_cutoff.attempted, at_cutoff.included, at_cutoff.excluded) == (5, 4, 1)
    assert at_cutoff.exclusion_fraction == 0.2
    assert at_cutoff.status == "supported"
    assert at_cutoff.exclusion_counts == (("excluded_angle_limit", 1),)
    assert (above_cutoff.attempted, above_cutoff.included, above_cutoff.excluded) == (4, 3, 1)
    assert above_cutoff.exclusion_fraction == 0.25
    assert above_cutoff.status == "under_supported"


def test_delta_support_rejects_recomputed_values_outside_predeclared_grid():
    with pytest.raises(ValueError, match="exact member of the predeclared M1 grid"):
        summarize_delta_support([(0.1 * 3, "included")])
