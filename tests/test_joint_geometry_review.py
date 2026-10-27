"""Independent tiny-CPU checks of coordinates, Jacobians and input contracts."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_geometry import (  # noqa: E402
    FeatureFrame,
    geometry_loss,
    sample_baseline_bands,
)


def _curved_inputs():
    dtype = torch.float64
    yy, xx = torch.meshgrid(
        torch.arange(16, dtype=dtype), torch.arange(16, dtype=dtype), indexing="ij"
    )
    feature = torch.stack((torch.sin(xx / 3) + yy.square() / 31, xx * yy / 43))[None]
    baseline = torch.tensor(
        [[[9.4, 14.2], [15.3, 12.7], [23.8, 16.1], [29.1, 15.4]]],
        dtype=dtype, requires_grad=True,
    )
    height = torch.tensor([4.3], dtype=dtype, requires_grad=True)
    frame = FeatureFrame((48, 40), (3.25, 4.5), (2.3, 1.7))
    return feature, baseline, height, frame


def test_curved_sampler_jacobian_matches_central_finite_differences():
    feature, baseline, height, frame = _curved_inputs()

    def sample(controls, full_height):
        return sample_baseline_bands(
            feature, controls, full_height, frame,
            output_size=(3, 7), baseline_fraction=0.75,
        ).features

    # Interior points avoid the nondifferentiable feature-cell boundaries.
    assert torch.autograd.gradcheck(
        sample, (baseline, height), eps=1e-6, atol=2e-5, rtol=1e-4
    )


def test_nonzero_origin_anisotropic_stride_preserves_affine_pixel_field():
    _, baseline, height, frame = _curved_inputs()
    iy, ix = torch.meshgrid(
        torch.arange(16, dtype=torch.float64),
        torch.arange(16, dtype=torch.float64), indexing="ij",
    )
    x = frame.origin_xy[0] + ix * frame.stride_xy[0]
    y = frame.origin_xy[1] + iy * frame.stride_xy[1]
    feature = torch.stack((x, y, 2 * x - 3 * y + 1))[None]
    sampled = sample_baseline_bands(
        feature, baseline, height, frame, output_size=(3, 7), baseline_fraction=0.75,
    )
    px, py = sampled.pixel_xy.unbind(dim=-1)
    expected = torch.stack((px, py, 2 * px - 3 * py + 1), dim=1)
    torch.testing.assert_close(sampled.features, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("baseline_fraction", [0.0, 0.5, 0.75, 1.0])
def test_baseline_fraction_coordinates_and_gradient_direction(baseline_fraction):
    dtype = torch.float64
    yy = torch.arange(20, dtype=dtype).view(1, 1, 20, 1).expand(1, 1, 20, 20)
    baseline = torch.tensor([[[3.2, 10.2], [14.4, 10.2]]], dtype=dtype, requires_grad=True)
    height = torch.tensor([4.0], dtype=dtype, requires_grad=True)
    result = sample_baseline_bands(
        yy, baseline, height, FeatureFrame((20, 20), (0, 0), (1, 1)),
        output_size=(3, 5), baseline_fraction=baseline_fraction,
    )
    expected_y = torch.linspace(
        10.2 - 4 * baseline_fraction, 10.2 + 4 * (1 - baseline_fraction),
        3, dtype=dtype,
    ).view(1, 3, 1).expand(1, 3, 5)
    torch.testing.assert_close(result.pixel_xy[..., 1], expected_y)
    result.features.mean().backward()
    torch.testing.assert_close(
        baseline.grad, torch.tensor([[[0.0, 0.5], [0.0, 0.5]]], dtype=dtype),
        atol=1e-12, rtol=1e-12,
    )
    torch.testing.assert_close(
        height.grad, torch.tensor([0.5 - baseline_fraction], dtype=dtype),
        atol=1e-12, rtol=1e-12,
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_baseline_fraction_is_rejected(fraction):
    feature, baseline, height, frame = _curved_inputs()
    with pytest.raises(ValueError, match="baseline_fraction"):
        sample_baseline_bands(
            feature, baseline, height, frame,
            output_size=(3, 7), baseline_fraction=fraction,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sampler_rejects_reduced_precision_coordinates_before_sampling(dtype):
    feature, baseline, height, frame = _curved_inputs()
    with pytest.raises(ValueError, match="float32 or float64"):
        sample_baseline_bands(
            feature.to(dtype), baseline.to(dtype), height.to(dtype), frame,
            output_size=(3, 7),
        )


def _loss_inputs():
    baseline = torch.tensor([[[3.0, 8.0], [12.0, 8.0]]])
    height = torch.tensor([4.0])
    return baseline, height, baseline.clone(), height.clone()


@pytest.mark.parametrize("weight_name", ["baseline_weight", "height_weight"])
@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_geometry_loss_rejects_invalid_weights(weight_name, weight):
    with pytest.raises(ValueError, match="weights must be finite and nonnegative"):
        geometry_loss(*_loss_inputs(), **{weight_name: weight})


@pytest.mark.parametrize("shape", [(0, 2, 2), (1, 1, 2), (1, 2, 3)])
def test_geometry_loss_rejects_empty_or_malformed_baselines(shape):
    baseline = torch.zeros(shape)
    height = torch.ones(shape[0])
    with pytest.raises(ValueError, match="baselines must share shape"):
        geometry_loss(baseline, height, baseline.clone(), height.clone())


@pytest.mark.parametrize("height_index", [1, 3])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_geometry_loss_rejects_nonpositive_or_nonfinite_heights(height_index, value):
    inputs = list(_loss_inputs())
    inputs[height_index] = torch.tensor([value])
    with pytest.raises(ValueError, match="positive|finite"):
        geometry_loss(*inputs)


def test_geometry_loss_rejects_height_rank_hidden_by_flattening():
    baseline, height, target_baseline, target_height = _loss_inputs()
    with pytest.raises(ValueError, match="heights must each have shape"):
        geometry_loss(baseline, height.reshape(1, 1, 1), target_baseline, target_height)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_geometry_loss_rejects_unsupported_dtypes(dtype):
    with pytest.raises(ValueError, match="float32 or float64"):
        geometry_loss(*(tensor.to(dtype) for tensor in _loss_inputs()))


def test_geometry_loss_rejects_numerical_overflow_from_finite_scale():
    with pytest.raises(ValueError, match="finite"):
        geometry_loss(*_loss_inputs(), coordinate_scale_px=1e-300)
