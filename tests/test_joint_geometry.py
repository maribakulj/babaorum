"""Tiny CPU checks for geometry sampling; no OCR data or model weights."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_geometry import (  # noqa: E402
    FeatureFrame,
    geometry_loss,
    sample_baseline_bands,
)


def synthetic_field():
    yy, xx = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xx = xx.to(torch.float32)
    yy = yy.to(torch.float32)
    blank = torch.zeros_like(xx)
    first = 4.0 - 0.15 * (xx - 3.0).square() - 0.25 * (yy - 5.0).square()
    second = 4.0 - 0.15 * (xx - 8.0).square() - 0.25 * (yy - 5.0).square()
    return torch.stack((blank, first, second), dim=0).unsqueeze(0)


def synthetic_geometry(requires_grad=True):
    baseline = torch.tensor([[[1.0, 4.0], [5.0, 4.0], [10.0, 4.0]]], requires_grad=requires_grad)
    height = torch.tensor([3.0], requires_grad=requires_grad)
    return baseline, height


def ctc_loss(sample):
    # Synthetic spatial class evidence; this checks autograd routing, not OCR.
    logits = sample.features.mean(dim=2).permute(2, 0, 1).contiguous()
    log_probs = logits.log_softmax(dim=-1)
    target = torch.tensor([1, 2], dtype=torch.long)
    return torch.nn.functional.ctc_loss(
        log_probs,
        target,
        input_lengths=torch.tensor([logits.shape[0]], dtype=torch.long),
        target_lengths=torch.tensor([2], dtype=torch.long),
        blank=0,
        reduction="sum",
        zero_infinity=False,
    )


def test_band_shape_coordinates_and_ctc_gradients_to_geometry():
    feature = synthetic_field().requires_grad_()
    baseline, height = synthetic_geometry()
    result = sample_baseline_bands(
        feature,
        baseline,
        height,
        FeatureFrame((12, 12), (0.0, 0.0), (1.0, 1.0)),
        output_size=(3, 8),
    )
    assert result.features.shape == (1, 3, 3, 8)
    assert result.pixel_xy.shape == (1, 3, 8, 2)
    assert torch.allclose(result.pixel_xy[0, 1, :, 1], torch.full((8,), 4.0), atol=1e-5)
    assert torch.allclose(result.pixel_xy[0, 0, :, 1], torch.full((8,), 2.5), atol=1e-5)
    assert result.as_ctc_sequence().shape == (8, 1, 9)

    loss = ctc_loss(result)
    loss.backward()
    assert baseline.grad is not None and baseline.grad.abs().sum().item() > 1e-7
    assert height.grad is not None and height.grad.abs().sum().item() > 1e-7
    assert feature.grad is not None and feature.grad.abs().sum().item() > 1e-7


def test_detached_ctc_geometry_and_supervised_geometry_gradient():
    feature = synthetic_field().requires_grad_()
    baseline, height = synthetic_geometry()
    result = sample_baseline_bands(
        feature,
        baseline,
        height,
        FeatureFrame((12, 12), (0.0, 0.0), (1.0, 1.0)),
        output_size=(3, 8),
        detach_geometry=True,
    )
    ctc_loss(result).backward()
    assert baseline.grad is None
    assert height.grad is None
    assert feature.grad is not None and feature.grad.abs().sum().item() > 1e-7

    pred_baseline = torch.tensor(
        [[[1.0, 4.5], [5.0, 4.5], [10.0, 4.5]]], requires_grad=True
    )
    pred_height = torch.tensor([3.5], requires_grad=True)
    supervised = geometry_loss(
        pred_baseline,
        pred_height,
        torch.tensor([[[1.0, 4.0], [5.0, 4.0], [10.0, 4.0]]]),
        torch.tensor([3.0]),
        coordinate_scale_px=12.0,
    )
    supervised.total.backward()
    assert supervised.total.item() > 0
    assert pred_baseline.grad is not None and pred_baseline.grad.abs().sum().item() > 0
    assert pred_height.grad is not None and pred_height.grad.abs().sum().item() > 0


@pytest.mark.parametrize(
    "baseline,height,frame,output_size",
    [
        (torch.tensor([[[1.0, 4.0], [10.0, 4.0]]]), torch.tensor([0.0]), FeatureFrame((12, 12), (0, 0), (1, 1)), (3, 8)),
        (torch.tensor([[[1.0, 4.0], [10.0, 4.0]]]), torch.tensor([-1.0]), FeatureFrame((12, 12), (0, 0), (1, 1)), (3, 8)),
        (torch.tensor([[[-1.0, 4.0], [10.0, 4.0]]]), torch.tensor([2.0]), FeatureFrame((12, 12), (0, 0), (1, 1)), (3, 8)),
        (torch.tensor([[[1.0, 4.0], [10.0, 4.0]]]), torch.tensor([2.0]), FeatureFrame((12, 12), (0, 0), (0.5, 0.5)), (3, 8)),
        (torch.tensor([[[1.0, 4.0], [1.0, 4.0]]]), torch.tensor([2.0]), FeatureFrame((12, 12), (0, 0), (1, 1)), (3, 8)),
        (torch.tensor([[[1.0, 4.0], [10.0, 4.0]]]), torch.tensor([2.0]), FeatureFrame((12, 12), (0, 0), (1, 1)), (1, 8)),
    ],
)
def test_invalid_geometry_is_rejected(baseline, height, frame, output_size):
    with pytest.raises(ValueError):
        sample_baseline_bands(
            synthetic_field(), baseline, height, frame, output_size=output_size
        )


def test_nonfinite_baseline_is_rejected():
    baseline, height = synthetic_geometry(requires_grad=False)
    baseline[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sample_baseline_bands(
            synthetic_field(), baseline, height,
            FeatureFrame((12, 12), (0, 0), (1, 1)), output_size=(3, 8)
        )
