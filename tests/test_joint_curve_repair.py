"""Explicit synthetic repair prior for collapsed predicted curve vertices."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_curve_repair import repair_degenerate_curve_loss  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def xy(values, **kwargs):
    return torch.tensor(values, dtype=torch.float64, **kwargs)


def test_all_four_collapsed_vertices_separate_after_one_repair_step():
    predicted = xy([[5, 0]] * 4, requires_grad=True)
    result = repair_degenerate_curve_loss(
        predicted, xy([[0, 0], [10, 0]]), target_height_px=2.0
    )
    assert result.reason == "zero_length_predicted_segment"
    assert result.target_anchors_xy.shape == predicted.shape
    torch.testing.assert_close(result.target_anchors_xy[:, 0], torch.linspace(0, 10, 4, dtype=torch.float64))
    result.total.backward()
    assert predicted.grad is not None
    repaired = predicted.detach() - 0.1 * predicted.grad
    assert (torch.linalg.vector_norm(repaired[1:] - repaired[:-1], dim=-1) > 0).all()
    assert (repaired[1:, 0] > repaired[:-1, 0]).all()
    assert repaired[1, 0] < 5 < repaired[2, 0]


def test_repair_anchors_and_loss_ignore_nonuniform_target_densification():
    predicted = xy([[5, 1]] * 4)
    coarse = xy([[0, 0], [4, 3], [10, 3]])
    dense = xy([[0, 0], [0.4, 0.3], [0.8, 0.6], [3.2, 2.4], [4, 3], [4, 3], [10, 3]])
    first = repair_degenerate_curve_loss(predicted, coarse, target_height_px=4.0)
    second = repair_degenerate_curve_loss(predicted, dense, target_height_px=4.0)
    expected_anchors = xy([[0, 0], [44 / 15, 11 / 5], [19 / 3, 3], [10, 3]])
    torch.testing.assert_close(first.target_anchors_xy, expected_anchors, atol=1e-12, rtol=0)
    for name in ("total", "anchors", "endpoints", "height", "target_anchors_xy"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), atol=1e-12, rtol=0)


def test_repair_detaches_all_target_geometry_but_trains_prediction_and_height():
    predicted = xy([[5, 1]] * 4, requires_grad=True)
    predicted_height = xy(3.0, requires_grad=True)
    target = xy([[0, 0], [0, 0], [10, 0]], requires_grad=True)
    target_height = xy(4.0, requires_grad=True)
    result = repair_degenerate_curve_loss(
        predicted, target, target_height_px=target_height, predicted_height_px=predicted_height
    )
    assert not result.target_anchors_xy.requires_grad
    result.total.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert predicted_height.grad is not None and predicted_height.grad < 0
    assert target.grad is None
    assert target_height.grad is None


def test_repair_refuses_valid_prediction_instead_of_substituting_an_index_prior():
    with pytest.raises(ValueError):
        repair_degenerate_curve_loss(
            xy([[0, 1], [3, 1], [7, 1], [10, 1]]),
            xy([[0, 0], [10, 0]]), target_height_px=4.0,
        )


def test_explicit_near_degenerate_cutoff_repairs_tiny_but_nonzero_segments():
    predicted = xy([[5 + index * 1e-6, 0] for index in range(4)], requires_grad=True)
    target = xy([[0, 0], [10, 0]])
    with pytest.raises(ValueError):
        repair_degenerate_curve_loss(predicted, target, target_height_px=2.0)
    result = repair_degenerate_curve_loss(
        predicted, target, target_height_px=2.0, min_segment_length_px=1e-4
    )
    assert result.reason == "near_degenerate_predicted_segment"
    result.total.backward()
    repaired = predicted.detach() - 0.1 * predicted.grad
    assert (torch.linalg.vector_norm(repaired[1:] - repaired[:-1], dim=-1) > 1e-4).all()
