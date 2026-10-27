#!/usr/bin/env python3
"""Run a tiny synthetic CPU autograd probe; this is not OCR evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_geometry import (  # noqa: E402
    FeatureFrame,
    geometry_loss,
    sample_baseline_bands,
)


def synthetic_field():
    yy, xx = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    xx, yy = xx.float(), yy.float()
    blank = torch.zeros_like(xx)
    first = 4.0 - 0.15 * (xx - 3.0).square() - 0.25 * (yy - 5.0).square()
    second = 4.0 - 0.15 * (xx - 8.0).square() - 0.25 * (yy - 5.0).square()
    return torch.stack((blank, first, second), dim=0).unsqueeze(0)


def ctc_loss(features):
    logits = features.mean(dim=2).permute(2, 0, 1).contiguous()
    lp = logits.log_softmax(dim=-1)
    return torch.nn.functional.ctc_loss(
        lp, torch.tensor([1, 2]), torch.tensor([logits.shape[0]]),
        torch.tensor([2]), blank=0, reduction="sum", zero_infinity=False,
    )


def norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach()).item())


def run_probe():
    torch.set_num_threads(1)
    frame = FeatureFrame((12, 12), (0.0, 0.0), (1.0, 1.0))
    field = synthetic_field().requires_grad_()
    baseline = torch.tensor([[[1.0, 4.0], [5.0, 4.0], [10.0, 4.0]]], requires_grad=True)
    height = torch.tensor([3.0], requires_grad=True)
    band = sample_baseline_bands(field, baseline, height, frame, output_size=(3, 8))
    recognition = ctc_loss(band.features)
    recognition.backward()
    ctc_live = {
        "loss": float(recognition.detach().item()),
        "baseline_gradient_l2": norm(baseline.grad),
        "height_gradient_l2": norm(height.grad),
        "feature_gradient_l2": norm(field.grad),
    }

    field_detached = synthetic_field().requires_grad_()
    baseline_detached = torch.tensor([[[1.0, 4.0], [5.0, 4.0], [10.0, 4.0]]], requires_grad=True)
    height_detached = torch.tensor([3.0], requires_grad=True)
    band_detached = sample_baseline_bands(
        field_detached, baseline_detached, height_detached, frame,
        output_size=(3, 8), detach_geometry=True,
    )
    detached_loss = ctc_loss(band_detached.features)
    detached_loss.backward()

    baseline_pred = torch.tensor([[[1.0, 4.5], [5.0, 4.5], [10.0, 4.5]]], requires_grad=True)
    height_pred = torch.tensor([3.5], requires_grad=True)
    geometry = geometry_loss(
        baseline_pred, height_pred,
        torch.tensor([[[1.0, 4.0], [5.0, 4.0], [10.0, 4.0]]]),
        torch.tensor([3.0]), coordinate_scale_px=12.0,
    )
    geometry.total.backward()

    gradients = {
        "ctc_to_geometry_enabled": ctc_live["baseline_gradient_l2"] > 0 and ctc_live["height_gradient_l2"] > 0,
        "ctc_to_geometry_disabled": baseline_detached.grad is None and height_detached.grad is None,
        "detached_feature_gradient_l2": norm(field_detached.grad),
        "geometry_loss": float(geometry.total.detach().item()),
        "geometry_baseline_gradient_l2": norm(baseline_pred.grad),
        "geometry_height_gradient_l2": norm(height_pred.grad),
    }
    passed = (
        gradients["ctc_to_geometry_enabled"]
        and gradients["ctc_to_geometry_disabled"]
        and gradients["detached_feature_gradient_l2"] > 0
        and gradients["geometry_loss"] > 0
        and gradients["geometry_baseline_gradient_l2"] > 0
        and gradients["geometry_height_gradient_l2"] > 0
    )
    return {
        "schema_version": "joint_geometry_probe_v1",
        "device": "cpu",
        "torch_version": torch.__version__,
        "synthetic_only": True,
        "pretrained_weights": False,
        "ocr_performance_claim": False,
        "frame": {
            "coordinate_system": "original_encoded_pixels_top_left_xy_before_exif",
            "original_size_wh": list(frame.original_size_wh),
            "feature_origin_xy": list(frame.origin_xy),
            "feature_stride_xy": list(frame.stride_xy),
        },
        "shapes": {
            "feature_map": [1, 3, 12, 12],
            "baseline_controls": [1, 3, 2],
            "sampled_band": list(band.features.shape),
            "sampled_pixel_xy": list(band.pixel_xy.shape),
            "ctc_sequence": list(band.as_ctc_sequence().shape),
        },
        "ctc_live": ctc_live,
        "gradient_checks": gradients,
        "result": "PASS_ENGINEERING_FEASIBILITY_ONLY" if passed else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_geometry_probe.json")
    args = parser.parse_args()
    result = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
