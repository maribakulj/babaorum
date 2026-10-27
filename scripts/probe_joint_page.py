#!/usr/bin/env python3
"""Bounded CPU page wiring and constructed geometry/CTC conflict witness.

No page dataset, pretrained weights or OCR metrics are consumed. A failed
assertion produces a FAIL report; numeric outputs are engineering diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_geometry import (  # noqa: E402
    FeatureFrame, geometry_loss, sample_baseline_bands,
)
from babaorum.experimental.joint_page import (  # noqa: E402
    PageTargets, TinyJointPage, export_page, match_lines, page_losses,
)


def norm(value):
    return 0.0 if value is None else float(value.detach().norm())


def gradient_diagnostics(loss, prediction, model):
    names = ("baseline", "height", "features", "geometry_head", "encoder")
    inputs = (prediction.baseline_xy, prediction.height_px, prediction.feature_map,
              model.geometry_head.weight, next(model.encoder.parameters()))
    grads = torch.autograd.grad(loss, inputs, allow_unused=True, retain_graph=True)
    return {name: {"connected": grad is not None, "l2": norm(grad)}
            for name, grad in zip(names, grads)}


def page_wiring(config):
    torch.manual_seed(config["seed"])
    width, height = config["image_size_wh"]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    # A smooth nonconstant input, with no rendered text or OCR annotations.
    image = (torch.sin(xx.float() * 0.31) + torch.cos(yy.float() * 0.47))[None, None]
    model = TinyJointPage(**{key: config[key] for key in (
        "num_queries", "channels", "alphabet_size", "band_rows", "sequence_steps",
        "baseline_fraction",
    )})
    targets = PageTargets(
        baseline_xy=torch.tensor([[[3., 7.], [28., 7.]], [[3., 17.], [28., 17.]]]),
        height_px=torch.tensor([4., 4.]),
        texts=((1, 2), (2, 3)),
        order=torch.tensor([[-1, 1], [0, -1]]),
    )
    predictions, diagnostics, losses_by_mode, matches = {}, {}, {}, {}
    for mode, detach in (("B3_detached", True), ("B4_coupled", False)):
        pred = model(image, detach_geometry=detach)
        losses = page_losses(pred, targets)
        predictions[mode] = pred
        matches[mode] = list(match_lines(pred, targets))
        losses_by_mode[mode] = {name: float(value.detach()) for name, value in losses.items()}
        diagnostics[mode] = gradient_diagnostics(losses["ctc"], pred, model)
    detached, coupled = predictions["B3_detached"], predictions["B4_coupled"]
    coupled_losses = page_losses(coupled, targets)
    auxiliary = {
        key: gradient_diagnostics(coupled_losses[key], coupled, model)
        for key in ("geometry", "presence", "order")
    }
    # Export every query to inspect geometry without tuning presence thresholds.
    page = export_page(
        coupled, image_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
        page_key="synthetic:joint-page-17", document_key="synthetic:joint-issue",
        presence_threshold=0.0,
    )
    # There is no encoded image in this experiment: this hash identifies the
    # actual model input, not PNG/JPEG source bytes required by the pipeline.
    page["image_sha256_scope"] = "synthetic_float32_tensor_bytes_C_order_not_encoded_image"
    page["input_tensor"] = {"shape": list(image.shape), "dtype": str(image.dtype),
                            "byte_order": sys.byteorder}
    checks = {
        "identical_forward": bool(torch.equal(detached.ctc_logits, coupled.ctc_logits)),
        "identical_matching": matches["B3_detached"] == matches["B4_coupled"],
        "ctc_coupled_geometry": all(diagnostics["B4_coupled"][key]["l2"] > 0
                                    for key in ("baseline", "height", "geometry_head")),
        "ctc_detached_geometry": all(not diagnostics["B3_detached"][key]["connected"]
                                     for key in ("baseline", "height", "geometry_head")),
        "ctc_features_both_modes": all(diagnostics[mode][key]["l2"] > 0
                                       for mode in diagnostics for key in ("features", "encoder")),
        "geometry_head_supervised": auxiliary["geometry"]["geometry_head"]["l2"] > 0,
        "all_losses_finite": all(math.isfinite(v) for row in losses_by_mode.values() for v in row.values()),
        "left_to_right_horizontal": bool((coupled.baseline_xy[:, 1, 0] > coupled.baseline_xy[:, 0, 0]).all()
                                         and torch.equal(coupled.baseline_xy[:, 1, 1], coupled.baseline_xy[:, 0, 1])),
        "pixel_grid_inside_page": bool((coupled.pixel_xy[..., 0] >= 0).all()
                                       and (coupled.pixel_xy[..., 0] <= width - 1).all()
                                       and (coupled.pixel_xy[..., 1] >= 0).all()
                                       and (coupled.pixel_xy[..., 1] <= height - 1).all()),
    }
    return {"result": "PASS_WIRING_ONLY" if all(checks.values()) else "FAIL",
            "checks": checks, "ctc_gradients": diagnostics,
            "auxiliary_gradients": auxiliary, "losses": losses_by_mode,
            "matching": matches, "parameters": sum(p.numel() for p in model.parameters()),
            "export": page}


def drift_witness(config):
    cfg = config["drift_witness"]
    yy, xx = torch.meshgrid(torch.arange(16, dtype=torch.float64),
                            torch.arange(16, dtype=torch.float64), indexing="ij")
    y_evidence = cfg["evidence_center_y"]
    field = torch.stack((torch.zeros_like(xx),
                         5 - .15 * (xx - 4).square() - .3 * (yy - y_evidence).square(),
                         5 - .15 * (xx - 11).square() - .3 * (yy - y_evidence).square()))[None]
    frame = FeatureFrame((16, 16), (0., 0.), (1., 1.))
    height = torch.tensor([cfg["height_px"]], dtype=torch.float64)
    x = torch.tensor([2., 13.], dtype=torch.float64)
    target_baseline = torch.stack((x, torch.full_like(x, cfg["initial_baseline_y"])), dim=-1)[None]

    def losses_at(y, detach):
        baseline = torch.stack((x, y.expand(2)), dim=-1)[None]
        bands = sample_baseline_bands(field, baseline, height, frame,
                                     output_size=(3, 12), detach_geometry=detach,
                                     baseline_fraction=config["baseline_fraction"])
        logits = bands.features.mean(dim=2).permute(2, 0, 1)
        ctc = torch.nn.functional.ctc_loss(
            logits.log_softmax(-1), torch.tensor([1, 2]), torch.tensor([12]),
            torch.tensor([2]), blank=0, reduction="mean", zero_infinity=False,
        )
        geom = geometry_loss(baseline, height, target_baseline, height,
                             coordinate_scale_px=cfg["coordinate_scale_px"]).total
        return ctc, geom

    arms = {}
    for name, detach, weight in (("B3_detached", True, cfg["geometry_weight"]),
                                 ("B4_coupled", False, cfg["geometry_weight"]),
                                 ("B4g_no_geometry", False, 0.0)):
        y = torch.tensor(cfg["initial_baseline_y"], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([y], lr=cfg["learning_rate"])
        initial_ctc, _ = losses_at(y, detach)
        for _ in range(cfg["steps"]):
            optimizer.zero_grad()
            ctc, geom = losses_at(y, detach)
            objective = ctc + weight * geom
            if not bool(torch.isfinite(objective)):
                raise ValueError("Nonfinite constructed-witness loss")
            objective.backward()
            optimizer.step()
        final_ctc, final_geom = losses_at(y, detach)
        arms[name] = {"final_baseline_y": float(y.detach()),
                      "displacement_px": abs(float(y.detach()) - cfg["initial_baseline_y"]),
                      "initial_ctc": float(initial_ctc.detach()), "final_ctc": float(final_ctc.detach()),
                      "final_geometry_loss": float(final_geom.detach()),
                      "geometry_weight": weight, "steps": cfg["steps"]}
    checks = {
        "detached_no_displacement": arms["B3_detached"]["displacement_px"] == 0.0,
        "unregularized_drift": arms["B4g_no_geometry"]["displacement_px"] > cfg["min_unregularized_displacement_px"],
        "regularized_bounded": arms["B4_coupled"]["displacement_px"] < cfg["max_regularized_displacement_px"],
        "ctc_can_improve_while_geometry_worsens": (
            arms["B4g_no_geometry"]["final_ctc"] < arms["B4g_no_geometry"]["initial_ctc"]
            and arms["B4g_no_geometry"]["displacement_px"] > 0
            and arms["B4g_no_geometry"]["final_geometry_loss"] > 0
        ),
    }
    return {"result": "PASS_CONSTRUCTED_WITNESS_ONLY" if all(checks.values()) else "FAIL_PREDECLARED_CHECK",
            "interpretation": "Constructed conflicting evidence, not OCR or proof of universal regularization benefit.",
            "trainable": "one scalar baseline y; features, height and reader fixed",
            "checks": checks, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/joint_page_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/joint_page_probe.json")
    args = parser.parse_args()
    started = time.monotonic()
    report = {"schema_version": "joint_page_probe_v1", "result": "FAIL",
              "created_at": datetime.now(timezone.utc).isoformat(), "synthetic_only": True,
              "device": "cpu", "ocr_performance_claim": False, "pretrained_weights": False,
              "torch_version": torch.__version__, "python_executable": sys.executable}
    report["changes_after_initial_v1_observation"] = [
        "The CTC-improves/geometry-worsens check was strengthened after review to require positive displacement and geometry loss as well as lower CTC. It does not change the FAIL verdict.",
        "The predeclared drift threshold, optimizer, number of steps and all configuration values are unchanged.",
        "Export now declares pixel-center coordinates, input tensor hash scope and rejects inconsistent sampling geometry; ambiguous order has no definitive linearization.",
    ]
    try:
        raw = args.config.read_bytes()
        config = json.loads(raw)
        if config["device"] != "cpu" or config["threads"] != 1 or config["synthetic_only"] is not True:
            raise ValueError("Probe only permits single-thread synthetic CPU work")
        if config["image_size_wh"] != [32, 24] or not 1 <= config["drift_witness"]["steps"] <= 100:
            raise ValueError("Probe dimensions and step budget are intentionally bounded")
        if not 1 <= config["num_queries"] <= 6 or not 2 <= config["channels"] <= 16:
            raise ValueError("Probe query/channel budget exceeded")
        if not 2 <= config["sequence_steps"] <= 32 or not 2 <= config["band_rows"] <= 8:
            raise ValueError("Probe sampling budget exceeded")
        torch.set_num_threads(1)
        report["config"] = config
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        report["source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "src/babaorum/experimental/joint_page.py",
                         ROOT / "src/babaorum/experimental/joint_geometry.py")
        }
        report["page_wiring"] = page_wiring(config)
        report["drift_witness"] = drift_witness(config)
        passed = all(value for section in ("page_wiring", "drift_witness")
                     for value in report[section]["checks"].values())
        report["result"] = "PASS_ENGINEERING_FEASIBILITY_ONLY" if passed else "FAIL"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"result": report["result"], "output": str(args.output),
                      "error": report.get("error"), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["result"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
