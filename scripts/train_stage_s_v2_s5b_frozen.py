#!/usr/bin/env python3
"""Stage S smoke training and evaluation, protocol v2 (step S-L5b). SMOKE_ONLY_NO_CLAIM.

Differences from train_stage_s.py (v1, kept unchanged):
- 768-px tiles, optionally centred on a drawn readable line (line_centered_windows);
- detection targets from every line intersecting the tile, baselines clipped to the tile;
- geometry and CTC losses only on the lines entirely inside the tile;
- evaluation on the full scaled page, with all page lines as ground truth.
Helpers shared with v1 are imported from train_stage_s.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_stage_s import (  # noqa: E402
    MODULES, detection_loss, encode_text, greedy_decode, levenshtein, load_scaled_page,
    lines_in_tile, perturb_baseline, sha256, sigma_for_step,
)
from babaorum.experimental.joint_curves import curve_geometry_loss  # noqa: E402
from babaorum.experimental.stage_s_arms import StageSModel  # noqa: E402
from babaorum.experimental.stage_s_data import build_alphabet  # noqa: E402
from babaorum.experimental.stage_s_model import refine_lines, resample_k  # noqa: E402
from babaorum.experimental.stage_s_reader import band_features  # noqa: E402
from babaorum.experimental.stage_s_targets import extract_polylines, rasterize_baselines  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

DRAW_TRIES = 10


def clip_polyline_to_box(baseline, x0, y0, size):
    """Pieces of a polyline inside the box [x0, x0+size] x [y0, y0+size], dense sampling at 1 px."""
    b = np.asarray(baseline, dtype=np.float64)
    pts = []
    for (ax, ay), (bx, by) in zip(b[:-1], b[1:]):
        n = max(2, int(np.hypot(bx - ax, by - ay)) + 1)
        t = np.linspace(0.0, 1.0, n)
        seg = np.stack((ax + t * (bx - ax), ay + t * (by - ay)), axis=-1)
        pts.append(seg if not pts else seg[1:])
    dense = np.concatenate(pts, axis=0)
    inside = (dense[:, 0] >= x0) & (dense[:, 0] <= x0 + size) & (dense[:, 1] >= y0) & (dense[:, 1] <= y0 + size)
    pieces, run = [], []
    for p, ok in zip(dense, inside):
        if ok:
            run.append(p)
        elif run:
            pieces.append(run)
            run = []
    if run:
        pieces.append(run)
    return [[[float(x - x0), float(y - y0)] for x, y in piece] for piece in pieces if len(piece) >= 2]


def intersects_tile(line, x0, y0, size):
    b = np.asarray(line["baseline"], dtype=np.float64)
    ymin = b[:, 1].min() - line["normal_negative_px"]
    ymax = b[:, 1].max() + line["normal_positive_px"]
    return not (b[:, 0].max() < x0 or b[:, 0].min() > x0 + size or ymax < y0 or ymin > y0 + size)


def draw_window(page, rng, tile, centred):
    """Window origin and tile lines. Centred mode centres on a readable line and keeps it entirely inside."""
    h_img, w_img = page["image"].shape
    readable = [ln for ln in page["lines"] if ln["legibility"] == "readable" and ln["text"]]
    if centred and readable:
        for _ in range(DRAW_TRIES):
            ln = readable[int(rng.integers(len(readable)))]
            b = np.asarray(ln["baseline"], dtype=np.float64)
            cx, cy = b[:, 0].mean(), b[:, 1].mean()
            x0 = int(np.clip(round(cx - tile / 2), 0, w_img - tile))
            y0 = int(np.clip(round(cy - tile / 2), 0, h_img - tile))
            inside = lines_in_tile(page["lines"], x0, y0, tile)
            if any(i["id"] == ln["id"] for i in inside):
                return x0, y0
    x0 = int(rng.integers(0, w_img - tile + 1))
    y0 = int(rng.integers(0, h_img - tile + 1))
    return x0, y0


def train_v2(model, optimizer, pages, char_to_idx, cfg, rng, steps, target_h, log_stop=None):
    rows = []
    tile = cfg["tiles"]["size_px"]
    centred = cfg["tiles"]["line_centered_windows"]
    for step in range(steps):
        t0 = time.monotonic()
        page = pages[int(rng.integers(len(pages)))]
        x0, y0 = draw_window(page, rng, tile, centred)
        image = torch.as_tensor(page["image"][y0:y0 + tile, x0:x0 + tile])[None, None]

        detect_lines, intersecting_ids = [], set()
        for ln in page["lines"]:
            if not intersects_tile(ln, x0, y0, tile):
                continue
            intersecting_ids.add(ln["id"])
            for piece in clip_polyline_to_box(ln["baseline"], x0, y0, tile):
                detect_lines.append({"baseline": piece, "normal_negative_px": ln["normal_negative_px"],
                                     "normal_positive_px": ln["normal_positive_px"]})
        raster = rasterize_baselines(detect_lines, (tile, tile), stride=cfg["model"]["stride"])

        inside = lines_in_tile(page["lines"], x0, y0, tile)
        readable = [ln for ln in inside if ln["legibility"] == "readable" and ln["text"]]
        take = min(cfg["ctc_lines_per_step_max"], len(readable))
        chosen = [readable[i] for i in rng.permutation(len(readable))[:take]] if take else []
        polylines = [perturb_baseline(ln["baseline"], ln["normal_negative_px"], ln["normal_positive_px"], rng)
                     for ln in chosen]
        labels = [encode_text(ln["text"], char_to_idx) for ln in chosen]

        sigma = sigma_for_step(step, steps, target_h)
        out = model.forward_lines(image, polylines, labels, sigma_px=sigma)
        l_det = detection_loss(out["det"], raster)

        if chosen:
            geo_terms = []
            for i, ln in enumerate(chosen):
                gt = torch.as_tensor(resample_k(ln["baseline"], model.K), dtype=torch.float32)
                h_gt = ln["normal_negative_px"] + ln["normal_positive_px"]
                geo_terms.append(curve_geometry_loss(out["refined"]["baseline"][i], gt,
                                                     target_height_px=h_gt, spacing_px=4.0).total)
            l_geo = sum(geo_terms) / len(geo_terms)
        else:
            l_geo = torch.zeros(())

        included = [x for x in out["ctc_losses"] if x is not None]
        counts = {}
        for s in out["statuses"]:
            counts[s] = counts.get(s, 0) + 1
        l_ctc = sum(included) / len(included) if included else torch.zeros(())

        total = l_det + l_geo + l_ctc
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        seconds = time.monotonic() - t0
        rows.append({"step": step, "page_id": page["page_id"], "window_xy": [x0, y0],
                     "sigma_px": sigma, "n_detect_pieces": len(detect_lines),
                     "n_lines_intersecting": len(intersecting_ids),
                     "n_lines_inside": len(inside),
                     "L_det": float(l_det.detach()), "L_geo": float(l_geo.detach()),
                     "L_ctc": float(l_ctc.detach()), "L_total": float(total.detach()),
                     "n_ctc_included": len(included), "n_ctc_excluded": len(out["statuses"]) - len(included),
                     "status_counts": counts, "seconds": seconds})
        if log_stop is not None and log_stop():
            break
    return rows


def evaluate_full_page(model, page, cfg):
    """Full scaled page: all lines are ground truth; before/after refinement; CER after."""
    t0 = time.monotonic()
    image = torch.as_tensor(page["image"])[None, None]
    gts = [GroundTruthLine(id=ln["id"], baseline=ln["baseline"],
                           height_px=ln["normal_negative_px"] + ln["normal_positive_px"],
                           baseline_source="ground_truth") for ln in page["lines"]]
    gt_by_id = {g.id: ln for g, ln in zip(gts, page["lines"])}
    model.eval()
    with torch.no_grad():
        features = model.enc_det(image)
        logit = model.head_d(features)[0]
        prob = torch.sigmoid(logit)[0, 0].numpy()
        extracted = [p for p in extract_polylines(prob, stride=cfg["extraction"]["stride"],
                                                  threshold=cfg["extraction"]["threshold"],
                                                  min_cells=cfg["extraction"]["min_cells"]) if len(p) >= 2]
        before = [PredictedLine(id=f"p{i:05d}", baseline=p, baseline_source="predicted")
                  for i, p in enumerate(extracted)]
        after, refined = [], None
        if extracted:
            refined = refine_lines(features, extracted, model.head_g, head_d=model.head_d, K=model.K)
            after = [PredictedLine(id=f"p{i:05d}", baseline=poly, baseline_source="predicted")
                     for i, poly in enumerate(refined["baseline"].tolist())]
        features_read = model.enc_read(image) if model.arm == "S2" else features
        alphabet = page["alphabet"]

    metrics = {}
    for tag, preds in (("before", before), ("after", after)):
        res = evaluate_line_geometry(gts, preds) if preds else None
        if res is None:
            metrics[tag] = {"recall": 0.0, "precision": None, "matched": 0, "n_gt": len(gts), "n_pred": 0,
                            "mean_normalized_distance": None}
            continue
        norm = [m.normalized_symmetric_distance for m in res.matches]
        metrics[tag] = {"recall": res.recall, "precision": res.precision, "matched": res.matched_count,
                        "n_gt": res.n_gt, "n_pred": res.n_pred,
                        "mean_normalized_distance": (sum(norm) / len(norm)) if norm else None}
        if tag == "after":
            pred_index = {p.id: i for i, p in enumerate(preds)}
            edits = total_len = 0
            with torch.no_grad():
                for m in res.matches:
                    ln = gt_by_id[m.gt_id]
                    if ln["legibility"] != "readable" or not ln["text"]:
                        continue
                    i = pred_index[m.pred_id]
                    band = band_features(features_read, refined["baseline"][i], refined["neg"][i], refined["pos"][i])
                    if band is None:
                        continue
                    hyp = greedy_decode(model.reader(band), alphabet)
                    edits += levenshtein(ln["text"], hyp)
                    total_len += len(ln["text"])
            metrics["cer_after_readable_matched"] = (edits / total_len) if total_len else None
    return {"page_id": page["page_id"], "n_gt_lines": len(gts), "n_extracted": len(extracted),
            "metrics": metrics, "seconds": time.monotonic() - t0}


def run_v2(cfg, arm, output, *, train_pages=None, eval_pages=None, steps=None, root=None):
    started = time.monotonic()
    root = Path(root or Path(cfg["source"]["root"]).expanduser())
    target_h = cfg["scale"]["target_median_height_px"]
    train_ids = train_pages or cfg["split"]["train_pages"]
    eval_ids = eval_pages or cfg["split"]["eval_pages"]
    steps = steps or cfg["steps"]

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    torch.set_num_threads(cfg["threads"])

    train_loaded = [load_scaled_page(root, pid, target_h) for pid in train_ids]
    eval_loaded = [load_scaled_page(root, pid, target_h) for pid in eval_ids]
    alphabet = build_alphabet(train_loaded)
    char_to_idx = {c: i for i, c in enumerate(alphabet) if i > 0}
    for p in train_loaded + eval_loaded:
        p["alphabet"] = alphabet

    model = StageSModel(arm, len(alphabet), channels=cfg["model"]["channels"], K=cfg["model"]["K"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["learning_rate"])
    limit = cfg["time_limit_seconds_per_run"]
    stop = {"hit": False}

    def time_up():
        stop["hit"] = (time.monotonic() - started) > limit
        return stop["hit"]

    t_train = time.monotonic()
    rows = train_v2(model, optimizer, train_loaded, char_to_idx, cfg, rng, steps, target_h, log_stop=time_up)
    train_seconds = time.monotonic() - t_train

    evals = []
    if not stop["hit"]:
        for p in eval_loaded:
            evals.append(evaluate_full_page(model, p, cfg))
            if time.monotonic() - started > limit:
                stop["hit"] = True
                break

    cer_values = [e["metrics"]["cer_after_readable_matched"] for e in evals
                  if e["metrics"].get("cer_after_readable_matched") is not None]
    secs = [r["seconds"] for r in rows]
    report = {
        "schema_version": "stage_s_smoke_report_v2",
        "status": "SMOKE_ONLY_NO_CLAIM" if not stop["hit"] else "STOPPED_TIME_LIMIT_SMOKE_ONLY",
        "arm": arm,
        "protocol": "stage_s_smoke_config_v2",
        "tile_px": cfg["tiles"]["size_px"],
        "line_centered_windows": cfg["tiles"]["line_centered_windows"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps_requested": steps,
        "steps_completed": len(rows),
        "time_limit_seconds": limit,
        "elapsed_seconds": time.monotonic() - started,
        "train_seconds": train_seconds,
        "seconds_per_step_mean": (sum(secs) / len(secs)) if secs else None,
        "scale_factors": {p["page_id"]: p["factor"] for p in train_loaded + eval_loaded},
        "alphabet_size": len(alphabet),
        "steps": rows,
        "eval": evals,
        "cer_after_mean": (sum(cer_values) / len(cer_values)) if cer_values else None,
        "seconds_per_eval_page_mean": (sum(e["seconds"] for e in evals) / len(evals)) if evals else None,
        "provenance": {
            "config_v2_sha256": sha256(ROOT / "config/stage_s_smoke_v2.json"),
            "script_sha256": sha256(Path(__file__).resolve()),
            "helper_script_sha256": sha256(ROOT / "scripts/train_stage_s.py"),
            "diagnostic_sha256": sha256(ROOT / "reports/stage_s_scale_diagnostic.json"),
            "module_sha256": {m: sha256(ROOT / f"src/babaorum/experimental/{m}.py") for m in MODULES},
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform(), "threads": cfg["threads"]},
        "limitations": [
            "smoke run: a few CPU steps, untrained reader, no OCR or performance claim",
            "eval pages are smoke-only; the sealed volet 2 is not opened",
            "geometry and CTC losses use the up to 8 CTC-selected lines per step, drawn from the entirely-inside lines",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage_s_smoke_v2.json")
    parser.add_argument("--arm", required=True, choices=("S2", "B3", "B4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    report = run_v2(cfg, args.arm, args.output)
    print(json.dumps({"arm": args.arm, "status": report["status"], "steps_completed": report["steps_completed"],
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
