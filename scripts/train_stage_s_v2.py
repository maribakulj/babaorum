#!/usr/bin/env python3
"""Stage S training and evaluation, protocol v2 (steps S-L5b and S-L6a). CPU or CUDA.

Logic as validated in S-L5b (frozen copy: train_stage_s_v2_s5b_frozen.py):
- 768-px tiles, optionally centred on a drawn readable line (line_centered_windows);
- detection targets from every line intersecting the tile, baselines clipped to the tile;
- geometry and CTC losses only on the lines entirely inside the tile;
- evaluation on the full scaled page, with all page lines as ground truth.
S-L6a adds: --device, --steps, --time-cap-min, --eval-every, --eval-pages,
--train-pages, --root, --log-dir (checkpoint, JSONL journal every 100 steps).
Helpers shared with v1 are imported from train_stage_s.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_stage_s import (  # noqa: E402
    MODULES, encode_text, greedy_decode, levenshtein, load_scaled_page,
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
JOURNAL_EVERY = 100
GPU_CALIBRATION_CONFIG = ROOT / "config/stage_s_gpu_calibration.json"


def load_cap_scope() -> dict:
    """Final-evaluation ratio and hard-kill margin, read from the pre-registration cap_scope text."""
    text = json.loads(GPU_CALIBRATION_CONFIG.read_text())["schedule"]["cap_scope"]
    ratio = float(re.search(r"(\d+(?:\.\d+)?) x the duration", text).group(1))
    kill_margin = float(re.search(r"cap \+ (\d+(?:\.\d+)?) min", text).group(1))
    return {"final_eval_min_ratio": ratio, "kill_margin_min": kill_margin}


def final_status_word(final_status: str, stopped: bool, device, time_cap_min) -> str:
    if final_status == "FINAL_EVAL_SKIPPED_BUDGET":
        return final_status
    budgeted = device.type == "cuda" or time_cap_min is not None
    if stopped:
        return "STOPPED_TIME_CAP" if budgeted else "STOPPED_TIME_LIMIT_SMOKE_ONLY"
    return "COMPLETED" if budgeted else "SMOKE_ONLY_NO_CLAIM"


def final_eval_allowed(remaining_s: float, last_eval_s, ratio: float) -> bool:
    """Run the final evaluation only if the remaining cap covers ratio x the last periodic evaluation."""
    if last_eval_s is None:
        return remaining_s > 0
    return remaining_s >= ratio * last_eval_s


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
    """Window origin. Centred mode centres on a readable line and keeps it entirely inside."""
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


def detection_loss_dev(det, raster, device):
    """Same objective as train_stage_s.detection_loss, tensors placed on `device`."""
    logit = det["logit"]
    tgt = torch.as_tensor(raster["baseline"], device=device)[None, None]
    n_pos = float(tgt.sum())
    n_neg = float(tgt.numel()) - n_pos
    pos_weight = torch.tensor(min(50.0, n_neg / n_pos) if n_pos > 0 else 1.0, device=device)
    bce = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pos_weight)
    mask = torch.as_tensor(raster["mask"], dtype=torch.float32, device=device)[None, None]
    denom = float(mask.sum())
    if denom > 0:
        tn = torch.as_tensor(raster["neg"], device=device)[None, None]
        tp = torch.as_tensor(raster["pos"], device=device)[None, None]
        l1 = ((det["neg"] - tn).abs() * mask).sum() / denom + ((det["pos"] - tp).abs() * mask).sum() / denom
    else:
        l1 = torch.zeros((), device=device)
    return bce + l1


def peak_gpu_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return None


def train_v2(model, optimizer, pages, char_to_idx, cfg, rng, steps, target_h, device,
             stop_check=None, on_step=None):
    rows = []
    tile = cfg["tiles"]["size_px"]
    centred = cfg["tiles"]["line_centered_windows"]
    for step in range(steps):
        t0 = time.monotonic()
        page = pages[int(rng.integers(len(pages)))]
        x0, y0 = draw_window(page, rng, tile, centred)
        image = torch.as_tensor(page["image"][y0:y0 + tile, x0:x0 + tile], device=device)[None, None]

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
        l_det = detection_loss_dev(out["det"], raster, device)

        if chosen:
            geo_terms = []
            for i, ln in enumerate(chosen):
                gt = torch.as_tensor(resample_k(ln["baseline"], model.K), dtype=torch.float32, device=device)
                h_gt = ln["normal_negative_px"] + ln["normal_positive_px"]
                geo_terms.append(curve_geometry_loss(out["refined"]["baseline"][i], gt,
                                                     target_height_px=h_gt, spacing_px=4.0).total)
            l_geo = sum(geo_terms) / len(geo_terms)
        else:
            l_geo = torch.zeros((), device=device)

        included = [x for x in out["ctc_losses"] if x is not None]
        counts = {}
        for s in out["statuses"]:
            counts[s] = counts.get(s, 0) + 1
        l_ctc = sum(included) / len(included) if included else torch.zeros((), device=device)

        total = l_det + l_geo + l_ctc
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - t0
        rows.append({"step": step + 1, "page_id": page["page_id"], "window_xy": [x0, y0],
                     "sigma_px": sigma, "n_detect_pieces": len(detect_lines),
                     "n_lines_intersecting": len(intersecting_ids), "n_lines_inside": len(inside),
                     "L_det": float(l_det.detach()), "L_geo": float(l_geo.detach()),
                     "L_ctc": float(l_ctc.detach()), "L_total": float(total.detach()),
                     "n_ctc_included": len(included), "n_ctc_excluded": len(out["statuses"]) - len(included),
                     "status_counts": counts, "seconds": seconds})
        if on_step is not None:
            on_step(rows)
        if stop_check is not None and stop_check():
            break
    return rows


def evaluate_full_page(model, page, cfg, device):
    """Full scaled page: all lines are ground truth; before/after refinement; CER after."""
    t0 = time.monotonic()
    image = torch.as_tensor(page["image"], device=device)[None, None]
    gts = [GroundTruthLine(id=ln["id"], baseline=ln["baseline"],
                           height_px=ln["normal_negative_px"] + ln["normal_positive_px"],
                           baseline_source="ground_truth") for ln in page["lines"]]
    gt_by_id = {g.id: ln for g, ln in zip(gts, page["lines"])}
    model.eval()
    with torch.no_grad():
        features = model.enc_det(image)
        logit = model.head_d(features)[0]
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
        extracted = [p for p in extract_polylines(prob, stride=cfg["extraction"]["stride"],
                                                  threshold=cfg["extraction"]["threshold"],
                                                  min_cells=cfg["extraction"]["min_cells"]) if len(p) >= 2]
        before = [PredictedLine(id=f"p{i:05d}", baseline=p, baseline_source="predicted")
                  for i, p in enumerate(extracted)]
        after, refined = [], None
        if extracted:
            refined = refine_lines(features, extracted, model.head_g, head_d=model.head_d, K=model.K)
            after = [PredictedLine(id=f"p{i:05d}", baseline=poly, baseline_source="predicted")
                     for i, poly in enumerate(refined["baseline"].cpu().tolist())]
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
                    hyp = greedy_decode(model.reader(band).cpu(), alphabet)
                    edits += levenshtein(ln["text"], hyp)
                    total_len += len(ln["text"])
            metrics["cer_after_readable_matched"] = (edits / total_len) if total_len else None
    model.train()
    return {"page_id": page["page_id"], "n_gt_lines": len(gts), "n_extracted": len(extracted),
            "metrics": metrics, "seconds": time.monotonic() - t0}


def evaluate_all(model, eval_pages, cfg, device):
    return [evaluate_full_page(model, p, cfg, device) for p in eval_pages]


def run_v2(cfg, arm, output, *, train_pages=None, eval_pages=None, steps=None, root=None,
           device_name="cpu", time_cap_min=None, eval_every=0, log_dir=None, ctc_det_grad_scale=1.0):
    started = time.monotonic()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda requested but torch.cuda is not available")
    root = Path(root or Path(cfg["source"]["root"]).expanduser())
    target_h = cfg["scale"]["target_median_height_px"]
    train_ids = train_pages or cfg["split"]["train_pages"]
    eval_ids = eval_pages or cfg["split"]["eval_pages"]
    steps = steps or cfg["steps"]
    cap_seconds = time_cap_min * 60.0 if time_cap_min is not None else cfg["time_limit_seconds_per_run"]
    cap_scope = load_cap_scope()
    log_dir = Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_loaded = [load_scaled_page(root, pid, target_h) for pid in train_ids]
    eval_loaded = [load_scaled_page(root, pid, target_h) for pid in eval_ids]
    alphabet = build_alphabet(train_loaded)
    char_to_idx = {c: i for i, c in enumerate(alphabet) if i > 0}
    for p in train_loaded + eval_loaded:
        p["alphabet"] = alphabet

    model = StageSModel(arm, len(alphabet), channels=cfg["model"]["channels"], K=cfg["model"]["K"],
                        hidden=cfg["model"].get("hidden", 128),
                        ctc_det_grad_scale=ctc_det_grad_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["learning_rate"])

    journal_path = log_dir / "journal.jsonl" if log_dir else None
    if journal_path and journal_path.exists():
        journal_path.unlink()

    def write_journal(rows):
        block = rows[-JOURNAL_EVERY:]
        entry = {"arm": arm, "step": rows[-1]["step"], "elapsed_s": round(time.monotonic() - started, 1),
                 "mean_L_det": float(np.mean([r["L_det"] for r in block])),
                 "mean_L_geo": float(np.mean([r["L_geo"] for r in block])),
                 "mean_L_ctc": float(np.mean([r["L_ctc"] for r in block])),
                 "mean_seconds_per_step": float(np.mean([r["seconds"] for r in block])),
                 "gpu_peak_mb": peak_gpu_mb(device)}
        if journal_path:
            with journal_path.open("a") as f:
                f.write(json.dumps(entry, allow_nan=False) + "\n")

    def save_checkpoint(step):
        if log_dir:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "arm": arm}, log_dir / "checkpoint_last.pt")

    timeline = []
    stop = {"hit": False}

    def time_up():
        stop["hit"] = (time.monotonic() - started) > cap_seconds
        return stop["hit"]

    def maybe_eval(rows):
        step = rows[-1]["step"]
        if eval_every and step % eval_every == 0 and step < steps:
            t_eval = time.monotonic()
            evs = evaluate_all(model, eval_loaded, cfg, device)
            timeline.append({"step": step, "evals": evs, "eval_seconds": time.monotonic() - t_eval})
            save_checkpoint(step)

    def on_step(rows):
        if rows[-1]["step"] % JOURNAL_EVERY == 0:
            write_journal(rows)
        maybe_eval(rows)

    t_train = time.monotonic()
    rows = train_v2(model, optimizer, train_loaded, char_to_idx, cfg, rng, steps, target_h, device,
                    stop_check=time_up, on_step=on_step)
    train_seconds = time.monotonic() - t_train
    if rows and len(rows) % JOURNAL_EVERY != 0:
        write_journal(rows)
    save_checkpoint(rows[-1]["step"] if rows else 0)

    periodic = [t for t in timeline if "eval_seconds" in t]
    last_eval_s = periodic[-1]["eval_seconds"] if periodic else None
    remaining_s = cap_seconds - (time.monotonic() - started)
    final_runs = final_eval_allowed(remaining_s, last_eval_s, cap_scope["final_eval_min_ratio"])
    if final_runs:
        t_eval = time.monotonic()
        evals = evaluate_all(model, eval_loaded, cfg, device)
        timeline.append({"step": rows[-1]["step"] if rows else 0, "evals": evals, "final": True,
                         "eval_seconds": time.monotonic() - t_eval})
        save_checkpoint(rows[-1]["step"] if rows else 0)
    else:
        evals = []
    criterion_step = timeline[-1]["step"] if timeline else None
    final_status = "RUN" if final_runs else "FINAL_EVAL_SKIPPED_BUDGET"

    cer_values = [e["metrics"]["cer_after_readable_matched"] for e in evals
                  if e["metrics"].get("cer_after_readable_matched") is not None]
    secs = [r["seconds"] for r in rows]
    med_tail = [r["seconds"] for r in rows if r["step"] >= 100]
    stopped = stop["hit"] or len(rows) < steps
    report = {
        "schema_version": "stage_s_run_report_v2",
        "status": (final_status_word(final_status, stopped, device, time_cap_min)),
        "final_eval": final_status,
        "criterion_eval_step": criterion_step,
        "cap_scope": cap_scope,
        "remaining_s_before_final_eval": remaining_s,
        "last_periodic_eval_seconds": last_eval_s,
        "arm": arm, "device": device_name, "ctc_det_grad_scale": float(ctc_det_grad_scale),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_peak_mb": peak_gpu_mb(device),
        "protocol": "stage_s_smoke_config_v2",
        "tile_px": cfg["tiles"]["size_px"], "line_centered_windows": cfg["tiles"]["line_centered_windows"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps_requested": steps, "steps_completed": len(rows),
        "time_cap_seconds": cap_seconds, "elapsed_seconds": time.monotonic() - started,
        "train_seconds": train_seconds,
        "seconds_per_step_mean": (sum(secs) / len(secs)) if secs else None,
        "seconds_per_step_median_100_to_end": (float(np.median(med_tail)) if med_tail else None),
        "scale_factors": {p["page_id"]: p["factor"] for p in train_loaded + eval_loaded},
        "alphabet_size": len(alphabet),
        "train_pages": train_ids, "eval_pages": eval_ids,
        "steps": rows,
        "timeline": timeline,
        "eval": evals,
        "cer_after_mean": (sum(cer_values) / len(cer_values)) if cer_values else None,
        "seconds_per_eval_page_mean": (sum(e["seconds"] for e in evals) / len(evals)) if evals else None,
        "provenance": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "helper_script_sha256": sha256(ROOT / "scripts/train_stage_s.py"),
            "config_sha256": cfg.get("_config_sha256"),
            "module_sha256": {m: sha256(ROOT / f"src/babaorum/experimental/{m}.py") for m in MODULES},
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform(), "threads": cfg["threads"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage_s_smoke_v2.json")
    parser.add_argument("--arm", required=True, choices=("S2", "B3", "B4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--time-cap-min", type=float)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-pages", nargs="+")
    parser.add_argument("--train-pages", nargs="+")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--ctc-det-grad-scale", type=float, default=1.0,
                        help="multiplies the L_ctc gradient entering enc_det (B3, B4); 1.0 = unchanged (E2E-5: 1/6)")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    cfg["_config_sha256"] = sha256(args.config)
    report = run_v2(cfg, args.arm, args.output, train_pages=args.train_pages, eval_pages=args.eval_pages,
                    steps=args.steps, root=args.root, device_name=args.device,
                    time_cap_min=args.time_cap_min, eval_every=args.eval_every, log_dir=args.log_dir,
                    ctc_det_grad_scale=args.ctc_det_grad_scale)
    print(json.dumps({"arm": args.arm, "device": args.device, "status": report["status"],
                      "steps_completed": report["steps_completed"],
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}))
    return report


if __name__ == "__main__":
    main()
