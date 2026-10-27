#!/usr/bin/env python3
"""Stage S smoke training and evaluation (step S-L5). SMOKE_ONLY_NO_CLAIM.

Reads the pilot pages in place, rescales each page so the median band height is
the configured target, trains one arm for a few CPU steps, then evaluates the
central tile of each eval page before and after refinement. No claim of OCR
performance is made by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.experimental.joint_curves import curve_geometry_loss  # noqa: E402
from babaorum.experimental.stage_s_arms import StageSModel  # noqa: E402
from babaorum.experimental.stage_s_data import BLANK, build_alphabet, load_mf_page  # noqa: E402
from babaorum.experimental.stage_s_model import refine_lines, resample_k  # noqa: E402
from babaorum.experimental.stage_s_reader import band_features  # noqa: E402
from babaorum.experimental.stage_s_targets import extract_polylines, rasterize_baselines  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

MODULES = ("stage_s_data", "stage_s_targets", "stage_s_model", "stage_s_reader", "stage_s_arms")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scaled_page(root: Path, page_id: str, target_h: float):
    """Load a pilot page in place and rescale image and geometry in memory (no disk writes)."""
    page = load_mf_page(root / "pages" / f"{page_id}.json", root / "images" / f"{page_id}.png")
    heights = [line["normal_negative_px"] + line["normal_positive_px"] for line in page["lines"]]
    median = float(np.median(heights))
    factor = target_h / median
    image = page["image"]
    h, w = image.shape
    new_w, new_h = round(w * factor), round(h * factor)
    scaled = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0
    lines = []
    for line in page["lines"]:
        lines.append({
            "id": line["id"],
            "baseline": [[x * factor, y * factor] for x, y in line["baseline"]],
            "normal_negative_px": line["normal_negative_px"] * factor,
            "normal_positive_px": line["normal_positive_px"] * factor,
            "text": line["text"],
            "legibility": line["legibility"],
        })
    return {"page_id": page_id, "image": scaled, "lines": lines, "factor": factor,
            "median_height_before_px": median}


def lines_in_tile(lines, x0, y0, size, margin=8.0):
    """Lines whose baseline and band lie fully inside the tile, in tile coordinates."""
    out = []
    for line in lines:
        b = np.asarray(line["baseline"], dtype=np.float64)
        ymin = b[:, 1].min() - line["normal_negative_px"]
        ymax = b[:, 1].max() + line["normal_positive_px"]
        if (b[:, 0].min() < x0 + margin or b[:, 0].max() > x0 + size - margin
                or ymin < y0 + margin or ymax > y0 + size - margin):
            continue
        local = dict(line)
        local["baseline"] = [[x - x0, y - y0] for x, y in line["baseline"]]
        out.append(local)
    return out


def perturb_baseline(baseline_local, neg, pos, rng):
    """GT baseline shifted rigidly along the chord normal by N(0, (0.5 h)^2)."""
    h = neg + pos
    d = float(rng.normal(0.0, 0.5 * h))
    b = np.asarray(baseline_local, dtype=np.float64)
    t = b[-1] - b[0]
    t = t / np.linalg.norm(t)
    n = np.array([-t[1], t[0]])
    return (b + d * n[None, :]).tolist()


def encode_text(text, char_to_idx):
    return [char_to_idx[c] for c in text]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_decode(logits, alphabet):
    best = logits.argmax(dim=-1).tolist()
    out, prev = [], None
    for idx in best:
        if idx != prev and idx != 0:
            out.append(alphabet[idx])
        prev = idx
    return "".join(out)


def sigma_for_step(step, steps, target_h):
    """Geometric annealing 1 h -> 0.1 h over the steps (h = target median height)."""
    if steps <= 1:
        return target_h
    return target_h * (0.1 ** (step / (steps - 1)))


def detection_loss(det, raster):
    logit = det["logit"]
    tgt = torch.as_tensor(raster["baseline"])[None, None]
    n_pos = float(tgt.sum())
    n_neg = float(tgt.numel()) - n_pos
    pos_weight = torch.tensor(min(50.0, n_neg / n_pos) if n_pos > 0 else 1.0)
    bce = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pos_weight)
    mask = torch.as_tensor(raster["mask"], dtype=torch.float32)[None, None]
    denom = float(mask.sum())
    if denom > 0:
        tn = torch.as_tensor(raster["neg"])[None, None]
        tp = torch.as_tensor(raster["pos"])[None, None]
        l1 = ((det["neg"] - tn).abs() * mask).sum() / denom + ((det["pos"] - tp).abs() * mask).sum() / denom
    else:
        l1 = torch.zeros(())
    return bce + l1


def train(model, optimizer, pages, char_to_idx, cfg, rng, steps, target_h, log_stop=None):
    rows, log = [], []
    tile = cfg["tiles"]["size_px"]
    for step in range(steps):
        t0 = time.monotonic()
        lines = []
        for _ in range(10):
            page = pages[int(rng.integers(len(pages)))]
            h_img, w_img = page["image"].shape
            x0 = int(rng.integers(0, w_img - tile + 1))
            y0 = int(rng.integers(0, h_img - tile + 1))
            lines = lines_in_tile(page["lines"], x0, y0, tile)
            if lines:
                break
        image = torch.as_tensor(page["image"][y0:y0 + tile, x0:x0 + tile])[None, None]
        raster = rasterize_baselines(lines, (tile, tile), stride=cfg["model"]["stride"])

        readable = [ln for ln in lines if ln["legibility"] == "readable" and ln["text"]]
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
        statuses = out["statuses"]
        counts = {}
        for s in statuses:
            counts[s] = counts.get(s, 0) + 1
        l_ctc = sum(included) / len(included) if included else torch.zeros(())

        total = l_det + l_geo + l_ctc
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        seconds = time.monotonic() - t0
        rows.append({"step": step, "sigma_px": sigma,
                     "L_det": float(l_det.detach()), "L_geo": float(l_geo.detach()),
                     "L_ctc": float(l_ctc.detach()), "L_total": float(total.detach()),
                     "n_ctc_included": len(included),
                     "n_ctc_excluded": len(statuses) - len(included), "status_counts": counts,
                     "n_lines_in_tile": len(lines), "seconds": seconds})
        if log_stop is not None and log_stop():
            break
    return rows


def evaluate_page(model, page, cfg, target_h):
    tile = cfg["tiles"]["size_px"]
    h_img, w_img = page["image"].shape
    x0, y0 = (w_img - tile) // 2, (h_img - tile) // 2
    t0 = time.monotonic()
    lines = lines_in_tile(page["lines"], x0, y0, tile)
    image = torch.as_tensor(page["image"][y0:y0 + tile, x0:x0 + tile])[None, None]
    gts = [GroundTruthLine(id=ln["id"], baseline=ln["baseline"],
                           height_px=ln["normal_negative_px"] + ln["normal_positive_px"],
                           baseline_source="ground_truth") for ln in lines]
    model.eval()
    with torch.no_grad():
        features = model.enc_det(image)
        logit = model.head_d(features)[0]
        prob = torch.sigmoid(logit)[0, 0].numpy()
        extracted = [p for p in extract_polylines(prob, stride=cfg["extraction"]["stride"],
                                                  threshold=cfg["extraction"]["threshold"],
                                                  min_cells=cfg["extraction"]["min_cells"]) if len(p) >= 2]
        result = {"page_id": page["page_id"], "tile_xy": [x0, y0], "gt_lines_in_tile": len(gts),
                  "n_extracted": len(extracted)}
        before_preds = [PredictedLine(id=f"p{i:04d}", baseline=p, baseline_source="predicted")
                        for i, p in enumerate(extracted)]
        if extracted:
            refined = refine_lines(features, extracted, model.head_g, head_d=model.head_d, K=model.K)
            after_polys = refined["baseline"].tolist()
            after_preds = [PredictedLine(id=f"p{i:04d}", baseline=poly, baseline_source="predicted")
                           for i, poly in enumerate(after_polys)]
        else:
            refined, after_preds = None, []
        features_read = model.enc_read(image) if model.arm == "S2" else features

    alphabet = page["alphabet"]
    metrics = {}
    for tag, preds in (("before", before_preds), ("after", after_preds)):
        if not gts:
            metrics[tag] = None
            continue
        res = evaluate_line_geometry(gts, preds)
        norm = [m.normalized_symmetric_distance for m in res.matches]
        metrics[tag] = {"recall": res.recall, "precision": res.precision,
                        "matched": res.matched_count, "n_gt": res.n_gt, "n_pred": res.n_pred,
                        "mean_normalized_distance": (sum(norm) / len(norm)) if norm else None}
        if tag == "after":
            pred_index = {p.id: i for i, p in enumerate(preds)}
            gt_by_id = {g.id: ln for g, ln in zip(gts, lines)}
            edits = total_len = 0
            with torch.no_grad():
                for m in res.matches:
                    ln = gt_by_id[m.gt_id]
                    if ln["legibility"] != "readable" or not ln["text"]:
                        continue
                    i = pred_index[m.pred_id]
                    band = band_features(features_read, refined["baseline"][i],
                                         refined["neg"][i], refined["pos"][i])
                    if band is None:
                        continue
                    hyp = greedy_decode(model.reader(band), alphabet)
                    edits += levenshtein(ln["text"], hyp)
                    total_len += len(ln["text"])
            metrics["cer_after_readable_matched"] = (edits / total_len) if total_len else None

    result.update({"metrics": metrics, "seconds": time.monotonic() - t0})
    return result


def run(cfg, arm, output, *, train_pages=None, eval_pages=None, steps=None, root=None):
    started = time.monotonic()
    root = Path(root or Path(cfg["source"]["root"]).expanduser())
    target_h = cfg["scale"]["target_median_height_px"]
    train_ids = train_pages or cfg["split"]["train_pages"]
    eval_ids = eval_pages or cfg["split"]["eval_pages"]
    steps = steps or cfg["steps"]

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    torch.set_num_threads(cfg["threads"])

    train_pages_loaded = [load_scaled_page(root, pid, target_h) for pid in train_ids]
    eval_pages_loaded = [load_scaled_page(root, pid, target_h) for pid in eval_ids]
    all_train_lines = [ln for p in train_pages_loaded for ln in p["lines"]]
    alphabet = build_alphabet(train_pages_loaded)
    char_to_idx = {c: i for i, c in enumerate(alphabet) if i > 0}
    for p in train_pages_loaded:
        p["alphabet"] = alphabet
    for p in eval_pages_loaded:
        p["alphabet"] = alphabet

    model = StageSModel(arm, len(alphabet), channels=cfg["model"]["channels"], K=cfg["model"]["K"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["learning_rate"])
    limit = cfg["time_limit_seconds_per_run"]
    stop = {"hit": False}

    def time_up():
        stop["hit"] = (time.monotonic() - started) > limit
        return stop["hit"]

    t_train = time.monotonic()
    rows = train(model, optimizer, train_pages_loaded, char_to_idx, cfg, rng, steps, target_h,
                 log_stop=time_up)
    train_seconds = time.monotonic() - t_train

    evals = []
    if not stop["hit"]:
        for p in eval_pages_loaded:
            evals.append(evaluate_page(model, p, cfg, target_h))
            if time.monotonic() - started > limit:
                stop["hit"] = True
                break

    cer_values = [e["metrics"].get("cer_after_readable_matched") for e in evals
                  if e["metrics"].get("cer_after_readable_matched") is not None]
    secs = [r["seconds"] for r in rows]
    report = {
        "schema_version": "stage_s_smoke_report_v1",
        "status": "SMOKE_ONLY_NO_CLAIM" if not stop["hit"] else "STOPPED_TIME_LIMIT_SMOKE_ONLY",
        "arm": arm,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps_requested": steps,
        "steps_completed": len(rows),
        "time_limit_seconds": limit,
        "elapsed_seconds": time.monotonic() - started,
        "train_seconds": train_seconds,
        "seconds_per_step_mean": (sum(secs) / len(secs)) if secs else None,
        "scale_factors": {p["page_id"]: p["factor"] for p in train_pages_loaded + eval_pages_loaded},
        "alphabet_size": len(alphabet),
        "train_pages": train_ids,
        "eval_pages": eval_ids,
        "steps": rows,
        "eval": evals,
        "cer_after_mean": (sum(cer_values) / len(cer_values)) if cer_values else None,
        "seconds_per_eval_page_mean": (sum(e["seconds"] for e in evals) / len(evals)) if evals else None,
        "provenance": {
            "config_sha256": sha256(ROOT / "config/stage_s_smoke.json") if (ROOT / "config/stage_s_smoke.json").exists() else None,
            "script_sha256": sha256(Path(__file__).resolve()),
            "module_sha256": {m: sha256(ROOT / f"src/babaorum/experimental/{m}.py") for m in MODULES},
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform(),
                    "threads": cfg["threads"]},
        "limitations": [
            "smoke run: a few CPU steps, untrained reader, no OCR or performance claim",
            "eval pages are smoke-only; the sealed volet 2 is not opened",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage_s_smoke.json")
    parser.add_argument("--arm", required=True, choices=("S2", "B3", "B4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    report = run(cfg, args.arm, args.output)
    print(json.dumps({"status": report["status"], "steps_completed": report["steps_completed"],
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
