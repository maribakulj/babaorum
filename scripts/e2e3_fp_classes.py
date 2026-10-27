#!/usr/bin/env python3
"""E2E-3: why do B3 and B4 produce false detections? (docs/END_TO_END_E2E3_LOT.txt, amendement 1).

Steps (each run separately through ~/outils-seg/lourd.sh, never in reports/ before the final report):
  check     sha256 of the 51 tracked mille-feuilles files of the clone vs bundle v5 manifest
  replay    rebuild the pages (already done by gen), run B3 on the 10 eval pages, compare
            mean recall/precision with the stored values (tolerance 1e-3); mismatch -> stop
  classes   D/F/H/R classes of the unmatched predictions of B3, B4, S2 (after refinement)
  cosine    cos(grad L_det, grad L_ctc) on enc_det over 20 training batches (B3, B4; S2 control = 0)
  final     assemble reports/e2e3_fp_classes.json (only with --confirmed-go after the GO of w2:p6)

Pure geometry and statistics import without torch or kraken (fast tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from g_diag_mf import dist_to_polyline, gt_samples  # noqa: E402

CONFIG = ROOT / "config/e2e3_fp_classes.json"
TMP = Path("/private/tmp/e2e3")
CAP_MIN = 60.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- pure: classes D/F/H/R

def mean_pred_to_gt(pred_poly, gt_poly, step: float = 4.0) -> float:
    """Mean distance (px) from the sampled points of the prediction to the GT polyline: how far the
    prediction lies from the GT line. Used for 'near' (<= 0.5 h), for H (> 1 h) and for h."""
    ps = gt_samples(pred_poly, step)
    return sum(dist_to_polyline(p, gt_poly) for p in ps) / len(ps)


def coverage(pred_poly, gt_poly, tol: float) -> float:
    """Fraction of the GT arc length covered by the prediction: the GT baseline is the reference;
    each prediction segment whose two endpoints lie within tol of the GT contributes the interval of
    GT arc positions between their projections; the union of intervals divided by the GT length.
    Overlaps are counted once."""
    seg_len = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(gt_poly, gt_poly[1:])]
    total = sum(seg_len)
    if total == 0:
        return 0.0
    cum = [0.0]
    for L in seg_len:
        cum.append(cum[-1] + L)

    def project(pt):
        best = (math.inf, 0.0)  # (distance, arc position)
        for i, (a, b) in enumerate(zip(gt_poly, gt_poly[1:])):
            dx, dy = b[0] - a[0], b[1] - a[1]
            den = dx * dx + dy * dy
            t = 0.0 if den == 0 else max(0.0, min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / den))
            cx, cy = a[0] + t * dx, a[1] + t * dy
            d = math.hypot(pt[0] - cx, pt[1] - cy)
            if d < best[0]:
                best = (d, cum[i] + t * seg_len[i])
        return best

    intervals = []
    for a, b in zip(pred_poly, pred_poly[1:]):
        da, sa = project(a)
        db, sb = project(b)
        if da <= tol and db <= tol:
            lo, hi = sorted((sa, sb))
            intervals.append((lo, hi))
    intervals.sort()
    covered, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in intervals:
        if cur_lo is None:
            cur_lo, cur_hi = lo, hi
        elif lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            covered += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    if cur_lo is not None:
        covered += cur_hi - cur_lo
    return min(1.0, covered / total)


def reference_line(pred_poly, gts, step: float = 4.0):
    """The GT line whose normalised distance (prediction-to-GT mean / its own height) is minimal;
    ties go to the smallest GT id. Returns (gt_id, gt_baseline, gt_height, normalised_distance)."""
    best = None
    for gid, b, h in gts:
        nd = mean_pred_to_gt(pred_poly, b, step) / h
        key = (nd, gid)
        if best is None or key < best[0]:
            best = (key, (gid, b, h, nd))
    return best[1]


def classify_unmatched(pred_poly, gts, matched_gt_to_other: set) -> str:
    """Class of one UNMATCHED prediction, order D, F, H, R. Only the reference line is used:
    D: its normalised distance <= 0.5 and it is matched to another prediction;
    F: its normalised distance <= 0.5 and coverage(pred, ref) < 0.5;
    H: its normalised distance > 1; R: the rest (reported, never reread)."""
    gid, b, h, nd = reference_line(pred_poly, gts)
    if nd <= 0.5 and gid in matched_gt_to_other:
        return "D"
    if nd <= 0.5 and coverage(pred_poly, b, 0.5 * h) < 0.5:
        return "F"
    if nd > 1.0:
        return "H"
    return "R"


def count_classes(classes: list[str]) -> dict:
    counts = {"D": 0, "F": 0, "H": 0, "R": 0}
    for c in classes:
        counts[c] += 1
    assert sum(counts.values()) == len(classes), "partition broken"
    return counts


# ---------------------------------------------------------------- pure: readings

def ratio(n_extracted: int, n_gt: int) -> float:
    return n_extracted / n_gt if n_gt else float("nan")


def h1_reading(ratio_b3: float, ratio_s2: float) -> str:
    if ratio_b3 > 1.5 and ratio_s2 <= 1.2:
        return "supported"
    if ratio_b3 <= ratio_s2 * 1.2:
        return "contradicted"
    return "intermediate"


def r1_reading(counts_b3: dict) -> dict:
    """R1 on B3 false positives. R >= 50 % -> not explained here; R never chooses the next lot."""
    total = sum(counts_b3.values())
    if total == 0:
        return {"dominant": None, "text": "no false positive"}
    if counts_b3["R"] / total >= 0.5:
        return {"dominant": "R", "text": "non explique ici (R >= 50 %)"}
    for cls, action in (("D", "post-processing (NMS / fusion at extraction), no retraining"),
                        ("F", "extraction first (connectivity, min_cells), then the map"),
                        ("H", "cut or weight the CTC gradient into enc_det (H1 objective)")):
        if counts_b3[cls] / total >= 0.5:
            return {"dominant": cls, "text": action}
    return {"dominant": None, "text": "not explained here (no class reaches 50 %)"}


def cosine(u, v) -> float:
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return float("nan")
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def bootstrap_mean_ci(values, resamples: int, seed: int):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples) - 1]
    return sum(values) / n, lo, hi


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def cosine_reading(lo: float, hi: float, median_ratio: float, median_rho: float) -> str:
    """R4 (amendement 3): 'conflict supported' ONLY IF the 95 % CI of cos is entirely < 0 AND the
    median of rho >= 0.1. A median ratio < 0.1 means a negligible CTC gradient (checked first)."""
    if median_ratio < 0.1:
        return "CTC gradient negligible (median ratio < 0.1)"
    if hi < 0 and median_rho >= 0.1:
        return "conflict supported (CI entirely < 0 and median rho >= 0.1)"
    if lo > 0:
        return "gradients aligned (CI > 0): contradicts the conflict mechanism"
    if lo <= 0 <= hi:
        return "no measurable conflict (CI contains 0)"
    return "conflict not supported (CI < 0 but median rho < 0.1)"


def replay_status(recall: float, precision: float, ref_recall: float, ref_precision: float, tol: float) -> str:
    ok = abs(recall - ref_recall) <= tol and abs(precision - ref_precision) <= tol
    return "OK" if ok else "MISMATCH"


def page_cache_status(root: Path, n: int = 60) -> dict:
    """Run 2 accepts ONLY a complete cache: exactly n JSON pages and n PNG images, with sha256 per file.
    Anything else is refused (run 2 never generates pages)."""
    pages = sorted((root / "pages").glob("*.json")) if (root / "pages").is_dir() else []
    images = sorted((root / "images").glob("*.png")) if (root / "images").is_dir() else []
    if len(pages) != n or len(images) != n:
        return {"status": "FAILED", "reason": f"cache incomplete: {len(pages)} JSON, {len(images)} PNG (need {n} each)"}
    hashes = {str(f.relative_to(root)): sha256(f) for f in pages + images}
    return {"status": "ACCEPTED", "files": len(hashes), "sha256": hashes}


def live_processes(ps_text: str, own_pid: int) -> list:
    """Lines of a 'pid ppid command' listing that belong to an e2e3 run or to mille-feuilles
    generation. The current process and ALL its ancestors (up to pid 1) are excluded: their command
    lines (lourd.sh, time, the shell) contain the script name but they are not runs."""
    parent, lines = {}, {}
    for ln in ps_text.splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, ppid, cmd = int(parts[0]), int(parts[1]), parts[2]
        parent[pid] = ppid
        lines[pid] = cmd
    chain, cur = set(), own_pid
    while cur in parent and cur not in chain and cur > 1:
        chain.add(cur)
        cur = parent[cur]
    chain.add(own_pid)
    keys = ("scripts/e2e3_fp_classes.py", "mille_feuilles.cli")
    return [f"{pid} {cmd}" for pid, cmd in lines.items()
            if pid not in chain and any(k in cmd for k in keys) and "grep" not in cmd]


def guard_exceeded(elapsed_s: float, cap_min: float = CAP_MIN) -> bool:
    return elapsed_s > cap_min * 60.0


# ---------------------------------------------------------------- steps (torch env, CPU)

def step_check(clone: Path, bundle_manifest: dict) -> dict:
    """sha256 of the tracked mille-feuilles files of the clone against bundle v5 manifest (51 files)."""
    expected = bundle_manifest["mille_feuilles_sha256"]
    mismatch = []
    for rel in expected:
        if sha256(clone / rel[len("heritage-synth/"):]) != expected[rel]:
            mismatch.append(rel)
    return {"files": len(expected), "mismatch": mismatch, "status": "OK" if not mismatch else "MISMATCH"}


def load_arm(arm: str, ckpt_path: Path, alphabet_size: int):
    import torch
    from babaorum.experimental.stage_s_arms import StageSModel
    model = StageSModel(arm, alphabet_size, channels=64, K=8, hidden=128)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def alphabet_from_json(pages_json) -> list:
    """Same rule as stage_s_data.build_alphabet (readable lines, blank at index 0), from the JSON."""
    chars = set()
    for pg in pages_json:
        for ln in pg["lines"]:
            if ln.get("legibility") == "readable":
                chars.update(ln["text"])
    return ["<blank>", *sorted(chars)]


def predict_page(model, scaled_image, cfg_model: dict, extraction: dict):
    """Extracted polylines and refined (after) baselines, as in train_stage_s_v2.evaluate_full_page."""
    import torch
    from babaorum.experimental.stage_s_model import refine_lines
    from babaorum.experimental.stage_s_targets import extract_polylines
    image = torch.as_tensor(scaled_image)[None, None]
    with torch.no_grad():
        features = model.enc_det(image)
        logit = model.head_d(features)[0]
        prob = torch.sigmoid(logit)[0, 0].numpy()
        extracted = [p for p in extract_polylines(prob, stride=extraction["stride"],
                                                  threshold=extraction["threshold"],
                                                  min_cells=extraction["min_cells"]) if len(p) >= 2]
        refined = []
        if extracted:
            refined = refine_lines(features, extracted, model.head_g, head_d=model.head_d,
                                   K=cfg_model["K"])["baseline"].tolist()
    return extracted, refined


def gt_list(page_json) -> list:
    from babaorum.line_geometry import GroundTruthLine  # noqa: F401
    out = []
    for ln in page_json["lines"]:
        h = _band_height(ln)
        out.append((ln["id"], ln["baseline"], h))
    return out


def _band_height(ln) -> float:
    from babaorum.experimental.stage_s_data import _band
    neg, pos = _band(ln["baseline"], ln["polygon"])
    return neg + pos


def step_replay(cfg: dict, scaled_eval: dict, model, alphabet_size: int) -> dict:
    from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry
    rec, pre = [], []
    for pid, page in scaled_eval.items():
        extracted, refined = predict_page(model, page["image"], cfg["model"], cfg["extraction"])
        gts = [GroundTruthLine(id=i, baseline=b, height_px=h, baseline_source="ground_truth")
               for i, b, h in page["gt"] if h > 0]
        preds = [PredictedLine(id=f"p{k:05d}", baseline=b, baseline_source="predicted")
                 for k, b in enumerate(refined)]
        res = evaluate_line_geometry(gts, preds) if preds else None
        rec.append(res.recall if res else 0.0)
        pre.append(res.precision if (res and res.precision is not None) else 0.0)
    mr, mp = sum(rec) / len(rec), sum(pre) / len(pre)
    ref = cfg["replay_control"]
    status = replay_status(mr, mp, ref["B3_recall_mean_10_pages"], ref["B3_precision_mean_10_pages"],
                           ref["tolerance"])
    return {"recall_mean": mr, "precision_mean": mp, "reference": ref, "status": status,
            "alphabet_size": alphabet_size}


def step_classes(cfg: dict, scaled_eval: dict, models: dict) -> dict:
    from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry
    out = {}
    for arm, model in models.items():
        classes_all, n_ext, n_gt, per_page = [], 0, 0, []
        for pid, page in scaled_eval.items():
            extracted, refined = predict_page(model, page["image"], cfg["model"], cfg["extraction"])
            gts = page["gt"]
            gt_objs = [GroundTruthLine(id=i, baseline=b, height_px=h, baseline_source="ground_truth")
                       for i, b, h in gts if h > 0]
            preds = [PredictedLine(id=f"p{k:05d}", baseline=b, baseline_source="predicted")
                     for k, b in enumerate(refined)]
            res = evaluate_line_geometry(gt_objs, preds) if preds else None
            matched_pred = {m.pred_id for m in res.matches} if res else set()
            matched_gt = {m.gt_id for m in res.matches} if res else set()
            gts_for_class = [(i, b, h) for i, b, h in gts if h > 0]
            page_classes = []
            for k, b in enumerate(refined):
                pid_k = f"p{k:05d}"
                if pid_k in matched_pred:
                    continue
                page_classes.append(classify_unmatched(b, gts_for_class, matched_gt))
            classes_all.extend(page_classes)
            n_ext += len(extracted)
            n_gt += len(gts)
            per_page.append({"page_id": pid, "n_gt": len(gts), "n_extracted": len(extracted),
                             "counts": count_classes(page_classes)})
        out[arm] = {"counts": count_classes(classes_all), "ratio": ratio(n_ext, n_gt),
                    "n_extracted": n_ext, "n_gt": n_gt, "per_page": per_page}
    out["H1"] = h1_reading(out["B3"]["ratio"], out["S2"]["ratio"]) if {"B3", "S2"} <= set(out) else None
    out["R1"] = r1_reading(out["B3"]["counts"]) if "B3" in out else None
    return out


def pick_index(rng, n: int) -> int:
    """Page index drawn with the NumPy Generator used by train_stage_s_v2 (rng.integers)."""
    return int(rng.integers(n))


def pick_lines(rng, n: int, take: int) -> list:
    """Line indices drawn as in train_stage_s_v2 (rng.permutation)."""
    return [int(i) for i in rng.permutation(n)[:take]] if take else []


def make_batch(page, rng, cfg: dict, sigma_px: float):
    """One training batch as in train_stage_s_v2.train_v2: window, detection raster, perturbed CTC lines."""
    from train_stage_s_v2 import draw_window, clip_polyline_to_box, intersects_tile
    from train_stage_s import lines_in_tile, perturb_baseline, encode_text
    from babaorum.experimental.stage_s_targets import rasterize_baselines
    tile = 768
    x0, y0 = draw_window(page, rng, tile, True)
    image = page["image"][y0:y0 + tile, x0:x0 + tile]
    detect = []
    for ln in page["lines"]:
        if not intersects_tile(ln, x0, y0, tile):
            continue
        for piece in clip_polyline_to_box(ln["baseline"], x0, y0, tile):
            detect.append({"baseline": piece, "normal_negative_px": ln["normal_negative_px"],
                           "normal_positive_px": ln["normal_positive_px"]})
    raster = rasterize_baselines(detect, (tile, tile), stride=cfg["model"]["stride"])
    inside = lines_in_tile(page["lines"], x0, y0, tile)
    readable = [ln for ln in inside if ln["legibility"] == "readable" and ln["text"]]
    take = min(8, len(readable))
    chosen = [readable[i] for i in pick_lines(rng, len(readable), take)]
    polylines = [perturb_baseline(ln["baseline"], ln["normal_negative_px"], ln["normal_positive_px"], rng)
                 for ln in chosen]
    labels = [encode_text(ln["text"], page["char_to_idx"]) for ln in chosen]
    return image, raster, polylines, labels


def step_cosine(cfg: dict, pages_train: dict, models: dict) -> dict:
    """Per batch: two SEPARATE backward passes (L_det, then L_ctc) with zero_grad between them,
    on the flattened enc_det gradients. Records every batch. sigma = sigma_for_step(1999, 2000, h)."""
    import torch
    from train_stage_s import sigma_for_step
    from train_stage_s_v2 import detection_loss_dev
    cc = cfg["cosine_measurement"]
    h = cfg["scale"]["target_median_height_px"]
    sigma = sigma_for_step(1999, 2000, h)
    out = {}
    for arm, model in models.items():
        model.train()
        enc = list(model.enc_det.parameters())
        rng = np.random.default_rng(cc["seed"])
        keys = sorted(pages_train)
        batches, attempts, skipped = [], 0, 0
        while len(batches) < cc["batches"]:
            if attempts >= cc["attempt_cap"]:
                break
            attempts += 1
            page = pages_train[keys[pick_index(rng, len(keys))]]
            image, raster, polylines, labels = make_batch(page, rng, cfg, sigma_px=sigma)
            img_t = torch.as_tensor(image, dtype=torch.float32)[None, None]
            res = model.forward_lines(img_t, polylines, labels, sigma_px=sigma)
            incl = [x for x in res["ctc_losses"] if x is not None]
            if not incl:
                skipped += 1
                continue
            l_det = detection_loss_dev(res["det"], raster, torch.device("cpu"))
            l_ctc = sum(incl) / len(incl)
            model.zero_grad(set_to_none=True)
            l_det.backward(retain_graph=True)
            g_det = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in enc]).clone()
            model.zero_grad(set_to_none=True)
            l_ctc.backward()
            g_ctc = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in enc]).clone()
            model.zero_grad(set_to_none=True)
            nd, nc = float(g_det.norm()), float(g_ctc.norm())
            cos = float(torch.dot(g_det, g_ctc) / (g_det.norm() * g_ctc.norm() + 1e-30))
            ratio_b = nc / nd if nd > 0 else float("nan")
            batches.append({"page_id": page.get("page_id"), "cos": cos, "norm_det": nd, "norm_ctc": nc,
                            "ratio": ratio_b, "rho": -cos * ratio_b})
        n_ok = len(batches)
        if n_ok < cc["batches"]:
            out[arm] = {"status": "COSINE_INCOMPLETE", "batches_valid": n_ok, "attempts": attempts,
                        "batches_skipped_no_ctc": skipped, "per_batch": batches, "reading": None}
            continue
        cos_vals = [b["cos"] for b in batches]
        mean, lo, hi = bootstrap_mean_ci(cos_vals, cc["bootstrap"]["resamples"], cc["bootstrap"]["seed"])
        med_ratio = median([b["ratio"] for b in batches])
        med_rho = median([b["rho"] for b in batches])
        ctc_max = max(b["norm_ctc"] for b in batches)
        reading = cosine_reading(lo, hi, med_ratio, med_rho) if arm == "B3" else "descriptive (B4)"
        if arm == "S2":
            reading = "control: rho = 0 and norm_ctc = 0 required"
        out[arm] = {"status": "OK", "batches_valid": n_ok, "attempts": attempts,
                    "batches_skipped_no_ctc": skipped, "per_batch": batches,
                    "cos_mean": mean, "cos_ci95": [lo, hi], "median_ratio": med_ratio,
                    "median_rho": med_rho, "norm_ctc_max": ctc_max, "sigma_px": sigma, "reading": reading}
    return out


# ---------------------------------------------------------------- orchestration

BUNDLE_ZIP = Path("/Users/marcel/Library/CloudStorage/GoogleDrive-mrebg@hotmail.fr/Mon Drive/babaorum/stage_s/bundle/stage_s_bundle_v5.zip")
CALIBRATION = ROOT / "runs/e2e1_kaggle_output/calibration"
GEN_COMMAND = ["--output", None, "--pages", "60", "--seed", "20261008", "--width", "2680", "--height", "3698",
               "--dpi", "150", "--degradation", "mixed", "--jobs", "2"]
HERITAGE_PY = Path.home() / "heritage-synth/.venv/bin/python"


class _LazyPages:
    """Scaled training pages loaded one at a time (memory); keys are the training page ids."""

    def __init__(self, root: Path, ids, char_to_idx: dict, target_h: float):
        self.root, self.ids, self.char_to_idx, self.target_h = root, list(ids), char_to_idx, target_h

    def __iter__(self):
        return iter(self.ids)

    def __getitem__(self, pid):
        from train_stage_s import load_scaled_page
        page = load_scaled_page(self.root, pid, self.target_h)
        page["char_to_idx"] = self.char_to_idx
        return page


def _check_prefix(path: Path, prefix: str) -> bool:
    return sha256(path).startswith(prefix)


def run_all(args, report=None) -> dict:
    import zipfile
    import subprocess
    from train_stage_s import load_scaled_page  # noqa: F401  (used through _LazyPages)

    cfg = json.loads(CONFIG.read_text())
    started = time.monotonic()
    TMP.mkdir(parents=True, exist_ok=True)
    if report is None:
        report = {}
    report.update({"schema_version": "e2e3_fp_classes_report_v1", "status": None, "steps": report.get("steps", {}),
                   "provenance": {"script_sha256": sha256(Path(__file__).resolve()),
                                  "config_sha256": sha256(CONFIG),
                                  "tests_sha256": sha256(ROOT / "tests/test_e2e3_fp_classes.py")}})

    def stop(status, why):
        report["status"] = status
        report["stopped_because"] = why
        return report

    def over_cap():
        return guard_exceeded(time.monotonic() - started, cfg["budget"]["cap_minutes"])

    # 1. sources: the 51 tracked files of the clone against the bundle v5 manifest
    manifest = json.loads(zipfile.ZipFile(BUNDLE_ZIP).read("bundle_manifest.json"))
    report["bundle_sha256"] = sha256(BUNDLE_ZIP)
    check = step_check(args.clone, manifest)
    report["steps"]["check"] = check
    if check["status"] != "OK":
        return stop("MISMATCH", "source files differ from the bundle v5 manifest")

    # 2. page cache from run 1 (no generation in run 2); the 60-minute cap starts after this step
    root = TMP / "mf"
    cache = page_cache_status(root)
    report["steps"]["page_cache"] = cache
    if cache["status"] != "ACCEPTED":
        return stop("FAILED", cache["reason"])
    n_pages = len(list((root / "pages").glob("*.json")))
    n_img = len(list((root / "images").glob("*.png")))
    started = time.monotonic()  # the cap counts inference only

    # 3. alphabet exactly as run_v2: readable lines of the train pages; blank at index 0
    train_json = {pid: json.loads((root / "pages" / f"{pid}.json").read_text()) for pid in cfg["pages"]["train"]}
    from babaorum.experimental.stage_s_data import BLANK  # noqa: F401
    alphabet = alphabet_from_json(list(train_json.values()))
    char_to_idx = {c: i for i, c in enumerate(alphabet) if i > 0}
    report["alphabet_size"] = len(alphabet)

    # 4. checkpoints (CPU) with their pre-registered sha256 prefixes
    ckpts = {}
    for arm in ("B3", "B4", "S2"):
        info = cfg["source_checkpoints"][arm]
        path = CALIBRATION / info["checkpoint"]
        if not _check_prefix(path, info["sha256"]):
            return stop("MISMATCH", f"checkpoint {arm} sha256 does not match the pre-registered prefix")
        ckpts[arm] = path

    # 5. evaluation pages (scaled as in run_v2)
    eval_scaled = {}
    for pid in cfg["pages"]["eval"]:
        pg = load_scaled_page(root, pid, cfg["scale"]["target_median_height_px"])
        eval_scaled[pid] = {"image": pg["image"],
                            "gt": [(ln["id"], ln["baseline"], ln["normal_negative_px"] + ln["normal_positive_px"])
                                   for ln in pg["lines"]]}
    lazy_train = _LazyPages(root, cfg["pages"]["train"], char_to_idx, cfg["scale"]["target_median_height_px"])

    # 6. priority order: replay B3 -> classes B3 -> cosine B3 -> classes B4 -> cosine B4 -> classes S2 -> cosine S2
    model_b3 = load_arm("B3", ckpts["B3"], len(alphabet))
    rep = step_replay(cfg, eval_scaled, model_b3, len(alphabet))
    report["steps"]["replay_B3"] = rep
    if rep["status"] != "OK":
        return stop("MISMATCH", "replay control B3 failed: no reading")

    def guarded(name, fn):
        if over_cap():
            report["steps"][name] = {"status": "INCOMPLETE_CAP", "reading": None}
            return False
        report["steps"][name] = fn()
        return True

    guarded("classes_B3", lambda: step_classes(cfg, eval_scaled, {"B3": model_b3})["B3"])
    guarded("cosine_B3", lambda: step_cosine(cfg, lazy_train, {"B3": model_b3})["B3"])
    for arm in ("B4", "S2"):
        model = load_arm(arm, ckpts[arm], len(alphabet))
        guarded(f"classes_{arm}", lambda m=model, a=arm: step_classes(cfg, eval_scaled, {a: m})[a])
        if arm == "S2":  # keep the whole dict: it carries the S2 control (exact zero)
            guarded("cosine_S2", lambda m=model: step_cosine(cfg, lazy_train, {"S2": m}))
        else:
            guarded(f"cosine_{arm}", lambda m=model, a=arm: step_cosine(cfg, lazy_train, {a: m})[a])
        del model

    # 7. readings (ratios, R1, R4). H1 keeps its ratio status and waits for R1 and R4.
    b3c = report["steps"].get("classes_B3", {}).get("counts")
    report["R1"] = r1_reading(b3c) if b3c else None
    cos_b3 = report["steps"].get("cosine_B3", {})
    report["R4_B3"] = cos_b3.get("reading") if isinstance(cos_b3, dict) else None
    s2 = report["steps"].get("classes_S2", {})
    if "ratio" in report["steps"].get("classes_B3", {}) and "ratio" in s2:
        report["H1_ratio_status"] = h1_reading(report["steps"]["classes_B3"]["ratio"], s2["ratio"])
    report["H1"] = "not decided: the ratio status is recorded; the decision waits for R1 and R4 (section 3)"
    report["status"] = "COMPLETE" if not over_cap() else "INCOMPLETE"
    report["elapsed_minutes"] = (time.monotonic() - started) / 60.0
    return report


RUN3_REPORT = ROOT / "reports" / "e2e3_fp_classes_run3.json"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "run"])
    parser.add_argument("--clone", type=Path, default=None, help="mille-feuilles clone at 23875ab0")
    parser.add_argument("--confirmed-go", action="store_true")
    parser.add_argument("--tmp", type=Path, default=TMP, help="folder holding the page cache mf/ (default: Mac)")
    parser.add_argument("--bundle-zip", type=Path, default=BUNDLE_ZIP, help="bundle v5 zip (default: Mac Drive)")
    parser.add_argument("--heritage-py", type=Path, default=HERITAGE_PY, help="python of the mille-feuilles venv")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION, help="folder with B3/, B4/, S2/ checkpoints")
    parser.add_argument("--report", type=Path, default=RUN3_REPORT, help="output report (never an existing file)")
    return parser


def main(argv=None):
    global TMP, BUNDLE_ZIP, HERITAGE_PY, CALIBRATION
    args = build_parser().parse_args(argv)
    TMP, BUNDLE_ZIP, HERITAGE_PY, CALIBRATION = args.tmp, args.bundle_zip, args.heritage_py, args.calibration
    if args.step == "run" and not args.confirmed_go:
        raise SystemExit("refused: E2E-3 runs only with --confirmed-go after the GO of w2:p6")
    if args.step == "run":
        import os
        ps = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
        alive = live_processes(ps, os.getpid())
        if alive:
            raise SystemExit("refused: an e2e3 or mille-feuilles process is alive: " + alive[0][:120])
    if args.clone is None:
        raise SystemExit("--clone is required")
    if args.step == "check":
        import zipfile
        manifest = json.loads(zipfile.ZipFile(BUNDLE_ZIP).read("bundle_manifest.json"))
        print(json.dumps(step_check(args.clone, manifest)))
        return
    partial = {}
    try:
        report = run_all(args, partial)
    except Exception as exc:  # an exception is reported, and the steps already done are kept
        import traceback
        report = dict(partial)
        report.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:]})
    if report["status"] == "COMPLETE" or report.get("status") in ("MISMATCH", "INCOMPLETE", "FAILED", "REFUSED"):
        out = args.report  # run 3 by default; runs 1 and 2 reports are never touched
        if out.exists():
            raise SystemExit(f"refused: {out.name} already exists; a report is never overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, allow_nan=False, default=float) + "\n")
    print(json.dumps({"status": report["status"]}))


if __name__ == "__main__":
    main()
