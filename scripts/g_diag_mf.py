#!/usr/bin/env python3
"""G-diag: why does the whole-page BLLA pass miss so many synthetic Mille Feuilles lines?

Arms (global passes only, no block ground truth used in any prediction):
  D0  whole page at native resolution (the G-intra global pass), predictions saved;
  D1  two overlapping horizontal bands fed at native resolution; BLLA resizes each band
      internally, so lines are enlarged relative to D0 AND the vertical field of each call is
      reduced; a prediction is kept by the band whose core contains its centroid; mapped back to
      page edge coordinates; predictions saved. D1 reads as "the horizontal band helps", with the
      scale effect and the reduced vertical field mixed.
Each missed GT line (same matching as line_geometry.v1, all readable lines) gets exactly one class,
in the order a, b, c, d; their sum equals the number of misses (checked).
The column gutter g_page is measured per page on GT blocks before any prediction.

Run only with --confirmed-go, after the explicit GO of w2:p6. Pure geometry is importable
without kraken (fast tests). Synthetic only; no OCR claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

CONFIG = ROOT / "config/g_diag_mf_v1.json"
SAMPLE_STEP_PX = 4.0
TIME_LIMIT_S = 45 * 60


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- geometry (pure)

def _point_segment_distance(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def dist_to_polyline(p, poly) -> float:
    if len(poly) == 1:
        return math.hypot(p[0] - poly[0][0], p[1] - poly[0][1])
    return min(_point_segment_distance(p, a, b) for a, b in zip(poly, poly[1:]))


def gt_samples(baseline, step: float = SAMPLE_STEP_PX):
    """Points sampled every `step` px along the GT polyline (both ends included)."""
    pts = []
    for (ax, ay), (bx, by) in zip(baseline, baseline[1:]):
        seg = math.hypot(bx - ax, by - ay)
        n = max(1, int(math.ceil(seg / step)))
        for k in range(n):
            t = k / n
            pts.append((ax + t * (bx - ax), ay + t * (by - ay)))
    pts.append((baseline[-1][0], baseline[-1][1]))
    return pts


def coverage(gt_pts, pred_poly, tol: float) -> float:
    """Fraction of GT samples within tol of the prediction polyline (GT -> prediction)."""
    if not gt_pts:
        return 0.0
    return sum(1 for p in gt_pts if dist_to_polyline(p, pred_poly) <= tol) / len(gt_pts)


def near(gt_pts, pred_poly, tol: float) -> bool:
    return any(dist_to_polyline(p, pred_poly) <= tol for p in gt_pts)


def measure_gutter(block_bboxes):
    """Median horizontal gap between each block and its nearest horizontal neighbour among the
    blocks it vertically overlaps (gap > 0 only). None when no such pair exists. Measured on GT
    blocks before any prediction, per page."""
    gaps = []
    for i, (ax0, ay0, ax1, ay1) in enumerate(block_bboxes):
        best = None
        for j, (bx0, by0, bx1, by1) in enumerate(block_bboxes):
            if i == j or not (ay0 < by1 and by0 < ay1):
                continue
            gap = max(bx0 - ax1, ax0 - bx1)
            if gap > 0 and (best is None or gap < best):
                best = gap
        if best is not None:
            gaps.append(best)
    return statistics.median(gaps) if gaps else None


def classify_miss(gt_baseline, h: float, block_x, preds, g_page) -> str:
    """Class of a missed GT line, exactly one: a absent, b column fusion, c partial, d other.
    Order a, then b, then c, then d. b uses the candidate prediction (largest GT coverage) and the
    measured page gutter g_page; if g_page is None, b cannot be assigned."""
    tol = 0.5 * h
    pts = gt_samples(gt_baseline)
    if not any(near(pts, p, tol) for p in preds):
        return "a_absent"
    if g_page is not None and preds:
        cand = max(preds, key=lambda p: coverage(pts, p, tol))
        bx0, bx1 = block_x
        xs = [q[0] for q in cand]
        beyond = min(xs) < bx0 - g_page or max(xs) > bx1 + g_page
        if coverage(pts, cand, tol) >= 0.75 and beyond:
            return "b_fusion"
    max_cov = max(coverage(pts, p, tol) for p in preds)
    if max_cov < 0.75:
        return "c_partial"
    return "d_other"


def centroid_y(baseline) -> float:
    return sum(y for _, y in baseline) / len(baseline)


def keep_band_predictions(band_preds, y0: float, core0: float, core1: float):
    """Map band predictions to page coordinates and keep those whose centroid lies in the core."""
    kept = []
    for poly in band_preds:
        page = [[x, y + y0] for x, y in poly]
        if core0 <= centroid_y(page) < core1:
            kept.append(page)
    return kept


def band_specs(height: int, band_rows_fraction, core_rows_fraction):
    """(y0, y1, core0, core1) in page rows for each band."""
    return [(round(b0 * height), round(b1 * height), round(c0 * height), round(c1 * height))
            for (b0, b1), (c0, c1) in zip(band_rows_fraction, core_rows_fraction)]


def decision(rate_d0: float, rate_d1: float, class_b_share: float) -> dict:
    """Both rules are reported; they may hold together. Descriptive, no test."""
    rule_scale = rate_d1 <= rate_d0 / 2.0
    rule_columns = class_b_share >= 0.5
    if rule_scale and rule_columns:
        text = "both rules hold: D1 misses at most half of D0, and class b is at least half of D0 misses"
    elif rule_scale:
        text = "rule scale holds (rate D1 <= rate D0 / 2); rule columns does not"
    elif rule_columns:
        text = "rule columns holds (class b >= 50 % of D0 misses); rule scale does not"
    else:
        text = "neither rule holds: not explained here; the real-geometry volume (volet 3) decides"
    return {"rule_scale_D1_le_half_D0": rule_scale, "rule_columns_b_ge_half": rule_columns, "text": text,
            "reading_D1": "the horizontal band helps: scale and reduced vertical field mixed"}


def control_status(d0_matched_ids, reference_rows) -> dict:
    """D0 must reproduce global_hit of the G-intra report for the lines already sampled there."""
    mismatches = [r["id"] for r in reference_rows if (r["id"] in d0_matched_ids) != bool(r["global_hit"])]
    return {"n_checked": len(reference_rows), "mismatches": mismatches,
            "status": "OK" if not mismatches else "MISMATCH"}


def guard_exceeded(elapsed_s: float, limit_s: float = TIME_LIMIT_S) -> bool:
    return elapsed_s > limit_s


# ---------------------------------------------------------------- run (kraken env)

def run(cfg, out_path: Path):
    from PIL import Image
    import t5_mf  # same loader and BLLA call as the G-intra run

    started = time.monotonic()
    root = Path(cfg["source"]["root"]).expanduser()
    model_path = Path(cfg["model"]["path"]).expanduser()
    if sha256(model_path) != cfg["model"]["sha256"]:
        raise RuntimeError("BLLA model sha256 does not match the pre-registered value")
    ref = json.loads((ROOT / cfg["control"]["reference"]).read_text())
    ref_rows = {p["page_id"]: p["rows"] for p in ref["per_page"]}
    model = t5_mf.load_blla(model_path)

    per_page, guard_hit = [], False
    for pid in cfg["pages"]["ids"]:
        if guard_exceeded(time.monotonic() - started):
            guard_hit = True
            break
        t0 = time.monotonic()
        page = json.loads((root / "pages" / f"{pid}.json").read_text())
        H = page["image"]["height"]
        blocks = {b["id"]: b for b in page["blocks"]}
        bboxes = [(min(q[0] for q in b["polygon"]), min(q[1] for q in b["polygon"]),
                   max(q[0] for q in b["polygon"]), max(q[1] for q in b["polygon"])) for b in blocks.values()]
        g_page = measure_gutter(bboxes)  # before any prediction

        gt_all = [GroundTruthLine(id=ln["id"], baseline=ln["baseline"], height_px=t5_mf._height(ln),
                                  baseline_source="ground_truth") for ln in page["lines"] if t5_mf._height(ln) > 0]
        readable = [g for g, ln in zip(gt_all, [l for l in page["lines"] if t5_mf._height(l) > 0])
                    if ln.get("legibility") == "readable"]
        img = Image.open(root / "images" / f"{pid}.png").convert("L")

        d0 = t5_mf.blla_lines(img, model)
        d1, band_info = [], []
        for (y0, y1, c0, c1) in band_specs(H, cfg["arms"]["D1"]["band_rows_fraction"],
                                           cfg["arms"]["D1"]["core_rows_fraction"]):
            raw = t5_mf.blla_lines(img.crop((0, y0, img.width, y1)), model)
            kept = keep_band_predictions(raw, y0, c0, c1)
            band_info.append({"rows": [y0, y1], "core": [c0, c1], "raw": len(raw), "kept": len(kept)})
            d1.extend(kept)

        # control: D0 on the G-intra GT set (all lines with height > 0) must reproduce global_hit
        d0_all = {m.gt_id for m in evaluate_line_geometry(
            gt_all, [PredictedLine(id=f"g{i:05d}", baseline=p, baseline_source="predicted")
                     for i, p in enumerate(d0)]).matches} if d0 else set()
        control = control_status(d0_all, ref_rows.get(pid, []))

        def analyse(preds):
            pobjs = [PredictedLine(id=f"p{i:05d}", baseline=p, baseline_source="predicted")
                     for i, p in enumerate(preds)]
            gt_r = [g for g in gt_all if g.id in {r.id for r in readable}]
            matched = {m.gt_id for m in evaluate_line_geometry(gt_r, pobjs).matches} if pobjs else set()
            classes = {"a_absent": 0, "b_fusion": 0, "c_partial": 0, "d_other": 0}
            miss_ids = []
            for g in gt_r:
                if g.id in matched:
                    continue
                miss_ids.append(g.id)
                blk = blocks[next(l["block_id"] for l in page["lines"] if l["id"] == g.id)]
                bx = (min(q[0] for q in blk["polygon"]), max(q[0] for q in blk["polygon"]))
                classes[classify_miss(g.baseline, g.height_px, bx, preds, g_page)] += 1
            n = len(gt_r)
            partition_ok = sum(classes.values()) == len(miss_ids)
            return {"n_gt_readable": n, "n_pred": len(preds), "n_matched": len(matched),
                    "n_miss": len(miss_ids), "miss_rate": (len(miss_ids) / n) if n else None,
                    "classes": classes, "partition_ok": partition_ok, "miss_ids": miss_ids}

        per_page.append({"page_id": pid, "page_sha256": sha256(root / "pages" / f"{pid}.json"),
                         "image_sha256": sha256(root / "images" / f"{pid}.png"),
                         "height": H, "g_page_px": g_page, "control": control,
                         "D0": analyse(d0), "D1": analyse(d1), "band_info": band_info,
                         "seconds_page": time.monotonic() - t0,
                         "D0_predictions": d0, "D1_predictions": d1})

    controls_ok = all(p["control"]["status"] == "OK" for p in per_page)
    complete = (not guard_hit) and len(per_page) == len(cfg["pages"]["ids"])
    if not controls_ok:
        status = "MISMATCH"
    elif not complete:
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"

    report = {"schema_version": "g_diag_mf_report_v2", "status": status,
              "pages": [p["page_id"] for p in per_page], "guard_hit": guard_hit,
              "elapsed_seconds": time.monotonic() - started, "per_page": per_page,
              "provenance": {"config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__).resolve()),
                             "model_sha256": cfg["model"]["sha256"], "reference": cfg["control"]["reference"]}}
    if status == "COMPLETE":
        def pooled(arm):
            n = sum(p[arm]["n_gt_readable"] for p in per_page)
            m = sum(p[arm]["n_miss"] for p in per_page)
            cls = {k: sum(p[arm]["classes"][k] for p in per_page) for k in per_page[0][arm]["classes"]}
            return {"n_gt_readable": n, "n_miss": m, "miss_rate": (m / n) if n else None, "classes": cls,
                    "partition_ok": all(p[arm]["partition_ok"] for p in per_page)}
        d0_tot, d1_tot = pooled("D0"), pooled("D1")
        b_share = d0_tot["classes"]["b_fusion"] / (d0_tot["n_miss"] or 1)
        report.update({"D0": d0_tot, "D1": d1_tot, "class_b_share_of_D0_misses": b_share,
                       "decision": decision(d0_tot["miss_rate"], d1_tot["miss_rate"], b_share)})
    else:
        report["reading"] = f"NOT READ: status {status}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, allow_nan=False, default=float) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/g_diag_mf_v1.json")
    parser.add_argument("--confirmed-go", action="store_true", help="only after the explicit GO of w2:p6")
    args = parser.parse_args(argv)
    if not args.confirmed_go:
        raise SystemExit("refused: G-diag runs only with --confirmed-go after the GO of w2:p6")
    cfg = json.loads(args.config.read_text())
    report = run(cfg, args.output)
    print(json.dumps({"status": report["status"], "reading": report.get("decision", {}).get("text")}))


if __name__ == "__main__":
    main()
