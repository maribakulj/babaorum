#!/usr/bin/env python3
"""G-intra on the synthetic Mille Feuilles volume: observed union gain G inside GT blocks.

Modes:
  --pages 1            one-page cost probe: block crops may be capped and the probe is never a
                       result (no statistic is written); output reports/g_intra_mf_cost_probe.json.
  full (no --pages)    the registered 30 pages: caps refused; any interruption or the time guard
                       gives status INCOMPLETE and no statistic is read; needs run_gate full_run
                       OPEN; output reports/g_intra_mf_v1.json.
Per-page results go to a partial file next to the report. A resume is allowed only when the
config, script and model sha256 are identical to those recorded in the partial file.

Pure statistics and geometry import without kraken (fast tests). The BLLA calls need the kraken
environment: ~/outils-seg/kraken-env/bin/python. Synthetic only; no OCR claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.line_geometry import (  # noqa: E402
    GroundTruthLine, PredictedLine, evaluate_line_geometry, evaluate_line_geometry_sensitivity,
)

CONFIG = ROOT / "config/g_intra_mf_v2.json"
BLLA_HEIGHT = 1800
MARGIN = 24
CHRONO_OUTPUT = "g_intra_mf_cost_probe.json"
FULL_OUTPUT = "g_intra_mf_v2.json"
GUARD_SECONDS_DEFAULT = 5 * 3600
FUTILITY_UPPER = 0.02
MODULE_FILES = ["src/babaorum/__init__.py", "src/babaorum/line_geometry.py",
                "src/babaorum/experimental/__init__.py", "src/babaorum/experimental/stage_s_data.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- pure statistics

def stats_from_indicators(miss_g, miss_l) -> dict:
    """r, m, phi, union oracle recall and observed gain from paired miss indicators (0/1)."""
    g = np.asarray(miss_g, dtype=float)
    l = np.asarray(miss_l, dtype=float)
    r, m = g.mean(), l.mean()
    p_both = float((g * l).mean())
    denom = math.sqrt(r * (1 - r) * m * (1 - m))
    phi = (p_both - r * m) / denom if denom > 0 else float("nan")
    union_recall = 1.0 - p_both
    best_single = 1.0 - min(r, m)
    return {
        "n": int(len(g)), "r_miss_global": float(r), "m_miss_local": float(m),
        "p_both_miss": p_both, "phi": float(phi),
        "union_oracle_recall": float(union_recall),
        "best_single_recall": float(best_single),
        "G_observed": float(union_recall - best_single),
        "G_reconstructed": float(min(r, m) - r * m - phi * denom) if denom > 0 else float("nan"),
    }


def bootstrap_by_page(page_rows, resamples: int, seed: int) -> dict:
    """page_rows: list of (miss_g, miss_l) arrays per page; pages are resampled with replacement."""
    rng = np.random.default_rng(seed)
    k = len(page_rows)
    phis, gs = [], []
    for _ in range(resamples):
        idx = rng.integers(0, k, size=k)
        g = np.concatenate([page_rows[i][0] for i in idx])
        l = np.concatenate([page_rows[i][1] for i in idx])
        s = stats_from_indicators(g, l)
        if not math.isnan(s["phi"]):
            phis.append(s["phi"])
        gs.append(s["G_observed"])
    lo, hi = np.percentile(phis, [2.5, 97.5]) if phis else (float("nan"), float("nan"))
    glo, ghi = np.percentile(gs, [2.5, 97.5])
    return {"resamples": resamples, "phi_ci95": [float(lo), float(hi)],
            "G_observed_ci95": [float(glo), float(ghi)], "phi_defined_resamples": len(phis)}


# ---------------------------------------------------------------- crop geometry

def crop_transform(block_polygon, page_w: int, page_h: int, margin: float = MARGIN):
    """Crop box (clipped to the page) and the scale 1800 / H_page applied to it."""
    xs = [p[0] for p in block_polygon]
    ys = [p[1] for p in block_polygon]
    x0 = max(0, math.floor(min(xs)) - margin)
    y0 = max(0, math.floor(min(ys)) - margin)
    x1 = min(page_w, math.ceil(max(xs)) + margin)
    y1 = min(page_h, math.ceil(max(ys)) + margin)
    return {"origin": (float(x0), float(y0)), "size": (int(x1 - x0), int(y1 - y0)),
            "scale": BLLA_HEIGHT / float(page_h)}


def crop_box(tr):
    """Crop content box in page edge coordinates: (x0, y0, x1, y1)."""
    x0, y0 = tr["origin"]
    w, h = tr["size"]
    return (x0, y0, x0 + w, y0 + h)


def line_in_box(baseline, box) -> bool:
    """A GT line belongs to a crop when at least one of its baseline points lies inside the box."""
    x0, y0, x1, y1 = box
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in baseline)


def gt_lines_in_box(gt_lines, box):
    return [g for g in gt_lines if line_in_box(g.baseline, box)]


def crop_to_page(points, transform):
    """Canvas coordinates (scaled crop, padded) -> page edge coordinates."""
    ox, oy = transform["origin"]
    s = transform["scale"]
    return [[ox + x / s, oy + y / s] for x, y in points]


def page_to_crop_canvas(points, transform):
    ox, oy = transform["origin"]
    s = transform["scale"]
    return [[(x - ox) * s, (y - oy) * s] for x, y in points]


# ---------------------------------------------------------------- run rules (pure)

def validate_caps(pages_limit, max_crops, deadline_s) -> None:
    """Crop and deadline caps belong to the one-page chronometer only."""
    if (max_crops is not None or deadline_s is not None) and pages_limit != 1:
        raise RuntimeError("--max-crops and --deadline-s are refused outside the one-page chronometer")


def interpret_G(lo: float, hi: float) -> dict:
    """Reading fixed before the run: lower bound > 0 is an oracle reframing effect (not a system gain);
    an interval containing 0 is 'not distinguished from 0'; 'futile' only if the upper bound < 0.02."""
    return {"oracle_reframing_effect": bool(lo > 0),
            "not_distinguished_from_0": bool(lo <= 0 <= hi),
            "futile_upper_below_0_02": bool(hi < FUTILITY_UPPER),
            "ci95": [float(lo), float(hi)]}


def finalize_status(interrupted, guard_hit: bool, chrono: bool) -> str:
    if chrono:
        return "COST_PROBE_ONLY_NO_RESULT"
    if interrupted or guard_hit:
        return "INCOMPLETE"
    return "COMPLETE"


def resume_allowed(partial_meta: dict, current_meta: dict, partial_guard_hit: bool = False) -> bool:
    """Resume only if no guard was hit and config, script, model and the babaorum modules are identical."""
    if partial_guard_hit:
        return False
    keys = ("config_sha256", "script_sha256", "model_sha256", "module_sha256")
    return all(partial_meta.get(k) == current_meta.get(k) for k in keys)


def check_run_gate(cfg, pages_limit, out_path: Path) -> None:
    """One-page chronometer only, unless the registered run_gate is OPEN."""
    gate = cfg.get("run_gate", {})
    if pages_limit == 1:
        if out_path.name != CHRONO_OUTPUT:
            raise RuntimeError(f"one-page chronometer must write {CHRONO_OUTPUT}, not {out_path.name}")
        return
    if gate.get("full_run") != "OPEN":
        raise RuntimeError("full run refused: run_gate.full_run is not OPEN in the registered config")
    if out_path.name != FULL_OUTPUT:
        raise RuntimeError(f"full run must write {FULL_OUTPUT}, not {out_path.name}")


# ---------------------------------------------------------------- BLLA (kraken env only)

def load_blla(model_path: Path):
    """Load the BLLA model. The 2026 Zenodo file has no 'hyper_params' metadata, which kraken 7.1
    reads unconditionally; an empty mapping makes kraken use its own defaults (padding (0, 0))."""
    from kraken.lib.vgsl import TorchVGSLModel
    model = TorchVGSLModel.load_model(str(model_path))
    model.user_metadata.setdefault("hyper_params", {})
    return model


def blla_lines(image, model) -> list[list[list[float]]]:
    from kraken import blla
    seg = blla.segment(image, text_direction="horizontal-lr", model=model, device="cpu")
    out = []
    for line in seg.lines:
        baseline = getattr(line, "baseline", None)
        if baseline is None or len(baseline) < 2:
            continue
        out.append([[float(x), float(y)] for x, y in baseline])
    return out


def crop_image_for_blla(page_image, transform):
    from PIL import Image
    x0, y0 = map(int, transform["origin"])
    w, h = transform["size"]
    crop = page_image.crop((x0, y0, x0 + w, y0 + h))
    s = transform["scale"]
    new = (max(1, round(w * s)), max(1, round(h * s)))
    scaled = crop.resize(new, Image.BILINEAR)
    canvas = Image.new(scaled.mode, (scaled.width, BLLA_HEIGHT), color=255)
    canvas.paste(scaled, (0, 0))
    return canvas


# ---------------------------------------------------------------- sampling and heights

def sample_pages(cfg, page_ids):
    rng_pages = random.Random(cfg["seed"])
    return sorted(rng_pages.sample(sorted(page_ids), cfg["sampling"]["pages"]["n"]))


def sample_lines(page_json_lines, n, seed):
    readable = [ln for ln in page_json_lines if ln.get("legibility") == "readable"]
    rng = random.Random(seed)
    if len(readable) < n:
        return readable
    idx = sorted(rng.sample(range(len(readable)), n))
    return [readable[i] for i in idx]


def _height(ln):
    """Band height (neg + pos) derived from the baseline and polygon, as in stage_s_data."""
    from babaorum.experimental.stage_s_data import _band
    neg, pos = _band(ln["baseline"], ln["polygon"])
    return neg + pos


def _matched_gt(result):
    return {m.gt_id for m in result.matches}


# ---------------------------------------------------------------- pipeline

def write_partial(path: Path, meta: dict, per_page: list, elapsed_cumulative_s: float, guard_hit: bool) -> None:
    tmp = path.with_suffix(".tmp")
    payload = {"meta": meta, "per_page": per_page, "elapsed_cumulative_s": elapsed_cumulative_s,
               "guard_hit": guard_hit}
    tmp.write_text(json.dumps(payload, allow_nan=False, default=float) + "\n")
    tmp.replace(path)


def crop_entry(matched, gt_ids, n_pred: int, box) -> dict:
    """JSON-serialisable record of one block crop (sets are stored as sorted lists)."""
    return {"matched": sorted(matched), "gt_ids": sorted(gt_ids), "n_pred": int(n_pred),
            "box": [float(v) for v in box]}


def build_rows(sampled, crop_info, g_match, gs_match):
    """Rows for the statistics. A sampled line with no baseline point inside its block crop is
    excluded from r, m, phi and G and reported as excluded_not_in_crop."""
    rows, excluded = [], []
    for ln in sampled:
        ci = crop_info.get(ln["block_id"])
        in_set = bool(ci and ln["id"] in ci["gt_ids"])
        if not in_set:
            excluded.append({"id": ln["id"], "block_id": ln["block_id"], "reason": "excluded_not_in_crop"})
            continue
        rows.append({"id": ln["id"], "block_id": ln["block_id"],
                     "global_hit": ln["id"] in g_match,
                     "local_hit": bool(ln["id"] in ci["matched"]),
                     "in_local_gt_set": True,
                     "global_hit_1h": ln["id"] in gs_match})
    return rows, excluded


def page_pass(pid, root, cfg, model, skip_blla, max_crops, deadline_s, started, interrupted):
    """One page: global pass, then one local pass per sampled block.
    The local ground truth of a crop is every GT line with a baseline point inside the crop box."""
    t_page = time.monotonic()
    page = json.loads((root / "pages" / f"{pid}.json").read_text())
    w, h = page["image"]["width"], page["image"]["height"]
    gt_all = [GroundTruthLine(id=ln["id"], baseline=ln["baseline"], height_px=_height(ln),
                              baseline_source="ground_truth") for ln in page["lines"] if _height(ln) > 0]
    sampled = sample_lines(page["lines"], cfg["sampling"]["lines_per_page"]["n"], cfg["seed"])
    blocks = {b["id"]: b for b in page["blocks"]}

    seconds_global = 0.0
    seconds_crops = 0.0
    global_pred = []
    crop_info = {}
    n_crops = 0
    if not skip_blla:
        from PIL import Image
        page_image = Image.open(root / "images" / f"{pid}.png").convert("L")
        t0 = time.monotonic()
        global_pred = blla_lines(page_image, model)
        seconds_global = time.monotonic() - t0
        t1 = time.monotonic()
        for ln in sampled:
            bid = ln["block_id"]
            if bid in crop_info:
                continue
            if max_crops is not None and n_crops >= max_crops:
                interrupted.append({"page_id": pid, "line_id": ln["id"], "reason": "max_crops"})
                continue
            if deadline_s is not None and time.monotonic() - started > deadline_s:
                interrupted.append({"page_id": pid, "line_id": ln["id"], "reason": "deadline"})
                continue
            tr = crop_transform(blocks[bid]["polygon"], w, h)
            box = crop_box(tr)
            gt_crop = gt_lines_in_box(gt_all, box)
            crop_img = crop_image_for_blla(page_image, tr)
            preds = [crop_to_page(poly, tr) for poly in blla_lines(crop_img, model)]
            lp = [PredictedLine(id=f"l{bid}_{i:04d}", baseline=p, baseline_source="predicted")
                  for i, p in enumerate(preds)]
            matched = _matched_gt(evaluate_line_geometry(gt_crop, lp)) if (gt_crop and lp) else set()
            crop_info[bid] = crop_entry(matched, [g.id for g in gt_crop], len(lp), box)
            n_crops += 1
        seconds_crops = time.monotonic() - t1

    glob_preds = [PredictedLine(id=f"g{i:05d}", baseline=p, baseline_source="predicted")
                  for i, p in enumerate(global_pred)]
    g_match = _matched_gt(evaluate_line_geometry(gt_all, glob_preds)) if glob_preds else set()
    gs_match = _matched_gt(evaluate_line_geometry_sensitivity(gt_all, glob_preds)) if glob_preds else set()

    rows, excluded = build_rows(sampled, crop_info, g_match, gs_match)
    return {"page_id": pid, "page_sha256": sha256(root / "pages" / f"{pid}.json"),
            "image_sha256": sha256(root / "images" / f"{pid}.png"),
            "n_gt_lines": len(gt_all), "n_global_pred": len(global_pred), "n_crops": n_crops,
            "seconds_global": seconds_global, "seconds_crops": seconds_crops,
            "seconds_page": time.monotonic() - t_page,
            "crops": crop_info, "rows": rows, "excluded": excluded}


def kraken_version():
    try:
        import importlib.metadata as md
        return md.version("kraken")
    except Exception:
        return None


def module_hashes() -> dict:
    return {rel: sha256(ROOT / rel) for rel in MODULE_FILES}


def run(cfg, out_path: Path, pages_limit=None, skip_blla=False, max_crops=None, deadline_s=None,
        guard_seconds=GUARD_SECONDS_DEFAULT):
    check_run_gate(cfg, pages_limit, out_path)
    validate_caps(pages_limit, max_crops, deadline_s)
    chrono = pages_limit == 1
    started = time.monotonic()
    root = Path(cfg["source"]["root"]).expanduser()
    page_ids = sorted(p.stem for p in (root / "pages").glob("*.json"))
    chosen = sample_pages(cfg, page_ids)
    if pages_limit is not None:
        chosen = chosen[:pages_limit]
    model_path = Path(cfg["model"]["path"]).expanduser()
    model_sha = None if skip_blla else sha256(model_path)
    if not skip_blla and model_sha != cfg["model"]["sha256"]:
        raise RuntimeError("BLLA model sha256 does not match the pre-registered value")
    meta = {"config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__).resolve()),
            "model_sha256": model_sha, "module_sha256": module_hashes()}

    partial_path = out_path.with_name(out_path.stem + ".partial.json")
    done, resumed, elapsed_offset = {}, [], 0.0
    if partial_path.exists():
        prev = json.loads(partial_path.read_text())
        if prev.get("guard_hit"):
            raise RuntimeError("partial file records a time-guard hit: resume refused, status is final INCOMPLETE")
        if not resume_allowed(prev["meta"], meta, partial_guard_hit=False):
            raise RuntimeError("partial file was written with different config/script/model/module sha256: "
                               "refuse to resume; move the partial file away to start again")
        done = {p["page_id"]: p for p in prev["per_page"]}
        resumed = sorted(done)
        elapsed_offset = float(prev.get("elapsed_cumulative_s", 0.0))

    def elapsed():
        return elapsed_offset + (time.monotonic() - started)

    model = None if skip_blla else load_blla(model_path)
    per_page, interrupted, guard_hit = [], [], False
    for pid in chosen:
        if pid in done:
            per_page.append(done[pid])
            continue
        if elapsed() > guard_seconds:
            guard_hit = True
            break
        per_page.append(page_pass(pid, root, cfg, model, skip_blla, max_crops, deadline_s, started, interrupted))
        write_partial(partial_path, meta, per_page, elapsed(), guard_hit=False)
    if guard_hit:
        write_partial(partial_path, meta, per_page, elapsed(), guard_hit=True)

    status = finalize_status(interrupted, guard_hit or len(per_page) < len(chosen), chrono)
    report = {
        "schema_version": "g_intra_mf_report_v3",
        "status": status,
        "stratum": "C-in (GT blocks); C-out empty by construction",
        "estimand": "G = (1 - P(miss_G and miss_L)) - (1 - min(r, m)), inside GT blocks",
        "pages_planned": chosen,
        "pages_done": [p["page_id"] for p in per_page],
        "resumed": bool(resumed), "resumed_pages": resumed,
        "interrupted_lines": interrupted,
        "guard_seconds": guard_seconds, "guard_hit": guard_hit,
        "elapsed_cumulative_seconds": elapsed(),
        "timing_totals": {"seconds_global": sum(p["seconds_global"] for p in per_page),
                          "seconds_crops": sum(p["seconds_crops"] for p in per_page),
                          "n_crops": sum(p["n_crops"] for p in per_page)},
        "excluded_not_in_crop": [e for p in per_page for e in p.get("excluded", [])],
        "per_page": per_page,
        "provenance": {**meta, "kraken": kraken_version(), "python": platform.python_version(),
                       "numpy": np.__version__, "platform": platform.platform(),
                       "partial_file": partial_path.name},
    }
    if status == "COMPLETE":
        rows = [r for p in per_page for r in p["rows"]]
        miss_g = [0 if r["global_hit"] else 1 for r in rows]
        miss_l = [0 if r["local_hit"] else 1 for r in rows]
        report["primary_tolerance_0.5h"] = stats_from_indicators(miss_g, miss_l)
        page_rows = [(np.array([0 if r["global_hit"] else 1 for r in p["rows"]]),
                      np.array([0 if r["local_hit"] else 1 for r in p["rows"]])) for p in per_page]
        boot = bootstrap_by_page(page_rows, cfg["statistics"]["bootstrap"]["resamples"],
                                 cfg["statistics"]["bootstrap"]["seed"])
        report["bootstrap_by_page"] = boot
        report["G_interpretation"] = interpret_G(boot["G_observed_ci95"][0], boot["G_observed_ci95"][1])
        report["both_hit"] = sum(1 for r in rows if r["global_hit"] and r["local_hit"])
        report["duplicate_predictions"] = None
        report["duplicate_predictions_reason"] = ("not computed: evaluate_line_geometry assigns one-to-one, "
                                                  "so multi-matches cannot occur by construction")
    else:
        report["statistics"] = "NOT READ: status is not COMPLETE"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, allow_nan=False, default=float) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pages", type=int, default=None, help="1 = one-page chronometer; omit for the full run")
    parser.add_argument("--skip-blla", action="store_true", help="statistics and matching only, no model call")
    parser.add_argument("--max-crops", type=int, default=None, help="chronometer only")
    parser.add_argument("--deadline-s", type=float, default=None, help="chronometer only")
    args = parser.parse_args(argv)
    validate_caps(args.pages, args.max_crops, args.deadline_s)
    cfg = json.loads(args.config.read_text())
    output = args.output or ROOT / "reports" / (CHRONO_OUTPUT if args.pages == 1 else FULL_OUTPUT)
    report = run(cfg, output, pages_limit=args.pages, skip_blla=args.skip_blla,
                 max_crops=args.max_crops, deadline_s=args.deadline_s,
                 guard_seconds=cfg.get("time_guard_hours", 5) * 3600)
    print(json.dumps({"status": report["status"], "pages_done": len(report["pages_done"]),
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}))
    return report


if __name__ == "__main__":
    main()
