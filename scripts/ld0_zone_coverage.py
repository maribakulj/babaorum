"""LD-0 — couverture des lignes GT par les zones PP-DocLayoutV3 (mf_0000..mf_0049 seulement ; INDICATEUR, pas une preuve).

Lot : docs/END_TO_END_LD0_LOT.txt (p6, 18/10/2026), amendements 1 à 3 (§11, §12, §13). Classe A.
Pour chaque page : détecteur PP-DocLayoutV3 sur l'image NATIVE (révision fixée, seuil 0,1, 1 tuile) ; blocs gardés
(axel select_blocks, conf 0,3, dédoublonnage 0,8, is_read) ; deux découpages en zones (Z-PP-blocs : une zone = un bloc ;
Z-PP-colonnes : merge_columns, écart 0,02 x hauteur native, zone <= 2 400 px natifs) ; contrôle Z-GT (blocs GT contenant
une ligne). Zones mises à l'échelle (W_s/W, H_s/H) ; baselines GT par factor (load_scaled_page).

Affectation déclarée (§13 (2)) : une ligne admissible dans plusieurs zones est attribuée à UNE zone, celle où la part de
ses points est la plus grande (égalité : indice le plus petit). Le compte NON exclusif est rapporté en descriptif.

Entrées : découverte par discover_inputs (listes blanches explicites de config, sans recherche récursive) ; les 100
fichiers de mf_0000..mf_0049 sont COPIÉS par le carnet à partir de chemins construits depuis la liste codée.
Sorties sous --out-dir seulement (/kaggle/working sur Kaggle).
"""
from __future__ import annotations

import argparse
import datetime
import faulthandler
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ld0_zone_coverage.json"
KAGGLE_OUT = Path("/kaggle/working")
KIT_FILES = ["scripts/ld0_zone_coverage.py", "config/ld0_zone_coverage.json", "tests/test_ld0_zone_coverage.py"]
DIVISIONS = ("Z-PP-blocs", "Z-PP-colonnes", "Z-GT")
FORBIDDEN_SUBPATHS = ("e2e3/mf/pages", "e2e3/mf/images")   # §13 (1): never listed nor traversed
# §14 (10) and (11): set in os.environ before the first import of huggingface_hub, and in the env of the steps
HF_ENV = {"HF_HUB_DOWNLOAD_TIMEOUT": "60", "HF_HUB_ETAG_TIMEOUT": "30", "HF_HUB_DISABLE_XET": "1"}

# imported lazily (run time only): the bundle's babaorum and the axel vendor package must be on sys.path
is_read = normalize_blocks = Params = merge_columns = select_blocks = bbox = None
load_mf_page = GroundTruthLine = _sample = _validated_gt = None


def _load_deps() -> None:
    if bbox is not None:
        return
    sys.path.insert(0, str(ROOT / "scripts" / "ld0_vendor"))
    sys.path.insert(0, str(ROOT / "src"))
    from axel.layout.blocks import is_read as _is_read, normalize_blocks as _nb
    from axel.layout.chains import Params as _P, merge_columns as _mc, select_blocks as _sb
    from axel.layout.geom import bbox as _bbox
    from babaorum.experimental.stage_s_data import load_mf_page as _lmp
    from babaorum.line_geometry import GroundTruthLine as _GTL, _sample as _s, _validated_gt as _vg
    globals().update(is_read=_is_read, normalize_blocks=_nb, Params=_P, merge_columns=_mc, select_blocks=_sb,
                     bbox=_bbox, load_mf_page=_lmp, GroundTruthLine=_GTL, _sample=_s, _validated_gt=_vg)


class ResumeIncompatible(RuntimeError):
    """§13 (4): a stored measure comes from another launch (script, config, vendor, model or page inputs)."""


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------- discovery (§13 (1)) ----------

def _child_dirs(d: Path) -> list:
    """Sub-directories of d, listed by name. §14 (12): never listed: a directory that is or contains e2e3/mf (checked
    with is_dir on an explicit path, not by listing) and anything under e2e3/mf."""
    if d.name == "mf" and d.parent.name == "e2e3":
        return []
    if d.name.startswith("mf"):              # page-like folders (mf_0050 decoys included): never listed
        return []
    if (d / "e2e3" / "mf").is_dir():
        return []
    return sorted(c for c in d.iterdir() if c.is_dir())


def discover_inputs(input_root: Path = Path("/kaggle/input")) -> dict:
    """§13 (1) and §14 (12): lists /kaggle/input and its directories of levels 1, 2 and 3 (names only, never inside
    e2e3/mf), then checks the EXPLICIT relative paths of the white list in the config under each root: each must be
    found exactly once. Returns the found paths and the roots used, as names relative to input_root."""
    cfg = json.loads(CONFIG.read_text())
    relpaths = cfg["discovery"]["relpaths"]
    for name, rel in relpaths.items():
        if rel.startswith("/") or any(bad in rel for bad in FORBIDDEN_SUBPATHS):
            raise ValueError(f"white list entry refused: {name} = {rel}")
    roots = [input_root]
    level1 = _child_dirs(input_root)
    level2 = []
    for d in level1:
        level2 += _child_dirs(d)
    level3 = []
    for d in level2:
        level3 += _child_dirs(d)
    roots += level1 + level2 + level3
    roots = list(dict.fromkeys(roots))
    found, root_of = {}, {}
    for name, rel in relpaths.items():
        hits = [(r, r / rel) for r in roots if (r / rel).exists()]
        if len(hits) != 1:
            raise RuntimeError(f"discovery: {name} ({rel}) found {len(hits)} times, expected 1: "
                               f"{[str(h[1]) for h in hits]}")
        root_of[name], found[name] = hits[0]
    names = sorted({"." if r == input_root else str(r.relative_to(input_root)) for r in root_of.values()})
    found["roots_used"] = names                # §14 (12): folder names only, down to level 3
    return found


def find_kit_script(candidates) -> Path:
    """§14 (12): the kit script at one of the explicit candidate paths (is_file only, no listing); exactly one."""
    hits = [Path(c) for c in candidates if Path(c).is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"kit script found {len(hits)} times among the candidates, expected 1: {[str(h) for h in hits]}")
    return hits[0]


def check_vendor(cfg: dict) -> None:
    """The vendored axel files are the ones of the config (commit 7b75303), before any import from them."""
    for rel, want in cfg["vendor"]["files"].items():
        got = sha256(ROOT / "scripts" / "ld0_vendor" / rel)
        if got != want:
            raise RuntimeError(f"vendored file differs from the config: {rel} {got}")


def launch_identity(cfg: dict) -> dict:
    """§13 (4): what a stored measure must match: script, config, vendor, model revision and sha."""
    return {"script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(CONFIG),
            "vendor_sha256": sha256_text(json.dumps(cfg["vendor"]["files"], sort_keys=True)),
            "model_revision": cfg["detector"]["revision"],
            "model_sha256": cfg["detector"]["files"]["model.safetensors"]["sha256"]}


# ---------- geometry (pure) ----------

def scale_box(box, sx: float, sy: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def expand(box, margin: float):
    x0, y0, x1, y1 = box
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def zones_from_blocks(kept_boxes_native, H_native: float, merge_gap_fraction: float, max_h: float):
    """Z-PP-blocs = one zone per kept block; Z-PP-colonnes = merge_columns groups, zone = union of members
    (axel zones_of, e12_kraken.py). Native frame."""
    blocks = [list(b) for b in kept_boxes_native]
    divisions = {"Z-PP-blocs": [tuple(b) for b in blocks]}
    groups = merge_columns(blocks, merge_gap_fraction * H_native, max_h) if blocks else []
    divisions["Z-PP-colonnes"] = [union_box([blocks[i] for i in g]) for g in groups]
    return divisions, groups


def measure_lines(samples: list[np.ndarray], zones: list[tuple], margin: float, a_thr: float, a_prime_thr: float,
                  b_thr: float) -> dict:
    """Rules (a), (a'), (b), (c) over ALL validated GT lines of the page (§11 (6)).
    Assignment (§13 (2)): a covered line goes to ONE zone, its best zone (largest share, smallest index on ties).
    The non-exclusive count (every zone counts every admissible line) is reported as a descriptive figure."""
    n_lines = len(samples)
    if n_lines == 0:
        raise ValueError("no validated GT line on the page")
    ex = [expand(z, margin) for z in zones]
    one_frac = np.zeros((n_lines, len(ex)))
    union_frac = np.zeros(n_lines)
    for i, pts in enumerate(samples):
        if not len(ex):
            continue
        inside = np.zeros((pts.shape[0], len(ex)), dtype=bool)
        for z, (x0, y0, x1, y1) in enumerate(ex):
            inside[:, z] = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        one_frac[i] = inside.mean(axis=0)
        union_frac[i] = inside.any(axis=1).mean()
    best_zone = one_frac.argmax(axis=1) if len(ex) else np.full(n_lines, -1)
    best_frac = one_frac.max(axis=1) if len(ex) else np.zeros(n_lines)
    a = best_frac >= a_thr
    a_prime = best_frac >= a_prime_thr
    b = union_frac >= b_thr
    nonexclusive = (one_frac >= a_thr).sum(axis=0) if len(ex) else np.zeros(0, dtype=int)
    return {
        "n_lines": n_lines,
        "a": float(a.mean()), "a_prime": float(a_prime.mean()), "b": float(b.mean()),
        "c": float((b & ~a).mean()),
        "covered_one": a, "best_zone": best_zone, "best_frac": best_frac, "union_frac": union_frac,
        "nonexclusive_counts": [int(v) for v in nonexclusive],
    }


def lines_per_zone(measure: dict, n_zones: int, mask=None) -> list[int]:
    """(d): exclusive count per zone (§13 (2)); a line cut between zones is in none. mask restricts the lines."""
    counts = [0] * n_zones
    for i, (covered, z) in enumerate(zip(measure["covered_one"], measure["best_zone"])):
        if mask is not None and not mask[i]:
            continue
        if covered and z >= 0:
            counts[int(z)] += 1
    return counts


def fraction_in_small_zones(measure: dict, counts: list[int], limit: int) -> float:
    """§13 (3): share of ALL validated lines whose exclusive zone holds at most `limit` lines."""
    n = measure["n_lines"]
    kept = sum(1 for covered, z in zip(measure["covered_one"], measure["best_zone"])
               if covered and z >= 0 and counts[int(z)] <= limit)
    return kept / n


# ---------- reading (§11 (10)) ----------

def reading(stats: dict, cfg_reading: dict) -> dict:
    """stats[div]['a_mean'], stats[div]['p95'] for the two cutting divisions. Indicators, not the evaluation."""
    usable_thr = cfg_reading["usable_mean_a"]
    n_ok = cfg_reading["N_if_p95_ok"]
    cands = [d for d in ("Z-PP-blocs", "Z-PP-colonnes") if stats[d]["a_mean"] >= usable_thr]
    if not cands:
        return {"choice": None, "N": None, "action": "lot de correction des zones avant tout GPU",
                "usable": []}
    p95_ok = [d for d in cands if stats[d]["p95"] is not None and stats[d]["p95"] <= n_ok]
    if len(cands) == 1:
        pick, reason = cands[0], "un seul découpage atteint 0,92"
    elif len(p95_ok) == 2 or len(p95_ok) == 0:
        best = max(cands, key=lambda d: (stats[d]["a_mean"], d == "Z-PP-blocs"))
        pick, reason = best, ("les deux ont p95 <= 32 : plus grand a_mean" if p95_ok
                              else "aucun n'a p95 <= 32 : plus grand a_mean")
    else:
        pick, reason = p95_ok[0], "p95 <= 32 pour ce seul découpage"
    N = n_ok if stats[pick]["p95"] is not None and stats[pick]["p95"] <= n_ok else cfg_reading["N_else"]
    return {"choice": pick, "N": N, "reason": reason, "usable": cands,
            "caveat": "INDICATEUR sur pages d'entraînement : la couverture et le p95 ne prouvent pas que le rappel "
                      "0,90 est atteignable (zones très chargées possibles)."}


# ---------- I/O (the only place where files are opened; tests spy on these) ----------

def _read_page_json(root: Path, pid: str) -> dict:
    return json.loads((root / "pages" / f"{pid}.json").read_text())


def _load_page(root: Path, pid: str):
    """Page with line bands (load_mf_page, which checks the PNG sha against the JSON). Explicit paths only."""
    return load_mf_page(root / "pages" / f"{pid}.json", root / "images" / f"{pid}.png")


def _page_factor(lines, target_h: float) -> float:
    """Same rule as train_stage_s.load_scaled_page: factor = target / median(normal_negative + normal_positive)."""
    heights = [ln["normal_negative_px"] + ln["normal_positive_px"] for ln in lines]
    return float(target_h / float(np.median(heights)))


def download_detector(cfg_det: dict, cache_dir: Path) -> dict:
    """§14 (10): the files of the fixed revision, each checked by sha256. Returns {name: path}."""
    from huggingface_hub import hf_hub_download
    local = {}
    for name, info in cfg_det["files"].items():
        path = Path(hf_hub_download(cfg_det["repo"], name, revision=cfg_det["revision"], cache_dir=str(cache_dir)))
        got = sha256(path)
        if got != info["sha256"]:
            raise RuntimeError(f"sha256 of {name} differs: {got}")
        local[name] = path
    return local


def clean_partial_cache(cache_dir: Path, repo: str) -> int:
    """§14 (10): removes the partial downloads (*.incomplete) of the repository, and returns their number."""
    blobs = Path(cache_dir) / ("models--" + repo.replace("/", "--")) / "blobs"
    if not blobs.is_dir():
        return 0
    removed = 0
    for f in blobs.iterdir():
        if f.name.endswith(".incomplete"):
            f.unlink()
            removed += 1
    return removed


def load_detector(cfg_det: dict, cache_dir: Path):
    """Fixed revision; each file checked by sha256 after download. Returns (processor, model, effective_size)."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    local = download_detector(cfg_det, cache_dir)
    folder = local["config.json"].parent
    proc = AutoImageProcessor.from_pretrained(folder)
    model = AutoModelForObjectDetection.from_pretrained(folder).eval()
    size = getattr(proc, "size", None)
    return proc, model, size


def detect(proc, model, image_rgb: Image.Image, floor: float) -> dict:
    """e12_detect: one tile, native image, post_process with the native target size; boxes and polygon_points kept."""
    import torch
    inputs = proc(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    res = proc.post_process_object_detection(outputs, threshold=floor,
                                             target_sizes=[(image_rgb.height, image_rgb.width)])[0]
    polys = res.get("polygon_points")
    blocks = []
    for k, (score, label, box) in enumerate(zip(res["scores"], res["labels"], res["boxes"], strict=True)):
        x0, y0, x1, y1 = (float(v) for v in box.tolist())
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if polys is not None and len(polys) > k and polys[k] is not None:
            pp = polys[k].tolist() if hasattr(polys[k], "tolist") else polys[k]
            if len(pp) >= 3:
                poly = [[float(a), float(b)] for a, b in pp]
        blocks.append({"id": f"b{k}", "label": model.config.id2label[int(label)], "score": float(score),
                       "polygon": poly})
    return {"width": image_rgb.width, "height": image_rgb.height, "blocks": blocks}


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _majority(categories) -> str:
    counts: dict[str, int] = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ---------- one page ----------

def process_page(pid: str, root: Path, det_fn, cfg: dict) -> dict:
    t0 = time.monotonic()
    page = _load_page(root, pid)
    raw = _read_page_json(root, pid)
    W = int(raw["image"]["width"])
    H = int(raw["image"]["height"])
    factor = _page_factor(page["lines"], cfg["measure"]["target_median_height_px"])
    W_s, H_s = round(W * factor), round(H * factor)
    sx, sy = W_s / W, H_s / H
    t_read = time.monotonic() - t0
    t1 = time.monotonic()
    det_raw = det_fn(pid, page)
    t_det = time.monotonic() - t1
    t2 = time.monotonic()
    det = normalize_blocks(det_raw, image_size=(W, H))
    kept = select_blocks(det["blocks"], Params(conf=cfg["zones"]["conf"], dedup=cfg["zones"]["dedup"]))
    kept = [b for b in kept if is_read(b.get("label"))]
    kept_boxes = [bbox([(float(x), float(y)) for x, y in b["polygon"]]) for b in kept]
    kept_cats = [b["category"] for b in kept]
    divisions, groups = zones_from_blocks(kept_boxes, H, cfg["zones"]["gap_fraction_of_native_height"],
                                          cfg["zones"]["max_h_native_px"])
    cat_zones = {"Z-PP-blocs": list(kept_cats),
                 "Z-PP-colonnes": [_majority([kept_cats[i] for i in g]) for g in groups]}
    block_of = {ln["id"]: ln.get("block_id") for ln in raw["lines"]}
    cat_of_block = {blk["id"]: blk["category"] for blk in raw["blocks"]}
    gt_lines = []
    for ln, neg_pos in zip(raw["lines"], page["lines"]):
        baseline = [[x * factor, y * factor] for x, y in ln["baseline"]]
        height = (neg_pos["normal_negative_px"] + neg_pos["normal_positive_px"]) * factor
        gt_lines.append(GroundTruthLine(id=ln["id"], baseline=baseline, height_px=height,
                                        baseline_source="ground_truth", geometry_status="available"))
    evaluable, _incomplete = _validated_gt(gt_lines)
    samples, heights, line_cats = [], [], []
    for line, points, height in evaluable:
        spacing = min(cfg["measure"]["sample_spacing_max_px"], height / cfg["measure"]["sample_spacing_height_divisor"])
        samples.append(np.asarray(_sample(points, spacing), dtype=float))
        heights.append(height)
        line_cats.append(cat_of_block.get(block_of.get(line.id), "inconnue"))
    if not samples:
        raise ValueError(f"{pid}: no validated GT line")
    gt_block_boxes, gt_block_cats = [], []
    for blk in raw["blocks"]:
        if blk.get("line_ids"):
            gt_block_boxes.append(bbox([(float(x), float(y)) for x, y in blk["polygon"]]))
            gt_block_cats.append(blk["category"])
    divisions["Z-GT"] = [tuple(b) for b in gt_block_boxes]
    cat_zones["Z-GT"] = gt_block_cats
    scaled = {d: [scale_box(b, sx, sy) for b in boxes] for d, boxes in divisions.items()}
    t_det_side = time.monotonic() - t2
    t3 = time.monotonic()
    result = {"page_id": pid, "W": W, "H": H, "factor": factor, "W_s": W_s, "H_s": H_s,
              "n_detections": len(det["blocks"]), "n_kept": len(kept),
              "categories_kept": _count(kept_cats), "labels_kept": _count([b["label"] for b in kept]),
              "lines_by_category": _count(line_cats),
              "n_gt_lines_total": len(gt_lines), "n_gt_lines_validated": len(evaluable),
              "heights_scaled": heights,
              "scale_gap_px": {"dx": abs(W_s - W * factor), "dy": abs(H_s - H * factor)}}
    hb = np.asarray(heights) > cfg["measure"]["height_bucket_px"]
    measures = {}
    for div in DIVISIONS:
        measures[div] = {}
        nz = len(scaled[div])
        for mname, margin in (("marge16", cfg["measure"]["margin_px"]),
                              ("marge0", cfg["measure"]["margin_variant_px"])):
            m = measure_lines(samples, scaled[div], margin, cfg["measure"]["threshold_a"],
                              cfg["measure"]["threshold_a_prime"], cfg["measure"]["threshold_b"])
            lpz = lines_per_zone(m, nz)
            cov = m["covered_one"]
            result_m = {k: m[k] for k in ("n_lines", "a", "a_prime", "b", "c")}
            result_m["a_height_le_32"] = float(cov[~hb].mean()) if (~hb).any() else None
            result_m["a_height_gt_32"] = float(cov[hb].mean()) if hb.any() else None
            result_m["lines_le_32"] = int((~hb).sum())
            result_m["lines_gt_32"] = int(hb.sum())
            result_m["covered_le_32"] = int(cov[~hb].sum())
            result_m["covered_gt_32"] = int(cov[hb].sum())
            result_m["n_zones"] = nz
            result_m["lines_per_zone"] = lpz
            result_m["lines_per_zone_le_32"] = lines_per_zone(m, nz, mask=~hb)
            result_m["lines_per_zone_gt_32"] = lines_per_zone(m, nz, mask=hb)
            result_m["lines_per_zone_nonexclusive"] = m["nonexclusive_counts"]
            result_m["frac_lines_in_zone_le_32"] = fraction_in_small_zones(m, lpz, 32)
            result_m["frac_lines_in_zone_le_64"] = fraction_in_small_zones(m, lpz, 64)
            by_cat: dict[str, list[int]] = {}
            for n_in_zone, cat in zip(lpz, cat_zones[div]):
                by_cat.setdefault(cat, []).append(n_in_zone)
            result_m["lines_per_zone_by_category"] = by_cat
            measures[div][mname] = result_m
    result["measures"] = measures
    result["zone_sizes_scaled"] = {d: [[round(b[2] - b[0], 2), round(b[3] - b[1], 2)] for b in scaled[d]]
                                   for d in DIVISIONS}
    result["zones_native"] = {d: [list(map(float, b)) for b in divisions[d]] for d in DIVISIONS}
    result["zones_scaled"] = {d: [list(map(float, b)) for b in scaled[d]] for d in DIVISIONS}
    t_meas = t_det_side + (time.monotonic() - t3)
    result["timings_s"] = {"read_and_control": round(t_read, 3), "detector": round(t_det, 3),
                           "measures": round(t_meas, 3), "total": round(time.monotonic() - t0, 3)}
    return result


# ---------- aggregation (pooled over pages) ----------

def _dist(values) -> dict:
    if not values:
        return {"n_zones": 0, "median": None, "p95": None, "max": None}
    return {"n_zones": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),   # numpy.percentile, linear (§11 (10))
            "max": int(max(values))}


def aggregate(per_page: list[dict], cfg: dict) -> dict:
    agg: dict = {}
    for div in DIVISIONS:
        agg[div] = {}
        for mname in ("marge16", "marge0"):
            rows = [p["measures"][div][mname] for p in per_page]
            n_tot = sum(r["n_lines"] for r in rows)
            agg[div][mname] = {
                "a_mean": float(np.mean([r["a"] for r in rows])), "a_min": float(np.min([r["a"] for r in rows])),
                "a_prime_mean": float(np.mean([r["a_prime"] for r in rows])),
                "a_prime_min": float(np.min([r["a_prime"] for r in rows])),
                "b_mean": float(np.mean([r["b"] for r in rows])), "b_min": float(np.min([r["b"] for r in rows])),
                "c_mean": float(np.mean([r["c"] for r in rows])), "c_min": float(np.min([r["c"] for r in rows])),
                "n_zones_total": int(sum(r["n_zones"] for r in rows)),
                "n_lines_total": int(n_tot),
                "frac_lines_in_zone_le_32": float(sum(r["frac_lines_in_zone_le_32"] * r["n_lines"] for r in rows)
                                                  / n_tot),
                "frac_lines_in_zone_le_64": float(sum(r["frac_lines_in_zone_le_64"] * r["n_lines"] for r in rows)
                                                  / n_tot),
                "height_le_32": {"lines": int(sum(r["lines_le_32"] for r in rows)),
                                 "covered": int(sum(r["covered_le_32"] for r in rows))},
                "height_gt_32": {"lines": int(sum(r["lines_gt_32"] for r in rows)),
                                 "covered": int(sum(r["covered_gt_32"] for r in rows))},
            }
        pooled = [c for p in per_page for c in p["measures"][div]["marge16"]["lines_per_zone"]]
        agg[div]["lines_per_zone"] = _dist(pooled)
        agg[div]["lines_per_zone_nonexclusive"] = _dist(
            [c for p in per_page for c in p["measures"][div]["marge16"]["lines_per_zone_nonexclusive"]])
        agg[div]["lines_per_zone_by_height"] = {
            "le_32": _dist([c for p in per_page for c in p["measures"][div]["marge16"]["lines_per_zone_le_32"]]),
            "gt_32": _dist([c for p in per_page for c in p["measures"][div]["marge16"]["lines_per_zone_gt_32"]])}
        cats: dict[str, list[int]] = {}
        for p in per_page:
            for c, counts in p["measures"][div]["marge16"]["lines_per_zone_by_category"].items():
                cats.setdefault(c, []).extend(counts)
        agg[div]["by_category"] = {c: _dist(v) for c, v in sorted(cats.items())}
        sizes = [s for p in per_page for s in p["zone_sizes_scaled"][div]]
        if sizes:
            ws = [s[0] for s in sizes]
            hs = [s[1] for s in sizes]
            agg[div]["zone_size_scaled"] = {"width_median": float(np.median(ws)), "width_max": float(max(ws)),
                                            "height_median": float(np.median(hs)), "height_max": float(max(hs))}
        else:
            agg[div]["zone_size_scaled"] = None
    stats = {d: {"a_mean": agg[d]["marge16"]["a_mean"], "p95": agg[d]["lines_per_zone"]["p95"]} for d in DIVISIONS}
    agg["_stats_for_reading"] = stats
    agg["categories_kept_total"] = _sum_counts([p["categories_kept"] for p in per_page])
    agg["lines_by_category_total"] = _sum_counts([p["lines_by_category"] for p in per_page])
    agg["n_pages"] = len(per_page)
    agg["n_lines_validated_total"] = int(sum(p["n_gt_lines_validated"] for p in per_page))
    agg["timings_s_mean"] = {k: float(np.mean([p["timings_s"][k] for p in per_page]))
                             for k in ("read_and_control", "detector", "measures", "total")} if per_page else None
    dx = [p["scale_gap_px"]["dx"] for p in per_page]
    dy = [p["scale_gap_px"]["dy"] for p in per_page]
    agg["scale_gap_px"] = {"max_dx": max(dx) if dx else None, "max_dy": max(dy) if dy else None,
                           "pages_over_0_5_px": int(sum(1 for a, b in zip(dx, dy) if max(a, b) > 0.5))}
    return agg


def _sum_counts(dicts) -> dict:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def status_of(agg: dict, cfg: dict) -> tuple[str, str]:
    z_gt = agg["Z-GT"]["marge16"]["a_mean"]
    if z_gt < cfg["reading"]["z_gt_control_min_mean"]:
        return "MEASURE_CHECK_FAILED", f"Z-GT a_mean {z_gt:.4f} < {cfg['reading']['z_gt_control_min_mean']}"
    return "COMPLETE", ""


# ---------- outputs ----------

def check_output_dir(out, root, smoke: bool) -> None:
    """Never inside the copied input folder; a full run writes only under /kaggle/working."""
    out = Path(out).resolve()
    pages_root = Path(root).resolve()
    if out == pages_root or pages_root in out.parents:
        raise SystemExit(f"refused: output {out} inside the input folder {pages_root}")
    if not smoke and not (out == KAGGLE_OUT or KAGGLE_OUT in out.parents):
        raise SystemExit(f"refused: full run writes only under {KAGGLE_OUT}")


def page_input_sha(root: Path, pid: str) -> str:
    """Sha of the two input files of a page (JSON then PNG)."""
    return sha256_text(sha256(root / "pages" / f"{pid}.json") + sha256(root / "images" / f"{pid}.png"))


def check_reference(root: Path, pages: list[str], reference: dict) -> dict:
    """§11 (17): every page file read is checked against steps.page_cache.sha256 of the E2E-4 v3 report."""
    ref = reference["steps"]["page_cache"]["sha256"]
    checked, bad = {}, []
    for pid in pages:
        for key in (f"pages/{pid}.json", f"images/{pid}.png"):
            got = sha256(root / key)
            checked[key] = got
            if ref.get(key) != got:
                bad.append(key)
    return {"checked": checked, "mismatch": bad}


def _write_outputs(out: Path, cfg: dict, report: dict) -> None:
    """Report, then the manifest: sha of the report, of every measures/*.json and zones/*.json (§13 (5))."""
    (out / "ld0_zone_coverage_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    files = {"ld0_zone_coverage_report.json": sha256(out / "ld0_zone_coverage_report.json")}
    for pid in cfg["pages"]["list"]:
        mp = out / "measures" / f"{pid}.json"
        if mp.exists():
            files[str(mp.relative_to(out))] = sha256(mp)
        for div in DIVISIONS:
            zp = out / "zones" / div / f"{pid}.json"
            if zp.exists():
                files[str(zp.relative_to(out))] = sha256(zp)
    (out / "ld0_manifest.json").write_text(json.dumps({"status": report["status"], "identity": report.get("identity"),
                                                      "files_sha256": files}, indent=1))


def _report_base(status: str, reason: str, cfg: dict, out: Path, per_page, agg, reading_result, errors,
                 identity, smoke: bool, skipped, t_start) -> dict:
    return {
        "schema_version": "ld0_zone_coverage_report_v1", "status": status, "reason": reason,
        "smoke": smoke, "pages": [p["page_id"] for p in per_page], "errors": errors,
        "identity": identity,
        "indicator_caveat": "LD-0 est un INDICATEUR sur pages d'entraînement, pas une preuve que le rappel 0,90 "
                            "est atteignable ; l'affectation des lignes est exclusive (§13 (2)).",
        "classes_table": None if identity is None else identity.get("classes_table"),
        "aggregate": {k: v for k, v in agg.items() if not k.startswith("_")},
        "reading": reading_result,
        "reading_note": ("non lus : MEASURE_CHECK_FAILED" if status == "MEASURE_CHECK_FAILED"
                         else "lecture non faite" if status not in ("COMPLETE", "SMOKE") else ""),
        "per_page": [{k: v for k, v in p.items() if k not in ("zones_native", "zones_scaled")} for p in per_page],
        "pages_resumed_from_disk": skipped,
        "timing_seconds_total": round(time.monotonic() - t_start, 2),
    }


def run(args) -> dict:
    _load_deps()
    cfg = json.loads(CONFIG.read_text())
    check_vendor(cfg)
    pages_all = list(cfg["pages"]["list"])
    assert pages_all == [f"mf_{i:04d}" for i in range(50)], "page list is mf_0000..mf_0049"
    smoke = args.pages is not None
    pages = list(args.pages) if smoke else pages_all
    assert set(pages) <= set(pages_all) and not any(p.startswith(("mf_005")) for p in pages)
    if not smoke and not cfg["license"]["text"]:
        raise SystemExit("refused: license text empty in the config (p6 fills it before the run)")
    root = Path(args.root)
    out = Path(args.out_dir)
    check_output_dir(out, root, smoke)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    ident_launch = launch_identity(cfg)
    ref_path = Path(args.reference_report)
    if not sha256(ref_path).startswith(cfg["pages"]["reference_sha256_prefix"]):
        raise SystemExit("refused: E2E-4 reference report sha prefix differs from the config")
    page_check = check_reference(root, pages, json.loads(ref_path.read_text()))
    if page_check["mismatch"]:
        raise SystemExit(f"refused: page files differ from the E2E-4 reference: {page_check['mismatch']}")
    proc, model, size = load_detector(cfg["detector"], Path(args.cache_dir))
    classes_table = {str(k): {"label": v, "is_read": bool(is_read(v))} for k, v in model.config.id2label.items()}
    identity = dict(ident_launch, classes_table=classes_table)

    def det_fn(pid, page):
        return detect(proc, model, Image.open(root / "images" / f"{pid}.png").convert("RGB"),
                      cfg["detector"]["floor"])

    cap_s = cfg["budget"]["cap_minutes"] * 60
    t_loop = time.monotonic()          # §12 (5): the 90-minute cap is measured from the start of the page loop
    per_page, errors, skipped = [], [], []
    status, reason = "COMPLETE", ""
    for pid in pages:
        if time.monotonic() - t_loop > cap_s:
            status, reason = "INCOMPLETE", "90-minute cap reached"
            break
        page_ident = dict(ident_launch, input_sha256=page_input_sha(root, pid))
        mpath = out / "measures" / f"{pid}.json"
        if mpath.exists():
            stored = json.loads(mpath.read_text())
            if stored.get("identity") != page_ident:      # §13 (4): another launch: refuse, do not reuse
                raise ResumeIncompatible(f"reprise incompatible : {pid} a été mesuré par un autre lancement")
            res = stored["result"]
            skipped.append(pid)
        else:
            try:
                res = process_page(pid, root, det_fn, cfg)
            except Exception as exc:  # noqa: BLE001 — §12 (4): one page in error is recorded, the others go on
                errors.append({"page_id": pid, "error": f"{type(exc).__name__}: {exc}",
                               "trace": traceback.format_exc()[-2000:]})
                continue
            mpath.parent.mkdir(parents=True, exist_ok=True)
            mpath.write_text(json.dumps({"page_id": pid, "identity": page_ident, "result": res}))
        per_page.append(res)
        for div in DIVISIONS:
            zdir = out / "zones" / div
            zdir.mkdir(parents=True, exist_ok=True)
            (zdir / f"{pid}.json").write_text(json.dumps(
                {"page_id": pid, "division": div, "margin_px": cfg["measure"]["margin_px"],
                 "assignment": "exclusive: best zone, smallest index on ties (§13 (2))",
                 "zones_native": res["zones_native"][div], "zones_scaled": res["zones_scaled"][div],
                 "factor": res["factor"], "W_s": res["W_s"], "H_s": res["H_s"]}, indent=1))
    if errors and status == "COMPLETE":
        status, reason = "FAILED_PAGES", f"{len(errors)} page(s) in error ; no reading"
    if not per_page and status == "COMPLETE":
        status, reason = "FAILED", "no page completed"
    agg = aggregate(per_page, cfg) if per_page else {}
    agg_reading = None
    if status == "COMPLETE" and agg:
        st, why = status_of(agg, cfg)
        if st != "COMPLETE":
            status, reason = st, why
        else:
            agg_reading = reading(agg["_stats_for_reading"], cfg["reading"])
    if status == "COMPLETE" and smoke:
        status = "SMOKE"
    report = _report_base(status, reason, cfg, out, per_page, agg, agg_reading, errors, identity, smoke, skipped,
                          t_start)
    report.update({
        "detector": {"repo": cfg["detector"]["repo"], "revision": cfg["detector"]["revision"],
                     "model_safetensors_sha256": cfg["detector"]["files"]["model.safetensors"]["sha256"],
                     "effective_input_size": str(size), "floor": cfg["detector"]["floor"],
                     "classes": len(model.config.id2label),
                     "note": "D-219d tuned on Gallica and NewsEye pages, not Mille Feuilles"},
        "license": cfg["license"]["text"], "versions": _versions(),
        "kit_sha256": {rel: sha256(ROOT / rel) for rel in KIT_FILES},
        "vendor_sha256": cfg["vendor"]["files"], "page_sha_checked": page_check["checked"],
        "indicators_not_evaluation": "training pages mf_0000-0049 ; indicators only"})
    _write_outputs(out, cfg, report)
    return report


def aggregate_only(args) -> dict:
    """§13 (6): after the notebook's 110-minute hard stop. Reads measures/ only, checks the launch identity, writes an
    INCOMPLETE report. No page read, no reading, no decision."""
    _load_deps()
    cfg = json.loads(CONFIG.read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ident_launch = launch_identity(cfg)
    per_page, done = [], []
    for pid in cfg["pages"]["list"]:
        mp = out / "measures" / f"{pid}.json"
        if not mp.exists():
            continue
        stored = json.loads(mp.read_text())
        stored_launch = {k: stored.get("identity", {}).get(k) for k in ident_launch}
        if stored_launch != ident_launch:
            raise ResumeIncompatible(f"reprise incompatible : {pid} a été mesuré par un autre lancement")
        per_page.append(stored["result"])
        done.append(pid)
    agg = aggregate(per_page, cfg) if per_page else {}
    report = _report_base("INCOMPLETE", "notebook hard stop 110 min : aggregate-only, no reading, no decision",
                          cfg, out, per_page, agg, None, [], dict(ident_launch, classes_table=None), False, done,
                          time.monotonic())
    report["pages"] = done
    _write_outputs(out, cfg, report)
    return report


def _versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__, "PIL": Image.__version__}
    for mod in ("torch", "torchvision", "cv2", "transformers", "huggingface_hub"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 (absent outside the Kaggle/venv image)
            out[mod] = None
    return out


def build_parser():
    p = argparse.ArgumentParser(description="LD-0 : couverture des lignes par les zones PP-DocLayoutV3 (mf_0000-0049)")
    p.add_argument("--confirmed-go", action="store_true", required=True)
    p.add_argument("--root", default=None, help="folder with pages/ and images/ copied by the notebook")
    p.add_argument("--cache-dir", default=None, help="HF download cache (scratch or /kaggle/working)")
    p.add_argument("--reference-report", default=None, help="E2E-4 v3 report (page sha reference)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pages", default=None, help="smoke only: explicit comma list among mf_0000..mf_0049")
    p.add_argument("--aggregate-only", action="store_true", help="§13 (6): after the hard stop; reads measures/ only")
    p.add_argument("--download-only", action="store_true", help="§14 (10): the 'download' step; no model load")
    return p


def write_failed_report(args, exc: BaseException, trace: str) -> None:
    """§12 (4): an uncaught exception still writes the report (status FAILED), keeping the measures already written."""
    out = Path(args.out_dir)
    cfg = json.loads(CONFIG.read_text())
    out.mkdir(parents=True, exist_ok=True)
    done = [pid for pid in cfg["pages"]["list"] if (out / "measures" / f"{pid}.json").exists()]
    report = {"schema_version": "ld0_zone_coverage_report_v1", "status": "FAILED",
              "reason": f"{type(exc).__name__}: {exc}", "traceback": trace[-4000:],
              "pages_measured_kept": done, "reading": None, "reading_note": "lecture non faite"}
    _write_outputs(out, cfg, report)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.download_only:
        try:
            cfg = json.loads(CONFIG.read_text())
            files = download_detector(cfg["detector"], Path(args.cache_dir))
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "download_report.json").write_text(json.dumps(
                {"status": "OK", "files_sha256": {k: sha256(v) for k, v in files.items()}}, indent=2))
            print(json.dumps({"status": "OK", "files": sorted(files)}), flush=True)
            return 0
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 1
    if args.aggregate_only:
        try:
            report = aggregate_only(args)
        except Exception as exc:  # noqa: BLE001
            trace = traceback.format_exc()
            print("FAILED", flush=True)
            traceback.print_exc()
            write_failed_report(args, exc, trace)
            return 1
        print(json.dumps({"status": report["status"], "pages": len(report["pages"])}, indent=2), flush=True)
        return 0
    if args.pages is not None:
        args.pages = [x.strip() for x in args.pages.split(",") if x.strip()]
    if not (args.root and args.cache_dir and args.reference_report):
        print("refused: --root, --cache-dir and --reference-report are required for a run", flush=True)
        return 2
    try:
        report = run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        trace = traceback.format_exc()
        print("FAILED", flush=True)
        traceback.print_exc()
        try:
            write_failed_report(args, exc, trace)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return 1
    print(json.dumps({"status": report["status"], "timing_seconds_total": report["timing_seconds_total"],
                      "pages": len(report["pages"]), "errors": len(report["errors"])}, indent=2), flush=True)
    return 0



# ---------- pipeline of the notebook (§14): timeouts per step, process groups, logs, heartbeat, exit record ----------

def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _kill_group(proc: subprocess.Popen, grace_s: float) -> None:
    """§14 (2): the whole group (the child and its descendants) gets SIGTERM, then SIGKILL after grace_s."""
    pgid = proc.pid                       # start_new_session=True: the group id is the child pid
    _signal_group(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + grace_s
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _group_alive(pgid):
        _signal_group(pgid, signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _tail(path: Path, n: int = 200) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_step(name: str, argv: list, timeout_s: float, logs_dir: Path, cwd=None, env=None,
             kill_grace_s: float = 10.0) -> dict:
    """§14 (1)(2)(3): one subprocess in its own group, stdout and stderr in files, no pipe, no subprocess.run timeout."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path, err_path = logs_dir / f"{name}.stdout", logs_dir / f"{name}.stderr"
    t0 = time.monotonic()
    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        proc = subprocess.Popen([str(a) for a in argv], cwd=cwd, env=env, stdout=out_f, stderr=err_f,
                                start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc, kill_grace_s)
        else:
            _signal_group(proc.pid, signal.SIGKILL)       # leftover descendants of a finished step
    rc = "TIMEOUT" if timed_out else proc.returncode
    return {"step": name, "rc": rc, "seconds": round(time.monotonic() - t0, 3),
            "stdout": str(out_path), "stderr": str(err_path), "stderr_tail": _tail(err_path)}


class Heartbeat:
    """§14 (4): a daemon thread writes heartbeat.jsonl every interval seconds."""

    def __init__(self, out: Path, interval_s: float = 60.0):
        self.out = Path(out)
        self.interval = interval_s
        self.cfg = json.loads(CONFIG.read_text())
        self.step = "setup"
        self.step_t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    def set_step(self, name: str) -> None:
        self.step, self.step_t0 = name, time.monotonic()

    def beat(self) -> dict:
        n = sum(1 for pid in self.cfg["pages"]["list"] if (self.out / "measures" / f"{pid}.json").exists())
        line = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "step": self.step, "seconds_in_step": round(time.monotonic() - self.step_t0, 1),
                "measures_json": n}
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "heartbeat.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        return line

    def _loop(self) -> None:
        """§14 (11): the loop catches and logs its own errors; it never ends the pipeline."""
        while not self._stop.wait(self.interval):
            try:
                self.beat()
            except Exception as exc:  # noqa: BLE001
                try:
                    with open(self.out / "heartbeat_errors.txt", "a") as fh:
                        fh.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {exc!r}\n")
                except OSError:
                    pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ld0-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def execute_pipeline(out: Path, steps: list, heartbeat_interval_s: float = 60.0, cleanup=None) -> dict:
    """Runs the steps in order. A step is {"name", "fn"} (in process) or {"name", "argv", "timeout_s", "cwd", "env",
    "retry", "before_retry", "after_timeout"} (subprocess). argv/env may be callables evaluated at run time.
    §14 (10): a retry (once) runs after before_retry (partial cache removed); a TIMEOUT or a failure stops the list,
    except that the step's "after_timeout" step runs after a TIMEOUT. The cause goes into the exit record.
    §14 (11): faulthandler dumps the stacks after 600 s in process steps (re-armed per step, cancelled around
    subprocess steps); the heartbeat is built inside the try; the exit record is written in the finally."""
    out = Path(out)
    rec = {"status": "DONE", "step": None, "exit_code": 0, "durations_s": {}, "attempts": {}, "cause": None,
           "exception": None}
    logs = out / "logs"
    hb = None
    fh = None
    current = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        fh = open(out / "ld0_faulthandler.txt", "a")
        faulthandler.dump_traceback_later(600, exit=True, file=fh)
        hb = Heartbeat(out, heartbeat_interval_s)
        hb.start()
        for st in steps:
            current = st["name"]
            hb.set_step(current)
            if "fn" in st:
                faulthandler.dump_traceback_later(600, exit=True, file=fh)
                t0 = time.monotonic()
                st["fn"]()
                rec["durations_s"][current] = round(time.monotonic() - t0, 3)
                continue
            faulthandler.cancel_dump_traceback_later()
            tries = 1 + int(st.get("retry", 0))
            r = None
            for attempt in range(1, tries + 1):
                if attempt > 1 and st.get("before_retry"):
                    st["before_retry"]()
                argv = st["argv"]() if callable(st["argv"]) else st["argv"]
                env = st["env"]() if callable(st.get("env")) else st.get("env")
                r = run_step(current, argv, st["timeout_s"], logs, cwd=st.get("cwd"), env=env)
                rec["attempts"][current] = attempt
                rec["durations_s"][current] = r["seconds"]
                print(f"--- {current} attempt {attempt}: rc={r['rc']} in {r['seconds']} s ; "
                      f"last lines of its stderr:", flush=True)
                print(r["stderr_tail"], flush=True)
                if r["rc"] == 0:
                    break
            if r["rc"] == "TIMEOUT":
                rec.update(status="TIMEOUT", step=current, exit_code="TIMEOUT",
                           cause={"step": current, "rc": "TIMEOUT", "stderr_tail": r["stderr_tail"]})
                if st.get("after_timeout"):
                    nxt = st["after_timeout"]
                    hb.set_step(nxt["name"])
                    nargv = nxt["argv"]() if callable(nxt["argv"]) else nxt["argv"]
                    r2 = run_step(nxt["name"], nargv, nxt["timeout_s"], logs, cwd=nxt.get("cwd"),
                                  env=nxt["env"]() if callable(nxt.get("env")) else nxt.get("env"))
                    rec["durations_s"][nxt["name"]] = r2["seconds"]
                    print(f"--- {nxt['name']}: rc={r2['rc']} ; last lines of its stderr:", flush=True)
                    print(r2["stderr_tail"], flush=True)
                break
            if r["rc"] != 0:
                rec.update(status="FAILED", step=current, exit_code=r["rc"],
                           cause={"step": current, "rc": r["rc"], "stderr_tail": r["stderr_tail"]})
                break
    except BaseException as exc:  # noqa: BLE001 — the exit record must be written whatever happened
        rec.update(status="FAILED", step=current, exit_code="EXCEPTION",
                   exception=f"{type(exc).__name__}: {exc}",
                   cause={"step": current, "exception": f"{type(exc).__name__}: {exc}"})
        print(traceback.format_exc(), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        if hb is not None:
            hb.stop()
        if cleanup is not None:
            try:
                cleanup(rec)
            except Exception as exc:  # noqa: BLE001
                rec["cleanup_exception"] = f"{type(exc).__name__}: {exc}"
        try:
            rec["timeouts_s_from_config"] = json.loads(CONFIG.read_text()).get("timeouts_s")
            out.mkdir(parents=True, exist_ok=True)
            (out / "ld0_run_exit.json").write_text(json.dumps(rec, indent=2))
        finally:
            if fh is not None:
                fh.close()
    return rec

if __name__ == "__main__":
    sys.exit(main())
