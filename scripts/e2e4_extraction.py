"""E2E-4 — recoller les fragments : extraction des lignes avec fusion (CPU, sans réentraînement).

Lot : docs/END_TO_END_E2E4_LOT.txt (p6, 17/10/2026). Fonctions reprises de e2e3_fp_classes.py par import ;
aucune modification de line_geometry, des bras S ni de la config v2 d'extraction (a464e1b6).

Ordre d'exécution (run) :
  1. garde : pas de processus e2e3/e2e4/mille-feuilles vivant ; --confirmed-go obligatoire ;
  2. sources : 51 fichiers du clone contre le manifeste du bundle v5 ;
  3. pages : /<tmp>/mf déjà générées (60 JSON + 60 PNG, sha par fichier) : AUCUNE génération ici ;
  4. checkpoints B3, B4, S2 (préfixes de sha) ;
  5. réglage (mf_0000..0019, B3) : écarts intra-ligne des fragments, G (90e centile, plafond h_cible),
     référence des espaces e (boîtes de mots), lecture H-ESP ; G est écrit AVANT l'évaluation ;
  6. évaluation (mf_0050..0059) : B3 (contrôle de fidélité sur l'extraction d'origine, puis après
     fusion), S2 (après fusion, contrôle R6), B4 (descriptif) ;
  7. lectures R5 et R6, seulement si le contrôle de fidélité passe et si le plafond n'est pas dépassé.
Le plafond d'inférence (150 min, config) démarre après l'étape 3.
Rien n'est écrit sous --tmp (entrée en lecture seule sur Kaggle) : G fixé et fichiers de travail vont
dans le dossier du rapport (amendement 2, §8).
Options --setting-ids a,b / --eval-ids c : pages explicites pour l'essai de fumée (remplacent les listes de la
config) ; le statut est alors SMOKE, sans lecture ni contrôle de rejeu, et seules ces pages sont ouvertes.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import e2e3_fp_classes as E3  # noqa: E402  (reuse by import: checks, loading, classes, replay gate)

CONFIG = ROOT / "config/e2e4_extraction.json"
G_FILE = "e2e4_G_fixed.json"


# ---------- pure geometry: directional projection, intra-line gaps, fusion ----------

def orient(poly) -> list:
    """Points left to right (the reading direction of the lines)."""
    pts = [(float(x), float(y)) for x, y in poly]
    if len(pts) >= 2 and pts[-1][0] < pts[0][0]:
        pts.reverse()
    return pts


def gap(a, b) -> float:
    """Horizontal gap of amendment 1 (§7 E2E-4): x(start of B) - x(end of A), in px of the scaled page.
    Positive: a hole. Zero or negative: overlap or touching, never fused and never measured as a hole.
    The same function serves the measurement (§2) and the fusion (§3)."""
    A, B = orient(a), orient(b)
    return B[0][0] - A[-1][0]


def line_gaps(fragments) -> list:
    """Holes between consecutive fragments of one GT line, sorted by their start x (only gap > 0)."""
    ordered = sorted((orient(f) for f in fragments), key=lambda f: (f[0][0], f[0][1]))
    out = []
    for prev, nxt in zip(ordered, ordered[1:]):
        g = gap(prev, nxt)
        if g > 0:
            out.append(g)
    return out


def percentile(values, q: float) -> float:
    """Linear-interpolation percentile (numpy default)."""
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def compute_G(gaps, h_target: float, q: float = 90.0) -> dict:
    """§3 of the lot: G = 90th percentile of the intra-line gaps, capped at h_target. No gaps -> G = 0."""
    if not gaps:
        return {"G": 0.0, "p90": None, "n_gaps": 0, "cap": h_target, "rule": "no gaps: G = 0 (no fusion)"}
    p = percentile(gaps, q)
    return {"G": min(p, h_target), "p90": p, "n_gaps": len(gaps), "cap": h_target,
            "rule": f"min(p{q:g} of intra-line gaps, h_target)"}


def _end_angle(poly):
    for (ax, ay), (bx, by) in reversed(list(zip(poly, poly[1:]))):
        if (bx - ax) or (by - ay):
            return math.atan2(by - ay, bx - ax)
    return None


def _start_angle(poly):
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        if (bx - ax) or (by - ay):
            return math.atan2(by - ay, bx - ax)
    return None


def fuse_pair(a, b, G: float, h_target: float, angle_max_deg: float = 10.0, vertical_factor: float = 0.5):
    """Horizontal gap g if A then B may be fused, else None. The four conditions of §3."""
    if G <= 0:
        return None
    A, B = orient(a), orient(b)
    g = gap(A, B)                                     # (i)-(ii) the same gap() as the measurement
    if not (0 < g <= G):
        return None
    if abs(A[-1][1] - B[0][1]) > vertical_factor * h_target:   # (iii) vertical offset <= 0.5 h_target
        return None
    ta, tb = _end_angle(A), _start_angle(B)
    if ta is None or tb is None:
        return None
    diff = (tb - ta + math.pi) % (2 * math.pi) - math.pi
    if abs(math.degrees(diff)) > angle_max_deg:       # (iv) angle difference <= 10 degrees
        return None
    return g


def fuse_chain(polys, G: float, h_target: float):
    """Chain fusion, deterministic: sort by (start y, start x, extraction index); each chain grows with the
    candidate of smallest gap (ties: same sort key). G <= 0 returns the extraction unchanged."""
    if G <= 0:
        return [[list(p) for p in orient(poly)] for poly in polys], [[i] for i in range(len(polys))]
    items = [(i, orient(p)) for i, p in enumerate(polys)]
    items.sort(key=lambda t: (t[1][0][1], t[1][0][0], t[0]))
    used = [False] * len(items)
    out, members_out = [], []
    for pos in range(len(items)):
        if used[pos]:
            continue
        used[pos] = True
        cur = list(items[pos][1])
        members = [items[pos][0]]
        while True:
            best = None
            for q in range(len(items)):
                if used[q]:
                    continue
                g = fuse_pair(cur, items[q][1], G, h_target)
                if g is None:
                    continue
                key = (g, items[q][1][0][1], items[q][1][0][0], items[q][0])
                if best is None or key < best[0]:
                    best = (key, q)
            if best is None:
                break
            q = best[1]
            used[q] = True
            cur = cur + list(items[q][1])
            members.append(items[q][0])
        out.append([[x, y] for x, y in cur])
        members_out.append(members)
    return out, members_out


# ---------- H-ESP and the readings ----------

def word_gaps(words_by_line: dict) -> list:
    """Horizontal gaps between consecutive GT word boxes of each line (scaled px), with gap().
    words_by_line: id -> [(minx, maxx)]; a box is the segment from minx to maxx."""
    out = []
    for spans in words_by_line.values():
        boxes = sorted(([(lo, 0.0), (hi, 0.0)] for lo, hi in spans), key=lambda b: b[0][0])
        for prev, nxt in zip(boxes, boxes[1:]):
            g = gap(prev, nxt)
            if g > 0:
                out.append(g)
    return out


def hesp_reading(gaps_px, e_px: float, interval=(0.5, 2.0), min_fraction: float = 0.5) -> dict:
    """H-ESP (amendment 1): ratio r = g / e, g the horizontal hole in px (gap()), e the median of the GT word
    gaps in px of the same scale. Supported if the median of r is in [0.5, 2] AND at least 50 % of the r are
    in that interval. The histogram in units of e is reported in all cases."""
    if not gaps_px:
        return {"supported": False, "text": "pas d'écart mesurable", "histogram_units_e": [0] * 17, "n": 0,
                "e_px": e_px}
    ratios = [g / e_px for g in gaps_px]
    lo, hi = interval
    bins = [0] * 17  # 0.25 e wide from 0 to 4 e, then >= 4 e
    for r in ratios:
        bins[min(int(r / 0.25), 16)] += 1
    med = E3.median(ratios)
    frac = sum(lo <= r <= hi for r in ratios) / len(ratios)
    ok = lo <= med <= hi and frac >= min_fraction
    text = ("H-ESP soutenue : le masque de ligne se coupe aux espaces entre mots" if ok
            else "coupures non expliquées par les espaces")
    return {"supported": ok, "text": text, "median_units_e": med, "fraction_in_interval": frac,
            "histogram_units_e": bins, "n": len(ratios), "e_px": e_px}


def r5_reading(precision: float, recall: float, recall_ref: float = 0.5768, margin: float = 0.02,
               precision_min: float = 0.5) -> dict:
    """R5 (B3, après fusion): sufficient if precision >= 0.5 AND recall >= 0.5768 - 0.02."""
    suff = precision >= precision_min and recall >= recall_ref - margin - 1e-12
    text = ("extraction suffisante : pré-enregistrer une nouvelle calibration GPU (lot séparé, avis de pK)"
            if suff else
            "extraction non suffisante : remède pré-écrit = poids λ = 1/6 sur le gradient CTC entrant dans enc_det "
            "(lot GPU séparé, non lancé ici)")
    return {"sufficient": suff, "text": text, "precision": precision, "recall": recall}


def r6_reading(precision: float, recall: float, ref_precision: float, ref_recall: float,
               margin: float = 0.02) -> dict:
    """R6 (S2, après fusion): neither precision nor recall may drop by more than 0.02 from the original."""
    drop_p, drop_r = ref_precision - precision, ref_recall - recall
    degraded = drop_p > margin + 1e-12 or drop_r > margin + 1e-12
    text = ("fusion trop agressive : R5 n'est pas lue comme une réussite" if degraded
            else "S2 non dégradé par la fusion")
    return {"degraded": degraded, "text": text, "drop_precision": drop_p, "drop_recall": drop_r}


# ---------- inference (torch) ----------

def infer(model, image, extraction: dict):
    """Extraction of the fragments and the encoder features (same steps as predict_page)."""
    import torch
    from babaorum.experimental.stage_s_targets import extract_polylines
    with torch.no_grad():
        features = model.enc_det(torch.as_tensor(image)[None, None])
        prob = torch.sigmoid(model.head_d(features)[0])[0, 0].numpy()
        extracted = [p for p in extract_polylines(prob, stride=extraction["stride"],
                                                  threshold=extraction["threshold"],
                                                  min_cells=extraction["min_cells"]) if len(p) >= 2]
    return extracted, features


def refine(model, features, polys, cfg_model: dict) -> list:
    import torch
    if not polys:
        return []
    from babaorum.experimental.stage_s_model import refine_lines
    with torch.no_grad():
        out = refine_lines(features, polys, model.head_g, head_d=model.head_d, K=cfg_model["K"])["baseline"].tolist()
    assert len(out) == len(polys), ("refinement is not one-to-one with the fragments", len(out), len(polys))
    return out


# ---------- scores of one page ----------

def page_scores(gts, refined) -> dict:
    """gts: (id, baseline, h) with h > 0. Matching line_geometry.v1 on the refined baselines."""
    from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry
    gt_objs = [GroundTruthLine(id=i, baseline=b, height_px=h, baseline_source="ground_truth") for i, b, h in gts]
    preds = [PredictedLine(id=f"p{k:05d}", baseline=b, baseline_source="predicted") for k, b in enumerate(refined)]
    res = evaluate_line_geometry(gt_objs, preds) if preds else None
    recall = res.recall if res else 0.0
    precision = res.precision if (res and res.precision is not None) else 0.0
    dists = [m.normalized_symmetric_distance for m in res.matches] if res else []
    match_map = {m.pred_id: m.gt_id for m in res.matches} if res else {}
    matched_gt = set(match_map.values())
    gts_pos = [(i, b, h) for i, b, h in gts]
    classes = []
    for k, b in enumerate(refined):
        if f"p{k:05d}" not in match_map:
            classes.append(E3.classify_unmatched(b, gts_pos, matched_gt))
    return {"recall": recall, "precision": precision,
            "mean_normalized_distance": (sum(dists) / len(dists)) if dists else None,
            "match_map": match_map, "classes": E3.count_classes(classes), "n_refined": len(refined)}


def fragment_groups(refined, extracted, gts, match_map: dict) -> dict:
    """§2: fragments F and the predictions matched to a GT line, grouped by reference line.
    Matched predictions join their matched line; unmatched ones join their reference line only if class F."""
    matched_gt = set(match_map.values())
    groups = {}
    for k, b in enumerate(refined):
        pid = f"p{k:05d}"
        if pid in match_map:
            gid = match_map[pid]
        else:
            if E3.classify_unmatched(b, gts, matched_gt) != "F":
                continue
            gid = E3.reference_line(b, gts)[0]
        groups.setdefault(gid, []).append(extracted[k])
    return groups


# ---------- run ----------

def _stop(report: dict, status: str, why: str) -> dict:
    report["status"] = status
    report["stopped_because"] = why
    return report


@contextlib.contextmanager
def stage(report: dict, name: str):
    """Adds the wall-clock seconds of the block to report['timings_s'][name] (sums over pages)."""
    t0 = time.monotonic()
    try:
        yield
    finally:
        timings = report.setdefault("timings_s", {})
        timings[name] = round(timings.get(name, 0.0) + time.monotonic() - t0, 3)


def write_G_fixed(report_path: Path, payload: dict, tmp: Path) -> Path:
    """G fixed, written next to the report, never under --tmp (read-only input on Kaggle)."""
    out = Path(report_path).parent / G_FILE
    if Path(tmp).resolve() in out.resolve().parents:
        raise SystemExit("refused: G_fixed must not be written under --tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def live_e2e4_processes(ps_text: str, own_pid: int) -> list:
    """E3.live_processes matches the e2e3 script name only. The e2e4 command lines are mapped onto that name
    first, so the same ancestor-chain exclusion applies to e2e4 runs too."""
    mapped = ps_text.replace("scripts/e2e4_extraction.py", "scripts/e2e3_fp_classes.py")
    return E3.live_processes(mapped, own_pid)


def run_all(args, report: dict | None = None) -> dict:
    cfg = json.loads(CONFIG.read_text())
    report = {} if report is None else report
    report.update({"schema_version": "e2e4_extraction_report_v1", "status": None,
                   "steps": {}, "provenance": {"script_sha256": E3.sha256(Path(__file__).resolve()),
                                               "config_sha256": E3.sha256(CONFIG),
                                               "tests_sha256": E3.sha256(ROOT / "tests/test_e2e4_extraction.py")}})
    root = Path(args.tmp) / "mf"
    cap_min = cfg["budget"]["cap_minutes"]
    setting_ids = getattr(args, "setting_ids", None)
    eval_ids_arg = getattr(args, "eval_ids", None)
    subset = setting_ids is not None or eval_ids_arg is not None
    set_ids = setting_ids if setting_ids is not None else cfg["pages"]["setting"]
    eval_ids = eval_ids_arg if eval_ids_arg is not None else cfg["pages"]["eval"]
    if subset:
        report["subset"] = {"setting_pages": list(set_ids), "eval_pages": list(eval_ids),
                            "note": "smoke test: no reading, no replay gate, not the E2E-4 result"}

    # 1. sources
    manifest = json.loads(zipfile.ZipFile(args.bundle_zip).read("bundle_manifest.json"))
    report["bundle_sha256"] = E3.sha256(args.bundle_zip)
    check = E3.step_check(args.clone, manifest)
    report["steps"]["check"] = check
    if check["status"] != "OK":
        return _stop(report, "MISMATCH", "source files differ from the bundle v5 manifest")

    # 2. pages from the E2E-3 run 3 output: no generation here, nothing written under --tmp
    if subset:
        # smoke test: count the 60+60 files, hash only the pages that are used (no other page is opened)
        used = list(dict.fromkeys(set_ids + eval_ids + cfg["pages"]["alphabet"]))
        cache = {"status": "ACCEPTED", "subset_hashed": used,
                 "sha256": {f"pages/{pid}.json": E3.sha256(root / "pages" / f"{pid}.json") for pid in used},
                 "counts": [len(list((root / "pages").glob("*.json"))), len(list((root / "images").glob("*.png")))]}
        if cache["counts"] != [60, 60]:
            cache = {"status": "FAILED", "reason": f"cache incomplete: {cache['counts']}"}
    else:
        cache = E3.page_cache_status(root)
    report["steps"]["page_cache"] = cache
    if cache["status"] != "ACCEPTED":
        return _stop(report, "FAILED", cache["reason"])

    # 3. checkpoints
    ckpts = {}
    for arm in ("B3", "B4", "S2"):
        info = cfg["source_checkpoints"][arm]
        path = Path(args.calibration) / info["checkpoint"]
        if not E3._check_prefix(path, info["sha256"]):
            return _stop(report, "MISMATCH", f"checkpoint {arm} sha256 does not match the pre-registered prefix")
        ckpts[arm] = path

    # the inference cap starts here (after the page-cache step)
    started = time.monotonic()

    def over_cap():
        return E3.guard_exceeded(time.monotonic() - started, cap_min)

    cap_msg = f"{cap_min:g}-minute inference cap reached"

    # pages: alphabet as at E2E-1 (train pages 0..49), settings pages and evaluation pages
    from train_stage_s import load_scaled_page
    target = cfg["scale"]["target_median_height_px"]
    with stage(report, "pages_alphabet_json"):
        alpha_pages = [json.loads((root / "pages" / f"{pid}.json").read_text()) for pid in cfg["pages"]["alphabet"]]
        alphabet = E3.alphabet_from_json(alpha_pages)
    report["alphabet_size"] = len(alphabet)

    def load_set(pids):
        out = {}
        for pid in pids:
            pg = load_scaled_page(root, pid, target)
            raw = json.loads((root / "pages" / f"{pid}.json").read_text())
            f = pg["factor"]
            words = {}
            for w in raw["words"]:
                xs = [x * f for x, _ in w["polygon"]]
                words.setdefault(w["line_id"], []).append((min(xs), max(xs)))
            out[pid] = {"image": pg["image"], "factor": f,
                        "gt": [(ln["id"], ln["baseline"], ln["normal_negative_px"] + ln["normal_positive_px"])
                               for ln in pg["lines"]],
                        "words": words}
        return out

    with stage(report, "pages_load_setting"):
        setting = load_set(set_ids)
    with stage(report, "pages_load_eval"):
        evaluation = load_set(eval_ids)
    h_target = target

    # 4. models
    with stage(report, "models_load"):
        models = {arm: E3.load_arm(arm, ckpts[arm], len(alphabet)) for arm in ("B3", "S2", "B4")}

    # 5. setting pass on B3 (train pages only)
    gaps_intra, gaps_word, heights = [], [], []
    for pid, page in setting.items():
        if over_cap():
            return _stop(report, "INCOMPLETE", f"{cap_msg} during the setting pass")
        gts = [(i, b, h) for i, b, h in page["gt"] if h > 0]
        with stage(report, "setting_infer"):
            extracted, features = infer(models["B3"], page["image"], cfg["extraction"])
        with stage(report, "setting_refine"):
            refined = refine(models["B3"], features, extracted, cfg["model"])
        with stage(report, "setting_scores_and_groups"):
            sc = page_scores(gts, refined)
            groups = fragment_groups(refined, extracted, gts, sc["match_map"])
        with stage(report, "setting_gaps"):
            for frags in groups.values():
                gaps_intra.extend(line_gaps(frags))
            gaps_word.extend(word_gaps(page["words"]))
        heights.extend(h for _, _, h in gts)
    report["steps"]["setting_intra_line_gaps_px"] = len(gaps_intra)

    # reference e for H-ESP: median of GT word gaps (scaled px); fallback 0.3 h if no word box at all
    if gaps_word:
        e_px, e_mode = E3.median(gaps_word), "word boxes (median of GT word gaps, scaled px)"
    else:
        e_px, e_mode = 0.3 * E3.median(heights), "fallback 0.3 h (median GT height)"
    report["steps"]["hesp"] = hesp_reading(gaps_intra, e_px, tuple(cfg["hesp"]["interval"]),
                                           cfg["hesp"]["min_fraction"])
    report["steps"]["hesp"]["e_mode"] = e_mode

    # G is fixed from the setting pages, written next to the report BEFORE any evaluation
    G = compute_G(gaps_intra, h_target, cfg["fusion"]["G_percentile"])
    G["e_px"] = e_px
    G["G_units_e"] = G["G"] / e_px
    report["steps"]["G"] = G
    G_out = write_G_fixed(args.report, {"G": G, "hesp": report["steps"]["hesp"]}, Path(args.tmp))
    report["steps"]["G_file"] = G_out.name

    # 6. evaluation: original (before) and fused (after), per arm
    def eval_arm(arm):
        rows = {"before": [], "after": []}
        for pid, page in evaluation.items():
            if over_cap():
                return None, cap_msg + " during evaluation"
            gts = [(i, b, h) for i, b, h in page["gt"] if h > 0]
            with stage(report, f"eval_{arm}_infer"):
                extracted, features = infer(models[arm], page["image"], cfg["extraction"])
            with stage(report, f"eval_{arm}_refine_before"):
                ref_b = refine(models[arm], features, extracted, cfg["model"])
            with stage(report, f"eval_{arm}_scores_before"):
                sc_b = page_scores(gts, ref_b)
            with stage(report, f"eval_{arm}_fuse"):
                fused, members = fuse_chain(extracted, G["G"], h_target)
            with stage(report, f"eval_{arm}_refine_after"):
                ref_a = refine(models[arm], features, fused, cfg["model"])
            with stage(report, f"eval_{arm}_scores_after"):
                sc_a = page_scores(gts, ref_a)
            rows["before"].append({"page_id": pid, "n_extracted": len(extracted), **{k: sc_b[k] for k in
                                   ("recall", "precision", "mean_normalized_distance", "classes")}})
            rows["after"].append({"page_id": pid, "n_extracted": len(extracted), "n_fused": len(fused),
                                  "fusions": sum(len(m) - 1 for m in members),
                                  **{k: sc_a[k] for k in ("recall", "precision", "mean_normalized_distance", "classes")}})
        return rows, None

    def summarise(rows):
        n = len(rows)
        d = [r["mean_normalized_distance"] for r in rows if r["mean_normalized_distance"] is not None]
        counts = {"D": 0, "F": 0, "H": 0, "R": 0}
        for r in rows:
            for k in counts:
                counts[k] += r["classes"][k]
        return {"recall_mean": sum(r["recall"] for r in rows) / n,
                "precision_mean": sum(r["precision"] for r in rows) / n,
                "mean_normalized_distance": (sum(d) / len(d)) if d else None,
                "n_extracted_total": sum(r["n_extracted"] for r in rows), "classes_after_or_before": counts}

    report["steps"]["eval"] = {}
    b3, why = eval_arm("B3")
    if b3 is None:
        return _stop(report, "INCOMPLETE", why or "")
    report["steps"]["eval"]["B3"] = {"before": summarise(b3["before"]), "after": summarise(b3["after"]),
                                     "per_page": b3}
    ref = cfg["replay_control"]
    before = report["steps"]["eval"]["B3"]["before"]
    if subset:
        report["steps"]["replay_B3_original"] = {"status": "NOT_APPLICABLE_SUBSET"}
    else:
        replay = E3.replay_status(before["recall_mean"], before["precision_mean"], ref["B3_recall_mean_10_pages"],
                                  ref["B3_precision_mean_10_pages"], ref["tolerance"])
        report["steps"]["replay_B3_original"] = {"recall_mean": before["recall_mean"],
                                                 "precision_mean": before["precision_mean"], "status": replay}
        if replay != "OK":
            return _stop(report, "MISMATCH", "replay control of B3 with the original extraction failed: no reading")

    s2, why = eval_arm("S2")
    if s2 is None:
        return _stop(report, "INCOMPLETE", why or "")
    report["steps"]["eval"]["S2"] = {"before": summarise(s2["before"]), "after": summarise(s2["after"]),
                                     "per_page": s2}
    b4, why = eval_arm("B4")
    if b4 is None:
        return _stop(report, "INCOMPLETE", why or "")
    report["steps"]["eval"]["B4"] = {"before": summarise(b4["before"]), "after": summarise(b4["after"]),
                                     "per_page": b4}
    if subset:
        report["status"] = "SMOKE"
        return report

    # 7. readings (the S2 reference values are those of the E2E-1 report, precision then recall)
    rd = cfg["readings"]
    a_b3 = report["steps"]["eval"]["B3"]["after"]
    a_s2 = report["steps"]["eval"]["S2"]["after"]
    report["readings"] = {
        "R5": r5_reading(a_b3["precision_mean"], a_b3["recall_mean"], rd["R5_recall_ref"], rd["R5_margin"],
                         rd["R5_precision_min"]),
        "R6": r6_reading(a_s2["precision_mean"], a_s2["recall_mean"], rd["R6_ref"]["precision"],
                         rd["R6_ref"]["recall"], rd["R6_margin"]),
        "B4_descriptive": {"before": report["steps"]["eval"]["B4"]["before"],
                           "after": report["steps"]["eval"]["B4"]["after"]},
        "remedy_lambda": rd["remedy_lambda"],
    }
    if report["readings"]["R6"]["degraded"]:
        report["readings"]["R5_reading_valid"] = False
    else:
        report["readings"]["R5_reading_valid"] = True
    report["status"] = "COMPLETE"
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["check", "run"])
    p.add_argument("--clone", type=Path, required=False, help="heritage-synth clone at 23875ab0")
    p.add_argument("--confirmed-go", action="store_true")
    p.add_argument("--tmp", type=Path, required=False, help="folder containing mf/ (E2E-3 run 3 pages)")
    p.add_argument("--bundle-zip", type=Path, required=False, help="bundle v5 zip")
    p.add_argument("--calibration", type=Path, required=False, help="folder with B3/, B4/, S2/ checkpoints")
    p.add_argument("--report", type=Path, required=False, help="output report; never an existing file")
    p.add_argument("--setting-ids", type=lambda s: s.split(","), default=None,
                   help="smoke test only: comma-separated setting page ids (replaces the config list)")
    p.add_argument("--eval-ids", type=lambda s: s.split(","), default=None,
                   help="smoke test only: comma-separated evaluation page ids (replaces the config list)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.step == "run" and not args.confirmed_go:
        raise SystemExit("refused: E2E-4 runs only with --confirmed-go after the GO of w2:p6")
    if args.step == "run" and (args.setting_ids is not None or args.eval_ids is not None) and not (
            args.setting_ids and args.eval_ids):
        raise SystemExit("refused: a smoke test needs both --setting-ids and --eval-ids")
    if args.step == "run":
        for name in ("clone", "tmp", "bundle_zip", "calibration", "report"):
            if getattr(args, name) is None:
                raise SystemExit(f"refused: --{name.replace('_', '-')} is required")
        if args.report.exists():
            raise SystemExit(f"refused: {args.report.name} already exists; a report is never overwritten")
        import subprocess
        ps = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
        alive = live_e2e4_processes(ps, os.getpid())
        if alive:
            raise SystemExit("refused: an e2e3, e2e4 or mille-feuilles process is alive: " + alive[0][:120])
        report = run_all(args)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(json.dumps({"status": report["status"]}))
        return
    print("check: nothing to check without run; see docs/END_TO_END_E2E4_LOT.txt")


if __name__ == "__main__":
    main()
