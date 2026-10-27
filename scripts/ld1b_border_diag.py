"""LD-1B : rappel et précision par distance au bord de zone (docs/END_TO_END_LD1B_LOT.txt). CLASSE A, DESCRIPTIF.

N'importe line_detr et train_line_detr SANS les modifier, après contrôle de leurs sha (config/ld1b_border.json).
Pages mf_0045-0049 (jamais mf_0050-0059), checkpoint FINAL de LD-1 (pas S, sha de la config), seuil 0,5.
Les fonctions pures (tranches, distances, moyennes, règle d'usage) sont importables SANS torch : les modules lourds
ne sont chargés que par main(). Aucun réglage n'est tiré de ce diagnostic (§0, §2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ld1b_border.json"
BINS = [(0.0, 8.0, "0-8"), (8.0, 16.0, "8-16"), (16.0, 32.0, "16-32"), (32.0, None, ">32")]
BIN_NAMES = [b[2] for b in BINS]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- fonctions pures (sans torch) ----------

def bin_of(d: float) -> str:
    """Tranche d'une distance d (px à l'échelle de travail). d < 0 (bande qui sort de la zone) compte en 0-8."""
    for lo, hi, name in BINS:
        if hi is None or d < hi:
            if d >= lo or lo == 0.0:
                return name
    return BINS[-1][2]


def band_distance(baseline, neg: float, pos: float, box) -> float:
    """d = min(distance du haut de la bande au bord haut de la zone, distance du bas de la bande au bord bas).
    §5 (4) : haut de bande = min(y de la baseline) - neg ; bas = max(y de la baseline) + pos (tient compte des lignes inclinées)."""
    pts = np.asarray(baseline, dtype=np.float64)
    top, bottom = float(pts[:, 1].min()) - float(neg), float(pts[:, 1].max()) + float(pos)
    x0, y0, x1, y1 = box
    return min(top - y0, y1 - bottom)


def fraction_in_box(baseline, box) -> float:
    pts = np.asarray(baseline, dtype=np.float64)
    x0, y0, x1, y1 = box
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return float(inside.mean())


def page_counts(gt_rows: list, pred_rows: list) -> dict:
    """gt_rows : (bin, matched) par ligne GT ; pred_rows : (bin, matched) par prédiction. Effectifs et appariés par tranche."""
    out = {name: {"gt_n": 0, "gt_matched": 0, "pred_n": 0, "pred_matched": 0} for name in BIN_NAMES}
    for name, matched in gt_rows:
        out[name]["gt_n"] += 1
        out[name]["gt_matched"] += int(matched)
    for name, matched in pred_rows:
        out[name]["pred_n"] += 1
        out[name]["pred_matched"] += int(matched)
    return out


def aggregate(per_page: list) -> dict:
    """Moyennes NON pondérées par page : par tranche, seules les pages ayant au moins un élément de la tranche comptent."""
    result = {}
    for name in BIN_NAMES:
        recalls, precisions = [], []
        totals = {"gt_n": 0, "pred_n": 0}
        for counts in per_page:
            c = counts[name]
            totals["gt_n"] += c["gt_n"]
            totals["pred_n"] += c["pred_n"]
            if c["gt_n"] > 0:
                recalls.append(c["gt_matched"] / c["gt_n"])
            if c["pred_n"] > 0:
                precisions.append(c["pred_matched"] / c["pred_n"])
        result[name] = {"recall_mean": float(np.mean(recalls)) if recalls else None,
                        "precision_mean": float(np.mean(precisions)) if precisions else None,
                        "pages_with_gt": len(recalls), "pages_with_pred": len(precisions),
                        "gt_lines_total": totals["gt_n"], "pred_total": totals["pred_n"]}
    return result


def rule_of_use(summary: dict, rule: dict, family: str = "Z-GT") -> dict:
    """§2 et §5 (1) : la règle n'est calculée QUE pour Z-GT. Pour Z-PP : « non applicable ».
    Piste citée si rappel(0-8) <= rappel(>32) - 0,10, l'écart étant arrondi (round(gap, 9)) avant la comparaison."""
    if family != rule["zone"]:
        return {"status": "non applicable", "family": family, "zone_of_rule": rule["zone"]}
    low = summary[rule["bin_low"]]["recall_mean"]
    high = summary[rule["bin_high"]]["recall_mean"]
    if low is None or high is None:
        return {"status": "tranche vide : règle non applicable", "cited_as_track": None, "gap_high_minus_low": None}
    gap = round(high - low, 9)
    return {"status": "calculée", "cited_as_track": bool(gap >= rule["min_gap"]), "gap_high_minus_low": gap,
            "threshold": rule["min_gap"], "text": rule["text"]}


# ---------- checkpoint final (§4 du lot : identifié par la sortie de LD-1 elle-même) ----------

_STEP_NAME = re.compile(r"^ld1_step(\d+)(.*)\.pt$")


def resolve_checkpoint(ld1_dir, ld1_report) -> dict:
    """S lu dans ld1_timing_manifest.json ; exactement UN fichier ld1_step{S:06d}.pt (refus : absent, doublon,
    S invalide) ; sha256 du checkpoint, du manifeste et de ld1_report.json de LD-1.
    Le rapport (--ld1-report, sortie de ld1-v, amendement 2) doit nommer ce checkpoint et son sha256."""
    d = Path(ld1_dir)
    manifest_path = d / "ld1_timing_manifest.json"
    report_path = Path(ld1_report)
    if not manifest_path.is_file():
        raise SystemExit("refusé : ld1_timing_manifest.json absent")
    if not report_path.is_file():
        raise SystemExit("refusé : ld1_report.json absent")
    manifest = json.loads(manifest_path.read_text())
    steps = manifest.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise SystemExit("refusé : S invalide dans ld1_timing_manifest.json")
    named = d / f"ld1_step{steps:06d}.pt"
    matches = [n for n in os.listdir(d) if (m := _STEP_NAME.match(n)) and int(m.group(1)) == steps]
    if len(matches) != 1 or matches[0] != named.name:
        raise SystemExit(f"refusé : {len(matches)} fichier(s) pour le pas S={steps} (exactement 1 attendu)")
    if not named.is_file():
        raise SystemExit("refusé : checkpoint du pas S absent")
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or report.get("checkpoint") != named.name:
        raise SystemExit(f"refusé : le rapport LD-1 ne nomme pas le checkpoint {named.name}")
    ckpt_sha = sha256(named)
    if report.get("checkpoint_sha256") != ckpt_sha:
        raise SystemExit("refusé : checkpoint_sha256 du rapport LD-1 différent du sha256 calculé")
    return {"checkpoint_file": named.name, "checkpoint_sha256": ckpt_sha, "steps_S": steps,
            "ld1_timing_manifest_sha256": sha256(manifest_path), "ld1_report_sha256": sha256(report_path)}


# ---------- mesure (torch, LD-1 importé sans modification) ----------

def _load_ld1_modules():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import train_line_detr as TR
    from babaorum.experimental import line_detr as LD
    from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry
    return torch, TR, LD, GroundTruthLine, PredictedLine, evaluate_line_geometry


def check_pins(cfg: dict) -> dict:
    """Refus si line_detr.py ou train_line_detr.py ne sont pas les fichiers gelés (sha épinglés)."""
    got = {"line_detr_sha256": sha256(ROOT / "src" / "babaorum" / "experimental" / "line_detr.py"),
           "train_line_detr_sha256": sha256(ROOT / "scripts" / "train_line_detr.py")}
    for k, v in got.items():
        if v != cfg["pins"][k]:
            raise SystemExit(f"refusé : {k} différent du gel LD-1")
    return got


def measure(args, cfg: dict) -> dict:
    torch, TR, LD, GroundTruthLine, PredictedLine, evaluate_line_geometry = _load_ld1_modules()
    identity = resolve_checkpoint(args.ld1_dir, args.ld1_report)
    ckpt = Path(args.ld1_dir) / identity["checkpoint_file"]
    TR.guard_pages(cfg["pages"])
    device = torch.device("cpu")
    model = LD.LineDetr().to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    L0 = TR.load_l0(json.loads((ROOT / "config" / "ld1_line_detr.json").read_text()))
    cache = Path(args.cache_dir)
    thr = cfg["threshold"]
    per_page = {"Z-GT": [], "Z-PP-blocs": []}
    pages_out = {}
    for pid in cfg["pages"]:
        page = TR.load_scaled(Path(args.root), pid)
        zones = {"Z-GT": [(i, b["box"]) for i, b in enumerate(page["blocks"])],
                 "Z-PP-blocs": TR.pp_zones(page, Path(args.root), pid, cache, L0)}
        gt_objs = [GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]),
                                   baseline_source="ground_truth", geometry_status="available") for g in page["gt_lines"]]
        gt_by_id = {g["id"]: g for g in page["gt_lines"]}
        for family, zlist in zones.items():
            raw = []
            for zid, box in zlist:
                canvas, meta = LD.zone_canvas(page["image"], box, h_page=TR.H_T)
                raw.append((zid, meta, TR.raw_zone_outputs(model, canvas, LD.zone_mask(meta), device)))
            kept = TR.candidates_for_threshold(raw, thr)
            zbox = dict(zlist)
            # appartenance de chaque ligne GT à UNE zone (Z-GT : son bloc ; Z-PP : zone la plus recouverte)
            if family == "Z-GT":
                owner = {}
                for i, b in enumerate(page["blocks"]):
                    for ln in b["lines"]:
                        owner[ln["id"]] = i
            else:
                owner = {}
                for g in page["gt_lines"]:
                    best, best_frac = None, 0.0
                    for zid, box in zlist:
                        f = fraction_in_box(g["baseline"], box)
                        if f > best_frac:
                            best, best_frac = zid, f
                    if best is not None:
                        owner[g["id"]] = best
            if kept:
                pred_objs = [PredictedLine(id=f"p{i}", baseline=k["pts"].tolist(), baseline_source="predicted",
                                           geometry_status="available") for i, k in enumerate(kept)]
                res = evaluate_line_geometry(gt_objs, pred_objs)
                matched_gt = {m.gt_id for m in res.matches}
                matched_pred = {int(m.pred_id[1:]) for m in res.matches}
            else:
                matched_gt, matched_pred = set(), set()
            gt_rows = []
            hors = 0
            for g in page["gt_lines"]:
                zid = owner.get(g["id"])
                if zid is None:
                    hors += 1
                    continue
                d = band_distance(g["baseline"], g["neg"], g["pos"], zbox[zid])
                gt_rows.append((bin_of(d), g["id"] in matched_gt))
            pred_rows = []
            for i, k in enumerate(kept):
                d = band_distance(k["pts"], k["neg"], k["pos"], zbox[k["zone"]])
                pred_rows.append((bin_of(d), i in matched_pred))
            per_page[family].append(page_counts(gt_rows, pred_rows))
            pages_out.setdefault(family, {})[pid] = {"gt_hors_zone": hors, "gt_in_zones": len(gt_rows),
                                                     "predictions": len(pred_rows)}
    summary = {fam: aggregate(rows) for fam, rows in per_page.items()}
    return {"checkpoint_identity": identity, "schema": "ld1b_border_report_v1", "checkpoint": ckpt.name,
            "threshold": thr, "pages": cfg["pages"], "summary": summary,
            "rule_of_use": {fam: rule_of_use(summary[fam], cfg["rule_of_use"], fam) for fam in summary},
            "assignment": cfg["assignment"], "effectifs_note": cfg["effectifs_note"],
            "pages_detail": pages_out, "pins": check_pins(cfg), "descriptive_only": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ld1-dir", required=True)
    ap.add_argument("--ld1-report", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="/tmp/ld1b_hf")
    args = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    check_pins(cfg)
    report = measure(args, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ld1b_border_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"status": "COMPLETE", "report_sha256": sha256(out / "ld1b_border_report.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
