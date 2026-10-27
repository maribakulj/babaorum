"""LD-O1 (classe A, DESCRIPTIF) : que deviennent les lignes GT manquées, surtout celles du bord de zone ?
(docs/END_TO_END_LD_O1_LOT.txt ; source docs/END_TO_END_LD_REFLEXION.txt O1, hypothèses H1-H4). Il ne décide rien et ne règle rien.

Modèle : décodeur LD-2 gelé ld2_main_step003271.pt (e5afd143…), seuil 0,6, NMS de line_detr.nms_lines, marge M16 (MARGIN_H 0,5 inchangé).
Pages : mf_0045-0049 SEULEMENT (pages d'ENTRAÎNEMENT, vues par LD-2). Aucune page neuve ; mf_0050-0069 et mfn_* sont refusées.
Phases (étapes argv du carnet) : D (détecteur PP) ; RUN --pages … (écrit ld_o1_page_<pid>.json) ; REPORT (agrège EXACTEMENT les cinq pages).

Pour chaque ligne GT manquée par les prédictions finales (après NMS, appariement line_geometry.v1), UNE classe, dans cet ordre :
  C1_SUPPR_AUTRE_ZONE  un candidat de score >= 0,6 l'apparie à lui seul (evaluate_line_geometry([g], [c]) -> rappel 1) et la NMS l'a supprimé
                       au profit d'une prédiction gardée d'une AUTRE zone que la sienne ;
  C2_SUPPR_MEME_ZONE   idem, la prédiction gardée qui le supprime vient de la MÊME zone ;
  C3_CONFLIT           une prédiction GARDÉE l'apparie à elle seule, mais l'appariement global l'a donnée à une autre ligne GT ;
  C4_SOUS_SEUIL        seule une requête de score < 0,6 l'apparie à elle seule (meilleur score noté) ;
  C5_AUCUNE            aucune requête ne l'apparie (distance médiane normalisée de la requête la plus proche notée).
Plusieurs candidats supprimés qui apparient la ligne : C1 si l'un d'eux est supprimé par une autre zone, sinon C2.
Comptes par classe, par tranche de bord (définition et tranches de LD-1B : band_distance, BINS) et par source (Z-PP-blocs, Z-GT).
Annexe descriptive : P et R d'un filtre « zone propriétaire » (milieu de la baseline dans la boîte SANS marge de la zone qui l'a produite).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ld3 as E3  # noqa: E402
import ld1b_border_diag as LB  # noqa: E402
import ld2_core as C2  # noqa: E402
import train_ld2 as TD  # noqa: E402
import train_line_detr as TL  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

CONFIG = ROOT / "config" / "ld_o1.json"
SOURCES = E3.SOURCES
BIN_KEYS = LB.BIN_NAMES + ["hors_zone"]
PHASES = ["D", "RUN", "REPORT"]


def load_cfg(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


# ---------- NMS instrumentée : mêmes gardés que line_detr.nms_lines, plus le gardé qui supprime chaque candidat ----------

def instrumented_nms(preds: list) -> tuple:
    """Même tri et même règle que line_detr.nms_lines (line_detr._overlap_rule INCHANGÉE). Retourne (kept, suppressed) ;
    suppressed = [(candidat, premier gardé k, dans l'ordre des gardés, dont la règle de recouvrement est vraie)]."""
    order = sorted(preds, key=lambda p: (-p["score"], p["zone"], p["query"]))
    kept, suppressed = [], []
    for cand in order:
        hit = next((k for k in kept if LD._overlap_rule(cand["pts"], k["pts"], cand["h"], k["h"])), None)
        if hit is None:
            kept.append(cand)
        else:
            suppressed.append((cand, hit))
    return kept, suppressed


def all_queries(zones_raw: list) -> list:
    """TOUTES les requêtes de chaque zone (tout score), en repère de page comme candidates_for_threshold (zone_to_page)."""
    out = []
    for zid, meta, raw in zones_raw:
        for q in range(len(raw["prob"])):
            out.append({"score": float(raw["prob"][q]), "zone": zid, "query": int(q), "pts": LD.zone_to_page(raw["pts"][q], meta),
                        "h": float(raw["neg"][q] + raw["pos"][q]), "neg": float(raw["neg"][q]), "pos": float(raw["pos"][q])})
    return out


# ---------- appariement d'une ligne seule, distance ----------

def gt_object(g: dict) -> GroundTruthLine:
    return GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]),
                           baseline_source="ground_truth", geometry_status="available")


def pred_object(c: dict, i: str = "p") -> PredictedLine:
    return PredictedLine(id=i, baseline=c["pts"].tolist(), baseline_source="predicted", geometry_status="available")


def plausible(g: dict, c: dict, pf: dict) -> bool:
    """Pré-filtre (ne change aucune classe) : boîte de la requête à moins de factor x hauteur + px de la boîte de la ligne GT."""
    if not pf["enabled"]:
        return True
    m = pf["margin_height_factor"] * float(g["neg"] + g["pos"]) + pf["margin_px"]
    gb, cb = np.asarray(g["baseline"]), np.asarray(c["pts"])
    return bool(cb[:, 0].max() >= gb[:, 0].min() - m and cb[:, 0].min() <= gb[:, 0].max() + m
                and cb[:, 1].max() >= gb[:, 1].min() - m - float(g["neg"]) and cb[:, 1].min() <= gb[:, 1].max() + m + float(g["pos"]))


def alone_match(g: dict, c: dict) -> bool:
    res = evaluate_line_geometry([gt_object(g)], [pred_object(c)])
    return bool((res.recall or 0.0) >= 1.0)


def nearest_distance(g: dict, queries: list, samples: int) -> float:
    """Distance médiane (sur `samples` points de la ligne GT) à la requête la plus proche, normalisée par la hauteur GT."""
    a = LD._polyline_samples(np.asarray(g["baseline"], dtype=np.float64), samples)
    h = float(g["neg"] + g["pos"])
    best = None
    for c in queries:
        b = LD._polyline_samples(np.asarray(c["pts"], dtype=np.float64), samples)
        d = float(np.median(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1))) / h
        best = d if best is None or d < best else best
    return best


# ---------- classification d'une ligne manquée ----------

def classify_missed(g: dict, queries: list, thr: float, kept: list, suppressed: list, cfg: dict) -> dict:
    pf = cfg["prefilter"]
    hit = lambda c: plausible(g, c, pf) and alone_match(g, c)                              # noqa: E731
    sup = [(c, s) for c, s in suppressed if hit(c)]
    other = [(c, s) for c, s in sup if s["zone"] != c["zone"]]
    counts = {"n_suppressed_other_zone": len(other), "n_suppressed_same_zone": len(sup) - len(other)}
    if other or sup:
        c, s = (other or sup)[0]
        return dict(counts, **{"class": cfg["classes"][0 if other else 1], "candidate_zone": c["zone"], "candidate_score": c["score"],
                               "suppressor_zone": s["zone"], "n_suppressed_matching": len(sup)})
    kept_hit = [k for k in kept if hit(k)]
    if kept_hit:
        return {**counts, "class": cfg["classes"][2], "kept_zone": kept_hit[0]["zone"], "kept_score": kept_hit[0]["score"]}
    low = [q for q in queries if q["score"] < thr and hit(q)]
    if low:
        return {**counts, "class": cfg["classes"][3], "best_score": max(q["score"] for q in low), "n_below_threshold_matching": len(low)}
    return {**counts, "class": cfg["classes"][4], "nearest_distance_norm": nearest_distance(g, queries, cfg["nearest_distance"]["samples"])}


# ---------- une source (Z-PP-blocs ou Z-GT) d'une page ----------

def key(c: dict) -> tuple:
    return (c["zone"], c["query"])


def owner_filter(kept: list, zbox: dict) -> list:
    """Annexe : une ligne n'est gardée que si le milieu (en abscisse curviligne) de sa baseline est dans la boîte SANS marge de sa zone."""
    out = []
    for k in kept:
        mx, my = LD._polyline_samples(np.asarray(k["pts"], dtype=np.float64), 3)[1]
        x0, y0, x1, y1 = zbox[k["zone"]]
        if x0 <= mx <= x1 and y0 <= my <= y1:
            out.append(k)
    return out


def diagnose_source(page: dict, zones: list, raws: list, owner: dict, thr: float, cfg: dict) -> dict:
    queries = all_queries(raws)
    preds = [q for q in queries if q["score"] >= thr]
    kept, suppressed = instrumented_nms(preds)
    ref = TL.candidates_for_threshold(raws, thr)
    if [key(k) for k in kept] != [key(k) for k in ref] or not all(np.array_equal(a["pts"], b["pts"]) for a, b in zip(kept, ref)):
        raise RuntimeError("NMS instrumentée différente de nms_lines")
    score, matched = E3.score_with_matches(page, kept, raws)
    if score != TD.score_lines(page, kept, raws):
        raise RuntimeError("P/R recalculés différents de train_ld2.score_lines")
    zbox = dict(zones)
    rows, counts, gt_by_bin = [], {c: {b: 0 for b in BIN_KEYS} for c in cfg["classes"]}, {b: 0 for b in BIN_KEYS}
    missed_by_bin = {b: 0 for b in BIN_KEYS}
    for g in page["gt_lines"]:
        zid = owner.get(g["id"])
        b = "hors_zone" if zid is None else LB.bin_of(LB.band_distance(g["baseline"], g["neg"], g["pos"], zbox[zid]))
        gt_by_bin[b] += 1
        if g["id"] in matched:
            continue
        missed_by_bin[b] += 1
        info = classify_missed(g, queries, thr, kept, suppressed, cfg)
        counts[info["class"]][b] += 1
        rows.append(dict(info, id=g["id"], bin=b))
    filt = owner_filter(kept, zbox)
    fscore, _m = E3.score_with_matches(page, filt, raws)
    return {"score": {k: score[k] for k in ("n_pred", "n_gt", "precision", "recall", "splits", "merges")},
            "n_queries": len(queries), "n_candidates": len(preds), "n_kept": len(kept), "n_suppressed": len(suppressed),
            "gt_by_bin": gt_by_bin, "missed_by_bin": missed_by_bin, "counts": counts, "missed": rows,
            "annex_owner_filter": {k: fscore[k] for k in ("n_pred", "n_gt", "precision", "recall")} | {"n_removed": len(kept) - len(filt)}}


# ---------- agrégation (REPORT) ----------

def _pr(rows: list) -> dict:
    return C2.mean_pr([{"page": r["page"], "precision": r["precision"], "recall": r["recall"], "n_pred": r["n_pred"]} for r in rows],
                      len(rows))


def aggregate_source(pages: list, source: str, cfg: dict) -> dict:
    cls = cfg["classes"]
    counts = {c: {b: sum(p[source]["counts"][c][b] for p in pages) for b in BIN_KEYS} for c in cls}
    missed = {b: sum(p[source]["missed_by_bin"][b] for p in pages) for b in BIN_KEYS}
    gt = {b: sum(p[source]["gt_by_bin"][b] for p in pages) for b in BIN_KEYS}
    shares = {b: ({c: counts[c][b] / missed[b] for c in cls} if missed[b] else None) for b in BIN_KEYS}
    rows = [dict(r, page=p["page"]) for p in pages for r in p[source]["missed"]]
    c4 = [r["best_score"] for r in rows if r["class"] == cls[3]]
    c5 = [r["nearest_distance_norm"] for r in rows if r["class"] == cls[4] and r["nearest_distance_norm"] is not None]
    base = [dict(p[source]["score"], page=p["page"]) for p in pages]
    filt = [dict(p[source]["annex_owner_filter"], page=p["page"]) for p in pages]
    b_pr, f_pr = _pr(base), _pr(filt)
    extras = {b: {"c1_with_same_zone_too": sum(1 for r in rows if r["bin"] == b and r["class"] == cls[0] and r["n_suppressed_same_zone"] > 0),
                  "c5_near": sum(1 for r in rows if r["bin"] == b and r["class"] == cls[4] and r["nearest_distance_norm"] is not None
                                 and r["nearest_distance_norm"] <= cfg["c5_near_threshold"]),
                  "c5_far": sum(1 for r in rows if r["bin"] == b and r["class"] == cls[4] and r["nearest_distance_norm"] is not None
                                and r["nearest_distance_norm"] > cfg["c5_near_threshold"]),
                  "c5_no_query": sum(1 for r in rows if r["bin"] == b and r["class"] == cls[4] and r["nearest_distance_norm"] is None)}
              for b in BIN_KEYS}
    return {"counts": counts, "extras_by_bin": extras, "missed_by_bin": missed, "gt_by_bin": gt, "shares_of_missed_by_bin": shares,
            "missed_total": sum(missed.values()), "gt_total": sum(gt.values()),
            "C4_best_score": {"n": len(c4), "median": float(np.median(c4)) if c4 else None, "min": min(c4) if c4 else None,
                              "max": max(c4) if c4 else None},
            "C5_nearest_distance_norm": {"n": len(c5), "median": float(np.median(c5)) if c5 else None},
            "final": {"P": b_pr["P"], "R": b_pr["R"]},
            "annex_owner_filter": {"P": f_pr["P"], "R": f_pr["R"], "dP": f_pr["P"] - b_pr["P"], "dR": f_pr["R"] - b_pr["R"],
                                   "removed": sum(p[source]["annex_owner_filter"]["n_removed"] for p in pages),
                                   "descriptive_only": True},
            "suppressed_total": sum(p[source]["n_suppressed"] for p in pages)}


def run_identity(cfg: dict) -> dict:
    files = {rel: (C2.sha256_file(ROOT / rel) if (ROOT / rel).is_file() else None) for rel in cfg["run_identity_files"]}
    return {"files_sha256": files, "frozen_files_sha256": {rel: C2.sha256_file(ROOT / rel) for rel in cfg["frozen_files_sha256"]},
            "torch": torch.__version__}


def check_pins(cfg: dict) -> None:
    for rel, want in cfg["frozen_files_sha256"].items():
        if C2.sha256_file(ROOT / rel) != want:
            raise SystemExit(f"refusé : {rel} diffère de la version gelée ({want[:8]}…)")


def check_pages(pids: list, cfg: dict, exact: bool = False) -> list:
    """Seules mf_0045-0049 (training pages de LD-1B) ; REPORT exige EXACTEMENT les cinq."""
    allowed = set(cfg["pages"])
    bad = [p for p in pids if p not in allowed]
    if bad or len(set(pids)) != len(pids) or not pids:
        raise SystemExit(f"refusé : pages {bad or pids} hors de {cfg['pages'][0]}..{cfg['pages'][-1]} (ou doublons)")
    TL.guard_pages(pids)
    if exact and sorted(pids) != sorted(cfg["pages"]):
        raise SystemExit("refusé : le rapport exige exactement les cinq pages mf_0045..mf_0049")
    return list(pids)


def build_report(pages: list, cfg: dict, ident: dict, run_id: dict) -> dict:
    ld2r = [C2.RESERVES[i] for i in cfg["reserves_from_ld2"]]
    return {"schema": "ld_o1_report_v1", "checkpoint_sha256": ident["checkpoint_sha256"],
            "reserves": [cfg["reserves"][0]] + ld2r + [cfg["reserves"][1]],
            "prediction": {"source": cfg["prediction"]["source"], "lines": cfg["prediction"]["lines"]},
            "reading_rules": cfg["reading_rules"], "status": "COMPLETE", "descriptive_only": True, "threshold": cfg["threshold"], "margin_h": cfg["margin_h"],
            "pages": [p["page"] for p in pages], "classes": cfg["classes"], "bins": BIN_KEYS,
            "totals": {src: aggregate_source(pages, src, cfg) for src in SOURCES}, "run_identity": run_id,
            "ld2_identity": {k: v for k, v in ident.items() if k != "alphabet"}, "per_page": pages}


# ---------- phases ----------

def make_model(args, cfg: dict):
    ident = E3.check_ld2_identity(args.ld2_dir, cfg)
    ld2cfg = TD.load_all()
    model = TD.new_model("main", ld2cfg, ident["alphabet"], torch.device("cpu"))
    model.load_state_dict(torch.load(ident["checkpoint"], map_location="cpu")["model"])
    model.eval()
    return ident, ld2cfg, model


def phase_D(args, cfg) -> dict:
    ld2cfg = TD.load_all()
    L0 = TL.load_l0(ld2cfg["ld1"])
    local = L0.download_detector(ld2cfg["ld1"]["detector"], Path(args.cache_dir))
    exits = Path(args.out).parent / "exits"
    exits.mkdir(parents=True, exist_ok=True)
    (exits / "ld_o1_download.json").write_text(json.dumps({k: str(v.name) for k, v in local.items()}, indent=1))
    return {"download": "ok", "files": sorted(local)}


def phase_RUN(args, cfg, out: Path) -> dict:
    pids = check_pages(args.pages or [], cfg)
    check_pins(cfg)
    if LD.MARGIN_H != cfg["margin_h"]:
        raise SystemExit("refusé : MARGIN_H n'est pas 0,5 (M16, LD-2 tel quel)")
    ident, ld2cfg, model = make_model(args, cfg)
    L0 = TL.load_l0(ld2cfg["ld1"])
    det_fn = TD.make_det_fn(L0, Path(args.cache_dir))
    thr, root = ident["threshold"], Path(args.root)
    done = []
    for pid in pids:
        t0 = time.monotonic()
        page = TD.load_page_text(root, pid)
        pp = TL.pp_zones(page, root, pid, Path(args.cache_dir), L0, det_fn=det_fn)
        zones = {"Z-PP-blocs": pp, "Z-GT": [(i, b["box"]) for i, b in enumerate(page["blocks"])]}
        owners = {"Z-PP-blocs": E3.owner_pp(page["gt_lines"], pp), "Z-GT": E3.owner_gt(page)}
        res = {"page": pid, "checkpoint_sha256": ident["checkpoint_sha256"]}
        for src, zl in zones.items():
            raws = TD.zone_raws(model, page, zl, thr, ident["alphabet"], torch.device("cpu"))
            res[src] = dict(diagnose_source(page, zl, raws, owners[src], thr, cfg), n_zones=len(zl))
        res["seconds"] = round(time.monotonic() - t0, 1)
        C2.write_json(out / f"ld_o1_page_{pid}.json", res)
        done.append(pid)
    return {"pages_done": done}


def phase_REPORT(args, cfg, out: Path) -> dict:
    pids = check_pages(list(cfg["pages"]), cfg, exact=True)
    pages = [C2.read_json(out / f"ld_o1_page_{pid}.json") for pid in pids]          # chemins explicites, aucune lecture de dossier
    shas = {p["checkpoint_sha256"] for p in pages}
    ident = E3.check_ld2_identity(args.ld2_dir, cfg)
    if shas != {ident["checkpoint_sha256"]}:
        raise SystemExit("refusé : une page a été calculée avec un autre checkpoint")
    check_pins(cfg)
    C2.write_json(out / "ld_o1_report.json", build_report(pages, cfg, ident, run_identity(cfg)))
    return {"report": "ld_o1_report.json"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--root", default=None, help="pages d'ENTRAÎNEMENT (run 3 : e2e3/mf)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ld2-dir", default=None)
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--cache-dir", default="/private/tmp/ld_o1_hf_cache")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"phase": args.phase, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "RUNNING"}
    t0 = time.monotonic()
    try:
        if args.phase in ("RUN", "REPORT"):
            os.environ["HF_HUB_OFFLINE"] = "1"
        if args.phase == "D":
            rec.update(status="COMPLETE", **phase_D(args, cfg))
        elif args.phase == "RUN":
            rec.update(status="COMPLETE", **phase_RUN(args, cfg, out))
        else:
            rec.update(status="COMPLETE", **phase_REPORT(args, cfg, out))
    except SystemExit:
        rec.update(status="FAILED")
        raise
    except Exception as exc:  # noqa: BLE001
        rec.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        raise
    finally:
        rec["duration_s"] = round(time.monotonic() - t0, 1)
        rec["run_identity"] = run_identity(cfg)
        exits = out.parent / "exits"
        exits.mkdir(parents=True, exist_ok=True)
        (exits / f"ld_o1_run_exit_{args.phase}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: rec[k] for k in ("phase", "status")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
