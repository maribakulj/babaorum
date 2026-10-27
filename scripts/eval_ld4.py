"""LD-4 : fusion DETR (LD-2 gelé) + B3+λ (E2E-5 gelé), option O2 (docs/END_TO_END_LD4_PROPOSITION.txt §1-§8 ; pré-enregistrement :
ce document, prédiction comprise). Inférence seule, sans entraînement. Reprend eval_ld3 / data3_generate / train_ld2 par IMPORT (rien de
LD-1, LD-2, LD-3, LD-O1 n'est modifié ; leurs sha sont épinglés dans config/ld4.json).

Phases (étapes argv du carnet) : controls ; D (détecteur PP) ; CAL (UNE page d'ENTRAÎNEMENT : DETR + B3, estimation) ; EVAL (le SEUL endroit où
les pages mf_0070..mf_0079 sont générées puis lues, une fois).
Trois sorties sur les mêmes pages : DETR (« M16 / Z-PP-blocs » de LD-3 : seuil 0,6, NMS §10 (9), MARGIN_H 0,5), B3 (lignes « after » de
train_stage_s_v2.evaluate_full_page : extraction puis raffinement, page entière) et FUSION (§2) = lignes DETR gardées + chaque ligne B3 qu'AUCUNE
ligne DETR gardée ne recouvre (line_detr._overlap_rule(b, k, h_b, h_k), b candidate B3, k gardée DETR). Aucun autre filtre.
CONTRÔLE (§2) : pour chaque page, evaluate_line_geometry sur les lignes B3 extraites ici redonne EXACTEMENT les metrics.after d'evaluate_full_page
(rappel, précision, nombre d'appariées, n_pred) ; sinon refus. La référence B3 du critère est ce metrics.after calculé dans le lancement.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import data3_generate as D3  # noqa: E402
import eval_ld3 as E3  # noqa: E402
import ld1b_border_diag as LB  # noqa: E402
import ld2_core as C2  # noqa: E402
import train_ld2 as TD  # noqa: E402
import train_line_detr as TL  # noqa: E402
import train_stage_s_v2 as T2  # noqa: E402
from train_stage_s import load_scaled_page  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402
from babaorum.experimental.stage_s_model import refine_lines  # noqa: E402
from babaorum.experimental.stage_s_targets import extract_polylines  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

CONFIG = ROOT / "config" / "ld4.json"
EVAL_STARTED = "ld4_eval_started.json"
PHASES = ["controls", "D", "CAL", "EVAL"]
OUTPUTS = ("DETR", "B3", "FUSION")


def load_cfg(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


# ---------- déterminisme (le contrôle exact compare deux passes GPU) ----------

def set_determinism() -> dict:
    """cudnn déterministe, sans benchmark : appelé au début de controls, CAL et EVAL. Le contrôle exact des lignes B3 (§2) reste STRICT ;
    un refus après le marqueur d'EVAL consommerait l'unique T4. Le contrôle est déjà exécuté en CAL, AVANT le marqueur : une différence
    y apparaît (INCOMPLETE, aucune page générée)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return determinism_flags()


def determinism_flags() -> dict:
    return {"cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


def generation_summary(gen: dict, cfg: dict) -> dict:
    """Résumé de la génération pour le rapport. Le champ ld3_config_sha256 du manifeste de DATA-3 désigne config/ld3.json (valeur codée en
    dur dans data3_generate, laissé tel quel) : la config de LD-4 est donnée à côté."""
    out = {k: gen[k] for k in ("mode", "seed", "names", "files_sha256", "generator", "seconds_total")}
    out["overlap"] = {k: v for k, v in (gen["overlap"] or {}).items() if k != "per_page"}
    out["ld4_config_sha256"] = C2.sha256_file(CONFIG) if Path(CONFIG).is_file() else None
    out["note"] = ("le champ ld3_config_sha256 de generator (manifeste de data3_generate) désigne config/ld3.json, pas la config de LD-4 ; "
                   "ld4_config_sha256 est le sha de config/ld4.json")
    return out


# ---------- lignes B3 (extraction + raffinement d'evaluate_full_page) ----------

@torch.no_grad()
def b3_lines(model, page: dict, cfg: dict, device) -> list:
    """Reproduit EXACTEMENT l'extraction puis refine_lines de train_stage_s_v2.evaluate_full_page (page entière, config d'E2E-5) et
    retourne les polylignes « after » avec h = neg + pos. `page` : schéma d'E2E-5 (load_scaled_page) ; `cfg` : config d'E2E-5."""
    model.eval()
    image = torch.as_tensor(page["image"], device=device)[None, None]
    features = model.enc_det(image)
    logit = model.head_d(features)[0]
    prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    extracted = [p for p in extract_polylines(prob, stride=cfg["extraction"]["stride"], threshold=cfg["extraction"]["threshold"],
                                              min_cells=cfg["extraction"]["min_cells"]) if len(p) >= 2]
    if not extracted:
        return []
    refined = refine_lines(features, extracted, model.head_g, head_d=model.head_d, K=model.K)
    polys = refined["baseline"].cpu().tolist()
    neg, pos = refined["neg"].cpu().numpy(), refined["pos"].cpu().numpy()
    return [{"score": 1.0, "zone": "B3", "query": i, "pts": np.asarray(poly, dtype=np.float64), "h": float(neg[i] + pos[i]),
             "neg": float(neg[i]), "pos": float(pos[i])} for i, poly in enumerate(polys)]


# ---------- fusion (§2) ----------

def fuse(detr_kept: list, b3: list) -> tuple:
    """FUSION = lignes DETR gardées + chaque ligne B3 qu'aucune ligne DETR gardée ne recouvre. Retourne (fusion, ajoutées, recouvertes).
    Règle : line_detr._overlap_rule(b.pts, k.pts, b.h, k.h) (b candidate B3, k gardée DETR), inchangée."""
    added, covered = [], []
    for b in b3:
        (covered if any(LD._overlap_rule(b["pts"], k["pts"], b["h"], k["h"]) for k in detr_kept) else added).append(b)
    return list(detr_kept) + added, added, covered


# ---------- mesure ----------

def gt_objects(page: dict) -> list:
    return [GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]),
                            baseline_source="ground_truth", geometry_status="available") for g in page["gt_lines"]]


def score_geometry(page: dict, lines: list) -> tuple:
    """P, R, coupures, fusions de line_geometry.v1 (mêmes GroundTruthLine / PredictedLine que train_ld2.score_lines, sans CER).
    Retourne (score, ids GT appariées, indices des lignes prédites appariées). n_pred = 0 donne P = R = 0."""
    gt = gt_objects(page)
    if not lines:
        return {"n_pred": 0, "n_gt": len(gt), "precision": 0.0, "recall": 0.0, "precision_none": False, "matched": 0,
                "splits": 0, "merges": 0}, set(), set()
    pred = [PredictedLine(id=f"p{i}", baseline=np.asarray(l["pts"]).tolist(), baseline_source="predicted", geometry_status="available")
            for i, l in enumerate(lines)]
    res = evaluate_line_geometry(gt, pred)
    return ({"n_pred": len(pred), "n_gt": len(gt), "precision": None if res.precision is None else float(res.precision),
             "recall": float(res.recall or 0.0), "precision_none": res.precision is None, "matched": int(res.matched_count),
             "splits": int(res.split_candidate_gt_count or 0), "merges": int(res.merge_candidate_pred_count or 0)},
            {m.gt_id for m in res.matches}, {int(m.pred_id[1:]) for m in res.matches})


def check_b3_exact(score: dict, after: dict) -> None:
    """Refus si les lignes B3 extraites ici ne redonnent pas EXACTEMENT metrics.after d'evaluate_full_page."""
    same_p = (after["precision"] is None and (score["precision_none"] or score["n_pred"] == 0)) or \
        (after["precision"] is not None and score["precision"] == after["precision"])
    if not (same_p and score["recall"] == after["recall"] and score["matched"] == after["matched"] and score["n_pred"] == after["n_pred"]):
        raise SystemExit("refusé : les lignes B3 extraites ne redonnent pas exactement metrics.after d'evaluate_full_page")


def edge_table(page: dict, owner: dict, zbox: dict, matched: set) -> dict:
    return E3.edge_counts(page["gt_lines"], owner, zbox, matched)


def evaluate_page_ld4(pid: str, root: Path, ctx: dict) -> dict:
    """Une page : DETR (M16, zones PP), B3 (page entière), FUSION ; chiffres, descriptifs de fusion, table de bord par sortie."""
    thr, alphabet, device = ctx["threshold"], ctx["alphabet"], ctx["device"]
    t0 = time.monotonic()
    page = TD.load_page_text(root, pid)                                    # repère de LD-1 (TL.load_scaled)
    cfg_e2e5 = ctx["cfg_e2e5"]
    page_b3 = load_scaled_page(Path(root), pid, cfg_e2e5["scale"]["target_median_height_px"])      # repère d'E2E-5
    if abs(page_b3["factor"] - page["factor"]) > 1e-9 or page_b3["image"].shape != page["image"].shape:
        raise SystemExit(f"refusé : {pid} : échelles différentes entre le repère DETR et le repère B3")
    row = {"page": pid, "seconds": {"load": round(time.monotonic() - t0, 3)}}
    t = time.monotonic()
    pp = TL.pp_zones(page, root, pid, ctx["cache"], ctx["L0"], det_fn=ctx["det_fn"])
    row["seconds"]["detector"] = round(time.monotonic() - t, 3)
    t = time.monotonic()
    raws = TD.zone_raws(ctx["model"], page, pp, thr, alphabet, device)
    detr = TL.candidates_for_threshold(raws, thr)
    row["seconds"]["DETR"] = round(time.monotonic() - t, 3)
    t = time.monotonic()
    page_b3["alphabet"] = alphabet
    after = T2.evaluate_full_page(ctx["b3_model"], page_b3, cfg_e2e5, device)["metrics"]["after"]       # référence ET contrôle
    b3 = b3_lines(ctx["b3_model"], page_b3, cfg_e2e5, device)
    row["seconds"]["B3"] = round(time.monotonic() - t, 3)
    row["b3_after"] = {k: after[k] for k in ("recall", "precision", "matched", "n_pred", "n_gt")}
    fusion, added, covered = fuse(detr, b3)
    owner, zbox = E3.owner_pp(page["gt_lines"], pp), dict(pp)
    outs = {"DETR": detr, "B3": b3, "FUSION": fusion}
    for name, lines in outs.items():
        score, matched, matched_pred = score_geometry(page, lines)
        row[name] = dict(score, edge=edge_table(page, owner, zbox, matched))
        if name == "B3":
            check_b3_exact(score, after)
        if name == "FUSION":
            n_detr = len(detr)
            row[name].update(b3_total=len(b3), b3_covered=len(covered), b3_added=len(added),
                             b3_added_matched=sum(1 for i in matched_pred if i >= n_detr),
                             b3_added_unmatched=len(added) - sum(1 for i in matched_pred if i >= n_detr))
    row["n_zones"] = len(pp)
    row["seconds"]["total"] = round(time.monotonic() - t0, 3)
    return row


# ---------- critère (§5) ----------

def decoy_reference(ids: list, cfg: dict) -> dict:
    return {i: C2.page_pr(cfg["decoy_b3"]) for i in ids}


def principal_criterion(rows: list, ids: list, cfg: dict, smoke_decoy: bool = False) -> dict:
    """PRINCIPAL : critère de LD-2 À L'IDENTIQUE (C2.criterion), FUSION contre la B3 + λ calculée dans le lancement."""
    fusion = [dict(r["FUSION"], page=r["page"]) for r in rows]
    ref = decoy_reference(ids, cfg) if smoke_decoy else {r["page"]: C2.page_pr(r["b3_after"]) for r in rows}
    return C2.criterion(fusion, ref, ids, cfg)


def descriptive_fusion_vs_detr(rows: list, ids: list, cfg: dict) -> dict:
    """DESCRIPTIF : FUSION - DETR, même bootstrap apparié ; ne décide rien."""
    f = {r["page"]: C2.page_pr(r["FUSION"]) for r in rows}
    d = {r["page"]: C2.page_pr(r["DETR"]) for r in rows}
    bs = C2.paired_bootstrap([f[i][0] - d[i][0] for i in ids], [f[i][1] - d[i][1] for i in ids], cfg)
    return {"bootstrap": bs, "FUSION": {"P": float(np.mean([f[i][0] for i in ids])), "R": float(np.mean([f[i][1] for i in ids]))},
            "DETR": {"P": float(np.mean([d[i][0] for i in ids])), "R": float(np.mean([d[i][1] for i in ids]))}, "descriptive_only": True}


def reading_from_principal(principal: dict, cfg: dict) -> str:
    """Lecture (§5) calculée UNIQUEMENT à partir du verdict du critère principal."""
    return cfg["reading"]["met"] if principal["verdict"] == "CRITERE_REMPLI" else cfg["reading"]["not_met"]


def build_report(rows: list, ids: list, cfg: dict, gen: dict, ident: dict, smoke: bool, run_id: dict = None) -> dict:
    n = len(ids)
    decoy = bool(smoke and rows[0].get("b3_decoy_reference", False))          # référence factice : fumée SEULEMENT
    principal = principal_criterion(rows, ids, cfg, smoke_decoy=decoy)
    outputs = {}
    for name in OUTPUTS:
        flat = [dict(r[name], page=r["page"]) for r in rows]
        agg = C2.mean_pr(flat, n)
        agg.update(n_pred=sum(r["n_pred"] for r in flat), n_gt=sum(r["n_gt"] for r in flat), matched=sum(r["matched"] for r in flat),
                   splits=sum(r["splits"] for r in flat), merges=sum(r["merges"] for r in flat))
        outputs[name] = agg
    fus = [r["FUSION"] for r in rows]
    fusion_counts = {k: sum(f[k] for f in fus) for k in ("b3_total", "b3_covered", "b3_added", "b3_added_matched", "b3_added_unmatched")}
    edge = {name: LB.aggregate([r[name]["edge"]["bins"] for r in rows]) for name in OUTPUTS}
    edge_totals = {name: {b: {"gt_n": sum(r[name]["edge"]["bins"][b]["gt_n"] for r in rows),
                              "gt_matched": sum(r[name]["edge"]["bins"][b]["gt_matched"] for r in rows)} for b in LB.BIN_NAMES}
                   for name in OUTPUTS}
    ld2r = [C2.RESERVES[i] for i in cfg["reserves_from_ld2"]]
    return {"schema": "ld4_report_v1", "checkpoint_sha256": ident["checkpoint_sha256"], "reserves": ld2r + list(cfg["reserves"]),
            "prediction": {"source": cfg["prediction"]["source"], "lines": cfg["prediction"]["lines"]},
            "status": "COMPLETE", "smoke": smoke, "pages": ids, "principal": principal,
            "reading": reading_from_principal(principal, cfg), "descriptive_fusion_vs_detr": descriptive_fusion_vs_detr(rows, ids, cfg),
            "outputs": outputs, "fusion_counts": fusion_counts, "edge_summary": edge, "edge_totals": edge_totals,
            "b3_reference": {"per_page": {r["page"]: r["b3_after"] for r in rows}, "decoy_reference_used": decoy},
            "b3_exact_checked_in_CAL": True, "determinism": determinism_flags(), "run_identity": run_id, "generation": gen, "ld2_identity": {k: v for k, v in ident.items() if k != "alphabet"}, "per_page": rows}


# ---------- identités, contexte ----------

def run_identity(cfg: dict) -> dict:
    files = {rel: (C2.sha256_file(ROOT / rel) if (ROOT / rel).is_file() else None) for rel in cfg["run_identity_files"]}
    return {"files_sha256": files, "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, **determinism_flags()}


def make_context(args, cfg: dict, device) -> dict:
    t0 = time.monotonic()
    E3.check_code_identity(cfg)
    if LD.MARGIN_H != cfg["margin_h"]:
        raise SystemExit("refusé : MARGIN_H n'est pas 0,5 (M16, LD-2 tel quel)")
    m = E3.load_models(args, cfg, device)                       # LD-2 (sha, freeze, rapport), E2E-5 (sha, pas), alphabet reconstruit
    L0 = TL.load_l0(m["ld2cfg"]["ld1"])
    cache = Path(args.cache_dir)
    return {"model": m["model"], "b3_model": m["b3_model"], "cfg_e2e5": m["cfg_e2e5"], "alphabet": m["ident"]["alphabet"],
            "threshold": m["ident"]["threshold"], "device": device, "L0": L0, "det_fn": TD.make_det_fn(L0, cache), "cache": cache,
            "ident": m["ident"], "load_seconds": round(time.monotonic() - t0, 3)}


# ---------- phases ----------

def phase_controls(args, cfg, device) -> dict:
    set_determinism()
    code = E3.check_code_identity(cfg)
    m = E3.load_models(args, cfg, device)
    return {"run_identity": run_identity(cfg), "code_identity": code, "alphabet_size": len(m["ident"]["alphabet"]),
            "ld2": {k: v for k, v in m["ident"].items() if k != "alphabet"}}


def phase_D(args, cfg) -> dict:
    ld2cfg = TD.load_all()
    L0 = TL.load_l0(ld2cfg["ld1"])
    local = L0.download_detector(ld2cfg["ld1"]["detector"], Path(args.cache_dir))
    exits = Path(args.out).parent / "exits"
    exits.mkdir(parents=True, exist_ok=True)
    (exits / "ld4_download.json").write_text(json.dumps({k: str(v.name) for k, v in local.items()}, indent=1))
    return {"download": "ok", "files": sorted(local)}


def phase_CAL(args, cfg, device, root: Path) -> dict:
    """Étalonnage sur UNE page d'ENTRAÎNEMENT (jamais une page d'évaluation) : DETR + B3 (contrôle exact inclus) -> estimation d'EVAL."""
    set_determinism()
    pid = E3.check_cal_page(args.pages[0] if args.pages else cfg["cal"]["page"], cfg)
    ctx = make_context(args, cfg, device)
    t = time.monotonic()
    row = evaluate_page_ld4(pid, root, ctx)
    page_s = time.monotonic() - t
    n = len(args.eval_pages or [pid]) if args.smoke else cfg["cal"]["pages_estimate"]
    rec = {"page": pid, "load_seconds": ctx["load_seconds"], "page_seconds": round(page_s, 3), "pages_estimated": n,
           "estimate_s": round(ctx["load_seconds"] + n * page_s, 1), "seconds_by_step": row["seconds"], "b3_exact_checked": True,
           **determinism_flags()}
    cal = Path(args.cal_out)
    cal.mkdir(parents=True, exist_ok=True)
    C2.write_json(cal / "ld4_cal.json", rec)
    return rec


def phase_EVAL(args, cfg, device, out: Path, root: Path) -> dict:
    """Évaluation UNIQUE. Seul endroit où les pages mf_0070..mf_0079 sont générées puis lues (fumée : page leurre)."""
    out = Path(out)
    set_determinism()
    if (out / "ld4_report.json").exists():
        raise SystemExit("refusé : le rapport existe déjà, aucune nouvelle soumission")
    if (out / EVAL_STARTED).exists():
        raise SystemExit(f"refusé : {EVAL_STARTED} existe, l'évaluation a déjà démarré ; une nouvelle EVAL exige une décision "
                         "explicite de pA (le fichier n'est jamais supprimé automatiquement)")
    if args.decoy_reference and not args.smoke:
        raise SystemExit("refusé : --decoy-reference n'existe qu'en fumée")
    mode = "decoy" if args.smoke else "real"
    ctx = make_context(args, cfg, device)
    try:                                                              # marqueur créé AVANT la génération et la première page
        with open(out / EVAL_STARTED, "x") as f:
            json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode,
                       "checkpoint_sha256": ctx["ident"]["checkpoint_sha256"]}, f, indent=1)
    except FileExistsError:
        raise SystemExit(f"refusé : {EVAL_STARTED} existe, une nouvelle EVAL exige une décision explicite de pA") from None
    work = Path(args.work)
    pages_root = work / "ld4pages"
    ref_e2e4 = json.loads(Path(args.e2e4_report).read_text()) if args.e2e4_report else None
    try:
        gen = D3.produce_pages(mode, cfg, Path(args.gen_src), work, pages_root, called_from_eval_ld3=True,
                               n_pages=cfg["pages"]["decoy_n"] if mode == "decoy" else None, mf_dir=root, reference=ref_e2e4,
                               extra_libs=args.extra_libs, timeout_s=cfg["robustness"]["timeouts_s"]["generate"])
    except D3.GenerationFailed:
        if (work / "data3_generate_failure.json").is_file():
            shutil.copy2(work / "data3_generate_failure.json", out / "data3_generate_failure.json")
        raise
    for name in ("data3_generation_manifest.json", "data3_overlap_report.json"):
        if (work / name).is_file():
            shutil.copy2(work / name, out / name)
    if gen["overlap"] is not None and gen["overlap"]["blocking"]:
        raise SystemExit(f"refusé : recouvrement bloquant ({gen['overlap']['status']}) ; aucune page n'est évaluée")
    ids = E3.check_eval_ids(gen["names"], mode, cfg)
    rows, partial = [], out / "ld4_eval_pages.jsonl"
    for pid in ids:
        row = evaluate_page_ld4(pid, pages_root, ctx)
        row["b3_decoy_reference"] = bool(args.decoy_reference)
        rows.append(row)
        with partial.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    gen_summary = generation_summary(gen, cfg)
    report = build_report(rows, ids, cfg, gen_summary, ctx["ident"], bool(args.smoke), run_identity(cfg))
    C2.write_json(out / "ld4_report.json", report)
    C2.write_json(out / "ld4_edge_table.json", {"pages": ids, "edge_summary": report["edge_summary"], "edge_totals": report["edge_totals"],
                                                "per_page": {r["page"]: {o: r[o]["edge"] for o in OUTPUTS} for r in rows}})
    return {"principal": report["principal"]["verdict"], "reading": report["reading"]}


# ---------- point d'entrée ----------

def main(argv=None) -> int:
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--root", default=None, help="pages d'ENTRAÎNEMENT (run 3 : e2e3/mf) ; lecture de mf_0000-0049 seulement")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ld2-dir", default=None)
    ap.add_argument("--e2e5-checkpoint", default=None)
    ap.add_argument("--e2e4-report", default=None)
    ap.add_argument("--gen-src", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--extra-libs", default=None)
    ap.add_argument("--cal-out", default=None)
    ap.add_argument("--cache-dir", default="/private/tmp/ld4_hf_cache")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--eval-pages", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--decoy-reference", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"phase": args.phase, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "RUNNING"}
    t0 = time.monotonic()
    try:
        if args.phase in ("CAL", "EVAL"):
            os.environ["HF_HUB_OFFLINE"] = "1"
        if args.phase == "D":
            rec.update(status="COMPLETE", **phase_D(args, cfg))
            return 0
        TL.check_cuda(args.device)
        device = torch.device(args.device)
        if args.phase == "controls":
            rec.update(status="COMPLETE", controls=phase_controls(args, cfg, device))
        elif args.phase == "CAL":
            rec.update(status="COMPLETE", cal=phase_CAL(args, cfg, device, Path(args.root)))
        else:
            rec.update(status="COMPLETE", **phase_EVAL(args, cfg, device, out, Path(args.root)))
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
        (exits / f"ld4_run_exit_{args.phase}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: rec[k] for k in ("phase", "status")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
