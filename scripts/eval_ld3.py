"""LD-3 : la marge du recadrage (0,5 h contre 1,0 h) explique-t-elle le rappel perdu par LD-2 ? Inférence seule
(docs/END_TO_END_LD3_PROPOSITION.txt §1-§8 ; pré-enregistrement : ce document, prédiction comprise).

Phases (étapes argv du carnet, chacune dans son sous-processus) :
  controls  identités : sha du checkpoint LD-2 (e5afd143…), bloc freeze du manifeste de LD-2 (seuil 0,6) et rapport de LD-2,
            sha du checkpoint E2E-5, fichiers gelés de LD-2 et du code d'E2E-5 (sha de la provenance de e2e5_lambda_run_report.json) ;
  D         téléchargement du détecteur PP (comme LD-2 phase_D) ;
  CAL       UNE page d'ENTRAÎNEMENT (mf_0045) : deux bras, B3 + λ, durée -> estimation (porte avant EVAL) ;
  EVAL      le SEUL endroit où les pages mf_0060..mf_0069 sont générées (data3_generate, mode réel) puis lues, une fois.
Bras : M16 = MARGIN_H 0,5 (LD-2 tel quel, contrôle) ; M32 = MARGIN_H 1,0. Même modèle gelé, mêmes pages, mêmes zones PP-blocs
(calculées UNE fois par page). Pas de témoin. Les chiffres ne décident rien tant que le critère du §5 n'est pas lu.

RÉFÉRENCE B3 + λ (§2) : les valeurs de e2e5_lambda_run_report.json (eval[].metrics.after) viennent de
train_stage_s_v2.run_v2 -> evaluate_all(model, eval_loaded, cfg, device) -> evaluate_full_page(model, page, cfg, device) avec
cfg = config/stage_s_smoke_v2.json (sha a464e1b6…), pages chargées par train_stage_s.load_scaled_page(root, pid, 32), page["alphabet"]
= alphabet de mf_0000-0049 (74 caractères), modèle StageSModel("B3", 74, channels, K, hidden, ctc_det_grad_scale = 1/6) chargé du
checkpoint_last.pt de pas 2000 (sha ef5e8268…). Ici, b3_page appelle CETTE MÊME fonction evaluate_full_page, sans modification,
sur ces mêmes objets ; l'identité du code est contrôlée par les sha de la provenance du rapport (config freeze « code_pins »). Une
différence ne peut venir que de la version de torch / du GPU (non bit-à-bit garantie), pas du code.
"""
from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import re
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
import ld1b_border_diag as LB  # noqa: E402
import ld2_core as C2  # noqa: E402
import train_ld2 as TD  # noqa: E402
import train_line_detr as TL  # noqa: E402
import train_stage_s_v2 as T2  # noqa: E402
from train_stage_s import levenshtein, load_scaled_page  # noqa: E402,F401
from babaorum.experimental import line_detr as LD  # noqa: E402
from babaorum.experimental.stage_s_arms import StageSModel  # noqa: E402
from babaorum.experimental.stage_s_data import build_alphabet  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

CONFIG = ROOT / "config" / "ld3.json"
EVAL_STARTED = "ld3_eval_started.json"
PHASES = ["controls", "D", "CAL", "EVAL"]
SOURCES = ("Z-PP-blocs", "Z-GT")
LAMBDA = 1.0 / 6.0


def load_cfg(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def arms(cfg: dict) -> dict:
    return {"M16": cfg["margins"]["M16"], "M32": cfg["margins"]["M32"]}


# ---------- marge : paramètre de LD-3, valeur par défaut de line_detr inchangée ----------

@contextlib.contextmanager
def margin_h(value: float):
    """Fixe babaorum.experimental.line_detr.MARGIN_H (lu par zone_canvas à CHAQUE appel) et le rétablit TOUJOURS, exception comprise."""
    old = LD.MARGIN_H
    LD.MARGIN_H = value
    try:
        yield
    finally:
        LD.MARGIN_H = old


def zone_reduction(metas: list) -> dict:
    """Zones réduites dans le canevas 1536 x 512 (facteur f < 1) : nombre, part et facteur moyen (0,5 (fx + fy)), §2."""
    f = [0.5 * (m.fx + m.fy) for m in metas if (m.fx, m.fy) != (1.0, 1.0)]
    return {"zones": len(metas), "reduced": len(f), "reduced_share": (len(f) / len(metas)) if metas else None,
            "mean_f_reduced": float(np.mean(f)) if f else None, "min_f": float(min(f)) if f else None}


# ---------- distance au bord : appartenance des lignes GT à UNE zone (règles de ld1b_border_diag) ----------

def owner_gt(page: dict) -> dict:
    """Z-GT : le bloc GT qui contient la ligne ; une ligne hors des blocs gardés n'a pas de zone (gt_hors_zone)."""
    return {ln["id"]: i for i, b in enumerate(page["blocks"]) for ln in b["lines"]}


def owner_pp(gt_lines: list, zones: list) -> dict:
    """Z-PP : la zone qui contient la plus grande part des points de la baseline (fraction_in_box), le premier indice à égalité ;
    aucune zone (part nulle partout) : pas d'entrée (gt_hors_zone)."""
    owner = {}
    for g in gt_lines:
        best, best_frac = None, 0.0
        for zid, box in zones:
            f = LB.fraction_in_box(g["baseline"], box)
            if f > best_frac:
                best, best_frac = zid, f
        if best is not None:
            owner[g["id"]] = best
    return owner


def edge_counts(gt_lines: list, owner: dict, zbox: dict, matched: set) -> dict:
    """Effectifs et appariés par tranche de distance verticale au bord de la zone (band_distance, BINS de LD-1B)."""
    rows, hors = [], 0
    for g in gt_lines:
        zid = owner.get(g["id"])
        if zid is None:
            hors += 1
            continue
        rows.append((LB.bin_of(LB.band_distance(g["baseline"], g["neg"], g["pos"], zbox[zid])), g["id"] in matched))
    return {"bins": LB.page_counts(rows, []), "gt_hors_zone": hors, "gt_in_zones": len(rows)}


# ---------- mesure d'une page ----------

def score_with_matches(page: dict, kept: list, raws: list) -> tuple:
    """Mêmes chiffres que train_ld2.score_lines (copie de son corps, evaluate_line_geometry appelé UNE fois) et les identifiants des
    lignes GT appariées. Un test vérifie l'égalité des deux dictionnaires."""
    gt = [GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]),
                          baseline_source="ground_truth", geometry_status="available") for g in page["gt_lines"]]
    cer = {"edits": 0, "chars": 0, "lines": 0, "skipped": 0, "matched_readable_lines": 0, "matched_readable_chars": 0,
           "skipped_by_cause": {"degenerate_chord": 0, "band_outside_support": 0, "reader_exception": 0, "unknown": 0}}
    if not kept:
        return {"n_pred": 0, "n_gt": len(gt), "precision": 0.0, "recall": 0.0, "precision_none": False, "splits": 0,
                "merges": 0, "cer": cer}, set()
    pred = [PredictedLine(id=f"p{i}", baseline=k["pts"].tolist(), baseline_source="predicted", geometry_status="available")
            for i, k in enumerate(kept)]
    res = evaluate_line_geometry(gt, pred)
    gt_by_id = {g["id"]: g for g in page["gt_lines"]}
    raw_of = {zid: raw for zid, _meta, raw in raws}
    pred_idx = {p.id: i for i, p in enumerate(pred)}
    try:
        for m in res.matches:
            g = gt_by_id[m.gt_id]
            if not TD._readable(g["text"], g["legibility"]):
                continue
            k = kept[pred_idx[m.pred_id]]
            cer["matched_readable_lines"] += 1
            cer["matched_readable_chars"] += len(g["text"])
            hyp = raw_of[k["zone"]]["hyp"].get(k["query"])
            if hyp is None:
                cer["skipped"] += 1
                cer["skipped_by_cause"][raw_of[k["zone"]].get("hyp_none", {}).get(k["query"], "unknown")] += 1
                continue
            cer["edits"] += levenshtein(g["text"], hyp)
            cer["chars"] += len(g["text"])
            cer["lines"] += 1
    except Exception as exc:  # noqa: BLE001
        cer["error"] = f"{type(exc).__name__}: {exc}"[:200]
    cer["reading_errors"] = sum(len(raw.get("reading_errors", [])) for raw in raw_of.values())
    score = {"n_pred": len(pred), "n_gt": len(gt),
             "precision": None if res.precision is None else float(res.precision),
             "recall": float(res.recall or 0.0), "precision_none": res.precision is None,
             "splits": int(res.split_candidate_gt_count or 0), "merges": int(res.merge_candidate_pred_count or 0), "cer": cer}
    return score, {m.gt_id for m in res.matches}


def b3_page(model, cfg_e2e5: dict, root: Path, pid: str, alphabet: list, device) -> dict:
    """Référence B3 + λ : evaluate_full_page d'E2E-5, appelée telle quelle (voir l'en-tête du module)."""
    t0 = time.monotonic()
    page = load_scaled_page(Path(root), pid, cfg_e2e5["scale"]["target_median_height_px"])
    page["alphabet"] = alphabet
    after = T2.evaluate_full_page(model, page, cfg_e2e5, device)["metrics"]["after"]
    return {"precision": after["precision"], "recall": after["recall"], "n_pred": after["n_pred"], "n_gt": after["n_gt"],
            "matched": after["matched"], "seconds": round(time.monotonic() - t0, 3)}


def evaluate_page_ld3(pid: str, root: Path, ctx: dict) -> dict:
    """Une page : zones PP calculées UNE fois, partagées par les deux bras ; par bras et par source (Z-PP-blocs décisive, Z-GT
    descriptive) : chiffres de LD-2, réduction des zones, table de bord ; B3 + λ sur la même page."""
    thr, alphabet, device, model = ctx["threshold"], ctx["alphabet"], ctx["device"], ctx["model"]
    t0 = time.monotonic()
    page = TD.load_page_text(root, pid)
    row = {"page": pid, "seconds": {"load": round(time.monotonic() - t0, 3)}}
    t = time.monotonic()
    pp = TL.pp_zones(page, root, pid, ctx["cache"], ctx["L0"], det_fn=ctx["det_fn"])
    row["seconds"]["detector"] = round(time.monotonic() - t, 3)
    zones = {"Z-PP-blocs": pp, "Z-GT": [(i, b["box"]) for i, b in enumerate(page["blocks"])]}
    owners = {"Z-PP-blocs": owner_pp(page["gt_lines"], pp), "Z-GT": owner_gt(page)}
    for arm, m in ctx["arms"].items():
        row[arm] = {}
        for src, zl in zones.items():
            t = time.monotonic()
            with margin_h(m):
                raws = TD.zone_raws(model, page, zl, thr, alphabet, device)
            kept = TL.candidates_for_threshold(raws, thr)
            score, matched = score_with_matches(page, kept, raws)
            score.update(n_zones=len(zl), reduction=zone_reduction([meta for _z, meta, _r in raws]),
                         edge=edge_counts(page["gt_lines"], owners[src], dict(zl), matched))
            row[arm][src] = score
            row["seconds"][f"{arm}/{src}"] = round(time.monotonic() - t, 3)
    if ctx["b3"] is not None:
        t = time.monotonic()
        row["B3"] = ctx["b3"](pid)
        row["seconds"]["B3"] = round(time.monotonic() - t, 3)
    row["seconds"]["total"] = round(time.monotonic() - t0, 3)
    return row


# ---------- critère (§5) ----------

def principal_criterion(rows16: list, rows32: list, ids: list, cfg: dict) -> dict:
    """PRINCIPAL : bootstrap apparié de M32 - M16 en Z-PP-blocs (10 000 tirages, graine 20261018, percentiles 2,5 / 97,5) ;
    rempli ssi borne basse de ΔR > 0 ET borne basse de ΔP >= -0,01."""
    pr = cfg["criterion"]["principal"]
    a, b = {r["page"]: C2.page_pr(r) for r in rows16}, {r["page"]: C2.page_pr(r) for r in rows32}
    d_p = [b[i][0] - a[i][0] for i in ids]
    d_r = [b[i][1] - a[i][1] for i in ids]
    bs = C2.paired_bootstrap(d_p, d_r, cfg)
    met = bool(bs["dR"]["low"] > pr["dr_low_gt"] and bs["dP"]["low"] >= pr["dp_low_min"])
    return {"met": met, "verdict": "PRINCIPAL_REMPLI" if met else "PRINCIPAL_NON_REMPLI", "bootstrap": bs,
            "dr_low_gt": pr["dr_low_gt"], "dp_low_min": pr["dp_low_min"], "zones": pr["zones"], "paired": pr["paired"],
            "M16": {"P": float(np.mean([a[i][0] for i in ids])), "R": float(np.mean([a[i][1] for i in ids]))},
            "M32": {"P": float(np.mean([b[i][0] for i in ids])), "R": float(np.mean([b[i][1] for i in ids]))}}


def secondary_criterion(rows32: list, ref: dict, ids: list, cfg: dict) -> dict:
    """SECONDAIRE : critère (a)+(b) de LD-2 à l'identique (C2.criterion), M32 contre B3 + λ calculée dans le lancement."""
    return C2.criterion(rows32, ref, ids, cfg)


def reading_from_principal(principal: dict) -> str:
    """Lecture des issues (amendement 2 (2)) : calculée UNIQUEMENT à partir du verdict principal ; le rappel 0-8 px n'y entre pas."""
    return "LD3B_SUITE_LOGIQUE" if principal["met"] else "PISTE_DU_BORD_CLOSE"


def mechanism(summary: dict) -> dict:
    """Descriptif : rappel moyen de la tranche 0-8 px par bras et par source (moyenne par page, LD-1B aggregate)."""
    return {arm: {src: summary[arm][src]["0-8"] for src in SOURCES} for arm in summary}


# ---------- identités (controls) ----------

def check_code_identity(cfg: dict) -> dict:
    """Fichiers gelés de LD-1 / LD-2 et code d'E2E-5 : sha égaux aux valeurs pré-enregistrées (provenance du rapport d'E2E-5)."""
    out = {}
    for group in (cfg["frozen_files_sha256"], cfg["e2e5"]["code_pins"]):
        for rel, want in group.items():
            got = C2.sha256_file(ROOT / rel)
            if got != want:
                raise SystemExit(f"refusé : {rel} diffère de la version gelée ({want[:8]}…)")
            out[rel] = got
    return out


def check_ld2_identity(ld2_dir, cfg: dict) -> dict:
    """Sortie de LD-2 : checkpoint pas 3271 de sha e5afd143…, bloc freeze du manifeste (checkpoint, sha, seuil 0,6) et rapport
    ld2_report.json qui nomment le même checkpoint. Retourne l'alphabet et le seuil du manifeste."""
    d, c = Path(ld2_dir), cfg["ld2"]
    man = C2.read_json(d / c["manifest"])
    fz = man.get("freeze") or {}
    ck = d / c["checkpoint_file"]
    sha = C2.require_sha(ck, c["checkpoint_sha256"], "checkpoint LD-2")
    if fz.get("checkpoint") != c["checkpoint_file"] or fz.get("checkpoint_sha256") != c["checkpoint_sha256"]:
        raise SystemExit("refusé : le bloc freeze du manifeste de LD-2 ne nomme pas le checkpoint attendu")
    if fz.get("threshold") != cfg["threshold"] or man.get("S2") != c["steps"]:
        raise SystemExit("refusé : seuil ou nombre de pas de LD-2 différents de la pré-inscription")
    rep_path = d / c["report"]
    if not rep_path.is_file():
        raise SystemExit("refusé : ld2_report.json absent de la sortie de LD-2")
    rep = C2.read_json(rep_path)
    if rep.get("checkpoint_sha256") != sha:
        raise SystemExit("refusé : checkpoint_sha256 du rapport de LD-2 différent du checkpoint")
    return {"alphabet": man["alphabet"], "threshold": fz["threshold"], "checkpoint": str(ck), "checkpoint_sha256": sha,
            "ld2_report_sha256": C2.sha256_file(rep_path), "ld2_manifest_sha256": C2.sha256_file(d / c["manifest"])}


def rebuild_e2e5_alphabet(root, cfg: dict, target_h: float) -> list:
    """Alphabet d'E2E-5 reconstruit EXACTEMENT comme run_v2 : build_alphabet([load_scaled_page(root, pid, target_h) pour les pages
    d'entraînement du rapport]) ; pages d'ENTRAÎNEMENT lues une à une par chemins explicites."""
    return build_alphabet([load_scaled_page(Path(root), pid, target_h) for pid in cfg["e2e5"]["train_pages"]])


def check_alphabet(rebuilt: list, ld2_alphabet: list, cfg: dict) -> list:
    """Refus sauf égalité élément par élément (ordre compris) avec l'alphabet du manifeste de LD-2 ; taille 74."""
    if list(rebuilt) != list(ld2_alphabet):
        raise SystemExit("refusé : l'alphabet d'E2E-5 reconstruit diffère de celui du manifeste de LD-2")
    if len(rebuilt) != cfg["e2e5"]["alphabet_size"]:
        raise SystemExit("refusé : taille d'alphabet différente de celle d'E2E-5")
    return list(rebuilt)


def check_eval_ids(ids: list, mode: str, cfg: dict) -> list:
    """Avant la première page : réel = exactement les noms de DATA-3 et 10 pages ; fumée = noms leurres seulement."""
    if mode == "real":
        if list(ids) != D3.real_names(cfg) or len(ids) != cfg["criterion"]["a"]["pages"]:
            raise SystemExit("refusé : les pages réelles ne sont pas exactement mf_0060..mf_0069 (10)")
    else:
        pat = re.compile(re.escape(cfg["pages"]["decoy_prefix"]) + r"\d{4}")
        if not ids or any(not pat.fullmatch(i) for i in ids):
            raise SystemExit("refusé : la fumée n'évalue que des pages leurres")
    return list(ids)


def check_cal_page(pid: str, cfg: dict) -> str:
    """CAL : page d'ENTRAÎNEMENT seulement ; refus des pages générées, des leurres et de tout autre nom."""
    TL.guard_pages([pid])
    if not re.fullmatch(cfg["pages"]["cal_name_regex"], pid):
        raise SystemExit(f"refusé : la page d'étalonnage {pid!r} n'est pas une page d'entraînement mf_00[0-4]x")
    return pid


def run_identity(cfg: dict) -> dict:
    """sha256 des fichiers de LD-3 tels que lus à l'exécution, versions de torch et de CUDA."""
    files = {rel: (C2.sha256_file(ROOT / rel) if (ROOT / rel).is_file() else None) for rel in cfg["run_identity_files"]}
    return {"files_sha256": files, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_models(args, cfg: dict, device) -> dict:
    ident = check_ld2_identity(args.ld2_dir, cfg)
    ld2cfg = TD.load_all()
    model = TD.new_model("main", ld2cfg, ident["alphabet"], device)
    model.load_state_dict(torch.load(ident["checkpoint"], map_location="cpu")["model"])
    model.eval()
    e = cfg["e2e5"]
    C2.require_sha(args.e2e5_checkpoint, e["checkpoint_sha256"], "checkpoint E2E-5")
    ck = torch.load(args.e2e5_checkpoint, map_location="cpu")
    if ck["step"] != e["step"] or ck.get("arm") != e["arm"]:
        raise SystemExit("refusé : le checkpoint E2E-5 n'est pas B3 au pas 2000")
    cfg_e2e5 = json.loads((ROOT / "config" / "stage_s_smoke_v2.json").read_text())
    ident["alphabet"] = check_alphabet(rebuild_e2e5_alphabet(args.root, cfg, cfg_e2e5["scale"]["target_median_height_px"]),
                                       ident["alphabet"], cfg)                  # alphabet reconstruit, utilisé pour B3
    b3 = StageSModel("B3", len(ident["alphabet"]), channels=cfg_e2e5["model"]["channels"], K=cfg_e2e5["model"]["K"],
                     hidden=cfg_e2e5["model"].get("hidden", 128), ctc_det_grad_scale=LAMBDA).to(device)
    b3.load_state_dict(ck["model"])
    return {"ident": ident, "model": model, "b3_model": b3, "cfg_e2e5": cfg_e2e5, "ld2cfg": ld2cfg}


def make_context(args, cfg: dict, device, with_b3: bool = True, decoy_b3: bool = False) -> dict:
    t0 = time.monotonic()
    check_code_identity(cfg)
    m = load_models(args, cfg, device)
    L0 = TL.load_l0(m["ld2cfg"]["ld1"])
    cache = Path(args.cache_dir)

    def b3(pid, _m=m):
        if decoy_b3:                                                    # fumée : référence factice, jamais écrite comme un résultat
            return dict(cfg["decoy_b3"], decoy=True, seconds=0.0)
        return b3_page(_m["b3_model"], _m["cfg_e2e5"], ctx["pages_root"], pid, _m["ident"]["alphabet"], device)

    ctx = {"model": m["model"], "alphabet": m["ident"]["alphabet"], "threshold": m["ident"]["threshold"], "device": device,
           "L0": L0, "det_fn": TD.make_det_fn(L0, cache), "cache": cache, "arms": arms(cfg), "ident": m["ident"],
           "b3": b3 if with_b3 else None, "pages_root": None, "load_seconds": round(time.monotonic() - t0, 3)}
    return ctx


# ---------- phases ----------

def phase_controls(args, cfg, device) -> dict:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("refusé : --device cuda sans CUDA")
    code = check_code_identity(cfg)
    m = load_models(args, cfg, device)
    return {"run_identity": run_identity(cfg), "code_identity": code, "ld2": {k: v for k, v in m["ident"].items() if k != "alphabet"},
            "alphabet_size": len(m["ident"]["alphabet"]), "torch": torch.__version__, "margins": cfg["margins"]}


def phase_D(args, cfg) -> dict:
    ld2cfg = TD.load_all()
    L0 = TL.load_l0(ld2cfg["ld1"])
    local = L0.download_detector(ld2cfg["ld1"]["detector"], Path(args.cache_dir))
    exits = Path(args.out).parent / "exits"
    exits.mkdir(parents=True, exist_ok=True)
    (exits / "ld3_download.json").write_text(json.dumps({k: str(v.name) for k, v in local.items()}, indent=1))
    return {"download": "ok", "files": sorted(local)}


def phase_CAL(args, cfg, device, root: Path) -> dict:
    """Étalonnage sur UNE page d'ENTRAÎNEMENT (jamais une page d'évaluation) : deux bras + B3 ; estimation de l'EVAL."""
    pid = check_cal_page(args.pages[0] if args.pages else cfg["cal"]["page"], cfg)
    ctx = make_context(args, cfg, device)
    ctx["pages_root"] = root
    t = time.monotonic()
    row = evaluate_page_ld3(pid, root, ctx)
    page_s = time.monotonic() - t
    n = len(args.eval_pages or [pid]) if args.smoke else cfg["cal"]["pages_estimate"]
    rec = {"page": pid, "load_seconds": ctx["load_seconds"], "page_seconds": round(page_s, 3), "pages_estimated": n,
           "estimate_s": round(ctx["load_seconds"] + n * page_s, 1), "seconds_by_source": row["seconds"]}
    cal = Path(args.cal_out)
    cal.mkdir(parents=True, exist_ok=True)
    C2.write_json(cal / "ld3_cal.json", rec)
    return rec


def b3_reference(rows: list, ids: list) -> dict:
    return {r["page"]: C2.page_pr(r["B3"]) for r in rows if r["page"] in ids}


def arm_rows(rows: list, arm: str, src: str) -> list:
    """Lignes au format de train_ld2.aggregate / C2.criterion pour un bras et une source."""
    return [{"page": r["page"], "seconds": {src: r["seconds"][f"{arm}/{src}"]}, src: r[arm][src]} for r in rows]


def build_report(rows: list, ids: list, cfg: dict, gen: dict, ident: dict, smoke: bool, run_id: dict = None) -> dict:
    n = len(ids)
    agg = {arm: {src: TD.aggregate(arm_rows(rows, arm, src), src, n) for src in SOURCES} for arm in arms(cfg)}
    flat = {arm: {src: [dict(r[arm][src], page=r["page"]) for r in rows] for src in SOURCES} for arm in arms(cfg)}
    ref = b3_reference(rows, ids)
    edge_summary = {arm: {src: LB.aggregate([r[arm][src]["edge"]["bins"] for r in rows]) for src in SOURCES} for arm in arms(cfg)}
    edge_totals = {arm: {src: {b: {"gt_n": sum(r[arm][src]["edge"]["bins"][b]["gt_n"] for r in rows),
                                   "gt_matched": sum(r[arm][src]["edge"]["bins"][b]["gt_matched"] for r in rows)}
                              for b in LB.BIN_NAMES} for src in SOURCES} for arm in arms(cfg)}
    reduction = {arm: {src: reduction_total([r[arm][src]["reduction"] for r in rows]) for src in SOURCES} for arm in arms(cfg)}
    # ordre : plafond réel (LD-0R), bord (LD-1B), échelle par la GT, zones PP réglées sur Gallica / NewsEye, bootstrap, seuil,
    # réserves de LD-3, périmètre du CER
    ld2r = [C2.RESERVES[i] for i in cfg["reserves_from_ld2"]]            # [6] plafond, [7] bord, [0] échelle, [3] zones PP, [2] bootstrap, [5] seuil
    reserves = ld2r[:2] + [cfg["reserves"][0]] + ld2r[2:] + list(cfg["reserves"][1:]) + [C2.CER_SCOPE]
    principal = principal_criterion(flat["M16"]["Z-PP-blocs"], flat["M32"]["Z-PP-blocs"], ids, cfg)
    return {"schema": "ld3_report_v1", "checkpoint_sha256": ident["checkpoint_sha256"], "reserves": reserves,
            "prediction": {"source": cfg["prediction"]["source"], "lines": cfg["prediction"]["lines"],
                           "amendment": cfg["prediction"]["amendment"]},
            "status": "COMPLETE", "smoke": smoke, "pages": ids,
            "principal": principal,
            "reading": reading_from_principal(principal),
            "secondary": secondary_criterion(flat["M32"]["Z-PP-blocs"], ref, ids, cfg),
            "mechanism_bin_0_8": mechanism(edge_summary), "arms": agg, "edge_summary": edge_summary, "edge_totals": edge_totals,
            "zone_reduction": reduction,
            "B3": {"per_page": {r["page"]: r["B3"] for r in rows}, "P": float(np.mean([ref[i][0] for i in ids])),
                   "R": float(np.mean([ref[i][1] for i in ids])), "decoy": any(r["B3"].get("decoy") for r in rows)},
            "run_identity": run_id, "generation": gen, "ld2_identity": {k: v for k, v in ident.items() if k != "alphabet"}, "per_page": rows}


def reduction_total(per_page: list) -> dict:
    """Zones réduites sur toutes les pages : facteur moyen PONDÉRÉ par le nombre de zones réduites de chaque page
    (somme des mean_f x reduced / somme des reduced), minimum sur les pages."""
    n_red = sum(r["reduced"] for r in per_page)
    zones = sum(r["zones"] for r in per_page)
    mins = [r["min_f"] for r in per_page if r.get("min_f") is not None]
    return {"zones": zones, "reduced": n_red, "reduced_share": (n_red / zones) if zones else None,
            "mean_f_reduced": (sum(r["mean_f_reduced"] * r["reduced"] for r in per_page if r["reduced"]) / n_red) if n_red else None,
            "min_f": min(mins) if mins else None}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def phase_EVAL(args, cfg, device, out: Path, root: Path) -> dict:
    """Évaluation UNIQUE. Seul endroit où les pages mf_0060..mf_0069 sont générées puis lues (jamais en fumée : pages leurres)."""
    out = Path(out)
    if (out / "ld3_report.json").exists():
        raise SystemExit("refusé : le rapport existe déjà, aucune nouvelle soumission")
    if (out / EVAL_STARTED).exists():
        raise SystemExit(f"refusé : {EVAL_STARTED} existe, l'évaluation a déjà démarré ; une nouvelle EVAL exige une décision "
                         "explicite de pA (le fichier n'est jamais supprimé automatiquement)")
    if args.decoy_reference and not args.smoke:
        raise SystemExit("refusé : --decoy-reference n'existe qu'en fumée")
    mode = "decoy" if args.smoke else "real"
    ctx = make_context(args, cfg, device, decoy_b3=args.decoy_reference)
    try:                                                              # marqueur créé AVANT la génération et la première page
        with open(out / EVAL_STARTED, "x") as f:
            json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode,
                       "checkpoint_sha256": ctx["ident"]["checkpoint_sha256"]}, f, indent=1)
    except FileExistsError:
        raise SystemExit(f"refusé : {EVAL_STARTED} existe, une nouvelle EVAL exige une décision explicite de pA") from None
    work = Path(args.work)
    pages_root = work / "ld3pages"
    ref_e2e4 = json.loads(Path(args.e2e4_report).read_text()) if args.e2e4_report else None
    try:
        gen = D3.produce_pages(mode, cfg, Path(args.gen_src), work, pages_root, called_from_eval_ld3=True,
                               n_pages=cfg["pages"]["decoy_n"] if mode == "decoy" else None, mf_dir=root, reference=ref_e2e4,
                               extra_libs=args.extra_libs, timeout_s=cfg["robustness"]["timeouts_s"]["generate"])
    except D3.GenerationFailed:                                       # échec ou délai du générateur : INCOMPLETE, fins des journaux
        if (work / "data3_generate_failure.json").is_file():
            shutil.copy2(work / "data3_generate_failure.json", out / "data3_generate_failure.json")
        raise
    for name in ("data3_generation_manifest.json", "data3_overlap_report.json"):          # petits JSON, rapatriables
        if (work / name).is_file():
            shutil.copy2(work / name, out / name)
    if gen["overlap"] is not None and gen["overlap"]["blocking"]:
        raise SystemExit(f"refusé : recouvrement bloquant ({gen['overlap']['status']}) ; aucune page n'est évaluée")
    ids = check_eval_ids(gen["names"], mode, cfg)
    ctx["pages_root"] = pages_root
    rows, partial = [], out / "ld3_eval_pages.jsonl"
    for pid in ids:
        rows.append(evaluate_page_ld3(pid, pages_root, ctx))
        with partial.open("a") as f:
            f.write(json.dumps(rows[-1], default=str) + "\n")
    gen_summary = {k: gen[k] for k in ("mode", "seed", "names", "files_sha256", "generator", "seconds_total")}
    gen_summary["overlap"] = {k: v for k, v in (gen["overlap"] or {}).items() if k != "per_page"}
    report = build_report(rows, ids, cfg, gen_summary, ctx["ident"], bool(args.smoke), run_identity(cfg))
    C2.write_json(out / "ld3_report.json", report)
    C2.write_json(out / "ld3_edge_table.json", {"pages": ids, "edge_summary": report["edge_summary"],
                                                "edge_totals": report["edge_totals"],
                                                "per_page": {r["page"]: {a: {s: r[a][s]["edge"] for s in SOURCES} for a in arms(cfg)}
                                                             for r in rows}})
    return {"principal": report["principal"]["verdict"], "secondary": report["secondary"]["verdict"]}


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
    ap.add_argument("--cache-dir", default="/private/tmp/ld3_hf_cache")
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
            os.environ["HF_HUB_OFFLINE"] = "1"            # mesure hors ligne, avant tout import de huggingface_hub
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
        exits = out.parent / "exits"                     # sorties d'étape hors de RUN
        exits.mkdir(parents=True, exist_ok=True)
        (exits / f"ld3_run_exit_{args.phase}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: rec[k] for k in ("phase", "status")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
