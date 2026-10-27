"""LD-1 : LINE-DETR, entraînement de contrôle Z-GT sans CTC (docs/END_TO_END_LD1_LOT.txt §1-§5, §10, §12).

Phases (étapes argv de execute_pipeline, chacune dans son sous-processus) :
  D  téléchargement du détecteur PP (LD-0 gel 6, sha contrôlés), séparé de la mesure (§12 (5)).
  T  chronométrage et choix de la taille de lot (8, 4, 2 ; §12 (13)) ; S = min(20 000, floor(4 200 s / s_par_pas)) ;
     le manifeste (taille de lot, S, répartition des paramètres par bloc) est écrit AVANT E.
  E  entraînement de contrôle : journal toutes les 100 étapes ; checkpoints ; trois derniers gardés (§12 (17)).
  V  évaluation descriptive sur mf_0045-0049 (ou mf_0000 en fumée) : sorties calculées UNE fois par zone, seuils
     appliqués ensuite (§12 (7)) ; Z-GT et Z-PP-blocs (LD-0 gel 6) ; diagnostic de features (fumée).
Le DERNIER checkpoint est évalué (§10 (3)). Aucune page mf_0050-0059 n'est nommée, listée, ouverte ni hachée.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from babaorum.experimental.line_detr import (  # noqa: E402
    CANVAS_H, CANVAS_W, H_T, K, N_QUERIES, LineDetr, load_enc_det_from_e2e5, masked_encoder, match_zone, nms_lines,
    page_to_zone, params_by_block, zone_canvas, zone_mask, zone_to_page)
from babaorum.experimental.stage_s_data import load_mf_page  # noqa: E402
from babaorum.experimental.stage_s_model import resample_k  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

CONFIG = ROOT / "config" / "ld1_line_detr.json"
ZONE_CATEGORIES = ("texte", "titre", "annonce")
TRAIN_PAGES = [f"mf_{i:04d}" for i in range(0, 45)]      # mf_0000-0044 (exact, §12 (11))
HELD_PAGES = [f"mf_{i:04d}" for i in range(45, 50)]      # mf_0045-0049 (exact)
FORBIDDEN = {f"mf_{i:04d}" for i in range(50, 60)}


def guard_pages(pages) -> list:
    bad = [p for p in pages if p in FORBIDDEN or p.startswith("mf_005")]
    if bad:
        raise SystemExit("refusé : pages réservées (mf_0050-0059) demandées")
    return list(pages)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_cuda(device: str) -> None:
    """§12 (2) : le run complet exige CUDA ; la fumée reste sur cpu."""
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("refusé : --device cuda sans CUDA disponible")


# ---------- données ----------

def load_scaled(root: Path, pid: str, target_h: float = H_T):
    page = load_mf_page(root / "pages" / f"{pid}.json", root / "images" / f"{pid}.png")
    heights = [ln["normal_negative_px"] + ln["normal_positive_px"] for ln in page["lines"]]
    factor = float(target_h / np.median(heights))
    img = page["image"]
    h, w = img.shape
    nw, nh = round(w * factor), round(h * factor)
    scaled = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    raw = json.loads((root / "pages" / f"{pid}.json").read_text())
    lines_by_id = {ln["id"]: ln for ln in page["lines"]}
    blocks = []
    for blk in raw["blocks"]:
        if blk["category"] not in ZONE_CATEGORIES or not blk.get("line_ids"):
            continue
        xs = [q[0] for q in blk["polygon"]]
        ys = [q[1] for q in blk["polygon"]]
        box = (min(xs) * factor, min(ys) * factor, max(xs) * factor, max(ys) * factor)
        lines = []
        for lid in blk["line_ids"]:
            ln = lines_by_id[lid]
            lines.append({"id": lid, "baseline": np.asarray(ln["baseline"], dtype=np.float64) * factor,
                          "neg": ln["normal_negative_px"] * factor, "pos": ln["normal_positive_px"] * factor})
        blocks.append({"id": blk["id"], "category": blk["category"], "box": box, "lines": lines})
    gt_lines = [{"id": ln["id"], "baseline": np.asarray(ln["baseline"], dtype=np.float64) * factor,
                 "neg": ln["normal_negative_px"] * factor, "pos": ln["normal_positive_px"] * factor}
                for ln in page["lines"]]
    return {"page_id": pid, "image": scaled, "factor": factor, "blocks": blocks, "gt_lines": gt_lines,
            "scaled_wh": (nw, nh), "native_wh": (w, h), "truncated": 0}


def perturb_box(box, h: float, width: float, rng: np.random.Generator):
    """±0,5 h sur les bords verticaux, ±5 % de la largeur sur les bords horizontaux (§1)."""
    x0, y0, x1, y1 = box
    dw = 0.05 * width
    return (x0 + rng.uniform(-dw, dw), y0 + rng.uniform(-0.5 * h, 0.5 * h),
            x1 + rng.uniform(-dw, dw), y1 + rng.uniform(-0.5 * h, 0.5 * h))


def zone_sample(page: dict, block: dict, box):
    canvas, meta = zone_canvas(page["image"], box, h_page=H_T)
    scale = 0.5 * (meta.fx + meta.fy)
    pts, h_canvas, neg, pos = [], [], [], []
    for ln in block["lines"]:
        if len(ln["baseline"]) < 2:
            continue
        try:
            pts.append(page_to_zone(resample_k(ln["baseline"], K), meta))
        except ValueError:                              # segment nul : ligne écartée
            continue
        h_canvas.append((ln["neg"] + ln["pos"]) * scale)
        neg.append(ln["neg"])
        pos.append(ln["pos"])
    if not pts:
        return None
    truncated = 0
    if len(pts) > N_QUERIES:                            # §12 (14) : troncature à N, comptée
        truncated = len(pts) - N_QUERIES
        pts, h_canvas, neg, pos = pts[:N_QUERIES], h_canvas[:N_QUERIES], neg[:N_QUERIES], pos[:N_QUERIES]
    reduced = bool(meta.fx < 1.0 or meta.fy < 1.0)      # §12 (9) : zone réduite
    return {"canvas": canvas, "mask": zone_mask(meta), "meta": meta, "pts": np.stack(pts).astype(np.float64),
            "h": np.asarray(h_canvas, dtype=np.float64), "neg": np.asarray(neg), "pos": np.asarray(pos),
            "reduced": reduced, "truncated": truncated, "factor_xy": (meta.fx, meta.fy)}


# ---------- pertes (§3) ----------

def focal_loss(logit, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return a_t * (1 - p_t) ** gamma * ce


def layer_loss(out_l: dict, b: int, sample: dict, matches: list):
    logit = out_l["logit"][b]
    tgt = torch.zeros_like(logit)
    q_idx = torch.tensor([q for _, q in matches], dtype=torch.long, device=logit.device)
    g_idx = [g for g, _ in matches]
    n_pos = max(1, len(matches))
    if len(matches):
        tgt[q_idx] = 1.0
    l_cls = focal_loss(logit, tgt).sum() / n_pos
    if not len(matches):
        return l_cls + out_l["pts"].sum() * 0.0, l_cls, 0.0, 0.0
    pts_pred = out_l["pts"][b][q_idx]
    gt_norm = torch.as_tensor(sample["pts"][g_idx] / np.array([CANVAS_W, CANVAS_H]), dtype=pts_pred.dtype,
                              device=pts_pred.device)
    l_pts = (pts_pred - gt_norm).abs().mean()
    neg_p, pos_p = out_l["neg"][b][q_idx], out_l["pos"][b][q_idx]
    neg_t = torch.as_tensor(sample["neg"][g_idx], dtype=neg_p.dtype, device=neg_p.device)
    pos_t = torch.as_tensor(sample["pos"][g_idx], dtype=pos_p.dtype, device=pos_p.device)
    l_h = ((neg_p - neg_t).abs() + (pos_p - pos_t).abs()).mean() / H_T
    return l_cls + l_pts + l_h, l_cls, float(l_pts), float(l_h)


def batch_loss(model, samples: list, device):
    canv = torch.as_tensor(np.stack([s["canvas"] for s in samples]))[:, None].to(device)
    msk = torch.as_tensor(np.stack([s["mask"] for s in samples]))[:, None].to(device)
    outs = model(canv, msk)
    total = 0.0
    parts = {"cls": 0.0, "pts": 0.0, "h": 0.0}
    matched_frac = []
    n_terms = len(samples) * len(outs)
    for b, s in enumerate(samples):
        for out_l in outs:
            with torch.no_grad():
                pred_px = out_l["pts"][b].detach().cpu().numpy() * np.array([CANVAS_W, CANVAS_H])
                matches = match_zone(out_l["logit"][b].detach().cpu().numpy(), pred_px, s["pts"], s["h"])
            l, lc, lp, lh = layer_loss(out_l, b, s, matches)
            total = total + l
            parts["cls"] += float(lc)
            parts["pts"] += lp
            parts["h"] += lh
            matched_frac.append(len(matches) / N_QUERIES)
    loss = total / n_terms
    return loss, {k: v / n_terms for k, v in parts.items()}, float(np.mean(matched_frac))


# ---------- inférence : sorties une fois, seuils ensuite (§12 (7)) ----------

@torch.no_grad()
def raw_zone_outputs(model, canvas, mask, device) -> dict:
    outs = model(torch.as_tensor(canvas)[None, None].to(device), torch.as_tensor(mask)[None, None].to(device))
    last = outs[-1]
    return {"prob": torch.sigmoid(last["logit"][0]).cpu().numpy(),
            "pts": last["pts"][0].cpu().numpy() * np.array([CANVAS_W, CANVAS_H]),
            "neg": last["neg"][0].cpu().numpy(), "pos": last["pos"][0].cpu().numpy()}


def candidates_for_threshold(zones_raw: list, thr: float) -> list:
    """zones_raw : liste de (zone_id, meta, sorties brutes). Retourne les lignes NMS dans le repère de la page."""
    preds = []
    for zid, meta, raw in zones_raw:
        for q in np.where(raw["prob"] >= thr)[0]:
            page_pts = zone_to_page(raw["pts"][q], meta)
            preds.append({"score": float(raw["prob"][q]), "zone": zid, "query": int(q), "pts": page_pts,
                          "h": float(raw["neg"][q] + raw["pos"][q]), "neg": float(raw["neg"][q]),
                          "pos": float(raw["pos"][q])})
    return nms_lines(preds)


def score_page(page: dict, kept: list) -> dict:
    gt = [GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]),
                          baseline_source="ground_truth", geometry_status="available") for g in page["gt_lines"]]
    if not kept:
        return {"n_pred": 0, "n_gt": len(gt), "precision": 0.0, "recall": 0.0, "precision_none": False,
                "splits": 0, "merges": 0}
    pred = [PredictedLine(id=f"p{i}", baseline=k["pts"].tolist(), baseline_source="predicted",
                          geometry_status="available") for i, k in enumerate(kept)]
    res = evaluate_line_geometry(gt, pred)
    return {"n_pred": len(pred), "n_gt": len(gt),
            "precision": None if res.precision is None else float(res.precision),
            "recall": float(res.recall or 0.0), "precision_none": res.precision is None,
            "splits": int(res.split_candidate_gt_count or 0), "merges": int(res.merge_candidate_pred_count or 0)}


def mean_pr(rows: list) -> dict:
    """Moyennes non pondérées par page. Une précision None n'est PAS remplacée par 0 : elle est comptée à part (§12 (16))."""
    prec = [r["precision"] for r in rows if r["precision"] is not None]
    P = float(np.mean(prec)) if prec else 0.0
    R = float(np.mean([r["recall"] for r in rows]))
    return {"P": P, "R": R, "precision_none_pages": int(sum(1 for r in rows if r["precision"] is None)),
            "pages_without_prediction": int(sum(1 for r in rows if r.get("n_pred", 1) == 0)),
            "pages": len(rows)}


def f1(P: float, R: float) -> float:
    return 0.0 if P + R == 0 else 2 * P * R / (P + R)


def choose_threshold(grid_rows: dict) -> float:
    best = None
    for thr in sorted(grid_rows):
        P, R = grid_rows[thr]
        score = f1(P, R)
        if best is None or score > best[1] + 1e-12:
            best = (thr, score)
    return best[0]


def reading(P: float, R: float) -> str:
    m = min(P, R)
    if m >= 0.80:
        return "CONVERGE"
    if m >= 0.50:
        return "PARTIEL"
    return "ECHEC"


# ---------- Z-PP-blocs (LD-0 gel 6, §12 (5), §13 (N2)) ----------

L0_DETECTOR = None


def load_l0(cfg: dict):
    """Charge LD-0 gel 6 : dépendances (_load_deps) et environnement HF posés AVANT tout import de huggingface_hub."""
    global L0_DETECTOR
    import ld0_zone_coverage as L0
    L0._load_deps()
    os.environ.update(L0.HF_ENV)
    L0_DETECTOR = dict(cfg["detector"])
    return L0


def pp_zones(page: dict, root: Path, pid: str, detector_cache: Path, L0, det_fn=None) -> list:
    """Zones PP-DocLayoutV3 sur l'image native, puis mises à l'échelle (factor) : Z-PP-blocs, comme LD-0 gel 6.
    det_fn(native_rgb_array) -> det_raw : injectable (test) ; par défaut le détecteur réel de LD-0."""
    native = np.asarray(Image.open(root / "images" / f"{pid}.png").convert("RGB"))
    H, W = native.shape[:2]
    if det_fn is None:
        proc, model, _ = L0.load_detector(L0_DETECTOR, detector_cache)
        det_fn = lambda arr: L0.detect(proc, model, Image.fromarray(arr), L0_DETECTOR["floor"])  # noqa: E731
    det_raw = det_fn(native)
    det = L0.normalize_blocks(det_raw, image_size=(W, H))
    kept = L0.select_blocks(det["blocks"], L0.Params(conf=0.3, dedup=0.8))
    kept = [b for b in kept if L0.is_read(b.get("label"))]
    boxes = [L0.bbox([(float(x), float(y)) for x, y in b["polygon"]]) for b in kept]
    nw, nh = page["scaled_wh"]
    sx, sy = nw / W, nh / H
    return [(i, tuple(map(float, L0.scale_box(b, sx, sy)))) for i, b in enumerate(boxes)]


def encoder_diagnostic(model, page: dict, device) -> dict:
    """§13 (6) : alignement exact. Les features de la zone et celles de la page sont comparées au MÊME pixel de page :
    l'image de page est décalée de (ix0 mod 4, iy0 mod 4) pour que la grille stride 4 coïncide avec le canevas.
    Les zones touchant le bord de la page sont exclues (nombre rapporté). Descriptif."""
    bins = {"0-8": [], "8-16": [], "16-32": [], ">32": []}
    excluded_border = 0
    H, W = page["image"].shape
    shifted_cache = {}
    with torch.no_grad():
        for blk in page["blocks"][:8]:
            canvas, meta = zone_canvas(page["image"], blk["box"], h_page=H_T)
            if meta.fx != 1.0 or meta.fy != 1.0:
                continue
            if meta.ix0 < 0 or meta.iy0 < 0 or meta.ix0 + meta.crop_w > W or meta.iy0 + meta.crop_h > H:
                excluded_border += 1
                continue
            ox, oy = meta.ix0 % 4, meta.iy0 % 4
            if (ox, oy) not in shifted_cache:
                sub = page["image"][oy:, ox:]
                shifted_cache[(ox, oy)] = model.enc_det(torch.as_tensor(sub)[None, None].to(device))[0]
            f_page = shifted_cache[(ox, oy)]
            jb, ib = (meta.iy0 - oy) // 4, (meta.ix0 - ox) // 4
            m = zone_mask(meta)
            f_zone, _ = masked_encoder(model.enc_det, torch.as_tensor(canvas)[None, None].to(device),
                                       torch.as_tensor(m)[None, None].to(device))
            f_zone = f_zone[0]
            nh4, nw4 = int(round(meta.crop_h / 4)), int(round(meta.crop_w / 4))
            for j in range(min(nh4, f_zone.shape[1], f_page.shape[1] - jb)):
                for i in range(min(nw4, f_zone.shape[2], f_page.shape[2] - ib)):
                    dist_px = 4 * min(i, nw4 - 1 - i, j, nh4 - 1 - j)
                    key = "0-8" if dist_px < 8 else "8-16" if dist_px < 16 else "16-32" if dist_px < 32 else ">32"
                    d = float((f_zone[:, j, i] - f_page[:, jb + j, ib + i]).abs().mean())
                    bins[key].append(d)
    return {"excluded_zones_touching_page_border": excluded_border,
            "bins": {k: {"n": len(v), "mean_abs_diff": float(np.mean(v)) if v else None} for k, v in bins.items()}}


# ---------- phases ----------

def set_seeds(seed: int) -> None:
    """§15 (2) : graines (torch, numpy, random) posées AVANT toute construction de modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def require_empty_output(out) -> None:
    """§15 (3) : reprise interdite, la sortie doit être vide au départ."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusé : sortie non vide ({out}), reprise interdite")


def build_model(args, cfg, device):
    model = LineDetr().to(device)
    if sha256(args.init_checkpoint) != cfg["init"]["sha256_full"]:
        raise SystemExit("refusé : checkpoint d'initialisation de sha différent")
    ck = torch.load(args.init_checkpoint, map_location="cpu")
    load_enc_det_from_e2e5(model, ck["model"], expected_step=cfg["init"]["step"], step=ck["step"])
    enc = list(model.enc_det.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("enc_det.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr_encoder"]}, {"params": other, "lr": cfg["lr"]}],
                            weight_decay=cfg["weight_decay"])
    return model, opt


def sample_batch(pages, zone_index, rng, bs: int):
    samples = []
    picks = rng.choice(len(zone_index), size=bs, replace=True)
    for k in picks:
        pi, bi = zone_index[int(k)]
        page = pages[pi]
        blk = page["blocks"][bi]
        w = blk["box"][2] - blk["box"][0]
        s = zone_sample(page, blk, perturb_box(blk["box"], H_T, w, rng))
        if s is not None:
            samples.append(s)
    return samples


class DeviceError(RuntimeError):
    """§15 (1) : erreur de device au contrôle de T (statut FAILED à l'étape T, avec la cause)."""


class RepeatedOOM(RuntimeError):
    """§15 (1) : aucune taille de lot ne tient (statut INCOMPLETE)."""


def is_oom(exc) -> bool:
    """§15 (1) : seul un vrai dépassement de mémoire est un repli de lot ; toute autre erreur remonte."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc))


def device_control(model, opt, pages, zone_index, rng, device, cfg):
    """§15 (1) : au début de T, 2 pas de contrôle sur le vrai device (taille de lot la plus petite)."""
    saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    try:
        for _ in range(2):
            s = sample_batch(pages, zone_index, rng, cfg["batch_candidates"][-1])
            loss, _, _ = batch_loss(model, s, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
    except Exception as exc:
        if is_oom(exc):
            raise
        raise DeviceError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        model.load_state_dict(saved_model)
        opt.load_state_dict(saved_opt)


def choose_batch(model, opt, pages, zone_index, rng, device, cfg):
    """§12 (13) : essais 8, 4, 2 ; la plus grande qui tient, avec une marge de 20 % (mémoire CUDA)."""
    tried = []
    for bs in cfg["batch_candidates"]:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            s = sample_batch(pages, zone_index, rng, bs)
            loss, _, _ = batch_loss(model, s, device)
            opt.zero_grad()
            loss.backward()
            opt.zero_grad()
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            total = int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None
            fits = True if peak is None else peak * 1.2 <= total
            tried.append({"batch_size": bs, "peak_bytes": peak, "total_bytes": total, "fits": fits})
            if fits:
                return bs, tried
        except Exception as exc:                        # dépassement mémoire : bs suivant (§13 (7)) ; le reste remonte
            if not is_oom(exc):
                raise
            tried.append({"batch_size": bs, "error": str(exc)[:200], "fits": False})
            loss = None
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RepeatedOOM("aucune taille de lot ne tient")


def phase_D(args, cfg):
    L0 = load_l0(cfg)
    local = L0.download_detector(cfg["detector"], Path(args.cache_dir))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exits = out.parent / "exits"                         # §17 (2) : RUN reste vide
    exits.mkdir(parents=True, exist_ok=True)
    (exits / "ld1_download.json").write_text(json.dumps({k: str(v.name) for k, v in local.items()}, indent=1))
    return {"download": "ok", "files": sorted(local)}


def phase_T(args, cfg, device, out: Path, pages, zone_index, rng, model, opt):
    device_control(model, opt, pages, zone_index, rng, device, cfg)
    bs, tried = choose_batch(model, opt, pages, zone_index, rng, device, cfg)
    steps = args.timing_steps or cfg["timing_steps"]
    t1 = time.monotonic()
    for _ in range(steps):
        s = sample_batch(pages, zone_index, rng, bs)
        loss, _, _ = batch_loss(model, s, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    per = (time.monotonic() - t1) / steps
    S = min(args.max_steps or cfg["max_steps"], math.floor(cfg["train_budget_s"] / per))
    manifest = {"batch_size": bs, "batch_tries": tried, "s_per_step": per, "steps": S,
                "budget_s": cfg["train_budget_s"], "timing_steps": steps,
                "max_steps": args.max_steps or cfg["max_steps"],
                "fumee": bool(args.pages or args.timing_steps or args.max_steps),
                "init_checkpoint_sha256": sha256(args.init_checkpoint), "device": str(device),
                "params_by_block": params_by_block(model), "written_before_E": True}
    if device.type == "cuda":
        manifest["max_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    (out / "ld1_timing_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def phase_E(args, cfg, device, out: Path, pages, zone_index, rng, model, opt, manifest):
    S = manifest["steps"]
    bs = manifest["batch_size"]
    identity = {"seed": cfg["seed"], "steps_total": S, "batch_size": bs, "init_sha256": manifest["init_checkpoint_sha256"]}
    start = 0
    prev = sorted(out.glob("ld1_step*.pt"))
    if prev:                                              # reprise : identité identique, sinon refus (§4)
        ck = torch.load(prev[-1], map_location="cpu")
        if ck.get("identity") != identity:
            raise SystemExit("refusé : reprise avec une autre identité de lancement")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"]
        rng.bit_generator.state = ck["rng_state"]
    log = out / "ld1_train_log.jsonl"
    reduced_zones = truncated = total_zones = 0
    factors = []                                          # facteur moyen des zones réduites (§12 (9))
    for step in range(start + 1, S + 1):
        s = sample_batch(pages, zone_index, rng, bs)
        reduced_zones += sum(1 for z in s if z["reduced"])
        factors += [min(z["factor_xy"]) for z in s if z["reduced"]]
        total_zones += len(s)
        truncated += sum(z["truncated"] for z in s)
        loss, parts, matched = batch_loss(model, s, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == S:
            row = {"step": step, "loss": float(loss), "focal": parts["cls"], "L1_points": parts["pts"],
                   "L1_heights": parts["h"], "matched_query_share": matched,
                   "reduced_zone_share": reduced_zones / max(1, total_zones), "truncated_targets": truncated,
                   "reduced_zone_mean_factor": float(np.mean(factors)) if factors else None}
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == S:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                        "rng_state": rng.bit_generator.state, "identity": identity},
                       out / f"ld1_step{step:06d}.pt")
            for old in sorted(out.glob("ld1_step*.pt"))[:-3]:   # §12 (17) : trois derniers checkpoints
                old.unlink()
    return {"steps": S, "reduced_zone_share": reduced_zones / max(1, total_zones), "truncated_targets": truncated,
            "reduced_zone_mean_factor": float(np.mean(factors)) if factors else None}


def phase_V(args, cfg, device, out: Path, root: Path, model, pages_list, L0):
    S = json.loads((out / "ld1_timing_manifest.json").read_text())["steps"]      # §15 (3) : S lu dans le manifeste de T
    ckpts = [c for c in out.glob("ld1_step*.pt") if c.name == f"ld1_step{S:06d}.pt"]
    if len(ckpts) != 1:
        raise SystemExit(f"refusé : checkpoint ld1_step{S:06d}.pt absent ou non unique")
    last = ckpts[-1]
    model.load_state_dict(torch.load(last, map_location="cpu")["model"])
    model.eval()
    thresholds = cfg["threshold_grid"]
    rows = {"Z-GT": {t: [] for t in thresholds}, "Z-PP-blocs": {t: [] for t in thresholds}}
    none_counts = {"Z-GT": 0, "Z-PP-blocs": 0}
    diagnostics = {}
    cache = Path(args.cache_dir)
    for pid in guard_pages(pages_list):
        page = load_scaled(root, pid)
        gz = [(i, b["box"]) for i, b in enumerate(page["blocks"])]
        raw_gt = []
        for zid, box in gz:
            canvas, meta = zone_canvas(page["image"], box, h_page=H_T)
            raw_gt.append((zid, meta, raw_zone_outputs(model, canvas, zone_mask(meta), device)))
        pp_list = pp_zones(page, root, pid, cache, L0) if cfg.get("evaluate_pp", True) else []
        raw_pp = []
        for zid, box in pp_list:
            canvas, meta = zone_canvas(page["image"], box, h_page=H_T)
            raw_pp.append((zid, meta, raw_zone_outputs(model, canvas, zone_mask(meta), device)))
        if args.diagnostic:
            diagnostics[pid] = encoder_diagnostic(model, page, device)
        for thr in thresholds:
            r_gt = score_page(page, candidates_for_threshold(raw_gt, thr))
            r_pp = score_page(page, candidates_for_threshold(raw_pp, thr))
            for name, r in (("Z-GT", r_gt), ("Z-PP-blocs", r_pp)):
                if r["precision_none"]:
                    none_counts[name] += 1
                rows[name][thr].append(dict(r, page=pid))
    report = {"schema": "ld1_report_v1", "checkpoint": last.name, "checkpoint_sha256": sha256(last), "pages": list(pages_list),
              "thresholds": thresholds, "none_precision_counts": none_counts,
              "reserves": ["encodeur de E2E-5 a vu mf_0045-0049 (décodeur seul tenu à l'écart)",
                           "échelle oracle : factor = 32 / médiane de la GT (même limite que la référence)",
                           "plafond réel de couverture de LD-0R : ~0,93 (pire page 0,76), donc Z-PP plafonné"],
              "encoder_diagnostic": diagnostics}
    grid_gt, grid_pp = {}, {}
    for thr in thresholds:
        grid_gt[thr] = mean_pr(rows["Z-GT"][thr])
        grid_pp[thr] = mean_pr(rows["Z-PP-blocs"][thr])
    P, R = grid_gt[0.5]["P"], grid_gt[0.5]["R"]
    report.update({"Z-GT_at_0_5": grid_gt[0.5], "Z-PP-blocs_at_0_5": grid_pp.get(0.5),
                   "reading_Z-GT_0_5": reading(P, R),
                   "threshold_for_LD2": choose_threshold({t: (g["P"], g["R"]) for t, g in grid_gt.items()}),
                   "grid_Z-GT": {str(t): dict(g, F1=f1(g["P"], g["R"])) for t, g in grid_gt.items()},
                   "grid_Z-PP-blocs": {str(t): g for t, g in grid_pp.items()},
                   "per_page_Z-GT_0_5": rows["Z-GT"][0.5], "per_page_Z-PP-blocs_0_5": rows["Z-PP-blocs"][0.5]})
    (out / "ld1_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main(argv=None) -> int:
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["D", "T", "E", "V"], required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--cache-dir", default="/private/tmp/ld1_hf_cache")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--timing-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--diagnostic", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"phase": args.phase, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "RUNNING"}
    t0 = time.monotonic()
    try:
        if args.phase == "D":
            if args.device == "cuda":
                check_cuda("cuda")
            rec.update(status="COMPLETE", **phase_D(args, cfg))
            return 0
        if args.phase == "V":
            os.environ["HF_HUB_OFFLINE"] = "1"            # §12 (5) : mesure hors ligne, poids déjà téléchargés
        if args.phase == "T":
            require_empty_output(out)
        check_cuda(args.device)
        device = torch.device(args.device)
        root = Path(args.root)
        set_seeds(cfg["seed"])
        model, opt = build_model(args, cfg, device)
        L0 = load_l0(cfg) if args.phase == "V" else None
        pages_list = args.pages or (HELD_PAGES if args.phase == "V" else TRAIN_PAGES)
        rng = np.random.default_rng(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if args.phase in ("T", "E"):
            pages = [load_scaled(root, pid) for pid in guard_pages(pages_list)]
            zone_index = [(pi, bi) for pi, p in enumerate(pages) for bi in range(len(p["blocks"]))]
        if args.phase == "T":
            rec.update(status="COMPLETE", manifest=phase_T(args, cfg, device, out, pages, zone_index, rng, model, opt))
        elif args.phase == "E":
            manifest = json.loads((out / "ld1_timing_manifest.json").read_text())
            rec.update(status="COMPLETE", **phase_E(args, cfg, device, out, pages, zone_index, rng, model, opt, manifest))
        else:
            rep = phase_V(args, cfg, device, out, root, model, pages_list, L0)
            rec.update(status="COMPLETE", reading_Z_GT_0_5=rep["reading_Z-GT_0_5"],
                       threshold_for_LD2=rep["threshold_for_LD2"])
    except SystemExit:
        rec.update(status="FAILED")
        raise
    except DeviceError as exc:
        rec.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        raise
    except Exception as exc:  # noqa: BLE001
        rec.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        raise
    finally:
        rec["duration_s"] = round(time.monotonic() - t0, 1)
        exits = out.parent / "exits"                     # §17 (2) : sorties d'étape hors de RUN
        exits.mkdir(parents=True, exist_ok=True)
        (exits / f"ld1_run_exit_{args.phase}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: rec[k] for k in ("phase", "status")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
