"""LD-2 : LINE-DETR complet (géométrie + CTC, λ = 1/6) et évaluation UNIQUE pré-enregistrée (docs/END_TO_END_LD2_LOT.txt §0-§7).

Phases (étapes argv de execute_pipeline, chacune dans son sous-processus ; TW et EW sont enveloppées par le carnet : leur code de
sortie va dans un fichier et l'étape rend toujours 0, lot §8 (2)) :
  D       téléchargement du détecteur PP (comme LD-1), RUN reste vide.
  T       200 pas chronométrés du modèle principal (avec CTC), puis 200 pas du témoin sans zones ; S2 = min(20 000,
          floor(8 400 / s_par_pas)) et S_t = min(S2, floor(4 200 / s_par_pas_témoin)) sont écrits dans le manifeste AVANT
          tout entraînement ; sortie vide exigée.
  E       entraînement principal jusqu'à S2 (plafond 9 000 s ; INCOMPLETE si S2 n'est pas atteint : aucune évaluation).
  EW      entraînement du témoin jusqu'à S_t (plafond 4 800 s) ; échec ou délai rapportés tels quels (état dans
          ld2_witness_status.json), sans relance ; le processus sort toujours avec le code 0.
  FREEZE  seuil 0,6, NMS et checkpoint (le dernier, pas S2) écrits dans le manifeste ; rien ne dépend de mf_0050-0059.
  CAL     étalonnage du temps d'une page sur UNE page d'ENTRAÎNEMENT (jamais une page d'évaluation).
  EVAL    le SEUL endroit où les chemins des pages d'évaluation sont construits, après FREEZE : critère (a)+(b) en
          Z-PP-blocs, Z-GT et témoin descriptifs, CER du texte brut des lignes appariées.
Aucune autre phase ne nomme, ne liste, n'ouvre ni ne hache les pages d'évaluation. En fumée (--smoke), EVAL ne construit
jamais ces chemins : il évalue des pages de remplacement données par --eval-pages.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import functools
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import ld2_core as C2  # noqa: E402
import train_line_detr as TL  # noqa: E402
from train_stage_s import encode_text, greedy_decode, levenshtein, load_scaled_page, lines_in_tile, sigma_for_step  # noqa: E402
from train_stage_s_v2 import draw_window  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402
from babaorum.experimental.line_detr import CANVAS_H, CANVAS_W, H_T, K, LineDetr, zone_canvas, zone_mask  # noqa: E402
from babaorum.experimental.stage_s_arms import ROWS, GradScale  # noqa: E402
from babaorum.experimental.stage_s_data import build_alphabet, load_mf_page  # noqa: E402
from babaorum.experimental.stage_s_model import resample_k  # noqa: E402
from babaorum.experimental.stage_s_reader import Reader, band_features, expected_ctc_loss  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

LAMBDA = 1.0 / 6.0
TRAIN_ALL = TL.TRAIN_PAGES + TL.HELD_PAGES          # mf_0000-0049 : mf_0045-0049 réintégrées (conception §10 (13))
SIGMA_TIMING = H_T                                    # sigma de départ (1 h) pour le chronométrage
MIN_CHORD_PX = 1.0                                    # corde minimale d'une baseline prédite pour la CTC
EVAL_STARTED = "ld2_eval_started.json"
PHASES = ["D", "T", "TW", "E", "EW", "FREEZE", "CAL", "EVAL"]


# ---------- configuration ----------

def load_all(path=C2.CONFIG) -> dict:
    """config/ld2.json, plus les clés reprises de la config de LD-1 et la config d'E2E-5 (sha contrôlés)."""
    cfg = C2.load_config(path)
    inh = cfg["inherit_ld1"]
    if C2.sha256_file(ROOT / inh["path"]) != inh["sha256"]:
        raise SystemExit("refusé : la config de LD-1 diffère de celle pré-enregistrée")
    ld1 = json.loads((ROOT / inh["path"]).read_text())
    cfg["ld1"] = {k: ld1[k] for k in inh["keys"]}
    st = cfg["stage_s"]
    if not C2.sha256_file(ROOT / st["config"]).startswith(st["sha256_prefix"]):
        raise SystemExit("refusé : la config d'E2E-5 diffère de celle pré-enregistrée")
    cfg["stage"] = json.loads((ROOT / st["config"]).read_text())
    return cfg


# ---------- données ----------

def load_page_text(root: Path, pid: str) -> dict:
    """Page mise à l'échelle de LD-1 (image, blocs GT, lignes GT) avec le texte et la lisibilité de chaque ligne ; 'lines' au
    schéma d'E2E-5 (tuiles du témoin). Même facteur d'échelle dans les deux chargeurs."""
    page = TL.load_scaled(Path(root), pid)
    s = load_scaled_page(Path(root), pid, H_T)
    if abs(s["factor"] - page["factor"]) > 1e-9:
        raise RuntimeError(f"{pid} : facteurs d'échelle différents")
    meta = {ln["id"]: (ln["text"], ln["legibility"]) for ln in s["lines"]}
    for ln in page["gt_lines"] + [ln for b in page["blocks"] for ln in b["lines"]]:
        ln["text"], ln["legibility"] = meta[ln["id"]]
    page["lines"] = s["lines"]
    return page


def _readable(text, legibility) -> bool:
    return legibility == "readable" and bool(text)


CTC_MARGIN_PX = 8.0                                   # marge de lines_in_tile (train_stage_s) : défaut margin=8.0


def band_extreme_points(pts_edge, neg: float, pos: float):
    """Points extrêmes de la bande CTC d'une baseline [K,2] (coordonnées de bords du canevas), avec EXACTEMENT les règles de
    stage_s_reader.band_features / joint_geometry.sample_baseline_bands (recopiées : elles exigent torch et une carte de features) :
    steps = max(4, round(arc / 4)) ; centres = bords - 0,5 interpolés linéairement (linspace sur les K points) ; tangente par
    différences centrées, unilatérale aux extrémités ; normale n = (-t_y, t_x) ; la bande va de -neg à +pos le long de n à CHAQUE
    pas. Retourne (bas, haut) [steps,2] en coordonnées de bords, ou None si la corde n'est pas strictement gauche-droite ou si une
    tangente est nulle (le sampler lève alors une ValueError)."""
    p = np.asarray(pts_edge, dtype=np.float64)
    if p[-1, 0] <= p[0, 0]:
        return None
    arc = float(np.linalg.norm(p[1:] - p[:-1], axis=-1).sum())
    steps = max(4, round(arc / 4.0))
    c = p - 0.5
    k = len(c)
    param = np.linspace(0.0, k - 1, steps)
    lo = np.floor(param).astype(int)
    hi = np.minimum(lo + 1, k - 1)
    alpha = (param - lo)[:, None]
    center = c[lo] * (1.0 - alpha) + c[hi] * alpha
    tan = np.empty_like(center)
    tan[0], tan[-1] = center[1] - center[0], center[-1] - center[-2]
    tan[1:-1] = center[2:] - center[:-2]
    norm = np.linalg.norm(tan, axis=-1, keepdims=True)
    if bool((norm <= np.finfo(np.float32).eps).any()):
        return None
    tan = tan / norm
    normal = np.stack((-tan[:, 1], tan[:, 0]), axis=-1)
    return center - neg * normal + 0.5, center + pos * normal + 0.5


def ctc_inside(pts_canvas, neg_c: float, pos_c: float, nw: float, nh: float, margin: float = CTC_MARGIN_PX) -> bool:
    """Ligne ENTIÈREMENT dans le support du canevas (conception : « entièrement dans la zone ») : points de la baseline ET points
    extrêmes de la bande (le long de la normale par pas, neg / pos en px canevas, voir band_extreme_points) dans
    [margin, nw - margin] x [margin, nh - margin]. Inéligible si la corde n'est pas strictement gauche-droite ou si une tangente
    est nulle. Avec margin = 2 et le canevas entier, équivaut à « band_features n'est pas None » (test d'équivalence)."""
    ext = band_extreme_points(pts_canvas, neg_c, pos_c)
    if ext is None:
        return False
    xy = np.concatenate([np.asarray(pts_canvas, dtype=np.float64), *ext])
    return bool(xy[:, 0].min() >= margin and xy[:, 0].max() <= nw - margin
                and xy[:, 1].min() >= margin and xy[:, 1].max() <= nh - margin)


def zone_sample_ctc(page: dict, block: dict, box, n_max: int = LD.N_QUERIES):
    """Comme train_line_detr.zone_sample, avec le texte et la lisibilité des lignes gardées (CTC). `scale` (px canevas / px page)
    sert à la seule bande CTC ; une ligne coupée par le recadrage garde sa supervision géométrique mais n'est plus lisible
    pour la CTC (readable = False)."""
    canvas, meta = zone_canvas(page["image"], box, h_page=H_T)
    scale = 0.5 * (meta.fx + meta.fy)
    nw, nh = round(meta.crop_w * meta.fx), round(meta.crop_h * meta.fy)
    pts, h_canvas, neg, pos, texts, readable, cut_by_crop = [], [], [], [], [], [], []
    for ln in block["lines"]:
        if len(ln["baseline"]) < 2:
            continue
        try:
            pts.append(LD.page_to_zone(resample_k(ln["baseline"], K), meta))
        except ValueError:                              # segment nul : ligne écartée
            continue
        h_canvas.append((ln["neg"] + ln["pos"]) * scale)
        neg.append(ln["neg"])
        pos.append(ln["pos"])
        texts.append(ln["text"])
        inside = ctc_inside(pts[-1], ln["neg"] * scale, ln["pos"] * scale, nw, nh)
        readable.append(_readable(ln["text"], ln["legibility"]) and inside)
        cut_by_crop.append(_readable(ln["text"], ln["legibility"]) and not inside)         # compteur de lecture seule
    if not pts:
        return None
    truncated = max(0, len(pts) - n_max)
    cut = slice(0, n_max)
    return {"canvas": canvas, "mask": zone_mask(meta), "meta": meta, "pts": np.stack(pts[cut]).astype(np.float64),
            "h": np.asarray(h_canvas[cut], dtype=np.float64), "neg": np.asarray(neg[cut]), "pos": np.asarray(pos[cut]),
            "texts": texts[cut], "readable": readable[cut], "truncated": truncated, "scale": scale,
            "ctc_cut": int(sum(cut_by_crop[cut]))}


def tile_sample(page: dict, rng: np.random.Generator, tile: int, centred: bool, n_max: int):
    """Témoin : tuile de l'étape S (fenêtre d'E2E-5, centrée sur une ligne lisible), lignes ENTIÈREMENT dedans."""
    x0, y0 = draw_window(page, rng, tile, centred)
    image = np.ascontiguousarray(page["image"][y0:y0 + tile, x0:x0 + tile])
    pts, h, neg, pos, texts, readable = [], [], [], [], [], []
    for ln in lines_in_tile(page["lines"], x0, y0, tile):
        try:
            pts.append(resample_k(ln["baseline"], K))
        except ValueError:
            continue
        h.append(ln["normal_negative_px"] + ln["normal_positive_px"])
        neg.append(ln["normal_negative_px"])
        pos.append(ln["normal_positive_px"])
        texts.append(ln["text"])
        readable.append(_readable(ln["text"], ln["legibility"]))
    if not pts:
        return None
    cut = slice(0, n_max)
    meta = LD.ZoneMeta(ix0=x0, iy0=y0, fx=1.0, fy=1.0, crop_w=tile, crop_h=tile)
    return {"canvas": image, "mask": np.ones((tile, tile), dtype=np.float32), "meta": meta,
            "pts": np.stack(pts[cut]).astype(np.float64), "h": np.asarray(h[cut], dtype=np.float64),
            "neg": np.asarray(neg[cut]), "pos": np.asarray(pos[cut]), "texts": texts[cut], "readable": readable[cut],
            "truncated": max(0, len(pts) - n_max), "scale": 1.0}


# ---------- appariement et perte du témoin (canevas de taille quelconque ; N = 512) ----------

def hungarian_vec(cost) -> list:
    """Même algorithme que line_detr.hungarian, boucle intérieure vectorisée (N = 512 colonnes)."""
    c = np.asarray(cost, dtype=np.float64)
    n, m = c.shape
    if n > m:
        raise ValueError("hungarian: more rows than columns")
    if n == 0:
        return []
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            cur = c[p[j0] - 1] - u[p[j0]] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            masked = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


def match_wh(pred_logit, pred_pts_px, gt_pts_px, gt_h_px, wh, w_cls=2.0, w_pts=5.0, w_dist=1.0):
    """line_detr.match_zone pour un canevas wh quelconque : même coût (focal + L1 normalisé par wh + distance / h)."""
    prob = 1.0 / (1.0 + np.exp(-np.asarray(pred_logit, dtype=np.float64)))
    alpha, gamma = 0.25, 2.0
    neg_cost = (1 - alpha) * (prob ** gamma) * (-np.log(1 - prob + 1e-8))
    pos_cost = alpha * ((1 - prob) ** gamma) * (-np.log(prob + 1e-8))
    scale = np.asarray(wh, dtype=np.float64)
    cost = np.zeros((len(gt_pts_px), len(prob)))
    for t in range(len(gt_pts_px)):
        l1 = np.abs((pred_pts_px - gt_pts_px[t][None]) / scale).sum(axis=(1, 2))
        dense = LD._polyline_samples(gt_pts_px[t])
        P = pred_pts_px.reshape(-1, 2)                    # distance point-polyligne par produit matriciel (N = 512 requêtes)
        d2 = (P ** 2).sum(1)[:, None] + (dense ** 2).sum(1)[None, :] - 2.0 * P @ dense.T
        d = np.sqrt(np.maximum(d2.min(axis=1), 0.0)).reshape(len(prob), -1).mean(axis=1)
        cost[t] = w_cls * (pos_cost - neg_cost) + w_pts * l1 + w_dist * d / gt_h_px[t]
    return hungarian_vec(cost)


def layer_loss_wh(out_l: dict, b: int, sample: dict, matches: list, wh):
    """train_line_detr.layer_loss pour un canevas wh quelconque (points GT normalisés par wh)."""
    logit = out_l["logit"][b]
    tgt = torch.zeros_like(logit)
    q_idx = torch.tensor([q for _, q in matches], dtype=torch.long, device=logit.device)
    g_idx = [g for g, _ in matches]
    n_pos = max(1, len(matches))
    if len(matches):
        tgt[q_idx] = 1.0
    l_cls = TL.focal_loss(logit, tgt).sum() / n_pos
    if not len(matches):
        return l_cls + out_l["pts"].sum() * 0.0, l_cls, 0.0, 0.0
    pts_pred = out_l["pts"][b][q_idx]
    gt_norm = torch.as_tensor(sample["pts"][g_idx] / np.array(wh), dtype=pts_pred.dtype, device=pts_pred.device)
    l_pts = (pts_pred - gt_norm).abs().mean()
    neg_p, pos_p = out_l["neg"][b][q_idx], out_l["pos"][b][q_idx]
    neg_t = torch.as_tensor(sample["neg"][g_idx], dtype=neg_p.dtype, device=neg_p.device)
    pos_t = torch.as_tensor(sample["pos"][g_idx], dtype=pos_p.dtype, device=pos_p.device)
    l_h = ((neg_p - neg_t).abs() + (pos_p - pos_t).abs()).mean() / H_T
    return l_cls + l_pts + l_h, l_cls, float(l_pts), float(l_h)


# ---------- modèle : LINE-DETR de LD-1 + tête CTC de l'étape S ----------

class LineDetrCtc(nn.Module):
    """Le LineDetr de LD-1 (inchangé) et le Reader de l'étape S. Les features stride 4 de l'encodeur (entrée de down[0])
    sont capturées au passage pour la lecture : aucun calcul de l'encodeur n'est dupliqué."""

    def __init__(self, alphabet_size: int, n_queries: int, hidden: int = 128, channels: int = 64):
        super().__init__()
        self.detr = LineDetr(n_queries=n_queries)
        self.reader = Reader(channels * ROWS, alphabet_size, hidden=hidden)
        self._tap = {}
        self.detr.down[0].register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self._tap["f4"] = inputs[0]

    def forward(self, canvas, mask):
        self._tap.clear()
        outs = self.detr(canvas, mask)
        return outs, self._tap.pop("f4")


def new_model(kind: str, cfg: dict, alphabet: list, device):
    TL.set_seeds(cfg["seed"])                           # §4 : graine posée AVANT la construction des modules
    n = cfg["model"]["n_queries_main" if kind == "main" else "n_queries_witness"]
    return LineDetrCtc(len(alphabet), n, hidden=cfg["stage"]["model"].get("hidden", 128),
                       channels=cfg["stage"]["model"]["channels"]).to(device)


def load_inits(model: LineDetrCtc, kind: str, cfg: dict, args) -> dict:
    """Encodeur ET Reader : enc_det.* et reader.* du checkpoint d'E2E-5 pas 2000 (sha complet ; refus si une clé ou une forme
    manque ou diffère). Principal : tout le reste de LineDetr depuis le checkpoint final de LD-1 (sha complet, pas 1871) ;
    témoin : décodeur NEUF. Refus si un sha ou un pas diffère. Retourne les shas et la provenance des clés."""
    enc, dec = cfg["init"]["encoder"], cfg["init"]["decoder"]
    enc_sha = C2.require_sha(args.init_checkpoint, enc["sha256_full"], "checkpoint de l'encodeur (E2E-5)")
    ck = torch.load(args.init_checkpoint, map_location="cpu")
    LD.load_enc_det_from_e2e5(model.detr, ck["model"], expected_step=enc["step"], step=ck["step"])
    sub = {k[len("reader."):]: v for k, v in ck["model"].items() if k.startswith("reader.")}
    want = model.reader.state_dict()
    C2.check_reader_keys({k: v.shape for k, v in sub.items()}, {k: v.shape for k, v in want.items()})
    model.reader.load_state_dict(sub, strict=True)
    out = {"encoder_sha256": enc_sha, "decoder_sha256": None,
           "key_sources": {"enc_det.*": f"E2E-5 {enc_sha[:8]}", "reader.*": f"E2E-5 {enc_sha[:8]}"}}
    if kind == "main":
        out["decoder_sha256"] = C2.require_sha(args.decoder_checkpoint, dec["sha256_full"], "checkpoint du décodeur (LD-1)")
        ck1 = torch.load(args.decoder_checkpoint, map_location="cpu")
        if ck1["step"] != dec["step"] or Path(args.decoder_checkpoint).name != dec["file"]:
            raise SystemExit(f"refusé : checkpoint du décodeur {Path(args.decoder_checkpoint).name} pas {ck1['step']}")
        sd = {k: v for k, v in ck1["model"].items() if not k.startswith("enc_det.")}
        missing, unexpected = model.detr.load_state_dict(sd, strict=False)
        if unexpected or any(not k.startswith("enc_det.") for k in missing):
            raise SystemExit("refusé : clés du checkpoint de LD-1 différentes de LineDetr")
        out["key_sources"]["detr.* sauf enc_det"] = f"LD-1 {out['decoder_sha256'][:8]}"
    else:
        out["key_sources"]["detr.* sauf enc_det"] = "neuf (graine)"
    return out


def make_optimizer(model: LineDetrCtc, cfg: dict):
    ld1 = cfg["ld1"]
    enc = list(model.detr.enc_det.parameters())
    ids = {id(p) for p in enc}
    other = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": enc, "lr": ld1["lr_encoder"]}, {"params": other, "lr": ld1["lr"]}],
                             weight_decay=ld1["weight_decay"])


# ---------- tâches : échantillonnage et perte (principal par zones, témoin par tuiles) ----------

class Task:
    def __init__(self, kind: str, cfg: dict, pages: list, alphabet: list):
        self.kind, self.pages = kind, pages
        self.char_to_idx = {c: i for i, c in enumerate(alphabet) if i > 0}
        self.max_ctc = cfg["stage"]["ctc_lines_per_step_max"]
        if kind == "main":
            self.wh = (CANVAS_W, CANVAS_H)
            self.match, self.layer_loss = LD.match_zone, TL.layer_loss        # LD-1, tels quels
            self.zone_index = [(pi, bi) for pi, p in enumerate(pages) for bi in range(len(p["blocks"]))]
        else:
            tile = cfg["stage"]["tiles"]["size_px"]
            self.tile, self.centred = tile, cfg["stage"]["tiles"]["line_centered_windows"]
            self.n_max = cfg["model"]["n_queries_witness"]
            self.wh = (tile, tile)
            self.match = functools.partial(match_wh, wh=self.wh)
            self.layer_loss = functools.partial(layer_loss_wh, wh=self.wh)

    def sample(self, rng, bs: int) -> list:
        samples, tries = [], 0
        while len(samples) < bs and tries < 10 * bs:
            tries += 1
            if self.kind == "main":
                pi, bi = self.zone_index[int(rng.choice(len(self.zone_index)))]
                page = self.pages[pi]
                blk = page["blocks"][bi]
                s = zone_sample_ctc(page, blk, TL.perturb_box(blk["box"], H_T, blk["box"][2] - blk["box"][0], rng))
            else:
                s = tile_sample(self.pages[int(rng.integers(len(self.pages)))], rng, self.tile, self.centred, self.n_max)
            if s is not None:
                samples.append(s)
        if not samples:
            raise RuntimeError("aucun échantillon valide")
        return samples

    def loss(self, model, samples: list, sigma: float, rng, device):
        """Géométrie (focal + L1 points + L1 hauteurs, chaque couche) + CTC attendue sur les lignes appariées lisibles
        (dernière couche), géométrie DÉTACHÉE pour la CTC, gradient vers enc_det multiplié par λ = 1/6 (GradScale)."""
        canv = torch.as_tensor(np.stack([s["canvas"] for s in samples]))[:, None].to(device)
        msk = torch.as_tensor(np.stack([s["mask"] for s in samples]))[:, None].to(device)
        outs, f4 = model(canv, msk)
        wh = np.array(self.wh)
        total = 0.0
        parts = {"cls": 0.0, "pts": 0.0, "h": 0.0}
        matched_frac, last_matches = [], []
        n_terms = len(samples) * len(outs)
        for b, s in enumerate(samples):
            for out_l in outs:
                with torch.no_grad():
                    pred_px = out_l["pts"][b].detach().cpu().numpy() * wh
                    matches = self.match(out_l["logit"][b].detach().cpu().numpy(), pred_px, s["pts"], s["h"])
                l, lc, lp, lh = self.layer_loss(out_l, b, s, matches)
                total = total + l
                parts["cls"] += float(lc)
                parts["pts"] += lp
                parts["h"] += lh
                matched_frac.append(len(matches) / out_l["logit"].shape[1])
            last_matches.append(matches)
        geom = total / n_terms
        l_ctc, ctc = self.ctc_term(model, samples, last_matches, outs[-1], f4, sigma, rng, device)
        info = {k: v / n_terms for k, v in parts.items()}
        info.update(ctc=float(l_ctc), matched=float(np.mean(matched_frac)), ctc_cut_by_crop=int(sum(s.get("ctc_cut", 0) for s in samples)), **ctc)
        return geom + l_ctc, info

    def ctc_term(self, model, samples, last_matches, last, f4, sigma, rng, device):
        cands = [(b, g, q) for b, (s, m) in enumerate(zip(samples, last_matches)) for g, q in m if s["readable"][g]]
        take = min(self.max_ctc, len(cands))
        chosen = sorted(cands[i] for i in rng.permutation(len(cands))[:take]) if take else []
        feats, losses, counts = {}, [], {}
        wh = torch.as_tensor(self.wh, dtype=last["pts"].dtype, device=last["pts"].device)
        for b, g, q in chosen:
            base = last["pts"][b, q].detach() * wh                             # géométrie détachée (comme B3)
            if float(torch.linalg.norm(base[-1] - base[0])) < MIN_CHORD_PX:
                counts["excluded_degenerate_chord"] = counts.get("excluded_degenerate_chord", 0) + 1
                continue
            if b not in feats:
                feats[b] = GradScale.apply(f4[b:b + 1], LAMBDA)
            sc = samples[b]["scale"]                                           # px page -> px canevas, pour la bande CTC seule
            line = {"baseline": base, "neg": last["neg"][b, q].detach() * sc, "pos": last["pos"][b, q].detach() * sc}
            labels = encode_text(samples[b]["texts"][g], self.char_to_idx)
            try:
                loss, status = expected_ctc_loss(feats[b], line, model.reader, labels, sigma_px=sigma * sc)
            except ValueError as exc:                                  # SEULE erreur rattrapée : corde prédite non gauche-droite
                if "left-to-right" not in str(exc):
                    raise
                loss, status = None, "excluded_chord_not_ltr"
            if loss is not None and not bool(torch.isfinite(loss)):
                loss, status = None, "excluded_nonfinite"
            counts[status] = counts.get(status, 0) + 1
            if loss is not None:
                losses.append(loss)
        l_ctc = sum(losses) / len(losses) if losses else torch.zeros((), device=last["pts"].device)
        return l_ctc, {"ctc_included": len(losses), "ctc_excluded": sum(counts.values()) - len(losses),
                       "ctc_status": counts}


# ---------- chronométrage : contrôle du device et taille de lot (LD-1 gel 7) ----------

def device_control(task, model, opt, rng, device, cfg):
    """Au début de T, 2 pas sur le vrai device (taille de lot la plus petite) ; une erreur autre qu'un OOM est une DeviceError."""
    saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    try:
        for _ in range(2):
            loss, _info = task.loss(model, task.sample(rng, cfg["ld1"]["batch_candidates"][-1]), SIGMA_TIMING, rng, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
    except Exception as exc:
        if TL.is_oom(exc):
            raise
        raise TL.DeviceError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        model.load_state_dict(saved_model)
        opt.load_state_dict(saved_opt)


def choose_batch(task, model, opt, rng, device, cfg):
    """Essais 8, 4, 2 ; la plus grande taille qui tient, avec une marge de 20 % (mémoire CUDA). Seul un OOM fait reculer."""
    tried = []
    for bs in cfg["ld1"]["batch_candidates"]:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            loss, _info = task.loss(model, task.sample(rng, bs), SIGMA_TIMING, rng, device)
            opt.zero_grad()
            loss.backward()
            opt.zero_grad()
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            total = int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None
            fits = True if peak is None else peak * 1.2 <= total
            tried.append({"batch_size": bs, "peak_bytes": peak, "total_bytes": total, "fits": fits})
            if fits:
                return bs, tried
        except Exception as exc:
            if not TL.is_oom(exc):
                raise
            tried.append({"batch_size": bs, "error": str(exc)[:200], "fits": False})
            loss = None
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise TL.RepeatedOOM("aucune taille de lot ne tient")


def time_kind(kind: str, cfg: dict, args, device, pages: list, alphabet: list) -> dict:
    """200 pas chronométrés d'un modèle neuf (mêmes réglages que l'entraînement)."""
    model = new_model(kind, cfg, alphabet, device)
    inits = load_inits(model, kind, cfg, args)
    opt = make_optimizer(model, cfg)
    rng = np.random.default_rng(cfg["seed"])
    task = Task(kind, cfg, pages, alphabet)
    device_control(task, model, opt, rng, device, cfg)
    bs, tried = choose_batch(task, model, opt, rng, device, cfg)
    steps = args.timing_steps or cfg["steps"]["timing_steps"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.monotonic()
    for _ in range(steps):
        loss, _info = task.loss(model, task.sample(rng, bs), SIGMA_TIMING, rng, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    res = {"batch_size": bs, "batch_tries": tried, "s_per_step": (time.monotonic() - t1) / steps, "timing_steps": steps,
           "params_by_block": dict(LD.params_by_block(model.detr), reader=int(sum(p.numel() for p in model.reader.parameters()))),
           "n_queries": model.detr.n_queries, "key_sources": inits["key_sources"]}
    if device.type == "cuda":
        res["max_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return res


# ---------- phases ----------

def manifest_path(out: Path) -> Path:
    return Path(out) / "ld2_manifest.json"


def phase_D(args, cfg):
    L0 = TL.load_l0(cfg["ld1"])
    local = L0.download_detector(cfg["ld1"]["detector"], Path(args.cache_dir))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exits = out.parent / "exits"                         # RUN reste vide
    exits.mkdir(parents=True, exist_ok=True)
    (exits / "ld2_download.json").write_text(json.dumps({k: str(v.name) for k, v in local.items()}, indent=1))
    return {"download": "ok", "files": sorted(local)}


def training_alphabet(root) -> list:
    """Alphabet d'E2E-5 : caractères des lignes lisibles de mf_0000-0049 (le Reader d'E2E-5 en dépend). Pages d'ENTRAÎNEMENT."""
    return build_alphabet([load_mf_page(Path(root) / "pages" / f"{pid}.json", Path(root) / "images" / f"{pid}.png")
                           for pid in TL.guard_pages(TRAIN_ALL)])


def phase_T(args, cfg, device, out: Path, pages: list):
    """Chronométrage du principal seul ; S2 est écrit au manifeste dès la fin de ses 200 pas (§8 (1)). S_t = 0 jusqu'à TW."""
    alphabet = training_alphabet(args.root)
    cfg_steps = dict(cfg["steps"], max_steps=args.max_steps or cfg["steps"]["max_steps"])
    main = time_kind("main", cfg, args, device, pages, alphabet)
    s2 = C2.steps_main(main["s_per_step"], dict(cfg, steps=cfg_steps))
    man = {"schema": "ld2_manifest_v1", "S2": s2, "S_t": 0, "main": dict(main, budget_s=cfg["steps"]["main_budget_s"]),
           "witness": {"status": "NOT_TIMED", "budget_s": cfg["steps"]["witness_budget_s"]}, "max_steps": cfg_steps["max_steps"],
           "timing_steps": main["timing_steps"], "alphabet": alphabet, "device": str(device),
           "fumee": bool(args.pages or args.timing_steps or args.max_steps),
           "init": {"encoder_sha256": C2.sha256_file(args.init_checkpoint),
                    "decoder_sha256": C2.sha256_file(args.decoder_checkpoint), "key_sources": main["key_sources"],
                    "witness_key_sources": None},
           "witness_tiles": {"size_px": cfg["stage"]["tiles"]["size_px"], "overlap_px": cfg["stage_s"]["tile_overlap_px"]},
           "written_before_E": True}
    C2.write_json(manifest_path(out), man)
    return {"S2": s2, "s_per_step_main": main["s_per_step"]}


def phase_TW(args, cfg, out: Path, on_hard_fire=None):
    """Chronométrage du témoin, après T. Écrit S_t au manifeste ; tout échec ou dépassement donne S_t = 0 (témoin NOT_RUN),
    rapporté tel quel. Jamais d'exception ni de code non nul (§8 (1), (2))."""
    t0 = time.monotonic()

    def fail(reason: str, **kw):
        try:
            man = C2.read_json(manifest_path(out))
            man["S_t"] = 0
            man["witness"] = dict(man["witness"], status="TIMING_FAILED", reason=reason, **kw)
            C2.write_json(manifest_path(out), man)
        except Exception:  # noqa: BLE001 — sans manifeste lisible, il n'y a rien à marquer
            pass

    def hard():
        fail("délai dur de TW", seconds=round(time.monotonic() - t0, 1))
        (on_hard_fire or (lambda: os._exit(0)))()

    timer = None
    try:
        if args.deadline_s is not None:                           # armé DÈS LE DÉBUT : tire avant le délai de l'étape du carnet
            timer = arm_watchdog(args.deadline_s + cfg["robustness"]["ew_grace_s"] / 2, hard)
        man = C2.read_json(manifest_path(out))
        TL.check_cuda(args.device)
        device = torch.device(args.device)
        pages = [load_page_text(Path(args.root), pid) for pid in TL.guard_pages(args.pages or TRAIN_ALL)]
        if args.deadline_s is not None and args.deadline_s - (time.monotonic() - t0) <= 0:
            fail("délai atteint pendant le chargement des pages", seconds=round(time.monotonic() - t0, 1))
            return C2.read_json(manifest_path(out))["witness"]
        wit = time_kind("witness", cfg, args, device, pages, man["alphabet"])
        man = C2.read_json(manifest_path(out))
        man["S_t"] = C2.steps_witness(man["S2"], wit["s_per_step"], cfg)
        man["witness"] = dict(wit, status="TIMED", budget_s=cfg["steps"]["witness_budget_s"])
        man["init"]["witness_key_sources"] = wit["key_sources"]
        C2.write_json(manifest_path(out), man)
    except BaseException as exc:  # noqa: BLE001 — rapporté tel quel
        fail(f"{type(exc).__name__}: {exc}"[:400], trace=traceback.format_exc()[-1500:])
    finally:
        if timer is not None:
            timer.cancel()
    return C2.read_json(manifest_path(out))["witness"]


def save_checkpoint(model, step: int, tag: str, identity: dict, out: Path, keep: int) -> Path:
    path = Path(out) / f"ld2_{tag}_step{step:06d}.pt"
    torch.save({"model": model.state_dict(), "step": step, "kind": tag, "identity": identity}, path)
    for old in sorted(Path(out).glob(f"ld2_{tag}_step*.pt"))[:-keep]:
        old.unlink()
    return path


def train_kind(kind: str, cfg: dict, args, device, out: Path, pages: list, alphabet: list, steps: int, bs: int,
               deadline_s=None, on_start=None) -> dict:
    """Entraînement jusqu'à `steps` pas ; un délai (deadline_s) arrête proprement et le dit (NOT_REACHED)."""
    tag = "main" if kind == "main" else "witness"
    model = new_model(kind, cfg, alphabet, device)
    inits = load_inits(model, kind, cfg, args)
    opt = make_optimizer(model, cfg)
    rng = np.random.default_rng(cfg["seed"])
    task = Task(kind, cfg, pages, alphabet)
    identity = {"seed": cfg["seed"], "steps_total": steps, "batch_size": bs, "kind": kind, **inits}
    rb = cfg["robustness"]
    log = Path(out) / f"ld2_{tag}_train_log.jsonl"
    t0, done, reached = time.monotonic(), 0, True
    if on_start is not None:
        on_start()                                                  # le chien de garde compte à partir d'ici
    for step in range(1, steps + 1):
        if deadline_s is not None and time.monotonic() - t0 > deadline_s:
            reached = False
            break
        loss, info = task.loss(model, task.sample(rng, bs), sigma_for_step(step - 1, steps, H_T), rng, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        done = step
        if step % rb["log_every_steps"] == 0 or step == steps:
            row = {"step": step, "loss": float(loss), "focal": info["cls"], "L1_points": info["pts"],
                   "L1_heights": info["h"], "L_ctc": info["ctc"], "ctc_included": info["ctc_included"],
                   "ctc_excluded": info["ctc_excluded"],
                   "ctc_chord_not_ltr": info["ctc_status"].get("excluded_chord_not_ltr", 0), "ctc_cut_by_crop": info["ctc_cut_by_crop"], "matched_query_share": info["matched"],
                   "elapsed_s": round(time.monotonic() - t0, 1)}
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % cfg["ld1"]["checkpoint_every"] == 0 or step == steps:
            ck = save_checkpoint(model, step, tag, identity, out, rb["checkpoints_kept"])
    res = {"steps_done": done, "steps_target": steps, "reached": reached, "seconds": round(time.monotonic() - t0, 1)}
    if reached and done == steps:
        res.update(checkpoint=ck.name, checkpoint_sha256=C2.sha256_file(ck))
    return res


def require_no_checkpoint(out: Path, tag: str) -> None:
    """Reprise interdite (comme LD-1 §15 (3)) : aucun checkpoint de la série au départ."""
    if list(Path(out).glob(f"ld2_{tag}_step*.pt")):
        raise SystemExit(f"refusé : checkpoint ld2_{tag}_step*.pt déjà présent, reprise interdite")


def phase_E(args, cfg, device, out: Path, pages: list):
    man = C2.read_json(manifest_path(out))
    alphabet = man["alphabet"]
    if not set("".join(ln["text"] for p in pages for ln in p["lines"] if ln["legibility"] == "readable")) <= set(alphabet):
        raise SystemExit("refusé : des caractères des pages sont absents de l'alphabet écrit par T")
    require_no_checkpoint(out, "main")
    return train_kind("main", cfg, args, device, out, pages, alphabet, man["S2"], man["main"]["batch_size"])


def arm_watchdog(seconds: float, on_fire):
    """Dernier recours du témoin : appelle on_fire après `seconds` (par défaut, on écrit l'état et on sort avec le code 0)."""
    timer = threading.Timer(seconds, on_fire)
    timer.daemon = True
    timer.start()
    return timer


def phase_EW(args, cfg, out: Path, on_hard_fire=None):
    """Témoin : jamais d'exception ni de code de sortie non nul ; l'état est écrit tel quel (sans relance). Le chargement des
    pages, le contrôle CUDA et les refus sont DANS le try : un échec ici est un échec du témoin, pas du lancement."""
    status_path = Path(out) / "ld2_witness_status.json"
    st = {"status": "NOT_RUN"}
    t0 = time.monotonic()
    timers = []

    def write(**kw):
        st.update(kw, seconds=round(time.monotonic() - t0, 1))
        C2.write_json(status_path, st)

    def hard():
        write(status="NOT_REACHED", reason="délai dur du témoin")
        (on_hard_fire or (lambda: os._exit(0)))()

    deadline = args.deadline_s
    try:
        if deadline is not None:                                    # armé DÈS LE DÉBUT : tire avant le délai de l'étape du carnet
            timers.append(arm_watchdog(deadline + cfg["robustness"]["ew_grace_s"] / 2, hard))
        man = C2.read_json(manifest_path(out))
        st.update(S2=man["S2"], S_t=man["S_t"])
        if not man["S_t"]:
            write(status="NOT_RUN", reason="S_t = 0 : chronométrage du témoin absent ou en échec")
            return st
        TL.check_cuda(args.device)
        device = torch.device(args.device)
        pages = [load_page_text(Path(args.root), pid) for pid in TL.guard_pages(args.pages or TRAIN_ALL)]
        require_no_checkpoint(out, "witness")
        if deadline is not None:                                    # délai doux = délai moins le temps déjà passé à charger
            deadline = max(0.0, deadline - (time.monotonic() - t0))
        if deadline is not None and deadline <= 0:
            write(status="NOT_REACHED", reason="délai atteint pendant le chargement des pages", steps_done=0)
            return st
        res = train_kind("witness", cfg, args, device, out, pages, man["alphabet"], man["S_t"],
                         man["witness"]["batch_size"], deadline_s=deadline)
        write(status="COMPLETE" if res["reached"] else "NOT_REACHED", **res)
    except BaseException as exc:  # noqa: BLE001 — rapporté tel quel
        write(status="FAILED", error=f"{type(exc).__name__}: {exc}"[:600], trace=traceback.format_exc()[-1500:])
    finally:
        for t in timers:
            t.cancel()
    return st


def phase_FREEZE(args, cfg, out: Path):
    """Seuil, NMS et checkpoint (le DERNIER, pas S2) écrits dans le manifeste. Aucune page, aucun chemin d'évaluation."""
    man = C2.read_json(manifest_path(out))
    S = man["S2"]
    ck = Path(out) / f"ld2_main_step{S:06d}.pt"
    if not ck.is_file():
        raise SystemExit(f"refusé : checkpoint {ck.name} absent, S2 non atteint (INCOMPLETE)")
    frozen = {}
    for rel, prefix in cfg["freeze"]["frozen_files_sha256_prefix"].items():
        frozen[rel] = C2.sha256_file(ROOT / rel)
        if not frozen[rel].startswith(prefix):
            raise SystemExit(f"refusé : {rel} diffère de la version gelée ({prefix}…)")
    for rel in cfg["freeze"]["also_record_sha256"]:                  # §8 (5) : code et config de LD-2 (enregistrés, non comparés)
        frozen[rel] = C2.sha256_file(ROOT / rel)
    man["freeze"] = C2.freeze_record(cfg, ck.name, C2.sha256_file(ck), S, frozen)
    man["witness_status_at_freeze"] = C2.witness_report(out)           # codes de sortie de TW et EW lus tels quels
    C2.write_json(manifest_path(out), man)
    return man["freeze"]


# ---------- évaluation d'une page (CAL et EVAL partagent ce code) ----------

@torch.no_grad()
def raw_outputs_read(model: LineDetrCtc, canvas, mask, thr: float, alphabet: list, wh, device, scale: float = 1.0) -> dict:
    """Sorties de la dernière couche pour une zone ou une tuile, et lecture (texte brut, décodage glouton) des seules
    requêtes au-dessus du seuil, avec la géométrie prédite de la requête. `scale` (px canevas / px page) convertit neg et pos
    (px page) en px canevas pour la bande de lecture seulement."""
    outs, f4 = model(torch.as_tensor(canvas)[None, None].to(device), torch.as_tensor(mask)[None, None].to(device))
    last = outs[-1]
    prob = torch.sigmoid(last["logit"][0]).cpu().numpy()
    pts = last["pts"][0] * torch.as_tensor(wh, dtype=last["pts"].dtype, device=last["pts"].device)
    hyp, hyp_none, reading_errors = {}, {}, []
    for q in np.where(prob >= thr)[0]:
        if float(torch.linalg.norm(pts[q][-1] - pts[q][0])) < MIN_CHORD_PX:
            hyp[int(q)], hyp_none[int(q)] = None, "degenerate_chord"
            continue
        try:                                                         # lecture isolée : une erreur n'arrête pas la géométrie (§8 (3))
            band = band_features(f4, pts[q], last["neg"][0, q] * scale, last["pos"][0, q] * scale)
            hyp[int(q)] = None if band is None else greedy_decode(model.reader(band), alphabet)
            if band is None:
                hyp_none[int(q)] = "band_outside_support"
        except Exception as exc:  # noqa: BLE001
            hyp[int(q)], hyp_none[int(q)] = None, "reader_exception"
            reading_errors.append(f"{type(exc).__name__}: {exc}"[:200])
    return {"hyp_none": hyp_none, "reading_errors": reading_errors, "prob": prob, "pts": pts.cpu().numpy(), "neg": last["neg"][0].cpu().numpy(),
            "pos": last["pos"][0].cpu().numpy(), "hyp": hyp}


def tile_origins(size: int, tile: int, stride: int) -> list:
    last = max(0, size - tile)
    xs = list(range(0, last + 1, stride))
    return xs if xs[-1] == last else xs + [last]


def witness_raw(model, page: dict, thr: float, alphabet: list, cfg: dict, device) -> list:
    """Témoin : la page est découpée en tuiles (taille d'E2E-5, recouvrement de la config) ; (id de tuile, repère, sorties)."""
    tile, overlap = cfg["stage"]["tiles"]["size_px"], cfg["stage_s"]["tile_overlap_px"]
    img = page["image"]
    H, W = img.shape
    out = []
    for y0 in tile_origins(H, tile, tile - overlap):
        for x0 in tile_origins(W, tile, tile - overlap):
            canvas = np.full((tile, tile), LD.PAD_VALUE, dtype=np.float32)
            crop = img[y0:y0 + tile, x0:x0 + tile]
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            mask = np.zeros((tile, tile), dtype=np.float32)
            mask[:crop.shape[0], :crop.shape[1]] = 1.0
            meta = LD.ZoneMeta(ix0=x0, iy0=y0, fx=1.0, fy=1.0, crop_w=tile, crop_h=tile)
            out.append((len(out), meta, raw_outputs_read(model, canvas, mask, thr, alphabet, (tile, tile), device)))
    return out


def score_lines(page: dict, kept: list, raws: list) -> dict:
    """Mêmes chiffres que train_line_detr.score_page (line_geometry.v1), plus le CER du texte brut des lignes appariées
    (lisibles) : somme des distances d'édition / somme des longueurs ; les lignes dont la bande sort du support sont comptées."""
    gt = [GroundTruthLine(id=g["id"], baseline=g["baseline"].tolist(), height_px=float(g["neg"] + g["pos"]),
                          baseline_source="ground_truth", geometry_status="available") for g in page["gt_lines"]]
    cer = {"edits": 0, "chars": 0, "lines": 0, "skipped": 0, "matched_readable_lines": 0, "matched_readable_chars": 0,
           "skipped_by_cause": {"degenerate_chord": 0, "band_outside_support": 0, "reader_exception": 0, "unknown": 0}}
    if not kept:
        return {"n_pred": 0, "n_gt": len(gt), "precision": 0.0, "recall": 0.0, "precision_none": False, "splits": 0,
                "merges": 0, "cer": cer}
    pred = [PredictedLine(id=f"p{i}", baseline=k["pts"].tolist(), baseline_source="predicted", geometry_status="available")
            for i, k in enumerate(kept)]
    res = evaluate_line_geometry(gt, pred)
    gt_by_id = {g["id"]: g for g in page["gt_lines"]}
    raw_of = {zid: raw for zid, _meta, raw in raws}
    pred_idx = {p.id: i for i, p in enumerate(pred)}
    try:                                                             # CER isolé : une erreur est consignée, la géométrie reste (§8 (3))
        for m in res.matches:
            g = gt_by_id[m.gt_id]
            if not _readable(g["text"], g["legibility"]):
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
    return {"n_pred": len(pred), "n_gt": len(gt),
            "precision": None if res.precision is None else float(res.precision),
            "recall": float(res.recall or 0.0), "precision_none": res.precision is None,
            "splits": int(res.split_candidate_gt_count or 0), "merges": int(res.merge_candidate_pred_count or 0),
            "cer": cer}


def make_det_fn(L0, cache: Path):
    """Détecteur PP chargé UNE fois (mêmes sorties que le chargement page par page de train_line_detr.pp_zones)."""
    det = TL.L0_DETECTOR
    proc, model, _ = L0.load_detector(det, cache)
    return lambda arr: L0.detect(proc, model, Image.fromarray(arr), det["floor"])


def zone_raws(model, page, zones, thr, alphabet, device):
    raws = []
    for zid, box in zones:
        canvas, meta = zone_canvas(page["image"], box, h_page=H_T)
        raws.append((zid, meta, raw_outputs_read(model, canvas, zone_mask(meta), thr, alphabet, (CANVAS_W, CANVAS_H), device,
                                                 scale=0.5 * (meta.fx + meta.fy))))
    return raws


def evaluate_page(pid: str, root: Path, ctx: dict) -> dict:
    """Une page, trois sources : principal en Z-PP-blocs (critère), principal en Z-GT, témoin sans zones (descriptifs).
    Au seuil gelé. Temps de chaque source."""
    thr, alphabet, device = ctx["threshold"], ctx["alphabet"], ctx["device"]
    t0 = time.monotonic()
    page = load_page_text(root, pid)
    row = {"page": pid, "seconds": {"load": round(time.monotonic() - t0, 3)}}
    t = time.monotonic()
    gt_zones = [(i, b["box"]) for i, b in enumerate(page["blocks"])]
    raws = zone_raws(ctx["main"], page, gt_zones, thr, alphabet, device)
    row["Z-GT"] = score_lines(page, TL.candidates_for_threshold(raws, thr), raws)
    row["seconds"]["Z-GT"] = round(time.monotonic() - t, 3)
    t = time.monotonic()
    pp = TL.pp_zones(page, root, pid, ctx["cache"], ctx["L0"], det_fn=ctx["det_fn"])
    row["seconds"]["detector"] = round(time.monotonic() - t, 3)
    raws = zone_raws(ctx["main"], page, pp, thr, alphabet, device)
    row["Z-PP-blocs"] = score_lines(page, TL.candidates_for_threshold(raws, thr), raws)
    row["Z-PP-blocs"]["n_zones"] = len(pp)
    row["seconds"]["Z-PP-blocs"] = round(time.monotonic() - t, 3)
    if ctx.get("witness") is not None:
        t = time.monotonic()
        try:                                                         # témoin isolé page par page ; une erreur du principal reste fatale
            raws = witness_raw(ctx["witness"], page, thr, alphabet, ctx["cfg"], device)
            row["witness"] = score_lines(page, TL.candidates_for_threshold(raws, thr), raws)
        except Exception as exc:  # noqa: BLE001
            row["witness"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        row["seconds"]["witness"] = round(time.monotonic() - t, 3)
    row["seconds"]["total"] = round(time.monotonic() - t0, 3)
    return row


def load_eval_models(cfg, args, device, out: Path, man: dict):
    """Principal (checkpoint gelé, sha recontrôlé) et témoin (seulement si son état est COMPLETE et son sha identique)."""
    fz = man["freeze"]
    ck = Path(out) / fz["checkpoint"]
    C2.require_sha(ck, fz["checkpoint_sha256"], "checkpoint gelé")
    alphabet = man["alphabet"]
    main = new_model("main", cfg, alphabet, device)
    main.load_state_dict(torch.load(ck, map_location="cpu")["model"])
    main.eval()
    witness = None
    wst = C2.witness_report(out)                                     # état + codes de sortie de TW et EW, tels quels
    if wst.get("status") == "COMPLETE":
        wck = Path(out) / wst["checkpoint"]
        C2.require_sha(wck, wst["checkpoint_sha256"], "checkpoint du témoin")
        witness = new_model("witness", cfg, alphabet, device)
        witness.load_state_dict(torch.load(wck, map_location="cpu")["model"])
        witness.eval()
    return main, witness, wst


def eval_context(cfg, args, device, out: Path, man: dict) -> dict:
    t0 = time.monotonic()
    main, witness, wst = load_eval_models(cfg, args, device, out, man)
    L0 = TL.load_l0(cfg["ld1"])
    cache = Path(args.cache_dir)
    det_fn = make_det_fn(L0, cache)
    return {"main": main, "witness": witness, "witness_status": wst, "alphabet": man["alphabet"], "device": device,
            "threshold": man["freeze"]["threshold"], "L0": L0, "det_fn": det_fn, "cache": cache, "cfg": cfg,
            "load_seconds": round(time.monotonic() - t0, 3)}


def phase_CAL(args, cfg, device, out: Path, root: Path):
    """Étalonnage : UNE page d'ENTRAÎNEMENT (la première de --pages, sinon la page de la config), jamais une page d'évaluation."""
    man = C2.read_json(manifest_path(out))
    if "freeze" not in man:
        raise SystemExit("refusé : FREEZE n'a pas été écrit")
    pid = TL.guard_pages([(args.pages or [cfg["eval"]["cal_page"]])[0]])[0]
    ctx = eval_context(cfg, args, device, out, man)
    t = time.monotonic()
    row = evaluate_page(pid, root, ctx)
    page_s = time.monotonic() - t
    n = cfg["eval"]["cal_pages_estimate"] if not args.smoke else len(args.eval_pages or [pid])
    rec = {"page": pid, "load_seconds": ctx["load_seconds"], "page_seconds": round(page_s, 3), "pages_estimated": n,
           "estimate_s": round(ctx["load_seconds"] + n * page_s, 1), "seconds_by_source": row["seconds"],
           "witness_status": ctx["witness_status"].get("status")}
    cal_out = Path(args.cal_out)
    cal_out.mkdir(parents=True, exist_ok=True)
    C2.write_json(cal_out / "ld2_cal.json", rec)
    return rec


def _ratio(a, b):
    return (a / b) if b else None


def aggregate(rows: list, source: str, n_pages: int) -> dict:
    """Les pages dont la source a une erreur (témoin isolé, §8 (3)) sont listées à part ; la moyenne porte sur les autres."""
    errors = {r["page"]: r[source]["error"] for r in rows if "error" in r.get(source, {})}
    if source != "witness":                                          # principal : dénominateur FIXE, une erreur est fatale (§0 (a))
        if errors or len(rows) != n_pages:
            raise ValueError(f"{source} : {len(rows)} pages dont {len(errors)} en erreur, attendu {n_pages} sans erreur")
    rows = [r for r in rows if "error" not in r.get(source, {})]
    if not rows:
        return {"pages": 0, "pages_with_error": errors}
    srows = [dict(r[source], page=r["page"]) for r in rows]
    agg = C2.mean_pr(srows, len(rows))
    chars = sum(r["cer"]["chars"] for r in srows)
    agg.update(n_pred=sum(r["n_pred"] for r in srows), n_gt=sum(r["n_gt"] for r in srows),
               splits=sum(r["splits"] for r in srows), merges=sum(r["merges"] for r in srows),
               cer={"scope": C2.CER_SCOPE, "value": (sum(r["cer"]["edits"] for r in srows) / chars) if chars else None,
                    "lines": sum(r["cer"]["lines"] for r in srows), "chars": chars,
                    "matched_readable_lines": sum(r["cer"]["matched_readable_lines"] for r in srows),
                    "matched_readable_chars": sum(r["cer"]["matched_readable_chars"] for r in srows),
                    "coverage_lines": _ratio(sum(r["cer"]["lines"] for r in srows), sum(r["cer"]["matched_readable_lines"] for r in srows)),
                    "coverage_chars": _ratio(chars, sum(r["cer"]["matched_readable_chars"] for r in srows)),
                    "skipped_total": sum(r["cer"]["skipped"] for r in srows),
                    "skipped_by_cause": {c: sum(r["cer"]["skipped_by_cause"][c] for r in srows)
                                         for c in ("degenerate_chord", "band_outside_support", "reader_exception", "unknown")},
                    "pages_with_cer_error": sum(1 for r in srows if "error" in r["cer"])},
               seconds_per_page_mean=float(np.mean([r["seconds"][source] for r in rows])),
               pages_with_error=errors, per_page=srows)
    return agg


def phase_EVAL(args, cfg, device, out: Path, root: Path):
    """Évaluation UNIQUE. Seul endroit où les chemins des pages d'évaluation sont construits (jamais en fumée)."""
    man = C2.read_json(manifest_path(out))
    if "freeze" not in man or not man["freeze"].get("written_before_EVAL"):
        raise SystemExit("refusé : FREEZE doit être écrit dans le manifeste avant EVAL")
    if (Path(out) / "ld2_report.json").exists():
        raise SystemExit("refusé : le rapport existe déjà, aucune nouvelle soumission")
    if (Path(out) / EVAL_STARTED).exists():
        raise SystemExit(f"refusé : {EVAL_STARTED} existe, l'évaluation a déjà démarré ; une nouvelle EVAL exige une décision "
                         "explicite de pA (le fichier n'est jamais supprimé automatiquement)")
    if args.smoke:
        ids = TL.guard_pages(args.eval_pages or [])
        if not ids:
            raise SystemExit("refusé : --smoke exige --eval-pages (pages de remplacement)")
    else:
        if args.eval_pages:
            raise SystemExit("refusé : --eval-pages n'existe qu'en fumée")
        ids = C2.eval_page_ids()
        if len(ids) != cfg["criterion"]["a"]["pages"]:
            raise SystemExit("refusé : le nombre de pages d'évaluation diffère du lot")
    if not args.reference_report:
        raise SystemExit("refusé : --reference-report manquant")
    ref = C2.load_reference(args.reference_report, ids, cfg, smoke=args.smoke)       # sha contrôlé AVANT toute page
    ctx = eval_context(cfg, args, device, out, man)
    try:                                                              # marqueur créé AVANT la première page, jamais supprimé
        with open(Path(out) / EVAL_STARTED, "x") as f:
            json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "checkpoint_sha256": man["freeze"]["checkpoint_sha256"], "smoke": bool(args.smoke), "pages": ids}, f, indent=1)
    except FileExistsError:
        raise SystemExit(f"refusé : {EVAL_STARTED} existe, une nouvelle EVAL exige une décision explicite de pA") from None
    rows, partial = [], Path(out) / "ld2_eval_pages.jsonl"
    for pid in ids:
        rows.append(evaluate_page(pid, root, ctx))
        with partial.open("a") as f:
            f.write(json.dumps(rows[-1], default=str) + "\n")
    n = len(ids)
    main_rows = [dict(r["Z-PP-blocs"], page=r["page"]) for r in rows]
    report = {"schema": "ld2_report_v1", "checkpoint_sha256": man["freeze"]["checkpoint_sha256"],
              "reserves": C2.RESERVES, "limits": [C2.DECODER_PAIRING_LIMIT, cfg["witness"]["limit"], C2.TILE_SELECTION_LIMIT], "status": "COMPLETE", "smoke": bool(args.smoke),
              "checkpoint": man["freeze"]["checkpoint"], "cer_scope": C2.CER_SCOPE,
              "reference": {"file": cfg["reference"]["file"], "decoy": bool(args.smoke)},
              "criterion": C2.criterion(main_rows, ref, ids, cfg),
              "steps": {"S2": man["S2"], "S_t": man["S_t"], "s_per_step_main": man["main"]["s_per_step"],
                        "s_per_step_witness": man["witness"].get("s_per_step"), "batch_size_main": man["main"]["batch_size"],
                        "batch_size_witness": man["witness"].get("batch_size")},
              "frozen": man["freeze"], "init": man["init"], "pages": ids,
              "Z-PP-blocs": aggregate(rows, "Z-PP-blocs", n), "Z-GT": aggregate(rows, "Z-GT", n),
              "witness": dict(ctx["witness_status"], limit=cfg["witness"]["limit"], note="analogue de CurT dans nos conditions, pas une reproduction ; "
                              "descriptif, ne décide rien", tile_overlap_px=cfg["stage_s"]["tile_overlap_px"],
                              tile_overlap_note=cfg["stage_s"]["tile_overlap_note"]),
              "load_seconds": ctx["load_seconds"], "per_page": rows}
    if ctx["witness"] is not None:
        report["witness"]["result"] = aggregate(rows, "witness", n)
    C2.write_json(Path(out) / "ld2_report.json", report)
    return {"verdict": report["criterion"]["verdict"]}


# ---------- point d'entrée ----------

def main(argv=None) -> int:
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--decoder-checkpoint", default=None)
    ap.add_argument("--reference-report", default=None)
    ap.add_argument("--cal-out", default=None)
    ap.add_argument("--cache-dir", default="/private/tmp/ld2_hf_cache")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--eval-pages", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timing-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--deadline-s", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_all()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"phase": args.phase, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "RUNNING"}
    t0 = time.monotonic()
    try:
        if args.phase == "D":
            if args.device == "cuda":
                TL.check_cuda("cuda")
            rec.update(status="COMPLETE", **phase_D(args, cfg))
            return 0
        if args.phase == "EW":
            st = phase_EW(args, cfg, out)
            rec.update(status=st["status"], witness=st)
            return 0
        if args.phase == "TW":
            rec.update(status="COMPLETE", witness=phase_TW(args, cfg, out))
            return 0
        if args.phase in ("CAL", "EVAL"):
            os.environ["HF_HUB_OFFLINE"] = "1"            # mesure hors ligne (comme LD-1 V), avant tout import de huggingface_hub
        if args.phase == "T":
            TL.require_empty_output(out)
        if args.phase == "FREEZE":
            rec.update(status="COMPLETE", freeze=phase_FREEZE(args, cfg, out))
            return 0
        TL.check_cuda(args.device)
        device = torch.device(args.device)
        root = Path(args.root)
        if args.phase in ("T", "E"):
            pages = [load_page_text(root, pid) for pid in TL.guard_pages(args.pages or TRAIN_ALL)]
            if args.phase == "T":
                rec.update(status="COMPLETE", manifest=phase_T(args, cfg, device, out, pages))
            else:
                rec.update(status="COMPLETE", **phase_E(args, cfg, device, out, pages))
        elif args.phase == "CAL":
            rec.update(status="COMPLETE", cal=phase_CAL(args, cfg, device, out, root))
        else:
            rec.update(status="COMPLETE", **phase_EVAL(args, cfg, device, out, root))
    except SystemExit:
        rec.update(status="FAILED")
        raise
    except TL.DeviceError as exc:
        rec.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        raise
    except Exception as exc:  # noqa: BLE001
        rec.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        raise
    finally:
        rec["duration_s"] = round(time.monotonic() - t0, 1)
        exits = out.parent / "exits"                     # sorties d'étape hors de RUN
        exits.mkdir(parents=True, exist_ok=True)
        (exits / f"ld2_run_exit_{args.phase}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: rec[k] for k in ("phase", "status")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
