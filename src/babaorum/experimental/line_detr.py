"""LINE-DETR (LD-1, docs/END_TO_END_LD1_LOT.txt §2, §3, §10 ; conception §2-§4, §10).

Décodeur à requêtes PAR ZONE : chaque requête prédit une ligne (logit, K = 8 points de baseline normalisés dans le canevas
de la zone, neg et pos). Géométrie seule : aucune tête CTC dans LD-1.

Entrée : canevas de 1536 x 512 (fond 1.0), zone + marge de 0,5 h, réduit si la zone est plus grande (facteur noté).
Encodeur : enc_det de l'étape S (stride 4, 64 canaux), initialisé depuis E2E-5 au pas 2000 (contrôle sur Kaggle),
puis deux convolutions de stride 2 (stride 16, 256 canaux) et un encodeur transformer de 3 couches (positions 2D).
Décodeur : 6 couches, N = 32 requêtes ; points raffinés à chaque couche ; supervision à chaque couche.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from babaorum.experimental.stage_s_model import Encoder, resample_k

CANVAS_W = 1536
CANVAS_H = 512
K = 8                 # points de baseline par ligne
N_QUERIES = 32        # N = 32 (LD-0R confirme 32 : pas de 64)
D_MODEL = 256
ENC_LAYERS = 3
DEC_LAYERS = 6
N_HEADS = 8
H_T = 32.0            # hauteur de référence (px de la page mise à l'échelle) ; les hauteurs prédites sont en unités de H_T
MARGIN_H = 0.5        # marge de 0,5 h de chaque côté
PAD_VALUE = 1.0       # fond de page
ANCHOR_X = np.linspace(0.05, 0.95, K)


# ---------- zones : découpage, canevas, repères ----------

@dataclass(frozen=True)
class ZoneMeta:
    ix0: int          # origine entière du recadrage dans la page mise à l'échelle
    iy0: int
    fx: float         # facteur canevas / recadrage (1.0 si pas de réduction)
    fy: float
    crop_w: int
    crop_h: int


def zone_canvas(page: np.ndarray, box, h_page: float = H_T, pad: float = PAD_VALUE):
    """Zone = boîte + marge de 0,5 h ; recadrage avec pad hors page ; réduction si plus grand que le canevas."""
    x0, y0, x1, y1 = box
    m = MARGIN_H * h_page
    X0, Y0, X1, Y1 = x0 - m, y0 - m, x1 + m, y1 + m
    ix0, iy0 = int(math.floor(X0)), int(math.floor(Y0))
    ix1, iy1 = int(math.ceil(X1)), int(math.ceil(Y1))
    cw, ch = ix1 - ix0, iy1 - iy0
    crop = np.full((ch, cw), pad, dtype=np.float32)
    H, W = page.shape
    sx0, sy0 = max(ix0, 0), max(iy0, 0)
    sx1, sy1 = min(ix1, W), min(iy1, H)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = page[sy0:sy1, sx0:sx1]
    f = min(1.0, CANVAS_W / cw, CANVAS_H / ch)
    nw, nh = max(1, round(cw * f)), max(1, round(ch * f))
    if (nw, nh) != (cw, ch):
        resized = np.asarray(Image.fromarray(crop, mode="F").resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    else:
        resized = crop
    canvas = np.full((CANVAS_H, CANVAS_W), pad, dtype=np.float32)
    canvas[:nh, :nw] = resized
    meta = ZoneMeta(ix0=ix0, iy0=iy0, fx=nw / cw, fy=nh / ch, crop_w=cw, crop_h=ch)
    return canvas, meta


def page_to_zone(points_xy, meta: ZoneMeta) -> np.ndarray:
    p = np.asarray(points_xy, dtype=np.float64).copy()
    p[:, 0] = (p[:, 0] - meta.ix0) * meta.fx
    p[:, 1] = (p[:, 1] - meta.iy0) * meta.fy
    return p


def zone_to_page(points_xy, meta: ZoneMeta) -> np.ndarray:
    p = np.asarray(points_xy, dtype=np.float64).copy()
    p[:, 0] = p[:, 0] / meta.fx + meta.ix0
    p[:, 1] = p[:, 1] / meta.fy + meta.iy0
    return p


# ---------- appariement hongrois (sans scipy : fumee-venv n'a pas scipy) ----------

def hungarian(cost) -> list:
    """Affectation de coût minimal pour une matrice n x m (n <= m). Retourne [(ligne, colonne)] trié par ligne."""
    c = np.asarray(cost, dtype=np.float64)
    n, m = c.shape
    if n > m:
        raise ValueError("hungarian: more rows than columns")
    if n == 0:
        return []
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return sorted(pairs)


# ---------- coûts et pertes ----------

def _polyline_samples(pts: np.ndarray, n: int = 64) -> np.ndarray:
    seg = np.hypot(*(pts[1:] - pts[:-1]).T)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    t = np.linspace(0.0, cum[-1], n)
    return np.stack((np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])), axis=-1)


def point_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> float:
    """Distance moyenne des points prédits à la polyligne GT (échantillonnée), en px canevas."""
    dense = _polyline_samples(polyline)
    d = np.sqrt(((points[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(d.mean())


def match_zone(pred_logit, pred_pts_px, gt_pts_px, gt_h_px, w_cls=2.0, w_pts=5.0, w_dist=1.0):
    """Appariement hongrois par zone. pred_pts_px [N,K,2] et gt_pts_px [T,K,2] en px canevas ; gt_h_px [T].
    Coût = focal sur la classe + L1 des points (normalisés par le canevas) + distance point-polyligne / h."""
    prob = 1.0 / (1.0 + np.exp(-np.asarray(pred_logit, dtype=np.float64)))
    alpha, gamma = 0.25, 2.0
    neg_cost = (1 - alpha) * (prob ** gamma) * (-np.log(1 - prob + 1e-8))
    pos_cost = alpha * ((1 - prob) ** gamma) * (-np.log(prob + 1e-8))
    T, N = len(gt_pts_px), len(prob)
    cost = np.zeros((T, N))
    scale = np.array([CANVAS_W, CANVAS_H], dtype=np.float64)
    for t in range(T):
        l1 = np.abs((pred_pts_px - gt_pts_px[t][None]) / scale).sum(axis=(1, 2))
        dist = np.array([point_polyline_distance(pred_pts_px[q], gt_pts_px[t]) for q in range(N)]) / gt_h_px[t]
        cost[t] = w_cls * (pos_cost - neg_cost) + w_pts * l1 + w_dist * dist
    return hungarian(cost)


# ---------- modèle ----------

def sine_pos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """Positions sinusoïdales 2D, [h*w, d] (d/2 pour y, d/2 pour x)."""
    half = d // 2
    freq = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half))
    ys = torch.arange(h, dtype=torch.float32)[:, None] * freq[None]
    xs = torch.arange(w, dtype=torch.float32)[:, None] * freq[None]
    py = torch.cat([ys.sin(), ys.cos()], dim=1)          # [h, half]
    px = torch.cat([xs.sin(), xs.cos()], dim=1)          # [w, half]
    grid = torch.cat([py[:, None, :].expand(h, w, half), px[None, :, :].expand(h, w, half)], dim=-1)
    return grid.reshape(h * w, d)


def _inv_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def zone_mask(meta: "ZoneMeta") -> np.ndarray:
    """Masque du recadrage utile dans le canevas (1 = page, 0 = remplissage hors recadrage)."""
    nw = int(round(meta.crop_w * meta.fx))
    nh = int(round(meta.crop_h * meta.fy))
    m = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    m[:nh, :nw] = 1.0
    return m


def masked_group_norm(x: torch.Tensor, m: torch.Tensor, gn: nn.GroupNorm) -> torch.Tensor:
    """GroupNorm dont les statistiques ne portent que sur les pixels valides (m = 1). Avec m == 1 partout,
    identique à nn.GroupNorm (§6, 1e-6). Les sorties hors masque ne sont PAS remises à zéro (§12 (6))."""
    b, c, h, w = x.shape
    g = gn.num_groups
    dt = x.dtype
    x64 = x.double()
    m64 = m.double()
    xg = x64.reshape(b, g, c // g, h, w)
    mm = m64.reshape(b, 1, 1, h, w)
    cnt = (mm.sum(dim=(2, 3, 4), keepdim=True) * (c // g)).clamp_min(1.0)
    mean = (xg * mm).sum(dim=(2, 3, 4), keepdim=True) / cnt
    var = (((xg - mean) ** 2) * mm).sum(dim=(2, 3, 4), keepdim=True) / cnt
    y = ((xg - mean) / torch.sqrt(var + gn.eps)).reshape(b, c, h, w)
    y = y * gn.weight.double().view(1, c, 1, 1) + gn.bias.double().view(1, c, 1, 1)
    return y.to(dt)                                       # sorties hors recadrage NON remises à zéro (§12 (6))


def _down_mask(m: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return F.interpolate(m, size=h.shape[-2:], mode="nearest")


def masked_encoder(enc: Encoder, x: torch.Tensor, m: torch.Tensor):
    """Encoder de l'étape S (mêmes modules, mêmes clés) avec entrée et statistiques masquées.
    Hors recadrage : fond 1,0 à l'entrée, activations NON remises à zéro (§12 (6)) ; seules les statistiques
    de la GroupNorm et les jetons (dans le transformer et le décodeur) sont restreints au recadrage."""
    net = enc.net
    h = net[0](x)                                          # hors recadrage = fond 1,0, sans remise à zéro (§12 (6))
    mm = _down_mask(m, h)
    h = F.relu(masked_group_norm(h, mm, net[1]))
    h = net[3](h)
    mm = _down_mask(m, h)
    h = F.relu(masked_group_norm(h, mm, net[4]))
    for block in (net[6], net[7]):
        body = block.body
        y = body[0](h)
        y = F.relu(masked_group_norm(y, mm, body[1]))
        y = body[3](y)
        y = masked_group_norm(y, mm, body[4])
        h = F.relu(h + y)
    return h, mm


def masked_down(seq: nn.Sequential, h: torch.Tensor, m: torch.Tensor):
    h = seq[0](h)
    mm = _down_mask(m, h)
    h = F.relu(masked_group_norm(h, mm, seq[1]))
    h = seq[3](h)
    mm = _down_mask(m, h)
    h = F.relu(masked_group_norm(h, mm, seq[4]))
    return h, mm


class LineDetr(nn.Module):
    def __init__(self, n_queries: int = N_QUERIES, k: int = K, d: int = D_MODEL, enc_layers: int = ENC_LAYERS,
                 dec_layers: int = DEC_LAYERS, channels: int = 64):
        super().__init__()
        self.n_queries, self.k, self.d = n_queries, k, d
        self.enc_det = Encoder(channels)
        self.down = nn.Sequential(
            nn.Conv2d(channels, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, d, 3, stride=2, padding=1), nn.GroupNorm(8, d), nn.ReLU(inplace=True),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, N_HEADS, 4 * d, dropout=0.0, batch_first=True), enc_layers)
        self.queries = nn.Embedding(n_queries, d)
        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d, N_HEADS, 4 * d, dropout=0.0, batch_first=True) for _ in range(dec_layers)])
        self.cls = nn.Linear(d, 1)
        self.pts_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 2 * k))
        self.size_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 2))
        nn.init.zeros_(self.pts_head[-1].weight)
        nn.init.zeros_(self.pts_head[-1].bias)
        ys = (np.arange(n_queries) + 0.5) / n_queries                     # ancres : bandes horizontales
        anchors = np.stack([np.broadcast_to(ANCHOR_X, (n_queries, k)),
                            np.repeat(ys[:, None], k, axis=1)], axis=-1)   # [N, K, 2] (x, y) dans [0, 1]
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))

    def forward(self, canvas: torch.Tensor, mask: torch.Tensor = None):
        """canvas [B, 1, 512, 1536], mask [B, 1, 512, 1536] (1 = page). Sortie par couche : logit, pts, neg, pos."""
        if mask is None:
            mask = torch.ones_like(canvas)
        f4, m4 = masked_encoder(self.enc_det, canvas, mask)
        f16, m16 = masked_down(self.down, f4, m4)
        b, c, h, w = f16.shape
        valid = m16.flatten(2)[:, 0] > 0.5                                 # [B, h*w] jetons de la page
        tok = f16.flatten(2).transpose(1, 2) + sine_pos_2d(h, w, c).to(f16)[None]
        mem = self.encoder(tok, src_key_padding_mask=~valid)
        hs = self.queries.weight[None].expand(b, -1, -1)
        ref = self.anchors[None].expand(b, -1, -1, -1).clone()
        outs = []
        for layer in self.decoder_layers:
            hs = layer(hs, mem, memory_key_padding_mask=~valid)
            delta = self.pts_head(hs).view(b, self.n_queries, self.k, 2)
            ref = torch.sigmoid(_inv_sigmoid(ref.detach()) + delta)
            sz = F.softplus(self.size_head(hs)) * H_T
            outs.append({"logit": self.cls(hs)[..., 0], "pts": ref, "neg": sz[..., 0], "pos": sz[..., 1]})
        return outs


def params_by_block(model: "LineDetr") -> dict:
    """Répartition des paramètres par bloc, écrite dans le manifeste (§11 (c))."""
    blocks = {"enc_det": model.enc_det, "down": model.down, "encoder": model.encoder,
              "queries": model.queries, "decoder": model.decoder_layers, "cls": model.cls,
              "pts_head": model.pts_head, "size_head": model.size_head}
    out = {name: int(sum(p.numel() for p in mod.parameters())) for name, mod in blocks.items()}
    out["total"] = int(sum(p.numel() for p in model.parameters()))
    return out


def encoder_state_keys(model: LineDetr) -> dict:
    """Clés et formes de l'encodeur initialisé depuis E2E-5 (enc_det), pour le contrôle init_check (§7 (7))."""
    return {k: tuple(v.shape) for k, v in model.enc_det.state_dict().items()}


def load_enc_det_from_e2e5(model: LineDetr, state: dict, *, expected_step: int, step: int) -> None:
    """Refus si le pas diffère ; charge exactement les clés enc_det.* (préfixe du StageSModel de E2E-5)."""
    if step != expected_step:
        raise ValueError(f"checkpoint step {step} != {expected_step}")
    sub = {k[len("enc_det."):]: v for k, v in state.items() if k.startswith("enc_det.")}
    want = encoder_state_keys(model)
    if set(sub) != set(want):
        raise ValueError("enc_det keys differ from the E2E-5 checkpoint")
    for k, shape in want.items():
        if tuple(sub[k].shape) != shape:
            raise ValueError(f"enc_det shape differs for {k}")
    model.enc_det.load_state_dict(sub, strict=True)


# ---------- inférence : NMS des doublons (§10 (9)) et décodage ----------

def _overlap_rule(a_pts: np.ndarray, b_pts: np.ndarray, a_h: float, b_h: float) -> bool:
    """Relation de recouvrement (§10 (9), §12 (10)) : a = ligne CANDIDATE, b = ligne GARDÉE. o > 0,5·min(L_A, L_B)
    ET t ≤ 0,5·h ; t = médiane des distances des points de la candidate à la gardée."""
    da = a_pts[-1] - a_pts[0]
    db = b_pts[-1] - b_pts[0]
    d = da / (np.linalg.norm(da) + 1e-9) + db / (np.linalg.norm(db) + 1e-9)
    d = d / (np.linalg.norm(d) + 1e-9)
    sa, sb = a_pts @ d, b_pts @ d
    a0, a1, b0, b1 = sa.min(), sa.max(), sb.min(), sb.max()
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    L = min(a1 - a0, b1 - b0)
    h = 0.5 * (a_h + b_h)
    dense_b = _polyline_samples(b_pts)
    t = float(np.median([np.min(np.hypot(*(dense_b - p).T)) for p in a_pts]))
    return overlap > 0.5 * L and t <= 0.5 * h


def nms_lines(preds: list) -> list:
    """preds : dicts (score, zone, query, pts [K,2] page, h). Tri par score décroissant, égalité par (zone, requête).
    Glouton et déterministe : une prédiction est gardée si aucune prédiction déjà gardée ne la recouvre."""
    order = sorted(preds, key=lambda p: (-p["score"], p["zone"], p["query"]))
    kept = []
    for cand in order:
        if all(not _overlap_rule(cand["pts"], k["pts"], cand["h"], k["h"]) for k in kept):
            kept.append(cand)
    return kept
