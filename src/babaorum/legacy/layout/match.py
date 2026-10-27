"""Appariement des lignes et des blocs trouvés avec la vérité (D-211).

Trois mesures de lignes, fixées avant toute mesure :

- **ligne de base (principale)** : une ligne vraie ``g`` et une ligne
  trouvée ``h`` s'apparient si chacune couvre l'autre sur au moins 75 % de
  son étendue en abscisse à moins de ``½ × hauteur de g`` en ordonnée
  (:func:`axel.layout.geom.baseline_cover`) ; appariement un pour un,
  glouton par recouvrement décroissant. Une ligne coupée en deux ou deux
  colonnes fondues en une ne s'apparient pas.
- **IoU des polygones ≥ 0,5**, un pour un, glouton (rastérisé au ¼).
- **couverture souple** (à la cBAD) : part des points des lignes de base
  vraies qu'une ligne trouvée longe à la même tolérance (rappel), et
  l'inverse (précision).

Défauts comptés : **fusions** (une ligne trouvée longe à ≥ 50 % deux
lignes vraies côte à côte : colonnes fondues), **coupes** (une ligne vraie
longée à ≥ 25 % par deux lignes trouvées au moins).

Blocs : part des lignes vraies dont le centre tombe dans un bloc trouvé,
**blocs mêlés** (un bloc contient deux lignes vraies côte à côte), F1 des
régions vraies à IoU ≥ 0,5 (granularités différentes : indicatif).
Ordre : τ de Kendall entre rangs vrais et trouvés des lignes appariées.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from babaorum.legacy.layout.geom import (
    Box,
    Mask,
    Point,
    baseline_cover,
    bbox,
    boxes_touch,
    line_height,
    mask_iou,
    rasterize,
    synthetic_baseline,
)

COVER = 0.75
TOL = 0.5
IOU = 0.5
MERGE = 0.5
SPLIT = 0.25


@dataclass(frozen=True)
class Seg:
    """Une ligne (vraie ou trouvée) réduite à sa géométrie."""

    id: str
    polygon: tuple[Point, ...]
    baseline: tuple[Point, ...]

    @property
    def box(self) -> Box:
        return bbox(self.polygon or self.baseline)


def seg(id_: str, polygon: Sequence[Point], baseline: Sequence[Point]) -> Seg:
    poly = tuple((float(x), float(y)) for x, y in polygon)
    base = tuple((float(x), float(y)) for x, y in baseline)
    if len(base) < 2 and poly:
        base = synthetic_baseline(poly)
    if not poly and base:
        h = 10.0
        poly = tuple([(x, y - h) for x, y in base] + [(x, y) for x, y in base[::-1]])
    return Seg(id_, poly, base)


@dataclass
class LineScores:
    n_gt: int
    n_hyp: int
    pairs_bl: list[tuple[int, int]] = field(default_factory=list)
    pairs_iou: list[tuple[int, int]] = field(default_factory=list)
    soft_recall: float = 0.0
    soft_precision: float = 0.0
    merges: int = 0  # lignes trouvées qui fondent deux lignes vraies côte à côte
    splits: int = 0  # lignes vraies coupées entre deux lignes trouvées
    kendall: float | None = None

    @staticmethod
    def _prf(k: int, n_gt: int, n_hyp: int) -> tuple[float, float, float]:
        r = k / n_gt if n_gt else 0.0
        p = k / n_hyp if n_hyp else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    @property
    def bl(self) -> tuple[float, float, float]:
        return self._prf(len(self.pairs_bl), self.n_gt, self.n_hyp)

    @property
    def iou(self) -> tuple[float, float, float]:
        return self._prf(len(self.pairs_iou), self.n_gt, self.n_hyp)


def side_by_side(a: Box, b: Box) -> bool:
    """Deux boîtes de ligne à la même hauteur et disjointes en abscisse."""
    ov = min(a[3], b[3]) - max(a[1], b[1])
    hmin = min(a[3] - a[1], b[3] - b[1])
    return hmin > 0 and ov > 0.5 * hmin and (a[2] <= b[0] or b[2] <= a[0])


def _candidates(gt: Sequence[Seg], hyp: Sequence[Seg], grow: float) -> list[list[int]]:
    """Pour chaque ligne vraie, les lignes trouvées dont la boîte touche la
    sienne (élargie de ``grow`` × sa hauteur en ordonnée)."""
    if not hyp:
        return [[] for _ in gt]
    hb = np.array([h.box for h in hyp])
    out: list[list[int]] = []
    for g in gt:
        x0, y0, x1, y1 = g.box
        m = (y1 - y0) * grow
        ok = (
            (hb[:, 0] < x1)
            & (hb[:, 2] > x0)
            & (hb[:, 1] < y1 + m)
            & (hb[:, 3] > y0 - m)
        )
        out.append([int(i) for i in np.nonzero(ok)[0]])
    return out


def _greedy(scores: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    scores.sort(key=lambda t: -t[0])
    used_g: set[int] = set()
    used_h: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in scores:
        if i in used_g or j in used_h:
            continue
        used_g.add(i)
        used_h.add(j)
        pairs.append((i, j))
    return sorted(pairs)


def kendall_tau(a: Sequence[int], b: Sequence[int]) -> float | None:
    """τ de Kendall (sans ex aequo) entre deux suites de rangs."""
    n = len(a)
    if n < 2:
        return None
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    s = float((dx * dy)[np.triu_indices(n, 1)].sum())
    return s / (n * (n - 1) / 2)


def score_lines(gt: Sequence[Seg], hyp: Sequence[Seg]) -> LineScores:
    """Les mesures de lignes de D-211 ; ``gt`` et ``hyp`` sont dans leur
    ordre de lecture respectif."""
    res = LineScores(len(gt), len(hyp))
    heights = [line_height(g.polygon, g.baseline) for g in gt]
    cand = _candidates(gt, hyp, grow=1.0)
    bl_scores: list[tuple[float, int, int]] = []
    iou_scores: list[tuple[float, int, int]] = []
    gt_masks: dict[int, Mask] = {}
    hyp_masks: dict[int, Mask] = {}
    gt_cover = [0.0] * len(gt)
    partial: dict[int, list[int]] = {}  # h → lignes vraies longées à ≥ MERGE
    split_count = [0] * len(gt)
    for i, g in enumerate(gt):
        tol = TOL * heights[i]
        best_cov = 0.0
        for j in cand[i]:
            h = hyp[j]
            # IoU des polygones : mesure distincte de la ligne de base (D-211),
            # calculée même quand les lignes de base ne se longent pas (C2, L6)
            if g.polygon and h.polygon and boxes_touch(g.box, h.box):
                if i not in gt_masks:
                    gt_masks[i] = rasterize(g.polygon)
                if j not in hyp_masks:
                    hyp_masks[j] = rasterize(h.polygon)
                v = mask_iou(gt_masks[i], hyp_masks[j])
                if v >= IOU:
                    iou_scores.append((v, i, j))
            c_gh = baseline_cover(g.baseline, h.baseline, tol)
            if c_gh <= 0:
                continue
            c_hg = baseline_cover(h.baseline, g.baseline, tol)
            best_cov = max(best_cov, c_gh)
            if c_gh >= SPLIT:
                split_count[i] += 1
            if c_gh >= MERGE:
                partial.setdefault(j, []).append(i)
            if min(c_gh, c_hg) >= COVER:
                bl_scores.append((min(c_gh, c_hg), i, j))
        gt_cover[i] = best_cov
    res.pairs_bl = _greedy(bl_scores)
    res.pairs_iou = _greedy(iou_scores)
    # couverture souple : rappel = moyenne pondérée par la longueur des
    # lignes vraies de la meilleure couverture (approximation : la meilleure
    # ligne trouvée seule) ; précision = symétrique, ligne trouvée contre
    # toutes les lignes vraies proches.
    glen = [max(p[0] for p in g.baseline) - min(p[0] for p in g.baseline) for g in gt]
    tot = sum(glen)
    res.soft_recall = (
        sum(c * w for c, w in zip(gt_cover, glen, strict=True)) / tot if tot else 0.0
    )
    hcand = _candidates(hyp, gt, grow=1.0)
    hcov: list[float] = []
    hlen: list[float] = []
    for j, h in enumerate(hyp):
        best = 0.0
        for i in hcand[j]:
            best = max(
                best, baseline_cover(h.baseline, gt[i].baseline, TOL * heights[i])
            )
        hcov.append(best)
        hlen.append(max(p[0] for p in h.baseline) - min(p[0] for p in h.baseline))
    htot = sum(hlen)
    res.soft_precision = (
        sum(c * w for c, w in zip(hcov, hlen, strict=True)) / htot if htot else 0.0
    )
    for gs in partial.values():
        boxes = [gt[i].box for i in gs]
        if any(
            side_by_side(boxes[a], boxes[b])
            for a in range(len(boxes))
            for b in range(a + 1, len(boxes))
        ):
            res.merges += 1
    res.splits = sum(1 for k in split_count if k >= 2)
    if res.pairs_bl:
        res.kendall = kendall_tau(
            [i for i, _ in res.pairs_bl], [j for _, j in res.pairs_bl]
        )
    return res


def point_in_polygon(x: float, y: float, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


@dataclass
class BlockScores:
    n_blocks: int
    covered: float  # part des lignes vraies dont le centre est dans un bloc
    mixed: int  # blocs qui contiennent deux lignes vraies côte à côte
    region_p: float
    region_r: float
    region_f: float


def score_blocks(
    gt_lines: Sequence[Seg],
    gt_regions: Sequence[Sequence[Point]],
    blocks: Sequence[Sequence[Point]],
) -> BlockScores:
    centers = []
    for g in gt_lines:
        b = g.box
        centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    bboxes = [bbox(p) for p in blocks]
    members: list[list[int]] = [[] for _ in blocks]
    covered = 0
    for i, (x, y) in enumerate(centers):
        hit = False
        for k, poly in enumerate(blocks):
            bb = bboxes[k]
            if (
                bb[0] <= x <= bb[2]
                and bb[1] <= y <= bb[3]
                and point_in_polygon(x, y, poly)
            ):
                members[k].append(i)
                hit = True
        covered += hit
    mixed = 0
    for mem in members:
        boxes = sorted((gt_lines[i].box for i in mem), key=lambda b: b[1])
        found = False
        for ia in range(len(boxes)):
            for ib in range(ia + 1, len(boxes)):
                if boxes[ib][1] > boxes[ia][3]:
                    break
                if side_by_side(boxes[ia], boxes[ib]):
                    found = True
                    break
            if found:
                break
        mixed += found
    rmasks = [rasterize(p, 0.1) for p in gt_regions if len(p) >= 3]
    bmasks = [rasterize(p, 0.1) for p in blocks if len(p) >= 3]
    sc: list[tuple[float, int, int]] = []
    rb = [bbox(p) for p in gt_regions if len(p) >= 3]
    bb2 = [bbox(p) for p in blocks if len(p) >= 3]
    for i, rm in enumerate(rmasks):
        for j, bm in enumerate(bmasks):
            if boxes_touch(rb[i], bb2[j]):
                v = mask_iou(rm, bm)
                if v >= IOU:
                    sc.append((v, i, j))
    k = len(_greedy(sc))
    p, r, f = LineScores._prf(k, len(rmasks), len(bmasks))
    return BlockScores(
        len(blocks),
        covered / len(gt_lines) if gt_lines else 0.0,
        mixed,
        p,
        r,
        f,
    )
