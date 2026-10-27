"""Géométrie des lignes et des blocs : masques rastérisés, IoU, lignes de base.

Les polygones sont rastérisés à l'échelle ``scale`` (¼ par défaut) sur leur
seule boîte englobante : un masque par polygone, une IoU par paire de boîtes
qui se touchent. Les lignes de base sont des polylignes à peu près
horizontales ; ``y(x)`` s'obtient par interpolation linéaire.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def bbox(points: Sequence[Point]) -> Box:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def box_polygon(box: Box) -> tuple[Point, ...]:
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def boxes_touch(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_inter(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def box_iou(a: Box, b: Box) -> float:
    i = box_inter(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


@dataclass(frozen=True)
class Mask:
    """Un polygone rastérisé : ``array[y, x]`` à partir de ``(x0, y0)``."""

    x0: int
    y0: int
    array: NDArray[np.bool_]

    @property
    def area(self) -> int:
        return int(self.array.sum())


def rasterize(points: Sequence[Point], scale: float = 0.25) -> Mask:
    xs = [p[0] * scale for p in points]
    ys = [p[1] * scale for p in points]
    x0, y0 = int(np.floor(min(xs))), int(np.floor(min(ys)))
    w = max(1, int(np.ceil(max(xs))) - x0 + 1)
    h = max(1, int(np.ceil(max(ys))) - y0 + 1)
    im = Image.new("1", (w, h), 0)
    if len(points) >= 3:
        ImageDraw.Draw(im).polygon(
            [(x - x0, y - y0) for x, y in zip(xs, ys, strict=True)], fill=1, outline=1
        )
    else:
        ImageDraw.Draw(im).line(
            [(x - x0, y - y0) for x, y in zip(xs, ys, strict=True)], fill=1
        )
    return Mask(x0, y0, np.asarray(im, dtype=bool))


def mask_inter(a: Mask, b: Mask) -> int:
    ax1, ay1 = a.x0 + a.array.shape[1], a.y0 + a.array.shape[0]
    bx1, by1 = b.x0 + b.array.shape[1], b.y0 + b.array.shape[0]
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0
    sa = a.array[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0]
    sb = b.array[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0]
    return int(np.logical_and(sa, sb).sum())


def mask_iou(a: Mask, b: Mask) -> float:
    i = mask_inter(a, b)
    u = a.area + b.area - i
    return i / u if u > 0 else 0.0


def baseline_y(
    baseline: Sequence[Point], xs: NDArray[np.float64]
) -> NDArray[np.float64]:
    pts = sorted(baseline)
    bx = np.array([p[0] for p in pts], dtype=float)
    by = np.array([p[1] for p in pts], dtype=float)
    out: NDArray[np.float64] = np.asarray(np.interp(xs, bx, by), dtype=np.float64)
    return out


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def line_height(polygon: Sequence[Point], baseline: Sequence[Point]) -> float:
    """Hauteur d'une ligne : aire du polygone sur la longueur de sa ligne de
    base (hauteur de la boîte si l'un des deux manque)."""
    if polygon and baseline and len(baseline) >= 2:
        xs = [p[0] for p in baseline]
        length = max(xs) - min(xs)
        if length > 0:
            h = polygon_area(polygon) / length
            if h > 0:
                return h
    if polygon:
        b = bbox(polygon)
        return b[3] - b[1]
    return 0.0


def synthetic_baseline(polygon: Sequence[Point]) -> tuple[Point, ...]:
    """Ligne de base d'une ligne sans ligne de base (boîte Tesseract, bloc
    de VLM) : horizontale, aux quatre cinquièmes de la hauteur."""
    x0, y0, x1, y1 = bbox(polygon)
    y = y0 + 0.8 * (y1 - y0)
    return ((x0, y), (x1, y))


def baseline_cover(
    a: Sequence[Point],
    b: Sequence[Point],
    tol: float,
    step: float = 4.0,
) -> float:
    """Part de l'étendue en abscisse de ``a`` où ``b`` existe à moins de
    ``tol`` pixels en ordonnée (0 à 1)."""
    ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
    if ax1 <= ax0:
        return 0.0
    bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
    xs = np.arange(ax0, ax1 + step / 2, step, dtype=float)
    inside = (xs >= bx0) & (xs <= bx1)
    if not inside.any():
        return 0.0
    ya = baseline_y(a, xs[inside])
    yb = baseline_y(b, xs[inside])
    ok = np.abs(ya - yb) <= tol
    return float(ok.sum()) / len(xs)
