"""Word boxes from CTC character positions (LOT-P01c, v3). Stdlib only; no segmentation or reading is rerun.

Conventions (declared before any measure):
- words are the runs of non-space characters of the line text; the CTC blank is not a character and is discarded;
- x extent of a word = from the first to the last ``cut`` of its characters; characters without a cut contribute
  nothing; a word with NO cut at all has NO coordinates (nothing is prorated, nothing is imputed);
- y extent = from the baseline and the line height only, never by intersection with the line polygon:
  top = baseline y - up, bottom = baseline y + down, with (up, down) measured once per line from its polygon;
- the share of each word box that falls outside its line polygon is MEASURED afterwards, never corrected;
- a CTC peak is narrow, so the end of a word is uncertain (Huang et al., ICASSP 2024): this is reported, not hidden.
Every repair of a polygon is returned with a code. Nothing is repaired silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Sequence

Point = tuple[float, float]
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int
    x0: float | None
    x1: float | None
    conf: float | None
    chars: int
    cut_chars: int

    @property
    def has_box(self) -> bool:
        return self.x0 is not None and self.x1 is not None


def normalized(text: str) -> str:
    """The text a word sequence can reproduce: runs of whitespace collapsed, borders trimmed."""
    return " ".join(text.split())


def split_words(text: str, spans: Sequence[tuple[float, float] | None] | None = None,
                confs: Sequence[float | None] | None = None) -> list[Word]:
    """Cut ``text`` on whitespace and attach x extents from per-character spans. No prorata."""
    if spans is not None and len(spans) != len(text):
        raise ValueError(f"{len(spans)} positions for {len(text)} characters")
    if confs is not None and len(confs) != len(text):
        raise ValueError(f"{len(confs)} confidences for {len(text)} characters")
    words = []
    for m in _WORD.finditer(text):
        idx = range(m.start(), m.end())
        cuts = [spans[i] for i in idx if spans is not None and spans[i] is not None]
        confidence = [confs[i] for i in idx if confs is not None and confs[i] is not None]
        words.append(Word(m.group(), m.start(), m.end(),
                          min(c[0] for c in cuts) if cuts else None, max(c[1] for c in cuts) if cuts else None,
                          mean(confidence) if confidence else None, len(idx), len(cuts)))
    return words


def words_text(words: Sequence[Word]) -> str:
    return " ".join(w.text for w in words)


def baseline_y(baseline: Sequence[Point], x: float) -> float:
    """Baseline height at ``x`` (piecewise linear; clamped beyond the ends)."""
    pts = sorted(baseline)
    if x <= pts[0][0]:
        return pts[0][1]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if x <= xb:
            return ya if xb == xa else ya + (yb - ya) * (x - xa) / (xb - xa)
    return pts[-1][1]


def line_extent(polygon: Sequence[Point], baseline: Sequence[Point]) -> tuple[float, float]:
    """(up, down): how far the line polygon reaches above and below its mean baseline height."""
    by = mean(p[1] for p in baseline)
    ys = [p[1] for p in polygon]
    return max(by - min(ys), 0.0), max(max(ys) - by, 0.0)


def _clamp(points: list[Point], width: int, height: int) -> tuple[list[Point], bool]:
    out = [(min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))) for x, y in points]
    return out, out != points


def _area2(points: Sequence[Point]) -> float:
    return sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1]
               for i in range(len(points)))


def word_polygon(word: Word, baseline: Sequence[Point], up: float, down: float,
                 width: int, height: int) -> tuple[list[Point] | None, bool]:
    """Parallelogram following the baseline, from the word's cuts. None when the word has no usable box."""
    if not word.has_box or word.x1 <= word.x0 or up + down <= 0:  # type: ignore[operator]
        return None, False
    ya, yb = baseline_y(baseline, word.x0), baseline_y(baseline, word.x1)  # type: ignore[arg-type]
    pts = [(word.x0, ya - up), (word.x1, yb - up), (word.x1, yb + down), (word.x0, ya + down)]  # type: ignore[list-item]
    pts, clamped = _clamp(pts, width, height)
    return (pts, clamped) if abs(_area2(pts)) > 1e-9 else (None, clamped)


def _inside(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        (xa, ya), (xb, yb) = polygon[i], polygon[(i + 1) % n]
        if (ya > y) != (yb > y) and x < (xb - xa) * (y - ya) / (yb - ya) + xa:
            inside = not inside
    return inside


def outside_share(box: Sequence[Point], line_polygon: Sequence[Point], nx: int = 12, ny: int = 6) -> float:
    """Share of a word box (sampled on an nx x ny grid, even-odd rule) that lies outside its line polygon."""
    (x0, ya), (x1, yb), _, (_, yd) = box[0], box[1], box[2], box[3]
    height0 = yd - ya
    out = total = 0
    for i in range(nx):
        u = (i + 0.5) / nx
        x, top = x0 + u * (x1 - x0), ya + u * (yb - ya)
        for j in range(ny):
            total += 1
            out += not _inside((x, top + (j + 0.5) / ny * height0), line_polygon)
    return out / total


def distribution(shares: Sequence[float]) -> dict:
    if not shares:
        return {"n": 0, "median": None, "p95": None, "over_10pct": 0}
    ordered = sorted(shares)
    return {"n": len(ordered), "median": median(ordered), "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
            "over_10pct": sum(s > 0.10 for s in ordered)}


# --- polygon repairs (always declared) -------------------------------------------------

def dedupe_polygon(points: Sequence[Point]) -> list[Point]:
    """Drop consecutive duplicate vertices, including a first vertex repeated at the end. Lossless: same shape."""
    out: list[Point] = []
    for p in points:
        if not out or p != out[-1]:
            out.append(p)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def baseline_band(baseline: Sequence[Point], up: float, down: float, width: int, height: int) -> list[Point] | None:
    pts = sorted(baseline)
    ring = [(x, y - up) for x, y in pts] + [(x, y + down) for x, y in reversed(pts)]
    ring, _ = _clamp(ring, width, height)
    ring = dedupe_polygon(ring)
    return ring if len(ring) >= 3 and abs(_area2(ring)) > 1e-9 else None


def repair_polygon(points: Sequence[Point], is_valid: Callable[[list[Point]], bool], baseline: Sequence[Point],
                   up: float, down: float, width: int, height: int) -> tuple[list[Point] | None, str | None]:
    """(polygon, code). code is None when the polygon was already valid.

    ``polygon_repaired`` : only repeated vertices removed (same shape).
    ``polygon_from_baseline`` : a band deduced from the baseline and the line height (the shape changes).
    ``polygon_unrepairable`` : neither worked; the caller must keep the line and declare its geometry as an error.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if is_valid(pts):
        return pts, None
    fixed = dedupe_polygon(pts)
    if len(fixed) >= 3 and is_valid(fixed):
        return fixed, "polygon_repaired"
    band = baseline_band(baseline, up, down, width, height) if baseline and len(baseline) >= 2 else None
    if band and is_valid(band):
        return band, "polygon_from_baseline"
    return None, "polygon_unrepairable"
