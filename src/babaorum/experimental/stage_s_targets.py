"""Stage S targets (step S-L2): rasterise baselines to a stride-4 grid and extract them back.

CPU only, synthetic-probe scope. Coordinates are continuous page edges; a point
(x, y) falls in cell (floor(x/stride), floor(y/stride)); the cell centre is
((c + 0.5) * stride, (r + 0.5) * stride).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

_STEP_CELLS = 0.25


def rasterize_baselines(lines, image_hw, stride: int = 4) -> dict:
    height, width = image_hw
    hc, wc = math.ceil(height / stride), math.ceil(width / stride)
    baseline = np.zeros((hc, wc), dtype=np.float32)
    neg = np.zeros((hc, wc), dtype=np.float32)
    pos = np.zeros((hc, wc), dtype=np.float32)
    mask = np.zeros((hc, wc), dtype=bool)
    owner: dict[tuple[int, int], int] = {}
    collisions = 0

    for line_index, line in enumerate(lines):
        pts = line["baseline"]
        step = _STEP_CELLS * stride
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            count = max(1, math.ceil(seg / step))
            for k in range(count + 1):
                t = k / count
                x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                c, r = int(math.floor(x / stride)), int(math.floor(y / stride))
                if not (0 <= r < hc and 0 <= c < wc):
                    continue
                prev = owner.get((r, c))
                if prev is None:
                    owner[(r, c)] = line_index
                    baseline[r, c] = 1.0
                    neg[r, c] = float(line["normal_negative_px"])
                    pos[r, c] = float(line["normal_positive_px"])
                    mask[r, c] = True
                elif prev != line_index:
                    collisions += 1
    return {"baseline": baseline, "neg": neg, "pos": pos, "mask": mask, "collisions": collisions}


def extract_polylines(prob, *, stride: int = 4, threshold: float = 0.5, min_cells: int = 4) -> list:
    prob = np.asarray(prob)
    hc, wc = prob.shape
    fg = prob >= threshold
    seen = np.zeros((hc, wc), dtype=bool)
    components = []
    for r0 in range(hc):
        for c0 in range(wc):
            if not fg[r0, c0] or seen[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (dr or dc) and 0 <= rr < hc and 0 <= cc < wc and fg[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            queue.append((rr, cc))
            if len(cells) < min_cells:
                continue
            by_col: dict[int, list[int]] = {}
            for r, c in cells:
                by_col.setdefault(c, []).append(r)
            poly = []
            for c in sorted(by_col):
                rows = by_col[c]
                r_mean = sum(rows) / len(rows)
                poly.append([(c + 0.5) * stride, (r_mean + 0.5) * stride])
            components.append(poly)

    components.sort(key=lambda p: (min(q[1] for q in p), min(q[0] for q in p)))
    return components
