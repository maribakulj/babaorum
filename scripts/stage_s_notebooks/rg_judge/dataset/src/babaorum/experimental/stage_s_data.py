"""Stage S data loader (step S-L1): Mille Feuilles canonical page -> arrays and targets.

Reads canonical files in place; no data is copied into babaorum. CPU only.
Bands are derived per line from the polygon and baseline, never from a fixed height.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

BLANK = "<blank>"


def _band(baseline, polygon):
    """Normal distances of the polygon above and below the baseline, in pixels.

    n = (-t_y, t_x) with t the unit chord from the first to the last baseline point.
    """
    p0, p_last = baseline[0], baseline[-1]
    dx, dy = p_last[0] - p0[0], p_last[1] - p0[1]
    if dx < 0:
        raise ValueError("baseline chord runs right-to-left")
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("baseline chord has zero length")
    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx
    offsets = [(v[0] - p0[0]) * nx + (v[1] - p0[1]) * ny for v in polygon]
    return max(0.0, -min(offsets)), max(0.0, max(offsets))


def load_mf_page(page_json_path, image_path) -> dict:
    page_json_path, image_path = Path(page_json_path), Path(image_path)
    page = json.loads(page_json_path.read_text())

    raw = image_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != page["image"]["sha256"]:
        raise ValueError(f"image sha256 does not match {page_json_path.name}")
    image = Image.open(image_path)
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (page["image"]["height"], page["image"]["width"]):
        raise ValueError("image dimensions do not match the page JSON")

    lines = []
    for line in page["lines"]:
        neg, pos = _band(line["baseline"], line["polygon"])
        lines.append({
            "id": line["id"],
            "block_id": line["block_id"],
            "baseline": line["baseline"],
            "normal_negative_px": neg,
            "normal_positive_px": pos,
            "text": line["text"],
            "legibility": line["legibility"],
        })
    return {"page_id": page["page_id"], "image": array, "lines": lines, "scale_note": None}


def build_alphabet(pages) -> list[str]:
    """Sorted characters of readable line texts; index 0 is the CTC blank."""
    chars = set()
    for page in pages:
        for line in page["lines"]:
            if line["legibility"] == "readable":
                chars.update(line["text"])
    return [BLANK, *sorted(chars)]
