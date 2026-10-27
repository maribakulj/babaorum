"""Synthetic pages for the LOT-P02 tests: local_block proposals with an identity crop (rect = page, scale 1, no padding)."""
from __future__ import annotations

import hashlib
import random

from babaorum.experimental.joint_reconcile import CropTransform, LineProposal

PAGE_W, PAGE_H = 3000.0, 4000.0
SHA = hashlib.sha256(b"page").hexdigest()


def proposal(pid, baseline, block="z0", query=None, document="doc"):
    ct = CropTransform((0.0, 0.0, PAGE_W, PAGE_H), 1.0, (0.0, 0.0, 0.0, 0.0))
    return LineProposal(proposal_id=pid, source="local_block", page_key="p1", document_key=document, page_size_wh=(PAGE_W, PAGE_H),
                        page_sha256=SHA, query_id=query or pid, input_sha256=hashlib.sha256(block.encode()).hexdigest(),
                        checkpoint_sha256=None, geometry_status="available", baseline_xy=tuple(baseline),
                        band_normal_negative_px=None, band_normal_positive_px=None, orientation="ltr", text_status="available", text="x",
                        token_ids=None, ctc_status="feasible", block_id=block, crop_transform=ct)


def synthetic_page(seed: int, h: float = 20.0, n_columns: int = 2, rows: int = 14):
    """Columns of text lines at a pitch close to the reach limits, with duplicates, fragments, tilted and degenerate lines."""
    rng = random.Random(seed)
    props, k = [], 0

    def add(points, block):
        nonlocal k
        k += 1
        props.append(proposal(f"{block}:{k:05d}", points, block=block))

    for col in range(n_columns):
        x0 = 100.0 + col * 900.0
        pitch = h * rng.choice([0.45, 0.5, 0.55, 0.9, 1.0, 1.1, 1.3])
        for r in range(rows):
            y = 200.0 + r * pitch
            x1 = x0 + rng.uniform(300, 700)
            add([(x0, y), (x1, y + rng.uniform(-2, 2))], f"z{col}")
            roll = rng.random()
            if roll < 0.25:   # a duplicate, slightly shifted
                add([(x0 + rng.uniform(-3, 3), y + rng.uniform(-4, 4)), (x1 + rng.uniform(-3, 3), y + rng.uniform(-4, 4))], f"z{col}")
            elif roll < 0.45:  # a margin fragment inside the line
                a = rng.uniform(x0, x1 - 50)
                add([(a, y + 1), (a + rng.uniform(2, 12), y + 1)], f"z{col + 2}")
            elif roll < 0.6:   # the line cut in two, touching or overlapping at the junction
                m = (x0 + x1) / 2
                add([(x0, y), (m + rng.uniform(-8, 8), y)], f"z{col + 4}")
                add([(m + rng.uniform(-8, 8), y), (x1, y)], f"z{col + 4}")
            elif roll < 0.7:   # a tilted line crossing the next row
                add([(x0, y), (x0 + 200, y + h * rng.uniform(0.8, 1.6))], f"z{col + 6}")
    add([(2000.0, 300.0), (2000.0, 300.0)], "zdeg")   # degenerate (zero length)
    add([(2500.0, 300.0), (2500.0 + 60, 300.0), (2500.0 + 60, 310.0), (2500.0 + 120, 310.0)], "zpoly")  # polyline
    rng.shuffle(props)
    return props
