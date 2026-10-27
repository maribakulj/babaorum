"""Analytic line-recall bounds for block cascades and global line proposals."""

from __future__ import annotations

from itertools import product
from typing import Any


def line_recall_bounds(p_block_miss_line_weighted: float,
                       q_local_conditional_miss: float,
                       r_global_miss: float) -> dict[str, float]:
    """Compute cascade/union recalls and best-single-path gains.

    p is the probability that a uniformly selected GT line belongs to a
    missed block. The oracle union assumes no false positives and perfect
    matching. Independence is only used for the point estimate, not the
    Frechet bounds.
    """
    for name, value in (
        ("p_block_miss_line_weighted", p_block_miss_line_weighted),
        ("q_local_conditional_miss", q_local_conditional_miss),
        ("r_global_miss", r_global_miss),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    cascade = (1.0 - p_block_miss_line_weighted) * (1.0 - q_local_conditional_miss)
    global_only = 1.0 - r_global_miss
    independent_union = 1.0 - r_global_miss * (1.0 - cascade)
    frechet_lower = max(cascade, global_only)
    frechet_upper = min(1.0, cascade + global_only)
    best_single = max(cascade, global_only)
    return {
        "cascade_recall": cascade,
        "global_recall": global_only,
        "independent_oracle_union_recall": independent_union,
        "frechet_union_lower": frechet_lower,
        "frechet_union_upper": frechet_upper,
        "independent_gain_over_cascade": independent_union - cascade,
        "independent_gain_over_global": independent_union - global_only,
        "best_single_recall": best_single,
        "independent_gain_over_best_single": independent_union - best_single,
        "frechet_gain_over_best_single_lower": frechet_lower - best_single,
        "frechet_gain_over_best_single_upper": frechet_upper - best_single,
    }


def build_full_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    low, high, step = config["grid_min"], config["grid_max"], config["grid_step"]
    count = round((high - low) / step)
    values = [round(low + i * step, 10) for i in range(count + 1)]
    return [
        {
            "p": p,
            "q": q,
            "r": r,
            **line_recall_bounds(p, q, r),
        }
        for p, q, r in product(values, repeat=3)
    ]
