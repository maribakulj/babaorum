"""Analytic recall bounds for a cascade and an oracle-reconciled global pass."""

from __future__ import annotations

from typing import Any


def recall_bounds(p_block_miss: float, q_local_miss: float, r_global_miss: float) -> dict[str, float]:
    """Return cascade recall, independent union, and Frechet union bounds.

    Marginal miss probabilities are treated as inputs. The union assumes
    oracle matching and no false positives; it is a ceiling, not a model.
    """
    for name, value in (
        ("p_block_miss", p_block_miss),
        ("q_local_miss", q_local_miss),
        ("r_global_miss", r_global_miss),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    cascade = (1.0 - p_block_miss) * (1.0 - q_local_miss)
    global_only = 1.0 - r_global_miss
    independent_union = 1.0 - r_global_miss * (1.0 - cascade)
    frechet_lower = max(cascade, global_only)
    frechet_upper = min(1.0, cascade + global_only)
    return {
        "cascade_recall": cascade,
        "global_recall": global_only,
        "independent_oracle_union_recall": independent_union,
        "frechet_union_lower": frechet_lower,
        "frechet_union_upper": frechet_upper,
        "independent_gain_over_cascade": independent_union - cascade,
    }


def build_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    low, high, step = config["grid_min"], config["grid_max"], config["grid_step"]
    count = round((high - low) / step)
    values = [round(low + i * step, 10) for i in range(count + 1)]
    rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        for p in values:
            for q in values:
                mode = scenario["global_miss_mode"]
                r = q if mode == "q" else scenario["global_miss"]
                result = recall_bounds(p, q, r)
                rows.append({"scenario": scenario["name"], "p": p, "q": q, "r": r, **result})
    return rows
