"""Paired bootstrap for ratios with potentially different denominators."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from fractions import Fraction

from .contracts import ContractError, require, string


_FIELDS = {
    "page_key",
    "document_key",
    "numerator_reference",
    "denominator_reference",
    "numerator_candidate",
    "denominator_candidate",
}
MAX_RESAMPLES = 100_000


def _finite_float(value, label):
    require(type(value) in (int, float), f"{label} must be finite and numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        require(False, f"{label} must be representable as a finite float")
    require(math.isfinite(result), f"{label} must be finite")
    return result


def _validate_rows(rows):
    require(isinstance(rows, list) and bool(rows), "rows: expected non-empty list")
    seen = set()
    clean = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == _FIELDS,
                f"row {index}: fields must be exactly {sorted(_FIELDS)}")
        require(string(row["page_key"]), f"row {index}: invalid page_key")
        require(row["document_key"] is None or string(row["document_key"]),
                f"row {index}: document_key must be a non-empty string or null")
        require(row["page_key"] not in seen, f"duplicate page_key: {row['page_key']}")
        seen.add(row["page_key"])
        for source in ("reference", "candidate"):
            numerator = row[f"numerator_{source}"]
            denominator = row[f"denominator_{source}"]
            number = _finite_float(numerator, f"row {index}: numerator_{source}")
            require(number >= 0, f"row {index}: numerator_{source} must be non-negative")
            require(type(denominator) is int and denominator >= 0,
                    f"row {index}: denominator_{source} must be a non-negative integer")
            require(denominator != 0 or numerator == 0,
                    f"row {index}: denominator_{source}=0 requires numerator_{source}=0")
        clean.append(dict(row))
    return clean


def _aggregate(rows):
    values = {}
    for source in ("reference", "candidate"):
        try:
            numerator = math.fsum(float(row[f"numerator_{source}"]) for row in rows)
        except (OverflowError, ValueError) as exc:
            raise ContractError(f"aggregate numerator_{source} is not finite") from exc
        require(math.isfinite(numerator), f"aggregate numerator_{source} is not finite")
        denominator = sum(row[f"denominator_{source}"] for row in rows)
        values[source] = {"numerator": numerator, "denominator": denominator}
    values["reference"]["estimate"] = (
        _finite_ratio(values["reference"]["numerator"], values["reference"]["denominator"], "reference")
        if values["reference"]["denominator"] else None
    )
    values["candidate"]["estimate"] = (
        _finite_ratio(values["candidate"]["numerator"], values["candidate"]["denominator"], "candidate")
        if values["candidate"]["denominator"] else None
    )
    return values


def _finite_ratio(numerator, denominator, label):
    try:
        result = float(Fraction(numerator) / denominator)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ContractError(f"aggregate {label} ratio is not finite") from exc
    require(math.isfinite(result), f"aggregate {label} ratio is not finite")
    return result


def _quantile(sorted_values, probability):
    """Linear interpolation at (n - 1) * p (type 7)."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    left, right = sorted_values[lower], sorted_values[upper]
    if left == 0 or right == 0 or (left < 0) != (right < 0):
        value = left * (1.0 - weight) + right * weight
    else:
        # Same-sign interpolation cannot mathematically leave [left,right];
        # this form avoids overflowing an otherwise finite convex combination.
        value = left + (right - left) * weight
    require(math.isfinite(value), "bootstrap quantile is not finite")
    return value


def _delta(rows, scale):
    totals = _aggregate(rows)
    ref, cand = totals["reference"]["estimate"], totals["candidate"]["estimate"]
    if ref is None or cand is None:
        return None, totals
    value = scale * (cand - ref)
    require(math.isfinite(value), "scaled ratio difference is not finite")
    return value, totals


def _resample_interval(rows, unit, scale, resamples, seed, confidence):
    if unit == "page":
        clusters = [[row] for row in rows]
    else:
        if any(row["document_key"] is None for row in rows):
            return {
                "unit": unit,
                "clusters": None,
                "status": "unknown_document_identity",
                "reason": "one or more rows lack an audited document key",
                "interval": None,
                "resamples_requested": resamples,
                "resamples_used": 0,
                "skipped_invalid_draws": 0,
                "conditional_on_valid_draws": False,
                "clusters_support_reference": None,
                "clusters_support_candidate": None,
                "clusters_support_both": None,
                "seed": seed,
            }
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["document_key"]].append(row)
        clusters = [grouped[key] for key in sorted(grouped)]

    support_reference = sum(
        sum(row["denominator_reference"] for row in cluster) > 0 for cluster in clusters
    )
    support_candidate = sum(
        sum(row["denominator_candidate"] for row in cluster) > 0 for cluster in clusters
    )
    support_both = sum(
        sum(row["denominator_reference"] for row in cluster) > 0
        and sum(row["denominator_candidate"] for row in cluster) > 0
        for cluster in clusters
    )

    point, totals = _delta(rows, scale)
    counts = {
        "unit": unit,
        "clusters": len(clusters),
        "pages": len(rows),
        "documents": None if any(r["document_key"] is None for r in rows) else len({r["document_key"] for r in rows}),
        "clusters_support_reference": support_reference,
        "clusters_support_candidate": support_candidate,
        "clusters_support_both": support_both,
        "numerator_reference": totals["reference"]["numerator"],
        "denominator_reference": totals["reference"]["denominator"],
        "numerator_candidate": totals["candidate"]["numerator"],
        "denominator_candidate": totals["candidate"]["denominator"],
        "estimate_reference": totals["reference"]["estimate"],
        "estimate_candidate": totals["candidate"]["estimate"],
        "delta_candidate_minus_reference": point,
        "scale": scale,
        "confidence": confidence,
        "resamples_requested": resamples,
        "resamples_used": 0,
        "skipped_invalid_draws": 0,
        "conditional_on_valid_draws": False,
        "seed": seed,
        "quantile_method": "linear interpolation at (n-1)*p (type 7)",
        "interval": None,
        "status": "insufficient_clusters" if len(clusters) < 2 else "ok",
        "small_n_warning": "Bootstrap coverage is not guaranteed merely because there are two clusters; inspect cluster count and heterogeneity.",
    }
    if point is None:
        counts["status"] = "zero_source_support"
        counts["reason"] = "at least one source has zero total denominator; ratio difference is undefined"
        return counts
    if len(clusters) < 2:
        return counts

    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draw_rows = [row for cluster in sampled for row in cluster]
        try:
            delta, _ = _delta(draw_rows, scale)
        except ContractError:
            # A finite input sample can still overflow in a bootstrap draw
            # that repeats a very large cluster. Report it as an invalid draw.
            delta = None
        if delta is None:
            counts["skipped_invalid_draws"] += 1
        else:
            values.append(delta)
    counts["resamples_used"] = len(values)
    counts["conditional_on_valid_draws"] = bool(counts["skipped_invalid_draws"] and values)
    if not values:
        counts["status"] = "no_valid_draws"
        counts["reason"] = "all bootstrap draws had zero support or non-finite aggregate statistics"
    else:
        values.sort()
        tail = (1.0 - confidence) / 2.0
        counts["interval"] = [_quantile(values, tail), _quantile(values, 1.0 - tail)]
        if counts["conditional_on_valid_draws"]:
            counts["status"] = "conditional_valid_draws"
            counts["reason"] = "interval is conditional on draws with positive denominator for both sources"
    return counts


def paired_ratio_intervals(
    rows,
    metric_id,
    *,
    scale=1.0,
    resamples=5000,
    seed=0,
    confidence=0.95,
):
    """Paired cluster-bootstrap for a difference between two ratios.

    Rows must already be aggregated to one page and have audited canonical
    page/document keys. A null document key explicitly suppresses the
    document-level interval; it is never treated as a page-sized document.
    The statistic is ``scale * (sum(numerator_candidate)/sum(denominator_candidate)
    - sum(numerator_reference)/sum(denominator_reference))``.
    ``resamples`` is capped at 100,000 to bound requested bootstrap work; this
    cap is a quantity limit, not a runtime guarantee.
    """
    require(string(metric_id), "metric_id must be a non-empty string")
    scale = _finite_float(scale, "scale")
    require(scale >= 0, "scale must be non-negative")
    require(type(resamples) is int and 0 < resamples <= MAX_RESAMPLES,
            f"resamples must be an integer between 1 and {MAX_RESAMPLES}")
    require(type(seed) is int, "seed must be an integer")
    confidence = _finite_float(confidence, "confidence")
    require(0 < confidence < 1, "confidence must be between 0 and 1")
    clean = _validate_rows(rows)
    point, totals = _delta(clean, scale)
    return {
        "metric_id": metric_id,
        "statistic": "scale * (candidate ratio - reference ratio)",
        "unit": "scaled ratio difference",
        "scale": scale,
        "estimate_reference": totals["reference"]["estimate"],
        "estimate_candidate": totals["candidate"]["estimate"],
        "delta_candidate_minus_reference": point,
        "sums": totals,
        "decision": None,
        "interval_method": "paired percentile cluster bootstrap; type-7 quantiles",
        "document": _resample_interval(clean, "document", scale, resamples, seed, confidence),
        "page": _resample_interval(clean, "page", scale, resamples, seed, confidence),
    }
