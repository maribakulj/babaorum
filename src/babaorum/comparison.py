"""Paired cached-reader comparison with page/document-cluster bootstrap."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from .contracts import (
    ContractError,
    allow_development,
    number,
    parse,
    require,
    sha,
    string,
    validate_manifest,
    validate_predictions,
)
from .identity import IdentityError, identities, canonical_issue


_ROW_FIELDS = {"page_key", "document_key", "support", "errors_reference", "errors_candidate"}


def _validate_stat_rows(rows):
    require(isinstance(rows, list) and bool(rows), "rows: expected non-empty list")
    seen_pages = set()
    page_docs = {}
    normalized = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == _ROW_FIELDS,
                f"row {index}: fields must be exactly {sorted(_ROW_FIELDS)}")
        require(string(row["page_key"]), f"row {index}: invalid page_key")
        require(row["document_key"] is None or string(row["document_key"]),
                f"row {index}: invalid document_key (use null only when audited issue identity is unavailable)")
        require(row["page_key"] not in seen_pages, f"duplicate page_key: {row['page_key']}")
        seen_pages.add(row["page_key"])
        require(row["page_key"] not in page_docs or page_docs[row["page_key"]] == row["document_key"],
                f"page mapped to multiple documents: {row['page_key']}")
        page_docs[row["page_key"]] = row["document_key"]
        for field in ("support", "errors_reference", "errors_candidate"):
            value = row[field]
            require(type(value) is int and value >= 0, f"row {index}: {field} must be a non-negative integer")
        if row["support"] == 0:
            require(row["errors_reference"] == 0 and row["errors_candidate"] == 0,
                    f"row {index}: zero-support rows must have zero errors")
        normalized.append(dict(row))
    return normalized


def _ratio(rows):
    support = sum(row["support"] for row in rows)
    delta = sum(row["errors_candidate"] - row["errors_reference"] for row in rows)
    return (100.0 * delta / support if support else None), support


def _quantile(sorted_values, probability):
    """Linear interpolation at (n - 1) * p, matching common type-7 quantiles."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap(rows, unit, resamples, seed, confidence):
    if unit == "page":
        clusters = [[row] for row in rows]
    else:
        if any(row["document_key"] is None for row in rows):
            _, support = _ratio(rows)
            return {
                "unit": unit,
                "clusters": None,
                "pages": len(rows),
                "documents": None,
                "support_chars": support,
                "errors_reference": sum(row["errors_reference"] for row in rows),
                "errors_candidate": sum(row["errors_candidate"] for row in rows),
                "delta_candidate_minus_reference_pp": _ratio(rows)[0],
                "confidence": confidence,
                "resamples_requested": resamples,
                "resamples_used": 0,
                "skipped_zero_support_draws": 0,
                "seed": seed,
                "quantile_method": "linear interpolation at (n-1)*p (type 7)",
                "interval_pp": None,
                "status": "unknown_document_identity",
                "reason": "one or more rows lack an audited issue/document key; no document-level CI reported",
            }
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["document_key"]].append(row)
        clusters = [grouped[key] for key in sorted(grouped)]

    estimate, support = _ratio(rows)
    result = {
        "unit": unit,
        "clusters": len(clusters),
        "pages": len(rows),
        "documents": None if any(row["document_key"] is None for row in rows) else len({row["document_key"] for row in rows}),
        "support_chars": support,
        "errors_reference": sum(row["errors_reference"] for row in rows),
        "errors_candidate": sum(row["errors_candidate"] for row in rows),
        "delta_candidate_minus_reference_pp": estimate,
        "confidence": confidence,
        "resamples_requested": resamples,
        "resamples_used": 0,
        "skipped_zero_support_draws": 0,
        "seed": seed,
        "quantile_method": "linear interpolation at (n-1)*p (type 7)",
        "interval_pp": None,
        "status": "insufficient_clusters" if len(clusters) < 2 else "ok",
    }
    if len(clusters) < 2:
        return result

    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draw_rows = [row for cluster in sampled for row in cluster]
        value, draw_support = _ratio(draw_rows)
        if draw_support == 0:
            result["skipped_zero_support_draws"] += 1
        else:
            values.append(value)
    result["resamples_used"] = len(values)
    if values:
        values.sort()
        tail = (1.0 - confidence) / 2.0
        result["interval_pp"] = [_quantile(values, tail), _quantile(values, 1.0 - tail)]
    else:
        result["status"] = "no_valid_draws"
    return result


def paired_intervals(rows, metric_id, resamples=5000, seed=0, confidence=0.95):
    """Bootstrap candidate-minus-reference micro-CER difference.

    Rows are already aggregated to one paired observation per page, and
    document_key values must already be canonical audited issue/document
    identities. Use null only when that identity is unavailable; then the
    document interval is explicitly withheld. The
    page interval resamples pages; the document interval resamples documents
    and carries all pages of each sampled document together. Difference is
    ``100 * sum(candidate_errors-reference_errors) / sum(support)`` in CER
    percentage points. This returns descriptive intervals, never a decision.
    """
    require(string(metric_id), "metric_id must be a non-empty string")
    require(type(resamples) is int and resamples > 0, "resamples must be a positive integer")
    require(type(seed) is int, "seed must be an integer")
    require(number(confidence) and 0.0 < confidence < 1.0, "confidence must be between 0 and 1")
    rows = _validate_stat_rows(rows)
    return {
        "metric_id": metric_id,
        "estimate_definition": "candidate CER minus reference CER; micro CER difference in percentage points",
        "decision": None,
        "page": _bootstrap(rows, "page", resamples, seed, confidence),
        "document": _bootstrap(rows, "document", resamples, seed, confidence),
    }


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _metric_rows(manifest, reference, candidate, metric_id):
    from .legacy.reader.metric import NORMS, line_errors

    pages = {}
    system_counts = {"reference": {"ok": 0, "missing": 0, "error": 0},
                     "candidate": {"ok": 0, "missing": 0, "error": 0}}
    for counts in system_counts.values():
        counts["empty_truth_lines"] = 0
        counts["empty_truth_output_chars"] = 0
    for line in manifest["lines"]:
        line_id = line["id"]
        outputs = []
        for name, predictions in (("reference", reference), ("candidate", candidate)):
            prediction = predictions.get(line_id)
            if prediction is None:
                system_counts[name]["missing"] += 1
                text = ""  # Omission is suppression, and is still a covered manifest line.
            elif prediction["status"] == "error":
                system_counts[name]["error"] += 1
                text = ""  # Failed inference is scored as suppressed output.
            else:
                system_counts[name]["ok"] += 1
                text = prediction["text"]
            if not NORMS[metric_id](line["text"]):
                system_counts[name]["empty_truth_lines"] += 1
                system_counts[name]["empty_truth_output_chars"] += len(NORMS[metric_id](text))
            errors, support = line_errors(text, line["text"], metric_id)
            outputs.append((errors, support))
        if manifest["exposure"] == "fixture":
            page_key, document_key = line["page_id"], line["document_key"]
        else:
            try:
                page_key, page_issue = identities(line["page_id"])
                declared_issue = canonical_issue(line["document_key"])
            except IdentityError as exc:
                raise ContractError(f"unresolved C7 identity in comparison: {exc}") from exc
            require(page_issue == declared_issue,
                    f"page/document issue mismatch for {line['page_id']}: {page_issue} != {declared_issue}")
            # KB identities identify pages, not newspaper issues. Do not use
            # page-shaped KB keys as independent document bootstrap clusters.
            document_key = page_issue
        page = pages.setdefault(page_key, {
            "page_key": page_key,
            "document_key": document_key,
            "support": 0,
            "errors_reference": 0,
            "errors_candidate": 0,
        })
        require(page["document_key"] == document_key, "page has inconsistent document identity")
        require(outputs[0][1] == outputs[1][1], "reference/candidate support mismatch")
        page["support"] += outputs[0][1]
        page["errors_reference"] += outputs[0][0]
        page["errors_candidate"] += outputs[1][0]
    rows = [pages[key] for key in sorted(pages)]
    for name, counts in system_counts.items():
        counts["total_lines"] = len(manifest["lines"])
        counts["successful_lines"] = counts["ok"]
        counts["coverage"] = counts["ok"] / len(manifest["lines"])
    return rows, system_counts


def compare(manifest_path, reference_path, candidate_path, resamples=5000, seed=0, confidence=0.95):
    """Compare two cached same-crop reader outputs; no reader/model is run."""
    manifest_path = Path(manifest_path)
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = _sha256(manifest_bytes)
    manifest = validate_manifest(parse(manifest_bytes))
    allow_development(manifest)

    # Do not touch prediction/cache paths until the manifest is accepted.
    reference_bytes = reference_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    reference_hash = _sha256(reference_bytes)
    candidate_hash = _sha256(candidate_bytes)
    reference_data = parse(reference_bytes)
    candidate_data = parse(candidate_bytes)
    reference = validate_predictions(reference_data, manifest, manifest_hash)
    candidate = validate_predictions(candidate_data, manifest, manifest_hash)

    metrics = {}
    from .legacy.reader.metric import NORMS
    for metric_id in NORMS:
        rows, counts = _metric_rows(manifest, reference, candidate, metric_id)
        intervals = paired_intervals(rows, metric_id, resamples=resamples, seed=seed, confidence=confidence)
        point = intervals["page"]["delta_candidate_minus_reference_pp"]
        row_support = sum(row["support"] for row in rows)
        errors_reference = sum(row["errors_reference"] for row in rows)
        errors_candidate = sum(row["errors_candidate"] for row in rows)
        metrics[metric_id] = {
            "cer_reference_pct": 100.0 * errors_reference / row_support if row_support else None,
            "cer_candidate_pct": 100.0 * errors_candidate / row_support if row_support else None,
            "delta_candidate_minus_reference_pp": point,
            "support_chars": row_support,
            "errors_reference": errors_reference,
            "errors_candidate": errors_candidate,
            "coverage": counts,
            "paired_intervals": intervals,
            "rows": rows,
        }

    return {
        "schema_version": "0.1.0",
        "task": "same_crops_reader_comparison",
        "scope": "READER_ONLY",
        "manifest_sha256": manifest_hash,
        "exposure": manifest["exposure"],
        "reference": {"predictions_sha256": reference_hash, "model": reference_data["model"]},
        "candidate": {"predictions_sha256": candidate_hash, "model": candidate_data["model"]},
        "units": "CER in percent; paired differences in percentage points; coverage in [0,1]",
        "omission_policy": "missing/error prediction is scored as empty text (suppression); missing/error counts and successful-line coverage are reported separately",
        "normalization_source": "babaorum.legacy.reader.metric.line_errors / NORMS; normalization is not reimplemented here",
        "confidence_intervals": "percentile bootstrap; paired by page and by document; descriptive only, no equivalence decision",
        "limits": ["same-crop reader comparison only", "no layout/detector/segmentation CER", "D-150 and raw transcription metrics only"],
        "metrics": metrics,
    }
