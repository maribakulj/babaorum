"""Paired geometric diagnostics, independent of benchmark/data authorization.

This numerical API does not read a manifest or approve data. Its caller must
supply audited identities, complete GT and already aligned pixel coordinates.
No missing-GT page is silently dropped from an aggregate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum
from typing import Sequence

from .contracts import require, string
from .line_geometry import (
    GroundTruthLine, PredictedLine, evaluate_line_geometry,
    evaluate_line_geometry_sensitivity,
)
from .ratio_comparison import paired_ratio_intervals


@dataclass(frozen=True)
class GeometryPage:
    page_key: str
    document_key: str | None
    ground_truth: Sequence[GroundTruthLine]
    reference: Sequence[PredictedLine]
    candidate: Sequence[PredictedLine]
    gt_annotation_complete: bool


def _row(page, nr, dr, nc, dc):
    return dict(page_key=page.page_key, document_key=page.document_key,
                numerator_reference=nr, denominator_reference=dr,
                numerator_candidate=nc, denominator_candidate=dc)


def _compare(pages, scorer, *, resamples, seed, confidence):
    reports = []
    rows = {name: [] for name in ('line_recall', 'line_precision', 'baseline_distance_common_gt')}
    non_evaluable = []
    for page in pages:
        ref, cand = scorer(page.ground_truth, page.reference), scorer(page.ground_truth, page.candidate)
        evaluable = page.gt_annotation_complete and ref.status == cand.status == 'evaluated'
        rm, cm = ({m.gt_id: m for m in r.matches} for r in (ref, cand))
        common = sorted(rm.keys() & cm.keys())
        rsum = fsum(rm[k].normalized_symmetric_distance for k in common)
        csum = fsum(cm[k].normalized_symmetric_distance for k in common)
        own = {}
        for key, matches in (('reference', rm), ('candidate', cm)):
            total = fsum(m.normalized_symmetric_distance for m in matches.values())
            own[key] = dict(sum_normalized_distance=total, support_lines=len(matches),
                            mean_normalized_distance=total / len(matches) if matches else None)
        reports.append(dict(
            page_key=page.page_key, document_key=page.document_key,
            status='evaluated' if evaluable else 'non_evaluable',
            gt_annotation_complete=page.gt_annotation_complete,
            reference=asdict(ref), candidate=asdict(cand),
            common_gt_ids=common, common_matched_count=len(common),
            reference_only_matched_gt_ids=sorted(rm.keys() - cm.keys()),
            candidate_only_matched_gt_ids=sorted(cm.keys() - rm.keys()),
            own_matched_geometry=own,
        ))
        if not evaluable:
            non_evaluable.append(page.page_key)
            continue
        rows['line_recall'].append(_row(page, ref.matched_count, ref.n_gt, cand.matched_count, cand.n_gt))
        rows['line_precision'].append(_row(page, ref.matched_count, ref.n_pred, cand.matched_count, cand.n_pred))
        rows['baseline_distance_common_gt'].append(_row(page, rsum, len(common), csum, len(common)))
    metrics = None
    if not non_evaluable:
        metrics = {name: paired_ratio_intervals(
            metric_rows, reports[0]['reference']['metric_id'] + ':' + name,
            scale=1.0 if name == 'baseline_distance_common_gt' else 100.0,
            resamples=resamples, seed=seed, confidence=confidence,
        ) for name, metric_rows in rows.items()}
    orientation = {}
    for system in ('reference', 'candidate'):
        orientation[system] = dict(
            status='diagnostic_incomplete_gt' if non_evaluable else 'evaluated',
            matched_count=sum(r[system]['matched_count'] for r in reports),
            orientation_disagreement_count=sum(r[system]['orientation_disagreement_count'] for r in reports),
            orientation_unresolved_count=sum(r[system]['orientation_unresolved_count'] for r in reports),
        )
    return dict(
        orientation_by_system=orientation,
        status='non_evaluable' if non_evaluable else 'evaluated',
        non_evaluable_pages=non_evaluable,
        # m6: with any incomplete-GT page, complete-page rows are withheld too,
        # so no caller can rebuild a favorable complete-case aggregate from them.
        metrics=metrics, rows=None if non_evaluable else rows, pages=reports,
        units={'line_recall': 'percentage_points', 'line_precision': 'percentage_points',
               'baseline_distance_common_gt': 'GT_band_heights'},
        favorable_delta={'line_recall': 'positive', 'line_precision': 'positive',
                         'baseline_distance_common_gt': 'negative'},
    )


def compare_geometry(pages, *, resamples=5000, seed=0, confidence=0.95):
    """Compare paired predictions, with a fixed and a wider-tolerance diagnostic.

    Primary localization compares the SAME GT identities matched in both
    systems. Thus it is conditional on detection, never an omission metric.
    Recall and precision use the full GT and emitted-prediction denominators.
    If any page has incomplete GT, all aggregate metrics AND the per-page
    aggregation ``rows`` (including those of complete pages) are withheld
    (``rows=None``) for BOTH systems; per-page diagnostics remain visible in
    ``pages`` and orientation counts carry ``diagnostic_incomplete_gt``. A null document identity
    withholds document CIs through paired_ratio_intervals, without page fallback.
    """
    require(isinstance(pages, (tuple, list)) and bool(pages), 'pages must be a non-empty list/tuple')
    seen = set()
    for page in pages:
        require(isinstance(page, GeometryPage), 'each page must be GeometryPage')
        require(string(page.page_key), 'invalid page_key')
        require(page.document_key is None or string(page.document_key), 'invalid document_key')
        require(page.page_key not in seen, 'duplicate page_key')
        require(type(page.gt_annotation_complete) is bool, 'gt_annotation_complete must be explicit bool')
        seen.add(page.page_key)
    kwargs = dict(resamples=resamples, seed=seed, confidence=confidence)
    return dict(
        schema_version='babaorum.geometry_comparison.v1',
        scope='GEOMETRY_NUMERICAL_DIAGNOSTIC', decision=None, data_authorization=None,
        primary=_compare(pages, evaluate_line_geometry, **kwargs),
        sensitivity_1h=_compare(pages, evaluate_line_geometry_sensitivity, **kwargs),
        limitations=[
            'Requires audited page/document identities and GT; this API does not certify G0 or authorize data.',
            'All baselines and explicit GT band heights must use the same original continuous-pixel-edge frame.',
            'Matched localization is conditional and censored by the matching tolerance; read with recall/precision.',
            'The fixed 1h sensitivity is always reported, never substituted for the primary 0.5h rule.',
            'Missing GT withholds all aggregate metrics; no favorable complete-case page subset is selected.',
            'Page bootstrap is diagnostic under within-issue dependence; no fallback when document CI is unavailable.',
            'Few independent issues and rare matched support can yield under-covering or conditional percentile CIs.',
            'No equivalence/noninferiority decision or acceptance margin is inferred here.',
        ],
    )
