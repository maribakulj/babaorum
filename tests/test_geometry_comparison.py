from dataclasses import replace

import pytest

from babaorum.contracts import ContractError
from babaorum.geometry_comparison import GeometryPage, compare_geometry
from babaorum.line_geometry import GroundTruthLine, PredictedLine


def gt(name, y=20):
    return GroundTruthLine(name, [(0, y), (100, y)], 10, 'ground_truth')


def pred(name, y=20):
    return PredictedLine(name, [(0, y), (100, y)], 'predicted')


def page(key='p1', doc='d1', **kwargs):
    return GeometryPage(key, doc, **dict(ground_truth=[gt('g')], reference=[pred('r')],
                         candidate=[pred('c', 21)], gt_annotation_complete=True, **kwargs))


def test_geometry_common_support_and_distinct_precision_denominators():
    p1 = GeometryPage('p1', 'd1', [gt('g1'), gt('g2', 50)],
                      [pred('r1'), pred('r2', 52)], [pred('c1', 21)], True)
    p2 = GeometryPage('p2', 'd2', [gt('g3')], [pred('r3', 22)], [pred('c3', 21)], True)
    result = compare_geometry([p1, p2], resamples=40)
    primary = result['primary']
    distance = primary['metrics']['baseline_distance_common_gt']
    assert distance['estimate_reference'] == pytest.approx(.1)
    assert distance['estimate_candidate'] == pytest.approx(.1)
    assert distance['delta_candidate_minus_reference'] == pytest.approx(0)
    assert primary['pages'][0]['reference_only_matched_gt_ids'] == ['g2']
    recall = primary['metrics']['line_recall']
    assert recall['delta_candidate_minus_reference'] == pytest.approx(-100/3)
    precision = primary['metrics']['line_precision']
    assert precision['sums']['reference']['denominator'] == 3
    assert precision['sums']['candidate']['denominator'] == 2
    assert precision['delta_candidate_minus_reference'] == 0
    assert result['decision'] is None and result['data_authorization'] is None


def test_incomplete_gt_never_silently_drops_a_page():
    first = page()
    second = replace(page('p2', 'd2'), gt_annotation_complete=False)
    result = compare_geometry([first, second], resamples=10)
    for key in ('primary', 'sensitivity_1h'):
        assert result[key]['metrics'] is None
        assert result[key]['status'] == 'non_evaluable'
        assert result[key]['non_evaluable_pages'] == ['p2']
        assert len(result[key]['pages']) == 2


def test_missing_gt_geometry_overrides_claim_of_completeness():
    item = replace(page(), ground_truth=[GroundTruthLine('g', None, None, None, 'missing')])
    out = compare_geometry([item], resamples=10)
    assert out['primary']['metrics'] is None


def test_unknown_issue_and_same_issue_have_no_document_fallback():
    same = compare_geometry([page(), page('p2', 'd1')], resamples=10)
    m = same['primary']['metrics']['line_recall']
    assert m['document']['interval'] is None
    assert m['document']['status'] == 'insufficient_clusters'
    unknown = compare_geometry([page(), page('p2', None)], resamples=10)
    assert unknown['primary']['metrics']['line_recall']['document']['status'] == 'unknown_document_identity'


def test_fixed_sensitivity_exposes_censored_localization():
    shifted = replace(page(), candidate=[pred('c', 28)])
    result = compare_geometry([shifted], resamples=10)
    primary = result['primary']['metrics']
    assert primary['line_recall']['estimate_candidate'] == 0
    assert primary['baseline_distance_common_gt']['delta_candidate_minus_reference'] is None
    sensitivity = result['sensitivity_1h']['metrics']
    assert sensitivity['line_recall']['estimate_candidate'] == 1
    assert sensitivity['baseline_distance_common_gt']['estimate_candidate'] == pytest.approx(.8)
    assert 'sensitivity' in sensitivity['line_recall']['metric_id']


def test_missing_prediction_geometry_is_retained_in_precision_denominator():
    item = replace(page(), candidate=[pred('c'), PredictedLine('bad', None, None, 'error')])
    result = compare_geometry([item], resamples=10)
    m = result['primary']['metrics']['line_precision']
    assert m['estimate_candidate'] == .5
    assert m['sums']['candidate']['denominator'] == 2


def test_duplicate_pages_and_implicit_completeness_rejected():
    with pytest.raises(ContractError, match='duplicate page'):
        compare_geometry([page(), page()], resamples=10)
    with pytest.raises(ContractError, match='explicit bool'):
        compare_geometry([replace(page(), gt_annotation_complete=1)], resamples=10)


def test_orientation_counts_are_reported_per_system_and_do_not_change_distance():
    backward = replace(page(), candidate=[PredictedLine('c', [(100,20),(0,20)], 'predicted')])
    result = compare_geometry([backward, page('p2','d2')], resamples=10)
    for policy in ('primary', 'sensitivity_1h'):
        orientation = result[policy]['orientation_by_system']
        assert orientation['reference']['orientation_disagreement_count'] == 0
        assert orientation['candidate']['orientation_disagreement_count'] == 1
        assert orientation['candidate']['matched_count'] == 2
        assert result[policy]['metrics']['baseline_distance_common_gt']['estimate_candidate'] == pytest.approx(.05)


# --- G1 I2 / m6 (2026-10-08) -------------------------------------------------

def test_incomplete_gt_marks_orientation_diagnostic_and_withholds_all_rows():
    first = page()
    second = replace(page('p2', 'd2'), gt_annotation_complete=False)
    result = compare_geometry([first, second], resamples=10)
    for key in ('primary', 'sensitivity_1h'):
        block = result[key]
        for system in ('reference', 'candidate'):
            assert block['orientation_by_system'][system]['status'] == 'diagnostic_incomplete_gt'
        # Complete page p1 is not republished as an aggregable row (m6).
        assert block['rows'] is None
        assert block['metrics'] is None
        assert [p['status'] for p in block['pages']] == ['evaluated', 'non_evaluable']
        # Its page-level diagnostics stay visible.
        assert block['pages'][0]['common_matched_count'] == 1


def test_complete_gt_orientation_status_is_evaluated_and_rows_are_published():
    result = compare_geometry([page(), page('p2', 'd2')], resamples=10)
    for key in ('primary', 'sensitivity_1h'):
        block = result[key]
        for system in ('reference', 'candidate'):
            assert block['orientation_by_system'][system]['status'] == 'evaluated'
        assert [row['page_key'] for row in block['rows']['line_recall']] == ['p1', 'p2']
