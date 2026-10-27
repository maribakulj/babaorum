import math

import pytest

from babaorum.contracts import ContractError
from babaorum.ratio_comparison import _quantile, paired_ratio_intervals


def row(page, doc, nr, dr, nc, dc):
    return {
        "page_key": page,
        "document_key": doc,
        "numerator_reference": nr,
        "denominator_reference": dr,
        "numerator_candidate": nc,
        "denominator_candidate": dc,
    }


def test_ratio_uses_separate_micro_denominators_and_scale():
    result = paired_ratio_intervals([
        row("p1", "d1", 1, 10, 3, 20),
        row("p2", "d2", 9, 90, 5, 10),
    ], "recall", scale=100, resamples=100, seed=2)
    assert result["estimate_reference"] == pytest.approx(.1)
    assert result["estimate_candidate"] == pytest.approx(8 / 30)
    assert result["delta_candidate_minus_reference"] == pytest.approx((8 / 30 - .1) * 100)
    assert result["sums"]["reference"] == {"numerator": 10, "denominator": 100, "estimate": .1}
    assert result["sums"]["candidate"] == {"numerator": 8, "denominator": 30, "estimate": 8 / 30}


def test_document_bootstrap_samples_whole_issues_and_page_is_diagnostic():
    rows = [
        row("p1", "issue-a", 1, 10, 2, 10),
        row("p2", "issue-a", 3, 10, 4, 10),
        row("p3", "issue-b", 8, 10, 6, 10),
        row("p4", "issue-b", 2, 10, 4, 10),
    ]
    result = paired_ratio_intervals(rows, "precision", resamples=200, seed=4)
    assert result["document"]["clusters"] == 2
    assert result["document"]["documents"] == 2
    assert result["document"]["pages"] == 4
    assert result["page"]["clusters"] == 4
    duplicated = paired_ratio_intervals(rows + [row("p5", "issue-a", 1, 10, 2, 10)], "precision", resamples=200, seed=4)
    assert duplicated["document"]["clusters"] == 2
    assert duplicated["page"]["clusters"] == 5


def test_support_clusters_report_reference_candidate_and_overlap_separately():
    result = paired_ratio_intervals([
        row("p1", "d1", 0, 10, 0, 0),
        row("p2", "d1", 0, 0, 0, 5),
        row("p3", "d2", 0, 3, 0, 4),
    ], "recall", resamples=5, seed=2)
    assert result["page"]["clusters_support_reference"] == 2
    assert result["page"]["clusters_support_candidate"] == 2
    assert result["page"]["clusters_support_both"] == 1
    assert result["document"]["clusters_support_reference"] == 2
    assert result["document"]["clusters_support_candidate"] == 2
    assert result["document"]["clusters_support_both"] == 2


def test_swapping_sources_reverses_estimate_and_interval():
    rows = [row("p1", "d1", 1, 10, 4, 10), row("p2", "d2", 3, 10, 2, 10), row("p3", "d3", 1, 10, 5, 10)]
    forward = paired_ratio_intervals(rows, "distance", scale=1, resamples=400, seed=51)
    reverse = paired_ratio_intervals([
        row(r["page_key"], r["document_key"], r["numerator_candidate"], r["denominator_candidate"],
            r["numerator_reference"], r["denominator_reference"])
        for r in rows
    ], "distance", scale=1, resamples=400, seed=51)
    assert reverse["delta_candidate_minus_reference"] == pytest.approx(-forward["delta_candidate_minus_reference"])
    assert reverse["document"]["interval"] == pytest.approx([
        -forward["document"]["interval"][1], -forward["document"]["interval"][0]
    ])


def test_missing_document_identity_withholds_only_document_interval():
    result = paired_ratio_intervals([
        row("p1", None, 1, 2, 1, 3),
        row("p2", "d2", 1, 2, 2, 3),
    ], "recall", resamples=50)
    assert result["page"]["interval"] is not None
    assert result["document"]["status"] == "unknown_document_identity"
    assert result["document"]["interval"] is None
    assert result["document"]["clusters"] is None
    assert result["document"]["clusters_support_reference"] is None


def test_zero_origin_support_has_null_estimates_and_intervals():
    result = paired_ratio_intervals([
        row("p1", "d1", 0, 0, 0, 0),
        row("p2", "d2", 0, 0, 0, 0),
    ], "recall", resamples=100)
    assert result["estimate_reference"] is None
    assert result["estimate_candidate"] is None
    assert result["delta_candidate_minus_reference"] is None
    for unit in ("page", "document"):
        assert result[unit]["status"] == "zero_source_support"
        assert result[unit]["interval"] is None
        assert result[unit]["resamples_used"] == 0


def test_rare_support_reports_skips_and_conditional_interval():
    result = paired_ratio_intervals([
        row("p1", "supported", 1, 1, 1, 1),
        row("p2", "zero", 0, 0, 0, 0),
    ], "recall", resamples=500, seed=17)
    doc = result["document"]
    assert doc["skipped_invalid_draws"] > 0
    assert doc["resamples_used"] + doc["skipped_invalid_draws"] == 500
    assert doc["conditional_on_valid_draws"] is True
    assert doc["status"] == "conditional_valid_draws"
    assert doc["interval"] is not None


@pytest.mark.parametrize("rows,kwargs", [
    ([row("p", "d", 0, 1, 0, 1), {**row("p", "d", 0, 1, 0, 1)}], {}),
    ([{**row("p", "d", 0, 1, 0, 1), "other": 0}], {}),
    ([row("p", "d", -1, 1, 0, 1)], {}),
    ([row("p", "d", math.inf, 1, 0, 1)], {}),
    ([row("p", "d", 1, 0, 0, 1)], {}),
    ([row("p", "d", 0, 1.5, 0, 1)], {}),
    ([row("p", "d", 0, 1, 0, 1)], {"scale": -1}),
    ([row("p", "d", 0, 1, 0, 1)], {"resamples": 0}),
    ([row("p", "d", 0, 1, 0, 1)], {"seed": True}),
    ([row("p", "d", 0, 1, 0, 1)], {"confidence": 1}),
])
def test_invalid_inputs_rejected(rows, kwargs):
    with pytest.raises(ContractError):
        paired_ratio_intervals(rows, "metric", **kwargs)


def test_seed_determinism_and_small_n_warning():
    rows = [row("p1", "d1", 1, 2, 2, 3), row("p2", "d2", 2, 3, 1, 2)]
    a = paired_ratio_intervals(rows, "precision", resamples=50, seed=6)
    b = paired_ratio_intervals(rows, "precision", resamples=50, seed=6)
    assert a == b
    assert "not guaranteed" in a["document"]["small_n_warning"]
    assert a["document"]["clusters"] == 2


def test_large_integers_raise_contract_error_instead_of_overflow_error():
    with pytest.raises(ContractError, match="representable as a finite float"):
        paired_ratio_intervals([row("p", "d", 10**10000, 1, 0, 1)], "metric", resamples=1)
    with pytest.raises(ContractError, match="representable as a finite float"):
        paired_ratio_intervals([row("p", "d", 0, 1, 0, 1)], "metric", scale=10**10000, resamples=1)


def test_nonfinite_aggregate_and_scaled_delta_are_rejected():
    with pytest.raises(ContractError, match="aggregate numerator_reference is not finite"):
        paired_ratio_intervals([
            row("p1", "d1", 1e308, 1, 0, 1),
            row("p2", "d2", 1e308, 1, 0, 1),
        ], "metric", resamples=1)
    with pytest.raises(ContractError, match="scaled ratio difference is not finite"):
        paired_ratio_intervals([row("p", "d", 0, 1, 1e308, 1)], "metric", scale=2, resamples=1)


def test_large_finite_type7_quantile_stays_finite():
    value = _quantile([1.7e308, 1.79e308], 0.5)
    assert math.isfinite(value)
    assert 1.7e308 <= value <= 1.79e308


def test_resamples_work_cap_is_enforced_without_running_bootstrap():
    with pytest.raises(ContractError, match="between 1 and 100000"):
        paired_ratio_intervals([row("p", "d", 0, 1, 0, 1)], "metric", resamples=100001)


def test_nonfinite_resampled_aggregates_are_counted_as_invalid_draws():
    result = paired_ratio_intervals([
        row("p1", "d1", 0, 1, 0, 1),
        row("p2", "d2", 0, 1, 9e307, 1),
    ], "metric", resamples=50, seed=0)
    for unit in ("page", "document"):
        interval = result[unit]
        assert interval["skipped_invalid_draws"] > 0
        assert interval["conditional_on_valid_draws"] is True
        assert interval["status"] == "conditional_valid_draws"
