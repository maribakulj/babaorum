"""Bench part b helpers: tau with support, 'ligne réussie', ALTO checks (synthetic data)."""
import pytest

from babaorum.bench_metrics import (ORDER_NOT_APPLICABLE_PAGES, alto_checks, kendall_tau, levenshtein, line_cer,
                                    line_success, order_report, page_checks, texts_from_alto, texts_from_page)

PAGE = "La_Presse_0512130_T0000003"
GT_PAGE_NA = "Regards_7635840_D0000005"


def test_levenshtein_and_line_cer():
    assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "abc") == 3 and levenshtein("abc", "abc") == 0
    assert line_cer("abcd", "abxd") == pytest.approx(0.25)
    assert line_cer("", "") == 0.0 and line_cer("", "x") == 1.0


def test_kendall_tau_extremes_and_small_support():
    assert kendall_tau([(0, 0), (1, 1), (2, 2)]) == 1.0
    assert kendall_tau([(0, 2), (1, 1), (2, 0)]) == -1.0
    assert kendall_tau([(0, 0), (1, 2), (2, 1)]) == pytest.approx(1 / 3)
    assert kendall_tau([(0, 0)]) is None and kendall_tau([]) is None


def orders(ids):
    return {i: n for n, i in enumerate(ids)}


def test_order_report_gives_tau_and_support():
    gt = orders(["g1", "g2", "g3", "g4"])
    pred = orders(["p2", "p1", "p3"])
    matches = [("g1", "p1"), ("g2", "p2"), ("g3", "p3")]
    report = order_report(PAGE, matches, gt, pred, n_gt=4)
    assert report["status"] == "computed"
    assert report["tau"] == pytest.approx(1 / 3)
    assert report["support"] == {"matched": 3, "gt_lines": 4, "ratio": 0.75}


@pytest.mark.parametrize("page", sorted(ORDER_NOT_APPLICABLE_PAGES))
def test_order_is_not_applicable_on_the_two_1937_pages(page):
    report = order_report(page, [("g1", "p1"), ("g2", "p2")], orders(["g1", "g2"]), orders(["p2", "p1"]), n_gt=2)
    assert report["status"] == "non applicable" and report["tau"] is None
    assert report["support"]["matched"] == 2  # the support is still published


def test_line_success_requires_match_cer_under_15_percent_and_same_rank():
    gt_text = {"g1": "abcdefghij", "g2": "klmnopqrst", "g3": "uvwxyzabcd", "g4": "never matched"}
    pred_text = {"p1": "abcdefghij", "p2": "klmnopqrXX", "p3": "uvwxyzabcd"}  # p2 has CER 0.2
    gt_order, pred_order = orders(["g1", "g2", "g3", "g4"]), orders(["p1", "p2", "p3"])
    matches = [("g1", "p1"), ("g2", "p2"), ("g3", "p3")]
    result = line_success(PAGE, matches, gt_text, pred_text, gt_order, pred_order, n_gt=4)
    assert result["matched"] == 3 and result["matched_and_cer_under_threshold"] == 2
    assert result["matched_cer_and_same_rank"] == 2
    assert result["share_success_with_order"] == 0.5 and result["order_condition"] == "applicable"


def test_line_success_rank_condition_fails_when_the_order_is_swapped():
    gt_text = {"g1": "aaaa", "g2": "bbbb"}
    pred_text = {"p1": "aaaa", "p2": "bbbb"}
    swapped = line_success(PAGE, [("g1", "p1"), ("g2", "p2")], gt_text, pred_text,
                           orders(["g1", "g2"]), orders(["p2", "p1"]), n_gt=2)
    assert swapped["matched_and_cer_under_threshold"] == 2 and swapped["matched_cer_and_same_rank"] == 0


def test_line_success_marks_order_not_applicable_but_keeps_the_other_conditions():
    result = line_success(GT_PAGE_NA, [("g1", "p1")], {"g1": "abcd"}, {"p1": "abcd"},
                          orders(["g1"]), orders(["p1"]), n_gt=1)
    assert result["order_condition"] == "non applicable" and result["matched_cer_and_same_rank"] is None
    assert result["share_success_with_order"] is None and result["share_matched_and_cer"] == 1.0


ALTO_OK = ('<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page><PrintSpace><TextBlock ID="b1">'
           '<TextLine ID="l1" IDNEXT="l2" BASELINE="0 10 50 10"/><TextLine ID="l2" BASELINE="0 30 50 30"/>'
           '</TextBlock></PrintSpace></Page></Layout></alto>')


def test_alto_checks_baseline_order_and_lost_lines():
    checks = alto_checks(ALTO_OK, graph_line_count=3)
    assert checks["text_lines"] == 2 and checks["baseline_missing"] == 0
    assert checks["order"] == "consistent" and checks["lines_lost"] == 1


def test_alto_checks_detect_missing_baseline_dangling_and_cyclic_order():
    no_baseline = ALTO_OK.replace(' BASELINE="0 30 50 30"', "")
    assert alto_checks(no_baseline, 2)["baseline_missing"] == 1
    dangling = ALTO_OK.replace('IDNEXT="l2"', 'IDNEXT="zzz"')
    assert alto_checks(dangling, 2)["order"] == "inconsistent"
    cyclic = ALTO_OK.replace('<TextLine ID="l2" BASELINE', '<TextLine ID="l2" IDNEXT="l1" BASELINE')
    assert alto_checks(cyclic, 2)["order"] == "inconsistent"
    plain = ALTO_OK.replace(' IDNEXT="l2"', "")
    assert alto_checks(plain, 2)["order"] == "not_present"


def test_alto_checks_when_nothing_was_exported():
    assert alto_checks(None, 5) == {"status": "not_exported", "lines_lost": 5}


PAGE_OK = ('<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"><Page><TextRegion id="r1">'
           '<TextLine id="l1"><Baseline points="0,10 50,10"/></TextLine><TextLine id="l2"/></TextRegion></Page></PcGts>')


def test_page_checks_count_lines_baselines_and_lost_lines():
    checks = page_checks(PAGE_OK, graph_line_count=3)
    assert checks["text_lines"] == 2 and checks["baseline_present"] == 1 and checks["baseline_missing"] == 1
    assert checks["lines_lost"] == 1 and checks["reading_order_element"] is False
    assert page_checks(None, 4) == {"status": "not_exported", "lines_lost": 4}


ALTO_WORDS = ('<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page><PrintSpace><TextBlock ID="b1">'
              '<TextLine ID="l1"><String CONTENT="ab"/><SP/><String CONTENT="cd"/></TextLine>'
              '<TextLine ID="l2"><String CONTENT="x"/></TextLine></TextBlock></PrintSpace></Page></Layout></alto>')
PAGE_TEXT = ('<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"><Page><TextRegion id="r1">'
             '<TextLine id="l1"><TextEquiv><Unicode>ab cd</Unicode></TextEquiv></TextLine>'
             '<TextLine id="l2"><TextEquiv><Unicode>x</Unicode></TextEquiv></TextLine></TextRegion></Page></PcGts>')


def test_text_is_read_back_from_alto_and_page_in_file_order():
    assert texts_from_alto(ALTO_WORDS) == ["ab cd", "x"]
    assert texts_from_page(PAGE_TEXT) == ["ab cd", "x"]


def test_same_predecessor_variant_survives_one_extra_line_that_breaks_the_rank_condition():
    gt_text = {f"g{i}": f"line{i:02d}" * 2 for i in range(1, 5)}
    pred_text = {f"p{i}": f"line{i:02d}" * 2 for i in range(1, 5)}
    pred_text["pX"] = "an invented line"  # one extra predicted line, unmatched, placed first
    gt_order = orders(["g1", "g2", "g3", "g4"])
    pred_order = orders(["pX", "p1", "p2", "p3", "p4"])
    matches = [("g1", "p1"), ("g2", "p2"), ("g3", "p3"), ("g4", "p4")]
    result = line_success(PAGE, matches, gt_text, pred_text, gt_order, pred_order, n_gt=4)
    # the unmatched extra line does not shift ranks among MATCHED lines: both conditions hold for every line
    assert result["matched_cer_and_same_rank"] == 4 and result["matched_cer_and_same_predecessor"] == 4
    # a missed line in the middle of the prediction: ranks among matched lines shift, the predecessor of the
    # line after the gap is no longer the matched predecessor of its truth line
    matches2 = [("g1", "p1"), ("g3", "p3"), ("g4", "p4")]
    gap = line_success(PAGE, matches2, gt_text, pred_text, gt_order, orders(["p1", "p3", "p4"]), n_gt=4)
    assert gap["matched_cer_and_same_rank"] == 3 and gap["matched_cer_and_same_predecessor"] == 3
    # two matched lines whose predicted ORDER is inverted (texts equal): both order variants fail for both lines
    inverted = line_success(PAGE, [("g1", "p1"), ("g2", "p2")], gt_text, pred_text, orders(["g1", "g2"]), orders(["p2", "p1"]), n_gt=2)
    assert inverted["matched_and_cer_under_threshold"] == 2
    assert inverted["matched_cer_and_same_rank"] == 0 and inverted["matched_cer_and_same_predecessor"] == 0
    # same texts but the matching itself follows the inversion: the predecessor relation is preserved
    crossed = line_success(PAGE, [("g1", "p1"), ("g2", "p2")], gt_text, pred_text, orders(["g1", "g2"]), orders(["p1", "p2"]), n_gt=2)
    assert crossed["matched_cer_and_same_predecessor"] == 2


def test_same_predecessor_is_not_applicable_on_the_1937_pages():
    result = line_success(GT_PAGE_NA, [("g1", "p1")], {"g1": "abcd"}, {"p1": "abcd"}, orders(["g1"]), orders(["p1"]), n_gt=1)
    assert result["matched_cer_and_same_predecessor"] is None and result["share_success_with_predecessor"] is None
