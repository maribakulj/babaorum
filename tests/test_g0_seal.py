"""Sealing decisions for the presse_e7 test split (synthetic data only)."""
import pytest

from babaorum.g0_seal import (SealStop, check_headline_measure, decide, expect_counts, find_overlaps,
                              headline_pages)

OTHERS = {
    "presse_e7_train_dev": {"page": {"scan/x/f1"}, "issue": {"gallica:111"}, "title": {"mercure"}},
    "bnl_pack": {"page": None, "issue": None, "title": None},
    "pleias": {"page": None, "issue": {"gallica:222"}, "title": None},
}


def test_exact_counts_stop_the_seal_when_they_differ():
    expect_counts({"pages": 454, "with_lines": 410, "without_lines": 44}, {"pages": 454, "with_lines": 410, "without_lines": 44})
    with pytest.raises(SealStop, match="with_lines=409!=410"):
        expect_counts({"pages": 454, "with_lines": 409, "without_lines": 45}, {"pages": 454, "with_lines": 410})


def test_find_overlaps_keeps_every_match_at_every_level():
    page = {"page": ["scan/x/f1"], "issue": ["gallica:111", "gallica:222"], "title": ["mercure"]}
    found = find_overlaps(page, OTHERS)
    assert {(o["collection"], o["level"]) for o in found} == {
        ("presse_e7_train_dev", "page"), ("presse_e7_train_dev", "issue"), ("presse_e7_train_dev", "title"),
        ("pleias", "issue")}


def test_a_collection_without_keys_never_matches_and_never_hides_a_match():
    assert find_overlaps({"issue": ["gallica:999"]}, OTHERS) == []
    assert find_overlaps({"issue": ["gallica:222"]}, {"bnl_pack": OTHERS["bnl_pack"], **{"pleias": OTHERS["pleias"]}})[0]["collection"] == "pleias"


def test_page_without_truth_lines_is_refused_even_when_nothing_overlaps():
    result = decide(truth_lines=0, issue_keys=["gallica:5"], overlaps=[])
    assert result["decision"] == "refused" and result["refusal_reasons"] == ["no_truth_lines"] and result["headline"] is False


def test_page_or_issue_overlap_removes_the_page_from_the_seal_and_keeps_the_matches():
    overlaps = find_overlaps({"issue": ["gallica:111"]}, OTHERS)
    result = decide(truth_lines=10, issue_keys=["gallica:111"], overlaps=overlaps)
    assert result["decision"] == "refused" and result["refusal_reasons"] == ["overlap_issue"]
    assert result["overlaps"] == overlaps


def test_clean_credited_single_document_page_is_headline():
    result = decide(truth_lines=10, issue_keys=["gallica:5"], overlaps=[])
    assert result["decision"] == "sealed" and result["headline"] is True and result["headline_false_reasons"] == []


def test_montage_uncredited_and_title_seen_pages_are_sealed_with_headline_false():
    montage = decide(truth_lines=5, issue_keys=["gallica:5", "gallica:6"], overlaps=[])
    assert montage["decision"] == "sealed" and montage["headline"] is False and montage["montage"] is True
    uncredited = decide(truth_lines=5, issue_keys=[], overlaps=[])
    assert uncredited["headline"] is False and uncredited["headline_false_reasons"] == ["uncredited"]
    seen = decide(truth_lines=5, issue_keys=["gallica:5"],
                  overlaps=find_overlaps({"title": ["mercure"]}, OTHERS))
    assert seen["decision"] == "sealed" and seen["title_seen"] is True and seen["headline"] is False
    assert seen["headline_false_reasons"] == ["title_seen"]


def test_a_headline_measure_with_a_non_headline_page_fails():
    sealed = [{"page_id": "a", "headline": True}, {"page_id": "b", "headline": False}]
    assert headline_pages(sealed) == ["a"]
    check_headline_measure(["a"], sealed)
    with pytest.raises(SealStop, match="not headline=true"):
        check_headline_measure(["a", "b"], sealed)
    with pytest.raises(SealStop):
        check_headline_measure(["unknown"], sealed)
