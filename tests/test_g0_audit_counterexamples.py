"""LOT-G0-02 lot 1: the six counter-examples of the audit of presse_e7_test_v1 (pQ, pM), as tests.

Each test reproduces in memory a case of audit_codex/adversarial.py (nothing is written under audit_codex/)
and asserts the REFUSAL the audit expects. They were written first and failed on the code that was audited.
The four controls the audit found correct must stay correct. The frozen v1 files are read, never changed.

Decision of p1 (section v3 of LOT-G0-02.md): with the registry v2 (gallica_scope), rules 1 and 2 refuse EVERY page today, since
no collection with gallica_scope=possible has complete number keys. The positive control (a headline page admitted) is therefore
redefined on a registry fabricated IN MEMORY where those keys are complete, so that the guard stays testable both ways.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from babaorum.g0_collections import CollectionRegistryError, eligibility, resolve_path, verify_collections

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "config/g0_collections_v2_2.json").read_text(encoding="utf-8"))
SEAL = json.loads((ROOT / "config/sealed/presse_e7_test_v2_1.json").read_text(encoding="utf-8"))
HEADLINE = next(p for p in SEAL["pages"] if p["headline"])
UNCREDITED = next(p for p in SEAL["pages"] if not p["issue_keys"])
NEW_TITLE = "audit titre neuf"
NEW_ISSUE = "gallica:987654321"


def fresh():
    return copy.deepcopy(REGISTRY)


def headline_query():
    return dict(page_key=HEADLINE["scan_page_key"], issue_key=HEADLINE["issue_keys"][0], title_key=HEADLINE["title_key"])


def declare_sealed_page(registry, page_key):
    """The audit's fabricated claim: this page key is declared a member of the sealed manifest."""
    registry["collections"]["presse_e7_test"]["match_keys"]["page"] = {"name": "page", "values": [page_key]}


# --- the six counter-examples: each must be REFUSED ---------------------------------------------------

def test_1_an_exposed_collection_without_comparable_key_forces_a_cautious_refusal():
    registry = fresh()
    declare_sealed_page(registry, "audit:bnl_page")  # real source in BnL, which has no comparable key
    result = eligibility(registry, "test", page_key="audit:bnl_page", issue_key=NEW_ISSUE, title_key=NEW_TITLE)
    assert result["allowed"] is False


def test_2_a_partial_exposure_coverage_never_clears_a_page():
    registry = fresh()
    declare_sealed_page(registry, "audit:exposed_partial")
    registry["collections"]["pleias_french_pd_322"]["match_keys"]["issue"] = {
        "name": "issue", "values": ["gallica:12345"], "coverage": {"keyed": 1, "total": 2}}
    result = eligibility(registry, "test", page_key="audit:exposed_partial", issue_key=NEW_ISSUE, title_key=NEW_TITLE)
    assert result["allowed"] is False


def test_3_the_same_page_key_in_the_seal_and_in_its_reserved_sibling_is_refused():
    registry = fresh()
    registry["collections"]["presse_e7_test_refused"]["match_keys"]["page"] = {"name": "page", "values": [HEADLINE["scan_page_key"]]}
    assert eligibility(registry, "test", **headline_query())["allowed"] is False


def test_4_a_page_with_the_exact_title_of_a_complete_locked_collection_is_refused():
    registry = fresh()
    declare_sealed_page(registry, "audit:nzz")
    result = eligibility(registry, "test", page_key="audit:nzz", issue_key=NEW_ISSUE, title_key="neue zurcher zeitung")
    assert result["allowed"] is False


def test_5_a_real_page_without_a_source_document_is_refused_for_the_test_role():
    result = eligibility(REGISTRY, "test", page_key=UNCREDITED["scan_page_key"], title_key=UNCREDITED["title_key"])
    assert result["allowed"] is False


def test_6_verify_collections_refuses_a_registry_whose_keys_are_not_those_of_the_pinned_manifest(tmp_path):
    registry = fresh()
    spec = registry["collections"]["presse_e7_test"]["match_keys"]["page"]
    real_values = json.loads(resolve_path(spec["values_file"]["path"]).read_text(encoding="utf-8"))
    spec.pop("values_file")
    spec["values"] = real_values + ["audit:not_in_pinned_manifest"]
    spec["coverage"] = {"keyed": len(real_values) + 1, "total": len(real_values) + 1}
    fabricated = tmp_path / "adversarial_registry.json"
    fabricated.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CollectionRegistryError):
        verify_collections(fabricated)


# --- the four controls the audit found correct: they must stay correct ------------------------------

def complete_issue_keys(registry):
    """Fabricated, in memory only: every collection that may hold a Gallica document gets complete number keys."""
    for entry in registry["collections"].values():
        if entry["gallica_scope"]["value"] == "possible":
            spec = entry["match_keys"]["issue"]
            if spec == "aucune clé à ce niveau":
                spec = entry["match_keys"]["issue"] = {"name": "issue", "values": []}
            spec["coverage"] = {"keyed": 1, "total": 1}
    return registry


def test_control_a_real_headline_page_is_admitted_for_the_test_role_when_the_keys_are_complete():
    assert eligibility(complete_issue_keys(fresh()), "test", **headline_query())["allowed"] is True


def test_with_the_real_registry_no_page_is_admitted_today():
    """Assumed result (p1): published as is. G0-03 completes the number keys of every collection that is gallica_scope=possible."""
    admitted = [p["scan_page_key"] for p in SEAL["pages"] if p["issue_keys"] and
                eligibility(REGISTRY, "test", page_key=p["scan_page_key"], issue_key=p["issue_keys"][0], title_key=p["title_key"])["allowed"]]
    assert admitted == []


def test_control_a_query_without_any_identity_is_refused():
    with pytest.raises(CollectionRegistryError):
        eligibility(REGISTRY, "test")


def test_control_a_page_outside_the_seal_is_refused():
    result = eligibility(REGISTRY, "test", page_key="audit:outside", issue_key=NEW_ISSUE, title_key=NEW_TITLE)
    assert result["allowed"] is False


def test_control_a_known_issue_conflict_with_an_exposed_collection_is_refused():
    registry = fresh()
    registry["collections"]["pleias_french_pd_322"]["match_keys"]["issue"] = {"name": "issue", "values": [HEADLINE["issue_keys"][0]]}
    assert eligibility(registry, "test", **headline_query())["allowed"] is False
