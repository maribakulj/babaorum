"""G0-04 step B: eligibility(registry, "suivi", ...), one test per contract line (v4 + v4.1); fabricated inputs, pinned in the test only."""
import copy
import hashlib
import json
import re

import pytest

from babaorum import g0_suivi
from babaorum.g0_collections import _disk_registry, _key_values, _scope
from babaorum.g0_suivi import SuiviInputError, eligibility, load_inputs
from babaorum.g0_suivi_anchors import SUIVI_ANCHORS

SHA1 = "1" * 40  # synthetic: NOT a file of the sealed test (a real one is reserved whole)
F1, F2, DOC = "gallica:7636525/f1", "gallica:7636525/f2", "gallica:7636525"
SOURCES = [{"source": s, "key_spaces": ks, "status": "consumed", "files": [], "proof": "p"} for s, ks in
           (("press/train", ["gallica"]), ("R2_LM", ["gallica"]), ("e12_dev", ["gallica"]), ("e22", ["gallica"]), ("seg_as", ["as"]),
                    ("seg_kb", ["kb"]), ("presse_e7", ["wikimedia"]))]
NZZ = "nzz:17800719_0_0_a1_p1_1"
FAR = "gallica:9999999/f1"   # a page no test queries: home of the neutral usage lines


def C(seq, purpose="monitor", pages=("kb:99999999",)):
    """A complete consultation line of the frozen format."""
    return {"seq": seq, "purpose": purpose, "date": "2026-10-23", "agent": "p1", "pages": list(pages)}

NZZ_REQ = {"nzz": ["catmus", "ngram8"]}
NZZ_DONE = [{"page": NZZ, "check": c, "status": "complete", "findings": []} for c in ("catmus", "ngram8")]


def registry(complete=True):
    """The anchored registry; with complete=True, every `possible` collection is marked fully keyed IN MEMORY (B1 done)."""
    reg = copy.deepcopy(_disk_registry())
    if complete:
        for e in reg["collections"].values():
            if _scope(e) == "possible":
                if not isinstance(e["match_keys"]["issue"], dict):  # NO_KEY today: an empty, complete key set once B1 is done
                    e["match_keys"]["issue"] = {"name": "k", "values": []}
                e["match_keys"]["issue"]["coverage"] = {"keyed": 1, "total": 1}
    return reg


def write_inputs(tmp_path, usages=(), aliases=(), checks=(), limits=None, sources=None, required_checks=None, nzz_search=None,
                 journals=(), without_proof=(), consultations=(), seq_max=0, drop=(), unpin=(), mutate=None, declaration="aucune consultation",
                 raw_edit=None, neutral=True):
    docs = {
        "usage": {"schema": "babaorum.g0_page_usage.v1", "version": 1, "created": "2026-10-22", "consultations_seq_max": seq_max,
                  "usages": [dict(zip(("scope", "key", "usage", "source", "status"), u), proof="p") for u in usages],
                  "aliases": list(aliases), "checks": list(checks), "limits": limits or {}},
        "sources": {"schema": "babaorum.g0_usage_sources.v1", "sources": copy.deepcopy(SOURCES if sources is None else sources),
                    "required_checks": required_checks or {}, "nzz_search": nzz_search or {"done": True, "hits": 0, "files": ["x"]}},
        "journals": {"schema": "babaorum.g0_journal_review.v1", "journals": list(journals)},
        "select_evidence": {"schema": "babaorum.g0_select_evidence.v1", "evidence": [], "documented_without_proof": list(without_proof)},
        "consultations": {"schema": "babaorum.g0_suivi_consultations.v1", "consultations": list(consultations), "auteur": "p1",
                          "date": "2026-10-22", "periode_couverte": "2026-10-18/2026-10-22"},
    }
    if declaration is not None:
        docs["consultations"]["declaration"] = declaration
    used = {u["source"] for u in docs["usage"]["usages"]}   # a joining row needs a usage line (pA, 24/10): a neutral one on a far page
    for src in (docs["sources"]["sources"] if neutral else []):
        if src.get("status") in ("consumed", "allocated_only") and isinstance(src.get("source"), str) and src["source"] not in used:
            docs["usage"]["usages"].append({"scope": "page", "key": FAR, "usage": "eval", "source": src["source"], "status": "consumed", "proof": "p"})
    if mutate:
        mutate(docs)
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths, anchors = {}, {}
    for name, doc in docs.items():
        p = tmp_path / f"{name}.json"
        paths[name] = p
        if name in drop:
            continue
        text = json.dumps(doc)
        p.write_text(raw_edit(name, text) if raw_edit else text, encoding="utf-8")
        anchors[name] = None if name in unpin else hashlib.sha256(p.read_bytes()).hexdigest()
    return load_inputs(paths, anchors)


def call(inputs, **kw):
    """The anchors used to write the test files are put in the module for the call only (the code trusts only module anchors)."""
    saved = dict(g0_suivi.SUIVI_ANCHORS)
    g0_suivi.SUIVI_ANCHORS.clear()
    g0_suivi.SUIVI_ANCHORS.update({k: v for k, v in inputs.anchors.items()})
    try:
        out = eligibility(kw.pop("reg", None) or registry(), kw.pop("role", "suivi"), inputs=inputs, **kw)
        # AV-G0-SUIVI-LIMITES: every page carries the pretraining limits; checked on every call, then set aside for the older exact asserts
        assert set(g0_suivi.PRETRAINING_LIMITS) <= set(out["labels"]), out["labels"]
        out["labels"] = [x for x in out["labels"] if x not in g0_suivi.PRETRAINING_LIMITS]
        return out
    finally:
        g0_suivi.SUIVI_ANCHORS.clear()
        g0_suivi.SUIVI_ANCHORS.update(saved)


def codes(page, inputs, reg=None, **kw):
    return call(inputs, reg=reg, page_key=page, **kw)


# frozen interface and fail-closed loading (pA, 22/10)
def test_the_input_schema_file_is_pinned_and_the_b1_inputs_are_anchored():
    """B1 in place (25/10, pA agreement, GO pQ): every anchor is the SHA-256 of the file at its default path."""
    assert hashlib.sha256(g0_suivi.SCHEMA_PATH.read_bytes()).hexdigest() == g0_suivi.SCHEMA_SHA256
    assert set(SUIVI_ANCHORS) == set(g0_suivi.PINNED_FILES)
    for name, rel in g0_suivi.SUIVI_PATHS.items():
        assert SUIVI_ANCHORS[name] == hashlib.sha256((g0_suivi.REPO_ROOT / rel).read_bytes()).hexdigest(), name


def test_a_new_gallica_page_on_the_committed_b1_state():
    """The committed B1 files, read by the module through its own anchors (no test anchor). A page of an issue nobody used:
    (1) as is, only the known reserves refuse it: the incomplete number keys and the per-page ngram8 not yet produced (pA);
    (2) with complete number keys, only the two ngram8 remain;
    (3) with an ngram8 against both LM rows added in memory on top of these exact files, it is admitted."""
    new = "gallica:7799999/f1"
    r = eligibility(registry(complete=False), "suivi", page_key=new)
    lm = {"SUIVI_CHECK_INCOMPLETE:ngram8:pleias_french_pd_322", "SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm_corpus"}
    assert not r["allowed"] and set(r["refusal_codes"]) <= lm | {c for c in r["refusal_codes"] if c.startswith("INCOMPLETE_NUMBER_KEY:")}
    assert set(r["refusal_codes"]) & lm == lm
    r = eligibility(registry(complete=True), "suivi", page_key=new)
    assert set(r["refusal_codes"]) == lm, r["refusal_codes"]
    real = g0_suivi.load_inputs()
    usage = json.loads(real.raw["usage"])
    rows = {s_["source"]: s_ for s_ in json.loads(real.raw["sources"])["sources"] if "lm_texts" in s_}
    usage["checks"].append({"page": new, "check": "ngram8", "status": "complete", "against": sorted(rows), "findings": [],
                            "against_manifests": {n: g0_suivi.lm_fingerprint(rows[n]["lm_texts"]["files"]) for n in rows}})
    b = json.dumps(usage, ensure_ascii=False).encode("utf-8")
    inputs = g0_suivi.SuiviInputs(dict(real.raw, usage=b), real.names, dict(real.anchors, usage=hashlib.sha256(b).hexdigest()))
    out = codes(new, inputs)
    assert out["allowed"], out["refusal_codes"]


def test_an_absent_usage_registry_is_refused_without_fallback(tmp_path):
    inputs = write_inputs(tmp_path, drop=("usage",))
    r = codes(F1, inputs)
    assert not r["allowed"] and r["refusal_codes"] == ["SEAL_NOT_PINNED:usage.json"]


def test_an_unpinned_usage_registry_is_not_read(tmp_path):
    inputs = write_inputs(tmp_path, usages=[("page", F1, "train", "press/train", "consumed")], unpin=("usage",))
    assert inputs.raw["usage"] is None and codes(F1, inputs)["refusal_codes"] == ["SEAL_NOT_PINNED:usage.json"]


@pytest.mark.parametrize("mutate", [
    lambda d: d["usage"].pop("usages"),
    lambda d: d["usage"].pop("consultations_seq_max"),
    lambda d: d["usage"]["usages"].append({"scope": "page", "key": F1, "usage": "train", "source": "s", "status": "consumed"}),  # no proof
    lambda d: d["usage"]["usages"].append({"scope": "page", "key": "read_7636525-001", "usage": "train", "source": "s",
                                           "status": "consumed", "proof": "p"}),  # non-canonical key
    lambda d: d["sources"].pop("nzz_search"),
    lambda d: d["sources"]["sources"].append(dict(SOURCES[0])),  # duplicate source
    lambda d: d["journals"]["journals"].append({"journal": "j", "status": "maybe"}),
    lambda d: d["consultations"].pop("periode_couverte"),
    lambda d: d["consultations"]["consultations"].append({"seq": 1, "purpose": "peek"}),
    lambda d: d["usage"]["checks"].extend([{"page": F1, "check": "c", "status": "complete", "findings": [{"kind": "train_overlap", "source": "b"}]},
                                           {"page": F1, "check": "c", "status": "complete", "findings": []}]),   # duplicate (page, check)
    lambda d: d["usage"]["usages"].extend([{"scope": "page", "key": F1, "usage": "train", "source": "s", "status": s, "proof": "p"}
                                           for s in ("consumed", "allocated_only")]),                          # duplicate usage
    lambda d: d["journals"]["journals"].append({"journal": "j", "status": "pending", "concerns": []}),        # empty concerns
    lambda d: d["sources"]["required_checks"].update(nzz="ngram8"),                                          # not a list
    lambda d: d["sources"]["sources"][0]["files"].append({"path": "x", "sha256": "0"}),                       # bad file entry
    lambda d: d["select_evidence"]["evidence"].append({"target": "e22", "file": "LOT.md"}),                    # no line
])
def test_a_partial_or_malformed_input_raises_and_admits_nothing(tmp_path, mutate):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, mutate=mutate)


def test_a_clean_page_is_admitted_once_gallica_is_complete(tmp_path):
    r = codes(F1, write_inputs(tmp_path))
    assert r["allowed"] and r["refusal_codes"] == [] and r["labels"] == [] and r["usage_registry_version"] == 1


# ranks 3 and 4
def test_a_consultation_after_the_anchor_makes_the_registry_stale(tmp_path):
    inputs = write_inputs(tmp_path, consultations=[C(1, "monitor"), C(2, "choose")], seq_max=1)
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_REGISTRY_STALE"]


@pytest.mark.parametrize("key", ["gallica:7636525/f01", "read_7636525-001", "gallica:7636525", "impresso:x", None])
def test_no_canonical_page_key_is_refused(tmp_path, key):
    assert codes(key, write_inputs(tmp_path))["refusal_codes"] == ["SUIVI_NO_PAGE_KEY"]


# rank 5: spaces, and the number keys of G0-03 B1
def test_gallica_with_an_incomplete_number_key_is_refused_then_admitted_when_complete(tmp_path):
    inputs = write_inputs(tmp_path)
    reg = registry(complete=False)
    expected = [f"INCOMPLETE_NUMBER_KEY:{n}" for n, e in reg["collections"].items()
                if _scope(e) == "possible" and not g0_suivi._issue_keys_complete(e)]
    assert expected and codes(F1, inputs, reg)["refusal_codes"] == expected
    assert codes(F1, inputs)["allowed"]


def test_as_page_four_cases(tmp_path):
    p3 = "as:18840315_1/p3"
    assert codes(p3, write_inputs(tmp_path), registry(complete=False))["refusal_codes"][0] == "SUIVI_SPACE_UNCOVERED:as"
    r = codes(p3, write_inputs(tmp_path / "e", usages=[("document", "as:18840315_1", "eval", "seg_as", "consumed")]))
    assert r["allowed"] and r["labels"] == ["EVALUATED_BEFORE:seg_as"]
    r = codes(p3, write_inputs(tmp_path / "t", usages=[("page", p3, "train", "seg_as", "consumed")]))
    assert r["refusal_codes"] == ["SUIVI_TRAINED:seg_as"]
    assert codes(p3, write_inputs(tmp_path / "n"))["allowed"]


def test_nzz_coverage(tmp_path):
    ok = dict(required_checks=NZZ_REQ, checks=NZZ_DONE)
    assert codes(NZZ, write_inputs(tmp_path, **ok))["allowed"]
    for k, (kw, extra) in enumerate([({"nzz_search": {"done": False, "hits": 0, "files": ["x"]}}, []),
                                     ({"nzz_search": {"done": True, "hits": 2, "files": ["x"]}}, []),
                                     ({"nzz_search": {"done": True, "hits": 0, "files": []}}, []),
                                     ({"required_checks": {"nzz": ["catmus"]}}, [])]):
        args = {**ok, **kw}
        r = codes(NZZ, write_inputs(tmp_path / str(k), **args))
        assert r["refusal_codes"][0] == "SUIVI_SPACE_UNCOVERED:nzz", kw
    unknown = SOURCES + [{"source": "mystery", "key_spaces": ["unknown"], "status": "consumed", "files": [], "proof": "p"}]
    r = codes(NZZ, write_inputs(tmp_path / "b", sources=unknown, **ok))
    assert r["refusal_codes"] == ["SUIVI_SPACE_UNCOVERED:nzz", "SUIVI_SOURCE_UNRESOLVED:mystery"]


def test_spaces_outside_the_contract_and_an_unpinned_sources_table_are_uncovered(tmp_path):
    assert codes("bnl:000000", write_inputs(tmp_path))["refusal_codes"] == ["SUIVI_SPACE_UNCOVERED:bnl"]
    assert codes("newseye_photo:P1020460", write_inputs(tmp_path / "p"))["refusal_codes"] == ["SUIVI_SPACE_UNCOVERED:newseye_photo"]
    r = codes(F1, write_inputs(tmp_path / "u", unpin=("sources",)))
    assert r["refusal_codes"] == ["SEAL_NOT_PINNED:sources.json", "SUIVI_SPACE_UNCOVERED:gallica"]


# ranks 6 and 7
def test_an_unresolved_or_unjoinable_source_of_the_space_refuses(tmp_path):
    src = SOURCES + [{"source": "pseudo", "key_spaces": ["gallica"], "status": "non_resolue", "files": [], "proof": "p"},
                     {"source": "kb_only", "key_spaces": ["kb"], "status": "non_joignable", "files": [], "proof": "p"}]
    assert codes(F1, write_inputs(tmp_path, sources=src))["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:pseudo"]


def test_a_required_check_missing_or_incomplete_refuses(tmp_path):
    page = NZZ
    r = codes(page, write_inputs(tmp_path, required_checks=NZZ_REQ,
                                 checks=[{"page": page, "check": "catmus", "status": "incomplete", "findings": []}]))
    assert r["refusal_codes"] == ["SUIVI_CHECK_INCOMPLETE:catmus", "SUIVI_CHECK_INCOMPLETE:ngram8"]


# rank 8: the sealed test, reserved whole
def test_the_sealed_test_is_reserved_by_page_issue_and_alias(tmp_path):
    sealed = registry()["collections"]["presse_e7_test"]["match_keys"]
    page = sorted(_key_values(sealed["page"]))[0]
    assert codes(page, write_inputs(tmp_path))["refusal_codes"] == ["SUIVI_RESERVED:presse_e7_test"]
    issue = sorted(i for i in _key_values(sealed["issue"]) if i.startswith("gallica:"))[0]
    assert codes(f"{issue}/f9", write_inputs(tmp_path / "i"))["refusal_codes"] == ["SUIVI_RESERVED:presse_e7_test"]
    ns = "wikimedia" if page.startswith("wikimedia") else "gallica"
    alias = {"left": {"namespace": ns, "value": page}, "right": {"namespace": "gallica", "value": "gallica:1234567/f1"},
             "proven_page": False, "source": "concordance"}
    r = codes("gallica:1234567/f4", write_inputs(tmp_path / "a", aliases=[alias]))
    assert r["refusal_codes"] == ["SUIVI_RESERVED:presse_e7_test"]


def test_a_sealed_list_that_is_not_the_pinned_file_is_refused(tmp_path):
    reg = registry()
    reg["collections"]["presse_e7_test"]["match_keys"]["page"] = {"name": "k", "values": ["gallica:1111111/f1"]}
    assert codes(F1, write_inputs(tmp_path), reg)["refusal_codes"] == ["SEAL_NOT_PINNED:presse_e7_test"]


# ranks 9-11, and labels
def test_document_training_refuses_a_page_with_no_individual_entry(tmp_path):
    r = codes(F2, write_inputs(tmp_path, usages=[("document", DOC, "train", "R2_LM", "consumed")]))
    assert r["refusal_codes"] == ["SUIVI_TRAINED:R2_LM"]


def test_select_on_one_folio_refuses_it_and_labels_its_sibling(tmp_path):
    inputs = write_inputs(tmp_path, usages=[("page", F1, "select", "e12_dev", "consumed"), ("page", F2, "eval", "e22", "consumed")])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SELECTED:e12_dev"]
    r = codes(F2, inputs)
    assert r["allowed"] and r["labels"] == ["EVALUATED_BEFORE:e22", "ISSUE_SHARED:select:e12_dev"]


def test_allocated_only_refuses_with_its_label_and_codes_follow_the_table_order(tmp_path):
    inputs = write_inputs(tmp_path, usages=[("page", F1, "train", "R2_LM", "allocated_only"), ("page", F1, "train", "press/train", "consumed"),
                                            ("page", F1, "dev", "press/train", "consumed")])
    r = codes(F1, inputs)
    assert r["refusal_codes"] == ["SUIVI_TRAINED:press/train", "SUIVI_TRAINED:R2_LM", "SUIVI_DEV:press/train"]
    assert r["labels"] == ["ALLOCATION_ONLY:R2_LM"]


def test_a_promoted_page_is_selected_by_the_next_registry_version(tmp_path):
    rows = SOURCES + [{"source": "suivi_d1", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "choose seq 1"}]
    inputs = write_inputs(tmp_path, usages=[("page", F1, "select", "suivi_d1", "consumed")], sources=rows,
                          consultations=[C(1, "choose")], seq_max=1)
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SELECTED:suivi_d1"]


# rank 12
def test_pending_journals_and_choices_without_proof(tmp_path):
    assert codes(F1, write_inputs(tmp_path, journals=[{"journal": "DECISIONS.md", "status": "pending"}]))["refusal_codes"] == [
        "SUIVI_SELECTION_UNREVIEWED:DECISIONS.md"]
    assert codes(F1, write_inputs(tmp_path / "k", journals=[{"journal": "kb.md", "status": "pending", "concerns": ["kb"]}]))["allowed"]
    inputs = write_inputs(tmp_path / "w", usages=[("page", F1, "eval", "e22", "consumed")],
                          without_proof=[{"bench": "e22", "journal": "LOT-E22.md"}])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SELECTION_UNREVIEWED:LOT-E22.md"]


# rank 13, and check labels
def test_text_overlap_refuses_and_eval_or_third_party_overlap_only_labels(tmp_path):
    page = NZZ
    findings = [{"kind": "train_overlap", "source": "bnl"}, {"kind": "eval_overlap", "source": "icdar2017"},
                {"kind": "third_party_unresolved", "source": "hipe"}, {"kind": "third_party_eval", "source": "catmus_test"}]
    checks = [NZZ_DONE[0], {"page": page, "check": "ngram8", "status": "complete", "findings": findings}]
    r = codes(page, write_inputs(tmp_path, checks=checks, required_checks=NZZ_REQ, limits={"nzz": ["LIMIT:mccatmus_non_french_uncovered"]}))
    assert r["refusal_codes"] == ["SUIVI_TEXT_OVERLAP:bnl"]
    assert r["labels"] == ["LIMIT:mccatmus_non_french_uncovered", "TEXT_OVERLAP_EVAL:icdar2017", "THIRD_PARTY_EVAL:catmus_test",
                           "THIRD_PARTY_OVERLAP:hipe:unresolved"]


# the final order of v4, all codes rendered
def test_exact_order_of_several_refusals(tmp_path):
    src = SOURCES + [{"source": "pseudo", "key_spaces": ["gallica"], "status": "non_resolue", "files": [], "proof": "p"}]
    inputs = write_inputs(tmp_path, sources=src, seq_max=0, consultations=[C(1, "choose")],
                          usages=[("page", F1, "select", "e12_dev", "consumed"), ("document", DOC, "train", "R2_LM", "consumed"),
                                  ("page", F1, "dev", "press/train", "consumed")],
                          journals=[{"journal": "DECISIONS.md", "status": "pending", "concerns": ["e12_dev"]}],
                          required_checks={"gallica": ["ngram8"]},
                          checks=[{"page": F1, "check": "ngram8", "status": "complete", "findings": [{"kind": "train_overlap", "source": "bnl"}]}])
    reg = registry(complete=False)
    incomplete = [f"INCOMPLETE_NUMBER_KEY:{n}" for n, e in reg["collections"].items()
                  if _scope(e) == "possible" and not g0_suivi._issue_keys_complete(e)]
    assert codes(F1, inputs, reg)["refusal_codes"] == (
        ["SUIVI_REGISTRY_STALE"] + incomplete + ["SUIVI_SOURCE_UNRESOLVED:pseudo", "SUIVI_TRAINED:R2_LM", "SUIVI_DEV:press/train",
                                                 "SUIVI_SELECTED:e12_dev", "SUIVI_SELECTION_UNREVIEWED:DECISIONS.md", "SUIVI_TEXT_OVERLAP:bnl"])


def test_caller_errors_raise(tmp_path):
    inputs = write_inputs(tmp_path)
    with pytest.raises(ValueError):
        call(inputs, role="test", page_key=F1)
    with pytest.raises(ValueError):
        call(inputs, page_key=F1, issue_key="gallica:999")
    assert call(inputs, page_key=F1, issue_key=DOC)["allowed"]


# pA (22/10): an empty consultation journal is a declaration, never an absence
def test_an_empty_consultation_journal_without_declaration_is_invalid(tmp_path):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, declaration=None)
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path / "blank", declaration="  ")
    assert codes(F1, write_inputs(tmp_path / "ok"))["allowed"]  # declared "aucune consultation"
    assert codes(F1, write_inputs(tmp_path / "n", declaration=None, consultations=[C(1, "monitor")], seq_max=1))["allowed"]


def test_an_absent_or_unpinned_consultation_journal_is_refused(tmp_path):
    assert codes(F1, write_inputs(tmp_path, drop=("consultations",)))["refusal_codes"] == ["SEAL_NOT_PINNED:consultations.json"]
    assert codes(F1, write_inputs(tmp_path / "u", unpin=("consultations",)))["refusal_codes"] == ["SEAL_NOT_PINNED:consultations.json"]


# relecture Sonnet (22/10), NO-GO : fail-open, couverture vide, contrôles NZZ, doublons, promotion, mutations survivantes
def test_hand_made_or_edited_snapshots_are_never_read(tmp_path):
    inputs = write_inputs(tmp_path, usages=[("page", "kb:00531074", "train", "seg_kb", "consumed")])
    assert codes("kb:00531074", inputs)["refusal_codes"] == ["SUIVI_TRAINED:seg_kb"]
    with pytest.raises(TypeError):
        inputs.raw["usage"] = b"{}"                                             # the snapshot is read-only
    edited = json.loads(inputs.raw["usage"])
    edited["usages"] = []
    forged = g0_suivi.SuiviInputs({**inputs.raw, "usage": json.dumps(edited).encode()}, inputs.names, inputs.anchors)
    assert codes("kb:00531074", forged)["refusal_codes"] == ["SEAL_NOT_PINNED:usage.json"]   # pT B1: edited bytes no longer match the anchor
    self_pinned = g0_suivi.SuiviInputs(forged.raw, forged.names, {**inputs.anchors, "usage": hashlib.sha256(forged.raw["usage"]).hexdigest()})
    assert eligibility(registry(), "suivi", page_key="kb:00531074", inputs=self_pinned)["refusal_codes"][:1] == ["SEAL_NOT_PINNED:usage.json"]


def test_a_space_is_covered_only_by_a_joining_row_of_the_table(tmp_path):
    for page, ns in ((F1, "gallica"), ("kb:00531074", "kb"), (f"wikimedia-file-sha1:{SHA1}/f2", "wikimedia")):
        assert codes(page, write_inputs(tmp_path / ns))["allowed"], ns
        rows = [dict(s, status="non_joignable") if ns in s["key_spaces"] else s for s in SOURCES]
        r = codes(page, write_inputs(tmp_path / f"{ns}-nj", sources=rows))
        assert r["refusal_codes"][0] == f"SUIVI_SPACE_UNCOVERED:{ns}" and not r["allowed"], ns
        r = codes(page, write_inputs(tmp_path / f"{ns}-none", sources=[s for s in SOURCES if ns not in s["key_spaces"]]))
        assert r["refusal_codes"] == [f"SUIVI_SPACE_UNCOVERED:{ns}"], ns


def test_a_usage_whose_source_is_not_in_the_table_is_unresolved(tmp_path):
    r = codes(F1, write_inputs(tmp_path, usages=[("page", F1, "eval", "ghost", "consumed")]))
    assert r["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:ghost"]


def test_select_evidence_promotes_its_bench_or_page(tmp_path):
    ev = lambda target: (lambda d: d["select_evidence"]["evidence"].append({"target": target, "file": "LOT-E22.md", "line": 12}))  # noqa: E731
    inputs = write_inputs(tmp_path, usages=[("page", F1, "eval", "e22", "consumed")], mutate=ev("e22"))
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SELECTED:e22"]
    assert codes(F1, write_inputs(tmp_path / "p", mutate=ev(F1)))["refusal_codes"] == [f"SUIVI_SELECTED:{F1}"]
    assert codes(F2, write_inputs(tmp_path / "o", mutate=ev(F1)))["allowed"]


def test_a_proven_alias_accumulates_page_usages_and_an_unproven_one_does_not(tmp_path):
    wik = f"wikimedia-file-sha1:{SHA1}/f1"
    alias = lambda proven: {"left": {"namespace": "gallica", "value": F1}, "right": {"namespace": "wikimedia", "value": wik},  # noqa: E731
                            "proven_page": proven, "source": "concordance"}
    use = [("page", F1, "train", "press/train", "consumed")]
    assert codes(wik, write_inputs(tmp_path, usages=use, aliases=[alias(True)]))["refusal_codes"] == ["SUIVI_TRAINED:press/train"]
    assert codes(wik, write_inputs(tmp_path / "u", usages=use, aliases=[alias(False)]))["allowed"]
    doc = [("document", DOC, "train", "R2_LM", "consumed")]
    other_view = f"wikimedia-file-sha1:{SHA1}/f7"
    assert codes(other_view, write_inputs(tmp_path / "d", usages=doc, aliases=[alias(False)]))["refusal_codes"] == ["SUIVI_TRAINED:R2_LM"]


def test_file_scope_reaches_every_view_and_kb_has_no_document(tmp_path):
    f = f"wikimedia-file-sha1:{SHA1}"
    inputs = write_inputs(tmp_path, usages=[("fichier", f, "dev", "presse_e7", "consumed")])
    assert codes(f"{f}/f13", inputs)["refusal_codes"] == ["SUIVI_DEV:presse_e7"]
    inputs = write_inputs(tmp_path / "kb", usages=[("page", "kb:00531074", "train", "seg_kb", "consumed")])
    assert codes("kb:00531074", inputs)["refusal_codes"] == ["SUIVI_TRAINED:seg_kb"] and codes("kb:00531075", inputs)["allowed"]


def test_exact_order_of_the_first_ranks(tmp_path):
    sealed = registry()["collections"]["presse_e7_test"]["match_keys"]
    page = sorted(_key_values(sealed["page"]))[0]
    ns = "wikimedia" if page.startswith("wikimedia") else "gallica"
    inputs = write_inputs(tmp_path, unpin=("journals",), sources=[s for s in SOURCES if ns not in s["key_spaces"]],
                          required_checks={ns: ["ngram8"]})
    assert codes(page, inputs)["refusal_codes"] == ["SEAL_NOT_PINNED:journals.json", f"SUIVI_SPACE_UNCOVERED:{ns}",
                                                   "SUIVI_CHECK_INCOMPLETE:ngram8", "SUIVI_RESERVED:presse_e7_test"]


def test_the_schema_is_checked_at_run_time(tmp_path, monkeypatch):
    inputs = write_inputs(tmp_path)
    monkeypatch.setattr(g0_suivi, "SCHEMA_SHA256", "0" * 64)
    with pytest.raises(SuiviInputError):
        codes(F1, inputs)


def test_a_selection_evidence_reaches_a_page_only_through_a_proven_alias(tmp_path):
    wik = f"wikimedia-file-sha1:{SHA1}/f1"
    alias = lambda proven: [{"left": {"namespace": "gallica", "value": F1}, "right": {"namespace": "wikimedia", "value": wik},  # noqa: E731
                             "proven_page": proven, "source": "concordance"}]
    ev = lambda d: d["select_evidence"]["evidence"].append({"target": F1, "file": "LOT-E12.md", "line": 3})  # noqa: E731
    assert codes(wik, write_inputs(tmp_path, aliases=alias(True), mutate=ev))["refusal_codes"] == [f"SUIVI_SELECTED:{F1}"]
    assert codes(wik, write_inputs(tmp_path / "u", aliases=alias(False), mutate=ev))["allowed"]



# pT (23/10), NO-GO on 253ee687: B2 strict validation, B3 proven alias class, C1 choose consultations, C2 forced NZZ limit
@pytest.mark.parametrize("raw_edit", [
    lambda n, s: re.sub(r'"usages": \[.*?\]', '"usages": {}', s, count=1, flags=re.S) if n == "usage" else s,  # object instead of array
    lambda n, s: s.replace('"checks": []', '"checks": {}') if n == "usage" else s,
    lambda n, s: s.replace('"aliases": []', '"aliases": {}') if n == "usage" else s,
    lambda n, s: s.replace('"usages": [', '"usages": [{"scope": "page", "key": "kb:00531074", "usage": "train", "source": "seg_kb", '
                                          '"status": "consumed", "proof": "p"}], "usages": [', 1) if n == "usage" else s,  # duplicate JSON key
    lambda n, s: s[:-1] if n == "sources" else s,                                                        # invalid JSON
])
def test_malformed_pinned_files_raise_the_contract_error(tmp_path, raw_edit):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, raw_edit=raw_edit)


@pytest.mark.parametrize("line", [{"seq": 1, "purpose": "choose"}, {"seq": 1, "purpose": "choose", "date": "d", "agent": "a"},
                                  {"seq": 1, "purpose": "choose", "date": "d", "agent": "a", "pages": []}])
def test_a_consultation_without_all_its_fields_is_invalid(tmp_path, line):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, consultations=[line], seq_max=1)


def test_a_choose_consultation_selects_its_pages_even_without_a_promotion(tmp_path):
    line = {"seq": 1, "purpose": "choose", "date": "2026-10-23", "agent": "p9", "pages": ["kb:00531074"]}
    inputs = write_inputs(tmp_path, consultations=[line], seq_max=1)
    assert codes("kb:00531074", inputs)["refusal_codes"] == ["SUIVI_SELECTED:suivi_choose_seq1"]
    assert codes("kb:00531075", inputs)["allowed"]
    monitor = dict(line, purpose="monitor")
    assert codes("kb:00531074", write_inputs(tmp_path / "m", consultations=[monitor], seq_max=1))["allowed"]


def test_nzz_always_carries_the_mccatmus_limit(tmp_path):
    r = codes(NZZ, write_inputs(tmp_path, required_checks=NZZ_REQ, checks=NZZ_DONE))
    assert r["allowed"] and r["labels"] == ["LIMIT:mccatmus_non_french_uncovered"]


def _proven(proven=True):
    return [{"left": {"namespace": "gallica", "value": F1}, "right": {"namespace": "wikimedia", "value": f"wikimedia-file-sha1:{SHA1}/f1"},
             "proven_page": proven, "source": "concordance"}]


def test_every_page_constraint_follows_the_proven_alias_class(tmp_path):
    w = f"wikimedia-file-sha1:{SHA1}/f1"
    overlap = [{"page": F1, "check": "ngram8", "status": "complete", "findings": [{"kind": "train_overlap", "source": "r2"}]}]
    assert codes(w, write_inputs(tmp_path, aliases=_proven(), checks=overlap))["refusal_codes"] == ["SUIVI_TEXT_OVERLAP:r2"]
    reg = registry(complete=False)
    incomplete = [f"INCOMPLETE_NUMBER_KEY:{n}" for n, e in reg["collections"].items()
                  if _scope(e) == "possible" and not g0_suivi._issue_keys_complete(e)]
    assert codes(w, write_inputs(tmp_path / "i", aliases=_proven()), reg)["refusal_codes"] == incomplete
    pending = [{"journal": "DECISIONS.md", "status": "pending", "concerns": ["gallica"]}]
    assert codes(w, write_inputs(tmp_path / "j", aliases=_proven(), journals=pending))["refusal_codes"] == [
        "SUIVI_SELECTION_UNREVIEWED:DECISIONS.md"]
    r = codes(w, write_inputs(tmp_path / "c", aliases=_proven(), required_checks={"gallica": ["ngram8"]}))
    assert r["refusal_codes"] == ["SUIVI_CHECK_INCOMPLETE:ngram8"]
    rows = [s for s in SOURCES if "gallica" not in s["key_spaces"]]
    assert codes(w, write_inputs(tmp_path / "s", aliases=_proven(), sources=rows))["refusal_codes"] == ["SUIVI_SPACE_UNCOVERED:gallica"]


def test_a_document_only_link_carries_no_page_constraint(tmp_path):
    w = f"wikimedia-file-sha1:{SHA1}/f1"
    overlap = [{"page": F1, "check": "ngram8", "status": "complete", "findings": [{"kind": "train_overlap", "source": "r2"}]}]
    assert codes(w, write_inputs(tmp_path, aliases=_proven(False), checks=overlap))["allowed"]
    assert codes(w, write_inputs(tmp_path / "i", aliases=_proven(False)), registry(complete=False))["allowed"]



# relecture Sonnet (23/10) : tests des mutations survivantes
def test_a_choose_consultation_reaches_the_proven_alias(tmp_path):
    w = f"wikimedia-file-sha1:{SHA1}/f1"
    line = C(1, "choose", pages=[F1])
    assert codes(w, write_inputs(tmp_path, aliases=_proven(), consultations=[line], seq_max=1))["refusal_codes"] == [
        "SUIVI_SELECTED:suivi_choose_seq1"]
    assert codes(w, write_inputs(tmp_path / "u", aliases=_proven(False), consultations=[line], seq_max=1))["allowed"]


def test_an_unjoinable_source_of_another_space_of_the_class_refuses(tmp_path):
    w = f"wikimedia-file-sha1:{SHA1}/f1"
    rows = SOURCES + [{"source": "pseudo", "key_spaces": ["gallica"], "status": "non_resolue", "files": [], "proof": "p"}]
    assert codes(w, write_inputs(tmp_path, aliases=_proven(), sources=rows))["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:pseudo"]
    assert codes(w, write_inputs(tmp_path / "u", sources=rows))["allowed"]


def test_codes_list_the_queried_key_first(tmp_path):
    w = f"wikimedia-file-sha1:{SHA1}/f1"
    rows = [s for s in SOURCES if not {"gallica", "wikimedia"} & set(s["key_spaces"])]
    assert codes(w, write_inputs(tmp_path, aliases=_proven(), sources=rows))["refusal_codes"] == [
        "SUIVI_SPACE_UNCOVERED:wikimedia", "SUIVI_SPACE_UNCOVERED:gallica"]
    assert codes(F1, write_inputs(tmp_path / "f", aliases=_proven(), sources=rows))["refusal_codes"] == [
        "SUIVI_SPACE_UNCOVERED:gallica", "SUIVI_SPACE_UNCOVERED:wikimedia"]
    req = {"gallica": ["gcheck"], "wikimedia": ["wcheck"]}
    assert codes(w, write_inputs(tmp_path / "c", aliases=_proven(), required_checks=req))["refusal_codes"] == [
        "SUIVI_CHECK_INCOMPLETE:wcheck", "SUIVI_CHECK_INCOMPLETE:gcheck"]
    over = lambda p, src: {"page": p, "check": "ngram8", "status": "complete", "findings": [{"kind": "train_overlap", "source": src}]}  # noqa: E731
    r = codes(w, write_inputs(tmp_path / "o", aliases=_proven(), checks=[over(F1, "aa"), over(w, "zz")]))
    assert r["refusal_codes"] == ["SUIVI_TEXT_OVERLAP:zz", "SUIVI_TEXT_OVERLAP:aa"]


def test_the_nzz_limit_is_carried_through_a_proven_alias(tmp_path):
    al = [{"left": {"namespace": "nzz", "value": NZZ}, "right": {"namespace": "kb", "value": "kb:00531074"},
           "proven_page": True, "source": "concordance"}]
    r = codes("kb:00531074", write_inputs(tmp_path, aliases=al, required_checks=NZZ_REQ, checks=NZZ_DONE))
    assert r["allowed"] and r["labels"] == ["LIMIT:mccatmus_non_french_uncovered"]


# pT (23/10), NO-GO on 387f58d1: join keys must be exact (canonical pages, known spaces, no surrounding space)
@pytest.mark.parametrize("kw", [
    {"consultations": [C(1, "choose", pages=["kb:00531074 "])], "seq_max": 1},
    {"checks": [{"page": "kb:00531074 ", "check": "ngram8", "status": "complete", "findings": [{"kind": "train_overlap", "source": "r2"}]}]},
    {"sources": SOURCES + [{"source": "pseudo", "key_spaces": ["kb "], "status": "non_resolue", "files": [], "proof": "p"}]},
    {"required_checks": {"kb ": ["ngram8"]}},
    {"required_checks": {"kbx": ["ngram8"]}},
    {"sources": SOURCES + [{"source": "x", "key_spaces": ["kbx"], "status": "consumed", "files": [], "proof": "p"}]},
    {"sources": SOURCES + [{"source": [], "key_spaces": ["kb"], "status": "consumed", "files": [], "proof": "p"}]},   # not hashable
    {"mutate": lambda d: d["select_evidence"]["evidence"].append({"target": "kb:00531074 ", "file": "f", "line": 1})},
    {"mutate": lambda d: d["select_evidence"]["evidence"].append({"target": "kb:531074", "file": "f", "line": 1})},
    {"aliases": [{"left": {"namespace": "kb", "value": "kb:00531074 "}, "right": {"namespace": "gallica", "value": F1},
                  "proven_page": True, "source": "s"}]},
    {"aliases": [{"left": {"namespace": "gallica", "value": "kb:00531074"}, "right": {"namespace": "gallica", "value": F1},
                  "proven_page": True, "source": "s"}]},
    {"journals": [{"journal": "DECISIONS.md ", "status": "pending"}]},
    {"limits": {"nzz ": ["LIMIT:x"]}},
])
def test_join_keys_that_are_not_exact_are_invalid(tmp_path, kw):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, **kw)


def test_the_exact_keys_of_pT_repros_still_refuse(tmp_path):
    kb = "kb:00531074"
    assert codes(kb, write_inputs(tmp_path, consultations=[C(1, "choose", pages=[kb])], seq_max=1))["refusal_codes"] == [
        "SUIVI_SELECTED:suivi_choose_seq1"]
    over = [{"page": kb, "check": "ngram8", "status": "complete", "findings": [{"kind": "train_overlap", "source": "r2"}]}]
    assert codes(kb, write_inputs(tmp_path / "o", checks=over))["refusal_codes"] == ["SUIVI_TEXT_OVERLAP:r2"]
    rows = SOURCES + [{"source": "pseudo", "key_spaces": ["kb"], "status": "non_resolue", "files": [], "proof": "p"}]
    assert codes(kb, write_inputs(tmp_path / "s", sources=rows))["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:pseudo"]
    assert codes(kb, write_inputs(tmp_path / "r", required_checks={"kb": ["ngram8"]}))["refusal_codes"] == ["SUIVI_CHECK_INCOMPLETE:ngram8"]


# amendment AV-G0-SUIVI-JOURNAUX (24/10): ["__none__"] with the archived search, benches without a source line, closed by default
ARCHIVE = {"path": {"root": "repo", "path": "reports/g0_suivi_pB/journal_inventory.json"}, "sha256": "a" * 64}
NONE_PROOF = {"inventory_sha256": "a" * 64, "file_sha256": "b" * 64, "collections": [], "benches": []}
BBVLM = [{"bench": "BBVLM", "proof": "cité sans ligne source (inventaire des journaux)"}]


def _journals(**extra):
    return lambda d: d["journals"].update(extra)


def test_none_with_proof_concerns_no_bench(tmp_path):
    j = [{"journal": "NOTE.md", "status": "pending", "concerns": ["__none__"], "none_proof": NONE_PROOF}]
    assert codes(F1, write_inputs(tmp_path, journals=j, mutate=_journals(search_archive=ARCHIVE)))["allowed"]
    assert codes(F1, write_inputs(tmp_path / "u", journals=j, usages=[("page", F1, "eval", "e22", "consumed")],
                                  mutate=_journals(search_archive=ARCHIVE)))["refusal_codes"] == []


@pytest.mark.parametrize("journal, archive", [
    ({"journal": "N", "status": "pending", "concerns": ["__none__"]}, ARCHIVE),                                       # no proof
    ({"journal": "N", "status": "pending", "concerns": ["__none__"], "none_proof": NONE_PROOF}, None),                 # no archive
    ({"journal": "N", "status": "pending", "concerns": ["__none__", "e22"], "none_proof": NONE_PROOF}, ARCHIVE),      # not alone
    ({"journal": "N", "status": "pending", "concerns": ["__none__"], "none_proof": dict(NONE_PROOF, inventory_sha256="c" * 64)}, ARCHIVE),
    ({"journal": "N", "status": "pending", "concerns": ["__none__"], "none_proof": dict(NONE_PROOF, benches=["E12"])}, ARCHIVE),
    ({"journal": "N", "status": "pending", "concerns": ["__none__"], "none_proof": dict(NONE_PROOF, collections=["newseye"])}, ARCHIVE),
    ({"journal": "N", "status": "pending", "concerns": ["__none__"], "none_proof": dict(NONE_PROOF, file_sha256="x")}, ARCHIVE),
    ({"journal": "N", "status": "pending", "concerns": ["e22"], "none_proof": NONE_PROOF}, ARCHIVE),                   # proof without none
    ({"journal": "N", "status": "pending", "concerns": ["__none__"], "none_proof": NONE_PROOF}, dict(ARCHIVE, sha256="0")),
])
def test_none_without_a_valid_proof_is_invalid(tmp_path, journal, archive):
    extra = {} if archive is None else {"search_archive": archive}
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, journals=[journal], mutate=_journals(**extra))


def test_declared_bench_without_source_refuses_no_page(tmp_path):
    j = [{"journal": "BBVLM.md", "status": "pending", "concerns": ["BBVLM"]}]
    for i, page in enumerate((F1, NZZ)):
        assert codes(page, write_inputs(tmp_path / str(i), journals=j, usages=[("page", F1, "eval", "e22", "consumed")],
                                        mutate=_journals(benches_without_source=BBVLM)))["refusal_codes"].count("SUIVI_SELECTION_UNREVIEWED:BBVLM.md") == 0


def test_declared_bench_next_to_a_source_still_refuses_that_source(tmp_path):
    j = [{"journal": "MIX.md", "status": "pending", "concerns": ["BBVLM", "e22"]}]
    inputs = write_inputs(tmp_path, journals=j, usages=[("page", F1, "eval", "e22", "consumed")], mutate=_journals(benches_without_source=BBVLM))
    assert "SUIVI_SELECTION_UNREVIEWED:MIX.md" in codes(F1, inputs)["refusal_codes"]


@pytest.mark.parametrize("journals, declared", [
    ([{"journal": "T.md", "status": "pending", "concerns": ["BBVML"]}], BBVLM),                 # a typo: undeclared, unknown name
    ([{"journal": "T.md", "status": "pending", "concerns": ["hans"]}], None),                    # undeclared bench
    ([], [{"bench": "e22", "proof": "p"}]),                                                      # declared but it has a source line
    ([], [{"bench": "gallica", "proof": "p"}]),                                                  # a space
    ([], [{"bench": "__none__", "proof": "p"}]),                                                 # the reserved word
    ([], [{"bench": "BBVLM"}]),                                                                  # no proof
    ([], BBVLM + BBVLM),                                                                         # duplicate
    ([], [{"bench": "BBVLM ", "proof": "p"}]),                                                   # surrounding space
])
def test_unknown_or_badly_declared_names_are_invalid(tmp_path, journals, declared):
    extra = {} if declared is None else {"benches_without_source": declared}
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, journals=journals, mutate=_journals(**extra))


@pytest.mark.parametrize("extra", [
    {"benches_without_source": [{"bench": "Gallica", "proof": "p"}]},                    # case of a space
    {"benches_without_source": [{"bench": "E22", "proof": "p"}]},                        # case of a source
    {"benches_without_source": [{"bench": "e22​", "proof": "p"}]},                  # zero-width space
    {"benches_without_source": "BBVLM"},                                                 # not a list
    {"benches_without_source": ["BBVLM"]},                                               # not objects
    {"search_archive": {"path": "journal_inventory.json", "sha256": "a" * 64}},          # bad shape, even without __none__
    {"search_archive": "x"},
])
def test_colliding_or_malformed_declarations_are_invalid_even_unused(tmp_path, extra):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, mutate=_journals(**extra))


def test_eligibility_alone_rejects_a_bad_journals_file_from_pinned_bytes(tmp_path):
    good = write_inputs(tmp_path)
    bad = json.loads(good.raw["journals"])
    bad["journals"].append({"journal": "T.md", "status": "pending", "concerns": ["BBVML"]})
    b = json.dumps(bad).encode()
    inputs = g0_suivi.SuiviInputs(dict(good.raw, journals=b), good.names, dict(good.anchors, journals=hashlib.sha256(b).hexdigest()))
    with pytest.raises(SuiviInputError):
        codes(F1, inputs)


# pQ NO-GO on 0816a65f (24/10): B1 "unknown" is not a PageIdentity space; B2 whitespace left after removing Cf; null fields; a source __none__
@pytest.mark.parametrize("journals, extra, sources", [
    ([{"journal": "U.md", "status": "pending", "concerns": ["unknown"]}], {}, None),                            # B1
    ([], {"benches_without_source": [{"bench": "e22 ​", "proof": "p"}]}, None),                              # B2
    ([], {"benches_without_source": [{"bench": "e 22", "proof": "p"}]}, None),                               # no-break space inside
    ([], {"benches_without_source": [{"bench": "BBVLM", "proof": "p"}, {"bench": "bbvlm", "proof": "p"}]}, None),  # two that fold together
    ([], {"search_archive": None}, None),                                                                         # null archive
    ([{"journal": "N.md", "status": "pending", "concerns": ["e22"], "none_proof": None}], {}, None),               # null proof
    ([], {}, SOURCES + [{"source": "__none__", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"}]),
])
def test_pq_blockers_are_invalid(tmp_path, journals, extra, sources):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, journals=journals, sources=sources, mutate=_journals(**extra))


def test_a_pending_journal_on_e22_still_refuses_the_page_used_by_e22(tmp_path):
    j = [{"journal": "E22.md", "status": "pending", "concerns": ["e22"]}]
    inputs = write_inputs(tmp_path, journals=j, usages=[("page", F1, "eval", "e22", "consumed")], mutate=_journals(benches_without_source=BBVLM))
    assert "SUIVI_SELECTION_UNREVIEWED:E22.md" in codes(F1, inputs)["refusal_codes"]


# amendment AV-G0-SUIVI-LM (pA, 24/10): an unresolved source of unknown space whose texts are all on disk owes each page an ngram8 check
LM_FILES = [{"path": {"root": "reviews_root", "path": "lm/pleias_322.jsonl"}, "sha256": "c" * 64}]


def _lm(on_disk=True, **extra):
    return dict({"source": "reader_lm", "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "p",
                 "lm_texts": {"files": LM_FILES, "documents": 322, "all_on_disk": on_disk}}, **extra)


def _ngram8(page=F1, status="complete", against=("reader_lm",), findings=(), manifests=None):
    c = {"page": page, "check": "ngram8", "status": status, "against": list(against), "findings": list(findings)}
    if manifests is None and "reader_lm" in against:
        manifests = {"reader_lm": g0_suivi.lm_fingerprint(LM_FILES)}
    if manifests:
        c["against_manifests"] = manifests
    return c


def test_lm_texts_on_disk_and_no_overlap_lets_the_page_pass(tmp_path):
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[_ngram8()]))
    assert out["allowed"], out["refusal_codes"]


def test_lm_overlap_refuses_the_page(tmp_path):
    chk = _ngram8(findings=[{"kind": "lm_overlap", "source": "reader_lm"}])
    assert codes(F1, write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[chk]))["refusal_codes"] == ["SUIVI_LM_OVERLAP:reader_lm"]


def test_lm_texts_not_all_on_disk_keep_the_total_refusal(tmp_path):
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [_lm(on_disk=False)], checks=[_ngram8()]))
    assert out["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:reader_lm"]


def test_an_unknown_row_without_lm_texts_still_refuses(tmp_path):
    row = {k: v for k, v in _lm().items() if k != "lm_texts"}
    assert codes(F1, write_inputs(tmp_path, sources=SOURCES + [row]))["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:reader_lm"]


@pytest.mark.parametrize("checks", [
    [],                                                    # no ngram8 at all
    [_ngram8(status="incomplete")],                        # incomplete
    [_ngram8(against=("e22",))],                           # complete but not against the LM source
])
def test_a_missing_or_partial_ngram8_refuses(tmp_path, checks):
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=checks))
    assert out["refusal_codes"] == ["SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm"]


def test_every_member_of_the_class_owes_the_check(tmp_path):
    inputs = write_inputs(tmp_path, sources=SOURCES + [_lm()], aliases=_proven(), checks=[_ngram8()])
    assert "SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm" in codes(F1, inputs)["refusal_codes"]


@pytest.mark.parametrize("sources, checks", [
    (SOURCES + [_lm(status="consumed")], []),                                   # lm_texts on a joining row
    (SOURCES + [_lm(key_spaces=["kb"])], []),                                   # neither unknown nor gallica
    (SOURCES + [_lm(key_spaces=["unknown", "gallica"])], []),
    (SOURCES + [dict(_lm(), lm_texts={"files": [], "documents": 322, "all_on_disk": True})], []),
    (SOURCES + [dict(_lm(), lm_texts={"files": [{"path": {"root": "r", "path": "p"}, "sha256": "x"}], "documents": 322, "all_on_disk": True})], []),
    (SOURCES + [dict(_lm(), lm_texts={"files": LM_FILES, "documents": 0, "all_on_disk": True})], []),
    (SOURCES + [dict(_lm(), lm_texts={"files": LM_FILES, "documents": 322, "all_on_disk": "yes"})], []),
    (SOURCES + [_lm()], [_ngram8(findings=[{"kind": "lm_overlap", "source": "e22"}])]),    # lm_overlap on a source without lm_texts
    (SOURCES + [_lm()], [_ngram8(against=("nowhere",))]),                                  # against an unknown source
    (SOURCES + [_lm()], [dict(_ngram8(), against=[])]),                                    # empty against
])
def test_malformed_lm_inputs_are_invalid(tmp_path, sources, checks):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=sources, checks=checks)


def test_lm_overlap_on_an_alias_member_refuses(tmp_path):
    alias = f"wikimedia-file-sha1:{SHA1}/f1"                                 # the proven alias of F1 (_proven)
    checks = [_ngram8(), _ngram8(page=alias, findings=[{"kind": "lm_overlap", "source": "reader_lm"}])]
    inputs = write_inputs(tmp_path, sources=SOURCES + [_lm()], aliases=_proven(), checks=checks)
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_LM_OVERLAP:reader_lm"]


def test_the_lm_exemption_leaves_usage_refusals_in_place(tmp_path):
    inputs = write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[_ngram8()], usages=[("page", F1, "train", "press/train", "consumed")])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_TRAINED:press/train"]
    inputs = write_inputs(tmp_path / "u", sources=SOURCES + [_lm()], checks=[_ngram8()], usages=[("page", F1, "eval", "ghost", "consumed")])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:ghost"]


def test_an_lm_row_still_leaves_nzz_uncovered(tmp_path):
    out = codes(NZZ, write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[_ngram8(page=NZZ)],
                                  required_checks={"nzz": ["catmus", "ngram8"]}))
    assert "SUIVI_SPACE_UNCOVERED:nzz" in out["refusal_codes"]


# AV-G0-SUIVI-LM v2 (pQ NO-GO B1): a check is bound to the manifest of the texts it compared; a stale check is refused
def test_a_check_on_older_texts_is_stale_and_refused(tmp_path):
    newer = [{"path": {"root": "reviews_root", "path": "lm/pleias_322.jsonl"}, "sha256": "d" * 64}]   # only the SHA of the LM file changes
    row = dict(_lm(), lm_texts={"files": newer, "documents": 322, "all_on_disk": True})
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [row], checks=[_ngram8()]))
    assert out["refusal_codes"] == ["SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm"]


def test_a_check_without_manifest_is_refused(tmp_path):
    chk = _ngram8()
    del chk["against_manifests"]
    assert codes(F1, write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[chk]))["refusal_codes"] == [
        "SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm"]


def test_the_fingerprint_ignores_file_order_but_not_content():
    a = {"path": {"root": "r", "path": "a"}, "sha256": "a" * 64}
    b = {"path": {"root": "r", "path": "b"}, "sha256": "b" * 64}
    assert g0_suivi.lm_fingerprint([a, b]) == g0_suivi.lm_fingerprint([b, a])
    assert g0_suivi.lm_fingerprint([a]) != g0_suivi.lm_fingerprint([dict(a, sha256="c" * 64)])
    assert g0_suivi.lm_fingerprint([a]) != g0_suivi.lm_fingerprint([dict(a, path={"root": "r", "path": "z"})])


@pytest.mark.parametrize("manifests", [
    {"e22": "a" * 64},                                       # a manifest for a source without lm_texts (and not in against)
    {"reader_lm": "x"},                                      # malformed fingerprint
    {},                                                      # empty object
])
def test_malformed_manifests_are_invalid(tmp_path, manifests):
    chk = _ngram8()
    chk["against_manifests"] = manifests
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[chk])


def test_a_manifest_for_a_source_in_against_but_without_lm_texts_is_invalid(tmp_path):
    chk = _ngram8(against=("reader_lm", "e22"), manifests={"reader_lm": g0_suivi.lm_fingerprint(LM_FILES), "e22": "a" * 64})
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=SOURCES + [_lm()], checks=[chk])


# pA decision on reader_lm_corpus (24/10): a joining row without any usage line is forbidden; the LM issues are refused, a new page passes
@pytest.mark.parametrize("status", ["consumed", "allocated_only"])
def test_a_joining_row_without_any_usage_line_is_invalid(tmp_path, status):
    row = {"source": "reader_lm_corpus", "key_spaces": ["bnl", "gallica", "wikimedia"], "status": status, "files": [], "proof": "p"}
    others = [("page", FAR, "eval", s_["source"], "consumed") for s_ in SOURCES]     # every other joining row has its usage line
    with pytest.raises(SuiviInputError, match="reader_lm_corpus"):
        write_inputs(tmp_path, sources=SOURCES + [row], usages=others, neutral=False)


def test_a_joining_row_with_a_usage_line_is_accepted(tmp_path):
    row = {"source": "reader_lm_corpus", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"}
    inputs = write_inputs(tmp_path, sources=SOURCES + [row], usages=[("document", "gallica:7777777", "train", "reader_lm_corpus", "consumed")])
    assert inputs.raw["usage"] is not None


def test_r7a_shape_lm_issue_refused_new_page_admitted(tmp_path):
    lm_row = {"source": "reader_lm_corpus", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"}
    pleias = dict(_lm(), source="pleias_french_pd_322")
    trained = [("document", DOC, "train", "reader_lm_corpus", "consumed")]              # one of the 109 issues learnt by the LM
    chk = lambda page: _ngram8(page=page, against=("pleias_french_pd_322",),             # noqa: E731
                               manifests={"pleias_french_pd_322": g0_suivi.lm_fingerprint(LM_FILES)})
    inputs = write_inputs(tmp_path, sources=SOURCES + [lm_row, pleias], usages=trained, checks=[chk(F1)])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_TRAINED:reader_lm_corpus"]  # a page of a learnt issue: refused
    new = "gallica:8888888/f3"                                                           # a page of an issue the LM never read
    inputs = write_inputs(tmp_path / "n", sources=SOURCES + [lm_row, pleias], usages=trained, checks=[chk(new)])
    out = codes(new, inputs)
    assert out["allowed"], out["refusal_codes"]


# AV-G0-SUIVI-TEXTES (pA, 24/10): an unresolved gallica source whose texts are on disk: ngram8 replaces the refusal of its space
def _gal(on_disk=True):
    return dict(_lm(on_disk=on_disk), source="gallica_pseudo", key_spaces=["gallica"])


def _ng_gal(page=F1, findings=()):
    return _ngram8(page=page, against=("gallica_pseudo",), findings=findings,
                   manifests={"gallica_pseudo": g0_suivi.lm_fingerprint(LM_FILES)})


def test_gallica_texts_on_disk_and_no_overlap_admit_a_new_gallica_page(tmp_path):
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [_gal()], checks=[_ng_gal()]))
    assert out["allowed"], out["refusal_codes"]


def test_gallica_texts_overlap_refuses(tmp_path):
    chk = _ng_gal(findings=[{"kind": "lm_overlap", "source": "gallica_pseudo"}])
    assert codes(F1, write_inputs(tmp_path, sources=SOURCES + [_gal()], checks=[chk]))["refusal_codes"] == ["SUIVI_LM_OVERLAP:gallica_pseudo"]


def test_gallica_texts_not_on_disk_keep_the_refusal_of_the_space(tmp_path):
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [_gal(on_disk=False)], checks=[_ng_gal()]))
    assert out["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:gallica_pseudo"]


def test_a_gallica_row_with_texts_owes_no_check_to_a_kb_page(tmp_path):
    kb = "kb:00531074"
    out = codes(kb, write_inputs(tmp_path, sources=SOURCES + [_gal()]))
    assert "SUIVI_CHECK_INCOMPLETE:ngram8:gallica_pseudo" not in out["refusal_codes"]
    assert not any(c.startswith("SUIVI_SOURCE_UNRESOLVED:gallica_pseudo") for c in out["refusal_codes"])


def test_a_gallica_row_without_texts_refuses_gallica_pages_only(tmp_path):
    row = {k: v for k, v in _gal().items() if k != "lm_texts"}
    assert "SUIVI_SOURCE_UNRESOLVED:gallica_pseudo" in codes(F1, write_inputs(tmp_path, sources=SOURCES + [row]))["refusal_codes"]
    assert "SUIVI_SOURCE_UNRESOLVED:gallica_pseudo" not in codes("kb:00531074", write_inputs(tmp_path / "k", sources=SOURCES + [row]))["refusal_codes"]


def test_a_gallica_page_still_owes_the_check_to_an_unknown_lm_source(tmp_path):
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [_gal(), _lm()], checks=[_ng_gal()]))
    assert out["refusal_codes"] == ["SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm"]


def _kb_with_gallica_alias():
    return [{"left": {"namespace": "kb", "value": "kb:00531074"}, "right": {"namespace": "gallica", "value": F1},
             "proven_page": True, "source": "concordance"}]


def test_a_kb_page_with_a_gallica_alias_owes_the_check_on_its_gallica_member(tmp_path):
    out = codes("kb:00531074", write_inputs(tmp_path, sources=SOURCES + [_gal()], aliases=_kb_with_gallica_alias()))
    assert "SUIVI_CHECK_INCOMPLETE:ngram8:gallica_pseudo" in out["refusal_codes"]


def test_a_kb_page_with_a_gallica_alias_is_refused_by_a_gallica_row_without_texts(tmp_path):
    row = {k: v for k, v in _gal().items() if k != "lm_texts"}
    out = codes("kb:00531074", write_inputs(tmp_path, sources=SOURCES + [row], aliases=_kb_with_gallica_alias()))
    assert "SUIVI_SOURCE_UNRESOLVED:gallica_pseudo" in out["refusal_codes"]


def test_mixed_spaces_cannot_carry_texts(tmp_path):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=SOURCES + [dict(_gal(), key_spaces=["gallica", "kb"])])


# AV-G0-SUIVI-LIMITES (pA, 24/10): the contract's pretraining limits (+ dots.mocr, pA's addition) on every page, never a refusal
def test_pretraining_limits_on_every_page_without_refusal(tmp_path):
    inputs = write_inputs(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(g0_suivi, "SUIVI_ANCHORS", dict(inputs.anchors))
        for page in (F1, "kb:00531074", NZZ):
            r = eligibility(registry(), "suivi", page_key=page, inputs=inputs)
            assert set(g0_suivi.PRETRAINING_LIMITS) <= set(r["labels"])
            assert not any(c.startswith("LIMIT") for c in r["refusal_codes"])
        r = eligibility(registry(), "suivi", page_key=F1, inputs=inputs)
        assert r["allowed"] and r["labels"] == sorted(r["labels"])          # an admitted page keeps its limits, sorted with the other labels
    assert g0_suivi.PRETRAINING_LIMITS == ("LIMIT:PP-DocLayoutV3", "LIMIT:blla", "LIMIT:dots.mocr", "LIMIT:dots.ocr", "LIMIT:german_print")


def test_mccatmus_stays_as_the_contract_says(tmp_path):
    assert not any("mccatmus" in x for x in g0_suivi.PRETRAINING_LIMITS)   # McCATMuS: train:third_party refusal and the nzz limit
    assert {sp: v for sp, v in g0_suivi.FORCED_LIMITS.items()} == {sp: ("LIMIT:mccatmus_non_french_uncovered",) for sp in ("nzz", "kb", "bnl")}


def test_the_mccatmus_limit_reaches_kb_and_bnl_pages_but_not_gallica(tmp_path):
    inputs = write_inputs(tmp_path)
    assert "LIMIT:mccatmus_non_french_uncovered" in codes("kb:00531074", inputs)["labels"]
    assert "LIMIT:mccatmus_non_french_uncovered" not in codes(F1, inputs)["labels"]


# AV-G0-SUIVI-ESPACES (pA, 24/10): impresso, icdar, hipe are declarable spaces; an unresolved source refuses only its own space
@pytest.mark.parametrize("space", ["impresso", "icdar", "hipe"])
def test_an_unresolved_source_of_a_declared_space_refuses_no_gallica_page(tmp_path, space):
    row = {"source": f"{space}_src", "key_spaces": [space], "status": "non_resolue", "files": [], "proof": "p"}
    out = codes(F1, write_inputs(tmp_path, sources=SOURCES + [row]))
    assert out["allowed"], out["refusal_codes"]


def test_an_unknown_row_still_refuses_where_a_declared_space_would_not(tmp_path):
    row = {"source": "x_src", "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "p"}
    assert codes(F1, write_inputs(tmp_path, sources=SOURCES + [row]))["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:x_src"]


def test_declared_spaces_have_no_page_key(tmp_path):
    for space in ("impresso", "icdar", "hipe"):
        assert space not in g0_suivi.NAMESPACES
        assert codes(f"{space}:abc", write_inputs(tmp_path / space))["refusal_codes"] == ["SUIVI_NO_PAGE_KEY"]


def test_a_journal_may_concern_a_declared_space(tmp_path):
    j = [{"journal": "HIPE.md", "status": "pending", "concerns": ["hipe"]}]
    assert codes(F1, write_inputs(tmp_path, journals=j))["allowed"]


# pQ NO-GO on 51cf8cbf (B-A): a pending journal concerning a declared space reaches the pages used by a source declared on that space
def test_a_journal_on_hipe_reaches_a_page_used_by_a_hipe_source(tmp_path):
    row = {"source": "hipe_src", "key_spaces": ["hipe", "gallica"], "status": "consumed", "files": [], "proof": "p"}
    j = [{"journal": "HIPE.md", "status": "pending", "concerns": ["hipe"]}]
    inputs = write_inputs(tmp_path, sources=SOURCES + [row], journals=j, usages=[("page", F1, "eval", "hipe_src", "consumed")])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SELECTION_UNREVIEWED:HIPE.md"]


def test_a_journal_on_hipe_does_not_reach_a_page_without_a_hipe_source(tmp_path):
    row = {"source": "hipe_src", "key_spaces": ["hipe", "gallica"], "status": "consumed", "files": [], "proof": "p"}
    j = [{"journal": "HIPE.md", "status": "pending", "concerns": ["hipe"]}]
    inputs = write_inputs(tmp_path, sources=SOURCES + [row], journals=j, usages=[("page", F2, "eval", "hipe_src", "consumed")])
    assert codes(F1, inputs)["allowed"]                     # the hipe source is in the table but does not apply to F1


def test_an_applied_source_declared_on_kb_brings_kb_journals_to_a_gallica_page(tmp_path):
    row = {"source": "mix_src", "key_spaces": ["kb", "gallica"], "status": "consumed", "files": [], "proof": "p"}
    j = [{"journal": "KB.md", "status": "pending", "concerns": ["kb"]}]
    inputs = write_inputs(tmp_path, sources=SOURCES + [row], journals=j, usages=[("page", F1, "eval", "mix_src", "consumed")])
    assert codes(F1, inputs)["refusal_codes"] == ["SUIVI_SELECTION_UNREVIEWED:KB.md"]


# AV-G0-SUIVI-MODELES (pA, 24/10): a pretrained third-party model is not a source of pages: no refusal, only its LIMIT label
def _model(name="dots_ocr", limit="LIMIT:dots.ocr", **extra):
    return dict({"source": name, "key_spaces": ["unknown"], "status": "modele_tiers_preentraine", "files": [], "proof": "p", "limit": limit},
                **extra)


def test_a_pretrained_model_refuses_no_page_and_its_limit_is_rendered(tmp_path):
    inputs = write_inputs(tmp_path, sources=SOURCES + [_model(), _model("kraken_blla_2026", "LIMIT:blla")])
    out = codes(F1, inputs)
    assert out["allowed"], out["refusal_codes"]


def test_a_model_row_does_not_cover_a_space(tmp_path):
    srcs = [s for s in SOURCES if "kb" not in s["key_spaces"]] + [_model()]
    assert "SUIVI_SPACE_UNCOVERED:kb" in codes("kb:00531074", write_inputs(tmp_path, sources=srcs))["refusal_codes"]


@pytest.mark.parametrize("row, usages", [
    (_model(), [("page", F1, "train", "dots_ocr", "consumed")]),          # a model with pages: forbidden
    (_model(limit="LIMIT:mccatmus_non_french_uncovered"), []),            # McCATMuS is resolved by E6, not a model row
    (_model(limit="LIMIT:anything"), []),                                 # not a contract limit
    (_model(key_spaces=["gallica"]), []),                                 # a model has no page space
    (_model(files=[{"path": {"root": "r", "path": "p"}, "sha256": "a" * 64}]), []),
    ({"source": "x", "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "p", "limit": "LIMIT:blla"}, []),  # limit elsewhere
])
def test_malformed_model_rows_are_invalid(tmp_path, row, usages):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=SOURCES + [row], usages=usages)


def test_r7a_new_gallica_page_with_models_lm_and_mccatmus_is_admitted(tmp_path):
    """The R7(a) shape after pA's decisions: models as limits, McCATMuS resolved by E6 (train on a contaminated book), the LM issues trained,
    pleias texts on disk with an ngram8: a page of an issue nobody used is admitted."""
    mcc = {"source": "mccatmus", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "E6 contamination.json d0555757"}
    lm_row = {"source": "reader_lm_corpus", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"}
    pleias = dict(_lm(), source="pleias_french_pd_322")
    usages = [("document", "gallica:5611930", "train", "mccatmus", "consumed"), ("document", DOC, "train", "reader_lm_corpus", "consumed")]
    srcs = SOURCES + [_model(), _model("kraken_blla_2026", "LIMIT:blla"), mcc, lm_row, pleias]
    new = "gallica:8888888/f3"
    chk = _ngram8(page=new, against=("pleias_french_pd_322",), manifests={"pleias_french_pd_322": g0_suivi.lm_fingerprint(LM_FILES)})
    out = codes(new, write_inputs(tmp_path, sources=srcs, usages=usages, checks=[chk]))
    assert out["allowed"], out["refusal_codes"]
    assert codes(F1, write_inputs(tmp_path / "t", sources=srcs, usages=usages, checks=[_ngram8(against=("pleias_french_pd_322",),
                 manifests={"pleias_french_pd_322": g0_suivi.lm_fingerprint(LM_FILES)})]))["refusal_codes"] == ["SUIVI_TRAINED:reader_lm_corpus"]


# Sonnet (conditional GO): names bound to limits; a model named by a journal, check or evidence is invalid; mccatmus can never be a model
@pytest.mark.parametrize("row", [
    _model(name="mccatmus", limit="LIMIT:blla"),                # McCATMuS disguised as a model
    _model(name="dots_ocr", limit="LIMIT:blla"),                # a known model with another model's limit
    _model(name="some_source", limit="LIMIT:blla"),             # a name outside MODEL_LIMITS
])
def test_model_names_are_bound_to_their_limit(tmp_path, row):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=SOURCES + [row])


@pytest.mark.parametrize("extra", [
    {"journals": [{"journal": "D.md", "status": "pending", "concerns": ["dots_ocr"]}]},
    {"checks": [{"page": F1, "check": "ngram8", "status": "complete", "against": ["dots_ocr"], "findings": []}]},
    {"checks": [{"page": F1, "check": "ngram8", "status": "complete", "findings": [{"kind": "eval_overlap", "source": "dots_ocr"}]}]},
    {"without_proof": [{"bench": "dots_ocr", "journal": "LOT.md"}]},
])
def test_a_model_relied_on_elsewhere_is_invalid(tmp_path, extra):
    with pytest.raises(SuiviInputError):
        write_inputs(tmp_path, sources=SOURCES + [_model()], **extra)


def test_a_model_row_does_not_lift_another_unknown_rows_refusal(tmp_path):
    other = {"source": "x_src", "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "p"}
    assert codes(F1, write_inputs(tmp_path, sources=SOURCES + [_model(), other]))["refusal_codes"] == ["SUIVI_SOURCE_UNRESOLVED:x_src"]


def test_r7a_shape_refuses_a_page_of_a_mccatmus_contaminated_book(tmp_path):
    mcc = {"source": "mccatmus", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "E6"}
    page = "gallica:5611930/f2"
    out = codes(page, write_inputs(tmp_path, sources=SOURCES + [_model(), mcc], usages=[("document", "gallica:5611930", "train", "mccatmus", "consumed")]))
    assert out["refusal_codes"] == ["SUIVI_TRAINED:mccatmus"]


# AV-G0-SUIVI-TITRES (pA, 25/10): disjunction by title of the collections whose number keys are incomplete
NEW, NEW_DOC = "gallica:7799999/f1", "gallica:7799999"
PAGE_CB, OTHER_CB = "cb32700000b", "cb11111111c"
PROOF = {"path": {"root": "repo", "path": "x.json"}, "sha256": "a" * 64}


def incomplete_names(reg):
    return [n for n, e in reg["collections"].items() if _scope(e) == "possible" and not g0_suivi._issue_keys_complete(e)]


def disjunction(reg, cb_of=None, documents=None, drop=(), shorten=()):
    """A title_disjunction covering every incomplete collection of `reg` with the registry's uncovered count (1 unit without a key)."""
    cols = {}
    for n in incomplete_names(reg):
        if n in drop:
            continue
        spec = reg["collections"][n]["match_keys"]["issue"]
        cov = spec.get("coverage") if isinstance(spec, dict) else None
        k = cov["total"] - cov["keyed"] if cov else 1
        k -= 1 if n in shorten else 0
        cols[n] = {"uncovered": [{"unit": f"{n}#{i}", "title_cb": (cb_of or {}).get(n, OTHER_CB)} for i in range(k)], "proof": PROOF}
    docs = [{"document": NEW_DOC, "title_cb": PAGE_CB, "proof": PROOF}] if documents is None else documents
    return {"collections": cols, "documents": docs}


def with_td(td):
    def m(docs):
        docs["sources"]["title_disjunction"] = td
    return m


def test_title_disjunction_lifts_every_incomplete_collection_whose_titles_are_pinned_and_other(tmp_path):
    reg = registry(complete=False)
    names = incomplete_names(reg)
    assert len(names) == 9
    before = codes(NEW, write_inputs(tmp_path / "a"), reg)
    assert {c for c in before["refusal_codes"] if c.startswith("INCOMPLETE_NUMBER_KEY:")} == {f"INCOMPLETE_NUMBER_KEY:{n}" for n in names}
    r = codes(NEW, write_inputs(tmp_path / "b", mutate=with_td(disjunction(reg))), reg)
    assert r["allowed"], r["refusal_codes"]
    assert {f"TITLE_DISJOINT:{n}" for n in names} <= set(r["labels"])


def test_a_collection_holding_the_title_of_the_page_keeps_its_refusal(tmp_path):
    reg = registry(complete=False)
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(disjunction(reg, cb_of={"newseye": PAGE_CB}))), reg)
    assert r["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]
    assert "TITLE_DISJOINT:newseye" not in r["labels"]


@pytest.mark.parametrize("case", ["null_cb", "short_count", "absent_collection", "no_document", "other_document"])
def test_without_a_complete_proven_title_set_the_refusal_stays(tmp_path, case):
    reg = registry(complete=False)
    td = {"null_cb": lambda: disjunction(reg, cb_of={"gallicorpora": None}),
          "short_count": lambda: disjunction(reg, shorten=("gallicorpora",)),
          "absent_collection": lambda: disjunction(reg, drop=("gallicorpora",)),
          "no_document": lambda: disjunction(reg, documents=[]),
          "other_document": lambda: disjunction(reg, documents=[{"document": "gallica:7636525", "title_cb": PAGE_CB, "proof": PROOF}])}[case]()
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)
    expected = ({f"INCOMPLETE_NUMBER_KEY:{n}" for n in incomplete_names(reg)} if case in ("no_document", "other_document")
                else {"INCOMPLETE_NUMBER_KEY:gallicorpora"})
    assert set(r["refusal_codes"]) == expected, r["refusal_codes"]


def test_a_collection_without_any_number_key_is_lifted_on_its_declared_units_only(tmp_path):
    reg = registry(complete=False)
    assert reg["collections"]["gallica_pseudo"]["match_keys"]["issue"] == g0_suivi.NO_KEY
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(disjunction(reg, cb_of={"gallica_pseudo": PAGE_CB}))), reg)
    assert r["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:gallica_pseudo"]


@pytest.mark.parametrize("member_cb, allowed", [(OTHER_CB, False), ("cb22222222d", True), (None, False)])
def test_every_gallica_member_of_the_proven_class_must_be_of_another_title(tmp_path, member_cb, allowed):
    reg = registry(complete=False)
    docs = [{"document": NEW_DOC, "title_cb": PAGE_CB, "proof": PROOF}]
    if member_cb:
        docs.append({"document": DOC, "title_cb": member_cb, "proof": PROOF})
    alias = {"left": {"namespace": "gallica", "value": NEW}, "right": {"namespace": "gallica", "value": F1},
             "proven_page": True, "source": "concordance"}
    r = codes(NEW, write_inputs(tmp_path, aliases=[alias], mutate=with_td(disjunction(reg, documents=docs))), reg)
    assert r["allowed"] is allowed, r["refusal_codes"]
    if not allowed:
        assert set(r["refusal_codes"]) == {f"INCOMPLETE_NUMBER_KEY:{n}" for n in incomplete_names(reg)}


def test_the_as_side_is_not_touched_by_the_title_disjunction(tmp_path):
    reg = registry(complete=False)
    r = codes("as:18840315_1/p3", write_inputs(tmp_path, mutate=with_td(disjunction(reg))), reg)
    assert r["refusal_codes"][0] == "SUIVI_SPACE_UNCOVERED:as"


def test_the_title_disjunction_lifts_nothing_else(tmp_path):
    reg = registry(complete=False)
    r = codes(NEW, write_inputs(tmp_path, usages=[("page", NEW, "train", "press/train", "consumed")],
                                mutate=with_td(disjunction(reg))), reg)
    assert r["refusal_codes"] == ["SUIVI_TRAINED:press/train"]


@pytest.mark.parametrize("edit", [
    lambda t: t["documents"][0].update(title_cb="cb1234567"),            # 7 digits
    lambda t: t["documents"][0].update(title_cb="cb12345678a"),          # 'a' is not an ARK check character
    lambda t: t["documents"][0].update(document="gallica:7799999/f1"),   # a page, not a document
    lambda t: t["documents"][0].update(document="gallica:7799999 "),
    lambda t: t["documents"][0].pop("proof"),
    lambda t: t["documents"].append(dict(t["documents"][0])),            # duplicate document
    lambda t: t["collections"]["newseye"]["uncovered"].append(dict(t["collections"]["newseye"]["uncovered"][0])),
    lambda t: t["collections"]["newseye"].update(uncovered=[]),
    lambda t: t["collections"]["newseye"].update(proof={"path": {"root": "repo", "path": "x"}, "sha256": "A" * 64}),
    lambda t: t["collections"]["newseye"]["uncovered"][0].update(extra=1),
    lambda t: t.update(more=1),
    lambda t: t.pop("documents"),
])
def test_a_malformed_title_disjunction_is_invalid(tmp_path, edit):
    td = disjunction(registry(complete=False))
    edit(td)
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


def whole_titles(reg, n, rows):
    """Collection level (pA, 25/10): the complete set of the titles of `n`, with their unit counts."""
    td = disjunction(reg)
    td["collections"][n] = {"titles": [dict(zip(("title_cb", "units"), r)) for r in rows], "proof": PROOF}
    return td


def total_of(reg, n):
    return reg["collections"][n]["match_keys"]["issue"]["coverage"]["total"]


@pytest.mark.parametrize("rows, allowed", [
    (lambda t: [(OTHER_CB, t - 1), ("cb22222222d", 1)], True),          # complete, other titles
    (lambda t: [(OTHER_CB, t - 1), (PAGE_CB, 1)], False),               # the title of the page is in the collection
    (lambda t: [(OTHER_CB, t - 1)], False),                             # one unit short of the registry total
    (lambda t: [(OTHER_CB, t), ("cb22222222d", 1)], False),             # one unit too many
    (lambda t: [(OTHER_CB, t - 1), (None, 1)], False),                  # a title not known
])
def test_the_collection_level_proof_needs_the_complete_title_set(tmp_path, rows, allowed):
    reg = registry(complete=False)
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(whole_titles(reg, "newseye", rows(total_of(reg, "newseye"))))), reg)
    assert r["allowed"] is allowed, r["refusal_codes"]
    assert r["refusal_codes"] == ([] if allowed else ["INCOMPLETE_NUMBER_KEY:newseye"])


def test_only_the_last_uncovered_unit_carrying_the_page_title_refuses(tmp_path):
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"]["uncovered"][-1]["title_cb"] = PAGE_CB
    assert codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]


def test_one_uncovered_unit_too_many_keeps_the_refusal(tmp_path):
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"]["uncovered"].append({"unit": "newseye#extra", "title_cb": OTHER_CB})
    assert codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]


@pytest.mark.parametrize("cov", ["?", {"keyed": "149", "total": 203}, {"keyed": True, "total": 203}, "absent", None,
                                 {"keyed": -1, "total": 0}, {"keyed": 5, "total": 4}])
def test_a_malformed_registry_coverage_keeps_the_refusal(tmp_path, cov):
    """pQ M1: only NO_KEY is declarative; a key object needs a valid coverage with 0 <= keyed <= total, the count is checked against it."""
    reg = registry(complete=False)
    spec = reg["collections"]["newseye"]["match_keys"]["issue"]
    if cov == "absent":
        spec.pop("coverage")
    else:
        spec["coverage"] = cov
    n = (cov["total"] - cov["keyed"]) if isinstance(cov, dict) and all(isinstance(v, int) for v in cov.values()) else 1
    td = disjunction(reg) if n <= 0 else disjunction(registry(complete=False))
    td["collections"]["newseye"]["uncovered"] = [{"unit": f"u{i}", "title_cb": OTHER_CB} for i in range(max(n, 1))]
    assert g0_suivi._title_disjoint(reg["collections"]["newseye"], td, "newseye", [NEW]) is False


@pytest.mark.parametrize("unit", [NEW, NEW_DOC, NEW.upper()])
def test_a_member_listed_among_the_uncovered_units_keeps_the_refusal(tmp_path, unit):
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"]["uncovered"][0]["unit"] = unit
    assert codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]


@pytest.mark.parametrize("edit", [
    lambda c: c["uncovered"].__setitem__(1, {"unit": c["uncovered"][0]["unit"].upper(), "title_cb": OTHER_CB}),   # same unit, other case
    lambda c: c.update(titles=[{"title_cb": OTHER_CB, "units": 1}]),                                           # both levels at once
    lambda c: (c.pop("uncovered"), c.update(titles=[{"title_cb": OTHER_CB, "units": 0}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[{"title_cb": OTHER_CB, "units": True}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[{"title_cb": OTHER_CB, "units": 1}, {"title_cb": OTHER_CB, "units": 1}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[{"title_cb": OTHER_CB, "units": 1, "name": " "}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[])),
])
def test_a_malformed_collection_entry_is_invalid(tmp_path, edit):
    td = disjunction(registry(complete=False))
    edit(td["collections"]["newseye"])
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_real_bnf_title_records_are_accepted():
    # pB and pA (25/10): the "9-digit" records are 11 characters, cb + 8 digits + a check character that is itself a digit
    for cb in ("cb32837632m", "cb328378921", "cb327986698", "cb343488023", "cb32780022t", "cb40216887k"):
        assert g0_suivi.CB.fullmatch(cb)
    for bad in ("cb3283789210", "cb32837632mm", "cb3283763m", "CB32837632M"):   # 12 or 10 characters, upper case: not a BnF record
        assert not g0_suivi.CB.fullmatch(bad)


# AV-G0-SUIVI-TITRES-ESPACES (pA, 25/10, after e12_test: 10 gallica pages and kb:00531072, kb:00531073): a unit of another space counts
# with its space instead of a cb and never blocks the disjunction of a Gallica page
@pytest.mark.parametrize("rows, allowed", [
    (lambda t: [(OTHER_CB, t - 2), ("kb", 2)], True),                   # complete: gallica titles + 2 kb units
    (lambda t: [(OTHER_CB, t - 2), ("kb", 1), ("bnl", 1)], True),       # two other spaces
    (lambda t: [("kb", t)], True),                                      # no Gallica unit at all
    (lambda t: [(OTHER_CB, t - 2), ("kb", 1)], False),                  # the space units still count: one short
    (lambda t: [(OTHER_CB, t - 2), ("kb", 3)], False),                  # one too many
    (lambda t: [(PAGE_CB, t - 2), ("kb", 2)], False),                   # the Gallica units keep their cb check
    (lambda t: [(None, t - 2), ("kb", 2)], False),                      # and a null cb keeps the refusal
])
def test_units_of_another_space_count_for_the_title_set_by_their_space(tmp_path, rows, allowed):
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"] = {"titles": [{"space": a, "units": u} if a in g0_suivi.OTHER_SPACES else {"title_cb": a, "units": u}
                                               for a, u in rows(total_of(reg, "newseye"))], "proof": PROOF}
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)
    assert r["allowed"] is allowed, r["refusal_codes"]
    assert r["refusal_codes"] == ([] if allowed else ["INCOMPLETE_NUMBER_KEY:newseye"])


def test_uncovered_units_of_another_space_carry_their_space(tmp_path):
    reg = registry(complete=False)
    td = disjunction(reg)
    k = len(td["collections"]["newseye"]["uncovered"])
    td["collections"]["newseye"]["uncovered"] = [{"unit": f"kb:{531000 + i:08d}", "space": "kb"} for i in range(k)]
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)
    assert r["allowed"], r["refusal_codes"]
    assert "TITLE_DISJOINT:newseye" in r["labels"]


@pytest.mark.parametrize("edit", [
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": "gallica", "units": 1}])),                   # gallica needs a cb
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": "unknown", "units": 1}])),                   # not a known space
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": "KB", "units": 1}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": ["kb"], "units": 1}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": "kb", "units": 0}])),
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": "kb", "units": 1}, {"space": "kb", "units": 1}])),   # one entry per space
    lambda c: (c.pop("uncovered"), c.update(titles=[{"space": "kb", "title_cb": OTHER_CB, "units": 1}])),  # a space OR a cb
    lambda c: c["uncovered"].__setitem__(0, {"unit": "kb:00531072", "space": "bnl"}),                     # the space of the key
    lambda c: c["uncovered"].__setitem__(0, {"unit": "gallica:7636525/f2", "space": "kb"}),              # a Gallica unit
    lambda c: c["uncovered"].__setitem__(0, {"unit": "gallica:7636525/f2", "space": "gallica"}),
    lambda c: c["uncovered"].__setitem__(0, {"unit": "newseye#0", "space": "kb"}),                        # no space in the key
    lambda c: c["uncovered"].__setitem__(0, {"unit": "kb:00531072", "space": "kb", "title_cb": OTHER_CB}),
])
def test_a_malformed_space_entry_is_invalid(tmp_path, edit):
    td = disjunction(registry(complete=False))
    edit(td["collections"]["newseye"])
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_uncovered_units_may_mix_title_records_and_other_spaces(tmp_path):
    reg = registry(complete=False)
    td = disjunction(reg)
    unc = td["collections"]["newseye"]["uncovered"]
    assert len(unc) >= 2
    unc[0], unc[1] = {"unit": "kb:00531072", "space": "kb"}, {"unit": "bnl:123456", "space": "bnl"}
    assert codes(NEW, write_inputs(tmp_path / "a", mutate=with_td(td)), reg)["allowed"]
    unc.pop()   # the space units count: one short keeps the refusal
    assert codes(NEW, write_inputs(tmp_path / "b", mutate=with_td(td)), reg)["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]


def test_a_declared_space_without_a_key_format_is_collection_level_only(tmp_path):
    """Sonnet m1: impresso, icdar, hipe have no page key, so only {space, units} can name them; {unit, space} needs the key's space."""
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"] = {"titles": [{"title_cb": OTHER_CB, "units": total_of(reg, "newseye") - 1}, {"space": "impresso", "units": 1}],
                                    "proof": PROOF}
    assert codes(NEW, write_inputs(tmp_path / "a", mutate=with_td(td)), reg)["allowed"]
    td = disjunction(reg)
    td["collections"]["newseye"]["uncovered"][0] = {"unit": "impresso:x", "space": "impresso"}
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path / "b", mutate=with_td(td))


@pytest.mark.parametrize("edit", [
    lambda c: c["uncovered"].__setitem__(0, {"unit": "KB:00531072", "space": "kb"}),   # the key's space is exact, not case-folded
    lambda c: c["uncovered"].__setitem__(1, {"unit": "KB:00531072", "space": "kb"}) or c["uncovered"].__setitem__(0, {"unit": "kb:00531072", "title_cb": OTHER_CB}),
])
def test_space_entries_follow_the_unit_rules(tmp_path, edit):
    td = disjunction(registry(complete=False))
    edit(td["collections"]["newseye"])
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_a_collection_without_number_key_is_lifted_on_declared_space_units(tmp_path):
    """Sonnet m2, declared limit: without a coverage (NO_KEY) the count is declared, space units as title records."""
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["gallica_pseudo"] = {"titles": [{"space": "kb", "units": 1}], "proof": PROOF}
    r = codes(NEW, write_inputs(tmp_path, mutate=with_td(td)), reg)
    assert "TITLE_DISJOINT:gallica_pseudo" in r["labels"], r["refusal_codes"]


# pQ NO-GO on TITRES-ESPACES (25/10): B1 a space unit must be a real canonical page key; B2 a proven page alias of another space is the same
# physical page, so the whole class is confronted with the units, and a space counted without keys cannot hold a member of the class
@pytest.mark.parametrize("unit", ["kb:not-a-key-1", "kb:0053107", "kb:00531072 ", "kb:"])
def test_a_space_unit_needs_a_canonical_page_key(tmp_path, unit):
    td = disjunction(registry(complete=False))
    td["collections"]["newseye"]["uncovered"][0] = {"unit": unit, "space": "kb"}
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


KB_ALIAS = {"left": {"namespace": "gallica", "value": NEW}, "right": {"namespace": "kb", "value": "kb:00531072"},
            "proven_page": True, "source": "concordance"}


@pytest.mark.parametrize("level", ["unit", "collection"])
def test_a_proven_alias_in_another_space_keeps_the_refusal(tmp_path, level):
    reg = registry(complete=False)
    td = disjunction(reg)
    t = total_of(reg, "newseye")
    if level == "unit":
        td["collections"]["newseye"]["uncovered"][0] = {"unit": "kb:00531072", "space": "kb"}
    else:
        td["collections"]["newseye"] = {"titles": [{"title_cb": OTHER_CB, "units": t - 2}, {"space": "kb", "units": 2}], "proof": PROOF}
    assert codes(NEW, write_inputs(tmp_path / "a", mutate=with_td(td)), reg)["allowed"]
    r = codes(NEW, write_inputs(tmp_path / "b", aliases=[KB_ALIAS], mutate=with_td(td)), reg)
    assert r["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"], r["refusal_codes"]


def test_a_proven_alias_in_a_space_the_titles_do_not_count_still_lifts(tmp_path):
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"] = {"titles": [{"title_cb": OTHER_CB, "units": total_of(reg, "newseye") - 1}, {"space": "bnl", "units": 1}],
                                    "proof": PROOF}
    r = codes(NEW, write_inputs(tmp_path, aliases=[KB_ALIAS], mutate=with_td(td)), reg)
    assert r["allowed"], r["refusal_codes"]


def test_a_proven_alias_to_another_kb_page_than_the_listed_ones_still_lifts(tmp_path):
    """Sonnet m4: at unit level the uncovered list is exact (count tied to total - keyed); a kb alias not listed is not a unit of it."""
    reg = registry(complete=False)
    td = disjunction(reg)
    td["collections"]["newseye"]["uncovered"][0] = {"unit": "kb:00531072", "space": "kb"}
    alias = dict(KB_ALIAS, right={"namespace": "kb", "value": "kb:00531099"})
    r = codes(NEW, write_inputs(tmp_path, aliases=[alias], mutate=with_td(td)), reg)
    assert r["allowed"], r["refusal_codes"]


# AV-G0-SUIVI-TITRES-NOMS (pA, 25/10): disjunction by NAME, a distinct and weaker label TITLE_DISJOINT_NAME:<c>
REF_DOC = {"document": NEW_DOC, "ref_title": "Le Petit Troyen", "proof": PROOF}


def by_names(reg, n, names, method="file_names", documents=None):
    """Every incomplete collection by title record, except `n` given by the complete set of its title names (name or space, units)."""
    td = disjunction(reg, documents=[dict(REF_DOC, title_cb=PAGE_CB)] if documents is None else documents)
    td["collections"][n] = {"names": [{"space": a, "units": u} if a in g0_suivi.OTHER_SPACES else {"name": a, "units": u} for a, u in names],
                            "method": method, "proof": PROOF}
    return td


def name_codes(tmp_path, td, reg, aliases=()):
    return codes(NEW, write_inputs(tmp_path, aliases=list(aliases), mutate=with_td(td)), reg)


def test_a_complete_name_set_without_any_ref_word_lifts_with_the_weaker_label(tmp_path):
    reg = registry(complete=False)
    t = total_of(reg, "newseye")
    r = name_codes(tmp_path, by_names(reg, "newseye", [("Ce soir", t - 3), ("Regards", 1), ("L'Œuvre", 1), ("Le Matin", 1)]), reg)
    assert r["allowed"], r["refusal_codes"]
    assert "TITLE_DISJOINT_NAME:newseye" in r["labels"] and "TITLE_DISJOINT:newseye" not in r["labels"]


@pytest.mark.parametrize("ref, other", [("L'Écho de Paris", "L'ÉCHO"), ("L'Action pyrénéenne", "Journal des Pyrénées"),
                                        ("Le Progrès de la Côte-d'Or", "Côte d'Azur"), ("La Petite Gironde", "Gironde libre"),
                                        ("Le Petit Troyen", "Le Troyen"), ("Le Petit Marseillais", "Le Marseillais"),
                                        ("Le Petit Marseillais", "Marseille-Matin"), ("Le Petit Marseillais", "Le Petit"),
                                        ("La Petite Gironde", "Le Petit"), ("Le Petit Troyen", "Le Petit"), ("L'Écho de Paris", "Les Échos de Paris"),
                                        ("L'Écho de Paris", "L'Écho de Paris illustré"), ("Le Progrès de la Côte-d'Or", "Le Progrès"),
                                        ("L'Action pyrénéenne", "L'Action"), ("L'Écho de Paris", "Paris, l'Écho du soir"),
                                        ("L'Écho de Paris", "Les Échos")])   # since échos joined the generic words (pA, 25/10)
def test_a_distinctive_word_or_an_ambiguous_generic_name_refuses(tmp_path, ref, other):
    """NOMS v2 (pA, 25/10): a distinctive word of the REF title met, a generic-only REF title found whole, or a generic-only name inside it."""
    reg = registry(complete=False)
    t = total_of(reg, "newseye")
    docs = [dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)]
    r = name_codes(tmp_path, by_names(reg, "newseye", [("Ce soir", t - 1), (other, 1)], documents=docs), reg)
    assert r["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"], r["refusal_codes"]


@pytest.mark.parametrize("ref, other", [("Le Petit Marseillais", "Le Petit Parisien"), ("La Petite Gironde", "Le Petit Parisien"),
                                        ("Le Petit Troyen", "Le Petit Parisien"), ("L'Action pyrénéenne", "L'Action française"),
                                        ("L'Écho de Paris", "Paris-Soir"), ("Le Petit Troyen", "Le Petit Journal"),
                                        ("L'Action pyrénéenne", "La Réaction"), ("Le Progrès de la Côte-d'Or", "Le Progrès du Nord"),
                                        ("L'Écho de Paris", "Le Parisien"), ("Le Progrès de la Côte-d'Or", "Prog."),
                                        ("L'Écho de Paris", "Le Petit Parisien"), ("Le Petit Marseillais", "La Petite République")])
def test_shared_generic_words_alone_do_not_meet(tmp_path, ref, other):
    """NOMS v2 (pA, 25/10), the real cases first, written before computing them: two titles sharing only generic words are other titles."""
    reg = registry(complete=False)
    docs = [dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)]
    r = name_codes(tmp_path, by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye") - 1), (other, 1)], documents=docs), reg)
    assert r["allowed"], r["refusal_codes"]
    assert "TITLE_DISJOINT_NAME:newseye" in r["labels"]


def test_the_generic_words_are_the_closed_list_of_pa():
    assert g0_suivi.GENERIC == {"petit", "petite", "journal", "echo", "gazette", "courrier", "progres", "action", "nouvelliste", "republicain",
                                "moniteur", "soir", "matin", "nouvelles", "depeche", "presse", "paris", "france",
                                "petits", "petites", "journaux", "echos", "gazettes", "courriers", "depeches"}
    assert {k: [f - g0_suivi.GENERIC for f in v] for k, v in g0_suivi.REF_FORMS.items()} == {
        "L'Écho de Paris": [set()], "Le Petit Marseillais": [{"marseillais"}], "La Petite Gironde": [{"gironde"}],
        "Le Petit Troyen": [{"troyen"}], "L'Action pyrénéenne": [{"pyreneenne"}], "Le Progrès de la Côte-d'Or": [{"cote", "or"}]}


E12_NAMES = ["Ce soir", "Regards", "Paris-Soir", "Marianne", "L'Œuvre", "La Presse", "Le Matin"]


@pytest.mark.parametrize("ref, allowed", [("Le Petit Marseillais", True), ("La Petite Gironde", True), ("Le Petit Troyen", True),
                                          ("L'Action pyrénéenne", True), ("Le Progrès de la Côte-d'Or", True), ("L'Écho de Paris", True)])
def test_the_name_refusal_is_per_page(tmp_path, ref, allowed):
    """pA (25/10): NOMS refused the pages of L'Écho de Paris for « paris » of Paris-Soir; NOMS v2 admits them, paris and soir being generic."""
    reg = registry(complete=False)
    rows = [(n, 1) for n in E12_NAMES] + [("kb", 2)]
    rows[0] = ("Ce soir", total_of(reg, "newseye") - 8)
    docs = [dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)]
    r = name_codes(tmp_path, by_names(reg, "newseye", rows, documents=docs), reg)
    assert r["allowed"] is allowed, r["refusal_codes"]
    assert ("TITLE_DISJOINT_NAME:newseye" in r["labels"]) is allowed


def test_a_class_with_members_of_two_ref_titles_is_checked_against_both(tmp_path):
    reg = registry(complete=False)
    alias = {"left": {"namespace": "gallica", "value": NEW}, "right": {"namespace": "gallica", "value": F1}, "proven_page": True, "source": "c"}
    rows = [("Ce soir", total_of(reg, "newseye") - 1), ("L'Écho", 1)]
    docs = [dict(REF_DOC, ref_title="Le Petit Marseillais", title_cb=PAGE_CB),
            {"document": DOC, "title_cb": "cb22222222d", "ref_title": "L'Écho de Paris", "proof": PROOF}]
    r = name_codes(tmp_path, by_names(reg, "newseye", rows, documents=docs), reg, [alias])
    assert "INCOMPLETE_NUMBER_KEY:newseye" in r["refusal_codes"]


@pytest.mark.parametrize("case", ["short", "long", "no_ref_title", "other_document", "kb_alias", "no_gallica_member"])
def test_the_name_level_keeps_the_refusal_when_a_condition_fails(tmp_path, case):
    reg = registry(complete=False)
    t = total_of(reg, "newseye")
    rows = [("Ce soir", t - 2), ("kb", 2)]
    aliases, docs = [], None
    if case == "short":
        rows[0] = ("Ce soir", t - 3)
    if case == "long":
        rows[0] = ("Ce soir", t - 1)
    if case == "no_ref_title":
        docs = [{"document": NEW_DOC, "title_cb": PAGE_CB, "proof": PROOF}]
    if case == "other_document":
        docs = [dict(REF_DOC, document="gallica:7636525")]
    if case == "kb_alias":
        aliases = [KB_ALIAS]
    td = by_names(reg, "newseye", rows, documents=docs)
    page = "kb:00531099" if case == "no_gallica_member" else NEW
    r = codes(page, write_inputs(tmp_path, aliases=aliases, mutate=with_td(td)), reg)
    assert "TITLE_DISJOINT_NAME:newseye" not in r["labels"]
    if page == NEW:
        assert "INCOMPLETE_NUMBER_KEY:newseye" in r["refusal_codes"], r["refusal_codes"]


def test_every_gallica_member_must_be_a_page_of_a_ref_title(tmp_path):
    reg = registry(complete=False)
    alias = {"left": {"namespace": "gallica", "value": NEW}, "right": {"namespace": "gallica", "value": F1}, "proven_page": True, "source": "c"}
    rows = [("Ce soir", total_of(reg, "newseye"))]
    only_new = [dict(REF_DOC, title_cb=PAGE_CB), {"document": DOC, "title_cb": "cb22222222d", "proof": PROOF}]
    r = name_codes(tmp_path / "a", by_names(reg, "newseye", rows, documents=only_new), reg, [alias])
    assert "INCOMPLETE_NUMBER_KEY:newseye" in r["refusal_codes"]
    both = [dict(REF_DOC, title_cb=PAGE_CB), {"document": DOC, "title_cb": "cb22222222d", "ref_title": "La Petite Gironde", "proof": PROOF}]
    assert name_codes(tmp_path / "b", by_names(reg, "newseye", rows, documents=both), reg, [alias])["allowed"]


def test_a_collection_without_number_key_is_lifted_by_names_on_its_declared_units(tmp_path):
    reg = registry(complete=False)
    r = name_codes(tmp_path, by_names(reg, "gallica_pseudo", [("Ce soir", 1)]), reg)
    assert "TITLE_DISJOINT_NAME:gallica_pseudo" in r["labels"], r["refusal_codes"]


def test_the_name_level_lifts_nothing_else_and_never_gives_the_title_record_label(tmp_path):
    reg = registry(complete=False)
    td = by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye"))])
    assert g0_suivi._title_disjoint(reg["collections"]["newseye"], td, "newseye", [NEW]) is False
    r = codes(NEW, write_inputs(tmp_path, usages=[("page", NEW, "train", "press/train", "consumed")], mutate=with_td(td)), reg)
    assert r["refusal_codes"] == ["SUIVI_TRAINED:press/train"]


def test_the_ref_titles_are_pinned_with_their_words():
    assert g0_suivi.REF_WORDS == {"L'Écho de Paris": {"echo", "paris"}, "Le Petit Marseillais": {"petit", "marseillais"},
                                  "La Petite Gironde": {"petite", "gironde"}, "Le Petit Troyen": {"petit", "troyen"},
                                  "L'Action pyrénéenne": {"action", "pyreneenne"}, "Le Progrès de la Côte-d'Or": {"progres", "cote", "or"}}


@pytest.mark.parametrize("a, b, meet", [("petit", "petite", True), ("pyrenees", "pyreneenne", True), ("paris", "parisien", True),
                                        ("echo", "echos", True), ("or", "ordre", False), ("soir", "soiree", True), ("cote", "cotes", True),
                                        ("matin", "marseillais", False), ("presse", "progres", False)])
def test_near_forms_of_a_word_meet(a, b, meet):
    assert g0_suivi._words_overlap(a, b) is meet


@pytest.mark.parametrize("edit", [
    lambda c: c.update(method="guess"),
    lambda c: c.pop("method"),
    lambda c: c.update(names=[]),
    lambda c: c["names"].append({"name": "La", "units": 1}),                      # no significant word
    lambda c: c["names"].append({"name": "CE SOIR!", "units": 1}),                # same words as « Ce soir »
    lambda c: c["names"].append({"name": "Le Matin", "units": 0}),
    lambda c: c["names"].append({"name": "Le Matin", "title_cb": OTHER_CB, "units": 1}),
    lambda c: c["names"].append({"space": "gallica", "units": 1}),
    lambda c: c["names"].extend([{"space": "kb", "units": 1}, {"space": "kb", "units": 1}]),
    lambda c: c.update(titles=[{"title_cb": OTHER_CB, "units": 1}]),
])
def test_a_malformed_names_entry_is_invalid(tmp_path, edit):
    td = by_names(registry(complete=False), "newseye", [("Ce soir", 1)])
    edit(td["collections"]["newseye"])
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


@pytest.mark.parametrize("doc", [{"document": NEW_DOC, "proof": PROOF}, {"document": NEW_DOC, "ref_title": "Le Petit Parisien", "proof": PROOF},
                                 {"document": NEW_DOC, "ref_title": "le petit troyen", "proof": PROOF},
                                 {"document": NEW_DOC, "ref_title": "Le Petit Troyen", "title_cb": "cb1", "proof": PROOF}])
def test_a_document_needs_a_title_record_or_a_pinned_ref_title(tmp_path, doc):
    td = disjunction(registry(complete=False), documents=[doc])
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_names_are_compared_after_spelling_out_the_ligatures(tmp_path):
    assert g0_suivi._name_words("L'Œuvre") == g0_suivi._name_words("l'oeuvre") == {"oeuvre"}
    td = by_names(registry(complete=False), "newseye", [("L'Œuvre", 1), ("L'Oeuvre", 1)])   # one title written twice
    with pytest.raises(SuiviInputError, match="duplicate name, group or space"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_the_name_level_needs_a_gallica_member():
    reg = registry(complete=False)
    td = by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye"))])
    assert g0_suivi._name_disjoint(reg["collections"]["newseye"], td, "newseye", ["kb:00531099"]) is False


@pytest.mark.parametrize("bad", ["Pe­tit Journal", "Pe​tit Journal", "Éc‌ho", "Pe⁠tit", "Pe͏tit", "Рetit Journal",
                                 "Aсtion", "Λction", "Petıt", "Ce soir\t", "Le Matin\n"])
def test_a_name_with_an_invisible_or_lookalike_character_is_invalid(tmp_path, bad):
    """Sonnet B1, M1: such a name would cut or hide a REF word; it is INVALID rather than compared."""
    td = by_names(registry(complete=False), "newseye", [(bad, 1)])
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


@pytest.mark.parametrize("ref, other", [("Le Petit Troyen", "LePetitTroyen"), ("L'Écho de Paris", "LEchoDeParis"),
                                        ("L'Action pyrénéenne", "L'Aution pyrénéenne"), ("Le Petit Troyen", "Ｐｅｔｉｔ"),
                                        ("Le Petit Troyen", "Pétit Troyes"), ("La Petite Gironde", "« La Petite » (Gironde)")])
def test_glued_or_differently_written_ref_words_still_meet(tmp_path, ref, other):
    reg = registry(complete=False)
    docs = [dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)]
    r = name_codes(tmp_path, by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye") - 1), (other, 1)], documents=docs), reg)
    assert r["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"], r["refusal_codes"]


@pytest.mark.parametrize("ref, other", [("Le Petit Troyen", "Pe-tit"), ("Le Petit Troyen", "Pe'tit"), ("Le Petit Troyen", "Pe.tit"),
                                        ("Le Petit Troyen", "Pe tit"), ("Le Petit Troyen", "Pe/tit & Troy-es"), ("L'Écho de Paris", "Le Éc-ho"),
                                        ("L'Écho de Paris", "Éc-ho de Pa ris-Midi"), ("Le Progrès de la Côte-d'Or", "Co.te Nord")])
def test_a_ref_word_cut_by_punctuation_or_a_space_still_meets(tmp_path, ref, other):
    """Sonnet v2 B1: the name is also compared with every separator taken out."""
    reg = registry(complete=False)
    docs = [dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)]
    r = name_codes(tmp_path, by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye") - 1), (other, 1)], documents=docs), reg)
    assert r["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"], r["refusal_codes"]


@pytest.mark.parametrize("ref, near, cut, refused", [
    ("L'Action pyrénéenne", "Pyrénées", "Py-rénées", True), ("La Petite Gironde", "Petit", "Pe-tit", True),
    ("Le Progrès de la Côte-d'Or", "Prog.", "Pro-g.", False), ("La Petite Gironde", "Le Petit", "LePetit", True),
    ("Le Petit Troyen", "Troyes", "Tro yes", True), ("L'Écho de Paris", "Parisien", "Pa-ri-sien", False),
    ("Le Petit Marseillais", "Seillais", "Seil-lais", True), ("La Petite Gironde", "Ronde", "Ron-de", True),
    ("L'Écho de Paris", "L'Écho", "L'É-cho", True), ("L'Écho de Paris", "Paris-Soir", "Pa-ris-So ir", False)])
def test_near_forms_and_cuts_compose(tmp_path, ref, near, cut, refused):
    """pQ NO-GO B1 (25/10), kept in NOMS v2: a name cut by punctuation, a space or glued has the outcome of its uncut form."""
    reg = registry(complete=False)
    docs = [dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)]
    for i, other in enumerate((near, cut)):
        td = by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye") - 1), (other, 1)], documents=docs)
        r = name_codes(tmp_path / str(i), td, reg)
        assert r["refusal_codes"] == (["INCOMPLETE_NUMBER_KEY:newseye"] if refused else []), other


def test_every_cut_of_every_near_form_of_every_distinctive_ref_word_meets():
    """pQ B1, in general, NOMS v2: for each DISTINCTIVE word of at least 4 letters of a REF title, its near forms (first 4 letters, first 5 plus
    another end, plural, glued after an article, every piece of 5 letters or more), cut at every place by a hyphen, an apostrophe, a point or a
    space, are refused for a page of that REF title."""
    reg = registry(complete=False)
    entry = reg["collections"]["newseye"]
    for ref, words in g0_suivi.REF_WORDS.items():
        docs = [dict(REF_DOC, ref_title=ref)]
        for r in (w for w in words - g0_suivi.GENERIC if len(w) >= 4):
            inside = {r[i:j] for i in range(len(r)) for j in range(i + 5, len(r) + 1)}   # every piece of 5 letters or more (Sonnet v3)
            for near in {r[:4], r[:5] + "xyz", r + "s", "le" + r} | inside:
                for k in range(1, len(near)):
                    for sep in "-'. ":
                        name = near[:k] + sep + near[k:]
                        td = by_names(reg, "newseye", [("Ce soir", total_of(reg, "newseye") - 1), (name, 1)], documents=docs)
                        assert g0_suivi._name_disjoint(entry, td, "newseye", [NEW]) is False, (ref, name)


# AV-G0-SUIVI-TITRES-GROUPES (pA, 25/10, bnf_ocr_corrige_presse): a GROUP {names: [...], units: k} says that each of the k units is of one of
# these names; it counts k units and every one of its names is compared with the REF title of the page
def with_group(reg, group, units, singles=(("Ce soir", None),), ref="Le Petit Troyen"):
    t = total_of(reg, "newseye")
    rows = [{"name": n, "units": u if u is not None else t - units} for n, u in singles] + [{"names": list(group), "units": units}]
    td = disjunction(reg, documents=[dict(REF_DOC, ref_title=ref, title_cb=PAGE_CB)])
    td["collections"]["newseye"] = {"names": rows, "method": "workbook", "proof": PROOF}
    return td


def test_a_group_of_names_counts_its_units_and_lifts_when_no_name_meets(tmp_path):
    reg = registry(complete=False)
    td = with_group(reg, ["BO", "L'industriel alsacien", "La Charente", "Rouge-Midi"], 5)
    r = name_codes(tmp_path, td, reg)
    assert r["allowed"], r["refusal_codes"]
    assert "TITLE_DISJOINT_NAME:newseye" in r["labels"]


@pytest.mark.parametrize("ref, member", [("Le Petit Troyen", "Le Petit"), ("Le Petit Troyen", "Pe-tit"), ("La Petite Gironde", "Ron-de"),
                                         ("L'Écho de Paris", "L'Écho")])
def test_one_name_of_a_group_meeting_the_ref_title_refuses(tmp_path, ref, member):
    reg = registry(complete=False)
    td = with_group(reg, ["BO", "La Charente", member], 5, ref=ref)
    assert name_codes(tmp_path, td, reg)["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]


@pytest.mark.parametrize("delta", [-1, 1])
def test_the_units_of_a_group_count_exactly(tmp_path, delta):
    reg = registry(complete=False)
    td = with_group(reg, ["BO", "La Charente"], 5)
    td["collections"]["newseye"]["names"][-1]["units"] += delta
    assert name_codes(tmp_path, td, reg)["refusal_codes"] == ["INCOMPLETE_NUMBER_KEY:newseye"]


def test_a_group_may_repeat_a_name_given_exactly_elsewhere(tmp_path):
    reg = registry(complete=False)
    td = with_group(reg, ["La Charente", "BO"], 2, singles=(("Ce soir", None), ("La Charente", 1)))
    td["collections"]["newseye"]["names"][0]["units"] -= 1
    assert name_codes(tmp_path, td, reg)["allowed"]


@pytest.mark.parametrize("group", [
    {"names": ["BO"], "units": 1},                                   # one name: write {name, units}
    {"names": ["La Charente", "LA CHARENTE!"], "units": 1},          # the same words twice
    {"names": ["BO", "Рetit"], "units": 1},                     # a lookalike letter
    {"names": ["BO", "La"], "units": 1},                             # no significant word
    {"names": ["BO", 3], "units": 1},
    {"names": "BO, La Charente", "units": 1},
    {"names": ["BO", "La Charente"], "units": 0},
    {"names": ["BO", "La Charente"], "name": "BO", "units": 1},
    {"names": [], "units": 1},
])
def test_a_malformed_group_is_invalid(tmp_path, group):
    td = with_group(registry(complete=False), ["BO", "La Charente"], 5)
    td["collections"]["newseye"]["names"][-1] = group
    with pytest.raises(SuiviInputError, match="title_disjunction"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_a_generic_only_name_cut_anywhere_has_the_outcome_of_its_uncut_form():
    """NOMS v2: « Le <mot générique> » and « L'Écho de Paris », cut at every place by a hyphen, an apostrophe, a point or a space, or glued,
    meet each REF title exactly when the uncut name does (the cuts are glued back before the full-name comparison)."""
    names = [f"Le {w}" for w in g0_suivi.GENERIC] + ["L'Écho de Paris", "Paris-Soir", "Petit Journal"]
    for ref in g0_suivi.REF_TITLES:
        for name in names:
            plain = g0_suivi._folded(name).replace("-", " ").replace("'", " ")
            expected = g0_suivi._names_meet(name, ref)
            glued = plain.replace(" ", "")
            assert g0_suivi._names_meet(glued, ref) is expected, (ref, glued)
            for k in range(1, len(glued)):
                for sep in "-'. ":
                    assert g0_suivi._names_meet(glued[:k] + sep + glued[k:], ref) is expected, (ref, glued[:k] + sep + glued[k:])


def test_two_identical_groups_are_invalid(tmp_path):
    """Sonnet GROUPES m1: a group written twice (the same names, in any order or spelling) is one group."""
    td = with_group(registry(complete=False), ["BO", "La Charente"], 3)
    td["collections"]["newseye"]["names"].append({"names": ["la charente", "BO"], "units": 2})
    with pytest.raises(SuiviInputError, match="duplicate name, group or space"):
        write_inputs(tmp_path, mutate=with_td(td))


def test_too_many_readings_of_a_name_refuse(monkeypatch):
    """Sonnet NOMS v2 M2: past the bound on the readings of a separator-free name, the names meet (refusal), never the reverse."""
    assert g0_suivi._names_meet("Le Soir de Lyon", "Le Petit Troyen") is False
    monkeypatch.setattr(g0_suivi, "SPLITS_MAX", 0)
    assert g0_suivi._generic_splits("lesoir") is None
    assert g0_suivi._names_meet("Le Soir de Lyon", "Le Petit Troyen") is True


@pytest.mark.parametrize("ref", ["Le Petit Marseillais", "La Petite Gironde", "Le Petit Troyen"])
@pytest.mark.parametrize("other", ["Le Petit & Cie", "Le Petit 1900", "Le Petit (supplément)", "Le Petit, édition quotidien 3",
                                   "Le Petit illustré hebdomadaire", "Le Petit X", "Le Pe-tit & C-ie", "LePetit1900", "Le Petit - B"])
def test_form_words_numbers_and_single_letters_do_not_lift_an_ambiguous_name(tmp_path, ref, other):
    """pA (25/10, Sonnet M1): refused as « Le Petit » alone."""
    assert g0_suivi._names_meet(other, ref) is True, other


@pytest.mark.parametrize("other", ["Le Petit Parisien 1900", "Le Petit Parisien illustré", "Le Petit Parisien & Cie", "Paris-Soir 1939"])
def test_form_words_never_turn_a_distinctive_name_into_a_refusal(other):
    for ref in g0_suivi.REF_TITLES:
        assert g0_suivi._names_meet(other, ref) is False, (other, ref)


def test_the_form_words_are_the_closed_list_of_pa():
    assert g0_suivi.FORM == {"illustre", "supplement", "edition", "quotidien", "hebdomadaire", "cie", "suppl", "hebdo", "co", "editions",
                             "illustree", "illustres", "illustrees"}
    assert not g0_suivi.FORM & (g0_suivi.GENERIC | set().union(*g0_suivi.REF_WORDS.values()))


def test_the_form_words_are_exact_words_a_declared_limit():
    """Like the generic words, the form words are exact: « quotidienne », « illustrée », « suppléments » are not form words, so
    « Le Petit, édition quotidienne » is no longer a name made only of generic words (declared limit, consistent with the cuts)."""
    assert g0_suivi._names_meet("Le Petit, édition quotidien", "Le Petit Troyen") is True
    assert g0_suivi._names_meet("Le Petit, édition quotidienne", "Le Petit Troyen") is False


def test_a_letter_is_left_out_only_where_the_name_has_a_one_letter_word():
    """« Le Petit X » is « Le Petit », but the first letter of a real word is never skipped: « Le Petit Xpetit » holds a distinctive word."""
    assert g0_suivi._names_meet("Le Petit X", "Le Petit Troyen") is True
    assert g0_suivi._names_meet("Le Petit Xpetit", "Le Petit Troyen") is False


@pytest.mark.parametrize("ref", ["Le Petit Marseillais", "La Petite Gironde", "Le Petit Troyen"])
@pytest.mark.parametrize("other", ["Le Petit 2e", "Le Petit 1er", "Petit 1ère", "Le Petit 3eme", "Le Petit 3ème édition", "Le Petit Suppl.",
                                   "Le Petit Hebdo", "Le Petit & Co", "Le Petit Co", "Le Petit Éditions", "Le Petit Illustrée",
                                   "Les Petits Illustrés", "Le Petit, illustrées", "Le Petit 2-e"])
def test_ordinals_and_the_added_form_words_do_not_lift_an_ambiguous_name(ref, other):
    """pA (25/10, after Sonnet NOMS v2 m1-m2): ordinals are numbers; suppl, hebdo, co, éditions, illustrée(s), illustrés join the form words."""
    assert g0_suivi._names_meet(other, ref) is True, other   # « Les Petits Illustrés » too, since petits joined the generic words


def test_an_ordinal_ending_needs_its_digits():
    assert g0_suivi._names_meet("Le Petit Ere", "Le Petit Troyen") is False   # « ere » alone is a word, not an ordinal


# the names of the tables of pB existing on 25/10 (names_as_anno_183, names_bnf_ocr_corrige_presse, names_e12_test), copied: they are not tracked
PB_TABLE_NAMES = ["BO", "Ce soir", "JDPL", "L'Aurore", "L'Humanite", "L'Oeuvre", "L'action francaise", "L'industriel alsacien", "La Charente",
                  "La Croix", "La Presse", "Le Courrier colonial", "Le Figaro", "Le Gaulois", "Le Matin", "Le Petit parisien", "Le Reveil",
                  "Le Siecle", "Le Temps", "Marianne", "Ouest-Eclair", "Paris-Soir", "Quarante-quatre", "Regards", "Rouge-Midi"]


def test_the_added_plurals_only_tighten(monkeypatch):
    """pA (25/10): petits, petites, journaux, échos, gazettes, courriers, dépêches joined the generic words. A completion of the lists may only
    add refusals: over the names of the existing tables of pB and the names of these tests, no pair (name, REF title) refused before is admitted
    now. Before = the generic list without the plurals."""
    names = {"Les Petits Illustrés", "Les Échos", "Les Échos de Paris", "Journaux du soir", "Gazettes", "Les Dépêches", "Petites Nouvelles",
             "Le Petit Parisien", "Paris-Soir", "L'Action française", "Le Petit", "Courriers de la Gironde", "Echos Marseillais"}
    names.update(PB_TABLE_NAMES)
    now = {(n, r) for n in names for r in g0_suivi.REF_TITLES if g0_suivi._names_meet(n, r)}
    assert not {(n, r) for n, r in now if n in PB_TABLE_NAMES}   # the existing tables admit the 6 REF titles, before and after
    plurals = frozenset().union(*(g0_suivi._name_words(w) for w in ("petits", "petites", "journaux", "échos", "gazettes", "courriers", "dépêches")))
    assert not plurals & set().union(*g0_suivi.REF_WORDS.values())   # none is a word of a REF title, so none can remove a distinctive refusal
    monkeypatch.setattr(g0_suivi, "GENERIC", g0_suivi.GENERIC - plurals)
    monkeypatch.setattr(g0_suivi, "REF_FORMS", g0_suivi.REF_FORMS)
    before = {(n, r) for n in names for r in g0_suivi.REF_TITLES if g0_suivi._names_meet(n, r)}
    assert before <= now
    assert ("Les Petits Illustrés", "Le Petit Troyen") in now - before


# AV-G0-SUIVI-CONTENU (pA, 25/10): disjunction by CONTENT, label CONTENT_DISJOINT:<c> — no shared 8-gram between the clean VT of the page
# and ALL the pinned texts of the collection
CT_FILES = [{"path": {"root": "axel_root", "path": f"data/c{i}.jsonl"}, "sha256": f"{i}" * 64} for i in range(1, 4)]
VT = {"kind": "vt_bbvlm", "sha256": "e" * 64}


def content(reg, n, units=None, files=None, check=None, page=NEW, against=None, findings=(), status="complete", page_text=VT, drop_td=True):
    """Inputs where every incomplete collection is lifted by title records except `n`, given by its pinned texts and a content check."""
    files = [dict(f) for f in (CT_FILES if files is None else files)]   # never mutate the shared fixture (Sonnet m1)
    spec = reg["collections"][n]["match_keys"]["issue"]
    total = spec["coverage"]["total"] if isinstance(spec, dict) and "coverage" in spec else 7
    td = disjunction(reg, drop=(n,) if drop_td else ())
    chk = {"page": page, "check": g0_suivi.CONTENT_CHECK, "status": status, "page_text": page_text,
           "against": list(against or [n]), "against_manifests": {a: g0_suivi.lm_fingerprint(files) for a in (against or [n])},
           "findings": [{"kind": "content_overlap", "source": s} for s in findings]}
    if check is not None:
        check(chk)

    def m(docs):
        docs["sources"]["title_disjunction"] = td
        docs["sources"]["content_texts"] = {n: {"files": files, "units": total if units is None else units, "proof": PROOF}}
    return [chk], m


def test_no_shared_8gram_with_all_the_texts_lifts_with_its_own_label(tmp_path):
    reg = registry(complete=False)
    checks, m = content(reg, "newseye")
    r = codes(NEW, write_inputs(tmp_path, checks=checks, mutate=m), reg)
    assert r["allowed"], r["refusal_codes"]
    assert "CONTENT_DISJOINT:newseye" in r["labels"] and "TITLE_DISJOINT:newseye" not in r["labels"]


@pytest.mark.parametrize("case", ["overlap", "incomplete", "no_check", "other_texts", "not_against", "units_short", "units_long",
                                  "other_page", "no_texts"])
def test_the_content_disjunction_keeps_the_refusal_when_a_condition_fails(tmp_path, case):
    reg = registry(complete=False)
    kw = {"overlap": {"findings": ["newseye"]}, "incomplete": {"status": "incomplete"},
          "other_texts": {"check": lambda c: c["against_manifests"].update(newseye="f" * 64)},
          "units_short": {"units": total_of(reg, "newseye") - 1}, "units_long": {"units": total_of(reg, "newseye") + 1},
          "other_page": {"page": "gallica:7636525/f2"}}.get(case, {})
    checks, m = content(reg, "newseye", **kw)
    if case == "no_check":
        checks = []
    if case == "not_against":
        checks, m = content(reg, "newseye", against=["gallicorpora"])
        m0 = m

        def m(docs):
            m0(docs)
            docs["sources"]["content_texts"]["gallicorpora"] = docs["sources"]["content_texts"]["newseye"]
    if case == "no_texts":
        m0 = m

        def m(docs):
            m0(docs)
            docs["sources"]["content_texts"] = {"gallicorpora": docs["sources"]["content_texts"].pop("newseye")}
        checks = []
    r = codes(NEW, write_inputs(tmp_path, checks=checks, mutate=m), reg)
    assert "INCOMPLETE_NUMBER_KEY:newseye" in r["refusal_codes"], (case, r["refusal_codes"])
    assert "CONTENT_DISJOINT:newseye" not in r["labels"]


def test_every_gallica_member_needs_its_content_check(tmp_path):
    reg = registry(complete=False)
    alias = {"left": {"namespace": "gallica", "value": NEW}, "right": {"namespace": "gallica", "value": F1}, "proven_page": True, "source": "c"}
    checks, m = content(reg, "newseye")
    td_docs = [{"document": NEW_DOC, "title_cb": PAGE_CB, "proof": PROOF}, {"document": DOC, "title_cb": "cb22222222d", "proof": PROOF}]

    def m2(docs):
        m(docs)
        docs["sources"]["title_disjunction"]["documents"] = td_docs
    r = codes(NEW, write_inputs(tmp_path / "a", aliases=[alias], checks=checks, mutate=m2), reg)
    assert "INCOMPLETE_NUMBER_KEY:newseye" in r["refusal_codes"]
    both = checks + [dict(checks[0], page=F1)]
    assert codes(NEW, write_inputs(tmp_path / "b", aliases=[alias], checks=both, mutate=m2), reg)["allowed"]


def test_a_collection_without_number_key_is_lifted_on_its_declared_units(tmp_path):
    reg = registry(complete=False)
    checks, m = content(reg, "gallica_pseudo", units=5)
    assert "CONTENT_DISJOINT:gallica_pseudo" in codes(NEW, write_inputs(tmp_path, checks=checks, mutate=m), reg)["labels"]


def test_title_records_and_names_come_first(tmp_path):
    reg = registry(complete=False)
    checks, m = content(reg, "newseye", drop_td=False)
    r = codes(NEW, write_inputs(tmp_path, checks=checks, mutate=m), reg)
    assert "TITLE_DISJOINT:newseye" in r["labels"] and "CONTENT_DISJOINT:newseye" not in r["labels"]


def test_the_content_disjunction_lifts_nothing_else(tmp_path):
    reg = registry(complete=False)
    checks, m = content(reg, "newseye")
    r = codes(NEW, write_inputs(tmp_path, checks=checks, usages=[("page", NEW, "train", "press/train", "consumed")], mutate=m), reg)
    assert r["refusal_codes"] == ["SUIVI_TRAINED:press/train"]


@pytest.mark.parametrize("edit", [
    lambda c, d: c.update(page_text={"kind": "ocr", "sha256": "e" * 64}),          # never an OCR
    lambda c, d: c.pop("page_text"),
    lambda c, d: c.update(page_text={"kind": "vt_bbvlm", "sha256": "E" * 64}),
    lambda c, d: c.update(against_manifests={}),
    lambda c, d: c.update(findings=[{"kind": "lm_overlap", "source": "newseye"}]),
    lambda c, d: c.update(findings=[{"kind": "content_overlap", "source": "gallicorpora"}]),   # a finding outside its against
    lambda c, d: c.update(against=["gallicorpora"], against_manifests={"gallicorpora": "a" * 64}),   # no texts for it
    lambda c, d: d["sources"]["content_texts"]["newseye"].update(units=0),
    lambda c, d: d["sources"]["content_texts"]["newseye"].update(files=[]),
    lambda c, d: d["sources"]["content_texts"]["newseye"]["files"].append(dict(CT_FILES[0])),   # the same file twice
    lambda c, d: d["sources"]["content_texts"]["newseye"].pop("proof"),
    lambda c, d: d["sources"].update(content_texts={}),
])
def test_malformed_content_inputs_are_invalid(tmp_path, edit):
    reg = registry(complete=False)
    checks, m = content(reg, "newseye")

    def m2(docs):
        m(docs)
        edit(checks[0], docs)
    with pytest.raises(SuiviInputError, match="SUIVI_INPUT_INVALID"):
        write_inputs(tmp_path, checks=checks, mutate=m2)


def test_page_text_and_content_overlap_belong_to_the_content_check(tmp_path):
    chk = {"page": NEW, "check": "ngram8", "status": "complete", "findings": [{"kind": "content_overlap", "source": "x"}]}
    with pytest.raises(SuiviInputError, match="belong to the ngram8_content check only"):
        write_inputs(tmp_path, checks=[chk])


def test_the_content_disjunction_needs_a_gallica_member():
    reg = registry(complete=False)
    src = {"content_texts": {"newseye": {"files": CT_FILES, "units": total_of(reg, "newseye"), "proof": PROOF}}}
    assert g0_suivi._content_disjoint(reg["collections"]["newseye"], src, "newseye", ["kb:00531099"], {}) is False


@pytest.mark.parametrize("lifted_by", ["title_records", "content"])
def test_a_measured_content_overlap_always_refuses(tmp_path, lifted_by):
    """Sonnet CONTENU M1: a shared 8-gram measured with the texts of C refuses even when C is lifted by its title records."""
    reg = registry(complete=False)
    checks, m = content(reg, "newseye", findings=["newseye"], drop_td=lifted_by == "content")
    r = codes(NEW, write_inputs(tmp_path, checks=checks, mutate=m), reg)
    assert "SUIVI_CONTENT_OVERLAP:newseye" in r["refusal_codes"], r["refusal_codes"]
    assert not r["allowed"] and not any(x.startswith("THIRD_PARTY_OVERLAP:newseye") for x in r["labels"])
