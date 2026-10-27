"""Convertisseur producteur ngram8 -> checks du registre d'usage (module g0_suivi). Textes et tables SYNTHÉTIQUES."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from babaorum import g0_suivi  # noqa: E402
from babaorum.g0_suivi import lm_fingerprint  # noqa: E402
from test_g0_suivi import codes, write_inputs  # noqa: E402

SPEC = importlib.util.spec_from_file_location("g0_lm_checks", ROOT / "scripts/g0_lm_checks.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

FILES = [{"path": {"root": "corpus_root", "path": "french-pd-sample/text.jsonl.gz"}, "sha256": "a" * 64},
         {"path": {"root": "corpus_root", "path": "french-pd-sample/manifest.json"}, "sha256": "b" * 64}]
COMPARES = [{"path": "french-pd-sample/text.jsonl.gz", "sha256": "a" * 64},
            {"path": "french-pd-sample/manifest.json", "sha256": "b" * 64}]
GAL = "gallica:7636525/f1"
LM = "pleias_french_pd_322"


def table(lm_names=(LM,), extra=()):
    rows = [{"source": n, "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "R2",
             "lm_texts": {"files": FILES, "documents": 322, "all_on_disk": True}} for n in lm_names]
    rows += list(extra)
    return {"sources": rows}


def out(verdicts=(), **extra):
    base = {"index_textes": 322, "verdict": (verdicts[0] if verdicts else "PASSE"), "verdicts": list(verdicts),
            "par_source": {}, "comparés": COMPARES, "mots": 50}   # une page calculée a au moins 50 mots
    base.update(extra)
    return base


def test_an_own_page_computed_against_the_322_texts_gets_one_complete_entry():
    e = C.convert({GAL: out()}, table())
    assert [x["page"] for x in e] == [GAL]
    assert e[0]["status"] == "complete" and e[0]["against"] == [LM] and e[0]["findings"] == []
    assert e[0]["against_manifests"][LM] == lm_fingerprint(FILES)


def test_no_entry_for_a_page_not_computed_against_322_texts_or_without_a_verdict():
    assert C.convert({GAL: out(index_textes=321)}, table()) == []
    assert C.convert({GAL: {k: v for k, v in out().items() if k != "verdict"}}, table()) == []


def test_a_stale_or_missing_fingerprint_of_the_compared_files_gives_no_entry():
    assert C.convert({GAL: out(comparés=[{"path": "x", "sha256": "c" * 64}])}, table()) == []
    assert C.convert({GAL: {k: v for k, v in out().items() if k != "comparés"}}, table()) == []


def test_only_the_computed_page_is_admitted_never_another_member_of_its_class():
    other = "gallica:7636525/f2"
    e = C.convert({GAL: out()}, table())
    assert other not in [x["page"] for x in e]


def test_an_overlap_becomes_lm_overlap_for_each_lm_source():
    e = C.convert({GAL: out(["SUIVI_TEXT_OVERLAP:" + LM])}, table((LM, "reader_lm_corpus")))
    kinds = {(f["kind"], f["source"]) for f in e[0]["findings"]}
    assert kinds == {("lm_overlap", LM), ("lm_overlap", "reader_lm_corpus")}


def test_no_lm_source_on_disk_is_refused():
    with pytest.raises(ValueError):
        C.convert({GAL: out()}, {"sources": [{"source": "x", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"}]})


def test_two_different_fingerprints_for_one_source_are_an_error():
    a = C.convert({GAL: out()}, table())
    b = [dict(a[0], against_manifests={LM: "f" * 64})]
    with pytest.raises(ValueError):
        C.merge_ngram8(a + b)


def test_the_produced_checks_are_accepted_by_the_module_validator():
    e = C.convert({GAL: out(["SUIVI_TEXT_OVERLAP:" + LM])}, table())
    doc = {"schema": g0_suivi.SCHEMAS["usage"], "version": 1, "created": "2026-10-24", "consultations_seq_max": 0,
           "usages": [], "aliases": [], "checks": C.merge_ngram8(e), "limits": {}}
    g0_suivi._validate("usage", doc)


def test_a_fresh_gallica_press_page_whose_ngram8_passes_is_admitted_by_the_module(tmp_path):
    """Chaîne module : une page neuve (clé fictive) dont le ngram8 est PASSE contre pleias_322 est ADMISE quand la seule
    ligne LM est pleias_322 (unknown, lm_texts vérifiés, ngram obligatoire). Aucune ligne reader consumed sans usage (règle d)."""
    cover = {"source": "couverture_gallica_test", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "source test : couvre l'espace gallica"}
    tab = table((LM,), extra=[cover])
    checks = C.convert({GAL: out()}, tab)
    other = "gallica:7600001/f1"                       # une ligne d'usage de la source couvrante, sur une AUTRE page
    r = codes(GAL, write_inputs(tmp_path, checks=checks, sources=tab["sources"], usages=[("page", other, "train", cover["source"], "consumed")]))
    assert r["allowed"], r["refusal_codes"]



def test_an_incomplete_page_gives_no_entry_while_a_computed_one_does():
    # sortie incomplete (moins de 50 mots) : aucune entrée (donc refus) ; la même sortie calculée donne une entrée
    inc = out(["NGRAM8_INCOMPLETE"]); inc["verdict"] = "incomplete"
    assert C.convert({GAL: inc}, table()) == []
    assert [x["page"] for x in C.convert({GAL: out()}, table())] == [GAL]


def test_an_old_forty_nine_word_pass_is_never_converted_into_a_complete_check(tmp_path):
    # repro pQ : un ancien résultat à 49 mots, verdict PASSE, ne doit donner AUCUNE entrée ni admettre la page
    old49 = out(); old49["mots"] = 49
    assert C.convert({GAL: old49}, table()) == []
    for bad in (True, None, "50", 49.9):
        o = out(); o["mots"] = bad
        assert C.convert({GAL: o}, table()) == [], bad
    ok = out(); ok["mots"] = 50
    assert [x["page"] for x in C.convert({GAL: ok}, table())] == [GAL]


def test_the_forty_nine_word_old_pass_does_not_admit_the_gallica_page_in_g0(tmp_path):
    cover = {"source": "couverture_gallica_test", "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "source test : couvre l'espace gallica"}
    tab = table((LM,), extra=[cover])
    old49 = out(); old49["mots"] = 49
    checks = C.convert({GAL: old49}, tab)
    other = "gallica:7600001/f1"
    r = codes(GAL, write_inputs(tmp_path, checks=checks, sources=tab["sources"], usages=[("page", other, "train", cover["source"], "consumed")]))
    assert not r["allowed"]
    assert any(c.startswith("SUIVI_CHECK_INCOMPLETE") for c in r["refusal_codes"]), r["refusal_codes"]
