"""Règle (d) de pA (24/10) : une source `consumed` SANS aucune ligne d'usage est interdite. Contrôle sur les brouillons réels."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "reports/g0_suivi_pB"


def test_every_consumed_source_has_at_least_one_usage_line():
    sources = json.loads((DRAFTS / "sources_full.DRAFT.json").read_text(encoding="utf-8"))["sources"]
    usages = json.loads((DRAFTS / "usage_full.DRAFT.json").read_text(encoding="utf-8"))["usages"]
    used = {u["source"] for u in usages}
    orphans = [s["source"] for s in sources if s["status"] == "consumed" and s["source"] not in used]
    assert orphans == [], f"sources consumed sans aucune ligne d'usage : {orphans}"


def test_the_reader_lm_row_is_unknown_with_its_filtered_texts_on_disk():
    """Décision pA : reader_lm_corpus = unknown avec lm_texts (press/train + 109 numéros Gallica), all_on_disk vrai une fois vérifiés."""
    sources = {s["source"]: s for s in json.loads((DRAFTS / "sources_full.DRAFT.json").read_text(encoding="utf-8"))["sources"]}
    row = sources["reader_lm_corpus"]
    assert row["status"] == "non_resolue" and row["key_spaces"] == ["unknown"]
    assert row["lm_texts"]["all_on_disk"] is True           # décision pA (24/10) : text.jsonl.gz filtré + press/train, 3 SHA vérifiés
    assert len(row["lm_texts"]["files"]) == 3


import sys  # noqa: E402

import pytest  # noqa: E402

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from test_g0_suivi import codes, write_inputs  # noqa: E402


def test_a_fresh_gallica_page_is_refused_exactly_without_ngram8(tmp_path):
    # (a) sans contrôle ngram8 : refus EXACT, donc aucune page ne passe sans ngram8 (p1, 25/10)
    sources = json.loads((DRAFTS / "sources_full.DRAFT.json").read_text(encoding="utf-8"))["sources"]
    r = codes("gallica:7799999/f1", write_inputs(tmp_path, sources=sources))
    assert not r["allowed"]
    assert set(r["refusal_codes"]) == {"SUIVI_CHECK_INCOMPLETE:ngram8:pleias_french_pd_322",
                                       "SUIVI_CHECK_INCOMPLETE:ngram8:reader_lm_corpus"}, r["refusal_codes"]


def test_a_fresh_press_page_stays_admitted_on_the_real_table(tmp_path):
    # pA (25/10) : les 6 livres non canonisables ne bloquent pas la presse (gallica_book fermé)
    import json as _json
    from pathlib import Path as _P
    src = _json.loads(_P("reports/g0_suivi_pB/sources_full.DRAFT.json").read_text(encoding="utf-8"))["sources"]
    assert all(s["status"] != "non_resolue" or s["source"] in ("reader_lm_corpus",) or s["key_spaces"] for s in src)
    from tests.test_g0_suivi import codes, write_inputs
    import importlib.util
    spec = importlib.util.spec_from_file_location("c", "scripts/g0_lm_checks.py")
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    import hashlib
    f = lambda p: hashlib.sha256(_P(p).read_bytes()).hexdigest()
    comp = [{"path": "french-pd-sample/text.jsonl.gz", "sha256": f("/Users/marcel/corpus-vt/french-pd-sample/text.jsonl.gz")},
            {"path": "french-pd-sample/manifest.json", "sha256": f("/Users/marcel/corpus-vt/french-pd-sample/manifest.json")},
            {"path": "g0_b1_lecteur/R2_LM_documents_gallica.json", "sha256": f("/Users/marcel/Documents/reviews/babaorum-pipeline/g0_b1_lecteur/R2_LM_documents_gallica.json")},
            {"path": "data/prepared/press/train.jsonl", "sha256": f("/Users/marcel/axel/data/prepared/press/train.jsonl")}]
    K = "gallica:7799999/f1"
    checks = C.merge_ngram8(C.convert({K: {"index_textes": 322, "mots": 50, "verdict": "PASSE", "verdicts": [], "par_source": {}, "comparés": comp}}, {"sources": src}))
    r = codes(K, write_inputs(tmp_path, sources=src, checks=checks))
    assert r["allowed"], sorted(set(r["refusal_codes"]))


def test_a_fresh_gallica_page_is_allowed_with_a_complete_ngram8_per_row(tmp_path):
    # (b) ngram8 complet : against = les deux lignes LM, empreintes = lm_fingerprint des fichiers de CHAQUE ligne
    import hashlib
    import importlib.util
    from babaorum import g0_suivi
    sources = json.loads((DRAFTS / "sources_full.DRAFT.json").read_text(encoding="utf-8"))["sources"]
    rows = {x["source"]: x for x in sources}
    spec = importlib.util.spec_from_file_location("c", ROOT / "scripts/g0_lm_checks.py")
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    comp = [{"path": "french-pd-sample/text.jsonl.gz", "sha256": sha("/Users/marcel/corpus-vt/french-pd-sample/text.jsonl.gz")},
            {"path": "french-pd-sample/manifest.json", "sha256": sha("/Users/marcel/corpus-vt/french-pd-sample/manifest.json")},
            {"path": "g0_b1_lecteur/R2_LM_documents_gallica.json", "sha256": sha("/Users/marcel/Documents/reviews/babaorum-pipeline/g0_b1_lecteur/R2_LM_documents_gallica.json")},
            {"path": "data/prepared/press/train.jsonl", "sha256": sha("/Users/marcel/axel/data/prepared/press/train.jsonl")}]
    K = "gallica:7799999/f1"
    checks = C.merge_ngram8(C.convert({K: {"index_textes": 322, "mots": 50, "verdict": "PASSE", "verdicts": [], "par_source": {}, "comparés": comp}}, {"sources": sources}))
    assert checks[0]["against"] == ["pleias_french_pd_322", "reader_lm_corpus"]
    for n in checks[0]["against"]:
        assert checks[0]["against_manifests"][n] == g0_suivi.lm_fingerprint(rows[n]["lm_texts"]["files"])
    r = codes(K, write_inputs(tmp_path, sources=sources, checks=checks))
    assert r["allowed"], sorted(set(r["refusal_codes"]))
