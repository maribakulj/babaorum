"""Contrôle 8-grammes LM (contrat G0-04 v4.1 point 4) sur textes SYNTHÉTIQUES. Aucune page REF-GALLICA (elles n'existent pas)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("g0_lm_ngram8", ROOT / "scripts/g0_lm_ngram8.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

TRAIN = ("le conseil de la ville a décidé que les anciens règlements seraient abrogés et que chaque habitant "
         "pourrait désormais présenter une requête écrite au bureau central avant la fin du mois de juin")
NEUF = ("la compagnie des chemins de fer du nord annonce une baisse sensible des tarifs de transport pour les "
        "marchandises pondéreuses durant la saison d'hiver et au printemps prochain sur tout le réseau")
PLEIAS = "pleias_french_pd_322"


def cat_count(res, cat):
    """Plus grand nombre de grammes partagés d'UNE source, dans la catégorie (refus | eval)."""
    vals = [v["grammes_partages"] for k, v in res["par_source"].items() if k.endswith("|" + cat)]
    return max(vals, default=0)


def cat_part(res, cat):
    vals = [v["part"] for k, v in res["par_source"].items() if k.endswith("|" + cat) and v["part"] is not None]
    if not vals:
        return None
    return max(vals)


def tx(file_id, text, split="train", source=PLEIAS):
    return {"file_id": file_id, "source": source, "split": split, "text": text}


def index(split="train", source=PLEIAS):
    return M.build_index([tx("bpt6k_1", TRAIN, split, source)])


def long_page(words=60):
    base = (TRAIN + " " + NEUF).split()
    return [" ".join(base[i:i + 12]) for i in range(0, min(words, len(base)), 12)]


def test_a_verbatim_copy_of_a_training_text_is_refused_with_its_source():
    res = chk("p_copie", long_page(), index())
    assert res["verdict"] == f"SUIVI_TEXT_OVERLAP:{PLEIAS}"
    assert res["par_source"][f"{PLEIAS}|refus"]["textes"] == ["bpt6k_1"] and cat_count(res, "refus") >= 3


def test_a_new_text_passes():
    res = chk("p_neuve", [NEUF], index())
    assert res["verdict"] == "PASSE" and (cat_count(res, "refus") + cat_count(res, "eval")) == 0


def test_an_eval_match_is_labelled_not_refused():
    res = chk("p_eval", [TRAIN], index(split="eval"))
    assert res["verdict"] == f"TEXT_OVERLAP_EVAL:{PLEIAS}" and res["verdicts"] == [f"TEXT_OVERLAP_EVAL:{PLEIAS}"]


def test_the_refusal_names_the_source_of_each_text():
    idx = M.build_index([tx("lm_a", TRAIN, "train", "reader_lm_corpus")])
    assert chk("p", [TRAIN], idx)["verdict"] == "SUIVI_TEXT_OVERLAP:reader_lm_corpus"


def test_a_text_without_split_or_source_is_refused_at_build():
    with pytest.raises(ValueError):
        M.build_index([{"file_id": "x", "source": PLEIAS, "split": "prod", "text": TRAIN}])
    with pytest.raises(ValueError):
        M.build_index([{"file_id": "x", "split": "train", "text": TRAIN}])


def test_a_short_page_is_incomplete_whatever_its_overlap():
    five = TRAIN.split()[:12]
    assert M.check_page("p_courte", [" ".join(five)], index())["verdict"] == "incomplete"
    one = " ".join(TRAIN.split()[:8]) + " " + " ".join(NEUF.split()[:4])
    assert M.check_page("p_courte_1", [one], index())["verdict"] == "incomplete"


def test_the_share_rule_refuses_a_long_page_with_two_shared_grams_when_the_share_reaches_half_a_percent():
    words = (NEUF + " " + " ".join(TRAIN.split()[:9])).split()        # 9 mots de TRAIN : 2 8-grammes partagés
    res = chk("p_longue_2", [" ".join(words * 3)], index())
    assert res["mots"] >= 50 and (cat_count(res, "refus") + cat_count(res, "eval")) == 2
    assert cat_part(res, "refus") >= M.SHARE_MIN and res["verdict"] == f"SUIVI_TEXT_OVERLAP:{PLEIAS}"


def test_one_shared_gram_in_a_long_new_page_passes_under_the_share_and_the_floor():
    body = " ".join(f"zq{i}" for i in range(2000))                    # 2000 mots distincts : ~2000 8-grammes distincts
    res = chk("p_longue_1", [body + " " + " ".join(TRAIN.split()[:8])], index())
    assert (cat_count(res, "refus") + cat_count(res, "eval")) == 1 and cat_part(res, "refus") < M.SHARE_MIN and res["verdict"] == "PASSE"


def test_the_fifty_word_boundary_is_applied_to_the_normalised_text():
    fifty = (TRAIN + " " + NEUF).split()[:50]
    assert cat_part(chk("p50", [" ".join(fifty)], index()), "refus") is not None
    forty_nine = fifty[:49]
    assert M.check_page("p49", [" ".join(forty_nine)], index())["verdict"] == "incomplete"


def test_a_hyphen_at_line_end_is_joined_to_the_next_line():
    assert M.page_text(["reconnais-", "sance des textes"]) == M.norm("reconnaissance des textes")
    assert M.page_text(["reconnais⸗", "sance"]) == M.norm("reconnaissance")
    assert M.page_text(["reconnais¬", "sance"]) == M.norm("reconnaissance")
    assert M.page_text(["le texte ¬", "suite"]) == M.norm("le texte suite")


def test_the_grams_are_distinct_and_the_denominator_is_the_page_s_distinct_count():
    text = " ".join([TRAIN] * 3)
    res = chk("p_rep", [text], index())
    assert res["grammes_distincts"] == len(M.grams(M.norm(text)))


def test_the_output_folder_must_be_new_and_the_json_has_the_verdict(tmp_path):
    texts = tmp_path / "t.jsonl"; texts.write_text(json.dumps(tx("bpt6k_1", TRAIN)) + "\n", encoding="utf-8")
    pages = tmp_path / "p.jsonl"; pages.write_text(json.dumps({"id": "p", "lines": [TRAIN, PAD(60)]}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert M.main(["--texts", str(texts), "--pages", str(pages), "--out", str(out)]) == 0
    written = json.loads((out / "p.json").read_text(encoding="utf-8"))
    assert written["verdict"] == f"SUIVI_TEXT_OVERLAP:{PLEIAS}"
    assert M.main(["--texts", str(texts), "--pages", str(pages), "--out", str(out)]) == 1


def test_two_ids_with_the_same_sanitised_name_are_refused(tmp_path):
    texts = tmp_path / "t.jsonl"; texts.write_text(json.dumps(tx("bpt6k_1", TRAIN)) + "\n", encoding="utf-8")
    pages = tmp_path / "p.jsonl"
    pages.write_text(json.dumps({"id": "a b", "lines": [NEUF]}) + "\n" + json.dumps({"id": "a_b", "lines": [NEUF]}) + "\n", encoding="utf-8")
    assert M.main(["--texts", str(texts), "--pages", str(pages), "--out", str(tmp_path / "out")]) == 1


def test_norm_is_the_norm_of_e6_when_axel_is_available():
    axel = Path.home() / "axel" / "scripts"
    if not (axel / "e6_contamination.py").exists():
        pytest.skip("axel absent : égalité de norm non vérifiable ici")
    sys.path.insert(0, str(axel))
    e6 = importlib.import_module("e6_contamination")
    for s in ["Ceci est un Test ſ", "Éléphants, œufs ; (fin)", "a_b  c\tD"]:
        assert M.norm(s) == e6.norm(s)


def test_an_empty_index_is_refused_not_passed():
    with pytest.raises(ValueError):
        M.build_index([])


def test_lines_given_as_a_string_are_refused_not_copied():
    with pytest.raises(ValueError):
        M.page_text(TRAIN)


def test_an_empty_page_is_refused():
    with pytest.raises(ValueError):
        M.page_text([])


def test_the_thresholds_are_applied_per_split_group():
    eval_src = " ".join(f"ev{i} " + TRAIN for i in range(1))            # placeholder, remplacé ci-dessous
    one_train = TRAIN.split()[:8]
    idx = M.build_index([tx("tr_1", " ".join(one_train) + " mots de fin differents", "train", PLEIAS)] +
                        [tx(f"ev_{i}", TRAIN if i < 33 else NEUF, "eval", "eval_src") for i in range(33)])
    res = M.check_page("p_mixte", [TRAIN + " " + PAD(600)], idx)
    assert cat_count(res, "refus") <= 2 and res["verdicts"] == ["TEXT_OVERLAP_EVAL:eval_src"]
    assert not any(v.startswith("SUIVI_TEXT_OVERLAP") for v in res["verdicts"])


def test_a_unicode_line_separator_inside_a_text_does_not_split_the_record(tmp_path):
    texte = TRAIN + "\u2028" + "suite"
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps(tx("bpt6k_u", texte)) + "\n", encoding="utf-8")
    recs = M.read_jsonl(f)
    assert len(recs) == 1 and recs[0]["text"] == texte


def test_an_empty_text_file_is_refused_by_main(tmp_path):
    texts = tmp_path / "t.jsonl"; texts.write_text("", encoding="utf-8")
    pages = tmp_path / "p.jsonl"; pages.write_text(json.dumps({"id": "p", "lines": [TRAIN, PAD(60)]}) + "\n", encoding="utf-8")
    assert M.main(["--texts", str(texts), "--pages", str(pages), "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()


def _mini_corpus(tmp_path, texts):
    import gzip, hashlib
    d = tmp_path / "corpus" / "french-pd-sample"; d.mkdir(parents=True)
    ids = [f"bpt6k{i}" for i in range(len(texts))]
    with gzip.open(d / "text.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, t in zip(ids, texts):
            f.write(json.dumps({"file_id": i, "text": t}) + "\n")
    man = {"kept_file_ids": ids}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    lm = {"lm_texts": {"count_attendu": len(ids), "root": {"root": "corpus_root", "path": "french-pd-sample/text.jsonl.gz"},
          "manifest": {"path": "french-pd-sample/manifest.json", "sha256": sha(d / "manifest.json")},
          "sha256_fichier": sha(d / "text.jsonl.gz")},
          "per_text": [{"file_id": i, "sha256_texte": hashlib.sha256(t.encode("utf-8")).hexdigest()} for i, t in zip(ids, texts)]}
    lmj = tmp_path / "lm_texts.json"; lmj.write_text(json.dumps(lm), encoding="utf-8")
    return tmp_path / "corpus", lmj


def test_the_lm_index_is_built_only_when_every_hash_and_count_matches(tmp_path):
    corpus, lmj = _mini_corpus(tmp_path, [TRAIN, NEUF])
    entries = M.load_lm_texts(corpus, lmj)
    assert len(entries) == 2 and all(e["split"] == "train" and e["source"] == "pleias_french_pd_322" for e in entries)


def test_an_altered_text_or_a_wrong_count_is_refused(tmp_path):
    corpus, lmj = _mini_corpus(tmp_path, [TRAIN, NEUF])
    lm = json.loads(lmj.read_text(encoding="utf-8"))
    lm["per_text"][0]["sha256_texte"] = "0" * 64
    lmj.write_text(json.dumps(lm), encoding="utf-8")
    with pytest.raises(ValueError):
        M.load_lm_texts(corpus, lmj)
    corpus2, lmj2 = _mini_corpus(tmp_path / "b", [TRAIN, NEUF])
    lm2 = json.loads(lmj2.read_text(encoding="utf-8")); lm2["lm_texts"]["count_attendu"] = 322
    lmj2.write_text(json.dumps(lm2), encoding="utf-8")
    with pytest.raises(ValueError):
        M.load_lm_texts(corpus2, lmj2)


def test_a_one_gram_train_source_and_a_two_gram_train_source_are_not_summed():
    """Repro pQ : 1 gramme de train_a + 2 grammes de train_b ne doivent refuser AUCUNE des deux (aucune n'atteint 3)."""
    a_gram = " ".join(TRAIN.split()[:8])
    b_grams = " ".join(NEUF.split()[:9])
    idx = M.build_index([tx("ta", a_gram, "train", "train_a"), tx("tb", b_grams, "train", "train_b")])
    page = a_gram + " " + "mots intermédiaires uniques ici " + b_grams
    res = M.check_page("p_courte_deux", [page + " " + PAD(600)], idx)
    assert not any(v.startswith("SUIVI_TEXT_OVERLAP") for v in res["verdicts"]), res["verdicts"]
    assert all(v["grammes_partages"] < 3 for v in res["par_source"].values())


def test_the_reader_texts_are_exactly_the_filtered_records_plus_press_pages(tmp_path):
    import gzip
    ids = ["bpt6k_a", "bpt6k_b"]
    gz = tmp_path / "text.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for i, t in [("bpt6k_a", TRAIN), ("bpt6k_b", NEUF), ("bpt6k_HORS", "hors liste")]:
            f.write(json.dumps({"file_id": i, "text": t}) + "\n")
    flt = tmp_path / "filtre.json"; flt.write_text(json.dumps({"inclus": ids + ["x"] * 107}), encoding="utf-8")
    flt.write_text(json.dumps({"inclus": [f"bpt6k_{k}" for k in range(107)] + ids}), encoding="utf-8")
    press = tmp_path / "press.jsonl"
    press.write_text("\n".join(json.dumps(r) for r in [{"page": "P1", "truth": "une ligne"}, {"page": "P1", "truth": "suite"}]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        M.load_reader_entries(gz, flt, press)          # 109 identifiants exigés : ici 109 mais les autres absents
    ids = [f"bpt6k_{k}" for k in range(107)] + ["bpt6k_a", "bpt6k_b"]
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"file_id": i, "text": TRAIN if i == "bpt6k_a" else NEUF + " " + i}) + "\n")
        f.write(json.dumps({"file_id": "bpt6k_HORS", "text": "hors liste"}) + "\n")
    e = M.load_reader_entries(gz, flt, press)
    names = [x["file_id"] for x in e]
    assert "bpt6k_HORS" not in names and len(names) == 109 + 1
    assert [x for x in e if x["file_id"] == "press:P1"][0]["text"] == "une ligne suite"
    assert all(x["source"] == "reader_lm_corpus" and x["split"] == "train" for x in e)


def test_index_only_writes_the_index_and_a_summary_and_no_page(tmp_path):
    texts = tmp_path / "t.jsonl"
    texts.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"file_id": "a", "source": "reader_lm_corpus", "split": "train", "text": TRAIN},
        {"file_id": "b", "source": "pleias_french_pd_322", "split": "train", "text": NEUF}]) + "\n", encoding="utf-8")
    out = tmp_path / "idx"
    assert M.main(["--texts", str(texts), "--index-only", "--out", str(out)]) == 0
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summ["mode"] == "index-only" and summ["textes"] == 2 and summ["index_grammes"] > 0
    assert summ["par_source"] == {"pleias_french_pd_322": 1, "reader_lm_corpus": 1}
    assert not list(out.glob("*.json")) == [] and [p.name for p in out.glob("*.json")] == ["summary.json"]
    assert (out / "index.jsonl.gz").exists()
    pages = tmp_path / "p.jsonl"; pages.write_text("", encoding="utf-8")
    assert M.main(["--texts", str(texts), "--index-only", "--pages", str(pages), "--out", str(tmp_path / "x")]) == 1


def PAD(n):
    """n mots sans rapport avec le corpus : allonge le dénominateur sans créer de gramme partagé."""
    return " ".join(f"zqx{k}" for k in range(n))


def chk(page_id, lines, idx):
    """Complète une page à 50 mots avec des mots sans rapport (aucun gramme partagé) : seuls les tests anciens l'utilisent,
    pour rester sur la règle des 50 mots (une page plus courte est « incomplete », testée à part)."""
    text = " ".join(lines)
    if len(M.norm(text).split()) < M.MIN_WORDS:
        pad = " ".join(f"zqx{k}" for k in range(M.MIN_WORDS))
        lines = [text, pad]
    return M.check_page(page_id, lines, idx)


def _words(n, prefix="mot"):
    return " ".join(f"{prefix}{i}" for i in range(n))


@pytest.mark.parametrize("n", [0, 7, 49])
def test_a_page_under_fifty_normalised_words_is_incomplete_and_never_passes(n):
    idx = M.build_index([{"file_id": "a", "source": "pleias_french_pd_322", "split": "train", "text": TRAIN}])
    res = M.check_page("gallica:1/f1", [_words(n)] if n else [" "], idx)
    assert res["verdict"] == "incomplete" and res["verdict"] != "PASSE"
    assert res["verdicts"] == ["NGRAM8_INCOMPLETE"] and res["par_source"] == {}


@pytest.mark.parametrize("n", [50, 80])
def test_a_page_with_fifty_or_more_words_is_computed(n):
    idx = M.build_index([{"file_id": "a", "source": "pleias_french_pd_322", "split": "train", "text": TRAIN}])
    res = M.check_page("gallica:1/f1", [_words(n)], idx)
    assert res["verdict"] != "incomplete" and res["mots"] == n

