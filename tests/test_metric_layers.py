from babaorum.reader.metric_layers import Tally, classify, layers


def test_layers_cumulative():
    t = layers("  L’Étoile  du ſoir — œuvre, ETOIT ")
    assert t["L0_brut"] == "L’Étoile du ſoir — œuvre, ETOIT"
    assert t["L1_casse"] == "l’étoile du ſoir — œuvre, etoit"
    assert t["L2_accents"] == "l’etoile du ſoir — œuvre, etoit"
    assert t["L3_typographie"] == "l'etoile du soir - oeuvre, etoit"
    assert t["L4_ponctuation"] == "l etoile du soir oeuvre etoit"
    assert t["L5_orthographe"] == "l etoile du soir oeuvre etait"


def test_classify_kinds():
    assert classify("Paris", "paris") == {"casse": 1}
    assert classify("Etat", "État") == {"accents": 1}
    assert classify("l'eau", "l’eau") == {"typographie": 1}
    assert classify("fin.", "fin") == {"ponctuation": 1}
    assert classify("il etoit", "il était") == {"orthographe": 1}
    assert classify("porte monnaie", "portemonnaie") == {"segmentation": 2}
    assert classify("le chat", "le chien") == {"substitution": 1}
    assert classify("le", "le chat") == {"mot_manquant": 1}
    assert classify("le gros chat", "le chat") == {"mot_invente": 1}
    assert classify("a b", "a b") == {}


def test_tally_rates():
    t = Tally()
    t.add("Le Chat", "le chat")
    t.add("", "*")  # vérité vide après L4 : hors support à L4, compte à L0
    r = t.report()
    assert r["cer_pc"]["L0_brut"] == 100 * 3 / 8  # « * » compte à L0 : 1 erreur sur 1
    assert r["cer_pc"]["L1_casse"] == 100 * 1 / 8  # « Le Chat » rattrapé ; reste « * »
    assert r["cer_support"]["L4_ponctuation"] == 7
    assert r["sac_de_mots"] == {"manquants": 0, "inventes": 0, "mots_verite": 2, "taux_pc": 0.0}


def test_review_cases():
    # relecture indépendante du 17/10 : objections 1 à 5
    assert classify("abc d", "ab cd") == {"segmentation": 2}
    assert classify("peutêtre", "peut-être") == {"segmentation": 1}
    assert classify("mot", "Mot ,") == {"ponctuation": 1, "casse": 1}
    assert classify("mót y", ", Mot x") == {"ponctuation": 1, "accents": 1, "substitution": 1}
    assert classify("Straße", "strasse") == {"typographie": 1}
    assert classify("François", "Français") == {"substitution": 1}
    t = Tally()
    t.add("a b c", "a b c d e f")
    t.add("a b c d e f", "a b c")
    assert t.report()["sac_de_mots"]["taux_pc"] == 100 * 6 / 9
    t2 = Tally()
    t2.add("zz", "")
    t2.add("a", "a")
    r = t2.report()
    assert r["differences"] == {} and r["sac_de_mots"]["taux_pc"] == 0.0
    assert r["verite_vide"]["lignes_verite_vide_L0"] == 1 and r["verite_vide"]["mots_ecrits_L0"] == 1


def test_class_sum_equals_unit_counts():
    # une substitution, un mot manquant, une casse : 3 mots de vérité concernés
    assert sum(classify("Le chien", "le chat noir").values()) == 3


def test_punctuation_words_second_review():
    assert classify("a - b", "a — b") == {"typographie": 1}
    assert classify("a ; b", "a , b") == {"ponctuation": 1}
    assert classify("xx", "—") == {"substitution": 1}
    assert classify("mot -", "mot —") == {"typographie": 1}


def test_class_sum_equals_wer_l0_fuzz():
    import random

    from rapidfuzz.distance import Levenshtein

    rng = random.Random(0)
    vocab = ["le", "Le", "lé", "chat", "Chat", "chât", ",", ";", "—", "-", "'", "peut-être", "peutêtre",
             "etoit", "était", "ab", "cd", "abcd", "x", "œuvre", "oeuvre", "«", "»"]
    bad = 0
    for _ in range(5000):
        t = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
        h = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 6)))
        wer = Levenshtein.distance(layers(h)["L0_brut"].split(), layers(t)["L0_brut"].split())
        n = sum(classify(h, t).values())
        tw, hw = layers(t)["L0_brut"].split(), layers(h)["L0_brut"].split()
        assert n <= len(tw) + len(hw)  # chaque mot compté au plus une fois
        if wer == 0:
            assert n == 0
        bad += n > wer
    assert bad == 0, bad  # repli sur l'alignement brut : jamais plus que le WER L0
