"""Journaux du lecteur (décision pA, 24/10) : pending avec concerns = les 14 sources du lecteur.

Condition 2 : une page neuve est ADMISE, et une page de CHACUNE des 14 sources est REFUSÉE par le journal concerné.
Les entrées sont synthétiques (même écriture que test_g0_suivi.py) : les cinq fichiers réels ne sont pas encore épinglés.
"""
import pytest

from test_g0_suivi import F1, SOURCES, codes, write_inputs

LECTEUR = ["lecteur_A1_E17_train", "lecteur_press_train_alloue", "lecteur_A1_E17_dev", "lecteur_TUNE", "lecteur_porte_2d",
           "lecteur_portes_1b_r1e", "lecteur_second_signal_abst200", "lecteur_R3_tune", "lecteur_decision_P3_ABSTv2_gel",
           "lecteur_T3_geometrie", "lecteur_eval_e8", "lecteur_negatifs_pert200", "lecteur_R3_eval", "lecteur_illisibles_P07b"]
JOURNALS = [{"journal": "docs/READER_LOOP.txt", "status": "pending", "concerns": LECTEUR},
            {"journal": "docs/READER_DOTS_P3_REPORT.txt", "status": "pending", "concerns": LECTEUR}]
# one canonical page per source; a new page is not used by any source
PAGE_OF = {s: f"gallica:7700{i:03d}/f1" for i, s in enumerate(LECTEUR)}
NEW_PAGE = "gallica:7799999/f1"
SOURCES14 = SOURCES + [{"source": s, "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"} for s in LECTEUR]


def test_a_new_page_is_admitted_with_the_two_lecteur_journals_pending(tmp_path):
    r = codes(NEW_PAGE, write_inputs(tmp_path, journals=JOURNALS, sources=SOURCES14))
    assert r["allowed"] and not any(c.startswith("SUIVI_SELECTION_UNREVIEWED") for c in r["refusal_codes"])


@pytest.mark.parametrize("source", LECTEUR)
def test_a_page_of_each_lecteur_source_is_refused_by_the_pending_journals(tmp_path, source):
    usages = [("page", PAGE_OF[source], "eval", source, "consumed")]
    r = codes(PAGE_OF[source], write_inputs(tmp_path, usages=usages, journals=JOURNALS, sources=SOURCES14))
    assert not r["allowed"]
    assert "SUIVI_SELECTION_UNREVIEWED:docs/READER_LOOP.txt" in r["refusal_codes"]
