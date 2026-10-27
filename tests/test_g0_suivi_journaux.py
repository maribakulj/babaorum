"""Journaux écrits avant tout résultat REF-GALLICA ou mfn (pA, 24/10) : concerns = la liste EXPLICITE des 46 sources existantes.

Une page de chacune des 46 sources est refusée par le journal ; une page neuve (hors des 46) est admise.
Écriture synthétique (même mécanique que test_g0_suivi_lecteur.py) : les cinq fichiers réels ne sont pas épinglés ici.
"""
import pytest

from test_g0_suivi import codes, write_inputs

import json
from pathlib import Path

NAMES = sorted(x["source"] for x in json.loads(
    (Path(__file__).resolve().parents[1] / "reports/g0_suivi_pB/sources_full.DRAFT.json").read_text(encoding="utf-8"))["sources"])
JOURNAL = [{"journal": "AV-GATE-liste-complete", "status": "pending", "concerns": NAMES}]
SOURCES = [{"source": n, "key_spaces": ["gallica"], "status": "consumed", "files": [], "proof": "p"} for n in NAMES]
PAGE_OF = {n: f"gallica:7800{i:03d}/f1" for i, n in enumerate(NAMES)}
NEW_PAGE = "gallica:7899999/f1"


def test_the_list_is_the_explicit_list_of_all_existing_sources():
    assert len(NAMES) == 45 and len(set(NAMES)) == 45


def test_a_new_page_is_admitted_with_the_journal_pending(tmp_path):
    r = codes(NEW_PAGE, write_inputs(tmp_path, journals=JOURNAL, sources=SOURCES))
    assert r["allowed"] and not any(c.startswith("SUIVI_SELECTION_UNREVIEWED") for c in r["refusal_codes"])


@pytest.mark.parametrize("source", NAMES)
def test_a_page_of_each_existing_source_is_refused(tmp_path, source):
    usages = [("page", PAGE_OF[source], "eval", source, "consumed")]
    r = codes(PAGE_OF[source], write_inputs(tmp_path, usages=usages, journals=JOURNAL, sources=SOURCES))
    assert not r["allowed"]
    assert "SUIVI_SELECTION_UNREVIEWED:AV-GATE-liste-complete" in r["refusal_codes"]


# --- R2 (pA, 24/10) : les 322 documents pleias / reader_lm sont non_resolue ; une page de presse gallica NEUVE n'est pas refusée à cause d'eux

R2_SOURCES = [{"source": "reader_lm_corpus", "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "R2 : 322 documents, non résolus"},
              {"source": "pleias_french_pd_322", "key_spaces": ["unknown"], "status": "non_resolue", "files": [], "proof": "R2 : 322 documents, non résolus"}]


