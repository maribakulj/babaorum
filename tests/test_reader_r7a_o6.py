"""Règle O6 du constructeur R7(a) (scripts/reader_r7a_build.py) : LIMIT:* admis ; TITLE_DISJOINT:<c> admis seulement
pour une collection incomplète du registre dont la preuve de disjonction est épinglée (pA 25/10) ; toute autre
étiquette refusée ; décompte par collection avec les deux preuves (titres de la collection, titre du document) ;
étiquettes jamais effacées ; TITLE_DISJOINT_NAME:<c> (pA 25/10) aux mêmes conditions, preuve au niveau des NOMS,
pages rapportées à part ; CONTENT_DISJOINT:<c> (pA 25/10, AV-G0-SUIVI-CONTENU) aux mêmes conditions, tous les textes
de <c> épinglés, contrôle ngram8_content relu, pages rapportées à part."""

import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reader_r7a_build", HERE / "scripts/reader_r7a_build.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

H = "ab" * 32
DISJ = {"gallicorpora": H, "newseye": None}   # newseye : incomplète au registre, sans preuve épinglée
LIMITS = ["LIMIT:PP-DocLayoutV3", "LIMIT:blla", "LIMIT:dots.mocr"]


def env(*labels):
    return {"role": "suivi", "allowed": True, "refusals": [], "refusal_codes": [], "labels": sorted([*LIMITS, *labels])}


def test_limit_seules_admises():
    assert b.label_refus(env(), DISJ) == []


def test_titre_disjoint_admis_avec_preuve():
    assert b.label_refus(env("TITLE_DISJOINT:gallicorpora"), DISJ) == []


def test_titre_disjoint_sans_preuve_refuse():
    assert b.label_refus(env("TITLE_DISJOINT:newseye"), DISJ) == ["TITLE_DISJOINT:newseye"]


def test_titre_disjoint_collection_hors_registre_ou_complete_refuse():
    assert b.label_refus(env("TITLE_DISJOINT:inconnue"), DISJ) == ["TITLE_DISJOINT:inconnue"]


def test_preuve_mal_formee_refusee():
    for bad in ("", "AB" * 32, "ab" * 31, 7):
        assert b.label_refus(env("TITLE_DISJOINT:gallicorpora"), {"gallicorpora": bad}) == ["TITLE_DISJOINT:gallicorpora"]


def test_autres_etiquettes_refusees():
    for t in ("ISSUE_SHARED", "ISSUE_SHARED:gallicorpora", "TITLE_DISJOINT", "title_disjoint:gallicorpora", "LIMIT"):
        assert b.label_refus(env(t), DISJ) == [t]


G = "cd" * 32
CTX = {"collections": DISJ, "documents": {"gallica:1": G, "gallica:3": G}, "classes": {}}


def test_decompte_par_collection_deux_preuves_et_etiquettes_intactes():
    preuves = {"gallica:1/f1": env("TITLE_DISJOINT:gallicorpora"), "gallica:2/f1": env(),
               "gallica:3/f1": env("TITLE_DISJOINT:gallicorpora"), "gallica:4/f1": env("TITLE_DISJOINT:gallicorpora")}
    avant = copy.deepcopy(preuves)
    r = b.titres_disjoints(preuves, CTX, {"gallica:1/f1", "gallica:2/f1", "gallica:3/f1"})  # 4 : refusée ailleurs (O7)
    assert r == {"gallicorpora": {"preuve_titres_collection_sha256": H, "n_pages": 2,
                                  "pages": [{"page": "gallica:1/f1", "preuves_titre_document_sha256": {"gallica:1": G}},
                                            {"page": "gallica:3/f1", "preuves_titre_document_sha256": {"gallica:3": G}}]}}
    assert preuves == avant   # rien n'est effacé de l'enveloppe archivée


def test_preuve_du_document_pour_chaque_membre_gallica_de_la_classe():
    ctx = {**CTX, "classes": {"gallica:1/f1": {"gallica:1/f1", "gallica:3/f1", "bnl:x/f1"}}}
    assert b.preuves_documents("gallica:1/f1", ctx) == {"gallica:1": G, "gallica:3": G}


def test_document_sans_preuve_arrete():
    with pytest.raises(b.BuildError):
        b.preuves_documents("gallica:2/f1", CTX)
    ctx = {**CTX, "classes": {"gallica:1/f1": {"gallica:1/f1", "gallica:2/f1"}}}   # un membre de la classe sans preuve
    with pytest.raises(b.BuildError):
        b.preuves_documents("gallica:1/f1", ctx)


def test_disjonctions_lues_sur_le_registre_ancre():
    d = b.g0_disjonctions()   # pièces G0 épinglées, sources épinglées (sinon arrêt)
    assert set(d) == {"collections", "methodes", "documents", "classes", "contenus", "controles_contenu"} and d["collections"]
    assert all(v is None or (isinstance(v, str) and len(v) == 64) for v in d["collections"].values())
    assert all(isinstance(v, str) and len(v) == 64 for v in d["documents"].values())


def test_sources_non_epinglees_arrete(monkeypatch):
    import babaorum.g0_suivi as gs
    monkeypatch.setattr(gs, "_documents", lambda inputs: {k: None for k in gs.PINNED_FILES})
    with pytest.raises(b.BuildError, match="non épinglé"):
        b.g0_disjonctions()


# --- disjonction par NOM (pA 25/10, AV-G0-SUIVI-TITRES-NOMS) : mêmes conditions, preuve au niveau des NOMS, rapportée à part
DISJ_N = {"gallicorpora": H, "kb_noms": "ef" * 32, "newseye": None}
METH = {"gallicorpora": None, "kb_noms": "mastheads_read_twice", "newseye": None}


def test_titre_disjoint_nom_admis_avec_preuve_des_noms():
    assert b.label_refus(env("TITLE_DISJOINT_NAME:kb_noms"), DISJ_N, METH) == []
    assert b.label_refus(env("TITLE_DISJOINT_NAME:kb_noms", "TITLE_DISJOINT:gallicorpora"), DISJ_N, METH) == []


def test_titre_disjoint_nom_refuse_sans_preuve_ou_sans_niveau_noms():
    for t in ("TITLE_DISJOINT_NAME:newseye", "TITLE_DISJOINT_NAME:gallicorpora", "TITLE_DISJOINT_NAME:inconnue",
              "TITLE_DISJOINT_NAME:", "title_disjoint_name:kb_noms", "TITLE_DISJOINT_NAMES:kb_noms"):
        assert b.label_refus(env(t), DISJ_N, METH) == [t]
    assert b.label_refus(env("TITLE_DISJOINT_NAME:kb_noms"), DISJ_N) == ["TITLE_DISJOINT_NAME:kb_noms"]   # méthode inconnue


def test_titre_disjoint_refuse_si_la_preuve_est_au_niveau_des_noms():
    assert b.label_refus(env("TITLE_DISJOINT:kb_noms"), DISJ_N, METH) == ["TITLE_DISJOINT:kb_noms"]


def test_pages_par_nom_rapportees_a_part():
    ctx = {"collections": DISJ_N, "methodes": METH, "documents": {"gallica:1": G, "gallica:2": G, "gallica:3": G}, "classes": {}}
    preuves = {"gallica:1/f1": env("TITLE_DISJOINT_NAME:kb_noms"), "gallica:2/f1": env("TITLE_DISJOINT:gallicorpora"),
               "gallica:3/f1": env("TITLE_DISJOINT_NAME:kb_noms", "TITLE_DISJOINT:gallicorpora")}
    avant = copy.deepcopy(preuves)
    gardees = set(preuves)
    nom = b.titres_disjoints_nom(preuves, ctx, gardees)
    assert nom["mention"] == "disjonction par NOM, preuve plus faible" and nom["pages"] == ["gallica:1/f1", "gallica:3/f1"]
    assert nom["collections"] == {"kb_noms": {"preuve_noms_collection_sha256": "ef" * 32, "methode": "mastheads_read_twice",
                                              "n_pages": 2,
                                              "pages": [{"page": "gallica:1/f1", "preuves_titre_document_sha256": {"gallica:1": G}},
                                                        {"page": "gallica:3/f1", "preuves_titre_document_sha256": {"gallica:3": G}}]}}
    assert list(b.titres_disjoints(preuves, ctx, gardees)) == ["gallicorpora"]   # les TITLE_DISJOINT restent comptés à part
    assert preuves == avant


def test_methodes_lues_sur_le_registre_ancre():
    d = b.g0_disjonctions()
    assert set(d["methodes"]) == set(d["collections"])
    assert all(m is None or isinstance(m, str) for m in d["methodes"].values())


# --- disjonction par CONTENU (pA 25/10, AV-G0-SUIVI-CONTENU) : tous les textes épinglés, contrôle ngram8_content relu, rapportée à part
E, P, V = "12" * 32, "34" * 32, "56" * 32
CONT = {"gallicorpora": None, "hipe_ocrepair": {"empreinte": E, "unites": 4570, "preuve": P}, "newseye": None}


def ctrl(page, **kw):
    return {"page": page, "check": "ngram8_content", "status": "complete", "against": ["hipe_ocrepair"],
            "against_manifests": {"hipe_ocrepair": E}, "findings": [], "page_text": {"kind": "vt_bbvlm", "sha256": V}} | kw


def test_contenu_disjoint_admis_si_textes_epingles():
    assert b.label_refus(env("CONTENT_DISJOINT:hipe_ocrepair"), DISJ, None, CONT) == []
    assert b.label_refus(env("CONTENT_DISJOINT:hipe_ocrepair", "TITLE_DISJOINT:gallicorpora"), DISJ, None, CONT) == []


def test_contenu_disjoint_refuse_sans_textes_ou_mal_forme():
    for t in ("CONTENT_DISJOINT:gallicorpora", "CONTENT_DISJOINT:inconnue", "CONTENT_DISJOINT:", "content_disjoint:hipe_ocrepair",
              "CONTENT_DISJOINTS:hipe_ocrepair", "SUIVI_CONTENT_OVERLAP:hipe_ocrepair"):
        assert b.label_refus(env(t), DISJ, None, CONT) == [t]
    assert b.label_refus(env("CONTENT_DISJOINT:hipe_ocrepair"), DISJ) == ["CONTENT_DISJOINT:hipe_ocrepair"]   # contenus inconnus
    for bad in ({"empreinte": "AB" * 32, "preuve": P}, {"empreinte": E}, {"empreinte": E, "preuve": 7}, E):
        assert b.label_refus(env("CONTENT_DISJOINT:hipe_ocrepair"), DISJ, None, {"hipe_ocrepair": bad}) == ["CONTENT_DISJOINT:hipe_ocrepair"]


def test_pages_par_contenu_rapportees_a_part_avec_vt():
    ctx = {"collections": DISJ, "methodes": {}, "documents": {"gallica:1": G}, "contenus": CONT,
           "classes": {"gallica:2/f1": {"gallica:2/f1", "gallica:3/f1", "bnl:x/f1"}},
           "controles_contenu": {"gallica:2/f1": ctrl("gallica:2/f1"), "gallica:3/f1": ctrl("gallica:3/f1")}}
    preuves = {"gallica:1/f1": env("TITLE_DISJOINT:gallicorpora"), "gallica:2/f1": env("CONTENT_DISJOINT:hipe_ocrepair")}
    avant = copy.deepcopy(preuves)
    r = b.contenus_disjoints(preuves, ctx, set(preuves), {"gallica:1/f1": H, "gallica:2/f1": V})
    assert r["mention"].startswith("disjonction par CONTENU") and r["pages"] == ["gallica:2/f1"]
    assert r["collections"] == {"hipe_ocrepair": {"empreinte_textes": E, "unites": 4570, "preuve_compte_sha256": P, "n_pages": 1,
                                                  "pages": [{"page": "gallica:2/f1", "vt_sha256": {"gallica:2/f1": V, "gallica:3/f1": V}}]}}
    assert list(b.titres_disjoints(preuves, ctx, set(preuves))) == ["gallicorpora"] and preuves == avant


def test_controle_contenu_invalide_arrete():
    base = {"contenus": CONT, "classes": {}}
    for k in (None, ctrl("gallica:2/f1", status="incomplete"), ctrl("gallica:2/f1", against=["newseye"]),
              ctrl("gallica:2/f1", against_manifests={"hipe_ocrepair": H}),
              ctrl("gallica:2/f1", findings=[{"kind": "content_overlap", "source": "hipe_ocrepair"}]),
              ctrl("gallica:2/f1", page_text={"kind": "ocr", "sha256": V})):
        ctx = base | {"controles_contenu": {} if k is None else {"gallica:2/f1": k}}
        with pytest.raises(b.BuildError):
            b.vt_contenu("gallica:2/f1", "hipe_ocrepair", ctx, V)
    with pytest.raises(b.BuildError):   # classe sans membre Gallica
        b.vt_contenu("bnl:x/f1", "hipe_ocrepair", base | {"controles_contenu": {}}, V)


def test_vt_controlee_doit_etre_la_vt_lue():
    """Décision pA (25/10, M1) : page_text.sha256 == truth_sha256 de la page, pour elle ET chaque membre Gallica de sa classe."""
    ctx = {"contenus": CONT, "classes": {"gallica:2/f1": {"gallica:2/f1", "gallica:3/f1"}},
           "controles_contenu": {"gallica:2/f1": ctrl("gallica:2/f1"), "gallica:3/f1": ctrl("gallica:3/f1")}}
    assert b.vt_contenu("gallica:2/f1", "hipe_ocrepair", ctx, V) == {"gallica:2/f1": V, "gallica:3/f1": V}
    with pytest.raises(b.BuildError, match="VT lue"):   # VT livrée en v2 après un contrôle sur la v1
        b.vt_contenu("gallica:2/f1", "hipe_ocrepair", ctx, H)
    ctx["controles_contenu"]["gallica:3/f1"] = ctrl("gallica:3/f1", page_text={"kind": "vt_bbvlm", "sha256": H})
    with pytest.raises(b.BuildError, match="VT lue"):   # un alias contrôlé sur une autre VT
        b.vt_contenu("gallica:2/f1", "hipe_ocrepair", ctx, V)


def test_contenus_lus_sur_le_registre_ancre():
    d = b.g0_disjonctions()
    assert set(d["contenus"]) == set(d["collections"])
    assert all(v is None or (set(v) == {"empreinte", "unites", "preuve"} and len(v["empreinte"]) == 64) for v in d["contenus"].values())
    assert all(k["check"] == "ngram8_content" for k in d["controles_contenu"].values())
