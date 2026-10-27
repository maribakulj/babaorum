"""G0-04 B1 — part du LECTEUR (p9) pour le registre d'usage réel (accord pA du 23/10 ; assemblage et relecture par pB).
Hors réseau ; ne lit que des fichiers locaux ; n'écrit AUCUNE ancre (config/ n'est pas touché).

    PYTHONPATH=src python scripts/g0_b1_lecteur.py --sortie ~/Documents/reviews/babaorum-pipeline/g0_b1_lecteur

Règle (pA, pB, 23/10) : ce que les modèles du lecteur ont VU (entraînement, choix, évaluation) est un USAGE (train, dev,
select, eval), jamais une « consultation » ; le lecteur n'a consulté AUCUNE page du corpus de suivi (REF-GALLICA non
reçu) : journal des consultations vide, avec déclaration et preuve.
Sorties (une par bloc du schéma config/g0_suivi_inputs_schema_v1.json, 4ed97eb8) :
  usage_lecteur_p9.partial.json          lignes usages[] (scope page, clé canonique page_identity, usage, source, status
                                         consumed, proof = fichier + sha256 + ligne de journal)
  sources_lecteur_p9.partial.json        lignes sources[] (fichiers {root, path} + sha256 ; racines : babaorum, axel, reviews)
  select_evidence_lecteur_p9.partial.json lignes evidence[] (choix documentés : target = source, file, line, effet)
  consultations_lecteur_p9.partial.json  bloc consultations (liste vide + déclaration)
  consultations_lecteur_p9.evidence.json preuves de la déclaration
  couverture_lecteur_p9.json             BALAYAGE : toute page citée par un fichier reports/reader_* est-elle couverte par
                                         au moins une ligne d'usage ? les pages non couvertes sont LISTÉES (jamais tues)
  MANIFEST.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
REVIEWS = Path.home() / "Documents/reviews/babaorum-pipeline"
ROOTS = {"babaorum": HERE, "axel": AXEL, "reviews": REVIEWS}
STEM = re.compile(r"(newseye_read|bnl_pack)__(.+?)__")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fichier(root: str, rel: str) -> dict:
    p = ROOTS[root] / rel
    assert p.is_file(), p
    return {"path": {"root": root, "path": rel}, "sha256": sha(p)}


def ligne(root: str, rel: str, motif: str) -> int:
    """Première ligne du journal qui contient le motif (preuve citée) ; absence = arrêt."""
    for k, x in enumerate((ROOTS[root] / rel).read_text(encoding="utf-8").splitlines(), 1):
        if motif in x:
            return k
    raise SystemExit(f"motif de preuve introuvable dans {rel} : {motif!r}")


def main() -> None:
    from babaorum.g0_suivi_identity import page_identity

    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", required=True)
    out = Path(ap.parse_args().sortie)
    assert not any(out.glob("*.json")), "dossier de sortie déjà rempli"
    out.mkdir(parents=True, exist_ok=True)

    # pages de press/train et press/dev : clé canonique par l'adaptateur (espace explicite, comme le constructeur R7(a))
    def espace(corpus, group):
        return "bnl" if corpus == "bnl_pack" else ("newseye_photo" if group == "photo_1983" else "gallica")
    split, cle = {}, {}
    for sp in ("train", "dev"):
        for x in open(AXEL / f"data/prepared/press/{sp}.jsonl", encoding="utf-8"):
            r = json.loads(x)
            k = page_identity(r["page"], espace(r["corpus"], r.get("group"))).page
            split.setdefault(sp, set()).add(k)
            cle[(r["corpus"], r["page"])] = k
    bnl_par_id = {p.split("-")[-1]: k for (c, p), k in cle.items() if c == "bnl_pack"}

    def pages_de_tiges(tiges):
        res, inconnues = set(), set()
        for t in tiges:
            m = STEM.match(t if t.count("__") >= 2 else t + "__")
            k = cle.get((m.group(1), m.group(2))) if m else None
            (res.add(k) if k else inconnues.add(t))
        assert not inconnues, f"tiges hors press/train et press/dev : {sorted(inconnues)[:3]}"
        return res

    def tiges_jsonl(rel, gz=False):
        op = gzip.open if gz else open
        return [json.loads(x)["stem"] for x in op(HERE / rel, "rt", encoding="utf-8")]

    def pages_items(rel):
        return {cle[(r["corpus"], r["page"])] for r in (json.loads(x) for x in open(HERE / rel, encoding="utf-8"))}

    def tiges_texte(rel):  # toutes les tiges citées par un fichier de rapport (lecture en flux)
        op = gzip.open if rel.endswith(".gz") else open
        with op(HERE / rel, "rt", encoding="utf-8") as fh:
            return [f"{m.group(1)}__{m.group(2)}__" for x in fh for m in STEM.finditer(x)]

    with gzip.open(AXEL / "data/e8/kaggle/e8_items.jsonl.gz", "rt", encoding="utf-8") as fh:  # données d'entraînement d'A1
        a1_train = pages_de_tiges(r["stem"] + "__" for r in (json.loads(x) for x in fh) if r["split"] == "train")
    t3 = json.loads((HERE / "reports/reader_t3_part2.json").read_text())["pages"]
    t3_pages = {page_identity(x["page"], "gallica").page for x in t3}
    tune = pages_de_tiges(r["stem"] for r in json.loads((HERE / "reports/reader_r1e_tune.json").read_text())["rows"])
    gate = pages_de_tiges(r["stem"] for r in json.loads((HERE / "reports/reader_gate_calib_2d_rows.json").read_text())["true"])
    r3 = json.loads((HERE / "reports/reader_r3_split.json").read_text())
    r3_cle = lambda i: bnl_par_id.get(i) or cle[("newseye_read", i)]  # noqa: E731  (ids BnL ou noms de page NewsEye)
    r3_tune = {r3_cle(i) for i in r3["tune"]}
    r3_eval = {r3_cle(i) for i in r3["eval"]}
    eval_e8 = pages_de_tiges([t for b in range(6) for t in tiges_jsonl(f"reports/reader_r6_B{b}_eval.jsonl.gz", gz=True)]
                             + tiges_jsonl("reports/reader_r7b_k1_eval.jsonl.gz", gz=True))
    runs26 = {cle[("newseye_read", json.loads(p.read_text())["page"])]
              for p in sorted((REVIEWS / "rappel_pages/runs_26").glob("*_[TD]0*.json"))}

    J = "docs/READER_LOOP.txt"
    SOURCES = [  # (source, usage, pages, fichiers, [preuves : (racine, journal, motif)], description)
        ("lecteur_A1_E17_train", "train", a1_train,
         [fichier("axel", "data/e8/kaggle/e8_items.jsonl.gz"), fichier("axel", "runs/e17/kaggle/a1/head_a1.pt"),
          fichier("axel", "docs/e17.md")],
         [("axel", "docs/e17.md", "entraîné sur tout le train")],
         "tête A1 (axel E17) entraînée sur le split train d'e8_items (39 559 lignes ; aucune ligne photo_1983 : "
         "READER_LOOP « A1 n'a pas vu photo_1983 ») ; fournit les treillis du guide et le système « A1 seul »"),
        ("lecteur_press_train_alloue", "train", split["train"] - a1_train, [fichier("axel", "data/prepared/press/train.jsonl")],
         [("babaorum", J, "kraken affiné sur notre train")],
         "PRUDENCE : pages de press/train hors du split d'A1 ; d'autres modèles du lecteur (R6, K1/R7b, LM press8x) ont pu "
         "les voir sans preuve page à page : déclarées allocated_only (refus SUIVI_TRAINED + étiquette ALLOCATION_ONLY)"),
        ("lecteur_A1_E17_dev", "dev", split["dev"], [fichier("axel", "data/prepared/press/dev.jsonl")],
         [("babaorum", "docs/READER_DOTS_P3_REPORT.txt", "choisie sur TOUT press/dev")],
         "tête A1 choisie sur TOUT press/dev (réserve B4) ; ligne prudente sur toutes les pages de press/dev"),
        ("lecteur_TUNE", "select", tune, [fichier("babaorum", "reports/reader_r1e_tune.json"),
                                          fichier("babaorum", "reports/reader_dots_p2_tune.json"),
                                          fichier("babaorum", "reports/reader_dots_abst_v2.json")],
         [("babaorum", J, "CER NORM sur les 400 lignes TUNE"), ("babaorum", J, "R1e étape 2 FINIE")],
         "400 lignes TUNE : grilles R1d/R1e, λ* = 0,3 (P2), contraintes de τ (ABST v2) et δ (second signal)"),
        ("lecteur_porte_2d", "select", gate, [fichier("babaorum", "reports/reader_gate_calib_2d_rows.json"),
                                              fichier("babaorum", "reports/reader_gate_calib_2d.json")],
         [("babaorum", J, "θ = 0,1 (à 0,2"), ("babaorum", J, "Lot G LANCÉ puis FINI")],
         "porte du lot 2d (θ) et lot G (seuils ajustés) sur ces lignes photo_1983"),
        ("lecteur_portes_1b_r1e", "select", pages_de_tiges(tiges_texte("reports/reader_gate_calib_rows.json")
                                                           + tiges_texte("reports/reader_gate_calib_r1e_rows.json")),
         [fichier("babaorum", "reports/reader_gate_calib_rows.json"), fichier("babaorum", "reports/reader_gate_calib_r1e_rows.json")],
         [("babaorum", J, "Lot 1b FINI"), ("babaorum", J, "R1e étape 2 FINIE")], "portes du lot 1b et de R1e : θ choisi sur ces lignes"),
        ("lecteur_second_signal_abst200", "select", pages_items("reports/reader_items_photo1983_abst200.jsonl"),
         [fichier("babaorum", "reports/reader_items_photo1983_abst200.jsonl"), fichier("babaorum", "reports/reader_dots_ss.json"),
          fichier("babaorum", "reports/reader_dots_abst_v2.json")],
         [("babaorum", J, "SECOND SIGNAL rendu"), ("babaorum", J, "ABST v2 FAIT")],
         "200 lignes de sélection (abst200) : τ (ABST v2) puis δ_a, δ_r (second signal) choisis"),
        ("lecteur_R3_tune", "select", r3_tune, [fichier("babaorum", "reports/reader_r3_split.json"),
                                                fichier("babaorum", "reports/reader_r3_grid_tune300.json")],
         [("babaorum", J, "R3 complet FINI")], "R3 : grille (α, couloir) réglée sur la moitié « tune » ; point d'arrêt R6 choisi dessus"),
        ("lecteur_decision_P3_ABSTv2_gel", "select", eval_e8,
         [fichier("babaorum", "reports/reader_dots_p3_eval.json"), fichier("babaorum", "reports/reader_dots_abst_v2.json"),
          fichier("babaorum", "docs/READER_DOTS_FREEZE_R7A.txt")],
         [("babaorum", J, "P3 FINI (Kaggle T4"), ("babaorum", J, "ABST v2 FAIT"), ("babaorum", J, "GEL pour R7(a) accordé")],
         "éval e8 (2 405 lignes) qui a servi à DÉCIDER : continuer (P3), adopter l'abstention (ABST v2), geler le dispositif"),
        ("lecteur_T3_geometrie", "select", t3_pages, [fichier("babaorum", "reports/reader_t3_part2.json")],
         [("babaorum", J, "8 octobre — T3 + T8")],
         "T3 partie 2 (E12-dev, géométrie seule, aucune image lue par un modèle) : a orienté la priorité R5 ; ligne prudente"),
        ("lecteur_eval_e8", "eval", eval_e8,
         [fichier("babaorum", f"reports/reader_r6_B{b}_eval.jsonl.gz") for b in range(6)]
         + [fichier("babaorum", "reports/reader_r7b_k1_eval.jsonl.gz"), fichier("babaorum", "reports/reader_dots_p3_eval.json")],
         [("babaorum", "docs/READER_DOTS_P3_REPORT.txt", "ÉTIQUETTE : SECOND REGARD")],
         "éval e8 (R6, R7b, P3 sur 2 405 lignes) : déjà exposée, relue plusieurs fois (et select ci-dessus)"),
        ("lecteur_negatifs_pert200", "eval", pages_items("reports/reader_items_photo1983_pert200.jsonl"),
         [fichier("babaorum", "reports/reader_items_photo1983_pert200.jsonl"), fichier("babaorum", "reports/reader_dots_abst_v2.json")],
         [("babaorum", J, "ABST v2 FAIT")], "négatifs Q2 de P3 (200 lignes perturbées) ; mêmes 27 pages que la sélection"),
        ("lecteur_R3_eval", "eval", r3_eval, [fichier("babaorum", "reports/reader_r3_split.json"),
                                              fichier("babaorum", "reports/reader_r3_report.json")],
         [("babaorum", J, "R3 complet FINI")], "R3 : moitié « eval »"),
        ("lecteur_illisibles_P07b", "eval", runs26, [fichier("babaorum", "reports/reader_illisibles_p07b.json")],
         [("babaorum", J, "illisibles ; patch e41836cf")], "mesure descriptive des lignes illisibles (26 pages P-07b)"),
    ]
    usages, sources, evidence = [], [], []
    for nom, usage, pages, fichiers, preuves, quoi in SOURCES:
        refs = [(root, journal, ligne(root, journal, motif)) for root, journal, motif in preuves]
        cites = ", ".join(f"{j}:{n}" for _, j, n in refs)
        root, journal, n = refs[0]
        preuve = f"{quoi} ; {cites} ; " + ", ".join(f"{f['path']['path']} {f['sha256'][:16]}" for f in fichiers)
        statut = "allocated_only" if nom == "lecteur_press_train_alloue" else "consumed"
        sources.append({"source": nom, "key_spaces": sorted({p.split(":")[0] for p in pages}), "status": statut,
                        "files": fichiers, "proof": preuve})
        usages += [{"scope": "page", "key": p, "usage": usage, "source": nom, "status": statut,
                    "proof": f"{nom} ({cites})"} for p in sorted(pages)]
        if usage == "select":
            evidence += [{"target": nom, "file": j, "line": k,
                          "effet": f"promeut en select les pages de {nom} (SUIVI_SELECTED:{nom}) : {quoi}"} for r, j, k in refs]
    # BALAYAGE : pages citées par tout fichier reports/reader_* (tiges corpus__page__ligne) contre les pages couvertes
    couvertes = {u["key"] for u in usages}
    hors, par_fichier, sans_tige = set(), {}, []
    for p in sorted((HERE / "reports").glob("reader_*")):
        op = gzip.open if p.suffix == ".gz" else open
        vus = set()
        try:  # lecture EN FLUX, ligne par ligne (mémoire bornée : certains fichiers dépassent 90 Mo compressés)
            with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                for x in fh:
                    vus |= {(m.group(1), m.group(2)) for m in STEM.finditer(x)}
        except OSError:
            continue
        if not vus:
            sans_tige.append(p.name)
            continue
        ks = {cle.get(v) for v in vus}
        par_fichier[p.name] = {"pages": len(ks - {None}), "hors_press": sorted(f"{c}__{q}" for c, q in vus if (c, q) not in cle)}
        hors |= {k for k in ks if k and k not in couvertes}
    couverture = {"fichiers_balayes": len(par_fichier), "pages_citees_non_couvertes": sorted(hors),
                  "tiges_hors_press_par_fichier": {k: v["hors_press"] for k, v in par_fichier.items() if v["hors_press"]},
                  "fichiers_sans_tige_non_balayes": sans_tige,
                  "limite": ("le balayage ne reconnaît que les tiges corpus__page__ligne ; les fichiers qui citent des pages "
                             "autrement sont couverts par les sources explicites (photo_1983, pert100 dev, R3, T3) ; à part : le LM "
                             "R2 (pleias_french_pd_322, reader_lm_corpus) relève du registre des collections, pas d'usages de page"),
                  "regle": "toute page citée doit être couverte par au moins une ligne d'usage ; sinon elle est listée ici"}
    consult = {"consultations": [], "auteur": "p9 (lecteur babaorum)", "date": "2026-10-23",
               "periode_couverte": "depuis le début du lecteur (07/10/2026, première entrée de docs/READER_LOOP.txt) jusqu'au 23/10/2026",
               "declaration": ("aucune page du corpus de suivi consultée par le lecteur (p9) : REF-GALLICA non reçu ; "
                               "R7(a) S0-S5 écrits et audités sans données")}
    consult_preuve = {"preuves": [fichier("babaorum", "docs/READER_R7A_PREREG.txt"), fichier("babaorum", J)],
                      "lignes": {J: ligne("babaorum", J, "Toute la chaîne R7(a) S0–S5 est écrite et auditée sans données")}}
    for nom, obj in (("usage_lecteur_p9.partial.json", {"usages": usages}),
                     ("sources_lecteur_p9.partial.json", {"sources": sources}),
                     ("select_evidence_lecteur_p9.partial.json", {"evidence": evidence}),
                     ("consultations_lecteur_p9.partial.json", consult),
                     ("consultations_lecteur_p9.evidence.json", consult_preuve),
                     ("couverture_lecteur_p9.json", couverture)):
        (out / nom).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    man = {"script": sha(Path(__file__)), "sorties": {p.name: sha(p) for p in sorted(out.glob("*.json"))}}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"usages": len(usages), "par_source": {s["source"]: sum(u["source"] == s["source"] for u in usages)
                                                            for s in sources},
                      "pages_citees_non_couvertes": len(hors), "fichiers_balayes": len(par_fichier)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
