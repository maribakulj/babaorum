"""R7(a) — S5 : rapport (docs/READER_R7A_PREREG.txt ; docs/READER_DOTS_FREEZE_R7A.txt ; CLASSE B ; accord pA du 23/10).
Écrit SANS DONNÉES ; lancé UNE fois sur les vraies sorties, après relecture Sonnet et pT.

    # 1) extraction du témoin A1 (léger, en flux ; là où se trouve a1.jsonl.gz, p. ex. Kaggle CPU) :
    python scripts/reader_r7a_s5_report.py extraire --s3-sortie <dossier> --s3-manifest-sha256 <SHA> --sortie <dossier>
    # 2) rapport (Mac, CPU, quelques minutes) :
    python scripts/reader_r7a_s5_report.py rapport --build <dossier> --build-manifest-sha256 <SHA> \\
        --exclusion <dossier> --exclusion-manifest-sha256 <SHA> --s4-sortie <dossier> --s4-manifest-sha256 <SHA> \\
        --a1 <dossier d'extraction> --a1-manifest-sha256 <SHA> --sortie <dossier vide, hors des entrées>

Systèmes (PREREG §3, TÉMOIN A1, PRÉCISIONS) : dots libre (clean de la config libre) ; dispositif GELÉ (clean de
guide_l0.3, sortie vide si σ < τ) ; A1 seul (axel.reader.ctc.greedy sur le logp float16 relu, sans abstention ; avec
abstention en descriptif). Mesure : NORM gelée (reader_dots_p3_report.norm, épinglé par le gel) ; CER = 100 × Σ erreurs
(Levenshtein, rapidfuzz) / Σ longueurs de vérité NORM ; CER brut (sans NORM) à côté.
DEUX CER par système (règle de Marcel) : (a) sur le TEXTE BRUT produit ; (b) sur le texte RELU DEPUIS L'ALTO LIVRÉ :
chaque page et chaque système passent par la chaîne de livraison du pipeline (bench_pipeline.build_graph + add_words,
mots sans boîte, puis export_page.export_alto en mode strict), et le texte est relu dans le fichier livré (String par
TextLine, dans l'ordre du fichier) ; une ligne absente de l'ALTO (vide omise, export refusé) est relue vide. Le verdict
pré-enregistré porte sur (a) ; (b) est rapporté à côté, avec les pertes d'export.
Critère primaire : K1 = CER(libre) − CER(gelé), K2 = CER(A1) − CER(gelé) sur les pages de PRESSE RELUES (marcel_corrige) ;
IC 95 % apparié PAR PAGE (pages tirées avec remise, 10 000 tirages, random.Random(1), indices 250 et 9 749) ; MDE
observée = 2,8 × écart type des tirages ; IC par LIGNE à côté (descriptif, même procédé sur les lignes). Lecture : « sens
confirmé » (IC > 0), « sens non confirmé » (IC contient 0), « sens contraire » (IC < 0). K1 sous l'hypothèse M3.
Zones vides : part inventée (compteur NOUVEAU de reader_dots_abst : ≥ 3 jetons) du gelé AVEC abstention ; verdict :
BORNE HAUTE de Clopper-Pearson à 95 % ≤ 5 % (décision pA du 23/10 ; la part et son IC rapportés à côté), seulement si n ≥ 200 (plancher ; au-dessous : effectif et IC de Clopper-Pearson SANS verdict) ; tout taux
de Clopper-Pearson n'a de verdict que si n ≥ 100 ; IC par page (bootstrap des pages) ; règle « fragile » ; pages avec
au moins une zone non abstenue et une zone inventée ; par famille, côté, type de page ; sans abstention ; dots libre et
A1 ; part abstenue ; zones « difficiles » à part ; exclusions par motif (exclusions.json).
INTERRUPTION (PREREG) : passage S4 interrompu, fin hors {eos, plafond}, tige sans ses deux configurations, ou couverture
inexacte : AUCUN verdict (statut écrit), retour à pA.
O6 (pA 25/10) : pages admises par TITLE_DISJOINT:<c>, par collection, avec les DEUX preuves (sha de la preuve des titres
de la collection ; sha de la preuve du titre du document de chaque page) (repris du constructeur, o6_titres_disjoints).
DISJONCTION PAR NOM (pA 25/10) : pages admises par TITLE_DISJOINT_NAME:<c> rapportées À PART (« disjonction par NOM, preuve
plus faible », méthode et deux preuves) ; critère primaire et critère des zones donnés AVEC elles (verdict) et SANS elles
(sensibilité) ; un écart de lecture est écrit en tête du rapport (sensibilite_disjonction_par_nom_ecart) ; si les deux
CONCLUENT différemment (passe d'un côté, échoue de l'autre), le verdict reçoit l'étiquette FRAGILE, en tête
(verdict_fragile_disjonction_par_nom).
DISJONCTION PAR CONTENU (pA 25/10, AV-G0-SUIVI-CONTENU) : pages admises par CONTENT_DISJOINT:<c> rapportées À PART de la
même façon (empreinte des textes de la collection, preuve du compte, sha de la VT par membre), sensibilité SANS elles,
écart en tête (sensibilite_disjonction_par_contenu_ecart), FRAGILE si les deux lectures concluent différemment
(verdict_fragile_disjonction_par_contenu). Les deux sensibilités sont calculées séparément, sur le résultat principal
avant toute étiquette FRAGILE.
Sorties : r7a_rapport.json, planche_s5.png (contrôle : zones où le gelé invente, puis 20 lignes primaires tirées au
hasard, chacune avec vérité / libre / gelé / A1), MANIFEST.json en dernier.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import random
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(AXEL / "src"))
FREEZE_SHA = "53e17a2bcb78133315ac4d309d03e39ae71e37d7c4fc8a53ab410bad182e42e7"
TIMING_SHA = "9cc562fecfe015aadf87243329233615da9c87f023751745e7fe98525521155f"  # timing.json d'A1 (chars, L = 32)
CTC_SHA = "1213796968a6d207de822b184849d289aa107084398adb3e891f5da05ce9a3d1"  # axel/reader/ctc.py (greedy)
DEPENDANCES = {  # B1 pT : mesure (compteur, Clopper-Pearson) et chaîne de livraison ALTO, épinglées et archivées au rapport
    "scripts/reader_dots_abst.py": "821855f2af427a9af1835dabd36865933296699a06f20fbfc7de7e3255cee754",
    "src/babaorum/page_graph.py": "6128ffe558b546c5dab8e7a02b7d74d74b1163ca4cabbcfa31ab729603bd0fa2",
    "src/babaorum/export_page.py": "2fac2d200ace94891d545f3192dd8d8af54379c07c27dd128d04d3da2897cc78",
    "src/babaorum/bench_pipeline.py": "ca85ce23ccf57757aca1e207866acfb9d3863ac7de771aeb92e4cf526fc0cb8f",
    "src/babaorum/words_from_ctc.py": "bd425acbd95c4a2d4ffbe8dfd6012f41b00a2cb791804b65a9b0b24561a43fcb"}
TIRAGES, BAS, HAUT, GRAINE = 10_000, 250, 9_749, 1
SEUIL_ZONES, PLANCHER, N_VERDICT = 0.05, 200, 100
CONFIGS = ("libre", "guide_l0.3")
FINS_OK = {"eos", "plafond"}
MENTION_NOM = "disjonction par NOM, preuve plus faible"   # PREREG, clarification datée du 25/10 (pA)
FRAGILE_NOM = "FRAGILE (règle fragile du PREREG) : AVEC et SANS les pages par nom, les lectures concluent différemment"
MENTION_CONTENU = "disjonction par CONTENU (aucun 8-gramme commun mesuré avec tous les textes épinglés de la collection)"
FRAGILE_CONTENU = "FRAGILE (règle fragile du PREREG) : AVEC et SANS les pages par contenu, les lectures concluent différemment"
RESERVE_B4 = ("RÉSERVE B4 (EN TÊTE) : la tête A1, qui fournit les treillis du guide et le système « A1 seul », a été "
              "choisie sur TOUT press/dev (axel E17). R7(a) lit des pages NEUVES, hors de press/dev, mais la tête n'a pas "
              "été choisie sur un panneau indépendant.")
ETIQUETTE = ("CONFIRMATION R7(a), PREMIER REGARD : lecture unique des pages neuves avec le dispositif gelé. Toute "
             "analyse ajoutée ou refaite après ce rapport est étiquetée SECOND REGARD et ne change pas le verdict.")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load("exclure", HERE / "scripts/reader_r7a_exclure.py")
need, sha = ex.need, ex.sha


def gel() -> dict:
    """Document du gel épinglé et ses 9 empreintes (NORM, abstention, compteurs viennent de fichiers du gel)."""
    raw = (HERE / "docs/READER_DOTS_FREEZE_R7A.txt").read_bytes()
    need(hashlib.sha256(raw).hexdigest() == FREEZE_SHA, "document du gel modifié")
    ent = {m.group(2): m.group(1) for m in (re.match(r"\s+([0-9a-f]{64})\s+(\S+)", x) for x in raw.decode().splitlines()) if m}
    need(len(ent) == 9 and all(sha(HERE / p) == h for p, h in ent.items()), "gel rompu")
    for p, h in DEPENDANCES.items():
        need(sha(HERE / p) == h, f"dépendance de mesure ou d'export modifiée : {p}")
    return {**ent, **DEPENDANCES}


# ---------------------------------------------------------------- extraction du témoin A1
def extraire(a) -> None:
    import numpy as np

    s3, out = Path(a.s3_sortie), Path(a.sortie)
    need(out.resolve() != s3.resolve() and s3.resolve() not in out.resolve().parents, "sortie dans la sortie S3")
    need(not out.exists() or not any(out.iterdir()), "dossier de sortie non vide")
    mp = s3 / "MANIFEST.json"
    need(sha(mp) == a.s3_manifest_sha256, "MANIFEST de la sortie S3 différent du SHA épinglé")
    man = json.loads(mp.read_text())
    for n, h in man["sorties"].items():
        need(sha(s3 / n) == h, f"sortie S3 modifiée : {n}")
    tp = Path(a.timing) if a.timing else AXEL / "runs/e17/kaggle/a1/timing.json"
    cp_ = Path(a.ctc) if a.ctc else AXEL / "src/axel/reader/ctc.py"
    need(sha(tp) == TIMING_SHA and sha(cp_) == CTC_SHA, "timing.json ou ctc.py d'A1 différents des pièces épinglées")
    timing = json.loads(tp.read_text())
    need(timing.get("L") == 32, "timing.json n'est pas celui d'A1")
    chars = timing["chars"]
    ctc = _load("ctc_a1", cp_)
    import torch

    out.mkdir(parents=True, exist_ok=True)
    n, diff, vus = 0, 0, set()
    with gzip.open(s3 / "a1.jsonl.gz", "rt", encoding="utf-8") as f, open(out / "a1_glouton.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            need(r["stem"] not in vus, f"tige en double : {r['stem']}")
            vus.add(r["stem"])
            a1 = ctc.greedy(torch.from_numpy(np.asarray(r["logp"], dtype=np.float32)), chars)
            g.write(json.dumps({"stem": r["stem"], "part": r["part"], "a1": a1, "hyp_s3": r["hyp"]}, ensure_ascii=False) + "\n")
            n += 1
            diff += a1 != r["hyp"]
    res = {"entrees": {"s3_sortie_manifest": a.s3_manifest_sha256, "timing": TIMING_SHA, "ctc": CTC_SHA,
                       "script": sha(Path(__file__))},
           "comptes": {"tiges": n, "glouton_float16_different_de_hyp_s3": diff},
           "sorties": {"a1_glouton.jsonl": sha(out / "a1_glouton.jsonl")}}
    (out / "MANIFEST.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- mesures
def boot(unites, stat, graine=GRAINE):
    """IC 95 % par tirage avec remise des unités (pages ou lignes) ; écart type des tirages (MDE = 2,8 × sd)."""
    if not unites:
        return {"ic95": [None, None], "sd": None, "mde": None}
    rng = random.Random(graine)
    d = sorted(stat([rng.choice(unites) for _ in unites]) for _ in range(TIRAGES))
    d = [x for x in d if x is not None]
    if len(d) < TIRAGES:
        return {"ic95": [None, None], "sd": None, "mde": None, "tirages_vides": TIRAGES - len(d)}
    sd = statistics.pstdev(d)
    return {"ic95": [d[BAS], d[HAUT]], "sd": sd, "mde": 2.8 * sd}


def sens(ic) -> str:
    lo, hi = ic
    if lo is None:
        return "indéterminé"
    return "sens confirmé" if lo > 0 else ("sens contraire" if hi < 0 else "sens non confirmé")


def cer(rows, sysname, champ="e"):
    n = sum(r["n"] for r in rows)
    return 100 * sum(r[champ][sysname] for r in rows) / n if n else None


def alto_relu(page_lines, textes, image_meta, image_nom, pk, systeme, dossier: Path, base: str):
    """Chaîne de livraison du pipeline : graphe de page (lignes dans l'ordre du panneau), mots sans boîte, ALTO strict,
    puis relecture du texte livré par TextLine. Renvoie (textes relus par ligne, compte rendu d'export)."""
    from babaorum.bench_pipeline import add_words, build_graph
    from babaorum.export_page import ExportError, export_alto
    from babaorum.words_from_ctc import split_words

    units, words = [], {}
    for k, it in enumerate(page_lines):
        x0, y0, x1, y1 = it["box"]
        poly = it["polygon"] or [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        units.append({"id": f"l{k}", "text": textes[k], "polygon": poly, "baseline": [[x0, y1], [x1, y1]], "block": "b0"})
        words[f"l{k}"] = [(w.text, None) for w in split_words(textes[k])]
    prov = {"run_id": f"r7a-s5:{systeme}:{pk}", "components": [{"name": systeme, "code_revision": "r7a_s5",
                                                               "config_sha256": None, "checkpoint_sha256": None, "license": None}]}
    cr = {"accepte": False}
    try:
        graph, _ = build_graph(units, role="prediction", identity={"page_key": pk, "document_key": None, "title_key": None},
                               image=image_meta, provenance=prov,
                               block_labels={"b0": "TextBlock"})  # un bloc par page : ALTO exige un parent à chaque ligne
        add_words(graph, words)
        res = export_alto(graph, image_filename=image_nom, physical_img_nr=1, mode="strict")
    except Exception as exc:  # export refusé : rien de livré, toutes les lignes relues vides (compté)
        cr["refus"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return [""] * len(page_lines), cr
    cr = {"accepte": bool(res.accepted), "pertes": {},
          "pertes_detail": [{"gravite": l.severity, "code": l.code, "chemin": l.path} for l in res.losses if l.severity != "info"
                            or l.code in ("empty_line_omitted", "blank_line_omitted", "word_geometry_unavailable")]}
    for loss in res.losses:
        cr["pertes"][f"{loss.severity}:{loss.code}"] = cr["pertes"].get(f"{loss.severity}:{loss.code}", 0) + 1
    if not res.accepted:
        return [""] * len(page_lines), cr
    # B2 pT : l'ALTO LIVRÉ et son graphe annexe sont écrits, liés aux tiges, puis RELUS depuis les octets écrits
    (dossier / f"{base}.alto.xml").write_text(res.xml, encoding="utf-8")
    (dossier / f"{base}.sidecar.json").write_text(res.sidecar_json, encoding="utf-8")
    (dossier / f"{base}.tiges.json").write_text(json.dumps({f"n_l{k}": it["stem"] for k, it in enumerate(page_lines)},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    cr["fichiers"] = [f"alto/{base}.alto.xml", f"alto/{base}.sidecar.json", f"alto/{base}.tiges.json"]
    loc = lambda t: t.rsplit("}", 1)[-1]  # noqa: E731
    lu = {}
    try:
        racine = ET.fromstring((dossier / f"{base}.alto.xml").read_bytes())
    except ET.ParseError as exc:  # mineur 6 Sonnet : ALTO illisible = rien de livré, compté
        cr["refus"] = f"ParseError: {exc}"
        cr["accepte"] = False
        return [""] * len(page_lines), cr
    for el in racine.iter():
        if loc(el.tag) == "TextLine":
            lu[el.get("ID")] = " ".join(c.get("CONTENT", "") for c in el if loc(c.tag) == "String")
    return [lu.get(f"n_l{k}", "") for k in range(len(page_lines))], cr


def verdict_zones(n, ic_cp, ic_pages, verdict_ok) -> str:
    """Critère des zones (PREREG, clarification pA du 23/10) : borne haute CP 95 % ≤ 5 %, si n ≥ plancher ; « fragile »
    si la borne haute de l'IC par page dépasse 5 % alors que le binomial passe."""
    if not verdict_ok:
        return "AUCUN VERDICT (passage interrompu)"
    if n < PLANCHER:
        return f"SOUS LE PLANCHER ({n} < {PLANCHER}) : effectif et IC sans verdict"
    passe = ic_cp[1] <= SEUIL_ZONES
    fragile = passe and ic_pages[1] is not None and ic_pages[1] > SEUIL_ZONES
    return ("PASSE" if passe else "ÉCHOUE") + (" — fragile (zones concentrées sur peu de pages)" if fragile else "")


def rapport(a) -> None:
    from rapidfuzz.distance import Levenshtein

    entrees_gel = gel()
    p3r = _load("p3r_norm", HERE / "scripts/reader_dots_p3_report.py")  # NORM gelée (fichier du gel)
    ab = _load("abst", HERE / "scripts/reader_dots_abst.py")  # compteur NOUVEAU et Clopper-Pearson
    norm, NEW, invented, cp = p3r.norm, ab.NEW, ab.invented, ab.cp
    tau = json.loads((HERE / "reports/reader_dots_abst_v2.json").read_text())["tau"]
    need(tau == 20.827896720535783, "τ du gel")
    build, excl, s4, a1d, out = Path(a.build), Path(a.exclusion), Path(a.s4_sortie), Path(a.a1), Path(a.sortie)
    for d in (build, excl, s4, a1d):
        need(out.resolve() != d.resolve() and d.resolve() not in out.resolve().parents, f"sortie dans {d}")
    need(not out.exists() or not any(out.iterdir()), "dossier de sortie non vide")
    bman = ex.check_build(build, a.build_manifest_sha256)
    eman = ex.check_build(excl, a.exclusion_manifest_sha256)
    need(eman["entrees"]["build_MANIFEST.json"] == a.build_manifest_sha256, "l'exclusion ne cite pas ce constructeur")
    sm = s4 / "r7a_s4_MANIFEST.json"
    need(sha(sm) == a.s4_manifest_sha256, "MANIFEST de S4 différent du SHA épinglé")
    s4m = json.loads(sm.read_text())
    for n, h in s4m["sorties"].items():
        need(sha(s4 / n) == h, f"sortie S4 modifiée : {n}")
    need(s4m["entrees"]["exclusion"] == a.exclusion_manifest_sha256 and s4m["entrees"]["gel"] == FREEZE_SHA,
         "S4 n'a pas lu cette exclusion ou ce gel")
    am = a1d / "MANIFEST.json"
    need(sha(am) == a.a1_manifest_sha256, "MANIFEST de l'extraction A1 différent du SHA épinglé")
    a1m = json.loads(am.read_text())
    need(sha(a1d / "a1_glouton.jsonl") == a1m["sorties"]["a1_glouton.jsonl"], "extraction A1 modifiée")
    need(a1m["entrees"]["s3_sortie_manifest"] == s4m["entrees"]["s3_sortie"], "l'extraction A1 ne vient pas du treillis lu par S4")
    need(a1m["entrees"].get("script") == sha(Path(__file__)) and a1m["entrees"].get("timing") == TIMING_SHA
         and a1m["entrees"].get("ctc") == CTC_SHA, "extraction A1 produite par une autre version du script ou d'A1")

    items = [json.loads(x) for x in open(excl / "items_retenus.jsonl", encoding="utf-8")]
    excl_rep = json.loads((excl / "exclusions.json").read_text())
    build_rep = json.loads((build / "r7a_build.json").read_text())
    need(isinstance(build_rep.get("o6_titres_disjoints"), dict), "r7a_build.json sans o6_titres_disjoints (constructeur antérieur à O6 du 25/10)")
    need(isinstance(build_rep.get("o6_titres_disjoints_nom"), dict) and isinstance(build_rep["o6_titres_disjoints_nom"].get("pages"), list),
         "r7a_build.json sans o6_titres_disjoints_nom (constructeur antérieur à la disjonction par NOM du 25/10)")
    pages_nom = set(build_rep["o6_titres_disjoints_nom"]["pages"])
    need(isinstance(build_rep.get("o6_contenus_disjoints"), dict) and isinstance(build_rep["o6_contenus_disjoints"].get("pages"), list),
         "r7a_build.json sans o6_contenus_disjoints (constructeur antérieur à la disjonction par CONTENU du 25/10)")
    pages_contenu = set(build_rep["o6_contenus_disjoints"]["pages"])
    sources = json.loads((build / "sources.json").read_text())
    run = json.loads((s4 / "r7a_s4_run.json").read_text())
    sorties, n_s4, doublons = {}, 0, 0
    for line in open(s4 / "r7a_s4.jsonl", encoding="utf-8"):
        r = json.loads(line)
        n_s4 += 1
        doublons += r["config"] in sorties.get(r["stem"], {})
        sorties.setdefault(r["stem"], {})[r["config"]] = r
    a1_lignes = [json.loads(x) for x in open(a1d / "a1_glouton.jsonl", encoding="utf-8")]
    a1 = {r["stem"]: r for r in a1_lignes}
    stems = {it["stem"] for it in items}
    # INTERRUPTION (PREREG) : tout manquement = aucun verdict
    defauts = []
    if run.get("interrompu") is not False or run.get("fins_anormales") != []:  # B2 Sonnet : clés exigées, pas d'ouverture
        defauts.append(f"passage S4 interrompu ou journal incomplet (interrompu = {run.get('interrompu')!r}, "
                       f"fins_anormales = {len(run.get('fins_anormales') or [])})")
    if doublons or n_s4 != 2 * len(sorties):
        defauts.append(f"sorties S4 en double ou en trop ({doublons} doublons, {n_s4} lignes pour {len(sorties)} tiges)")
    if set(sorties) != stems:
        defauts.append(f"couverture S4 inexacte ({len(stems - set(sorties))} tiges manquantes, {len(set(sorties) - stems)} en plus)")
    if any(set(v) != set(CONFIGS) for v in sorties.values()):
        defauts.append("tige sans ses deux configurations")
    fins = [c["fin"] for v in sorties.values() for c in v.values()]
    if any(f not in FINS_OK for f in fins):
        defauts.append(f"fin hors eos/plafond : {sum(f not in FINS_OK for f in fins)}")
    if set(a1) != stems or len(a1_lignes) != len(a1):  # M1 pT : doublons A1 refusés
        defauts.append(f"couverture A1 inexacte ou doublons ({len(a1_lignes) - len(a1)} doublons)")
    statut = "COMPLET" if not defauts else "INTERROMPU — AUCUN VERDICT (PREREG) ; retour à pA : " + " ; ".join(defauts)
    verdict_ok = not defauts
    if not verdict_ok:  # B1 Sonnet : PREREG « décision écrite avant toute lecture des CER » : AUCUN CER calculé ni écrit
        rep = {"reserve_B4": RESERVE_B4, "etiquette": ETIQUETTE, "statut": statut, "defauts": defauts,
               "couverture": {"items_retenus": len(stems), "tiges_s4": len(sorties), "lignes_s4": n_s4, "tiges_a1": len(a1)},
               "lancement_S4": {k: run.get(k) for k in ("dtype", "gpu", "secondes", "interrompu", "fins_anormales")},
               "entrees": {"build": a.build_manifest_sha256, "exclusion": a.exclusion_manifest_sha256,
                           "s4": a.s4_manifest_sha256, "a1": a.a1_manifest_sha256, "gel": FREEZE_SHA, "script": sha(Path(__file__))}}
        out.mkdir(parents=True, exist_ok=True)
        (out / "r7a_rapport.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        (out / "MANIFEST.json").write_text(json.dumps({"entrees": rep["entrees"],
                                                       "sorties": {"r7a_rapport.json": sha(out / "r7a_rapport.json")}}, indent=1))
        print(statut)
        sys.exit(3)

    def textes(it):
        o = sorties.get(it["stem"], {})
        g = "" if it["abstention"] else o.get("guide_l0.3", {}).get("clean", "")
        a1t = a1.get(it["stem"], {}).get("a1", "")
        return {"libre": o.get("libre", {}).get("clean", ""), "gele": g, "a1": a1t, "a1_abst": "" if it["abstention"] else a1t}

    SYS = ("libre", "gele", "a1", "a1_abst")
    # ---------- lignes de texte : CER brut texte, CER NORM texte, CER NORM relu dans l'ALTO livré
    lignes = [it for it in items if it["part"] in ("presse", "livre")]
    par_page = {}
    for it in lignes:
        par_page.setdefault(it["page_key"], []).append(it)
    rows, export = [], {}
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    out.mkdir(parents=True, exist_ok=True)
    (out / "alto").mkdir()
    for ip, (pk, pl) in enumerate(sorted(par_page.items())):
        src = sources[pk]
        need(sha(Path(src["image"])) == src["image_sha256"], f"image modifiée : {pk}")  # mineur 5 Sonnet
        with Image.open(src["image"]) as im:
            meta = {"sha256": src["image_sha256"], "width": im.size[0], "height": im.size[1]}
        tx = [textes(it) for it in pl]
        relus = {}
        for s in SYS:
            relus[s], export[f"{pk}|{s}"] = alto_relu(pl, [t[s] for t in tx], meta, Path(src["image"]).name, pk, s,
                                                      out / "alto", f"p{ip:03d}_{s}")
        for k, it in enumerate(pl):
            vt = norm(it["truth"])
            rows.append({"stem": it["stem"], "page": pk, "type": it["part"], "niveau": it["truth_level"],
                         "n": len(vt), "n_brut": len(it["truth"]),
                         "e": {s: Levenshtein.distance(norm(tx[k][s]), vt) for s in SYS},
                         "e_alto": {s: Levenshtein.distance(norm(relus[s][k]), vt) for s in SYS},
                         "e_brut": {s: Levenshtein.distance(tx[k][s], it["truth"]) for s in SYS},
                         "abst": it["abstention"]})
    for r in rows:
        r["n_alto"] = r["n"]

    def bloc(rs, titre):
        res = {"lignes": len(rs), "pages": len({r["page"] for r in rs}), "caracteres_verite_norm": sum(r["n"] for r in rs)}
        for champ, nom in (("e", "cer_norm_texte"), ("e_alto", "cer_norm_alto_livre"), ("e_brut", "cer_brut_texte")):
            if champ == "e_brut":
                nb = sum(r["n_brut"] for r in rs)
                res[nom] = {s: (100 * sum(r[champ][s] for r in rs) / nb if nb else None) for s in SYS}
            else:
                res[nom] = {s: cer(rs, s, champ) for s in SYS}
        pages = sorted({r["page"] for r in rs})
        byp = {p: [r for r in rs if r["page"] == p] for p in pages}
        for k_nom, autre in (("K1", "libre"), ("K2", "a1")):
            for champ, suffixe in (("e", ""), ("e_alto", "_alto_livre")):
                def stat_pages(tirage, autre=autre, champ=champ):
                    sel = [r for p in tirage for r in byp[p]]
                    n = sum(r["n"] for r in sel)
                    return 100 * (sum(r[champ][autre] for r in sel) - sum(r[champ]["gele"] for r in sel)) / n if n else None

                def stat_lignes(tirage, autre=autre, champ=champ):
                    n = sum(r["n"] for r in tirage)
                    return 100 * (sum(r[champ][autre] for r in tirage) - sum(r[champ]["gele"] for r in tirage)) / n if n else None
                val = stat_pages(pages)
                bp, bl = boot(pages, stat_pages), boot(rs, stat_lignes)
                res[f"{k_nom}{suffixe}"] = {"valeur": val, "ic95_par_page": bp["ic95"], "mde_observee_par_page": bp["mde"],
                                            "ic95_par_ligne": bl["ic95"], "lecture": sens(bp["ic95"])}
        res["titre"] = titre
        return res

    primaire = [r for r in rows if r["type"] == "presse" and r["niveau"] == "marcel_corrige"]
    crit = bloc(primaire, "CRITÈRE PRIMAIRE : presse relue (marcel_corrige)")
    if not verdict_ok:
        for k in ("K1", "K2", "K1_alto_livre", "K2_alto_livre"):
            crit[k]["lecture"] = "AUCUN VERDICT (passage interrompu)"
    for k in ("K1_alto_livre", "K2_alto_livre"):
        crit[k]["lecture"] = "rapporté à côté (le verdict pré-enregistré porte sur le texte brut) : " + crit[k]["lecture"]
    crit["K1"]["note_M3"] = "K1 positif rapporté « sous l'hypothèse que la circularité VLM ne favorise pas le guidé » (M3 pT)"
    descriptifs = {f"{t}/{lv}": bloc([r for r in rows if r["type"] == t and r["niveau"] == lv], f"descriptif {t} {lv}")
                   for t in ("presse", "livre") for lv in ("marcel_corrige", "bbvlm_sample_validated")
                   if any(r["type"] == t and r["niveau"] == lv for r in rows)}
    descriptifs["presse/toutes"] = bloc([r for r in rows if r["type"] == "presse"], "descriptif presse, tous niveaux")
    pertes_export = {}
    for v in export.values():
        for c, n in v.get("pertes", {}).items():
            pertes_export[c] = pertes_export.get(c, 0) + n
    export_res = {"exports": len(export), "refuses": sum(not v["accepte"] for v in export.values()), "pertes": pertes_export,
                  "refus": {k: v.get("refus") for k, v in export.items() if not v["accepte"]}}
    # ---------- zones vides
    zones = [it for it in items if it["part"] == "vide"]
    diff_ = {d["stem"]: d["nature"] for d in excl_rep.get("difficiles", [])}
    zr = []
    for it in zones:
        t = textes(it)
        zr.append({"stem": it["stem"], "page": it["page_key"], "famille": it["famille"], "cote": it["cote"],
                   "type": it["type_page"], "abst": it["abstention"], "difficile": diff_.get(it["stem"]),
                   "inv": {s: invented(t[s], NEW) for s in SYS},
                   "inv_sans_abst": invented(sorties.get(it["stem"], {}).get("guide_l0.3", {}).get("clean", ""), NEW)})

    def taux(sel, cle="gele"):
        k = sum(z["inv"][cle] if cle in z["inv"] else z[cle] for z in sel)
        n = len(sel)
        return {"k": k, "n": n, "taux": k / n if n else None, "ic95_cp": cp(k, n),
                # décision pA du 23/10 : verdict = BORNE HAUTE de Clopper-Pearson à 95 % ≤ 5 %, si n ≥ 100 ; part rapportée
                "repere_borne_cp_5pc": ((cp(k, n)[1] <= SEUIL_ZONES) if n >= N_VERDICT else "n < 100 : pas de repère")}
    def critere_zones(zr):
        zpages = sorted({z["page"] for z in zr})
        zbp = {p: [z for z in zr if z["page"] == p] for p in zpages}

        def stat_z(tirage):
            sel = [z for p in tirage for z in zbp[p]]
            return sum(z["inv"]["gele"] for z in sel) / len(sel) if sel else None
        crit_z = taux(zr)
        ic_pages = boot(zpages, stat_z)["ic95"]
        crit_z["verdict"] = verdict_zones(crit_z["n"], crit_z["ic95_cp"], ic_pages, verdict_ok)
        crit_z.pop("repere_borne_cp_5pc")
        crit_z["ic95_par_page"] = ic_pages
        crit_z["portee"] = ("critère NOMINAL sous hypothèse binomiale (zones indépendantes) ; sans garantie générale sous "
                            "dépendance entre zones d'une même page : l'IC par page et la règle « fragile » ne détectent pas "
                            "une dépendance forte quand aucune zone n'invente (C1 pT)")
        crit_z["pages"] = len(zpages)
        crit_z["pages_avec_zone_non_abstenue"] = sum(any(not z["abst"] for z in zbp[p]) for p in zpages)
        crit_z["pages_avec_zone_inventee"] = sum(any(z["inv"]["gele"] for z in zbp[p]) for p in zpages)
        return crit_z
    crit_z = critere_zones(zr)
    # SENSIBILITÉ (pA 25/10, PREREG « DISJONCTION PAR NOM » et « DISJONCTION PAR CONTENU ») : le résultat principal AVEC les pages
    # par nom (resp. par contenu) (verdict) et SANS elles (même calcul sur les autres pages) ; un écart de lecture est écrit en
    # tête, sans changer le verdict ; les deux sont calculées sur le résultat principal AVANT toute étiquette FRAGILE
    def categorie(v):   # PASSE, ÉCHOUE, SOUS LE PLANCHER ou AUCUN VERDICT : sans effectif ni annotation « — fragile »
        return v.split(" —")[0].split(" (")[0]

    # pA 25/10 : si AVEC et SANS CONCLUENT différemment (passe d'un côté, échoue de l'autre : « sens confirmé » ou non pour
    # K1/K2 ; PASSE / ÉCHOUE pour les zones, un côté sous le plancher n'est pas une conclusion), le verdict est FRAGILE
    def conclut(v):
        return {"PASSE": True, "ÉCHOUE": False}.get(categorie(v))

    def sensibilite_sans(pages_x, mention, quoi):
        sans_crit = bloc([r for r in primaire if r["page"] not in pages_x], f"SENSIBILITÉ : critère primaire SANS les pages {quoi}")
        sans_z = critere_zones([z for z in zr if z["page"] not in pages_x])
        ecarts = [k for k in ("K1", "K2") if sans_crit[k]["lecture"] != crit[k]["lecture"]] + \
            (["zones"] if categorie(sans_z["verdict"]) != categorie(crit_z["verdict"]) else [])   # catégorie, pas l'effectif (M1 pT)
        fragiles = [k for k in ("K1", "K2") if (sans_crit[k]["lecture"] == "sens confirmé") != (crit[k]["lecture"] == "sens confirmé")]
        if None not in (conclut(crit_z["verdict"]), conclut(sans_z["verdict"])) and conclut(crit_z["verdict"]) != conclut(sans_z["verdict"]):
            fragiles.append("zones")
        return {"mention": mention, f"pages_{quoi.replace(' ', '_')}": sorted(pages_x), "verdicts_fragiles": fragiles,
                f"pages_{quoi.replace(' ', '_')}_dans_le_primaire": len({r["page"] for r in primaire} & pages_x),
                f"zones_des_pages_{quoi.replace(' ', '_')}": sum(z["page"] in pages_x for z in zr),
                "critere_primaire_sans": sans_crit, "zones_sans": sans_z,
                "ecart_de_lecture": "sans objet (passage interrompu)" if not verdict_ok else (f"ÉCART entre AVEC et SANS les pages {quoi} : {', '.join(ecarts)} (le verdict reste "
                                     f"celui AVEC elles)" if ecarts else "aucun")}
    sensibilite = sensibilite_sans(pages_nom, MENTION_NOM, "par nom")
    sensibilite_c = sensibilite_sans(pages_contenu, MENTION_CONTENU, "par contenu")
    for sens_x, etiquette_x in ((sensibilite, FRAGILE_NOM), (sensibilite_c, FRAGILE_CONTENU)):
        for k in sens_x["verdicts_fragiles"]:
            if k == "zones":
                crit_z["verdict"] += " — " + etiquette_x
            else:
                crit[k]["lecture"] += " — " + etiquette_x
    zdesc = {"sans_abstention_guide": taux(zr, "inv_sans_abst"), "dots_libre": taux(zr, "libre"), "a1_seul": taux(zr, "a1"),
             "a1_avec_abstention": taux(zr, "a1_abst"), "part_abstenue": {"k": sum(z["abst"] for z in zr), "n": len(zr),
                                                                           "ic95_cp": cp(sum(z["abst"] for z in zr), len(zr))},
             "par_famille": {f: taux([z for z in zr if z["famille"] == f]) for f in ("F1", "F2")},
             "par_cote": {c: taux([z for z in zr if z["cote"] == c]) for c in sorted({z["cote"] for z in zr})},
             "par_type_page": {t: taux([z for z in zr if z["type"] == t]) for t in sorted({z["type"] for z in zr})},
             "difficiles": {n: taux([z for z in zr if z["difficile"] == n]) for n in ("tache", "transparence", "trou")}}
    rep = {"reserve_B4": RESERVE_B4, "etiquette": ETIQUETTE, "statut": statut,
           "sensibilite_disjonction_par_nom_ecart": sensibilite["ecart_de_lecture"],
           "verdict_fragile_disjonction_par_nom": ([f"{k} : {FRAGILE_NOM}" for k in sensibilite["verdicts_fragiles"]] or "non"),
           "sensibilite_disjonction_par_contenu_ecart": sensibilite_c["ecart_de_lecture"],
           "verdict_fragile_disjonction_par_contenu": ([f"{k} : {FRAGILE_CONTENU}" for k in sensibilite_c["verdicts_fragiles"]] or "non"),
           "lancement_S4": {k: run.get(k) for k in ("dtype", "device_map", "gpu", "secondes", "interrompu", "fins_anormales",
                                                     "torch", "transformers", "lignes_prevues", "lignes_traitees")},
           "temoin_A1": a1m["comptes"],
           "critere_primaire": crit, "zones_vides": {"critere": crit_z, "descriptif": zdesc},
           "sensibilite_disjonction_par_nom": sensibilite,
           "sensibilite_disjonction_par_contenu": sensibilite_c,
           "exclusions": {k: excl_rep[k] for k in ("zones", "exclues", "retenues", "exclues_par_motif",
                                                    "exclues_par_motif_type_famille", "retenues_par_type_famille",
                                                    "retenues_par_page", "pages_avec_zone_retenue", "plancher",
                                                    "difficiles_par_nature")},
           "o6_preuves_disjonction_titres": {c: {"titres_collection": v["preuve_titres_collection_sha256"],
                                                  "titre_document_par_page": {x["page"]: x["preuves_titre_document_sha256"]
                                                                              for x in v["pages"]}}
                                             for c, v in build_rep["o6_titres_disjoints"].items()},
           "o6_preuves_disjonction_par_nom": {"mention": MENTION_NOM,
                                              "collections": {c: {"noms_collection": v["preuve_noms_collection_sha256"],
                                                                  "methode": v["methode"],
                                                                  "titre_document_par_page": {x["page"]: x["preuves_titre_document_sha256"]
                                                                                              for x in v["pages"]}}
                                                              for c, v in build_rep["o6_titres_disjoints_nom"]["collections"].items()}},
           "o6_preuves_disjonction_par_contenu": {"mention": MENTION_CONTENU,
                                                  "collections": {c: {"empreinte_textes": v["empreinte_textes"], "unites": v["unites"],
                                                                      "preuve_compte": v["preuve_compte_sha256"],
                                                                      "vt_par_page": {x["page"]: x["vt_sha256"] for x in v["pages"]}}
                                                                  for c, v in build_rep["o6_contenus_disjoints"]["collections"].items()}},
           "constructeur": {k: build_rep.get(k) for k in ("pages", "pages_gardees", "pages_refusees", "pages_refusees_union",
                                                         "o6_titres_disjoints", "o6_titres_disjoints_nom", "o6_contenus_disjoints",
                                                         "livres_refuses", "lignes", "lignes_par_type_et_niveau",
                                                         "pages_par_type_et_niveau", "pages_presse_relues",
                                                         "zones_total", "zones_attendues", "zones_par_page")},
           "descriptifs_non_produits": ("D-150, couches et « par source » (PREREG §4) ne sont PAS calculés par ce script ; "
                                        "s'ils sont produits, ce sera après ce rapport, étiquetés SECOND REGARD"),
           "descriptifs": descriptifs, "export_alto": export_res,
           "entrees": {"build": a.build_manifest_sha256, "exclusion": a.exclusion_manifest_sha256, "s4": a.s4_manifest_sha256,
                       "a1": a.a1_manifest_sha256, "gel": FREEZE_SHA, "script": sha(Path(__file__)), "gel_fichiers": entrees_gel}}
    (out / "r7a_rapport.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    planche(out, items, sources, zr, rows, textes)
    man = {"entrees": rep["entrees"],
           "sorties": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}}  # sous-dossier alto/ compris
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"statut": statut, "K1": crit["K1"], "K2": crit["K2"], "zones": crit_z}, indent=1, ensure_ascii=False))


def planche(out, items, sources, zr, rows, textes):
    """Planche de CONTRÔLE : toutes les zones où le gelé invente (40 au plus), puis 20 lignes primaires tirées au hasard."""
    from PIL import Image, ImageDraw, ImageFont

    from axel.image.crops import crop_line

    police = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")  # accents, ligatures, ſ (police système macOS)
    font = ImageFont.truetype(str(police), 14) if police.exists() else ImageFont.load_default(size=14)
    by = {it["stem"]: it for it in items}
    inv = [z["stem"] for z in zr if z["inv"]["gele"]][:40]
    prim = sorted(r["stem"] for r in rows if r["type"] == "presse" and r["niveau"] == "marcel_corrige")
    tir = random.Random("r7a:s5:planche").sample(prim, min(20, len(prim)))
    choix = [("ZONE inventée", s) for s in inv] + [("ligne", s) for s in tir]
    sheet = Image.new("L", (1800, 110 * len(choix) + 10), 255)
    d = ImageDraw.Draw(sheet)
    pages = {}
    for k, (genre, s) in enumerate(choix):
        it = by[s]
        img = sources[it["page_key"]]["image"]
        if img not in pages:
            pages.clear()
            with Image.open(img) as im:
                pages[img] = im.convert("L")
        c = crop_line(pages[img], tuple(it["box"]), it.get("polygon"), pad=4).image.copy()
        c.thumbnail((900, 50))
        y = 110 * k + 5
        t = textes(it)
        sheet.paste(c, (5, y))
        d.text((5, y + 55), f"#{k} {genre} {s[-50:]}  sigma={it['sigma']:.1f}", fill=0, font=font)
        d.text((920, y), f"vérité : {it.get('truth', '')[:80]!r}", fill=0, font=font)
        d.text((920, y + 18), f"libre  : {t['libre'][:80]!r}", fill=0, font=font)
        d.text((920, y + 36), f"gelé   : {t['gele'][:80]!r}", fill=0, font=font)
        d.text((920, y + 54), f"A1     : {t['a1'][:80]!r}", fill=0, font=font)
    sheet.save(out / "planche_s5.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="etape", required=True)
    e = sp.add_parser("extraire")
    e.add_argument("--s3-sortie", required=True)
    e.add_argument("--s3-manifest-sha256", required=True)
    e.add_argument("--timing")
    e.add_argument("--ctc")
    e.add_argument("--sortie", required=True)
    r = sp.add_parser("rapport")
    for n in ("build", "build-manifest-sha256", "exclusion", "exclusion-manifest-sha256", "s4-sortie", "s4-manifest-sha256",
              "a1", "a1-manifest-sha256", "sortie"):
        r.add_argument(f"--{n}", required=True)
    a = ap.parse_args()
    (extraire if a.etape == "extraire" else rapport)(a)


if __name__ == "__main__":
    main()
