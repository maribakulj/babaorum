"""R7(a) — constructeur, étapes S0 à S2 (docs/READER_R7A_PREREG.txt, docs/READER_R7A_BUILD_DESIGN.txt ; CLASSE B ;
écrit SANS DONNÉES : rien ne tourne sur de vraies pages avant leur livraison, la relecture Sonnet et l'audit pT).
v2 (22/10) : corrections du NO-GO pT du 21/10 (B1-B5, M1-M2 ; ~/Documents/reviews/babaorum-audit-pT/2026-10-21-R7a-S0-S2.txt).

    ~/outils-seg/dots-hf-env/bin/python scripts/reader_r7a_build.py --panel <panel.jsonl> --liste <liste_v1.json> \\
        --liste-sha256 <SHA publié avec la livraison REF-GALLICA> --sortie <dossier vide>

liste_v1.json (livraison REF-GALLICA, épinglée par --liste-sha256) : liste de 40 objets, schéma STRICT, chacun avec
  page_key (canonique : page_identity(page_key, NS[type]).page == page_key), type (presse | livre),
  image_sha256, alto_sha256, truth_sha256 (64 hex : les SOURCES de la page, B2), truth_level (statut de relecture de la
  PAGE, B4 : marcel_corrige | bbvlm_sample_validated), et g0 (enveloppe G0 « suivi », B1, ci-dessous).
panel.jsonl (une ligne par ligne de texte) : stem, page_key, type, image, image_sha256, alto, alto_sha256, truth_file,
  truth_sha256, truth_id, box [x0, y0, x1, y1], polygon (null ou ≥ 3 points), truth (texte non vide), truth_level.
truth_file (VT de la page, épinglée par truth_sha256 dans la liste ; B4 pT 22/10) : JSON UTF-8
  {"page_key": <clé>, "lines": [{"id": <chaîne non vide, unique>, "text": <chaîne non vide>}, …]}. Règle de conversion
  FIXÉE : l'identité. Chaque ligne du panneau est jointe par truth_id à UNE ligne de la VT, et truth == text EXACTEMENT
  (aucune normalisation) ; les truth_id d'une page sont uniques et couvrent EXACTEMENT les id de sa VT (ni ajout, ni
  oubli). Une VT livrée sous un autre format passe par un convertisseur relu à part (condition de livraison O3).
S0 Contrôles :
   - gel (docs/READER_DOTS_FREEZE_R7A.txt épinglé en entier, ses 9 empreintes revérifiées) ; identité (adaptateur G0-04
     de p1, épinglé en entier ; page_identity et .page/.document, plus _closure et _namespace_of pour O6, comme G0) ;
   - B1 : l'enveloppe G0 de chaque page est RECALCULÉE ici par babaorum.g0_suivi.eligibility (G0-04 étape B, à 6ede0e5)
     sur le registre ancré (g0_collections._disk_registry ; la péremption, SUIVI_REGISTRY_STALE, est un refus du
     recalcul) ; on exige role == "suivi", allowed is True, refusals == refusal_codes == [], supplied.page == page_key
     (sinon arrêt) et l'égalité EXACTE avec l'enveloppe déclarée par la liste (usage_registry_version comprise) ;
     preuves archivées (g0_preuves.json). Code G0 (g0_suivi.py, g0_collections.py) et DONNÉES (g0_suivi_anchors.py)
     épinglés à part ; ancres B1 remplies au 24/10 (6ede0e5) ; toute nouvelle version des
     données se repince par un lot relu ; les étiquettes LIMIT:* que G0 ajoute passent la règle O6 ; aucun contournement ;
   - B2 : par page_key, UN SEUL jeu (image, image_sha256, alto, alto_sha256, truth_file, truth_sha256) sur toutes ses
     lignes, égal aux empreintes de la liste ; chaque fichier est haché ;
   - B3 : types fixés (tiges : chaîne non vide hors de l'espace réservé « r7a-vide: » ; truth : texte non vide ; box :
     4 nombres finis, x0 < x1, y0 < y1, dans l'image ; polygon : null ou ≥ 3 points finis dans l'image) ; ALTO : une
     seule Page, dimensions > 0, éléments lus DANS cette Page seulement, géométries finies (hors Page : O7) ;
   - B4 : toutes les lignes d'une page portent le truth_level de la PAGE dans la liste ; « page relue » = page dont la
     liste dit marcel_corrige, raccordée à son fichier de VT haché ;
   - O6 et O7 (pA 22/10, prudence confirmatoire) : une page dont l'enveloppe G0 porte une étiquette autre que LIMIT:
     (ISSUE_SHARED compris : numéro partagé = risque d'exposition), ou dont l'ALTO a un élément hors de la Page (sans
     tolérance), est REFUSÉE ; O6 admet aussi TITLE_DISJOINT:<c> (pA 25/10, AV-G0-SUIVI-TITRES) si <c> est une collection
     du registre ancré marquée incomplète ET si les sources épinglées portent sa preuve de disjonction ; pages admises
     ainsi rapportées par collection avec les DEUX preuves (titres de la collection ; titre du document de la page)
     (r7a_build.json, o6_titres_disjoints) ; O6 admet AUSSI TITLE_DISJOINT_NAME:<c> (pA 25/10, AV-G0-SUIVI-TITRES-NOMS),
     aux mêmes conditions (collection incomplète, preuve des NOMS épinglée avec sa méthode) : pages rapportées À PART,
     « disjonction par NOM, preuve plus faible » (r7a_build.json, o6_titres_disjoints_nom) ; O6 admet AUSSI
     CONTENT_DISJOINT:<c> (pA 25/10, AV-G0-SUIVI-CONTENU), aux mêmes conditions (collection incomplète, TOUS ses textes
     épinglés dans content_texts) : pages rapportées À PART avec l'empreinte des textes, la preuve du compte et le sha de
     la VT de chaque membre Gallica, contrôle ngram8_content relu, VT contrôlée = VT lue (page_text.sha256 == truth_sha256,
     décision pA M1) (r7a_build.json, o6_contenus_disjoints) ; chaque règle est jugée sur toutes les pages et comptée à part (r7a_build.json,
     pages_refusees), pour vérifier le plancher de 200 zones ; les pages refusées n'entrent ni en S1 ni en S2 ;
   - LIVRES (pM, pA 22/10) : l'espace gallica_book est réservé (adaptateur 431b47f1) ; toute page de type livre est
     REFUSÉE avec le code LIVRE_NON_ADMIS:gallica_book_reserve, sans identité ni G0, ses lignes écartées et comptées
     (r7a_build.json, livres_refuses) ; les livres n'entrent qu'avec un avenant audité (READER_R7A_PREREG) ;
   - appartenance : aucune page NI aucun numéro de press/dev, press/train (axel data/prepared/press), espace de noms
     explicite par ligne (namespace_of).
S1 Recadrages comme les lecteurs (axel crop_line, polygone, pad 4), MD5 des pixels, σ ; abstention prévue = σ < τ du gel.
S2 Zones vides (READER_R7A_PREREG, « ZONES VIDES » et « AJUSTEMENT ») : vide = hors de toute boîte de l'ALTO GALLICA de la
   page (String, TextLine, TextBlock, Illustration, GraphicalElement : les cinq types du pré-enregistrement, M1 pT ;
   ComposedBlock n'est PAS lu), élargie de 8 px ; forme h × w médiane des TextLine ; F1 marges (3 haut, 3 bas, bande
   ≤ 3 h) ; F2 blancs intérieurs (6, dans le rectangle imprimé = union des TextBlock et Illustration) ;
   Random("r7a:vide:<famille>:<page_key>") ; 500 essais par zone, manque compté. Tiges des zones
   « r7a-vide:<page_key>:<k> » ; unicité de TOUS les items vérifiée après S2 (B3).
Sorties (B5), dans cet ordre : items.jsonl ; sources.json (par page : image, ALTO, VT et leurs SHA contrôlés) ;
   g0_preuves.json ; zones/<n>.png (recadrage pleine résolution de chaque zone, M2) et contact.json (vignette n → tige,
   page, famille, fichier) ; contact.png ; r7a_build.json (comptes par type ET truth_level, pages relues, zones par page,
   gel, entrées) ; MANIFEST.json EN DERNIER, sur tous ces fichiers et les SHA des entrées (panel, liste, constructeur,
   pré-enregistrement, conception, gel, identité, G0, recadrage). reader_r7a_exclure.py consommera cet état.
Les étapes S3 (treillis A1, Kaggle T4 fp16), S4 (lecture dots) et S5 (rapport) viendront avec les pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
R = HERE / "reports"
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))
FIELDS = ("stem", "page_key", "type", "image", "image_sha256", "alto", "alto_sha256", "truth_file", "truth_sha256",
          "truth_id", "box", "polygon", "truth", "truth_level")
LISTE_FIELDS = ("page_key", "type", "image_sha256", "alto_sha256", "truth_sha256", "truth_level", "g0")
SOURCES = ("image", "image_sha256", "alto", "alto_sha256", "truth_file", "truth_sha256")
TYPES, LEVELS = ("presse", "livre"), ("marcel_corrige", "bbvlm_sample_validated")
NS = {"presse": "gallica", "livre": "gallica_book"}  # espace de noms de l'adaptateur de p1 par type de page
LIVRE_REFUS = "LIVRE_NON_ADMIS:gallica_book_reserve"  # pM/pA 22/10 : livres refusés tant qu'aucun avenant audité ne les ouvre
F1_COTES, F2_N, ESSAIS, MARGE, BANDE = ("haut",) * 3 + ("bas",) * 3, 6, 500, 8, 3
ALTO_TAGS = ("String", "TextLine", "TextBlock", "Illustration", "GraphicalElement")  # M1 pT : les cinq du pré-enregistrement
VIDE = "r7a-vide:"  # espace de noms réservé aux tiges des zones (B3)
HEX64 = re.compile(r"[0-9a-f]{64}")

FREEZE_SHA = "53e17a2bcb78133315ac4d309d03e39ae71e37d7c4fc8a53ab410bad182e42e7"  # docs/READER_DOTS_FREEZE_R7A.txt
IDENTITY_SHA = "431b47f1850788b317c773e7731b48fb914c20335678b3d8b1e289b390f2e1a9"  # g0_suivi_identity.py (page_identity), GO pT 22/10
G0_SHA = "41a016adaeec9d397177f857254baf47a686d1df7198a9ac977052dbd4810cad"  # g0_suivi.py à d5ce34c (v4 : étape B + avenants audités, TITRES, -ESPACES, -NOMS v2, -GROUPES, CONTENU)
G0_COLLECTIONS_SHA = "57fb85bb25f70fe97bd53f550c0f6e802f10e33fbd832345f74ff4d21aa2ceac"  # g0_collections.py à 6ede0e5 (registre ancré)
IDENTITY_CORE_SHA = "2dacd640bcab6d3098dcbca91aba3af94fbaefa30157c4b1382bef2af71d5a43"  # identity.py (fonctions Gallica/KB strictes, importées par l'adaptateur et G0)
G0_ANCRES_SHA = "30db662c15567b30d6fe4a4abf6cdc86ea585303fe95a56172498163f78faaa2"  # g0_suivi_anchors.py : DONNÉES B1 (registre
# v3, f9df7cc) ; à repincer, par un lot relu, à chaque nouvelle version des données


TITRE_DISJOINT = "TITLE_DISJOINT:"
TITRE_DISJOINT_NOM = "TITLE_DISJOINT_NAME:"   # AV-G0-SUIVI-TITRES-NOMS : preuve plus faible, rapportée à part (pA 25/10)
MENTION_NOM = "disjonction par NOM, preuve plus faible"
CONTENU_DISJOINT = "CONTENT_DISJOINT:"   # AV-G0-SUIVI-CONTENU : rapportée à part comme les noms (pA 25/10)
MENTION_CONTENU = "disjonction par CONTENU (aucun 8-gramme commun mesuré avec tous les textes épinglés de la collection)"


class BuildError(Exception):
    pass


class HorsPage(BuildError):
    """Règle O7 (pA 22/10) : un élément ALTO hors de la Page fait refuser la PAGE, sans tolérance ; refus compté."""


def need(cond, msg) -> None:
    """Contrôle qui survit à python -O (B2 Sonnet)."""
    if not cond:
        raise BuildError(msg)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_alto(path: Path, img_size):
    """Boîtes ALTO (image px) par type ; une seule Page, lue seule ; échelle = taille image / taille Page (B3)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    loc = lambda e: e.tag.rsplit("}", 1)[-1]  # noqa: E731
    pages = [e for e in root.iter() if loc(e) == "Page"]
    need(len(pages) == 1, f"ALTO à {len(pages)} Page : {path}")
    try:
        pw, ph = float(pages[0].get("WIDTH")), float(pages[0].get("HEIGHT"))
    except (TypeError, ValueError) as exc:
        raise BuildError(f"Page sans dimensions : {path}") from exc
    need(finite(pw) and finite(ph) and pw > 0 and ph > 0, f"dimensions de Page invalides : {path}")
    sx, sy = img_size[0] / pw, img_size[1] / ph
    need(abs(sx - sy) / sx < 0.02, f"échelles ALTO incohérentes {sx:.3f} {sy:.3f} : {path}")
    boxes = {t: [] for t in ALTO_TAGS}
    for e in pages[0].iter():
        t = loc(e)
        if t in boxes:  # mineur pM : un élément des cinq types sans géométrie arrête, il n'est plus ignoré
            try:
                x, y, w, h = (float(e.get(k)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
            except (TypeError, ValueError) as exc:
                raise BuildError(f"{t} sans géométrie : {path}") from exc
            need(all(finite(v) for v in (x, y, w, h)) and w >= 0 and h >= 0, f"{t} géométrie invalide : {path}")
            if not (x >= 0 and y >= 0 and x + w <= pw and y + h <= ph):
                raise HorsPage(f"{t} hors de la Page ({x}, {y}, {w}, {h})")
            boxes[t].append([int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)])
    return boxes, [0, 0, img_size[0], img_size[1]]


def freeze_ok() -> dict:
    """Empreintes du gel : chaque ligne « <sha256 64 hex>  <chemin> » de docs/READER_DOTS_FREEZE_R7A.txt est revérifiée."""
    raw = (HERE / "docs/READER_DOTS_FREEZE_R7A.txt").read_bytes()
    need(hashlib.sha256(raw).hexdigest() == FREEZE_SHA, "document du gel modifié")
    out = {}
    for line in raw.decode().splitlines():
        m = re.match(r"\s+([0-9a-f]{64})\s+(\S+)", line)
        if m:
            need(sha(HERE / m.group(2)) == m.group(1), f"gel rompu : {m.group(2)}")
            out[m.group(2)] = m.group(1)
    need(len(out) == 9, f"{len(out)} empreintes de gel lues, 9 attendues")
    return out


def g0_pieces() -> None:
    """Code (g0_suivi.py, g0_collections.py) et DONNÉES (ancres) de G0 épinglés séparément."""
    for rel, h in (("src/babaorum/g0_suivi.py", G0_SHA), ("src/babaorum/g0_collections.py", G0_COLLECTIONS_SHA),
                   ("src/babaorum/g0_suivi_anchors.py", G0_ANCRES_SHA),
                   ("src/babaorum/identity.py", IDENTITY_CORE_SHA)):
        need(sha(HERE / rel) == h, f"pièce G0 modifiée : {rel}")


def g0_suivi(pk: str, document) -> dict:
    """B1 : enveloppe G0 RECALCULÉE : babaorum.g0_suivi.eligibility(registre G0 ancré = g0_collections._disk_registry(),
    "suivi", page_key canonique, issue_key = .document). Pièces épinglées (g0_pieces) ; une entrée invalide, un numéro
    incohérent ou un registre dont l'ancre a changé = arrêt."""
    g0_pieces()
    from babaorum import g0_collections as g0
    from babaorum import g0_suivi as gs

    try:
        return gs.eligibility(g0._disk_registry(), "suivi", page_key=pk, issue_key=document)
    except (ValueError, g0.CollectionRegistryError) as exc:  # SuiviInputError hérite de ValueError
        raise BuildError(f"G0 « suivi » : {type(exc).__name__} : {exc} ({pk})") from exc


def namespace_of(ln) -> str:
    """Espace de noms EXPLICITE d'une ligne de press (pas d'essai successif) : bnl_pack → bnl ; NewsEye/READ groupe
    photo_1983 → newseye_photo ; autres NewsEye/READ (BnF) → gallica."""
    if ln["corpus"] == "bnl_pack":
        return "bnl"
    if ln["corpus"] == "newseye_read":
        return "newseye_photo" if ln["group"] == "photo_1983" else "gallica"
    raise BuildError(f"corpus de press sans espace de noms : {ln['corpus']}")


def press_identities():
    """Pages et numéros de press/train et press/dev par l'adaptateur G0-04 (page_identity, p1) ; comptes par espace."""
    from babaorum.g0_suivi_identity import page_identity

    pages, docs, par_espace = set(), set(), {}
    for sp in ("train", "dev"):
        for x in open(AXEL / f"data/prepared/press/{sp}.jsonl"):
            ln = json.loads(x)
            ident = page_identity(ln["page"], namespace_of(ln))  # IdentityError = arrêt
            pages.add(ident.page)
            if ident.document is not None:
                docs.add(ident.document)
            par_espace.setdefault(ident.namespace, set()).add(ident.page)
    return pages, docs, {k: len(v) for k, v in par_espace.items()}


def overlaps(a, b, m=4) -> bool:
    return not (a[2] + m <= b[0] or b[2] + m <= a[0] or a[3] + m <= b[1] or b[3] + m <= a[1])


def check_liste(liste):
    """Schéma strict de liste_v1 ; renvoie ({page_key: (entrée, identité)} des pages de presse, {page_key: code} des livres).
    Livres (pM, pA 22/10) : l'espace gallica_book est réservé et refusé par l'adaptateur tant que Marcel n'a pas tranché
    REF-GALLICA ; chaque page de livre est REFUSÉE avec le code LIVRE_REFUS, sans appel d'identité ni G0, et comptée."""
    from babaorum.g0_suivi_identity import page_identity
    from babaorum.identity import IdentityError

    need(isinstance(liste, list) and len(liste) == 40, "liste_v1 : une liste de 40 entrées attendue")
    ok, livres = {}, {}
    for e in liste:
        need(isinstance(e, dict) and set(LISTE_FIELDS) <= set(e), f"entrée de liste incomplète : {e}")
        pk = e["page_key"]
        need(isinstance(pk, str) and pk and e["type"] in TYPES and e["truth_level"] in LEVELS, f"clé, type ou niveau : {pk}")
        need(pk not in ok and pk not in livres, f"page en double dans la liste : {pk}")
        if e["type"] == "livre":
            livres[pk] = LIVRE_REFUS
            continue
        try:
            ident = page_identity(pk, NS[e["type"]])
        except IdentityError as exc:
            raise BuildError(f"clé de page refusée par l'identité ({NS[e['type']]}) : {pk}") from exc
        need(ident.page == pk, f"clé non canonique : {pk}")
        need(all(isinstance(e[k], str) and HEX64.fullmatch(e[k]) for k in ("image_sha256", "alto_sha256", "truth_sha256")),
             f"empreintes de sources invalides : {pk}")
        ok[pk] = (e, ident)
    need(len(ok) >= 20, "liste : moins de 20 pages de presse")
    return ok, livres


def check_g0(pk, entree, ident) -> dict:
    """B1 : enveloppe déclarée == enveloppe recalculée, rôle suivi, admise, sans refus, page concernée (sinon arrêt).
    Les étiquettes sont jugées à part (règle O6, label_refus)."""
    decl, calc = entree["g0"], g0_suivi(pk, ident.document)
    need(isinstance(calc, dict) and calc.get("role") == "suivi" and calc.get("allowed") is True
         and calc.get("refusals") == [] and calc.get("refusal_codes") == [], f"G0 « suivi » refuse : {pk} {calc.get('refusal_codes')}")
    need((calc.get("supplied") or {}).get("page") == pk, f"G0 : résultat d'une autre page : {pk}")
    need(isinstance(calc.get("labels"), list) and all(isinstance(t, str) for t in calc["labels"]), f"étiquettes G0 : {pk}")
    need(decl == calc, f"enveloppe G0 de la liste différente du recalcul : {pk}")
    return calc


def g0_disjonctions() -> dict:
    """O6 (pA 25/10) : les DEUX preuves épinglées de TITLE_DISJOINT, lues dans les octets des sources que lit eligibility
    (_documents(load_inputs()), ancres du module) : "collections" = {collection du registre ancré marquée incomplète
    (gallica_scope possible, clés de numéro incomplètes, même règle que G0) : sha de la preuve de ses titres, ou None} ;
    "methodes" = {même collection : méthode de la preuve si la table la donne au niveau des NOMS, sinon None} ;
    "documents" = {document Gallica : sha de la preuve de son titre (cb et/ou titre REF)} ; "classes" = pages d'une même page physique
    (alias prouvés, comme G0) ; "contenus" = {même collection : {empreinte, unites, preuve} de ses content_texts (empreinte =
    lm_fingerprint de G0), ou None} ; "controles_contenu" = {page : son contrôle ngram8_content}. Sources non épinglées = arrêt."""
    g0_pieces()
    from babaorum import g0_collections as g0
    from babaorum import g0_suivi as gs
    from babaorum.g0_suivi_identity import _closure

    try:
        reg, docs = g0._disk_registry(), gs._documents(gs.load_inputs())
        _, aliases, _ = gs._derived(docs)
    except (ValueError, g0.CollectionRegistryError) as exc:
        raise BuildError(f"G0 « suivi » : {type(exc).__name__} : {exc}") from exc
    need(docs["sources"] is not None, "G0 : fichier des sources non épinglé (preuves de disjonction illisibles)")
    need(docs["usage"] is not None, "G0 : fichier des usages non épinglé (alias, donc classes, illisibles)")
    t = docs["sources"].get("title_disjunction") or {"collections": {}, "documents": []}
    inc = [n for n, e in reg["collections"].items() if g0._scope(e) == "possible" and not g0._issue_keys_complete(e)]
    ct = docs["sources"].get("content_texts") or {}
    return {"contenus": {n: ({"empreinte": gs.lm_fingerprint(ct[n]["files"]), "unites": ct[n]["units"],
                              "preuve": ct[n]["proof"]["sha256"]} if n in ct else None) for n in inc},
            "controles_contenu": {c["page"]: c for c in docs["usage"]["checks"] if c["check"] == gs.CONTENT_CHECK},
            "collections": {n: ((t["collections"].get(n) or {}).get("proof") or {}).get("sha256") for n in inc},
            "methodes": {n: (t["collections"].get(n) or {}).get("method") for n in inc},   # niveau NOMS seulement
            "documents": {d["document"]: d["proof"]["sha256"] for d in t["documents"]},
            "classes": _closure((a.left.page, a.right.page) for a in aliases if a.proven_page)}


def preuves_documents(pk, ctx) -> dict:
    """{document : sha de la preuve de son titre} pour chaque membre Gallica de la classe de la page (comme G0) ; une page
    admise par TITLE_DISJOINT sans cette preuve pour l'un d'eux = arrêt (G0 ne devrait pas l'avoir admise)."""
    from babaorum.g0_suivi_identity import _namespace_of, page_identity

    out = {}
    for m in sorted(ctx["classes"].get(pk, {pk})):
        if _namespace_of(m) == "gallica":
            doc = page_identity(m, "gallica").document
            need(doc in ctx["documents"], f"TITLE_DISJOINT(_NAME) sans preuve du titre du document {doc} : {pk}")
            out[doc] = ctx["documents"][doc]
    need(out, f"TITLE_DISJOINT(_NAME) sans membre Gallica : {pk}")
    return out


def label_refus(calc, disj, methodes=None, contenus=None) -> list:
    """Règle O6 (pA 22/10 et 25/10) : sont admises les étiquettes LIMIT:, TITLE_DISJOINT:<c> et TITLE_DISJOINT_NAME:<c> si
    <c> est une collection incomplète du registre dont la preuve de disjonction est épinglée (disj, g0_disjonctions) ; la
    preuve d'un TITLE_DISJOINT_NAME est au niveau des NOMS (méthode donnée), celle d'un TITLE_DISJOINT ne l'est pas, comme
    dans G0 ; CONTENT_DISJOINT:<c> (pA 25/10) si tous les textes de <c> sont épinglés (contenus, empreinte) ; ISSUE_SHARED
    et toute autre font refuser la page. Les étiquettes restent dans l'enveloppe archivée."""
    methodes, contenus = methodes or {}, contenus or {}

    def preuve(c):
        return isinstance(disj.get(c), str) and len(disj[c]) == 64 and all(x in "0123456789abcdef" for x in disj[c])

    def admise(t):
        if t.startswith("LIMIT:"):
            return True
        if t.startswith(TITRE_DISJOINT):
            c = t.removeprefix(TITRE_DISJOINT)
            return preuve(c) and methodes.get(c) is None
        if t.startswith(TITRE_DISJOINT_NOM):
            c = t.removeprefix(TITRE_DISJOINT_NOM)
            return preuve(c) and isinstance(methodes.get(c), str) and bool(methodes[c])
        if t.startswith(CONTENU_DISJOINT):
            v = contenus.get(t.removeprefix(CONTENU_DISJOINT))
            return isinstance(v, dict) and all(isinstance(v.get(k), str) and len(v[k]) == 64
                                               and all(x in "0123456789abcdef" for x in v[k]) for k in ("empreinte", "preuve"))
        return False
    return [t for t in calc["labels"] if not admise(t)]


def titres_disjoints(preuves, ctx, gardees, prefixe=TITRE_DISJOINT) -> dict:
    """O6 (pA 25/10) : par collection, les DEUX preuves (titres, ou NOMS, de la collection ; titre du document de chaque
    membre Gallica de la classe de la page) et les pages GARDÉES admises par l'étiquette `prefixe`<c>."""
    par = {}
    for pk in sorted(gardees):
        for t in preuves[pk]["labels"]:
            if t.startswith(prefixe):
                par.setdefault(t.removeprefix(prefixe), []).append(
                    {"page": pk, "preuves_titre_document_sha256": preuves_documents(pk, ctx)})
    return {c: {"preuve_titres_collection_sha256": ctx["collections"][c], "n_pages": len(v), "pages": v}
            for c, v in sorted(par.items())}


def titres_disjoints_nom(preuves, ctx, gardees) -> dict:
    """O6 (pA 25/10) : pages GARDÉES admises par au moins un TITLE_DISJOINT_NAME, rapportées À PART (preuve plus faible),
    par collection avec la méthode et les deux preuves ; « pages » = liste des pages par nom (sensibilité S5)."""
    par = titres_disjoints(preuves, ctx, gardees, TITRE_DISJOINT_NOM)
    for c, v in par.items():
        v["preuve_noms_collection_sha256"] = v.pop("preuve_titres_collection_sha256")
        v["methode"] = ctx["methodes"][c]
    return {"mention": MENTION_NOM, "collections": par, "pages": sorted({x["page"] for v in par.values() for x in v["pages"]})}


def vt_contenu(pk, c, ctx, truth_sha) -> dict:
    """{membre Gallica de la classe de la page : sha de sa VT BBVLM} pour CONTENT_DISJOINT:<c>, après relecture de son contrôle
    ngram8_content (complet, contre <c> avec l'empreinte exacte de ses textes, aucun constat sur <c>, VT BBVLM), comme G0 ;
    décision pA (25/10, M1) : la VT contrôlée est la VT LUE, page_text.sha256 == truth_sha256 de la page dans la liste (sortie
    du convertisseur), pour la page ET chaque membre Gallica de sa classe ; sinon arrêt (G0 ne devrait pas l'avoir admise ;
    une nouvelle version de la VT impose un nouveau contrôle)."""
    from babaorum.g0_suivi_identity import _namespace_of

    out = {}
    for m in sorted(ctx["classes"].get(pk, {pk})):
        if _namespace_of(m) == "gallica":
            k = ctx["controles_contenu"].get(m)
            need(k is not None and k["status"] == "complete" and c in k["against"]
                 and k["against_manifests"].get(c) == ctx["contenus"][c]["empreinte"]
                 and not any(f["source"] == c for f in k["findings"]) and k["page_text"]["kind"] == "vt_bbvlm",
                 f"CONTENT_DISJOINT:{c} sans contrôle ngram8_content valable pour {m} : {pk}")
            need(k["page_text"]["sha256"] == truth_sha,
                 f"CONTENT_DISJOINT:{c} : la VT contrôlée pour {m} n'est pas la VT lue (truth_sha256 de la liste) : {pk}")
            out[m] = k["page_text"]["sha256"]
    need(out, f"CONTENT_DISJOINT:{c} sans membre Gallica : {pk}")
    return out


def contenus_disjoints(preuves, ctx, gardees, truth) -> dict:
    """O6 (pA 25/10) : pages GARDÉES admises par au moins un CONTENT_DISJOINT, rapportées À PART comme les noms, par collection
    avec l'empreinte de ses textes, son compte d'unités et sa preuve, et le sha de la VT de chaque membre Gallica de la classe
    (= truth_sha256 de la page, truth = {page : truth_sha256}) ;
    « pages » = liste des pages par contenu (sensibilité S5)."""
    par = {}
    for pk in sorted(gardees):
        for t in preuves[pk]["labels"]:
            if t.startswith(CONTENU_DISJOINT):
                c = t.removeprefix(CONTENU_DISJOINT)
                par.setdefault(c, []).append({"page": pk, "vt_sha256": vt_contenu(pk, c, ctx, truth[pk])})
    cols = {c: {"empreinte_textes": ctx["contenus"][c]["empreinte"], "unites": ctx["contenus"][c]["unites"],
                "preuve_compte_sha256": ctx["contenus"][c]["preuve"], "n_pages": len(v), "pages": v} for c, v in sorted(par.items())}
    return {"mention": MENTION_CONTENU, "collections": cols, "pages": sorted({x["page"] for v in cols.values() for x in v["pages"]})}


def check_vt(pk, path: Path, lines) -> None:
    """B4 pT 22/10 : les textes du panneau sont ceux de la VT épinglée (identifiant stable, égalité exacte, couverture exacte)."""
    try:
        vt = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BuildError(f"VT illisible (JSON UTF-8 attendu) : {pk}") from exc
    need(isinstance(vt, dict) and vt.get("page_key") == pk and isinstance(vt.get("lines"), list) and vt["lines"],
         f"VT : format ou page_key : {pk}")
    texts = {}
    for x in vt["lines"]:
        need(isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"] and isinstance(x.get("text"), str)
             and x["text"].strip(), f"VT : ligne invalide : {pk}")
        need(x["id"] not in texts, f"VT : id en double : {pk} {x['id']}")
        texts[x["id"]] = x["text"]
    ids = [r["truth_id"] for r in lines]
    need(len(ids) == len(set(ids)), f"truth_id en double dans le panneau : {pk}")
    need(set(ids) == set(texts), f"panneau et VT ne couvrent pas les mêmes lignes : {pk} "
                                 f"({len(set(ids) - set(texts))} hors VT, {len(set(texts) - set(ids))} absentes)")
    for r in lines:
        need(r["truth"] == texts[r["truth_id"]], f"truth du panneau différent de la VT épinglée : {r['stem']}")


def check_row(r, ok, size_of) -> None:
    """B3 : types et géométries d'une ligne du panneau (size_of(image) = (largeur, hauteur))."""
    need(isinstance(r, dict) and set(FIELDS) <= set(r), f"ligne de panneau incomplète : {str(r)[:120]}")
    s = r["stem"]
    need(isinstance(s, str) and s.strip() == s and s and not s.startswith(VIDE), f"tige invalide ou réservée : {s!r}")
    need(r["page_key"] in ok, f"page hors liste : {r['page_key']}")
    e = ok[r["page_key"]][0]
    need(r["type"] == e["type"], f"type différent de la liste : {s}")
    need(r["truth_level"] == e["truth_level"], f"truth_level différent de celui de la PAGE (B4) : {s}")
    need(isinstance(r["truth"], str) and r["truth"].strip(), f"truth non textuel ou vide : {s}")
    need(isinstance(r["truth_id"], str) and r["truth_id"], f"truth_id invalide : {s}")
    need(all(isinstance(r[k], str) and r[k] for k in ("image", "alto", "truth_file")), f"chemins de sources : {s}")
    W, H = size_of(r["image"])
    b = r["box"]
    need(isinstance(b, list) and len(b) == 4 and all(finite(v) for v in b) and 0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H,
         f"box invalide ou hors image : {s} {b}")
    p = r["polygon"]
    need(p is None or (isinstance(p, list) and len(p) >= 3 and all(
        isinstance(q, list) and len(q) == 2 and all(finite(v) for v in q) and 0 <= q[0] <= W and 0 <= q[1] <= H for q in p)),
         f"polygon invalide ou hors image : {s}")


def main() -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from axel.image.crops import crop_line

    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--liste", required=True)
    ap.add_argument("--liste-sha256", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    out = Path(a.sortie)
    need(not out.exists() or not any(out.iterdir()), "dossier de sortie non vide")

    # S0 — entrées épinglées
    gel = freeze_ok()
    need(sha(HERE / "src/babaorum/g0_suivi_identity.py") == IDENTITY_SHA, "adaptateur d'identité modifié")
    need(sha(HERE / "src/babaorum/identity.py") == IDENTITY_CORE_SHA, "identity.py modifié")
    tau = json.loads((R / "reader_dots_abst_v2.json").read_text())["tau"]
    need(sha(Path(a.liste)) == a.liste_sha256, "liste_v1 différente de la livraison épinglée")
    ok, livres = check_liste(json.loads(Path(a.liste).read_text()))
    preuves = {pk: check_g0(pk, e, ident) for pk, (e, ident) in sorted(ok.items())}
    ctx = g0_disjonctions()
    disj = ctx["collections"]

    # S0 — panneau
    rows = [json.loads(x) for x in open(a.panel)]
    sizes = {}

    def size_of(img):
        if img not in sizes:
            with Image.open(img) as im:
                sizes[img] = im.size
        return sizes[img]

    lignes_livres = sum(isinstance(r, dict) and r.get("page_key") in livres for r in rows)
    rows = [r for r in rows if not (isinstance(r, dict) and r.get("page_key") in livres)]  # lignes des livres refusés
    for r in rows:
        check_row(r, ok, size_of)
    need(len({r["stem"] for r in rows}) == len(rows), "tiges en double")
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page_key"], []).append(r)
    need(set(by_page) == set(ok), "pages de la liste sans ligne au panneau")
    # O6 et O7 (pA 22/10) : chaque règle est jugée sur TOUTES les pages et comptée à part ; refusées = leur union
    refus = {"O6_etiquette_non_admise": {pk: label_refus(c, disj, ctx["methodes"], ctx["contenus"]) for pk, c in preuves.items()
                                         if label_refus(c, disj, ctx["methodes"], ctx["contenus"])},
             "O7_alto_hors_page": {}}
    altos = {}
    sources = {}
    for pk, lines in sorted(by_page.items()):  # B2 : un seul jeu de sources par page, égal à la liste, haché
        jeux = {tuple(r[k] for k in SOURCES) for r in lines}
        need(len(jeux) == 1, f"sources multiples pour la page {pk} : {len(jeux)} jeux")
        src = dict(zip(SOURCES, jeux.pop()))
        e = ok[pk][0]
        for f, h in (("image", "image_sha256"), ("alto", "alto_sha256"), ("truth_file", "truth_sha256")):
            need(src[h] == e[h], f"{h} du panneau différent de la liste : {pk}")
            need(sha(Path(src[f])) == e[h], f"{f} modifié : {pk}")
        sources[pk] = {**src, "type": e["type"], "truth_level": e["truth_level"]}
        check_vt(pk, Path(src["truth_file"]), lines)
        try:
            altos[pk] = read_alto(Path(src["alto"]), size_of(src["image"]))
            need(altos[pk][0]["TextLine"], f"ALTO sans TextLine : {pk}")
        except HorsPage as exc:
            refus["O7_alto_hors_page"][pk] = str(exc)
    refusees = set(refus["O6_etiquette_non_admise"]) | set(refus["O7_alto_hors_page"])
    for pk in refusees:
        sources[pk]["refusee"] = True
    pages_vues, numeros_vus, hors_gallica = press_identities()
    clash = sorted(pk for pk, (_, ident) in ok.items()
                   if pk in pages_vues or (ident.document is not None and ident.document in numeros_vus))
    need(not clash, f"pages ou numéros déjà vus dans press : {clash[:5]}")

    # S1 et S2
    items, zones_count, zones = [], {}, []
    for pk, lines in sorted(by_page.items()):
        if pk in refusees:
            continue
        src = sources[pk]
        with Image.open(src["image"]) as im:
            page = im.convert("L")
        for r in lines:
            crop = crop_line(page, tuple(r["box"]), r["polygon"], pad=4).image
            sig = float(np.asarray(crop, dtype=np.float64).std())
            items.append({"stem": r["stem"], "part": r["type"], "page_key": pk, "image": src["image"], "box": r["box"],
                          "polygon": r["polygon"], "truth": r["truth"], "truth_level": r["truth_level"],
                          "crop_md5": hashlib.md5(crop.tobytes()).hexdigest(), "sigma": sig, "abstention": sig < tau})
        boxes, pg = altos[pk]
        h = int(statistics.median(b[3] - b[1] for b in boxes["TextLine"]))
        w = int(statistics.median(b[2] - b[0] for b in boxes["TextLine"]))
        need(h > 0 and w > 0, f"forme ligne nulle : {pk}")
        obst = [b for t in ALTO_TAGS for b in boxes[t]]
        blocs = boxes["TextBlock"] + boxes["Illustration"] or boxes["TextLine"]
        px0, py0 = min(b[0] for b in blocs), min(b[1] for b in blocs)
        px1, py1 = max(b[2] for b in blocs), max(b[3] for b in blocs)
        band = BANDE * h
        regions = {"haut": (px0, max(pg[1], py0 - band), px1, py0), "bas": (px0, py1, px1, min(pg[3], py1 + band))}
        got, manque = [], 0

        def draw(rng, reg):
            x_lo, y_lo, x_hi, y_hi = reg
            for _ in range(ESSAIS):
                if x_hi - w < x_lo or y_hi - h < y_lo:
                    return None
                x0, y0 = rng.randint(x_lo, x_hi - w), rng.randint(y_lo, y_hi - h)
                c = [x0, y0, x0 + w, y0 + h]
                if not any(overlaps(c, b, MARGE) for b in obst) and not any(overlaps(c, g[0], MARGE) for g in got):
                    return c
            return None
        rng1 = random.Random(f"r7a:vide:F1:{pk}")
        for cote in F1_COTES:
            c = draw(rng1, regions[cote])
            if c:
                got.append((c, "F1", cote))
            else:
                manque += 1
        rng2 = random.Random(f"r7a:vide:F2:{pk}")
        for _ in range(F2_N):
            c = draw(rng2, (px0, py0, px1, py1))
            if c:
                got.append((c, "F2", "interieur"))
            else:
                manque += 1
        zones_count[pk] = {"retenues": len(got), "manque": manque, "h": h, "w": w}
        for k, (b, fam, cote) in enumerate(got):
            crop = crop_line(page, tuple(b), None, pad=4).image
            sig = float(np.asarray(crop, dtype=np.float64).std())
            stem = f"{VIDE}{pk}:{k}"
            items.append({"stem": stem, "part": "vide", "famille": fam, "cote": cote, "type_page": src["type"],
                          "page_key": pk, "image": src["image"], "box": b, "polygon": None, "truth": "",
                          "truth_level": "zone_vide", "crop_md5": hashlib.md5(crop.tobytes()).hexdigest(), "sigma": sig,
                          "abstention": sig < tau})
            zones.append((stem, pk, fam, cote, crop))
    need(len({i["stem"] for i in items}) == len(items), "tiges en double après S2")  # B3

    # Sorties (B5) : planche et preuves AVANT le manifeste, qui vient en dernier
    out.mkdir(parents=True, exist_ok=True)
    (out / "zones").mkdir()
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    (out / "sources.json").write_text(json.dumps(sources, indent=1, ensure_ascii=False))
    (out / "g0_preuves.json").write_text(json.dumps(preuves, indent=1, ensure_ascii=False))
    contact = []
    sheet = Image.new("L", (6 * 410, 60 * ((len(zones) + 5) // 6) + 10), 255)
    d = ImageDraw.Draw(sheet)
    for n, (stem, pk, fam, cote, c) in enumerate(zones):
        c.save(out / "zones" / f"{n}.png")  # pleine résolution, pour les exclusions (M2)
        contact.append({"n": n, "stem": stem, "page_key": pk, "famille": fam, "cote": cote, "fichier": f"zones/{n}.png"})
        t = c.copy()
        t.thumbnail((400, 40))
        x, y = 410 * (n % 6) + 5, 60 * (n // 6) + 5
        sheet.paste(t, (x, y + 14))
        d.text((x, y), f"{n} {pk} {fam}"[:60], fill=0)
    (out / "contact.json").write_text(json.dumps(contact, indent=1, ensure_ascii=False))
    sheet.save(out / "contact.png")
    gardees = {pk: s for pk, s in sources.items() if pk not in refusees}
    relues = sorted(pk for pk, s in gardees.items() if s["truth_level"] == "marcel_corrige" and s["type"] == "presse")
    entrees = {"panel": sha(Path(a.panel)), "liste_v1": a.liste_sha256,
               **{p: sha(HERE / p) for p in ("scripts/reader_r7a_build.py", "docs/READER_R7A_PREREG.txt",
                                             "docs/READER_R7A_BUILD_DESIGN.txt", "docs/READER_DOTS_FREEZE_R7A.txt",
                                             "src/babaorum/g0_suivi_identity.py", "src/babaorum/g0_collections.py",
                                             "src/babaorum/g0_suivi.py", "src/babaorum/g0_suivi_anchors.py",
                                             "src/babaorum/identity.py")},
               "axel/src/axel/image/crops.py": sha(AXEL / "src/axel/image/crops.py")}
    stats = {"lignes": {t: sum(i["part"] == t for i in items) for t in TYPES + ("vide",)},
             "lignes_par_type_et_niveau": {f"{t}/{lv}": sum(i["part"] == t and i["truth_level"] == lv for i in items)
                                           for t in TYPES for lv in LEVELS},
             "pages": {t: sum(s["type"] == t for s in sources.values()) for t in TYPES},
             "pages_par_type_et_niveau": {f"{t}/{lv}": sum(s["type"] == t and s["truth_level"] == lv for s in sources.values())
                                          for t in TYPES for lv in LEVELS},
             "pages_gardees": {t: sum(s["type"] == t for s in gardees.values()) for t in TYPES},
             "pages_refusees": {r: {"n": len(v), "par_type": {t: sum(sources[pk]["type"] == t for pk in v) for t in TYPES},
                                    "pages": v} for r, v in refus.items()},
             "pages_refusees_union": len(refusees),
             "o6_titres_disjoints": titres_disjoints(preuves, ctx, gardees),
             "o6_titres_disjoints_nom": titres_disjoints_nom(preuves, ctx, gardees),
             "o6_contenus_disjoints": contenus_disjoints(preuves, ctx, gardees, {pk: e["truth_sha256"] for pk, (e, _) in ok.items()}),
             "livres_refuses": {"code": LIVRE_REFUS, "n": len(livres), "pages": sorted(livres), "lignes_ecartees": lignes_livres},
             "pages_presse_relues": relues, "zones_par_page": zones_count,
             "zones_total": len(zones), "zones_attendues": 12 * len(zones_count), "tau": tau, "gel": gel,
             "entrees": entrees, "press_pages_par_espace": hors_gallica,
             "abstentions_prevues": {t: sum(i["abstention"] for i in items if i["part"] == t) for t in TYPES + ("vide",)}}
    (out / "r7a_build.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    manifest = {"entrees": entrees,
                "sorties": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("gel", "zones_par_page", "entrees")}, indent=1,
                     ensure_ascii=False))
    if stats["zones_total"] < 200:  # pas d'arrêt (pA) : sous le plancher, IC sans verdict au rapport
        print(f"SOUS LE PLANCHER : {stats['zones_total']} zones avant exclusions ; rapport sans verdict", flush=True)


if __name__ == "__main__":
    main()
