"""REF-GALLICA → R7(a) : convertisseur de livraison de la VT (LOT-REF-GALLICA v2.3 ; accord pA du 24/10 ; relu Sonnet puis pT).

    PYTHONPATH=src python scripts/ref_gallica_vt_convert.py --entree <training.jsonl relu> \\
        --source-bbvlm <training.jsonl livré par BBVLM, avant relecture> --page-key <clé> --type <presse|livre> \\
        --sortie <fichier .json, inexistant>

Entrée (HYPOTHÈSE retenue provisoirement par pA le 24/10, soumise à Marcel) : le training.jsonl d'UNE page livré par
BBVLM, une ligne JSON par ligne de texte {"page_id", "line_id", "text", ...}, dans lequel la relecture ne change que
« text », et peut ajouter « statut » : « supprimee_par_relecteur » (ce n'est pas une ligne de texte : ornement, filet,
fausse détection de kraken). La clé de page est donnée explicitement (--page-key) : le page_id de BBVLM est une empreinte.
CONTRÔLE DU NOMBRE DE LIGNES (v2.3) : le fichier relu doit avoir EXACTEMENT la suite de line_id (ordre compris) et le
page_id du fichier livré par BBVLM avant relecture (--source-bbvlm) : la relecture n'ajoute, ne retire ni ne déplace
aucune ligne ; elle en marque seulement.
Sortie (format de check_vt du constructeur R7(a)) : {"page_key": <clé>, "lines": [{"id": <line_id>, "text": <text>}]}.
Aucun texte n'est touché (ni NFC, ni espaces) ; découpage du fichier sur l'octet « \\n » seulement (un « \\r » final, CRLF,
est retiré ; U+2028, U+2029, U+0085 restent DANS le texte) ; une ligne de fichier vide est ignorée.
Statut : une ligne « supprimee_par_relecteur » est RETIRÉE et COMPTÉE à part (ids au rapport), son texte peut être vide ;
une ligne de texte vide ou blanche SANS ce statut est REFUSÉE (TEXTE_VIDE) ; tout autre statut est refusé.
Refus CODÉS (VTRefus, .code) : BOM_INTERDIT, ENCODAGE_INVALIDE, JSON_INVALIDE, CLE_JSON_DOUBLE, CHAMP_MANQUANT,
LIGNE_SANS_ID, ID_INVALIDE, ID_EN_DOUBLE, PAGE_ID_INVALIDE, PAGE_ID_MULTIPLES, TEXTE_INVALIDE, TEXTE_VIDE, STATUT_INCONNU,
PAGE_VIDE, NB_LIGNES, IDS_DIFFERENTS, PAGE_ID_DIFFERENT, CLE_REFUSEE, CLE_NON_CANONIQUE, TYPE_INCONNU, SORTIE_EXISTANTE,
SORTIE_INVALIDE, ENTREE_ILLISIBLE.
Les lignes retirées ne doivent pas entrer au panneau du constructeur (couverture exacte de check_vt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

NS = {"presse": "gallica", "livre": "gallica_book"}
SUPPRIMEE = "supprimee_par_relecteur"


class VTRefus(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code} : {message}")
        self.code = code


def refus(cond, code, message) -> None:
    if not cond:
        raise VTRefus(code, message)


def cle_canonique(page_key: str, type_page: str) -> str:
    from babaorum.g0_suivi_identity import page_identity
    from babaorum.identity import IdentityError

    refus(type_page in NS, "TYPE_INCONNU", f"type de page inconnu : {type_page!r}")
    try:
        ident = page_identity(page_key, NS[type_page])
    except IdentityError as exc:
        raise VTRefus("CLE_REFUSEE", f"clé refusée par page_identity ({NS[type_page]}) : {page_key!r} ({exc})") from exc
    refus(ident.page == page_key, "CLE_NON_CANONIQUE", f"clé non canonique : {page_key!r} (canonique : {ident.page!r})")
    return ident.page


def _objet(pairs):
    cles = [k for k, _ in pairs]
    refus(len(cles) == len(set(cles)), "CLE_JSON_DOUBLE", f"clé JSON en double : {cles}")
    return dict(pairs)


def lire_lignes(raw: bytes, nom: str) -> list[dict]:
    """Objets JSON d'un training.jsonl, découpé sur l'octet « \\n » seulement (jamais str.splitlines)."""
    refus(not raw.startswith(b"\xef\xbb\xbf"), "BOM_INTERDIT", f"{nom} : marque d'ordre d'octets UTF-8 en tête")
    objets = []
    for k, brut in enumerate(raw.split(b"\n"), 1):
        if brut.endswith(b"\r"):
            brut = brut[:-1]  # fin de ligne CRLF
        if not brut.strip():
            continue  # ligne de FICHIER vide : ce n'est pas une ligne de texte
        try:
            txt = brut.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VTRefus("ENCODAGE_INVALIDE", f"{nom}, ligne {k} : {exc}") from exc
        try:
            r = json.loads(txt, object_pairs_hook=_objet)
        except ValueError as exc:
            raise VTRefus("JSON_INVALIDE", f"{nom}, ligne {k} : {exc}") from exc
        refus(isinstance(r, dict), "JSON_INVALIDE", f"{nom}, ligne {k} : objet attendu")
        r["_ligne"] = k
        objets.append(r)
    return objets


def _ids_et_page(objets, nom) -> tuple[list[str], str]:
    ids, page_ids = [], set()
    for r in objets:
        k = r["_ligne"]
        refus("line_id" in r and r["line_id"] is not None and r["line_id"] != "", "LIGNE_SANS_ID", f"{nom}, ligne {k}")
        refus(isinstance(r["line_id"], str) and r["line_id"].strip() == r["line_id"], "ID_INVALIDE",
              f"{nom}, ligne {k} : line_id doit être une chaîne sans blanc autour : {r['line_id']!r}")
        try:
            r["line_id"].encode("utf-8")
        except UnicodeEncodeError as exc:  # substitut isolé dans l'id (mineur pT) : refus codé, pas d'erreur brute à l'écriture
            raise VTRefus("ENCODAGE_INVALIDE", f"{nom}, ligne {k} : line_id {exc}") from exc
        refus(r["line_id"] not in ids, "ID_EN_DOUBLE", f"{nom} : line_id en double : {r['line_id']!r}")
        refus("page_id" in r, "CHAMP_MANQUANT", f"{nom}, ligne {k} : page_id exigé")
        refus(isinstance(r["page_id"], str) and r["page_id"] != "", "PAGE_ID_INVALIDE", f"{nom}, ligne {k} : {r['page_id']!r}")
        ids.append(r["line_id"])
        page_ids.add(r["page_id"])
    refus(ids, "PAGE_VIDE", f"{nom} : aucune ligne")
    refus(len(page_ids) == 1, "PAGE_ID_MULTIPLES", f"{nom} : plusieurs page_id : {sorted(page_ids)[:3]}")
    return ids, page_ids.pop()


def convertir(raw: bytes, raw_source: bytes, page_key: str, type_page: str) -> tuple[dict, list[str]]:
    """(training.jsonl relu, training.jsonl livré par BBVLM) → (objet VT de R7(a), line_id retirées par le relecteur)."""
    cle = cle_canonique(page_key, type_page)
    relu, source = lire_lignes(raw, "relu"), lire_lignes(raw_source, "source BBVLM")
    ids, page_id = _ids_et_page(relu, "relu")
    ids_src, page_id_src = _ids_et_page(source, "source BBVLM")
    refus(len(ids) == len(ids_src), "NB_LIGNES", f"{len(ids)} lignes relues pour {len(ids_src)} livrées par BBVLM")
    refus(ids == ids_src, "IDS_DIFFERENTS", "la suite des line_id (ordre compris) diffère de la livraison BBVLM")
    refus(page_id == page_id_src, "PAGE_ID_DIFFERENT", f"page_id {page_id!r} au lieu de {page_id_src!r}")
    lignes, retirees = [], []
    for r in relu:
        k, i = r["_ligne"], r["line_id"]
        refus(r.get("statut") in (None, SUPPRIMEE), "STATUT_INCONNU", f"ligne {k} ({i}) : statut {r.get('statut')!r}")
        if r.get("statut") == SUPPRIMEE:  # pas une ligne de texte selon le relecteur : retirée, comptée à part
            retirees.append(i)
            continue
        refus("text" in r, "CHAMP_MANQUANT", f"ligne {k} ({i}) : text exigé")
        refus(isinstance(r["text"], str), "TEXTE_INVALIDE", f"ligne {k} ({i}) : texte non chaîne")
        try:
            r["text"].encode("utf-8")
        except UnicodeEncodeError as exc:  # substitut isolé (\\ud800…) accepté par json.loads, illisible ensuite
            raise VTRefus("ENCODAGE_INVALIDE", f"ligne {k} ({i}) : {exc}") from exc
        refus(r["text"].strip() != "", "TEXTE_VIDE", f"ligne {k} ({i}) : texte vide ou blanc sans statut de retrait")
        lignes.append({"id": i, "text": r["text"]})
    refus(lignes, "PAGE_VIDE", "aucune ligne de texte après les retraits du relecteur")
    return {"page_key": cle, "lines": lignes}, retirees


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entree", required=True)
    ap.add_argument("--source-bbvlm", required=True)
    ap.add_argument("--page-key", required=True)
    ap.add_argument("--type", required=True, choices=sorted(NS))
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    out = Path(a.sortie)
    try:
        refus(out.parent.is_dir(), "SORTIE_INVALIDE", f"dossier de sortie absent : {out.parent}")
        try:
            raw, raw_src = Path(a.entree).read_bytes(), Path(a.source_bbvlm).read_bytes()
        except OSError as exc:
            raise VTRefus("ENTREE_ILLISIBLE", str(exc)) from exc
        vt, retirees = convertir(raw, raw_src, a.page_key, a.type)
        data = json.dumps(vt, ensure_ascii=False, indent=1).encode("utf-8")
        refus(not out.exists(), "SORTIE_EXISTANTE", f"{out} existe déjà")
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".vt-", suffix=".tmp")  # écriture complète, puis lien exclusif
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp, out)  # échoue si la sortie existe : jamais d'écrasement, jamais de fichier partiel
            except FileExistsError as exc:
                raise VTRefus("SORTIE_EXISTANTE", f"{out} existe déjà") from exc
        finally:
            os.unlink(tmp)
    except VTRefus as exc:
        print(f"REFUS {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"page_key": vt["page_key"], "lignes": len(vt["lines"]), "retirees_par_relecteur": retirees,
                      "entree_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": hashlib.sha256(raw_src).hexdigest(),
                      "sortie_sha256": hashlib.sha256(data).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
