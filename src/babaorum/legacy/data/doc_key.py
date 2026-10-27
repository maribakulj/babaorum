"""Clé de document d'une page, la même quelle que soit la source (C7, D-506).

Une même page de Gallica arrive sous plusieurs noms : ``6330162-001``
(NewsEye/READ ``ocr_bnf`` et paquet BnF « OCR corrigé presse »),
``Ce-soir_7633477_D0000001`` (READ ``transkribus_1937``),
``La_Presse_0512130_T0000003`` (READ ``corrected_T``), ``268644`` (paquet
BnF, page unique), ou un ark ``ark:/12148/bpt6k63301626`` et un folio. Une
exclusion écrite sur le **nom** d'une source laisse passer la même page
venue d'une autre : c'est la fuite de C7 (``read_6330162-00{1,2}`` en train
d'E21 alors que ``bnf_6330162-00{1,2}`` sont au test d'E7).

La clé est ``gallica:<identifiant>/f<folio>`` : identifiant Gallica sans
zéros de tête (ark ``bpt6k<identifiant><contrôle>`` sans son caractère de
contrôle NOID), folio en entier (``-001``, ``-01``, ``T0000001`` → 1 ; nom
sans folio → 1). IMPACT KB : ``kb:<identifiant de page>``. Toute autre page
n'a pas de clé (``None``) : seule la règle des 8-grammes la protège.

Vérifié le 7 octobre sur les 53 groupes de pages du manifeste d'E21 qui
partagent une clé : les numérotations ``-NNN`` et ``[DT]NNNNNNN`` désignent
la même vue (partage de 8-grammes de 0,2 à 1,0 entre les membres).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

NOID = "0123456789bcdfghjkmnpqrstvwxz"

_TD = re.compile(r"(?:.+_)?(\d{6,7})_[DT](\d+)")
_DASH = re.compile(r"(\d{5,7})(?:-(\d{1,4}))?")
_ARK = re.compile(r"bpt6k(\w+)")
_PREFIX = re.compile(r"^(?:read_|bnf_|newseye_read:|kb_)")


def noid_check(name: str) -> str:
    """Caractère de contrôle NOID d'un nom d'ark BnF (sans le NAAN), comme
    ``scripts/e13_gallica.noid_check``."""
    s = sum((i + 1) * (NOID.index(c) if c in NOID else 0) for i, c in enumerate(name))
    return NOID[s % 29]


def gallica_id_of_ark(ark: str) -> str | None:
    """``ark:/12148/bpt6k63301626`` (ou ``bpt6k63301626``) → ``6330162`` ;
    le dernier caractère n'est retiré que s'il est bien le contrôle NOID."""
    m = _ARK.search(ark)
    if not m:
        return None
    name = "bpt6k" + m.group(1)
    if noid_check(name[:-1]) == name[-1]:
        return m.group(1)[:-1].lstrip("0") or "0"
    return m.group(1).lstrip("0") or "0"


def gallica_ark(gallica_id: str) -> str:
    """Identifiant Gallica → nom d'ark complet (``bpt6k…`` + contrôle)."""
    base = "bpt6k" + gallica_id.lstrip("0")
    return base + noid_check(base)


def gallica_page(name: str) -> tuple[str, int] | None:
    """Identifiant Gallica et folio d'un nom de page READ ou BnF (préfixe de
    source toléré), ou ``None``."""
    stem = _PREFIX.sub("", name)
    m = _TD.fullmatch(stem) or _DASH.fullmatch(stem)
    if not m:
        return None
    return m.group(1).lstrip("0") or "0", int(m.group(2) or 1)


def doc_key(source: str, name: str, ark: str | None = None) -> str | None:
    """Clé de document d'une page (``source`` : ``read``, ``bnf``, ``kb``,
    ``as`` — pour AS, ``name`` est la page READ appariée, ``read_page``).
    Avec un ``ark``, l'identifiant vient de l'ark et le folio du nom."""
    if source == "kb":
        return f"kb:{_PREFIX.sub('', name)}"
    if source not in ("read", "bnf", "as", "e12", "readval"):
        return None
    page = gallica_page(name)
    if ark:
        gid = gallica_id_of_ark(ark)
        if gid is not None:
            return f"gallica:{gid}/f{page[1] if page else 1}"
    if page is None:
        return None
    return f"gallica:{page[0]}/f{page[1]}"


def issue_key(source: str, name: str, ark: str | None = None) -> str | None:
    """Clé de **numéro** (ark sans folio), distincte de la clé de page
    :func:`doc_key` : ``gallica:<identifiant>`` ; une page f1 en train et une
    page f2 du même numéro en test ont la même clé de numéro. IMPACT KB : le
    paquet ne dit pas le numéro, la clé est celle de la page
    (``kb:<page>``)."""
    page = doc_key(source, name, ark)
    if page is None:
        return None
    return page.split("/f", 1)[0] if page.startswith("gallica:") else page


def duplicates(rows: Iterable[tuple[str, str | None]]) -> dict[str, list[str]]:
    """Clés portées par plus d'une page : ``(page, clé)`` → clé → pages."""
    by: dict[str, list[str]] = defaultdict(list)
    for page, key in rows:
        if key is not None:
            by[key].append(page)
    return {k: v for k, v in sorted(by.items()) if len(v) > 1}
