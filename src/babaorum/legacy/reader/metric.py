"""Une seule mesure de CER pour les lecteurs (C3, D-422).

D-150 d'abord : NFC, puis normalisation du scorer HIPE (minuscules,
ligatures pliées, non-mot → espace, blancs compactés) avec ``ſ`` → ``s`` en
premier. Le brut (NFC, blancs compactés) est l'annexe. CER micro : somme
des distances d'édition sur somme des longueurs de vérité normalisées.

Ligne dont la vérité est **vide après normalisation** (``*``, ``. —``) :
elle ne compte ni dans les erreurs ni dans le support du CER (convention de
``scripts/e8_report.py``, celle des chiffres E8 et de D-280). Elle n'est pas
oubliée pour autant : :func:`score` rend à part le nombre de ces lignes et
le nombre de caractères que le lecteur y a écrits (une sortie sur une ligne
sans texte est une invention possible).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from rapidfuzz.distance import Levenshtein

#: Pliages du scorer HIPE (D-11) ; ``ſ`` est replié avant, par :func:`d150`.
HIPE_TABLE = {
    "_": " ",
    "aͤ": "ä",
    "oͤ": "ö",
    "uͤ": "ü",
    "ß": "ss",
    "ꝛ": "r",
    "œ": "oe",
    "æ": "ae",
}


def normalize_hipe(text: str) -> str:
    """La normalisation du scorer HIPE : minuscules, pliages ß/ꝛ/œ/æ,
    tout non-mot devient une espace, blancs compactés. ``ſ`` est gardé."""
    text = text.lower()
    for src, dst in HIPE_TABLE.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^\w]", " ", text)
    return " ".join(text.split())


def d150(text: str) -> str:
    """Texte normalisé selon D-150 : NFC, puis HIPE avec ``ſ`` replié (sans
    NFC, HIPE change un accent décomposé en espace : D-422)."""
    return normalize_hipe(unicodedata.normalize("NFC", text).replace("ſ", "s"))


def raw(text: str) -> str:
    """Texte brut de l'annexe : NFC, blancs compactés."""
    return " ".join(unicodedata.normalize("NFC", text).split())


NORMS: dict[str, Callable[[str], str]] = {"d150": d150, "raw": raw}


def lev(a: str, b: str) -> int:
    return int(Levenshtein.distance(a, b))


def line_errors(hyp: str | None, truth: str, mode: str = "d150") -> tuple[int, int]:
    """(erreurs, support) d'une ligne ; ``(0, 0)`` si la vérité normalisée
    est vide (la ligne sort du CER, voir :func:`score`)."""
    f = NORMS[mode]
    h, t = f(hyp or ""), f(truth)
    if not t:
        return 0, 0
    return lev(h, t), len(t)


def cer(pairs: Iterable[tuple[str | None, str]], mode: str = "d150") -> float:
    """CER micro en %, ``pairs`` = [(lecture, vérité)]."""
    e = n = 0
    for h, t in pairs:
        a, b = line_errors(h, t, mode)
        e += a
        n += b
    return 100 * e / n if n else 0.0


def score(pairs: Iterable[tuple[str | None, str]]) -> dict[str, float | int]:
    """D-150 d'abord, brut en annexe, et les lignes à vérité vide à part."""
    ps = list(pairs)
    out: dict[str, float | int] = {
        "cer_d150": cer(ps, "d150"),
        "cer_raw": cer(ps, "raw"),
        "lines": len(ps),
    }
    for mode in ("d150", "raw"):
        f = NORMS[mode]
        empty = [(h, t) for h, t in ps if not f(t)]
        out[f"empty_truth_lines_{mode}"] = len(empty)
        out[f"empty_truth_output_chars_{mode}"] = sum(len(f(h or "")) for h, _ in empty)
    return out
