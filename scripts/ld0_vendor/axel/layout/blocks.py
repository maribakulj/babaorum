"""Format commun des blocs d'un détecteur, du détecteur à l'ALTO (C2, L5).

Tout détecteur de blocs — PP-DocLayoutV3 et Heron (E12, ``scripts/e12_detect.py``),
le détecteur affiné sur NewsEye AS (E21, ``predict_pages`` du carnet), les
régions de ``blla`` — écrit, par page, un fichier JSON
``runs/<exp>/blocks/<modèle>/<page>.json`` :

.. code-block:: text

    {
      "model":  "ppdl3" | "e21_b" | …,        # facultatif
      "width":  5362, "height": 7363,          # repère des coordonnées (px)
      "blocks": [
        {
          "id":       "b0",                    # unique dans la page
          "polygon":  [[x, y], …],             # ≥ 3 sommets, repère ci-dessus
          "label":    "paragraph_title",       # étiquette brute du modèle
          "category": "titre",                 # catégorie commune (ci-dessous)
          "score":    0.93,                    # confiance (1 si sans objet)
          "rank":     0,                       # rang de sortie (ordre du modèle)
          "article":  "a3"                     # facultatif : article (E21)
        }, …
      ]
    }

- **Repère** : pixels de l'image de la page **telle que l'expérience la
  lit**, origine en haut à gauche. ``width``/``height`` disent ce repère ;
  :func:`normalize_blocks` ramène les polygones au repère de l'image quand
  les deux diffèrent (E21 prédit dans le repère du PAGE d'origine, qui peut
  différer de l'image). Sans ``width``/``height``, le repère est supposé
  être celui de l'image (``"frame": "supposé"``).
- **Catégories** (:data:`CATEGORIES`) : les six types de blocs de l'OLR
  (D-231 : ``titre``, ``texte``, ``legende``, ``annonce``, ``tableau``,
  ``illustration``), plus ``separateur`` (filets, E21 D-303) et ``autre``
  (en-tête, pied, folio : du texte pour l'OLR). :data:`LABEL_CATEGORY` est
  la **seule table** étiquette → catégorie du dépôt ; elle fixe aussi le
  type de bloc de l'IR (:func:`axel.pipeline.sources.e12_type`) et la
  catégorie évaluée par l'OLR (:mod:`axel.olr.evaluate`).
- **Lecture** : un bloc est lu (cherché par ``blla``, ses lignes relues) si
  son étiquette n'est pas dans :data:`NOT_READ` et que sa catégorie n'est
  ni ``illustration`` ni ``separateur`` (:func:`is_read`).
- **Identifiants** : un bloc sans ``id`` en reçoit un (``b<rang>``), et un
  doublon est suffixé ; E12 (zones, ``members``), l'IR et l'ALTO s'y réfèrent.
- **Articles** : ``article`` passe tel quel des blocs aux lignes
  (``axel.pipeline.sources.page_from_e12``), puis aux ``StructureTag`` de
  l'ALTO et au METS.
- **Segmentation des lignes** : ``scripts/e12_kraken.py blocks`` lit ce
  format ; ``--blla-model`` choisit le point ``blla`` (par défaut celui de
  kraken), ``--windows`` applique le découpage de l'entraînement d'E21
  (D-304 : fenêtres de 1 800 px au pas de 1 500, :mod:`axel.layout.segment`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CATEGORIES = (
    "titre",
    "texte",
    "legende",
    "annonce",
    "tableau",
    "illustration",
    "separateur",
    "autre",
)

#: étiquette brute (minuscules) → catégorie commune. Détecteurs génériques
#: (PP-DocLayout, Heron, dots.mocr), ``blla``, E21 (nos catégories, telles
#: quelles), vérité PAGE.
LABEL_CATEGORY: dict[str, str] = {
    # titres
    "doc_title": "titre",
    "paragraph_title": "titre",
    "title": "titre",
    "section_header": "titre",
    "section-header": "titre",
    "heading": "titre",
    # texte
    "text": "texte",
    "abstract": "texte",
    "content": "texte",
    "aside_text": "texte",
    "reference": "texte",
    "reference_content": "texte",
    "footnote": "texte",
    "vertical_text": "texte",
    "formula": "texte",
    "algorithm": "texte",
    "list_item": "texte",
    "list-item": "texte",
    "paragraph": "texte",
    "textregion": "texte",
    "zone": "texte",
    "page": "texte",
    # légendes
    "figure_title": "legende",
    "table_title": "legende",
    "chart_title": "legende",
    "vision_footnote": "legende",
    "caption": "legende",
    # page (en-tête, pied, folio)
    "header": "autre",
    "footer": "autre",
    "number": "autre",
    "formula_number": "autre",
    "page-header": "autre",
    "page-footer": "autre",
    "page_header": "autre",
    "page_footer": "autre",
    # tableaux, annonces
    "table": "tableau",
    "advert": "annonce",
    "advertisement": "annonce",
    # illustrations
    "image": "illustration",
    "picture": "illustration",
    "figure": "illustration",
    "chart": "illustration",
    "header_image": "illustration",
    "footer_image": "illustration",
    "seal": "illustration",
    "graphic": "illustration",
    # filets
    "separator": "separateur",
}
LABEL_CATEGORY.update({c: c for c in CATEGORIES})

#: étiquettes de blocs qu'aucune chaîne ne lit (E12, D-214), en plus des
#: catégories ``illustration`` et ``separateur``
NOT_READ = frozenset(
    {
        "image",
        "picture",
        "figure",
        "chart",
        "seal",
        "formula",
        "display_formula",
        "inline_formula",
        "Picture",
        "Formula",
        "algorithm",
        "footer_image",
        "header_image",
    }
)


def category_of(label: str | None) -> str:
    """La catégorie commune d'une étiquette (``texte`` si inconnue)."""
    return LABEL_CATEGORY.get((label or "").strip().lower(), "texte")


def olr_type(category: str) -> str | None:
    """La catégorie OLR (D-231) : ``autre`` est du texte ; ``None`` pour un
    filet (hors blocs de l'OLR)."""
    if category == "separateur":
        return None
    return "texte" if category == "autre" else category


def is_read(label: str | None) -> bool:
    """Un bloc de cette étiquette est-il cherché et lu ligne à ligne ?"""
    if label in NOT_READ:
        return False
    return category_of(label) not in ("illustration", "separateur")


def majority_category(labels: Sequence[str | None]) -> str:
    """Catégorie d'une zone : celle la plus fréquente parmi ses blocs (à
    égalité, la première rencontrée) ; ``texte`` sans bloc."""
    cats = [category_of(lab) for lab in labels]
    if not cats:
        return "texte"
    counts = Counter(cats)
    best = max(counts.values())
    return next(c for c in cats if counts[c] == best)


def normalize_blocks(
    raw: Mapping[str, Any],
    image_size: tuple[int, int] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Un fichier de blocs au format commun : ``id`` uniques, ``category``,
    ``rank``, ``score``, polygones dans le repère de l'image (``image_size``
    = largeur, hauteur). Les autres clés des blocs sont gardées."""
    width = raw.get("width")
    height = raw.get("height")
    sx = sy = 1.0
    frame = "fichier"
    if image_size is not None:
        if width and height:
            sx = image_size[0] / float(width)
            sy = image_size[1] / float(height)
        else:
            frame = "supposé"
        width, height = image_size
    seen: set[str] = set()
    blocks: list[dict[str, Any]] = []
    for k, b in enumerate(raw.get("blocks", ())):
        poly = [[float(p[0]) * sx, float(p[1]) * sy] for p in b.get("polygon") or ()]
        bid = str(b.get("id") or f"b{k}")
        base, n = bid, 1
        while bid in seen:
            n += 1
            bid = f"{base}_{n}"
        seen.add(bid)
        label = b.get("label") or b.get("category") or "text"
        blocks.append(
            {
                **b,
                "id": bid,
                "polygon": poly,
                "label": label,
                "category": category_of(label),
                "score": float(b.get("score", 1.0)),
                "rank": int(b.get("rank", k)),
            }
        )
    out: dict[str, Any] = {
        k: v for k, v in raw.items() if k not in ("blocks", "width", "height")
    }
    out.update(
        {
            "model": model or raw.get("model", ""),
            "width": width,
            "height": height,
            "frame": frame,
            "blocks": blocks,
        }
    )
    return out


def load_blocks(
    path: Path, image_size: tuple[int, int] | None = None
) -> dict[str, Any]:
    """:func:`normalize_blocks` sur un fichier."""
    return normalize_blocks(
        json.loads(path.read_text(encoding="utf-8")), image_size, path.parent.name
    )
