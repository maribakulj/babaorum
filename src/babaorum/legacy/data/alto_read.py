"""Lecture d'un ALTO de production (Gallica, Tesseract) par un vrai analyseur XML.

Source unique (C6, D-496) pour E13 (``scripts/e13_eval.py``), E12
(``e12_report.gallica_chain``), E22 et P3 (``scripts/p3_pseudo.py``).
Noms locaux (toute forme d'espace de noms, préfixe ou non), attributs dans
n'importe quel ordre, entités XML décodées une seule fois par lxml.

Une ligne a pour texte ses ``String/@CONTENT`` joints par une espace, suivis
d'un ``-`` si elle porte un ``HYP`` et ne finit pas déjà par un trait
(convention d'E13, D-223).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lxml import etree


class AltoError(ValueError):
    """ALTO illisible : XML invalide, pas de ``Page``, dimensions absentes."""


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1]


def read_xml(raw: bytes) -> etree._Element:
    """L'ALTO de Gallica déclare ISO-8859-1 et contient de l'UTF-8 : on
    essaie l'UTF-8 d'abord, puis l'encodage déclaré."""
    fixed = (
        re.sub(rb"encoding=[\"'][^\"']+[\"']", b'encoding="UTF-8"', raw[:200], count=1)
        + raw[200:]
    )
    try:
        return etree.fromstring(fixed)
    except etree.XMLSyntaxError:
        try:
            return etree.fromstring(raw)
        except etree.XMLSyntaxError as exc:
            raise AltoError(f"XML invalide : {exc}") from exc


def _f(el: etree._Element, k: str) -> float:
    return float(el.get(k, "0") or 0)


def parse_alto(path: Path | bytes) -> dict[str, Any]:
    """Page, blocs (dans l'ordre du document) et lignes d'un ALTO.

    ``{"unit", "width", "height", "blocks", "lines", "tags"}`` ; une ligne :
    ``{"id", "block", "box": (x0, y0, x1, y1), "text"}`` dans les unités de
    l'ALTO. Lève :class:`AltoError` si le XML est invalide ou sans ``Page``."""
    raw = path if isinstance(path, bytes) else path.read_bytes()
    root = read_xml(raw)
    page = next((e for e in root.iter() if _local(e.tag) == "Page"), None)
    if page is None:
        raise AltoError("pas d'élément Page")
    unit = next(
        (e.text for e in root.iter() if _local(e.tag) == "MeasurementUnit"), "pixel"
    )
    tags: dict[str, str] = {}
    for e in root.iter():
        if _local(e.tag) in ("OtherTag", "StructureTag", "LayoutTag", "NamedEntityTag"):
            tags[e.get("ID", "")] = (
                f"{_local(e.tag)}:{e.get('LABEL') or e.get('TYPE') or ''}"
            )
    blocks: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    for el in page.iter():
        kind = _local(el.tag)
        if kind in ("TextBlock", "Illustration", "GraphicalElement", "ComposedBlock"):
            box = (
                _f(el, "HPOS"),
                _f(el, "VPOS"),
                _f(el, "HPOS") + _f(el, "WIDTH"),
                _f(el, "VPOS") + _f(el, "HEIGHT"),
            )
            blocks.append(
                {
                    "id": el.get("ID", ""),
                    "kind": kind,
                    "type": el.get("TYPE", ""),
                    "tagrefs": [
                        tags.get(t, t) for t in (el.get("TAGREFS") or "").split()
                    ],
                    "stylerefs": el.get("STYLEREFS", ""),
                    "box": box,
                    "nlines": sum(1 for c in el.iter() if _local(c.tag) == "TextLine"),
                }
            )
        if kind == "TextLine":
            words: list[str] = []
            hyp = False
            for c in el:
                t = _local(c.tag)
                if t == "String":
                    words.append(c.get("CONTENT", ""))
                elif t == "HYP":
                    hyp = True
            text = " ".join(w for w in words if w)
            if hyp and not text.endswith("-"):
                text += "-"
            parent = el.getparent()
            lines.append(
                {
                    "id": el.get("ID", ""),
                    "block": parent.get("ID", "") if parent is not None else "",
                    "box": (
                        _f(el, "HPOS"),
                        _f(el, "VPOS"),
                        _f(el, "HPOS") + _f(el, "WIDTH"),
                        _f(el, "VPOS") + _f(el, "HEIGHT"),
                    ),
                    "text": text,
                }
            )
    return {
        "unit": unit,
        "width": _f(page, "WIDTH"),
        "height": _f(page, "HEIGHT"),
        "blocks": blocks,
        "lines": lines,
        "tags": tags,
    }


def lines_in_pixels(
    path: Path | bytes, width: int, height: int
) -> list[tuple[list[float], str]]:
    """(boîte dans les pixels d'une image ``width`` × ``height``, texte) de
    chaque ``TextLine`` ; échelle = taille de l'image / ``Page@WIDTH``,
    ``Page@HEIGHT`` (dans l'unité de l'ALTO, quelle qu'elle soit).

    Lève :class:`AltoError` si l'ALTO est invalide ou si la page n'a pas de
    dimensions positives ; un ALTO valide sans ligne rend ``[]``."""
    alto = parse_alto(path)
    if alto["width"] <= 0 or alto["height"] <= 0:
        raise AltoError("Page sans WIDTH/HEIGHT positifs")
    sx, sy = width / alto["width"], height / alto["height"]
    out: list[tuple[list[float], str]] = []
    for ln in alto["lines"]:
        x0, y0, x1, y1 = ln["box"]
        out.append(([x0 * sx, y0 * sy, x1 * sx, y1 * sy], ln["text"]))
    return out
