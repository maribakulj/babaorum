"""Vérité de mise en page lue dans un fichier PAGE (2010 ou 2013).

On garde ce qu'il faut pour juger une segmentation de bout en bout :

- les **régions** (``TextRegion`` et les régions non textuelles), avec leur
  polygone ;
- les **lignes** (``TextLine``) avec polygone, ligne de base et texte ;
- l'**ordre de lecture** des régions (``ReadingOrder``, groupes pris dans
  l'ordre du document), puis celui des lignes dans chaque région (attribut
  Transkribus ``custom="readingOrder {index:N;}"``, sinon ordre du
  document).

Une page sans ``TextLine`` (IMPACT KB) garde le texte de chaque région,
coupé aux sauts de ligne : ses « lignes » n'ont alors ni polygone ni ligne
de base (``polygon == ()``), et ne servent qu'au texte de la page.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

Point = tuple[float, float]

_RO = re.compile(r"readingOrder\s*\{\s*index:\s*(\d+)")


@dataclass(frozen=True)
class GtLine:
    id: str
    region: str
    polygon: tuple[Point, ...]
    baseline: tuple[Point, ...]
    text: str


@dataclass(frozen=True)
class GtRegion:
    id: str
    kind: str  # TextRegion, ImageRegion, SeparatorRegion…
    polygon: tuple[Point, ...]
    lines: tuple[GtLine, ...]
    text: str = ""


@dataclass
class GtPage:
    width: int
    height: int
    image_name: str
    regions: list[GtRegion] = field(default_factory=list)
    #: identifiants des régions dans l'ordre de lecture (toutes les régions
    #: textuelles y sont : celles que ``ReadingOrder`` oublie suivent, dans
    #: l'ordre du document)
    order: list[str] = field(default_factory=list)

    @property
    def text_regions(self) -> list[GtRegion]:
        by_id = {r.id: r for r in self.regions}
        return [by_id[i] for i in self.order if by_id[i].kind == "TextRegion"]

    def lines(self) -> list[GtLine]:
        """Les lignes à texte, dans l'ordre de lecture."""
        out: list[GtLine] = []
        for reg in self.text_regions:
            out.extend(ln for ln in reg.lines if ln.text.strip())
        return out

    @property
    def has_line_geometry(self) -> bool:
        return any(ln.polygon for ln in self.lines())


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1]


def parse_points(points: str) -> tuple[Point, ...]:
    out: list[Point] = []
    for pair in points.split():
        x, _, y = pair.partition(",")
        out.append((float(x), float(y)))
    return tuple(out)


def _coords(el: etree._Element) -> tuple[Point, ...]:
    for child in el:
        if _local(child.tag) == "Coords":
            pts = child.get("points")
            if pts:
                return parse_points(pts)
            # PAGE 2010 : <Point x= y=/>
            return tuple(
                (float(p.get("x", 0)), float(p.get("y", 0)))
                for p in child
                if _local(p.tag) == "Point"
            )
    return ()


def _baseline(el: etree._Element) -> tuple[Point, ...]:
    for child in el:
        if _local(child.tag) == "Baseline" and child.get("points"):
            return parse_points(child.get("points", ""))
    return ()


def _own_text(el: etree._Element) -> str:
    """Le ``TextEquiv/Unicode`` enfant direct (pas celui des mots)."""
    for child in el:
        if _local(child.tag) == "TextEquiv":
            for u in child:
                if _local(u.tag) == "Unicode":
                    return unicodedata.normalize("NFC", u.text or "")
    return ""


def _line_index(el: etree._Element, default: int) -> int:
    m = _RO.search(el.get("custom", ""))
    return int(m.group(1)) if m else default


def load_page(path: Path) -> GtPage:
    root = etree.parse(str(path)).getroot()
    page_el = next(e for e in root.iter() if _local(e.tag) == "Page")
    page = GtPage(
        width=int(page_el.get("imageWidth", 0)),
        height=int(page_el.get("imageHeight", 0)),
        image_name=page_el.get("imageFilename", ""),
    )
    doc_order: list[str] = []
    for el in page_el.iter():
        kind = _local(el.tag)
        if not kind.endswith("Region") or el.get("id") is None:
            continue
        rid = el.get("id", "")
        lines: list[GtLine] = []
        if kind == "TextRegion":
            raw = [c for c in el if _local(c.tag) == "TextLine"]
            ranked = sorted(
                enumerate(raw), key=lambda kv: (_line_index(kv[1], kv[0]), kv[0])
            )
            for _, ln in ranked:
                lines.append(
                    GtLine(
                        id=ln.get("id", ""),
                        region=rid,
                        polygon=_coords(ln),
                        baseline=_baseline(ln),
                        text=_own_text(ln),
                    )
                )
            text = _own_text(el)
            if not raw and text.strip():
                # page sans lignes (IMPACT KB) : le texte de la région, coupé
                for k, part in enumerate(text.splitlines()):
                    if part.strip():
                        lines.append(GtLine(f"{rid}_l{k}", rid, (), (), part))
        else:
            text = ""
        page.regions.append(GtRegion(rid, kind, _coords(el), tuple(lines), text))
        doc_order.append(rid)
    ro: list[str] = []
    for el in page_el.iter():
        if _local(el.tag) != "ReadingOrder":
            continue
        for grp in el.iter():
            if _local(grp.tag) not in ("OrderedGroup", "UnorderedGroup"):
                continue
            refs = [
                c
                for c in grp
                if _local(c.tag) in ("RegionRefIndexed", "RegionRef")
                and c.get("regionRef")
            ]
            refs.sort(key=lambda c: int(c.get("index", 0)))
            ro.extend(c.get("regionRef", "") for c in refs)
    known = set(doc_order)
    seen: set[str] = set()
    order: list[str] = []
    for rid in ro + doc_order:
        if rid in known and rid not in seen:
            seen.add(rid)
            order.append(rid)
    page.order = order
    return page
