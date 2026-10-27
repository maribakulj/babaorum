"""Les chaînes de segmentation d'E12, assemblées depuis les sorties en cache.

Toutes les sorties lourdes sont calculées une fois (``runs/e12/``) :

- ``blocks/<det>/<page>.json`` : détections brutes d'un détecteur
  (``scripts/e12_detect.py``), score et rang compris ;
- ``lines/<src>/<page>.json`` : lignes ``blla`` (page entière, ou dans
  chaque bloc détecté au-dessus d'un plancher), ``scripts/e12_kraken.py`` ;
- ``read/<src>/<page>.json`` : lecture ``german_print_s`` de chaque ligne ;
- ``tesseract/<page>.xml`` : ALTO Tesseract (sa segmentation et sa lecture) ;
- ``runs/e3/dots_mocr/layout*/<page>.json`` : mise en page et texte de
  dots.mocr (E3).

Une chaîne choisit ses blocs (seuil de confiance, étiquettes non textuelles
écartées, doublons fondus), les ordonne (rang du détecteur ou coupe XY), et
garde les lignes de chaque bloc dans l'ordre de ``blla`` ; une ligne déjà
couverte par une ligne gardée (bloc chevauchant) est écartée. Option
« comblement » : les lignes ``blla`` de la page entière qui ne tombent dans
aucun bloc gardé sont ajoutées, placées par la coupe XY. Les réglages se
choisissent sur dev (D-213) ; :data:`FROZEN` les garde.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from babaorum.legacy.layout.blocks import NOT_READ, is_read
from babaorum.legacy.layout.geom import (
    Box,
    Point,
    baseline_cover,
    bbox,
    box_area,
    box_inter,
    line_height,
)
from babaorum.legacy.layout.order import xy_cut

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "e12"
E3 = ROOT / "runs" / "e3" / "dots_mocr"

#: étiquettes de blocs sans texte à lire (les deux détecteurs, dots.mocr) ;
#: la règle complète est :func:`axel.layout.blocks.is_read` (catégories
#: ``illustration`` et ``separateur`` comprises, C2 L5)
NON_TEXT = NOT_READ


@dataclass
class Block:
    id: str
    polygon: tuple[Point, ...]
    label: str
    score: float
    rank: int

    @property
    def box(self) -> Box:
        return bbox(self.polygon)


@dataclass
class Unit:
    """Une unité de texte dans l'ordre de lecture : une ligne, ou un bloc
    entier pour un lecteur de page (dots.mocr, sans lignes)."""

    id: str
    polygon: tuple[Point, ...]
    baseline: tuple[Point, ...]
    text: str
    block: str | None = None


@dataclass
class Chain:
    name: str
    page: str
    blocks: list[Block] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    has_lines: bool = True
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Params:
    conf: float = 0.5
    dedup: float = 0.7  # bloc écarté si cette part de son aire est dans un bloc gardé
    order: str = "xycut"  # ou "rank" (ordre du détecteur)
    fill: bool = False


#: réglages gelés après dev (D-213) ; remplis par la décision
FROZEN: dict[str, Params] = {
    # D-219f : rang du détecteur (min des rangs des blocs de la zone), sans comblement
    "ppdl3_z": Params(conf=0.0, dedup=0.8, order="rank", fill=False),
    # D-219f : Heron ne rend aucun ordre ; coupe XY fixée a priori, sans comblement
    "heron_z": Params(conf=0.0, dedup=0.8, order="xycut", fill=False),
}


def _tup(seq: Any) -> tuple[Point, ...]:
    return tuple((float(a), float(b)) for a, b in seq)


def _read(src: str, page: str) -> tuple[dict[str, str], float]:
    p = RUNS / "read" / src / f"{page}.json"
    if not p.exists():
        return {}, 0.0
    d = json.loads(p.read_text())
    return d["texts"], float(d.get("seconds", 0.0))


def select_blocks(raw: list[dict[str, Any]], prm: Params) -> list[dict[str, Any]]:
    """Seuil, étiquettes textuelles, fusion des doublons (score décroissant)."""
    cand = [b for b in raw if b["score"] >= prm.conf and is_read(b.get("label"))]
    cand.sort(key=lambda b: -b["score"])
    kept: list[dict[str, Any]] = []
    boxes: list[Box] = []
    for b in cand:
        bb = bbox(_tup(b["polygon"]))
        a = box_area(bb)
        if a <= 0:
            continue
        inside = max((box_inter(bb, k) for k in boxes), default=0.0)
        if inside >= prm.dedup * a:
            continue
        kept.append(b)
        boxes.append(bb)
    return kept


def _covered(ln: dict[str, Any], kept: list[dict[str, Any]]) -> bool:
    """Une ligne déjà longée à ≥ 50 % par une ligne gardée (bloc voisin)."""
    bl = _tup(ln["baseline"])
    tol = 0.5 * line_height(_tup(ln["boundary"]), bl)
    x0, y0, x1, y1 = bbox(_tup(ln["boundary"]))
    for k in kept:
        kb = k["_box"]
        touch = kb[0] < x1 and x0 < kb[2] and kb[1] < y1 and y0 < kb[3]
        if touch and baseline_cover(bl, _tup(k["baseline"]), tol) >= 0.5:
            return True
    return False


def detector_chain(
    det: str, page: str, prm: Params, name: str | None = None
) -> Chain | None:
    path = RUNS / "lines" / det / f"{page}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    texts, t_read = _read(det, page)
    if not (RUNS / "read" / det / f"{page}.json").exists():
        return None  # lignes non lues : rien à mesurer
    blocks = select_blocks(data["blocks"], prm)
    order = (
        xy_cut([bbox(_tup(b["polygon"])) for b in blocks])
        if prm.order == "xycut"
        else sorted(range(len(blocks)), key=lambda i: blocks[i].get("rank", i))
    )
    blocks = [blocks[i] for i in order]
    ch = Chain(name or det, page)
    kept: list[dict[str, Any]] = []
    for b in blocks:
        ch.blocks.append(
            Block(
                b["id"],
                _tup(b["polygon"]),
                b.get("label", ""),
                b["score"],
                b.get("rank", 0),
            )
        )
        for ln in b["lines"]:
            if len(ln["baseline"]) < 2 or len(ln["boundary"]) < 3:
                continue
            if _covered(ln, kept):
                continue
            ln = {**ln, "_box": bbox(_tup(ln["boundary"])), "_block": b["id"]}
            kept.append(ln)
    if prm.fill:
        full = RUNS / "lines" / "blla" / f"{page}.json"
        ftexts, _ = _read("blla", page)
        if full.exists():
            bboxes = [blk.box for blk in ch.blocks]
            extra = []
            for ln in json.loads(full.read_text())["blocks"][0]["lines"]:
                bb = bbox(_tup(ln["boundary"]))
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in bboxes):
                    continue
                if _covered(ln, kept):
                    continue
                extra.append(
                    {
                        **ln,
                        "_box": bb,
                        "_block": None,
                        "_text": ftexts.get(ln["id"], ""),
                    }
                )
            if extra:
                # blocs et lignes ajoutées ordonnés ensemble par la coupe XY
                groups: list[list[dict[str, Any]]] = []
                gboxes: list[Box] = []
                for blk in ch.blocks:
                    groups.append([k for k in kept if k["_block"] == blk.id])
                    gboxes.append(blk.box)
                for e in extra:
                    groups.append([e])
                    gboxes.append(e["_box"])
                kept = [ln for gi in xy_cut(gboxes) for ln in groups[gi]]
    for ln in kept:
        text = ln["_text"] if "_text" in ln else texts.get(ln["id"], "")
        ch.units.append(
            Unit(
                ln["id"], _tup(ln["boundary"]), _tup(ln["baseline"]), text, ln["_block"]
            )
        )
    det_json = RUNS / "blocks" / det / f"{page}.json"
    if det_json.exists():
        ch.seconds["detect"] = float(
            json.loads(det_json.read_text()).get("seconds", 0.0)
        )
    ch.seconds["lines"] = float(data.get("seconds", 0.0))
    ch.seconds["read"] = t_read
    return ch


def blla_chain(page: str) -> Chain | None:
    path = RUNS / "lines" / "blla" / f"{page}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    texts, t_read = _read("blla", page)
    if not (RUNS / "read" / "blla" / f"{page}.json").exists():
        return None
    ch = Chain("blla", page, seconds={"lines": float(data["seconds"]), "read": t_read})
    for r in data.get("regions", []):
        ch.blocks.append(Block(r["id"], _tup(r["polygon"]), r["label"], 1.0, 0))
    for ln in data["blocks"][0]["lines"]:
        if len(ln["baseline"]) < 2 or len(ln["boundary"]) < 3:
            continue
        ch.units.append(
            Unit(
                ln["id"],
                _tup(ln["boundary"]),
                _tup(ln["baseline"]),
                texts.get(ln["id"], ""),
            )
        )
    return ch


def human_chain(page: str, xml: Path) -> Chain | None:
    from babaorum.legacy.layout.gt import load_page

    gt = load_page(xml)
    if (
        not gt.has_line_geometry
        or not (RUNS / "read" / "human" / f"{page}.json").exists()
    ):
        return None  # pas de lignes humaines (IMPACT KB) ou pas encore lues
    texts, t_read = _read("human", page)
    ch = Chain("human", page, seconds={"read": t_read})
    for r in gt.text_regions:
        ch.blocks.append(Block(r.id, r.polygon, "TextRegion", 1.0, 0))
    for ln in gt.lines():
        ch.units.append(
            Unit(ln.id, ln.polygon, ln.baseline, texts.get(ln.id, ""), ln.region)
        )
    return ch


def _alto(path: Path) -> list[tuple[Box, list[tuple[Box, str]]]]:
    from lxml import etree

    def box_of(el: Any) -> Box:
        x, y = float(el.get("HPOS") or 0), float(el.get("VPOS") or 0)
        return (x, y, x + float(el.get("WIDTH") or 0), y + float(el.get("HEIGHT") or 0))

    out: list[tuple[Box, list[tuple[Box, str]]]] = []
    for el in etree.parse(str(path)).iter():
        if not isinstance(el.tag, str) or etree.QName(el).localname != "TextBlock":
            continue
        lines = []
        for ln in el.iter():
            if isinstance(ln.tag, str) and etree.QName(ln).localname == "TextLine":
                words = [
                    s.get("CONTENT") or ""
                    for s in ln.iter()
                    if isinstance(s.tag, str) and etree.QName(s).localname == "String"
                ]
                lines.append((box_of(ln), " ".join(words)))
        out.append((box_of(el), lines))
    return out


def _xml_error() -> type[Exception]:
    from lxml import etree

    err: type[Exception] = etree.XMLSyntaxError
    return err


def tesseract_chain(page: str, reread: bool = False) -> Chain | None:
    """L'OCR Tesseract tel quel (``reread`` : ses lignes lues par
    ``german_print_s``, source ``tess``)."""
    path = RUNS / "tesseract" / f"{page}.xml"
    if not path.exists() or path.stat().st_size == 0:
        return None  # vide = Tesseract interrompu : référence absente (D-491)
    try:
        blocks = _alto(path)
    except _xml_error() as e:  # tronqué : absent aussi, jamais une exception
        print(f"ALTO Tesseract illisible, tenu pour absent : {path} ({e})")
        return None
    texts, t_read = _read("tess", page) if reread else ({}, 0.0)
    if reread and not texts:
        return None
    ch = Chain("tess_gps" if reread else "tesseract", page, seconds={"read": t_read})
    k = 0
    for bi, (bb, lines) in enumerate(blocks):
        ch.blocks.append(Block(f"tb{bi}", _box_poly(bb), "TextBlock", 1.0, bi))
        for lb, text in lines:
            lid = f"tl{k}"
            k += 1
            poly = _box_poly(lb)
            base = (
                (lb[0], lb[1] + 0.8 * (lb[3] - lb[1])),
                (lb[2], lb[1] + 0.8 * (lb[3] - lb[1])),
            )
            ch.units.append(
                Unit(lid, poly, base, texts.get(lid, "") if reread else text, f"tb{bi}")
            )
    return ch


def tesseract_lines_json(page: str) -> dict[str, Any] | None:
    """Les lignes Tesseract au format ``lines/<src>`` (pour la relecture)."""
    ch = tesseract_chain(page)
    if ch is None:
        return None
    return {
        "seconds": 0.0,
        "blocks": [
            {
                "id": "tess",
                "polygon": None,
                "label": "page",
                "score": 1.0,
                "lines": [
                    {
                        "id": u.id,
                        "baseline": [list(p) for p in u.baseline],
                        "boundary": [list(p) for p in u.polygon],
                    }
                    for u in ch.units
                ],
            }
        ],
    }


def _box_poly(b: Box) -> tuple[Point, ...]:
    return ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))


def dots_chain(page: str) -> Chain | None:
    """dots.mocr page entière (E3) : ses blocs dans son ordre, son texte."""
    from babaorum.legacy.image.vlm import parse_layout

    path = None
    for tag in ("layout", "layout_kaggle", "layout_tiles"):
        p = E3 / tag / f"{page}.json"
        if p.exists():
            path = p
            break
    if path is None:
        return None
    data = json.loads(path.read_text())
    ch = Chain("dots_page", page, has_lines=False)
    secs = 0.0
    k = 0
    for t in data["tiles"]:
        x0, y0, x1, y1 = t["box"]
        w, h = t["seen"]
        bs, _ = parse_layout(t["raw"], (x1 - x0) / w, (y1 - y0) / h)
        secs += float(t.get("seconds", 0.0))
        for b in bs:
            poly = _box_poly((b.x0 + x0, b.y0 + y0, b.x1 + x0, b.y1 + y0))
            ch.blocks.append(Block(f"d{k}", poly, b.category, 1.0, k))
            if b.text.strip() and b.category not in NON_TEXT:
                ch.units.append(Unit(f"d{k}", poly, (), b.text, f"d{k}"))
            k += 1
    ch.seconds["page"] = secs
    return ch


def load_chain(name: str, page: str, prm: Params | None = None) -> Chain | None:
    """Une chaîne par son nom (réglages gelés pour les détecteurs)."""
    if name == "blla":
        return blla_chain(page)
    if name == "tesseract":
        return tesseract_chain(page)
    if name == "tess_gps":
        return tesseract_chain(page, reread=True)
    if name == "dots_page":
        return dots_chain(page)
    if name == "human":
        pages = {p["page"]: p for p in json.loads((RUNS / "pages.json").read_text())}
        return human_chain(page, Path(pages[page]["xml"]))
    det = name.split("+")[0]
    return detector_chain(det, page, prm or FROZEN.get(name, Params()), name)


def merge_columns(
    boxes: list[Box], gap: float, max_h: float = 3600.0
) -> list[list[int]]:
    """D-219d : regroupe les blocs d'une même colonne en **zones** où chercher
    les lignes (moins d'appels à ``blla``). Glouton de haut en bas : un bloc
    rejoint la zone dont le bas est à moins de ``gap`` pixels au-dessus de
    lui et qui recouvre au moins 70 % de la plus étroite des deux largeurs,
    si la zone agrandie ne mord sur aucun autre bloc (à 10 % de son aire
    près) et ne dépasse pas ``max_h`` pixels de haut (``blla`` y travaille
    au plus à 2 400 px : une zone plus haute serait réduite, D-214) ; sinon
    il ouvre une zone. Rend les indices de ``boxes`` par zone."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    zones: list[list[int]] = []
    zbox: list[Box] = []
    for i in order:
        b = boxes[i]
        best = None
        for z, zb in enumerate(zbox):
            ov = min(b[2], zb[2]) - max(b[0], zb[0])
            narrow = min(b[2] - b[0], zb[2] - zb[0])
            if narrow <= 0 or ov < 0.7 * narrow:
                continue
            if not (-gap <= b[1] - zb[3] <= gap):
                continue
            u = (min(b[0], zb[0]), min(b[1], zb[1]), max(b[2], zb[2]), max(b[3], zb[3]))
            if u[3] - u[1] > max_h:
                continue
            members = set(zones[z]) | {i}
            if any(
                box_inter(u, boxes[k]) > 0.1 * box_area(boxes[k])
                for k in range(len(boxes))
                if k not in members
            ):
                continue
            best = (z, u)
            break
        if best is None:
            zones.append([i])
            zbox.append(b)
        else:
            zones[best[0]].append(i)
            zbox[best[0]] = best[1]
    return zones
