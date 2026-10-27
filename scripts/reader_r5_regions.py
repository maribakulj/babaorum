"""R5 — manifeste de régions d'apprentissage (géométrie + texte, aucune image).

Source : fichiers PAGE NewsEye/READ (~/corpus-vt/newseye-read), lignes
gardées seulement si elles sont dans axel press/train.jsonl, jeu par défaut
(D-144 : transkribus_1937, corrected_T) : les exclusions d'E6/C7 restent
celles d'axel. Pour chaque région de texte, fenêtres de k lignes
consécutives (k = 1..8) dans l'ordre de la région ; chaque élément porte la
boîte de la fenêtre (union des polygones, marge), et par ligne : identifiant,
texte, baseline, polygone. Variantes d'erreurs de découpe, tirées (graine
fixe) : « coupée » (bord droit ou gauche ramené dans une ligne : la ligne
touchée est marquée partielle, son texte n'est plus une cible exacte),
« décalée » (boîte décalée en y d'une fraction de hauteur de ligne : encre
des voisines), « vide » (aucune). Sortie : reports/reader_r5_regions.jsonl.gz.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

AXEL = Path.home() / "axel"
SRC = Path.home() / "corpus-vt/newseye-read/ATR_TrainingSet_BnF_Newseye_M2+"
OUT = Path(__file__).resolve().parents[1] / "reports" / "reader_r5_regions.jsonl.gz"
NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
DEFAULT = {"transkribus_1937", "corrected_T"}


def pts(s: str) -> list[list[float]]:
    return [[float(a), float(b)] for a, b in (p.split(",") for p in s.split())]


def bbox(polys: list[list[list[float]]], pad: float) -> list[float]:
    xs = [x for p in polys for x, _ in p]
    ys = [y for p in polys for _, y in p]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def main() -> None:
    keep: dict[tuple[str, str], dict] = {}
    for line in open(AXEL / "data/prepared/press/train.jsonl"):
        r = json.loads(line)
        if r["corpus"] == "newseye_read" and r.get("group") in DEFAULT:
            keep[(r["page"], r["line_id"])] = r
    pages = sorted({p for p, _ in keep})
    rng = random.Random(5)
    n_items = n_lines = 0
    stats = {"none": 0, "cut": 0, "shift": 0}
    with gzip.open(OUT, "wt") as g:
        for page in pages:
            xml = SRC / f"{page}.xml"
            if not xml.exists():
                continue
            root = ET.parse(xml).getroot()
            for reg in root.iter(f"{{{NS['p']}}}TextRegion"):
                lines = []
                for tl in reg.findall("p:TextLine", NS):
                    r = keep.get((page, tl.get("id")))
                    bl = tl.find("p:Baseline", NS)
                    co = tl.find("p:Coords", NS)
                    if r is None or bl is None or co is None:
                        lines.append(None)  # ligne hors train : coupe la fenêtre
                        continue
                    lines.append({"line_id": r["line_id"], "truth": r["truth"],
                                  "baseline": pts(bl.get("points")), "polygon": pts(co.get("points"))})
                runs, cur = [], []
                for ln in lines + [None]:
                    if ln is None:
                        if cur:
                            runs.append(cur)
                        cur = []
                    else:
                        cur.append(ln)
                for run in runs:
                    i = 0
                    while i < len(run):
                        k = min(len(run) - i, rng.choice((1, 1, 2, 3, 4, 6, 8)))
                        win = [dict(x) for x in run[i : i + k]]
                        i += k
                        hs = [max(y for _, y in w["polygon"]) - min(y for _, y in w["polygon"]) for w in win]
                        h = sorted(hs)[len(hs) // 2]
                        box = bbox([w["polygon"] for w in win], pad=0.15 * h)
                        kind = rng.choices(("none", "cut", "shift"), (0.6, 0.2, 0.2))[0]
                        if kind == "cut":
                            j = rng.randrange(k)
                            xs = [x for x, _ in win[j]["polygon"]]
                            frac = rng.uniform(0.2, 0.8)
                            if rng.random() < 0.5:
                                box[2] = min(xs) + frac * (max(xs) - min(xs))
                            else:
                                box[0] = min(xs) + frac * (max(xs) - min(xs))
                            for w in win:
                                wx = [x for x, _ in w["polygon"]]
                                if max(wx) > box[2] or min(wx) < box[0]:
                                    w["partial"] = True
                        elif kind == "shift":
                            dy = rng.uniform(-0.5, 0.5) * h
                            box[1] += dy
                            box[3] += dy
                        stats[kind] += 1
                        g.write(json.dumps({"page": page, "region": reg.get("id"), "kind": kind,
                                            "box": [round(v, 1) for v in box], "lines": win},
                                           ensure_ascii=False) + "\n")
                        n_items += 1
                        n_lines += k
    print(json.dumps({"pages": len(pages), "items": n_items, "lines": n_lines, "kinds": stats, "out": str(OUT)}))


if __name__ == "__main__":
    main()
