"""T3 partie 2 — lignes vraies manquées dans un trou de colonne (docs/READER_T3_PROTOCOL.txt).

Dev d'E12 (5 pages, split « dev » seulement ; aucun test), chaîne ppdl3_z,
décomposition d'E27 (axel scripts/e27_decompose, lecture seule). Pour chaque
ligne vraie classée « manquee » : parmi les lignes détectées du même bloc
détecté qui recouvrent son abscisse, deux consécutives (en y de baseline)
l'encadrent-elles avec Δy/ℓ ∈ [1,7 ; 2,3] (ℓ = médiane des Δy du bloc) ?
Hasard : part de l'étendue verticale du bloc couverte par de tels trous
(position uniforme dans la colonne).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(AXEL / "scripts"))
OUT = Path(__file__).resolve().parents[1] / "reports/reader_t3_part2.json"


def ymean(line) -> float:
    return float(np.mean([y for _, y in line.baseline]))


def xr(line) -> tuple[float, float]:
    xs = [x for x, _ in line.polygon]
    return min(xs), max(xs)


def main() -> None:
    e27 = importlib.import_module("e27_decompose")
    paths = importlib.import_module("e22_paths")
    from axel.eval.seg_errors import decompose, make_line
    from axel.layout.gt import load_page

    rows, n_missed, n_gap, chance = [], 0, 0, []
    for p in paths.pages("e12", "dev"):
        ch = e27.load("ppdl3_z", p)
        if ch is None:
            continue
        gt = load_page(Path(p["xml"]))
        glines = gt.lines()
        G = [make_line(g.id, g.polygon, g.baseline, g.text) for g in glines]
        H = [make_line(u.id, u.polygon, u.baseline, u.text) for u in ch.units]
        d = decompose(G, H, [r.polygon for r in gt.text_regions], "nearest")
        blocks: dict[str | None, list[int]] = {}
        for j, u in enumerate(ch.units):
            blocks.setdefault(u.block, []).append(j)
        missed = [i for c in d.components if c.kind == "manquee" for i in c.gt]
        page_gap = 0
        for i in missed:
            g = G[i]
            gy, (gx0, gx1) = ymean(g), xr(g)
            hit = False
            best_chance = 0.0
            for js in blocks.values():
                cols = [H[j] for j in js if min(xr(H[j])[1], gx1) - max(xr(H[j])[0], gx0) > 0]
                if len(cols) < 3:
                    continue
                ys = sorted(ymean(h) for h in cols)
                dy = np.diff(ys)
                dy = dy[dy > 0]
                if len(dy) < 2:
                    continue
                ell = float(np.median(dy))
                span = ys[-1] - ys[0]
                gaps = [(a, b) for a, b in zip(ys, ys[1:]) if 1.7 <= (b - a) / ell <= 2.3]
                if span > 0:
                    best_chance = max(best_chance, sum(b - a for a, b in gaps) / span)
                if ys[0] < gy < ys[-1]:
                    if any(a < gy < b for a, b in gaps):
                        hit = True
            n_missed += 1
            n_gap += hit
            page_gap += hit
            chance.append(best_chance)
        rows.append({"page": p["page"], "missed": len(missed), "in_gap": page_gap})
    rep = {"pages": rows, "missed": n_missed, "in_gap_1.7_2.3": n_gap,
           "frac": n_gap / max(1, n_missed), "chance_mean": float(np.mean(chance)) if chance else None}
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
