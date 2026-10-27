"""T3 + T8 partie 1 — périodicité de l'interligne et quantum (docs/READER_T3_PROTOCOL.txt).

GT PAGE de NewsEye, pages du jeu par défaut de press/train (lecture seule).
"""
from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

AXEL = Path.home() / "axel"
SRC = Path.home() / "corpus-vt/newseye-read/ATR_TrainingSet_BnF_Newseye_M2+"
OUT = Path(__file__).resolve().parents[1] / "reports"
NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
DEFAULT = {"transkribus_1937", "corrected_T"}
QS = np.arange(2.0, 12.0001, 0.05)


def rayleigh(vals: np.ndarray) -> np.ndarray:
    ph = 2 * np.pi * vals[None, :] / QS[:, None]
    return np.abs(np.exp(1j * ph).mean(1))


def scan(vals: np.ndarray, rng: random.Random, draws: int = 1000) -> dict:
    if len(vals) < 8:
        return {"n": int(len(vals))}
    r = rayleigh(vals)
    k = int(r.argmax())
    obs = float(r.max())
    null = []
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    for _ in range(draws):
        null.append(rayleigh(vals + nprng.uniform(0, 1000, len(vals))).max())
    p = (1 + sum(x >= obs for x in null)) / (draws + 1)
    return {"n": int(len(vals)), "q": float(QS[k]), "R": obs, "p_max": p}


def main() -> None:
    pages = set()
    for line in open(AXEL / "data/prepared/press/train.jsonl"):
        r = json.loads(line)
        if r["corpus"] == "newseye_read" and r.get("group") in DEFAULT:
            pages.add(r["page"])
    ratios, per_page = [], {}
    rng = random.Random(8)
    for page in sorted(pages):
        xml = SRC / f"{page}.xml"
        if not xml.exists():
            continue
        root = ET.parse(xml).getroot()
        dys, lefts, ells = [], [], []
        for reg in root.iter(f"{NS}TextRegion"):
            ys = []
            for tl in reg.findall(f"{NS}TextLine"):
                bl = tl.find(f"{NS}Baseline")
                if bl is None:
                    continue
                pts = [tuple(map(float, p.split(","))) for p in bl.get("points").split()]
                ys.append(float(np.mean([y for _, y in pts])))
            co = reg.find(f"{NS}Coords")
            if co is not None:
                lefts.append(min(float(p.split(",")[0]) for p in co.get("points").split()))
            if len(ys) < 4:
                continue
            ys.sort()
            d = np.diff(ys)
            d = d[d > 0]
            if len(d) < 3:
                continue
            ell = float(np.median(d))
            ratios += list(d / ell)
            ells.append(ell)
            dys += list(d)
        img = next((SRC / f"{page}{ext}" for ext in (".jpg", ".tif", ".png") if (SRC / f"{page}{ext}").exists()), None)
        size = Image.open(img).size if img else None
        per_page[page] = {"size": size, "ell": scan(np.array(ells), rng), "ell_std": float(np.std(ells)) if ells else None,
                          "left": scan(np.array(lefts), rng)}
    ratios = np.array(ratios)
    rep = {
        "pages": len(per_page),
        "gaps": int(len(ratios)),
        "frac_0.9_1.1": float(((ratios >= 0.9) & (ratios <= 1.1)).mean()),
        "frac_1.7_2.3": float(((ratios >= 1.7) & (ratios <= 2.3)).mean()),
        "frac_gt_2.3": float((ratios > 2.3).mean()),
        "quantiles": {str(q): float(np.quantile(ratios, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "hist_0_4": np.histogram(ratios, bins=40, range=(0, 4))[0].tolist(),
        "per_page": per_page,
    }
    (OUT / "reader_t3_part1.json").write_text(json.dumps(rep, indent=1))
    sig = {p: (v["ell"], v["ell_std"], v["left"]) for p, v in per_page.items()
           if v["ell"].get("p_max", 1) < 0.01 or v["left"].get("p_max", 1) < 0.01}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_page", "hist_0_4")}, indent=1))
    print("pages à quantum (ℓ ou bords) p<0,01 :", len(sig), "sur", len(per_page))
    for p, v in sig.items():
        print(" ", p, per_page[p]["size"], v)


if __name__ == "__main__":
    main()
