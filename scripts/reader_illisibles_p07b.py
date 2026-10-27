"""Mesure descriptive « lignes illisibles dans le vrai pipeline » (docs/READER_ILLISIBLES_P07B_PREREG.txt ; CLASSE A).

    PYTHONPATH=~/axel/src:src ~/outils-seg/dots-hf-env/bin/python scripts/reader_illisibles_p07b.py planche --sortie <dossier>
    ... scripts/reader_illisibles_p07b.py comptes --sortie <même dossier>

« planche » calcule les traits de chaque ligne finale (lines_c1_ma, in_final c1+ma) des 26 pages de runs_26, les écrit
dans traits.jsonl SANS afficher aucun compte, et rend planche.png (30 exemples, 6 par famille I1-I5, tirage seedé).
« comptes » ne se lance qu'après que pA a vu la planche : comptes par famille et par page, IC de Clopper-Pearson.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
RUNS = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages/runs_26"
TAU = 20.827896720535783
FAM = ("I1", "I2", "I3", "I4", "I5")


def families(t) -> list[str]:
    f = []
    if TAU <= t["sigma"] < TAU + 5:
        f.append("I1")
    if len("".join((t["text"] or "").split())) < 3:
        f.append("I2")
    if t["w"] < 1.5 * t["h"]:
        f.append("I3")
    if t["h"] < 0.4 * t["h_med"]:
        f.append("I4")
    if t["h"] > 2.5 * t["h_med"]:
        f.append("I5")
    return f


def planche(out: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from axel.image.crops import crop_line

    Image.MAX_IMAGE_PIXELS = None
    pages = sorted(glob.glob(str(RUNS / "*_[TD]0*.json")))
    assert len(pages) == 26, len(pages)
    rows, crops = [], {}
    for pf in pages:
        d = json.loads(Path(pf).read_text())
        assert hashlib.sha256(Path(d["image"]).read_bytes()).hexdigest() == d["image_sha256"], d["page"]
        lines = [ln for ln in d["lines_c1_ma"] if ln["in_final"]["c1+ma"]]
        hs = [ln["line_crop"]["box_page"][3] - ln["line_crop"]["box_page"][1] for ln in lines]
        h_med = statistics.median(hs)
        with Image.open(d["image"]) as im:
            page = im.convert("L")
        for ln in lines:
            b = [int(round(v)) for v in ln["line_crop"]["box_page"]]
            c = crop_line(page, tuple(b), ln["polygon"], pad=4).image
            a = np.asarray(c, dtype=np.uint8)
            hist = np.bincount(a.ravel(), minlength=256).astype(float)
            w_ = np.cumsum(hist)
            m_ = np.cumsum(hist * np.arange(256))
            with np.errstate(divide="ignore", invalid="ignore"):
                var = (m_[-1] * w_ / w_[-1] - m_) ** 2 / (w_ * (w_[-1] - w_))
            thr = -1 if a.std() == 0 else int(np.nanargmax(var))  # recadrage uniforme : part sombre 0 (comme ABST v2)
            t = {"page": d["page"], "line_id": ln["line_id"], "box": b, "w": b[2] - b[0], "h": b[3] - b[1], "h_med": h_med,
                 "sigma": float(a.astype(np.float64).std()), "otsu_sombre": float((a <= thr).mean()), "text": ln["text"]}
            t["familles"] = families(t)
            rows.append(t)
            if t["familles"]:
                crops[(t["page"], t["line_id"])] = c
    (out / "traits.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    rng = random.Random("illisibles:planche")
    picked = []
    for f in FAM:
        pool = sorted((r for r in rows if f in r["familles"]), key=lambda r: (r["page"], r["line_id"]))
        for r in rng.sample(pool, min(6, len(pool))):
            picked.append((f, r))
    sheet = Image.new("L", (1700, 90 * len(picked) + 10), 255)
    dr = ImageDraw.Draw(sheet)
    for n, (f, r) in enumerate(picked):
        c = crops[(r["page"], r["line_id"])].copy()
        c.thumbnail((1000, 70))
        y = 90 * n + 5
        dr.text((5, y), f"#{n} {f} sigma={r['sigma']:.1f} {r['w']}x{r['h']} (med {r['h_med']:.0f})", fill=0)
        dr.text((5, y + 14), f"{r['page'][:28]}  lu : {(r['text'] or '')[:40]!r}", fill=0)
        sheet.paste(c, (520, y))
    sheet.save(out / "planche.png")
    (out / "planche.json").write_text(json.dumps([{"n": n, "famille": f, **{k: r[k] for k in ("page", "line_id", "box", "text")}}
                                                  for n, (f, r) in enumerate(picked)], ensure_ascii=False, indent=1))
    print(f"traits écrits ({len(rows)} lignes) ; planche de {len(picked)} exemples : {out / 'planche.png'}")


def comptes(out: Path) -> None:
    from reader_dots_abst import cp  # Clopper-Pearson pur Python (ABST v2)

    rows = [json.loads(x) for x in open(out / "traits.jsonl")]
    n = len(rows)
    res = {"lignes_finales": n, "pages": len({r["page"] for r in rows})}
    for f in FAM + ("toutes",):
        k = sum((f in r["familles"]) if f != "toutes" else bool(r["familles"]) for r in rows)
        res[f] = {"k": k, "part": k / n, "ic95": cp(k, n)}
    res["deja_abstenues_sigma"] = sum(r["sigma"] < TAU for r in rows)
    res["par_page"] = {p: {f: sum(f in r["familles"] for r in rows if r["page"] == p) for f in FAM}
                       | {"n": sum(r["page"] == p for r in rows)} for p in sorted({r["page"] for r in rows})}
    # (1) caractères parasites (ajout pA après la planche) : strict = I4 ; large = I1-I4 lus en ≤ 2 caractères
    from reader_dots_abst import norm

    nb = lambda t: len("".join(norm(t or "").split()))  # noqa: E731
    juge = RUNS / "juge"
    vt = {p: sum(nb(tl["text"]) for tl in json.loads((juge / f"{p}.json").read_text())["true_lines"])
          for p in sorted({r["page"] for r in rows})}
    par = {}
    for niveau, sel in (("strict", lambda r: "I4" in r["familles"]),
                        ("large", lambda r: bool(set(r["familles"]) & {"I1", "I2", "I3", "I4"}) and nb(r["text"]) <= 2)):
        pp = {p: sum(nb(r["text"]) for r in rows if r["page"] == p and sel(r)) for p in vt}
        par[niveau] = {"lignes": sum(sel(r) for r in rows), "caracteres": sum(pp.values()),
                       "borne_cer_points_global": 100 * sum(pp.values()) / sum(vt.values()),
                       "borne_cer_points_par_page": {p: 100 * pp[p] / vt[p] for p in vt},
                       "borne_cer_points_page_max": max(100 * pp[p] / vt[p] for p in vt)}
    res["parasites"] = par
    res["caracteres_verite"] = sum(vt.values())
    res["note"] = "fragments lus par le lecteur du PIPELINE (C1+M+A), pas par dots guidé"
    (HERE / "reports/reader_illisibles_p07b.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "par_page"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etape", choices=("planche", "comptes"))
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    o = Path(a.sortie)
    o.mkdir(parents=True, exist_ok=True)
    (planche if a.etape == "planche" else comptes)(o)
