"""Treillis kraken (mccatmus_press d'E8) de lignes données, écrits dans babaorum.

Même sortie que reader_a1_read.py (stem, truth, hyp, logp float16) plus le
codec en première ligne ({"labels": [...]}). Code de lecture d'axel
(axel.image.kraken_read, normalisation D-190), lecture seule ; Python de
kraken. Calcul lourd : par ~/outils-seg/lourd.sh.

    ~/outils-seg/lourd.sh ~/outils-seg/kraken-env/bin/python scripts/reader_kraken_read.py \\
        --items reports/reader_items_photo1983.jsonl --out reports/reader_kraken_photo1983.jsonl.gz
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))
from PIL import Image  # noqa: E402

from axel.image.crops import crop_line  # noqa: E402
from axel.image.kraken_read import decode, load_reader, read_line  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from babaorum.reader.perturb import perturb  # noqa: E402

MODEL = AXEL / "runs/e8/models/mccatmus_press.mlmodel"
Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perturb", default="")
    ap.add_argument("--suffix", default="#r-gate")
    a = ap.parse_args()
    rows = sorted((json.loads(x) for x in open(a.items)), key=lambda r: r["image"])
    reader = load_reader(MODEL)
    t0 = time.time()
    page, path = None, None
    with gzip.open(a.out, "wt") as g:
        first = True
        for i, r in enumerate(rows):
            if r["image"] != path:
                path = r["image"]
                with Image.open(path) as im:
                    page = im.convert("L")
            item = dict(r, source=r.get("corpus", ""))
            stem = r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"
            if a.perturb:
                crop = crop_line(page, tuple(r["box"]), r.get("polygon"), pad=4).image
                donor = (crop_line(page, tuple(r["donor_box"]), r.get("donor_polygon"), pad=4).image
                         if "donor_box" in r else None)  # ligne donneuse (mode occlude_donor seulement)
                im = perturb(crop, a.perturb, stem + a.suffix, donor=donor)
                item = dict(item, box=[0, 0, im.width, im.height], polygon=None)
                lat = read_line(reader, im, item)
            else:
                lat = read_line(reader, page, item)
            if first and lat is not None:
                g.write(json.dumps({"labels": list(lat.labels)}, ensure_ascii=False) + "\n")
                first = False
            stem = r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"
            g.write(json.dumps({"stem": stem, "truth": r["truth"],
                                "hyp": decode(lat).text if lat is not None else None,
                                "logp": lat.logprobs.astype("float16").tolist() if lat is not None else None},
                               ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(i, f"{time.time() - t0:.0f} s", flush=True)
    print(json.dumps({"lines": len(rows), "s": time.time() - t0}))


if __name__ == "__main__":
    main()
