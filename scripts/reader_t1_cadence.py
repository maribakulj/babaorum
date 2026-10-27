"""T1 (conseil w2:pA §3) — faisabilité CTC par cadence, sans modèle.

Pour chaque ligne de press/dev : largeur exactement comme axel.reader.vit.prep
(arrondi, minimum 28, plafond 2 800, toile complétée au multiple de 28), trames T = W / pas pour un pas de 28, 14 et 7 px ;
contrainte CTC : T ≥ U + R (U = caractères de la vérité NFC, R = paires de
caractères adjacents identiques, qui exigent un blanc entre eux). Fraction de
lignes infaisables par pas. Lecture seule d'axel ; aucune image.
"""
import json
import math
import unicodedata
from pathlib import Path

from PIL import Image

AXEL = Path.home() / "axel"
rows = [json.loads(x) for x in open(AXEL / "data/prepared/press/dev.jsonl")]
sizes = {}


def image_size(path):
    # seule la taille est lue, pas les pixels
    if path not in sizes:
        with Image.open(path) as im:
            sizes[path] = im.size
    return sizes[path]


def clamp_box(box, width, height, pad=0):
    # copie de axel.image.crops.clamp_box : +pad, +1, borné à l'image
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, x0 - pad))
    y0 = max(0, min(height - 1, y0 - pad))
    x1 = max(x0 + 1, min(width, x1 + pad + 1))
    y1 = max(y0 + 1, min(height, y1 + pad + 1))
    return x0, y0, x1, y1


res = {}
for step in (28, 14, 7):
    bad = n = chars_bad = chars = 0
    for r in rows:
        t = unicodedata.normalize("NFC", " ".join(r["truth"].split()))
        if not t:
            continue
        img_w, img_h = image_size(r["image"])
        x0, y0, x1, y1 = clamp_box(r["box"], img_w, img_h, pad=4)  # axel.image.crops.crop_line
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        w56 = min(2800, max(28, round(w * 56 / h)))  # axel.reader.vit.prep
        W = math.ceil(w56 / 28) * 28  # toile complétée au multiple de 28
        T = W // step
        need = len(t) + sum(a == b for a, b in zip(t, t[1:]))
        n += 1
        chars += len(t)
        if T < need:
            bad += 1
            chars_bad += len(t)
    res[step] = {"lines": n, "infeasible_lines": bad, "frac_lines": bad / n, "frac_chars": chars_bad / chars}
print(json.dumps(res, indent=1))
(Path(__file__).resolve().parents[1] / "reports/reader_t1_cadence.json").write_text(json.dumps(res, indent=1))
