"""Lecture A1 (E17 : encodeur dots.mocr coupé à 32 blocs + tête BiLSTM) sur le Mac.

Produit, pour des lignes données (JSONL d'axel : image, box, polygon, truth),
le treillis complet d'A1 au format de dev_pred_a1 (stem, truth, hyp, logp
float16). Les poids sont lus directement dans les deux fichiers de
~/outils-seg/modeles/DotsMOCR (pas de copie : disque presque plein). Le code
de lecture est celui d'axel (lecture seule) : axel.reader.vit/ctc, crops.
Calcul lourd : toujours par ~/outils-seg/lourd.sh.

    ~/outils-seg/lourd.sh ~/axel/.venv/bin/python scripts/reader_a1_read.py \\
        --items ~/axel/data/prepared/press/dev.jsonl --n 24 --out reports/reader_a1_check.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))

from axel.image.crops import crop_line  # noqa: E402
from axel.reader.ctc import Head, read_lines, white_value  # noqa: E402
from axel.reader.vit import MEAN, SPECS, STD, NaViT, prep, rename_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from babaorum.reader.perturb import perturb  # noqa: E402

DOTS = Path.home() / "outils-seg/modeles/DotsMOCR"
HEAD = AXEL / "runs/e17/kaggle/a1/head_a1.pt"
TIMING = AXEL / "runs/e17/kaggle/a1/timing.json"
L = 32
Image.MAX_IMAGE_PIXELS = None


def load_encoder(dtype: torch.dtype, device: str) -> NaViT:
    from safetensors import safe_open

    s = SPECS["dots_mocr"]
    m = NaViT(s["dim"], s["heads"], s["ffn"], L, s["rope"])
    wmap = json.load(open(DOTS / "model.safetensors.index.json"))["weight_map"]
    ren = rename_map(s["prefix"], L)
    by_file: dict[str, list[str]] = {}
    for k in ren:
        by_file.setdefault(wmap[k], []).append(k)
    sd = {}
    for f, keys in by_file.items():
        with safe_open(str(DOTS / f), framework="pt") as fh:
            for k in keys:
                sd[ren[k]] = fh.get_tensor(k)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if k != "inv_freq"], (missing, unexpected)
    return m.to(device, dtype).eval()


def stem_of(r: dict) -> str:
    return r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--perturb", default="", help="mode de négatif (babaorum.reader.perturb)")
    ap.add_argument("--suffix", default="#r-gate")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.items)]
    if a.n:
        rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    chars = json.load(open(TIMING))["chars"]
    dtype = torch.bfloat16 if a.device != "cpu" else torch.float32
    enc = load_encoder(dtype, a.device)
    head = Head(len(chars) + 1, up=2)
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    head = head.to(a.device).eval()
    white = white_value(MEAN, STD)
    t0 = time.time()
    pages: dict[str, Image.Image] = {}
    rows.sort(key=lambda r: r["image"])
    done = 0
    with gzip.open(a.out, "wt") as g:
        batch: list[tuple[dict, torch.Tensor]] = []

        def flush() -> None:
            nonlocal done
            if not batch:
                return
            with torch.no_grad():
                outs = read_lines(enc, head, [x for _, x in batch], L, chars, white, a.device)
            for (r, _), (hyp, lp) in zip(batch, outs):
                g.write(json.dumps({"stem": stem_of(r), "truth": r["truth"], "hyp": hyp,
                                    "logp": lp.float().cpu().half().numpy().tolist()},
                                   ensure_ascii=False) + "\n")
            done += len(batch)
            batch.clear()
            print(done, f"{time.time() - t0:.0f} s", flush=True)

        for r in rows:
            if r["image"] not in pages:
                pages.clear()
                with Image.open(r["image"]) as im:
                    pages[r["image"]] = im.convert("L")
            page = pages[r["image"]]
            im = page if r.get("corpus") == "bnl_pack" else crop_line(page, tuple(r["box"]), r.get("polygon"), pad=4).image
            if a.perturb:
                donor = (crop_line(page, tuple(r["donor_box"]), r.get("donor_polygon"), pad=4).image
                         if "donor_box" in r else None)  # ligne donneuse (mode occlude_donor seulement)
                im = perturb(im, a.perturb, stem_of(r) + a.suffix, donor=donor)
            batch.append((r, prep(im, 56, 2800)))
            if len(batch) >= a.bs:
                flush()
        flush()
    print(json.dumps({"lines": done, "s_per_line": (time.time() - t0) / max(1, done), "device": a.device}))


if __name__ == "__main__":
    main()
