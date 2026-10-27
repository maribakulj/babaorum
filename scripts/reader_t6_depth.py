"""T6 — M2 (stabilité au décalage sous-jeton) et sonde linéaire de position par
profondeur de l'encodeur dots (docs/READER_T6_PROTOCOL.txt). Calcul lourd :

    ~/outils-seg/lourd.sh ~/axel/.venv/bin/python scripts/reader_t6_depth.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

AXEL = Path.home() / "axel"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))
from axel.image.crops import crop_line  # noqa: E402
from axel.image.lattice import encode  # noqa: E402
from axel.image.lattice import load as load_lat  # noqa: E402
from axel.reader.vit import MEAN, SPECS, STD, NaViT, prep, rename_map  # noqa: E402
from babaorum.reader.ancre import forced_align  # noqa: E402

TAPS = (8, 16, 24, 32, 42)
DOTS = Path.home() / "outils-seg/modeles/DotsMOCR"
E8LAT = AXEL / "runs/e1/mccatmus_press/lattices/press"
OUT = HERE / "reports/reader_t6.json"
Image.MAX_IMAGE_PIXELS = None


def load_encoder(dev: str) -> NaViT:
    from safetensors import safe_open

    s = SPECS["dots_mocr"]
    L = max(TAPS)
    m = NaViT(s["dim"], s["heads"], s["ffn"], L, s["rope"])
    wmap = json.load(open(DOTS / "model.safetensors.index.json"))["weight_map"]
    ren = rename_map(s["prefix"], L)
    by: dict[str, list[str]] = {}
    for k in ren:
        by.setdefault(wmap[k], []).append(k)
    sd = {}
    for f, ks in by.items():
        with safe_open(str(DOTS / f), framework="pt") as fh:
            for k in ks:
                sd[ren[k]] = fh.get_tensor(k)
    m.load_state_dict(sd, strict=False)
    return m.to(dev, torch.bfloat16).eval()


@torch.no_grad()
def feats(enc: NaViT, im: Image.Image, dev: str) -> dict[int, torch.Tensor]:
    x = prep(im, 56, 2800)
    w = x.shape[-1] // 14
    out = enc(x[None].to(dev, torch.bfloat16), TAPS, widths=[w])
    return {t: out[t][0].float().mean(0)[:w].cpu() for t in TAPS}


def labels_for(r: dict, crop_x0: float, crop_h: float, n_tok: int) -> list[str] | None:
    p = E8LAT / f"newseye_read__{r['page']}__{r['line_id']}.npz"
    if not p.exists():
        p = E8LAT / f"{r['corpus']}__{r['page']}__{r['line_id']}.npz"
    if not p.exists():
        return None
    lat = load_lat(p)
    ids = encode(lat, r["truth"])
    if not ids:
        return None
    sp = forced_align(lat.logprobs.astype(np.float64), ids)
    if not sp:
        return None
    lab = ["∅"] * n_tok
    for (t0, t1, _), c in zip(sp, ids):
        ch = lat.labels[c]
        if unicodedata.combining(ch[0]) or ch == " ":
            continue
        xp = float(lat.x[(t0 + t1) // 2])
        j = int((xp - crop_x0) * 56 / crop_h // 14)
        if 0 <= j < n_tok:
            lab[j] = ch.lower()
    return lab


def standardized_m2(pairs: dict[int, dict[int, list]], ks) -> dict:
    """Cosinus après standardisation par canal (moyenne/écart type estimés sur
    les jetons des lignes non décalées de la même profondeur)."""
    out = {}
    for t, byk in pairs.items():
        allf = torch.cat([a for a, _ in byk[ks[0]]])
        mu, sd = allf.mean(0), allf.std(0) + 1e-6
        out[t] = {}
        for k in ks:
            cs = [torch.nn.functional.cosine_similarity((a - mu) / sd, (b - mu) / sd, dim=-1).mean().item()
                  for a, b in byk[k]]
            out[t][k] = float(np.mean(cs))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14")
    ap.add_argument("--standardized-only", action="store_true", help="contrôle M2 standardisé, sans sonde")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [json.loads(x) for x in open(AXEL / "data/prepared/press/dev.jsonl")]
    rows = [r for r in rows if r["corpus"] == "newseye_read" and 20 <= len(r["truth"]) <= 120]
    rows = random.Random(6).sample(rows, 200)[: a.n]
    enc = load_encoder(dev)
    t0 = time.time()
    m2 = {t: {k: [] for k in ks} for t in TAPS}
    pairs = {t: {k: [] for k in ks} for t in TAPS}
    probe: list[tuple[dict[int, torch.Tensor], list[str]]] = []
    page_cache: dict[str, Image.Image] = {}
    for i, r in enumerate(rows):
        if r["image"] not in page_cache:
            page_cache.clear()
            page_cache[r["image"]] = Image.open(r["image"]).convert("L")
        lc = crop_line(page_cache[r["image"]], tuple(r["box"]), r.get("polygon"), pad=4)
        im = lc.image
        f0 = feats(enc, im, dev)
        for k in ks:
            sh = Image.new("L", (im.width + round(k * im.height / 56), im.height), 255)
            sh.paste(im, (round(k * im.height / 56), 0))
            fk = feats(enc, sh, dev)
            for t in TAPS:
                n = min(len(f0[t]), len(fk[t]))
                c = torch.nn.functional.cosine_similarity(f0[t][:n], fk[t][:n], dim=-1).mean().item()
                m2[t][k].append(c)
                if a.standardized_only:
                    pairs[t][k].append((f0[t][:n], fk[t][:n]))
        lab = labels_for(r, lc.x0, im.height, len(f0[TAPS[0]]))
        if lab is not None:
            probe.append(({t: f0[t].half() for t in TAPS}, lab))
        if i % 20 == 0:
            print(i, f"{time.time() - t0:.0f} s", flush=True)
    if a.standardized_only:
        rep0 = {"lines": len(rows), "m2_raw": {t: {k: float(np.mean(v)) for k, v in m2[t].items()} for t in TAPS},
                "m2_standardized": standardized_m2(pairs, ks), "seconds": time.time() - t0}
        (HERE / "reports/reader_t6_m2std.json").write_text(json.dumps(rep0, indent=1))
        print(json.dumps(rep0, indent=1))
        return
    rep: dict = {"lines": len(rows), "probe_lines": len(probe),
                 "m2": {t: {k: float(np.mean(v)) for k, v in m2[t].items()} for t in TAPS}}
    vocab = sorted({c for _, lab in probe for c in lab})
    vi = {c: i for i, c in enumerate(vocab)}
    tr, te = probe[:150], probe[150:]
    rep["probe"] = {}
    for t in TAPS:
        X = torch.cat([f[t].float() for f, _ in tr])
        Y = torch.tensor([vi[c] for _, lab in tr for c in lab])
        Xe = torch.cat([f[t].float() for f, _ in te])
        Ye = torch.tensor([vi[c] for _, lab in te for c in lab])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        torch.manual_seed(0)
        lin = torch.nn.Linear(X.shape[1], len(vocab))
        opt = torch.optim.Adam(lin.parameters(), 1e-3, weight_decay=1e-4)
        for _ in range(300):
            loss = torch.nn.functional.cross_entropy(lin((X - mu) / sd), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred = lin((Xe - mu) / sd).argmax(1)
        nonvoid = Ye != vi["∅"]
        rep["probe"][t] = {"acc": float((pred == Ye).float().mean()),
                           "acc_char": float((pred[nonvoid] == Ye[nonvoid]).float().mean()),
                           "tokens_test": int(len(Ye))}
        print(t, rep["probe"][t], flush=True)
    rep["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
