"""R2a — 6-gramme caractère enrichi de presse Gallica (CPU).

LM = sources de press8x (vérité press/train + BnL, impresso, ICDAR 2017 de
train, avec le filtre D-82 d'axel) + ``--gallica-chars`` caractères de
l'échantillon French-PD-Newspapers (note OCR ≥ 95). Un document Gallica qui
partage un 8-gramme de mots avec la vérité de press/dev est écarté (les
dépêches d'agence se réimpriment d'un journal à l'autre). Aucune vérité de
test n'est lue ici. Sortie : ~/corpus-vt/french-pd-sample/lm.press8x_gallica.pkl

    ~/outils-seg/lourd.sh ~/axel/.venv/bin/python scripts/reader_r2_lm.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(AXEL / "scripts"))

from axel.image.fusion import EOS, lm_text  # noqa: E402
from axel.systems.charlm import CharLM  # noqa: E402

SAMPLE = Path.home() / "corpus-vt" / "french-pd-sample"


def grams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallica-chars", type=int, default=15_000_000)
    ap.add_argument("--order", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    import e2_fusion  # axel, lecture seule

    base = e2_fusion.lm_lines(("press", "bnl", "impresso", "icdar2017"), verbose=True)
    dev = [json.loads(x)["truth"] for x in open(AXEL / "data/prepared/press/dev.jsonl")]
    banned: set[tuple[str, ...]] = set()
    for t in dev:
        banned |= grams(t)
    extra: list[str] = []
    n = dropped = 0
    with gzip.open(SAMPLE / "text.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            text = unicodedata.normalize("NFC", r["text"])
            if grams(text) & banned:
                dropped += 1
                continue
            for ln in text.split("\n"):
                ln = " ".join(ln.split())
                if len(ln) < 12 or sum(c.isalpha() for c in ln) < 0.6 * len(ln):
                    continue
                extra.append(ln)
                n += len(ln)
            if n >= a.gallica_chars:
                break
    print(f"Gallica : {len(extra)} lignes, {n} caractères, {dropped} documents écartés (8-grammes dev)", flush=True)
    lm = CharLM(a.order)
    for src in base.values():
        lm.train(lm_text(t) + EOS for t in src if t.strip())
    lm.train(lm_text(t) + EOS for t in extra)
    out = SAMPLE / "lm.press8x_gallica.pkl"
    with open(out, "wb") as fh:
        pickle.dump(lm, fh)
    print("écrit", out, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
