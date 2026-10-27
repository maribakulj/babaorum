"""R2b — apprentissage du LM neuronal caractère (CharLSTM 2×512) sur MPS.

Texte : sources de press8x (press/train + BnL, impresso, ICDAR 2017 de train,
filtre D-82 d'axel) + l'échantillon Gallica de R2 (mêmes filtres que R2a :
documents à 8-gramme commun avec press/dev écartés). Une ligne = une
séquence (BOS … EOS). Validation : vérité de press/dev en bits/caractère
(c'est un dev, déjà exposé ; elle sert à l'arrêt, pas à un verdict).
Calcul lourd : ~/outils-seg/lourd.sh.

    ~/outils-seg/lourd.sh ~/axel/.venv/bin/python scripts/reader_r2b_lm.py --minutes 45
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch

AXEL = Path.home() / "axel"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(AXEL / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from babaorum.reader.neural_lm import BOS, EOS, PAD, CharLSTM  # noqa: E402

SAMPLE = Path.home() / "corpus-vt" / "french-pd-sample"
OUT = SAMPLE / "charlstm.pt"


def grams(text: str, n: int = 8) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def corpus(gallica_chars: int) -> tuple[list[str], list[str]]:
    import e2_fusion

    base = e2_fusion.lm_lines(("press", "bnl", "impresso", "icdar2017"))
    lines = [unicodedata.normalize("NFC", " ".join(t.split())) for src in base.values() for t in src if t.strip()]
    dev = [json.loads(x)["truth"] for x in open(AXEL / "data/prepared/press/dev.jsonl")]
    banned = set()
    for t in dev:
        banned |= grams(t)
    n = 0
    with gzip.open(SAMPLE / "text.jsonl.gz", "rt") as f:
        for line in f:
            text = unicodedata.normalize("NFC", json.loads(line)["text"])
            if grams(text) & banned:
                continue
            for ln in text.split("\n"):
                ln = " ".join(ln.split())
                if len(ln) >= 12 and sum(c.isalpha() for c in ln) >= 0.6 * len(ln):
                    lines.append(ln)
                    n += len(ln)
            if n >= gallica_chars:
                break
    return lines, [unicodedata.normalize("NFC", " ".join(t.split())) for t in dev if t.strip()]


def batchify(model: CharLSTM, lines: list[str], bs: int, maxlen: int = 200):
    random.shuffle(lines)
    lines = sorted(lines[: len(lines)], key=len)
    chunks = [lines[i : i + bs] for i in range(0, len(lines), bs)]
    random.shuffle(chunks)
    for ch in chunks:
        seqs = [[BOS] + model.ids(t[:maxlen]) + [EOS] for t in ch]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), PAD)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        yield x


def bits(model: CharLSTM, lines: list[str], dev: str) -> float:
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for x in batchify(model, list(lines), 64):
            x = x.to(dev)
            lp, _ = model(x[:, :-1])
            y = x[:, 1:]
            m = y != PAD
            tot += -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[m].sum().item()
            n += m.sum().item()
    model.train()
    return tot / n / math.log(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--gallica-chars", type=int, default=15_000_000)
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    lines, val = corpus(a.gallica_chars)
    cnt = Counter(c for t in lines for c in t)
    vocab = sorted(c for c, k in cnt.items() if k >= 20)
    model = CharLSTM(vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=0.01)
    print(f"{len(lines)} lignes, {sum(map(len, lines))} car., vocab {len(vocab)}, {dev}", flush=True)
    t0 = time.time()
    best = 1e9
    ep = 0
    while time.time() - t0 < a.minutes * 60:
        ep += 1
        for i, x in enumerate(batchify(model, lines, 128)):
            x = x.to(dev)
            lp, _ = model(x[:, :-1])
            loss = torch.nn.functional.nll_loss(lp.transpose(1, 2), x[:, 1:], ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if i % 500 == 0:
                print(ep, i, round(loss.item() / math.log(2), 3), f"{time.time() - t0:.0f} s", flush=True)
            if i % 2000 == 1999 or time.time() - t0 > a.minutes * 60:
                b = bits(model, val, dev)
                print("dev bits/car", round(b, 4), flush=True)
                if b < best:
                    best = b
                    torch.save({"vocab": vocab, "state": model.state_dict(), "dev_bits": b}, OUT)
                if time.time() - t0 > a.minutes * 60:
                    break
    print(json.dumps({"best_dev_bits": best, "minutes": (time.time() - t0) / 60, "out": str(OUT)}))


if __name__ == "__main__":
    main()
