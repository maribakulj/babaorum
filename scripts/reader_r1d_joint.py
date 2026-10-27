"""R1d — faisceau joint sur deux treillis CTC (A1 dots, mccatmus_press kraken) + LM.

Au lieu de re-noter des N-best (R1b), le faisceau avance caractère par
caractère avec, pour chaque préfixe, le score de préfixe CTC exact de chaque
treillis (cadences et codecs différents) : S = a·A1 + (1−a)·E8 + λ·LM + β·|g|.
Candidats d'un pas : caractères du codec A1 dont la probabilité dépasse 1e-3
sur au moins un pas de l'un des deux treillis. Réglage sur tune (sous-ensemble),
mesure sur eval ; mêmes moitiés que R1/R1b. Lecture seule d'axel.

    ~/axel/.venv/bin/python scripts/reader_r1d_joint.py --part tune --n 200 --a 0.8 --lm 0.35 --beta 2
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))

from axel.image.fusion import EOS, LMCache, lm_score  # noqa: E402
from axel.image.lattice import CODEC_FOLD  # noqa: E402
from axel.image.lattice import load as load_lat  # noqa: E402
from axel.reader.metric import score  # noqa: E402
from axel.systems.charlm import CharLM  # noqa: E402,F401
from babaorum.reader.ancre import multi_ctc_beam  # noqa: E402

PRED = AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
TIMING = AXEL / "runs/e17/kaggle/a1/timing.json"
E8LAT = AXEL / "runs/e1/mccatmus_press/lattices/press"
LM = AXEL / "runs/e2/lm.press8x.pkl"


def page_of(stem: str) -> str:
    return stem.split("__")[1]


def encoder_for(labels: tuple[str, ...], nfd: bool):
    index = {c: i for i, c in enumerate(labels) if i}
    cache: dict[str, list[int] | None] = {}

    def enc(ch: str) -> list[int] | None:
        if ch in cache:
            return cache[ch]
        out: list[int] | None = None
        if not nfd and ch in index:
            out = [index[ch]]
        else:
            units = unicodedata.normalize("NFD", ch)
            if all(u in index for u in units):
                out = [index[u] for u in units]
            else:
                fold = CODEC_FOLD.get(ch)
                if fold and all(u in index for u in fold):
                    out = [index[u] for u in fold]
        cache[ch] = out
        return out

    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("tune", "eval"), default="tune")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--lm", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--corridor", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lm-path", default=str(LM))
    a = ap.parse_args()
    labels = ("",) + tuple(json.load(open(TIMING))["chars"])
    enc_a = encoder_for(labels, nfd=False)
    lmc = LMCache(pickle.load(open(a.lm_path, "rb")))

    def lm(text: str, c: str | None) -> float:
        hist = list(unicodedata.normalize("NFD", text))
        if c is None:
            return lm_score(lmc, EOS, hist)
        tot = 0.0
        for u in unicodedata.normalize("NFD", c):
            tot += lm_score(lmc, u, hist)
            hist.append(u)
        return tot

    rows = [json.loads(x) for x in gzip.open(PRED, "rt")]
    pages = sorted({page_of(r["stem"]) for r in rows})
    random.Random(1).shuffle(pages)
    tp = set(pages[: len(pages) // 2])
    rows = [r for r in rows if (page_of(r["stem"]) in tp) == (a.part == "tune")]
    if a.n:
        rows = random.Random(5).sample(rows, min(a.n, len(rows)))
    t0 = time.time()
    pairs, outs = [], []
    for r in rows:
        la = np.asarray(r["logp"], dtype=np.float64)
        le_lat = load_lat(E8LAT / f"{r['stem']}.npz")
        le = le_lat.logprobs.astype(np.float64)
        enc_e = encoder_for(tuple(le_lat.labels), nfd=True)
        cand = {labels[i] for i in np.nonzero(la.max(0) > np.log(1e-3))[0] if i}
        cand |= {le_lat.labels[i] for i in np.nonzero(le.max(0) > np.log(1e-3))[0]
                 if i and not unicodedata.combining(le_lat.labels[i][0])}
        cand = sorted(c for c in cand if enc_a(c))
        hs = multi_ctc_beam([(la, a.a, enc_a), (le, 1 - a.a, enc_e)], cand, lm=lm, lm_weight=a.lm,
                            beta=a.beta, beam=a.beam, corridor=a.corridor,
                            max_len=la.shape[0] // 2 + 2)
        h = hs[0].text if hs else ""
        pairs.append((h, r["truth"]))
        outs.append({"stem": r["stem"], "truth": r["truth"], "joint": h, "a1": r["hyp"]})
    res = {"part": a.part, "n": len(rows), "cfg": vars(a), "joint": score(pairs),
           "a1_greedy": score((r["hyp"], r["truth"]) for r in rows), "seconds": time.time() - t0}
    print(json.dumps({k: (v["cer_d150"] if isinstance(v, dict) and "cer_d150" in v else v)
                      for k, v in res.items()}, ensure_ascii=False))
    out = HERE / "reports" / f"reader_r1d_{a.part}{a.tag}.json"
    res["rows"] = outs
    out.write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
