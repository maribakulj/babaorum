"""R1b — consensus de deux lecteurs CTC par re-notation d'hypothèses (CPU).

Lecteurs : E17 A1 (encodeur dots L = 32 + BiLSTM) et mccatmus_press (kraken
E8), mêmes 3 740 lignes de press/dev. Candidats = N-best de chacun (faisceau
16, fusion LM gelée d'E8 pour kraken, faisceau sans LM + avec LM pour A1).
Chaque candidat h reçoit S(h) = a·log P_A1(h) + (1−a)·log P_E8(h)
+ λ·log P_LM(h) + β·|h| ; aucun texte n'est écrit qui ne soit un pic d'un
des deux treillis. Réglage sur la moitié « tune » (pages tirées, graine 1,
comme R1), chiffres sur « eval ». Lecture seule d'axel.

    ~/axel/.venv/bin/python scripts/reader_r1b_consensus.py cands   # coûteux
    ~/axel/.venv/bin/python scripts/reader_r1b_consensus.py tune
"""

from __future__ import annotations

import gzip
import json
import math
import pickle
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))

from axel.image.fusion import EOS, FusionConfig, LMCache, decode, lm_score  # noqa: E402
from axel.image.lattice import BLANK, CODEC_FOLD, Lattice, ctc_logprob, encode  # noqa: E402
from axel.image.lattice import load as load_lat  # noqa: E402
from axel.reader.metric import line_errors, score  # noqa: E402
from axel.systems.charlm import CharLM  # noqa: E402,F401

PRED = AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
TIMING = AXEL / "runs/e17/kaggle/a1/timing.json"
E8LAT = AXEL / "runs/e1/mccatmus_press/lattices/press"
LM = AXEL / "runs/e2/lm.press8x.pkl"
HERE = Path(__file__).resolve().parents[1]
CANDS = HERE / "reports" / "reader_r1b_cands.jsonl.gz"
OUT = HERE / "reports" / "reader_r1b.json"
SEED = 1


def enc_nfc(index: dict[str, int], text: str) -> list[int] | None:
    """Codec d'A1 (classes NFC) : caractère tel quel, sinon NFD, sinon repli."""
    out: list[int] = []
    for ch in unicodedata.normalize("NFC", text):
        if ch in index:
            out.append(index[ch])
            continue
        units = unicodedata.normalize("NFD", ch)
        if all(u in index for u in units):
            out += [index[u] for u in units]
            continue
        fold = CODEC_FOLD.get(ch)
        if fold and all(u in index for u in fold):
            out += [index[u] for u in fold]
            continue
        return None
    return out


def lm_total(lm: LMCache, text: str) -> float:
    hist: list[str] = []
    tot = 0.0
    for ch in unicodedata.normalize("NFD", text):
        tot += lm_score(lm, ch, hist)
        hist.append(ch)
    return tot + lm_score(lm, EOS, hist)


def cmd_cands() -> None:
    t0 = time.time()
    labels = ("",) + tuple(json.load(open(TIMING))["chars"])
    index = {c: i for i, c in enumerate(labels) if i != BLANK}
    lm = LMCache(pickle.load(open(LM, "rb")))
    cfg_a_nolm = FusionConfig(beam=16, nbest=8)
    cfg_a_lm = FusionConfig(beam=16, nbest=8, lm_weight=0.35, insertion_bonus=0.5)
    cfg_e8 = FusionConfig(beam=16, nbest=8, lm_weight=0.5, insertion_bonus=3.0)
    n = miss = 0
    with gzip.open(PRED, "rt") as f, gzip.open(CANDS, "wt") as g:
        for line in f:
            r = json.loads(line)
            stem = r["stem"]
            p = E8LAT / f"{stem}.npz"
            if not p.exists():
                miss += 1
                continue
            la = Lattice(np.asarray(r["logp"], dtype=np.float16),
                         np.arange(len(r["logp"]), dtype=np.float32), 1.0, labels)
            le = load_lat(p)
            texts = {r["hyp"]}
            for lat, cfg, use in ((la, cfg_a_nolm, None), (la, cfg_a_lm, lm), (le, cfg_e8, lm)):
                texts.update(h.text for h in decode(lat, cfg, use))
            e8_greedy = decode(le, FusionConfig(beam=1, nbest=1), None)
            if e8_greedy:
                texts.add(e8_greedy[0].text)
            cands = []
            for t in texts:
                ia, ie = enc_nfc(index, t), encode(le, t)
                cands.append({
                    "t": t,
                    "a": ctc_logprob(la, ia) if ia is not None else -1e9,
                    "e": ctc_logprob(le, ie) if ie is not None else -1e9,
                    "lm": lm_total(lm, t),
                    "err": line_errors(t, r["truth"], "d150")[0],
                })
            g.write(json.dumps({"stem": stem, "truth": r["truth"], "a1": r["hyp"],
                                "e8": e8_greedy[0].text if e8_greedy else "",
                                "cands": cands}, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    print("lignes", n, "sans treillis E8", miss)


def page_of(stem: str) -> str:
    return stem.split("__")[1] if "__" in stem else stem


def cmd_tune() -> None:
    rows = [json.loads(x) for x in gzip.open(CANDS, "rt")]
    pages = sorted({page_of(r["stem"]) for r in rows})
    random.Random(SEED).shuffle(pages)
    tp = set(pages[: len(pages) // 2])
    tune = [r for r in rows if page_of(r["stem"]) in tp]
    ev = [r for r in rows if page_of(r["stem"]) not in tp]

    def pick(rs, a, lam, beta):
        out = []
        for r in rs:
            best = max(r["cands"], key=lambda c: a * c["a"] + (1 - a) * c["e"]
                       + lam * c["lm"] + beta * len(c["t"]))
            out.append((best["t"], r["truth"]))
        return out

    rep: dict = {"lines": len(rows), "tune": len(tune), "eval": len(ev)}
    for name, rs in (("tune", tune), ("eval", ev)):
        rep[f"a1_greedy_{name}"] = score((r["a1"], r["truth"]) for r in rs)["cer_d150"]
        rep[f"e8_greedy_{name}"] = score((r["e8"], r["truth"]) for r in rs)["cer_d150"]
        rep[f"oracle_pool_{name}"] = score(
            (min(r["cands"], key=lambda c: c["err"])["t"], r["truth"]) for r in rs)["cer_d150"]
    grid = {}
    for a in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        for lam in (0.0, 0.2, 0.35, 0.5):
            for beta in (0.0, 0.5, 1.0, 2.0, 3.0):
                grid[(a, lam, beta)] = score(pick(tune, a, lam, beta))["cer_d150"]
    best = min(grid, key=grid.get)
    rep["chosen"] = dict(zip(("a", "lm", "beta"), best))
    rep["tune_best"] = grid[best]
    rep["eval"] = score(pick(ev, *best))
    # par corpus et par moitié : glouton A1, glouton E8, consensus choisi, A1 seul + LM
    by: dict = {}
    for name, rs in (("tune", tune), ("eval", ev)):
        for corp in sorted({r["stem"].split("__")[0] for r in rs}):
            sub = [r for r in rs if r["stem"].startswith(corp + "__")]
            by[f"{name}/{corp}"] = {
                "lines": len(sub),
                "a1": score((r["a1"], r["truth"]) for r in sub)["cer_d150"],
                "e8": score((r["e8"], r["truth"]) for r in sub)["cer_d150"],
                "consensus": score(pick(sub, *best))["cer_d150"],
                "oracle_pool": score((min(r["cands"], key=lambda c: c["err"])["t"], r["truth"]) for r in sub)["cer_d150"],
            }
    rep["by_corpus"] = by
    # bras de contrôle sur eval, réglés chacun sur tune
    for name, keep in (("a1_only", lambda k: k[0] == 1.0), ("e8_only", lambda k: k[0] == 0.0),
                       ("no_lm", lambda k: k[1] == 0.0)):
        sub = {k: v for k, v in grid.items() if keep(k)}
        kb = min(sub, key=sub.get)
        rep[f"eval_{name}"] = {"cfg": kb, "cer_d150": score(pick(ev, *kb))["cer_d150"]}
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"cands": cmd_cands, "tune": cmd_tune}[sys.argv[1]]()
