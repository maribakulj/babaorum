"""T2 — marges exactes de séquence, bornes « 3 mots » (docs/READER_T2_PROTOCOL.txt).

Treillis A1 (runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz) et kraken
(runs/e1/mccatmus_press/lattices/press/*.npz) de press/dev, lecture seule.
Flux ligne à ligne. Sortie : reports/reader_t2_rows.jsonl.gz (une ligne par
ligne de dev) puis résumé par scripts/reader_t2_report.py.

    ~/axel/.venv/bin/python scripts/reader_t2_margins.py [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))
from axel.image.lattice import Lattice, ctc_logprob  # noqa: E402
from axel.image.lattice import load as load_lat  # noqa: E402
from axel.reader.metric import line_errors  # noqa: E402
from babaorum.reader.ancre import forced_align  # noqa: E402

PRED = AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
TIMING = AXEL / "runs/e17/kaggle/a1/timing.json"
E8LAT = AXEL / "runs/e1/mccatmus_press/lattices/press"
OUT = HERE / "reports/reader_t2_rows.jsonl.gz"


def greedy_ids(lp: np.ndarray) -> list[int]:
    a = lp.argmax(1)
    out, prev = [], -1
    for c in a:
        if c != prev and c != 0:
            out.append(int(c))
        prev = c
    return out


def word_index(units: list[str]) -> list[int]:
    """Indice de mot de chaque unité (−1 pour une espace)."""
    w, out, inword = -1, [], False
    for u in units:
        if u.isspace():
            out.append(-1)
            inword = False
        else:
            if not inword:
                w += 1
                inword = True
            out.append(w)
    return out


def margins(lp: np.ndarray, labels: tuple[str, ...]) -> dict | None:
    lat = Lattice(lp.astype(np.float16), np.arange(len(lp), dtype=np.float32), 1.0, labels)
    ids = greedy_ids(lp)
    if not ids:
        return None
    base = ctc_logprob(lat, ids)
    spans = forced_align(lp, ids)
    if not spans:
        return None
    m, alt = [], []
    for i, (t0, t1, _) in enumerate(spans):
        pool = np.argsort(-lp[t0 : t1 + 1].max(0))
        cands = [int(c) for c in pool if c != 0 and c != ids[i]][:4]
        scores = [ctc_logprob(lat, ids[:i] + [c] + ids[i + 1 :]) for c in cands]
        j = int(np.argmax(scores))
        m.append(base - scores[j])
        alt.append(cands[j])
    units = [labels[c] for c in ids]
    return {"ids": ids, "units": units, "m": m, "alt": alt, "words": word_index(units), "base": base, "lat": lat}


def bounds(units: list[str], words: list[int], m: list[float]) -> dict:
    per_word: dict[int, float] = {}
    for w, x in zip(words, m):
        if w >= 0:
            per_word[w] = min(per_word.get(w, 1e9), x)
    wl: dict[int, int] = {}
    for w in words:
        if w >= 0:
            wl[w] = wl.get(w, 0) + 1
    if len(per_word) < 3:
        return {"S_w": None, "S_Nk": None, "n_words": len(per_word)}
    s_w = float(sum(sorted(per_word.values())[:3]))
    nk = sum(sorted(wl.values())[:3])
    s_nk = float(sum(sorted(x for w, x in zip(words, m) if w >= 0)[:nk]))
    return {"S_w": s_w, "S_Nk": s_nk, "n_words": len(per_word), "Nk": nk}


def additivity(r: dict) -> dict | None:
    """Perte exacte d'une substitution simultanée de 3 positions (la plus faible
    de chacun des 3 mots les plus faibles) contre la somme des marges isolées."""
    best: dict[int, int] = {}
    for i, (w, x) in enumerate(zip(r["words"], r["m"])):
        if w >= 0 and (w not in best or x < r["m"][best[w]]):
            best[w] = i
    if len(best) < 3:
        return None
    pos = sorted(best.values(), key=lambda i: r["m"][i])[:3]
    ids = list(r["ids"])
    for i in pos:
        ids[i] = r["alt"][i]
    joint = r["base"] - ctc_logprob(r["lat"], ids)
    return {"sum": float(sum(r["m"][i] for i in pos)), "joint": float(joint)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    la_labels = ("",) + tuple(json.load(open(TIMING))["chars"])
    t0 = time.time()
    n = 0
    with gzip.open(OUT, "wt") as g:
        for line in gzip.open(PRED, "rt"):
            r = json.loads(line)
            row: dict = {"stem": r["stem"], "truth": r["truth"]}
            for name, get in (("a1", lambda: (np.asarray(r["logp"], np.float64), la_labels)),
                              ("kr", lambda: (lambda L: (L.logprobs.astype(np.float64), tuple(L.labels)))(
                                  load_lat(E8LAT / f"{r['stem']}.npz")))):
                try:
                    lp, labels = get()
                except FileNotFoundError:
                    continue
                res = margins(lp, labels)
                if res is None:
                    continue
                hyp = "".join(res["units"])
                e, nn = line_errors(hyp, r["truth"], "d150")
                row[name] = {"hyp": hyp, "err": e, "n": nn, "m": [round(x, 3) for x in res["m"]],
                             "words": res["words"], **bounds(res["units"], res["words"], res["m"])}
                if e == 0 and n % 4 == 0:
                    add = additivity(res)
                    if add:
                        row[name]["add"] = add
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 250 == 0:
                print(n, f"{time.time() - t0:.0f} s", flush=True)
            if a.limit and n >= a.limit:
                break
    print("fini", n, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
