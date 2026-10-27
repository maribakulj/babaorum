"""R1 — décodages du treillis d'E17 A1 sur press/dev (CPU, sans GPU).

Question : sur le lecteur à encodeur dots (L = 32, A1), combien rapporte un
LM caractère fusionné dans le faisceau CTC, et quelle marge reste pour un LM
plus fort (oracle N-best = meilleure hypothèse du faisceau choisie avec la
vérité) ? Le dev est coupé en deux moitiés par PAGE (graine fixe) : réglage
sur « tune », chiffres rapportés sur « eval ». Lecture seule d'axel.

    ~/axel/.venv/bin/python scripts/reader_r1_decode.py [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))

from axel.image.fusion import FusionConfig, LMCache, decode  # noqa: E402
from axel.image.lattice import Lattice  # noqa: E402
from axel.reader.metric import score  # noqa: E402
from axel.systems.charlm import CharLM  # noqa: E402,F401  (dépicklage)

PRED = AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
TIMING = AXEL / "runs/e17/kaggle/a1/timing.json"
LM = AXEL / "runs/e2/lm.press8x.pkl"
OUT = Path(__file__).resolve().parents[1] / "reports" / "reader_r1.json"
SEED = 1


def load(limit: int | None) -> list[dict]:
    rows = []
    with gzip.open(PRED, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "stem": r["stem"],
                    "page": r["stem"].split("__")[1] if "__" in r["stem"] else r["stem"],
                    "truth": r["truth"],
                    "hyp": r["hyp"],
                    "logp": np.asarray(r["logp"], dtype=np.float16),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    pages = sorted({r["page"] for r in rows})
    random.Random(SEED).shuffle(pages)
    tune_pages = set(pages[: len(pages) // 2])
    tune = [r for r in rows if r["page"] in tune_pages]
    ev = [r for r in rows if r["page"] not in tune_pages]
    return tune, ev


def run(rows, labels, cfg, lm):
    best, oracle = [], []
    for r in rows:
        lat = Lattice(r["logp"], np.arange(len(r["logp"]), dtype=np.float32), 1.0, labels)
        hyps = decode(lat, cfg, lm)
        texts = [h.text for h in hyps] or [""]
        best.append((texts[0], r["truth"]))
        if cfg.nbest > 1:
            from axel.reader.metric import line_errors

            oracle.append((min(texts, key=lambda t: line_errors(t, r["truth"], "d150")[0]), r["truth"]))
    out = {"best": score(best)}
    if oracle:
        out["oracle"] = score(oracle)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    t0 = time.time()
    rows = load(a.limit)
    labels = ("",) + tuple(json.load(open(TIMING))["chars"])
    assert rows[0]["logp"].shape[1] == len(labels), (rows[0]["logp"].shape, len(labels))
    import pickle

    lm = LMCache(pickle.load(open(LM, "rb")))
    tune, ev = split(rows)
    rep: dict = {"lines": len(rows), "tune_lines": len(tune), "eval_lines": len(ev)}
    rep["greedy_eval"] = score([(r["hyp"], r["truth"]) for r in ev])
    rep["greedy_tune"] = score([(r["hyp"], r["truth"]) for r in tune])
    nolm = FusionConfig(beam=16, nbest=16)
    rep["beam_nolm_eval"] = run(ev, labels, nolm, None)
    grid = {}
    for lam in (0.2, 0.35, 0.5, 0.8):
        for beta in (0.5, 1.5, 3.0):
            cfg = FusionConfig(beam=16, nbest=1, lm_weight=lam, insertion_bonus=beta)
            grid[f"{lam}/{beta}"] = run(tune, labels, cfg, lm)["best"]["cer_d150"]
            print(f"tune λ={lam} β={beta}: {grid[f'{lam}/{beta}']:.3f}", flush=True)
    rep["grid_tune_d150"] = grid
    lam, beta = map(float, min(grid, key=grid.get).split("/"))
    rep["chosen"] = {"lm_weight": lam, "insertion_bonus": beta}
    cfg = FusionConfig(beam=16, nbest=16, lm_weight=lam, insertion_bonus=beta)
    rep["fusion_eval"] = run(ev, labels, cfg, lm)
    e8 = FusionConfig(beam=16, nbest=1, lm_weight=0.5, insertion_bonus=3.0)
    rep["fusion_e8_frozen_eval"] = run(ev, labels, e8, lm)["best"]
    rep["seconds"] = time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "grid_tune_d150"}, indent=1))


if __name__ == "__main__":
    main()
