"""Writes the lines (id, polygon, baseline, text) of the chains c1 (base) and c1+ma of one page, from the caches under AXEL_RUNS.
Used on the Mac (reference for the portability control) and on Kaggle (same code, same page): no truth is read.

    AXEL_RUNS=<abs>/runs <axel-venv python> p07b_chain_units.py --axel <axel dir> --page La_Presse_0545076_T0000002 --out units.json
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--axel", required=True)
ap.add_argument("--page", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert os.environ.get("AXEL_RUNS", "").startswith("/"), "AXEL_RUNS must be absolute"
ax = Path(a.axel)
sys.path[:0] = [str(ax / "src"), str(ax / "scripts")]
e32 = importlib.import_module("e32")
arms, _zones = e32.build_arms(e32.BENCHES["dev"], "c1", a.page)
pack = lambda ch: [{"id": u.id, "polygon": [list(q) for q in u.polygon], "baseline": [list(q) for q in u.baseline], "text": u.text} for u in ch.units]
Path(a.out).write_text(json.dumps({"page": a.page, "c1": pack(arms["base"]), "c1+ma": pack(arms["ma"])}, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"c1": len(arms["base"].units), "c1+ma": len(arms["ma"].units)}))
