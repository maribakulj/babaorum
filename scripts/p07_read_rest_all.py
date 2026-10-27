"""LOT-P07 v5: read with mccatmus_press ALL the rest-ink candidates of lines/rest_z that D-563 did not select.

Run through lourd.sh with the kraken Python, AXEL_RUNS exported (absolute):
    ~/outils-seg/lourd.sh ~/outils-seg/kraken-env/bin/python scripts/p07_read_rest_all.py

Reads under <AXEL_RUNS>/e28/lines/rest_z and the selections e28/add/a_z, e32/dev/add/{a_m,a_m_nomb}; writes ONLY
<rappel_pages>/read_rest_all/<page>.json (never read/, never add/, never ~/axel/runs). Nothing here changes a chain output or a measure.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

AX = Path.home() / "axel"
sys.path.insert(0, str(AX / "src"))
sys.path.insert(0, str(AX / "scripts"))
RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
assert RUNS.is_absolute() and RUNS.name == "runs", "AXEL_RUNS must be an absolute path ending in /runs (Mac: rappel_pages/runs, Kaggle: /kaggle/working/runs)"
OUT = RUNS.parent / "read_rest_all"
READER = "mccatmus_press"  # the six pages are all after 1837 (axel 39b169f)


def main() -> int:
    import torch
    from PIL import Image

    from axel.image.kraken_read import load_reader, read_lines, write_json_atomic
    from axel.pipeline.route import READERS

    Image.MAX_IMAGE_PIXELS = None
    torch.set_num_threads(4)
    pages = json.loads((RUNS / "e12/pages.json").read_text())
    reader = None
    for p in pages:
        name = p["page"]
        dest = OUT / f"{name}.json"
        if dest.exists():
            continue
        cand = [ln for b in json.loads((RUNS / f"e28/lines/rest_z/{name}.json").read_text())["blocks"] for ln in b["lines"]]
        selected = set()
        for sel in (f"e28/add/a_z/{name}.json", f"e32/dev/add/a_m/{name}.json", f"e32/dev/add/a_m_nomb/{name}.json"):
            f = RUNS / sel
            if f.exists():
                selected |= {ln["id"] for ln in json.loads(f.read_text())["lines"]}
        todo = [ln for ln in cand if ln["id"] not in selected]
        if reader is None:
            reader = load_reader(AX / READERS[READER], "cpu", name=READER)
        with Image.open(p["image"]) as raw:
            page = raw.convert("L")
        t0 = time.time()
        res = read_lines(reader, page, todo, page_id=name, source="p07_rest_all")
        dt = time.time() - t0
        OUT.mkdir(parents=True, exist_ok=True)
        write_json_atomic(dest, {**res.as_json(reader=READER, seconds=round(dt, 1)),
                                 "candidates": len(cand), "selected_by_chain": len(selected & {c['id'] for c in cand}), "read_here": len(todo)})
        print(f"{name}: {len(todo)} candidats non sélectionnés lus sur {len(cand)} en {dt:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
