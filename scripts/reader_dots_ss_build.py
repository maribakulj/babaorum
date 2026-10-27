"""Lot « second signal » (docs/READER_DOTS_SS_PREREG.txt ; CLASSE B) — entrées et carnet Kaggle du passage T4 : sorties du
guidé λ* = 0,3 (dispositif gelé, sans changement) sur les 200 négatifs SCRAMBLE de SÉLECTION (abst200, clé « #r-abst »).
Rien n'est lancé avant relecture Sonnet et audit pT.

    ~/axel/.venv/bin/python scripts/reader_dots_ss_build.py --inputs <dossier vide du scratchpad>

Réutilise le constructeur P3 (scripts/reader_dots_p3_build.py, commité, audité) : cellules CHRONO, SETUP, MODEL, IMAGES et
RUN, avec ces seuls remplacements vérifiés : panneau = 200 négatifs « neg » (mode scramble) ; contrôles de panneau = 200
tiges, couverture exacte des 200 treillis ; configurations des négatifs = guide_l0.3 seul ; pas d'éval. Les recadrages
d'origine sont reproduits sur Kaggle et contrôlés par MD5 avant et après perturbation (crop_md5, pert_md5, calculés ici).
Treillis A1 : reports/reader_a1_photo1983_abst_scramble.jsonl.gz (Mac, MPS bf16 ; même provenance que les négatifs P3).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
R = HERE / "reports"
NB = HERE / "scripts/reader_notebooks"
DATASET = "babaorum-dots-ss-inputs"

_spec = importlib.util.spec_from_file_location("p3", HERE / "scripts/reader_dots_p3_build.py")
p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3)
p1, sub = p3.p1, p3.sub

RUN = sub(p3.RUN, """assert len(stems) == len(set(stems)) == 3005, (len(stems), len(set(stems)))
assert sum(it['part'] == 'eval' for it in items) == 2405 and set(stems) == set(LAT), 'panneau ou treillis'""",
          """assert len(stems) == len(set(stems)) == 200, (len(stems), len(set(stems)))
assert all(it['part'] == 'neg' and it['mode'] == 'scramble' for it in items) and set(stems) == set(LAT), 'panneau'""")
RUN = sub(RUN, "CONFIGS_NEG = [('libre', None, 1), ('guide_l0.3', 0.3, 4)]",
          "CONFIGS_NEG = [('guide_l0.3', 0.3, 4)]  # second signal : guidé gelé seul, sur la sélection")


def notebook(manifest_sha: str) -> dict:
    head = ("# babaorum — second signal : guidé λ* = 0,3 sur 200 négatifs scramble de SÉLECTION\n\n"
            "docs/READER_DOTS_SS_PREREG.txt. Dispositif gelé inchangé ; T4 ×2 (« GPU T4 x2 », Internet ON) ; plafond dur 6 h.")
    return {"cells": [p1.md(head), p1.code(p3.CHRONO), p1.code(p1.SETUP.format(phase="EVAL", manifest_sha=manifest_sha)),
                      p1.code("%%bash\nset -e\npip install -q 'transformers==4.51.3' 'accelerate>=0.34' 2>&1 | tail -1"),
                      p1.code(p3.IMAGES), p1.code(p1.MODEL), p1.code(RUN)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args()
    out = Path(a.inputs)
    assert not out.exists() or not any(out.iterdir()), "dossier d'entrées non vide"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(AXEL / "src"))
    sys.path.insert(0, str(HERE / "src"))
    from PIL import Image

    from axel.image.crops import crop_line
    from babaorum.reader.perturb import perturb

    Image.MAX_IMAGE_PIXELS = None
    key = lambda r: r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"  # noqa: E731
    sel = [json.loads(x) for x in open(R / "reader_items_photo1983_abst200.jsonl")]
    pert = {key(json.loads(x)) for x in open(R / "reader_items_photo1983_pert200.jsonl")}
    assert len(sel) == 200 and not {key(r) for r in sel} & pert
    items = []
    for r in sel:
        with Image.open(r["image"]) as im:
            crop = crop_line(im.convert("L"), tuple(r["box"]), r.get("polygon"), pad=4).image
        p = perturb(crop, "scramble", key(r) + "#r-abst")
        items.append({"stem": f"{key(r)}#scramble", "part": "neg", "mode": "scramble", "src_stem": key(r),
                      "cle_perturbation": key(r) + "#r-abst", "image": "newseye/" + Path(r["image"]).name,
                      "box": r["box"], "polygon": r.get("polygon"), "truth": r["truth"],
                      "crop_md5": hashlib.md5(crop.tobytes()).hexdigest(), "pert_md5": hashlib.md5(p.tobytes()).hexdigest()})
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    want = {i["src_stem"] for i in items}
    got = 0
    with open(out / "a1.jsonl", "w") as g:
        for x in gzip.open(R / "reader_a1_photo1983_abst_scramble.jsonl.gz", "rt"):
            r = json.loads(x)
            if r.get("stem") in want:
                g.write(json.dumps({"stem": f"{r['stem']}#scramble", "logp": r["logp"]}) + "\n")
                got += 1
    assert got == 200, got
    shutil.copy(AXEL / "runs/e17/kaggle/a1/timing.json", out)
    for f in (HERE / "src/babaorum/reader/ctc_prefix.py", HERE / "src/babaorum/reader/dots_guided.py",
              HERE / "src/babaorum/reader/dots_beam.py", HERE / "src/babaorum/reader/perturb.py",
              AXEL / "src/axel/image/crops.py", AXEL / "src/axel/image/vlm.py", AXEL / "src/axel/image/lattice.py"):
        shutil.copy(f, out)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    manifest_sha = hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()
    (out / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"mabaku/{DATASET}",
                                                           "licenses": [{"name": "other"}]}))
    (NB / "kaggle_dots_ss_scramble.ipynb").write_text(json.dumps(notebook(manifest_sha), indent=1, ensure_ascii=False))
    print(json.dumps({"items": len(items), "a1": got, "manifest_sha256": manifest_sha}, indent=1))


if __name__ == "__main__":
    main()
