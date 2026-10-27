"""NOTE-B-P08-instrument-E27 v2.8, § 1: one blind planche per drawn group (p1). No link, no oracle name, no measure on the planche.

Crop: the bounding box of the group's true and predicted lines and of the true lines that line_geometry matches to the group's predicted lines,
plus a margin of 3 median line heights (of the group's true lines). EVERY true line whose polygon meets the crop is numbered V1, V2... in the
order of its centroid (y, then x, integer pixels), light-grey outline; only the group's predicted lines are drawn, P1, P2... in the same order,
dark-grey outline. Texts in a legend under the image. Large group (crop wider than 1600 px at full resolution, or more than 20 lines): two halves,
top and bottom, same numbering and legend; a line meeting both halves is drawn on both. Light PNG (16 greys), at most 1600 px wide, one copy only.
Judges see only the opaque copies planches_juges/<k>.png (k from random.Random(20261027) over the unique groups of the stage); the manifest
(V/P numbering -> ids, opaque number -> group) is written OUTSIDE that folder. Stage 2 makes planches only for groups not yet made.
    AXEL_RUNS=<abs>/rappel_pages/runs_26/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v9_instrument_planches.py \
        --draws runs/p08v9_instrument/draws.json --stage 1 --out-dir runs/p08v9_instrument
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
dm = importlib.import_module("p08_decompose")  # pB's functions as they are
from axel.layout.gt import load_page  # noqa: E402
from babaorum.line_geometry import evaluate_line_geometry  # noqa: E402

MAX_W = 1600
MAX_LINES = 20
ORDER_SEED = 20261027
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
V_GREY, P_GREY = 165, 40


def bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def meets(b, box) -> bool:
    return not (b[2] < box[0] or b[0] > box[2] or b[3] < box[1] or b[1] > box[3])


def centroid_key(poly) -> tuple[int, int]:
    return int(sum(p[1] for p in poly) / len(poly)), int(sum(p[0] for p in poly) / len(poly))


def numbering(lines: list, box) -> list:
    """Lines (objects with .id, .polygon) meeting the box, in centroid order (y, then x)."""
    kept = [ln for ln in lines if ln.polygon and meets(bbox(ln.polygon), box)]
    return sorted(kept, key=lambda ln: (centroid_key(ln.polygon), ln.id))


def crop_box(group_v, group_p, lg_v, width, height):
    polys = [ln.polygon for ln in group_v + group_p + lg_v if ln.polygon]
    bs = [bbox(p) for p in polys]
    h = statistics.median([bbox(ln.polygon)[3] - bbox(ln.polygon)[1] for ln in group_v if ln.polygon] or [30])
    m = 3 * h
    return (max(0, min(b[0] for b in bs) - m), max(0, min(b[1] for b in bs) - m),
            min(width, max(b[2] for b in bs) + m), min(height, max(b[3] for b in bs) + m))


def halves(box, n_lines: int) -> list:
    if box[2] - box[0] <= MAX_W and n_lines <= MAX_LINES:
        return [box]
    mid = (box[1] + box[3]) / 2
    return [(box[0], box[1], box[2], mid), (box[0], mid, box[2], box[3])]


def render(img: Image.Image, box, vs: list, ps: list, legend: list[str], font) -> Image.Image:
    crop = img.crop(tuple(int(round(v)) for v in box)).convert("L")
    d = ImageDraw.Draw(crop)
    ox, oy = box[0], box[1]
    for lines, grey, width, tag in ((vs, V_GREY, 2, "V"), (ps, P_GREY, 3, "P")):
        for k, ln in lines:
            pts = [(x - ox, y - oy) for x, y in ln.polygon]
            if not meets(bbox(ln.polygon), box):
                continue
            d.line(pts + [pts[0]], fill=grey, width=width)
            x0, y0, x1, y1 = bbox(pts)
            lx, ly = (x0 - 4, y0 - 2) if tag == "V" else (x1 + 4, y1 - 20)
            label = f"{tag}{k}"
            tw = d.textlength(label, font=font)
            lx = min(max(lx, 0), crop.width - tw - 2)                       # labels always on the canvas (Sonnet F5)
            ly = min(max(ly, 0), crop.height - font.size - 2)
            tb = d.textbbox((lx, ly), label, font=font)
            d.rectangle(tb, fill=255)
            d.text((lx, ly), label, fill=0 if tag == "P" else 90, font=font)
    if crop.width > MAX_W:
        crop = crop.resize((MAX_W, int(crop.height * MAX_W / crop.width)))
    wrapped = [w for line in legend for w in (textwrap.wrap(line, 150) or [""])]
    lh = font.size + 6
    out = Image.new("L", (crop.width, crop.height + lh * (len(wrapped) + 1)), 255)
    out.paste(crop, (0, 0))
    dl = ImageDraw.Draw(out)
    for i, line in enumerate(wrapped):
        dl.text((8, crop.height + 6 + i * lh), line, fill=0, font=font)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir).resolve()
    draws = json.loads(Path(a.draws).read_text())
    pop = draws["population"]
    manifest_p = out / f"planches_manifest_stage{a.stage}.json"
    if manifest_p.exists():
        raise SystemExit(f"{manifest_p} exists, refused")
    done = set()
    if a.stage == 2:
        done = {int(g) for g in json.loads((out / "planches_manifest_stage1.json").read_text())["groups"]}
    wanted = sorted(set(draws[f"stage{a.stage}"]) - done)
    order = list(wanted)
    random.Random(ORDER_SEED).shuffle(order)                                   # the seed of the note, for both stages
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    e32 = importlib.import_module("e32")
    bench = e32.BENCHES["dev"]
    font = ImageFont.truetype(FONT, 18)
    (out / "planches_juges").mkdir(parents=True, exist_ok=True)
    manifest = {"stage": a.stage, "draws_sha": hashlib.sha256(Path(a.draws).read_bytes()).hexdigest(),
                "script_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "groups": {}, "opaque": {}}
    cache = {}
    for k, gi in enumerate(order, start=1 + (100 if a.stage == 2 else 0)):
        g = pop[gi]
        name = g["page"]
        if name not in cache:
            gt = load_page(Path(pages[name]["xml"]))
            arms, _ = e32.build_arms(bench, "c1", name)
            units = arms["ma"].units
            lg = evaluate_line_geometry(dm.gt_geo_of(gt.lines()), dm.pred_geo_of(units))
            cache = {name: (gt, units, {m.pred_id: m.gt_id for m in lg.matches})}
        gt, units, lg_of = cache[name]
        gby = {ln.id: ln for ln in gt.lines()}
        uby = {u.id: u for u in units}
        gv = [gby[i] for i in g["gt_ids"]]
        gp = [uby[i] for i in g["pred_ids"]]
        if not all(ln.polygon for ln in gv + gp):                               # fail before any judge (Sonnet F9)
            raise SystemExit(f"group {gi} of {name}: a line without polygon")
        lgv = [gby[lg_of[p.id]] for p in gp if p.id in lg_of]
        img = Image.open(pages[name]["image"])
        box = crop_box(gv, gp, lgv, img.width, img.height)
        vs = list(enumerate(numbering(gt.lines(), box), start=1))
        ps = list(enumerate(sorted(gp, key=lambda u: (centroid_key(u.polygon), u.id)), start=1))
        legend = (["Lignes de référence (V), texte de référence :"] + [f"V{i} : {ln.text}" for i, ln in vs]
                  + ["", "Lignes lues par la machine (P), texte lu :"] + [f"P{i} : {u.text}" for i, u in ps])
        files = []
        for h, part in enumerate(halves(box, len(gv) + len(gp))):
            suffix = "" if (box == part) else ("a" if h == 0 else "b")
            pic = render(img, part, vs, ps, legend, font).convert("P", palette=Image.ADAPTIVE, colors=16)   # light: 16 greys
            stem = f"{k}{suffix}"
            pic.save(out / "planches_juges" / f"{stem}.png", optimize=True)    # one copy only (disk); the manifest maps it to the group
            files.append(f"{stem}.png")
        vnum = {ln.id: i for i, ln in vs}
        lg_pairs = {str(i): vnum[lg_of[u.id]] for i, u in ps if u.id in lg_of and lg_of[u.id] in vnum}   # for the reported LG accuracy
        manifest["groups"][str(gi)] = {"opaque": k, "page": name, "V": {str(i): ln.id for i, ln in vs},
                                       "P": {str(i): u.id for i, u in ps}, "LG": lg_pairs, "files": files}
        manifest["opaque"][str(k)] = gi
    manifest_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"stage": a.stage, "planches": len(order), "manifest_sha": hashlib.sha256(manifest_p.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
