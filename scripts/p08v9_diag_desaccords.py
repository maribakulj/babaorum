"""P-08 v9, note de classe B après le résultat sur runs_26 : diagnostic des pages écartées pour désaccord (p1, lecture seule).

Réutilise les fonctions de pB TELLES QUELLES (scripts/p08_decompose.py, src/babaorum/p08_oracles.py) ; seul le contrôle « ordre vrai »
est désactivé ici, par argument, pour obtenir les parts de ces pages en sensibilité. Rien n'est écrit hors de --out-dir.
    AXEL_RUNS=<abs>/rappel_pages/runs_26/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v9_diag_desaccords.py \
        --decomposition runs/p08v9_runs26/decomposition.json --out-dir runs/p08v9_diag
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
dec_mod = importlib.import_module("p08_decompose")  # puts axel and babaorum on sys.path
from babaorum import p08_oracles  # noqa: E402
from babaorum.p08_oracles import MISSED_CLASSES, bootstrap_by_groups  # noqa: E402
from axel.layout.gt import load_page  # noqa: E402
from axel.layout.pagetext import score_page  # noqa: E402

PARTS = ("lecture", "partielles", "decoupe", "manquees", "insertions", "ordre")
# the three candidate sub-parts of the choice rule (NOTE-DECISION v4-v7), absent from the group bootstrap of p08_decompose
SUBPARTS = tuple(f"manquees_{c}" for c in MISSED_CLASSES)


def uncovered(glines, lg, units, regions, polys, zones, classes):
    """True lines that no oracle touches, with what E27 and line_geometry say about each."""
    from axel.eval.seg_errors import make_line
    G = [make_line(g.id, g.polygon, g.baseline, g.text) for g in glines]
    H = [make_line(u.id, u.polygon, u.baseline, u.text) for u in units]
    d = dec_mod.decompose(G, H, [r.polygon for r in regions])
    kind_gt, comp_hyp = {}, {}
    for c in d.components:
        for i in c.gt:
            kind_gt[glines[i].id] = c.kind
            comp_hyp[glines[i].id] = [units[j].id for j in c.hyp]
    groups_g = {glines[i].id for c in d.components if c.kind in dec_mod.GROUP_KINDS for i in c.gt}
    groups_p = {units[j].id for c in d.components if c.kind in dec_mod.GROUP_KINDS for j in c.hyp}
    partial_g = {glines[c.gt[0]].id for c in d.components if c.kind == dec_mod.PARTIAL}
    missed_g = {glines[i].id for c in d.components if c.kind == dec_mod.MISSED for i in c.gt}
    lg_of_gt = {m.gt_id: m.pred_id for m in lg.matches}
    matched_g = {m.gt_id for m in lg.matches if m.pred_id not in groups_p and m.gt_id not in groups_g}
    out = []
    for g in glines:
        if g.id in matched_g | groups_g | partial_g | missed_g:
            continue
        lp = lg_of_gt.get(g.id)
        out.append({"gt_id": g.id, "text": g.text, "e27_kind": kind_gt.get(g.id), "e27_component_pred": comp_hyp.get(g.id, []),
                    "line_geometry_pred": lp, "line_geometry_pred_in_e27_group": lp in groups_p if lp else None,
                    "polygon": g.polygon, "baseline": g.baseline})
    return out


def planche(image_path, gline, units, link_pred_ids, out_png):
    from PIL import Image, ImageDraw
    im = Image.open(image_path).convert("RGB")
    xs = [p[0] for p in gline["polygon"]]
    ys = [p[1] for p in gline["polygon"]]
    h = max(ys) - min(ys)
    box = (max(0, min(xs) - 6 * h), max(0, min(ys) - 4 * h), min(im.width, max(xs) + 6 * h), min(im.height, max(ys) + 4 * h))
    dr = ImageDraw.Draw(im)
    for u in units:
        if u.polygon and any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in u.polygon):
            colour = (255, 140, 0) if u.id in link_pred_ids else (0, 120, 255)
            dr.line([*map(tuple, u.polygon), tuple(u.polygon[0])], fill=colour, width=3)
    dr.line([*map(tuple, gline["polygon"]), tuple(gline["polygon"][0])], fill=(220, 0, 0), width=4)
    if gline["baseline"]:
        dr.line(list(map(tuple, gline["baseline"])), fill=(0, 170, 0), width=3)
    crop = im.crop(tuple(int(v) for v in box))
    crop.save(out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decomposition", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise SystemExit(f"--out-dir exists, refused: {out}")
    out.mkdir(parents=True)
    done = json.loads(Path(args.decomposition).read_text())
    apart = sorted(done["disagreement_pages_reported_apart"])
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    R = RUNS.parent
    e32 = importlib.import_module("e32")
    bench = e32.BENCHES["dev"]
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    # the oracles with the true-order control off: parts of the pages set apart, for sensitivity only
    dec_mod.decompose_page = functools.partial(p08_oracles.decompose_page, check_true_order=False)
    report = {"pages_apart": {}, "sensitivity": {}}
    extra_rows = {"c1": [], "c1+ma": []}
    for name in apart:
        p = pages[name]
        gt = load_page(Path(p["xml"]))
        glines = gt.lines()
        arms, zones = e32.build_arms(bench, "c1", name)
        polys = dec_mod.truth_polys(gt)
        pool = [json.loads(l) for l in (R / "pool" / f"{name}.jsonl").open(encoding="utf-8")]
        classes = dec_mod.missed_classes(glines, dec_mod.pool_existence(glines, pool), zones)
        page_rep = {"image": p.get("image"), "xml": p["xml"], "chains": {}}
        for chain, arm in (("c1", "base"), ("c1+ma", "ma")):
            units = arms[arm].units
            dec, lg, info = dec_mod.analyse(units, glines, gt.text_regions, zones, polys, missed_class=classes)
            chars = score_page([u.text for u in units], [g.text for g in glines]).chars
            unc = uncovered(glines, lg, units, gt.text_regions, polys, zones, classes)
            for k, gl in enumerate(unc):
                if not gl["polygon"]:  # no polygon: no planche, the line is still reported
                    gl["planche"] = None
                    gl.pop("polygon"), gl.pop("baseline")
                    continue
                png = out / f"{name}_{chain.replace('+', '')}_{k}.png"
                planche(p["image"], gl, units, set(gl["e27_component_pred"]) | {gl["line_geometry_pred"]}, png)
                gl["planche"] = str(png)
                gl.pop("polygon"), gl.pop("baseline")
                pred_text = {u.id: u.text for u in units}
                gl["e27_component_pred_text"] = [pred_text[i] for i in gl["e27_component_pred"]]
                gl["line_geometry_pred_text"] = pred_text.get(gl["line_geometry_pred"])
            page_rep["chains"][chain] = {"true_order_cer": dec.true_order_cer, "cer": dec.cer, "parts": dec.parts, "chars": chars,
                                         "uncovered_true_lines": unc, "links": info["linked"]}  # left_without_link: preds with no link
            extra_rows[chain].append({"page": name, "chars": chars, "cer": dec.cer, "parts": dec.parts})
        report["pages_apart"][name] = page_rep
    groups = done["groups"]
    for chain in ("c1", "c1+ma"):
        kept = [{"page": n, "chars": v[chain]["forward"]["chars"], "cer": v[chain]["forward"]["cer"], "parts": v[chain]["forward"]["parts"]}
                for n, v in done["pages"].items()]
        both = {"sans_les_pages_ecartees": kept, "avec_les_pages_ecartees": kept + extra_rows[chain]}
        report["sensitivity"][chain] = {}
        for label, rows in both.items():
            w = sum(r["chars"] for r in rows)
            agg = {"pages": len(rows), "cer_0": sum(r["chars"] * r["cer"]["0"] for r in rows) / w}
            flat = []
            for r in rows:
                f = {"page": r["page"], "group": tuple(groups[r["page"]]), "chars": r["chars"], **{k: r["parts"].get(k, 0.0) for k in PARTS + SUBPARTS}}
                f["ordre"] = r["cer"]["4"]
                flat.append(f)
            for k in PARTS + SUBPARTS:
                agg[k] = sum(f["chars"] * f[k] for f in flat) / w
            agg["bootstrap_by_groups"] = bootstrap_by_groups(flat, list(PARTS + SUBPARTS))
            report["sensitivity"][chain][label] = agg
    (out / "diag.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({n: {c: [(u["gt_id"], u["e27_kind"], u["line_geometry_pred"]) for u in v["chains"][c]["uncovered_true_lines"]]
                          for c in v["chains"]} for n, v in report["pages_apart"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
