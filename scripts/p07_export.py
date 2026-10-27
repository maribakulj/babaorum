"""LOT-P07 v4/v5 exports: pool/<page>.jsonl (NO truth), <page>.json (no truth) and juge/<page>.json (truth, apart).

    AXEL_RUNS=<abs>/rappel_pages/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07_export.py

Read-only on every cache. Every exclusion motive is RECOMPUTED deterministically from the kept files with the code of the cited commits
(axel 39b169f, c723beb on top of E32); neither blla nor the reading is rerun. Each recomputation is cross-checked against the chain's own
output (selected ids, kept units); a mismatch stops the export. A motive that cannot be recomputed is written ``non_recuperable``.
Stable motive codes: zone level  : score_sous_0_3, etiquette_non_textuelle, doublon_fondu_80, membre_de_zone
                     D-563       : centre_dans_zone, recouvre_ligne_chaine, recouvre_candidate_acceptee
                     D-564       : hauteur_hors_intervalle, ligne_de_base_courte, alnum_insuffisant, confiance_sous_plancher, non_lue
                     dedup / M   : gardee, ecartee_par, centre_hors_rectangle_zone (M), bloc_ecarte_select_blocks, geometrie_invalide
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

AX = Path.home() / "axel"
sys.path[:0] = [str(AX / "src"), str(AX / "scripts"), str(Path(__file__).resolve().parents[1] / "src")]
RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
assert RUNS.is_absolute() and RUNS.name == "runs" and "rappel_pages" in str(RUNS)
R = RUNS.parent

from axel.layout import chains as C  # noqa: E402
from axel.layout import fill as F  # noqa: E402
from axel.layout.blocks import is_read, load_blocks  # noqa: E402
from axel.layout.chains import FROZEN, Params, bbox, select_blocks  # noqa: E402
from axel.layout.fill import baseline_cover, line_height, mean_conf, page_refs  # noqa: E402
from axel.layout.geom import box_area, box_inter  # noqa: E402
from axel.layout.segment import PAD, crop_window  # noqa: E402

assert C.RUNS == RUNS / "e12", "chains.RUNS must follow AXEL_RUNS (axel c723beb)"
e28 = importlib.import_module("e28_fill")
e32 = importlib.import_module("e32")
CODE = {"axel_commits": ["39b169f", "c723beb"], "constants": {"H_LOW": F.H_LOW, "H_HIGH": F.H_HIGH, "MIN_LEN": F.MIN_LEN, "MIN_ALNUM": F.MIN_ALNUM,
        "ALNUM_SHARE": F.ALNUM_SHARE, "CONF_PCT": F.CONF_PCT, "OVERLAP": F.OVERLAP}}


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tup(seq):
    return tuple((float(a), float(b)) for a, b in seq)


def centre_in(poly_box, rect) -> bool:
    cx, cy = (poly_box[0] + poly_box[2]) / 2, (poly_box[1] + poly_box[3]) / 2
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]


def covering(ln, kept):
    """Same test as chains._covered; returns the id of the covering kept line, or None."""
    bl = tup(ln["baseline"])
    tol = 0.5 * line_height(tup(ln["boundary"]), bl)
    x0, y0, x1, y1 = bbox(tup(ln["boundary"]))
    for k in kept:
        kb = k["_box"]
        if kb[0] < x1 and x0 < kb[2] and kb[1] < y1 and y0 < kb[3] and baseline_cover(bl, tup(k["baseline"]), tol) >= 0.5:
            return k["id"]
    return None


def assemble(blocks, with_m: bool):
    """Replicates detector_chain (with_m False) or e23_report.owned_chain (True): status of every line of the selected blocks."""
    status, kept = {}, []
    for b in blocks:
        rect = bbox(tup(b["polygon"]))
        for ln in b["lines"]:
            lid = ln["id"]
            if len(ln["baseline"]) < 2 or len(ln["boundary"]) < 3:
                status[lid] = {"status": "ecartee", "motif": "geometrie_invalide"}
                continue
            lb = bbox(tup(ln["boundary"]))
            if with_m and not centre_in(lb, rect):
                status[lid] = {"status": "ecartee", "motif": "centre_hors_rectangle_zone"}
                continue
            by = covering(ln, kept)
            if by is not None:
                status[lid] = {"status": "ecartee", "motif": "ecartee_par", "ecartee_par": by}
                continue
            kept.append({**ln, "_box": lb, "_block": b["id"]})
            status[lid] = {"status": "gardee", "motif": "gardee"}
    return status, kept


def zone_level(page, size):
    """Detections of e12/blocks/ppdl3 and what became of each one (D-219d)."""
    det = load_blocks(RUNS / "e12/blocks/ppdl3" / f"{page}.json", size)
    kept = select_blocks(det["blocks"], Params(conf=0.3, dedup=0.8))
    kept = [b for b in kept if is_read(b.get("label"))]
    kept_ids = {b["id"] for b in kept}
    z = json.loads((RUNS / "e12/lines/ppdl3_z" / f"{page}.json").read_text())["blocks"]
    member_of = {m: zb["id"] for zb in z for m in zb.get("members", ())}
    out = []
    for b in det["blocks"]:
        if b["id"] in kept_ids:
            motif = "membre_de_zone"
        elif b["score"] < 0.3:
            motif = "score_sous_0_3"
        elif not is_read(b.get("label")):
            motif = "etiquette_non_textuelle"
        else:
            motif = "doublon_fondu_80"
        out.append({"detection_id": b["id"], "label": b.get("label"), "category": b.get("category"), "score": b["score"],
                    "rank": b.get("rank"), "polygon": b["polygon"], "motif": motif, "zone": member_of.get(b["id"])})
    assert {o["detection_id"] for o in out if o["motif"] == "membre_de_zone"} == set(member_of), "zone members differ from the kept detections"
    return out


def select_with_reasons(cands, zones, kept):
    """D-563 replicated with the reason of each rejection; returns {id: (selected, motif, by)}."""
    taken = [(bbox(p), bl) for p, bl in kept if p and len(bl) >= 2]
    res = {}
    for c in cands:
        if F._centre_in(c.box, zones):
            res[c.id] = (False, "centre_dans_zone", None)
            continue
        hit = None
        cb, ca, tol = c.box, box_area(c.box), 0.5 * line_height(c.boundary, c.baseline)
        for k, (tb, tbl) in enumerate(taken):
            inter = box_inter(cb, tb)
            if inter <= 0:
                continue
            if inter > F.OVERLAP * min(ca, box_area(tb)) or baseline_cover(c.baseline, tbl, tol) >= 0.5:
                hit = k
                break
        if hit is not None:
            res[c.id] = (False, "recouvre_ligne_chaine" if hit < len(kept) else "recouvre_candidate_acceptee", hit)
            continue
        res[c.id] = (True, "selectionnee", None)
        taken.append((c.box, c.baseline))
    return res


def d564(c, text, conf, ref_h, floor):
    """Reasons of D-564 (same tests, same constants as fill.plausible)."""
    from axel.layout.pagetext import norm

    h = line_height(c.boundary, c.baseline)
    if ref_h > 0 and not (F.H_LOW * ref_h <= h <= F.H_HIGH * ref_h):
        return "hauteur_hors_intervalle"
    xs = [p[0] for p in c.baseline]
    if h <= 0 or max(xs) - min(xs) < F.MIN_LEN * h:
        return "ligne_de_base_courte"
    t = "".join(norm(text).split())
    alnum = sum(ch.isalnum() for ch in t)
    if alnum < F.MIN_ALNUM or alnum < F.ALNUM_SHARE * len(t):
        return "alnum_insuffisant"
    if conf is not None and conf < floor:
        return "confiance_sous_plancher"
    return None


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def main() -> int:
    from axel.layout.chains import Unit  # noqa: F401
    e23 = importlib.import_module("e23_report")
    from axel.layout.geom import bbox as gbbox  # noqa: F401
    from axel.layout.chains import FROZEN as FZ

    pages = json.loads((RUNS / "e12/pages.json").read_text())
    bench = e32.BENCHES["dev"]
    (R / "pool").mkdir(exist_ok=True)
    (R / "juge").mkdir(exist_ok=True)
    summary = {}
    for p in pages:
        name, W, H = p["page"], p["width"], p["height"]
        img_sha = sha(p["image"])
        data = json.loads((RUNS / "e12/lines/ppdl3_z" / f"{name}.json").read_text())
        blocks = select_blocks(data["blocks"], FZ["ppdl3_z"])
        blocks.sort(key=lambda b: b.get("rank", 0))
        kept_block_ids = {b["id"] for b in blocks}
        base_status, base_kept = assemble(blocks, False)
        m_status, m_kept = assemble(blocks, True)
        # cross-checks against the axel chains
        ref_base = C.detector_chain("ppdl3_z", name, FZ["ppdl3_z"])
        assert [u.id for u in ref_base.units] == [k["id"] for k in base_kept], f"{name}: base chain differs from the recomputed dedup"
        ref_m = e23.owned_chain("m", RUNS / "e12", "ppdl3_z", FZ["ppdl3_z"], name)
        assert [u.id for u in ref_m.units] == [k["id"] for k in m_kept], f"{name}: M chain differs from the recomputed rule"
        # C1 reads (all zone lines), E12 German reads are NOT used here
        c1 = json.loads((RUNS / "e22/read/ppdl3_z" / f"{name}.json").read_text())
        # rest candidates and their selections
        cands = e28.candidates("rest_z", name)
        base_zone_boxes = [b.box for b in ref_base.blocks]
        sel_a = select_with_reasons(cands, base_zone_boxes, [(u.polygon, u.baseline) for u in ref_base.units])
        # D-603: M+A is selected against the ZONES OF THE BASE and the LINES OF M
        # D-603 (2): only the candidates whose centre is in their rest region enter the M+A selection (e32.candidates member=True)
        rest_rect = {blk["id"]: bbox(tup(blk["polygon"])) for blk in json.loads((RUNS / "e28/lines/rest_z" / f"{name}.json").read_text())["blocks"]}
        member_c = [c for c in cands if centre_in(c.box, rest_rect[c.group])]
        sel_ma = {c.id: (False, "centre_hors_region_reste", None) for c in cands}
        sel_ma.update(select_with_reasons(member_c, base_zone_boxes, [(u.polygon, u.baseline) for u in ref_m.units]))
        chain_a = {l["id"] for l in json.loads((RUNS / "e28/add/a_z" / f"{name}.json").read_text())["lines"]}
        chain_ma = {l["id"] for l in json.loads((RUNS / "e32/dev/add/a_m" / f"{name}.json").read_text())["lines"]}
        assert {i for i, v in sel_a.items() if v[0]} == chain_a, f"{name}: D-563 (A) recomputation differs from add/a_z"
        assert {i for i, v in sel_ma.items() if v[0]} == chain_ma, f"{name}: D-563 (M+A) recomputation differs from add/a_m"
        # texts: selected lines from the chain reads, the others from read_rest_all (v5)
        chain_texts, chain_conf = {}, {}
        for d in (RUNS / "e28/read/mccatmus_press/a_z", RUNS / "e32/dev/read/mccatmus_press/extra"):
            j = read_json(d / f"{name}.json")
            if j:
                for k, v in j["texts"].items():
                    chain_texts.setdefault(k, v)
                for k, v in (j.get("ctc") or {}).items():
                    chain_conf.setdefault(k, v.get("conf"))
        extra = read_json(R / "read_rest_all" / f"{name}.json") or {"texts": {}, "ctc": {}}
        texts = {**extra["texts"], **chain_texts}
        confs = {**{k: v.get("conf") for k, v in extra["ctc"].items()}, **chain_conf}
        cuts = {**{k: v.get("cuts") for k, v in extra["ctc"].items()}}
        # D-564 references: A uses the base chain (C1 reads), M+A uses M
        def refs(units):
            lines = [(u.polygon, u.baseline) for u in units]
            cf = [mean_conf((c1["ctc"].get(u.id) or {}).get("conf")) for u in units]
            return page_refs(lines, cf)
        ref_a, ref_ma = refs(ref_base.units), refs(ref_m.units)
        arms = e32.build_arms(bench, "c1", name)[0]
        final = {"c1": {u.id for u in arms["base"].units}, "c1+m": {u.id for u in arms["m"].units},
                 "c1+a": {u.id for u in arms["a"].units}, "c1+ma": {u.id for u in arms["ma"].units}}

        def filt(c, arm_sel, refs_):
            sel = arm_sel[c.id][0]
            t = texts.get(c.id)
            if t is None:
                return {"status": "non_lue", "motif": "non_lue"}
            m = d564(c, t, mean_conf(confs.get(c.id)), *refs_)
            code = "d564_hypothetique" if not sel else None
            return {"status": ("gardee" if m is None else "exclue"), "motif": m or "ok", "hypothetique": (not sel), "code": code}

        recs = []
        zone_crop_cache = {}

        def zcrop(box):
            key = tuple(box)
            if key not in zone_crop_cache:
                w = crop_window(box, W, H, PAD)
                zone_crop_cache[key] = None if w is None else {
                    "x0": w.x0, "y0": w.y0, "x1": w.x1, "y1": w.y1, "marge_px": PAD, "canvas_h": w.canvas_h, "input_h": w.input_h,
                    "complete_de_blanc": w.padded, "facteur_echelle": w.input_h / w.canvas_h, "regle": "D-210/D-214, hauteur bornee a [600, 2400]"}
            return zone_crop_cache[key]

        def line_crop(ln):
            b = gbbox(tup(ln["boundary"]))
            return {"box_page": [b[0], b[1], b[2], b[3]], "marge_px": 4,
                    "box_recadree": [max(0, int(b[0]) - 4), max(0, int(b[1]) - 4), min(W, int(b[2]) + 4), min(H, int(b[3]) + 4)]}

        for zb in data["blocks"]:
            zbox = bbox(tup(zb["polygon"]))
            for ln in zb["lines"]:
                lid = ln["id"]
                bs = base_status.get(lid) or {"status": "ecartee", "motif": "bloc_ecarte_select_blocks"}
                ms = m_status.get(lid) or {"status": "ecartee", "motif": "bloc_ecarte_select_blocks"}
                recs.append({
                    "page": name, "image_sha256": img_sha, "candidate_id": f"zone:{lid}", "line_id": lid, "origin": "zone",
                    "zone_id": zb["id"], "zone_members": zb.get("members"), "zone_selected_by_select_blocks": zb["id"] in kept_block_ids,
                    "baseline": ln["baseline"], "polygon": ln["boundary"], "line_crop": line_crop(ln), "zone_crop": zcrop(zbox),
                    "m_membership": {"rule": "D-328b/D-603", **ms},
                    "dedup_covered": {"rule": "chains._covered", **bs},
                    "d563": None, "d564": {"status": "non_applicable", "motif": "ligne_de_zone"},
                    "text": c1["texts"].get(lid), "conf": (c1["ctc"].get(lid) or {}).get("conf"), "cuts": (c1["ctc"].get(lid) or {}).get("cuts"),
                    "text_motif": None if lid in c1["texts"] else "non_lue",
                    "in_final": {k: (lid in v) for k, v in final.items()}})
        rest = json.loads((RUNS / "e28/lines/rest_z" / f"{name}.json").read_text())["blocks"]
        by_id = {c.id: c for c in cands}
        for rb in rest:
            rbox = bbox(tup(rb["polygon"]))
            for ln in rb["lines"]:
                lid = ln["id"]
                c = by_id.get(lid)
                if c is None:
                    recs.append({"page": name, "image_sha256": img_sha, "candidate_id": f"rest:{lid}", "line_id": lid, "origin": "rest",
                                 "zone_id": rb["id"], "baseline": ln["baseline"], "polygon": ln["boundary"], "d563": {"status": "non_recuperable",
                                 "motif": "geometrie_invalide"}, "in_final": {k: False for k in final}})
                    continue
                sa, sm = sel_a[lid], sel_ma[lid]
                recs.append({
                    "page": name, "image_sha256": img_sha, "candidate_id": f"rest:{lid}", "line_id": lid, "origin": "rest",
                    "zone_id": rb["id"], "baseline": ln["baseline"], "polygon": ln["boundary"], "line_crop": line_crop(ln), "zone_crop": zcrop(rbox),
                    "m_membership": {"rule": "D-328b/D-603", "status": "non_applicable", "motif": "ligne_du_reste"},
                    "dedup_covered": {"rule": "chains._covered", "status": "non_applicable", "motif": "ligne_du_reste"},
                    "d563": {"arm_a": {"selected": sa[0], "motif": sa[1]}, "arm_ma": {"selected": sm[0], "motif": sm[1]}},
                    "d564": {"arm_a": filt(c, sel_a, ref_a), "arm_ma": filt(c, sel_ma, ref_ma)},
                    "text": texts.get(lid), "conf": confs.get(lid), "cuts": cuts.get(lid), "text_motif": None if lid in texts else "non_lue",
                    "in_final": {k: (lid in v) for k, v in final.items()}})
        # final cross-check: the recomputed D-564 equals what the chains kept, for the selected lines
        for arm, key, sel in (("arm_a", "c1+a", chain_a), ("arm_ma", "c1+ma", chain_ma)):
            for r in recs:
                if r["origin"] == "rest" and r["line_id"] in sel:
                    kept_by_filter = r["d564"][arm]["status"] == "gardee"
                    assert kept_by_filter == r["in_final"][key] or r["d564"][arm]["status"] == "non_lue", (
                        f"{name}: D-564 recomputation differs from {key} for {r['line_id']}")
        with (R / "pool" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        det = zone_level(name, (W, H))
        (R / f"{name}.json").write_text(json.dumps({
            "page": name, "image": p["image"], "image_sha256": img_sha, "identity": {"doc_key": p["page"].split("_")[-2], "issue_key": None,
            "note": "doc_key = doc_id NewsEye du nom de fichier ; issue_key non établie (voir MANIFESTE)"},
            "code": CODE, "zones": [{"zone_id": zb["id"], "polygon": zb["polygon"], "label": zb.get("label"), "rank": zb.get("rank"),
                                     "members": zb.get("members"), "selected_by_select_blocks": zb["id"] in kept_block_ids} for zb in data["blocks"]],
            "detections_zone_level": det,
            "lines_c1_ma": [r for r in recs if r["in_final"]["c1+ma"]],
            "pool_file": f"pool/{name}.jsonl", "truth": "absente de ce fichier : voir juge/ (non transmis)"}, ensure_ascii=False) + "\n", encoding="utf-8")
        summary[name] = {"pool": len(recs), "zone_lines": sum(r["origin"] == "zone" for r in recs), "rest_lines": sum(r["origin"] == "rest" for r in recs),
                         "detections": len(det), "zone_lines_kept_base": sum(r["dedup_covered"]["status"] == "gardee" for r in recs if r["origin"] == "zone"),
                         "c1+ma": len(final["c1+ma"]), "rest_without_text": sum(r.get("text_motif") == "non_lue" for r in recs if r["origin"] == "rest")}
        print(name, summary[name], flush=True)
    (R / "export_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
