"""LOT-P08: where does the page CER go (oracles) and what limits the recall? Computed on files, no inference.

Dev n° 2 (P-07):
    AXEL_RUNS=<abs>/rappel_pages/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08_decompose.py
E12 dev control (c1+ma must give 0.05188611919580012; run WITHOUT AXEL_RUNS, ~/axel/runs read only):
    env -u AXEL_RUNS PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08_decompose.py --e12-control

Definitions are those of LOT-P08 v1-v3 (nothing is tuned). Writes rappel_pages/*.json (dev n° 2) or e12_control.json.
"""
from __future__ import annotations

import argparse
import re
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

AX = Path.home() / "axel"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AX / "src"), str(AX / "scripts"), str(ROOT / "src")]
SEED_BOOT, N_BOOT = 20261017, 10000

from axel.eval.seg_errors import (EXTRA_IN, EXTRA_OUT, MERGE_COL, MERGE_VERT, MISSED, PARTIAL, SPLIT, TANGLE,  # noqa: E402
                                  decompose, make_line)
from axel.layout.gt import load_page  # noqa: E402
from axel.layout.pagetext import score_page  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402
from babaorum.p08_oracles import FORWARD, MISSED_CLASSES, REVERSE, Group, Links, check_pages_for, decompose_page, disagreement_table, parts_sum_error  # noqa: E402

GROUP_KINDS = (SPLIT, MERGE_VERT, MERGE_COL, TANGLE)
RLG_CER = 0.10  # seuil de suspicion, défini avant la classe A (note P-08 v2.3 § 1)
PART_KEYS = ("lecture", "partielles", "decoupe", "manquees", "insertions", "ordre")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(pred_texts, gt_texts) -> float:
    return score_page(list(pred_texts), list(gt_texts)).cer_reordered


def height(poly):
    ys = [q[1] for q in poly]
    return max(ys) - min(ys)


def bbox(poly):
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    return min(xs), min(ys), max(xs), max(ys)


def centre(poly):
    b = bbox(poly)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def in_poly(pt, poly) -> bool:
    x, y = pt
    inside, n = False, len(poly)
    for i in range(n):
        (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
        if (ya > y) != (yb > y) and x < (xb - xa) * (y - ya) / (yb - ya) + xa:
            inside = not inside
    return inside


def in_box(pt, b) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def gt_geo_of(glines):
    return [GroundTruthLine(g.id, g.baseline, height(g.polygon), "ground_truth") if g.baseline and len(g.baseline) >= 2 and g.polygon
            else GroundTruthLine(g.id, None, None, None, "missing") for g in glines]


def pred_geo_of(units):
    return [PredictedLine(u.id, [list(q) for q in u.baseline], "predicted") if u.baseline and len(u.baseline) >= 2
            else PredictedLine(u.id, None, None, "missing") for u in units]


def rlg_resolve(groups, matches, pred_text, gt_text, missed, extra, partial=None):
    """R-LG v10.1 (note P-08 v2.3 § 1, avenant v2.4 : règle P-E27). Un groupe suspect (une ligne prédite appariée 1:1
    par line_geometry à une ligne vraie, CER de ligne <= RLG_CER) est dissous en entier.
    Paire NOUVELLE (au moins une ligne dissoute) : admise dans matched seulement si SES DEUX lignes sont dissoutes ;
    sinon refusée, et la ligne dissoute suit le sort 3 (V -> manquée, P -> en trop). Les paires v9 sont inchangées.
    missed, extra, partial : classements E27 d'origine, lus AVANT dissolution."""
    partial = dict(partial or {})
    partner = {m.pred_id: m for m in matches}
    def suspect(g):
        return any(p in partner and score([pred_text[p]], [gt_text[partner[p].gt_id]]) <= RLG_CER for p in g.pred_ids)
    dissolved = [g for g in groups if suspect(g)]
    kept = [g for g in groups if not suspect(g)]
    dissolved_p = {p for g in dissolved for p in g.pred_ids}
    dissolved_g = {x for g in dissolved for x in g.gt_ids}
    kept_p = {p for g in kept for p in g.pred_ids}
    kept_g = {x for g in kept for x in g.gt_ids}
    # arrêt : une ligne dissoute ne doit figurer dans aucun classement E27 d'origine
    if (dissolved_p & (set(partial) | set(partial.values()) | set(extra))) or (dissolved_g & (set(partial.values()) | set(missed))):
        raise RuntimeError("R-LG : une ligne dissoute figure dans partial, missed ou extra d'origine")
    matched, admitted, refused = {}, [], []
    for m in matches:
        p, g = m.pred_id, m.gt_id
        if p in kept_p or g in kept_g:
            continue                      # une ligne dans un groupe gardé : exclue, comme en v9
        if p not in dissolved_p and g not in dissolved_g:
            matched[p] = g                # paire v9 (aucune ligne dissoute) : inchangée
        elif p in dissolved_p and g in dissolved_g:
            matched[p] = g                # paire nouvelle, deux lignes dissoutes : admise
            admitted.append((p, g))
        else:
            refused.append((p, g))        # paire nouvelle, une ligne non dissoute : refusée (règle P-E27)
    if any((p in dissolved_p) != (g in dissolved_g) for p, g in matched.items() if p in dissolved_p or g in dissolved_g):
        raise RuntimeError("R-LG : une paire de matched touchant une ligne dissoute a une ligne non dissoute")
    matched_gt = set(matched.values())
    rlg_missed = [x for x in dissolved_g if x not in matched_gt]
    rlg_extra = [p for p in dissolved_p if p not in matched]
    if set(rlg_missed) & matched_gt or set(rlg_extra) & set(matched):
        raise RuntimeError("R-LG : une ligne issue d'un groupe dissous est à la fois appariée et manquée/en trop")
    to_matched = sorted(dissolved_g & matched_gt)
    gt_of_pred = dict(matched)
    info = {"groups_dissolved": len(dissolved), "dissolved_gt": sorted(dissolved_g), "dissolved_pred": sorted(dissolved_p),
            "pairs_added": len(admitted), "pairs_refused": len(refused),
            "to_matched_gt": to_matched, "to_missed_gt": sorted(rlg_missed), "to_extra_pred": sorted(rlg_extra),
            # rapport de masse (§ 4.3) : caractères VRAIS vers matched (lus juste ou non) et vers missed ; caractères PRÉDITS vers extra
            "chars_gt_to_matched": sum(len(gt_text[g]) for g in to_matched),
            "chars_gt_to_matched_read_right": sum(len(gt_text[g]) for g in to_matched
                                                  if any(gt_of_pred[p] == g and pred_text[p] == gt_text[g] for p in gt_of_pred)),
            "chars_gt_to_missed": sum(len(gt_text[g]) for g in rlg_missed),
            "lines_pred_to_extra": len(rlg_extra), "chars_pred_to_extra": sum(len(pred_text[p]) for p in rlg_extra)}
    return kept, matched, list(missed) + rlg_missed, list(extra) + rlg_extra, info


def analyse(units, glines, regions, zones, truth_regions, order=FORWARD, missed_class=None):
    """All links for one chain on one page, then the oracle decomposition."""
    lg = evaluate_line_geometry(gt_geo_of(glines), pred_geo_of(units))
    G = [make_line(g.id, g.polygon, g.baseline, g.text) for g in glines]
    H = [make_line(u.id, u.polygon, u.baseline, u.text) for u in units]
    d = decompose(G, H, [r.polygon for r in regions])
    kind_of_pred, groups, missed, extra, partial = {}, [], [], [], {}
    for c in d.components:
        for j in c.hyp:
            kind_of_pred[units[j].id] = c.kind
        if c.kind in GROUP_KINDS:
            groups.append(Group(c.kind, tuple(units[j].id for j in c.hyp), tuple(glines[i].id for i in c.gt)))
        elif c.kind == PARTIAL:
            assert len(c.gt) == 1 and len(c.hyp) == 1, "a partial component must link one predicted to one true line"
            partial[units[c.hyp[0]].id] = glines[c.gt[0]].id
        elif c.kind == MISSED:
            missed.extend(glines[i].id for i in c.gt)
        elif c.kind in (EXTRA_IN, EXTRA_OUT):
            extra.extend(units[j].id for j in c.hyp)
    groups, matched, missed, extra, rlg = rlg_resolve(groups, lg.matches, {u.id: u.text for u in units},
                                                       {g.id: g.text for g in glines}, missed, extra, partial)
    in_group_p = {p for g in groups for p in g.pred_ids}
    links = Links(matched, groups, missed, extra, partial)
    zone_of = {g.id: any(in_box(centre(g.polygon), z) for z in zones) for g in glines}
    dec = decompose_page([(u.id, u.text) for u in units], [(g.id, g.text) for g in glines], links, score, order=order,
                         zone_of_missed=lambda x: zone_of[x], missed_class=missed_class)
    lg_matched_pred = {m.pred_id for m in lg.matches}
    unmatched = [u for u in units if u.id not in lg_matched_pred]
    by_kind = {}
    for u in unmatched:
        k = kind_of_pred.get(u.id, "sans_composante")
        inside = any(in_poly(centre(u.polygon), poly) for poly in truth_regions)
        key = by_kind.setdefault(k, {"dans_region_avec_verite": 0, "hors_emprise": 0})
        key["dans_region_avec_verite" if inside else "hors_emprise"] += 1
    return dec, lg, {"unmatched_pred": len(unmatched), "unmatched_by_e27_kind": by_kind,
                     "linked": {"matched_1to1": len(matched), "groups": len(groups), "partial": len(partial), "missed": len(missed), "extra_pred": len(extra),
                                "left_without_link": sum(1 for u in units if u.id not in matched and u.id not in in_group_p and u.id not in set(extra) and u.id not in partial)},
                     "e27_kinds_gt": {k: sum(1 for c in d.components if c.kind == k) for k in {c.kind for c in d.components}},
                     "rlg": rlg}


def pool_existence(glines, pool_rows) -> set[str]:
    """LOT-P08 v9 (R_pool, existence): the true lines that at least one candidate of the pool pairs with (line_geometry.v1, 0.5 h, pairwise,
    box filter only to speed up). The box filter never removes a pair that could match."""
    from babaorum.line_geometry import GroundTruthLine, PredictedLine

    cands = [r for r in pool_rows if r.get("baseline") and len(r["baseline"]) >= 2]
    def bb(bl):
        xs, ys = [q[0] for q in bl], [q[1] for q in bl]
        return min(xs), min(ys), max(xs), max(ys)
    boxes = [bb(r["baseline"]) for r in cands]
    found = set()
    for g in glines:
        if not (g.baseline and len(g.baseline) >= 2 and g.polygon):
            continue
        gg = GroundTruthLine(g.id, g.baseline, height(g.polygon), "ground_truth")
        gb = bb(g.baseline)
        m = 2 * height(g.polygon) + 20
        for r, rb in zip(cands, boxes):
            if rb[2] < gb[0] - m or rb[0] > gb[2] + m or rb[3] < gb[1] - m or rb[1] > gb[3] + m:
                continue
            res = evaluate_line_geometry([gg], [PredictedLine(str(r["candidate_id"]).replace(":", "_"), r["baseline"], "predicted")])
            if res.matched_count == 1:
                found.add(g.id)
                break
    return found


def missed_classes(glines, existing, zones) -> dict[str, str]:
    """ecartees / dans_zone / hors_zone / sans_centre, in the order of NOTE-DECISION v4 (centre of the true line in a base zone rectangle)."""
    out = {}
    for g in glines:
        if g.id in existing:
            out[g.id] = "ecartees"
        elif not (g.polygon and len(g.polygon) >= 3 and height(g.polygon) > 0):
            out[g.id] = "sans_centre"
        else:
            c = centre(g.polygon)
            out[g.id] = "dans_zone" if any(in_box(c, z) for z in zones) else "hors_zone"
    return out


def truth_polys(gt):
    return [r.polygon for r in gt.text_regions if r.lines]


def restrict(units, polys):
    keep = [u for u in units if any(in_poly(centre(u.polygon), poly) for poly in polys)]
    return keep, len(units) - len(keep)


def aggregate(rows, key_fn):
    """Weighted by true characters. rows: [{chars, cer:{0..4}, parts:{...}}]."""
    w = np.array([r["chars"] for r in rows], dtype=float)
    out = {"cer_0": float((w * [r["cer"]["0"] for r in rows]).sum() / w.sum()), "cer_4": float((w * [r["cer"]["4"] for r in rows]).sum() / w.sum())}
    for k in key_fn(rows):
        out[k] = float((w * [r["parts"][k] for r in rows]).sum() / w.sum())
    return out


def bootstrap(rows, keys):
    rng = np.random.default_rng(SEED_BOOT)
    n = len(rows)
    w = np.array([r["chars"] for r in rows], dtype=float)
    arr = {k: np.array([r["parts"][k] for r in rows]) for k in keys if not k.startswith("cer_")}
    arr.update({k: np.array([r["cer"][k[4:]] for r in rows]) for k in keys if k.startswith("cer_")})
    draws = {k: [] for k in arr}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ww = w[idx]
        for k, v in arr.items():
            draws[k].append((ww * v[idx]).sum() / ww.sum())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def summarize(rows, with_ci=True):
    keys = ["cer_0", "lecture", "partielles", "decoupe", "manquees", "insertions", "ordre"]
    agg = aggregate(rows, lambda rs: ("lecture", "partielles", "decoupe", "manquees", "manquees_dans_zone", "manquees_hors_zone", "insertions", "ordre", "part_positions"))
    agg["pages"] = [r["page"] for r in rows]
    agg["share_of_cer_0"] = {k: (agg[k] / agg["cer_0"] if agg["cer_0"] else None) for k in PART_KEYS}
    if with_ci:
        agg["ci95_pages_bootstrap"] = {"seed": SEED_BOOT, "draws": N_BOOT, "n_pages": len(rows), **bootstrap(rows, keys)}
    return agg


# ------------------------------------------------------------------ dev n° 2

def runs_root_from_env() -> Path:
    """Racine = AXEL_RUNS.parent (LOT-P08 v9). AXEL_RUNS must be set and end with ``runs``: no default location."""
    env = os.environ.get("AXEL_RUNS", "").strip()
    if not env:
        raise SystemExit("AXEL_RUNS is required (no default location, LOT-P08 v9)")
    runs = Path(env).expanduser().resolve()
    if runs.name != "runs":
        raise SystemExit(f"AXEL_RUNS must end with 'runs': {runs}")
    return runs.parent


def new_out_dir(path: str) -> Path:
    """The output folder is NEW: an existing folder is refused, so that no result (nor ) is ever rewritten."""
    out = Path(path).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"--out-dir exists, refused: {out}")
    out.mkdir(parents=True)
    return out


def dev2(out_dir: Path) -> tuple[dict, dict, list]:
    from babaorum.p08_oracles import bootstrap_by_groups
    from babaorum.g0_keys import title_of_prefix

    e32 = importlib.import_module("e32")
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    R = runs_root_from_env()
    bench = e32.BENCHES["dev"]
    pages = json.loads((RUNS / "e12/pages.json").read_text())
    names = sorted(p["page"] for p in pages)
    check_pages = list(check_pages_for(names))     # LOT-P08 v9: pages named by the request on 26 pages; seed 1 on 6
    bench_json = json.loads((ROOT / "reports/bench_pipeline_p07.json").read_text())
    bench_cer = {(r["chain"], r["page"]): r["cer_reordered"] for r in bench_json["chains"]}
    result = {"check_pages": check_pages, "pages": {}}
    rows = {c: {"full": [], "emprise": []} for c in ("c1", "c1+ma")}
    reverse_rows = {c: [] for c in ("c1", "c1+ma")}
    disagreement: dict[str, str] = {}
    for p in pages:
        name = p["page"]
        try:
            gt = load_page(Path(p["xml"]))
            glines = gt.lines()
            arms, zones = e32.build_arms(bench, "c1", name)
            polys = truth_polys(gt)
            pool = [json.loads(l) for l in (R / "pool" / f"{name}.jsonl").open(encoding="utf-8")]
            existing = pool_existence(glines, pool)
            classes = missed_classes(glines, existing, zones)
            page_out = {}
            for chain, arm in (("c1", "base"), ("c1+ma", "ma")):
                units = arms[arm].units
                dec, lg, info = analyse(units, glines, gt.text_regions, zones, polys, missed_class=classes)
                ref = bench_cer.get((chain, name))
                if ref is not None:
                    assert abs(dec.cer["0"] - ref) < 1e-9, f"{name} {chain}: CER_0 {dec.cer['0']} != bench {ref}"
                assert abs(parts_sum_error(dec)) < 1e-9
                chars = score_page([u.text for u in units], [g.text for g in glines]).chars
                rev, _, _ = analyse(units, glines, gt.text_regions, zones, polys, order=REVERSE, missed_class=classes)[0:3]
                kept, removed = restrict(units, polys)
                dec_e, _, info_e = analyse(kept, glines, gt.text_regions, zones, polys, missed_class=classes)
                share_out = removed / len(units) if units else 0.0
                row = {"page": name, "chars": chars, "cer": dec.cer, "parts": dec.parts, "counts": dec.counts}
                rows[chain]["full"].append(row)
                rows[chain]["emprise"].append({"page": name, "chars": chars, "cer": dec_e.cer, "parts": dec_e.parts, "counts": dec_e.counts,
                                               "predicted_removed_outside_emprise": removed, "share_removed": share_out})
                reverse_rows[chain].append({"page": name, "chars": chars, "cer": rev.cer, "parts": rev.parts})
                page_out[chain] = {"forward": {"cer": dec.cer, "parts": dec.parts, "counts": dec.counts, "chars": chars},
                                   "reverse_order": {"cer": rev.cer, "parts": rev.parts},
                                   "missed_classes": {c: sum(1 for v in classes.values() if v == c) for c in MISSED_CLASSES},
                                   "links": info, "emprise": {"cer": dec_e.cer, "parts": dec_e.parts, "predicted_removed": removed,
                                                              "share_removed": share_out, "links": info_e}}
        except ValueError as exc:
            if "true-order control failed" not in str(exc):
                raise
            disagreement[name] = str(exc)[:300]   # LOT-P08 v9 point 5: reported apart, the run goes on (all calls protected)
            continue
        result["pages"][name] = page_out
    result["disagreement_pages_reported_apart"] = disagreement
    keep = set(result["pages"])
    for chain in rows:
        for key in ("full", "emprise"):
            rows[chain][key] = [r for r in rows[chain][key] if r["page"] in keep]
        reverse_rows[chain] = [r for r in reverse_rows[chain] if r["page"] in keep]
    flag = {r["page"]: r["share_removed"] for r in rows["c1+ma"]["emprise"]}
    within = [n for n, s_ in flag.items() if s_ <= 0.10]
    result["emprise_rule"] = {"rule": "pages dont au plus 10 % des lignes prédites (c1+ma) sont hors de l'emprise transcrite ; étiquette seulement",
                              "share_outside": flag, "within_10pct": within, "over_10pct": [n for n in flag if n not in within]}
    result["groups"] = {n: [str(int(re.search(r"(?<!\d)(\d{6,8})(?!\d)", n).group(1))), title_of_prefix(n) or n] for n in names}   # (number, title)
    result["aggregates"] = {}
    for chain in rows:
        full = rows[chain]["full"]
        sel = lambda names_, src=full: [r for r in src if r["page"] in names_]
        result["aggregates"][chain] = {
            "all_pages": summarize(full),
            "pages_within_10pct_outside_emprise": summarize(sel(within)),
            "pages_over_10pct_reported_apart": {n: summarize(sel([n]), with_ci=False) for n in flag if n not in within},
            "all_pages_restricted_to_emprise": summarize(rows[chain]["emprise"]),
            "reverse_order_all_pages": aggregate(reverse_rows[chain], lambda rs: ("lecture", "partielles", "decoupe", "manquees", "insertions", "ordre"))}
        fwd = result["aggregates"][chain]["all_pages"]
        rev = result["aggregates"][chain]["reverse_order_all_pages"]
        result["aggregates"][chain]["order_sensitivity"] = {k: {"forward": fwd[k], "reverse": rev[k],
            "changes_more_than_30pct": (abs(rev[k] - fwd[k]) > 0.3 * abs(fwd[k])) if fwd[k] else None} for k in ("lecture", "partielles", "decoupe", "manquees", "insertions")}
        # LOT-P08 v9 point 4: bootstrap by groups (23 groups for 26 pages), pages of a group enter together
        grows = []
        for r in rows[chain]["full"]:
            flat = {"page": r["page"], "group": tuple(result["groups"][r["page"]]), "chars": r["chars"], **{k: r["parts"].get(k, 0.0) for k in PART_KEYS}}
            flat["ordre"] = r["cer"]["4"]
            grows.append(flat)
        result["aggregates"][chain]["bootstrap_by_groups"] = bootstrap_by_groups(grows, [*PART_KEYS]) if grows else None
    return result, rows, check_pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="NEW folder for the results (refused if it exists)")
    ap.add_argument("--e12-control", action="store_true")
    args = ap.parse_args()
    if args.e12_control:
        from p08_e12_control import run  # noqa: F401  (kept separate: scripts/p08_e12_control.py)
        return run(new_out_dir(args.out_dir))
    runs_root_from_env()          # fails first, before any output folder is created
    out = new_out_dir(args.out_dir)
    result, rows, check = dev2(out)
    (out / "decomposition.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    emprise = {"rule": result["emprise_rule"]["rule"], "per_page": {
        n: {"predicted_c1ma_removed": r["predicted_removed_outside_emprise"], "share": r["share_removed"],
            "label": "verite_partielle_probable" if r["share_removed"] > 0.10 else "ok"} for n, r in
        ((x["page"], x) for x in rows["c1+ma"]["emprise"])}}
    (out / "emprise.json").write_text(json.dumps(emprise, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "desaccords.md").write_text(disagreement_table(result["disagreement_pages_reported_apart"]), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "pages": len(result["pages"]), "apart": result["disagreement_pages_reported_apart"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
