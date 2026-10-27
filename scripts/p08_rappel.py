"""LOT-P08 section 2: recall study of the dev n° 2 (c1+ma): R_final, R_pool, motives of loss. Files only, no inference.

    AXEL_RUNS=<abs>/rappel_pages/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08_rappel.py
R_pool is an existence: each true line is tested against every neighbouring candidate (box filter, speed only) with the
line_geometry.v1 criterion at 0.5 h; best candidate = largest gt_to_pred_coverage, tie -> smallest candidate_id.
"""
from __future__ import annotations

import importlib
import json
import os
import random
import sys
from pathlib import Path

AX = Path.home() / "axel"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AX / "src"), str(AX / "scripts"), str(ROOT / "src"), str(ROOT / "scripts")]


from p08_decompose import centre, gt_geo_of, height, in_box, in_poly, pred_geo_of, truth_polys  # noqa: E402
from axel.layout.gt import load_page  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

MOTIF_ORDER = ("zone_non_retenue", "regle_M", "dedup_covered", "D-563", "D-564", "non_lu", "garde_mais_apparie_a_une_autre")


def safe(r):
    return r["candidate_id"].replace(":", "_")


def bb_baseline(bl):
    xs, ys = [q[0] for q in bl], [q[1] for q in bl]
    return min(xs), min(ys), max(xs), max(ys)


def best_candidates(glines, geo, pool, use_filter=True):
    """{gt_id: (candidate record, coverage)} for the true lines proposed by at least one candidate."""
    cands = [(r, bb_baseline(r["baseline"])) for r in pool if r.get("baseline") and len(r["baseline"]) >= 2]
    out = {}
    for g, gg in zip(glines, geo):
        if gg.geometry_status != "available":
            continue
        gb = bb_baseline(g.baseline)
        h = height(g.polygon)
        m = 2 * h + 20
        best = None
        for r, rb in cands:
            if use_filter and (rb[2] < gb[0] - m or rb[0] > gb[2] + m or rb[3] < gb[1] - m or rb[1] > gb[3] + m):
                continue
            res = evaluate_line_geometry([gg], [PredictedLine(safe(r), r["baseline"], "predicted")])
            if res.matched_count == 1:
                cov = res.matches[0].gt_to_pred_coverage
                key = (-cov, r["candidate_id"])
                if best is None or key < best[0]:
                    best = (key, r, cov)
        if best:
            out[g.id] = (best[1], best[2])
    return out


def motif(rec):
    if rec["origin"] == "zone":
        if not rec["zone_selected_by_select_blocks"]:
            return "zone_non_retenue"
        m = rec["m_membership"]
        if m["status"] == "ecartee":
            return {"centre_hors_rectangle_zone": "regle_M", "ecartee_par": "dedup_covered"}.get(m["motif"], "geometrie_invalide")
    else:
        if not rec["d563"]["arm_ma"]["selected"]:
            return "D-563"
        d = rec["d564"]["arm_ma"]
        if d["status"] == "exclue":
            return "D-564"
        if d["status"] == "non_lue" or rec.get("text") is None:
            return "non_lu"
    return "garde_mais_apparie_a_une_autre" if rec["in_final"]["c1+ma"] else "autre"


def study(name, p, e32, bench, restrict_emprise, use_filter=True):
    gt = load_page(Path(p["xml"]))
    glines = gt.lines()
    geo = gt_geo_of(glines)
    arms, zones = e32.build_arms(bench, "c1", name)
    units = arms["ma"].units
    pool = [json.loads(l) for l in (R / "pool" / f"{name}.jsonl").open(encoding="utf-8")]
    polys = truth_polys(gt)
    if restrict_emprise:
        units = [u for u in units if any(in_poly(centre(u.polygon), q) for q in polys)]
        pool = [r for r in pool if any(in_poly(centre(r["polygon"]), q) for q in polys)]
    lg = evaluate_line_geometry(geo, pred_geo_of(units))
    final = {m.gt_id for m in lg.matches}
    best = best_candidates(glines, geo, pool, use_filter)
    lost = [g for g in best if g not in final]
    reasons = {}
    for g in lost:
        reasons[motif(best[g][0])] = reasons.get(motif(best[g][0]), 0) + 1
    never = [g.id for g in glines if g.id not in best and geo[[x.id for x in glines].index(g.id)].geometry_status == "available"]
    gmap = {g.id: g for g in glines}
    vent = {"dans_une_zone": sum(any(in_box(centre(gmap[i].polygon), z) for z in zones) for i in never)}
    vent["hors_zone"] = len(never) - vent["dans_une_zone"]
    return {"n_gt": len(glines), "matched_final": len(final), "R_final": len(final) / len(glines), "in_pool": len(best),
            "R_pool": len(best) / len(glines), "gap_R_pool_minus_R_final": (len(best) - len(final & set(best))) / len(glines),
            "lost_between_pool_and_final": len(lost), "motives": reasons, "never_proposed": len(never), "never_proposed_split": vent,
            "best": {g: (b[0]["candidate_id"], round(b[1], 6)) for g, b in best.items()}, "pool_size": len(pool)}


def main() -> int:
    """LOT-P08 v9: root = AXEL_RUNS.parent (no default location); the output is a NEW folder given by --out-dir;
    emprise_rule is read from the decomposition.json of that same folder."""
    import argparse
    global R
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    env = os.environ.get("AXEL_RUNS", "").strip()
    if not env or Path(env).expanduser().resolve().name != "runs":
        raise SystemExit("AXEL_RUNS is required and must end with 'runs' (no default location)")
    R = Path(env).expanduser().resolve().parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not (out_dir / "decomposition.json").is_file():
        raise SystemExit(f"{out_dir} must be the folder of p08_decompose (decomposition.json missing)")
    if (out_dir / "rappel.json").exists():
        raise SystemExit(f"rappel.json already exists in {out_dir}, refused")
    e32 = importlib.import_module("e32")
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    assert RUNS == (R / "runs").resolve()
    bench = e32.BENCHES["dev"]
    from babaorum.p08_oracles import check_pages_for
    dec = json.loads((out_dir / "decomposition.json").read_text())
    apart = set(dec["disagreement_pages_reported_apart"])     # LOT-P08 v9 point 5: in no aggregate, listed in desaccords.md
    all_names = [p["page"] for p in json.loads((RUNS / "e12/pages.json").read_text())]
    p_decomp, p_pool = check_pages_for(all_names)     # on the full list: the check pages never switch silently
    if {p_decomp, p_pool} & apart:
        raise SystemExit(f"a check page is reported apart ({p_decomp}, {p_pool}): stop and ask pA")
    pages = [p for p in json.loads((RUNS / "e12/pages.json").read_text()) if p["page"] not in apart]
    emprise_rule = dec["emprise_rule"]
    out = {"check_pages": {"decomposition": p_decomp, "R_pool_filter_control": p_pool}, "pages": {}, "emprise_pages": {}}
    for p in pages:
        out["pages"][p["page"]] = study(p["page"], p, e32, bench, False)
        out["emprise_pages"][p["page"]] = study(p["page"], p, e32, bench, True)
        print(p["page"], {k: v for k, v in out["pages"][p["page"]].items() if k in ("R_final", "R_pool", "motives", "never_proposed_split")}, flush=True)
    chk = next(p for p in pages if p["page"] == p_pool)
    unfiltered = study(p_pool, chk, e32, bench, False, use_filter=False)
    same = unfiltered["best"] == out["pages"][p_pool]["best"] and unfiltered["R_pool"] == out["pages"][p_pool]["R_pool"]
    out["filter_control"] = {"page": p_pool, "identical_with_and_without_box_filter": same,
                             "R_pool_filtered": out["pages"][p_pool]["R_pool"], "R_pool_unfiltered": unfiltered["R_pool"]}
    assert same, "R_pool differs with and without the box filter: stop"
    # control: existence >= the 1-to-1 matching rate of juge/ (counts only)
    ctl = {}
    for name, s in out["pages"].items():
        j = json.loads((R / "juge" / f"{name}.json").read_text())
        one = sum(1 for r in j["pool_matching"] if r["status"] == "apparie") / s["n_gt"]
        ctl[name] = {"R_pool": s["R_pool"], "pool_1to1_rate": one, "ok": s["R_pool"] >= one - 1e-12}
    out["control_vs_juge_1to1"] = ctl
    assert all(v["ok"] for v in ctl.values()), "R_pool below the 1-to-1 rate of juge/"
    within = set(emprise_rule["within_10pct"])

    def agg(names, src):
        n = sum(src[x]["n_gt"] for x in names)
        res = {"pages": sorted(names), "R_final": sum(src[x]["matched_final"] for x in names) / n, "R_pool": sum(src[x]["in_pool"] for x in names) / n,
               "motives": {}, "never_proposed": sum(src[x]["never_proposed"] for x in names)}
        for x in names:
            for k, v in src[x]["motives"].items():
                res["motives"][k] = res["motives"].get(k, 0) + v
        res["never_proposed_split"] = {k: sum(src[x]["never_proposed_split"][k] for x in names) for k in ("dans_une_zone", "hors_zone")}
        res["lost_between_pool_and_final"] = sum(src[x]["lost_between_pool_and_final"] for x in names)
        return res

    allp = list(out["pages"])
    out["aggregates"] = {"6_pages": agg(allp, out["pages"]), "pages_within_10pct": agg([x for x in allp if x in within], out["pages"]),
                         "pages_over_10pct_apart": {x: agg([x], out["pages"]) for x in allp if x not in within},
                         "6_pages_emprise": agg(allp, out["emprise_pages"]),
                         "pages_within_10pct_emprise": agg([x for x in allp if x in within], out["emprise_pages"])}
    for s in list(out["pages"].values()) + list(out["emprise_pages"].values()):
        s.pop("best")
    out["disagreement_pages_reported_apart"] = sorted(apart)
    (out_dir / "rappel.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
