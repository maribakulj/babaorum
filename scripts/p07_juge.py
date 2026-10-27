"""LOT-P07 v4: juge/<page>.json — the TRUTH side, apart from pool/. NOT handed to pR until pA gives the signal.

    AXEL_RUNS=<abs>/rappel_pages/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07_juge.py

Per page: true lines with their region; eligibility (registry v2, role dev); the principal line_geometry.v1 matching (0.5 h) of EVERY pool
candidate against the truth; the status of every true line against the C1+M+A chain (apparie | manque_dans_zone | manque_hors_zone | coupe),
with the E27 decomposition kind beside it. Reads the pool/ files and the caches; writes only juge/.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

AX = Path.home() / "axel"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AX / "src"), str(AX / "scripts"), str(ROOT / "src")]
RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
R = RUNS.parent
assert RUNS.is_absolute() and "rappel_pages" in str(RUNS)

from axel.eval.seg_errors import MISSED, SPLIT, decompose, make_line  # noqa: E402
from axel.layout.gt import load_page  # noqa: E402
from babaorum.g0_collections import eligibility  # noqa: E402
from babaorum.g0_keys import ids_of, title_of_prefix  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

e32 = importlib.import_module("e32")
REGISTRY = json.loads((ROOT / "config/g0_collections_v2_2.json").read_text(encoding="utf-8"))


def safe(r):
    """line_geometry ids must match [A-Za-z_][A-Za-z0-9_.-]* : candidate_id minus the colon."""
    return r["candidate_id"].replace(":", "_")


def height(poly):
    ys = [q[1] for q in poly]
    return max(ys) - min(ys)


def bbox(poly):
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    pages = json.loads((RUNS / "e12/pages.json").read_text())
    bench = e32.BENCHES["dev"]
    (R / "juge").mkdir(exist_ok=True)
    for p in pages:
        name = p["page"]
        gt = load_page(Path(p["xml"]))
        glines = gt.lines()
        gt_geo = [GroundTruthLine(g.id, g.baseline, height(g.polygon), "ground_truth") if g.baseline and len(g.baseline) >= 2 and g.polygon
                  else GroundTruthLine(g.id, None, None, None, "missing") for g in glines]
        pool = [json.loads(l) for l in (R / "pool" / f"{name}.jsonl").open(encoding="utf-8")]
        back = {safe(r): r["candidate_id"] for r in pool}
        pred_geo = [PredictedLine(safe(r), r["baseline"], "predicted") if r.get("baseline") and len(r["baseline"]) >= 2
                    else PredictedLine(safe(r), None, None, "missing") for r in pool]
        lg_pool = evaluate_line_geometry(gt_geo, pred_geo)
        matched_pool = {back[m.pred_id]: m.gt_id for m in lg_pool.matches}
        # status of every true line against the C1+M+A chain
        arms, zones = e32.build_arms(bench, "c1", name)
        ch = arms["ma"]
        pred_ch = [PredictedLine(u.id, [list(q) for q in u.baseline], "predicted") if u.baseline and len(u.baseline) >= 2
                   else PredictedLine(u.id, None, None, "missing") for u in ch.units]
        lg_ch = evaluate_line_geometry(gt_geo, pred_ch)
        matched_ch = {m.gt_id: m.pred_id for m in lg_ch.matches}
        G = [make_line(g.id, g.polygon, g.baseline, g.text) for g in glines]
        H = [make_line(u.id, u.polygon, u.baseline, u.text) for u in ch.units]
        d = decompose(G, H, [r.polygon for r in gt.text_regions])
        kind = {}
        for c in d.components:
            for gi in c.gt:
                kind[glines[gi].id] = c.kind
        out = []
        for g in glines:
            b = bbox(g.polygon) if g.polygon else None
            in_zone = bool(b) and any(z[0] <= (b[0] + b[2]) / 2 <= z[2] and z[1] <= (b[1] + b[3]) / 2 <= z[3] for z in zones)
            if g.id in matched_ch:
                st = "apparie"
            elif kind.get(g.id) == SPLIT:
                st = "coupe"
            else:
                st = "manque_dans_zone" if in_zone else "manque_hors_zone"
            out.append({"gt_line_id": g.id, "region": g.region, "text": g.text, "baseline": [list(q) for q in g.baseline],
                        "polygon": [list(q) for q in g.polygon], "status_c1ma": st, "matched_pred_line": matched_ch.get(g.id),
                        "decompose_kind": kind.get(g.id), "centre_in_a_base_zone": in_zone})
        page_key, issue_key = ids_of(name)
        elig = eligibility(REGISTRY, "dev", page_key=page_key, issue_key=issue_key, title_key=title_of_prefix(name))
        (R / "juge" / f"{name}.json").write_text(json.dumps({
            "page": name, "TRUTH": "ce fichier contient la vérité : NE PAS transmettre à pR avant le signal de pA",
            "eligibility_dev": {"allowed": elig["allowed"], "refusals": elig["refusals"], "matches": elig["matched_collections"],
                                "keys": elig["supplied"], "registry": "config/g0_collections_v2_2.json"},
            "true_lines": out,
            "line_geometry_pool": {"metric": lg_pool.metric_id, "status": lg_pool.status, "n_gt": lg_pool.n_gt, "n_pool": lg_pool.n_pred,
                                   "matched": lg_pool.matched_count, "tolerance": "0,5 h (principal)"},
            "pool_matching": [{"candidate_id": r["candidate_id"], "origin": r["origin"], "gt_line_id": matched_pool.get(r["candidate_id"]),
                               "status": "apparie" if r["candidate_id"] in matched_pool else "non_apparie"} for r in pool],
            "chain_c1ma": {"n_pred": len(ch.units), "matched": lg_ch.matched_count, "status": lg_ch.status,
                           "counts": {k: sum(o["status_c1ma"] == k for o in out) for k in ("apparie", "manque_dans_zone", "manque_hors_zone", "coupe")}},
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(name, {k: sum(o["status_c1ma"] == k for o in out) for k in ("apparie", "manque_dans_zone", "manque_hors_zone", "coupe")},
              "pool apparié", lg_pool.matched_count, "/", lg_pool.n_pred, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
