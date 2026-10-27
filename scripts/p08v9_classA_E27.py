"""NOTE-B-P08v9-runs26-choix v3.4 (d1faff87, GO pJ + pQ), § 3: class-A count E27 / découpe, then the mechanical criterion of § 4.3 (p1, read only).

Equation (v3.4, unique): I = Σ_p S_p / Σ_p G_p over the 21 pages <= 10 % outside emprise, chain c1+ma.
  G_p: true characters (len(g.text), code points, text of load_page, no normalisation) of the true lines of page p in at least one E27 group
       (SPLIT, MERGE_VERT, MERGE_COL, TANGLE), each line once; S_p: the same restricted to SUSPECT groups.
  Suspect group: an E27 group holding at least one predicted line that line_geometry matches 1:1 to a true line, inside the group or not,
       with a line CER <= 0.10 (score of P-08: score_page(...).cer_reordered).
  G_p = 0 adds 0/0; Σ G_p = 0: I undefined, threshold not crossed. Threshold: I >= 0.50 (unrounded) -> stop, note B, nothing applied.
Below the threshold, § 4.3 is applied mechanically on unrounded values (destinations_unrounded.json and a fresh bootstrap of the candidates).
Only pB's functions are used, unchanged. Writes one small JSON in --out (refused if it exists).
    AXEL_RUNS=<abs>/rappel_pages/runs_26/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v9_classA_E27.py \
        --decomposition runs/p08v9_runs26/decomposition.json --destinations runs/p08v9_diag/destinations_unrounded.json \
        --out runs/p08v9_diag/classA_E27_and_criterion.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
dm = importlib.import_module("p08_decompose")  # puts axel and babaorum on sys.path; pB's functions as they are
from axel.eval.seg_errors import make_line  # noqa: E402
from axel.layout.gt import load_page  # noqa: E402
from babaorum.line_geometry import evaluate_line_geometry  # noqa: E402
from babaorum.p08_oracles import bootstrap_by_groups  # noqa: E402

THRESHOLD = 0.50
LINE_CER = 0.10
CLASS = {"P-04": 2, "P-03": 4, "P-02 élargi": 2}                                  # § 4.2 (P-04 class 2: route B decided by pA)
CANDIDATES = {"decoupe": "P-04", "manquees_dans_zone": "P-04", "ordre": "P-03", "manquees_hors_zone": "P-03", "manquees_ecartees": "P-02 élargi"}
DEST_KEYS = {"P-04": "P-04", "P-03": "P-03", "P-02 élargi": "P-02e"}           # keys of destinations_unrounded.json


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def count_page(name: str, page: dict, bench, e32) -> dict:
    gt = load_page(Path(page["xml"]))
    glines = gt.lines()
    arms, _ = e32.build_arms(bench, "c1", name)
    units = arms["ma"].units                                                      # c1+ma
    d = dm.decompose([make_line(g.id, g.polygon, g.baseline, g.text) for g in glines],
                     [make_line(u.id, u.polygon, u.baseline, u.text) for u in units], [r.polygon for r in gt.text_regions])
    lg = evaluate_line_geometry(dm.gt_geo_of(glines), dm.pred_geo_of(units))
    gt_text = {g.id: g.text for g in glines}
    pred_text = {u.id: u.text for u in units}
    good_pred = {m.pred_id for m in lg.matches if dm.score([pred_text[m.pred_id]], [gt_text[m.gt_id]]) <= LINE_CER}
    groups = [c for c in d.components if c.kind in dm.GROUP_KINDS]
    in_group, in_suspect = set(), set()
    n_suspect = 0
    for c in groups:
        gids = {glines[i].id for i in c.gt}
        in_group |= gids
        if any(units[j].id in good_pred for j in c.hyp):
            n_suspect += 1
            in_suspect |= gids
    G = sum(len(gt_text[i]) for i in in_group)
    S = sum(len(gt_text[i]) for i in in_suspect)
    return {"page": name, "G_p": G, "S_p": S, "groups": len(groups), "suspect_groups": n_suspect,
            "true_lines_in_groups": len(in_group), "true_lines_in_suspect_groups": len(in_suspect)}


def criterion(dec: dict, dest: dict) -> dict:
    """§ 4.3 on unrounded values: score = upper bound / class, with its three controls; tie at 10 %; class 1 in parallel (none today)."""
    within = set(dec["emprise_rule"]["within_10pct"])
    groups = dec["groups"]
    rows, rev = [], {}
    for n, v in dec["pages"].items():
        if n not in within:
            continue
        f = v["c1+ma"]["forward"]
        r = {"page": n, "group": tuple(groups[n]), "chars": f["chars"], **{k: f["parts"].get(k, 0.0) for k in CANDIDATES}}
        r["ordre"] = f["cer"]["4"]
        rows.append(r)
        rv = v["c1+ma"]["reverse_order"]
        rev[n] = {**{k: rv["parts"].get(k, 0.0) for k in CANDIDATES}, "ordre": rv["cer"]["4"], "chars": f["chars"]}
    boot = bootstrap_by_groups(rows, list(CANDIDATES))
    w = sum(r["chars"] for r in rows)
    indiv = {k: {"estimate": sum(r["chars"] * r[k] for r in rows) / w, "ci95": boot["ci95"][k], "lot": CANDIDATES[k]} for k in CANDIDATES}
    lots = {lot: {**dest["dest"][DEST_KEYS[lot]], "class": CLASS[lot]} for lot in CLASS}
    for lot, x in lots.items():
        x["score_upper"] = x["ci95"][1] / x["class"]
        x["score_estimate"] = x["estimate"] / x["class"]
        x["reverse_estimate"] = sum(rev[n]["chars"] * sum(rev[n][k] for k, l in CANDIDATES.items() if l == lot) for n in rev) / w
    by_upper = sorted(lots, key=lambda l: -lots[l]["score_upper"])
    top, second = by_upper[0], by_upper[1]
    tie = lots[second]["score_upper"] >= 0.9 * lots[top]["score_upper"]
    chosen = min((top, second), key=lambda l: lots[l]["class"]) if tie else top
    controls = {
        "estimate_ranking_top": max(lots, key=lambda l: lots[l]["score_estimate"]),
        "best_individual_candidate_lot": indiv[max(indiv, key=lambda k: indiv[k]["ci95"][1])]["lot"],
        "reverse_order_top": max(lots, key=lambda l: lots[l]["reverse_estimate"] / lots[l]["class"]),
    }
    agree = all(v == chosen for v in controls.values())
    return {"rows_pages": len(rows), "bootstrap_groups": boot["groups"], "candidates": indiv, "lots": lots, "ranking_upper": by_upper,
            "tie_within_10pct": tie, "controls": controls, "controls_agree": agree,
            "class_1_parallel": sorted(l for l in lots if lots[l]["class"] == 1),
            "designated": chosen if agree else None, "note_B_required": not agree}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decomposition", required=True)
    ap.add_argument("--destinations", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    dec_p, dest_p = Path(a.decomposition), Path(a.destinations)
    dec, dest = json.loads(dec_p.read_text()), json.loads(dest_p.read_text())
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    e32 = importlib.import_module("e32")
    bench = e32.BENCHES["dev"]
    within = sorted(dec["emprise_rule"]["within_10pct"])
    per_page = [count_page(n, pages[n], bench, e32) for n in within]
    sG, sS = sum(r["G_p"] for r in per_page), sum(r["S_p"] for r in per_page)
    I = sS / sG if sG else None
    crossed = I is not None and I >= THRESHOLD
    report = {"note": "NOTE-B-P08v9-runs26-choix.md v3.4 d1faff87", "inputs": {"decomposition": sha(dec_p), "destinations": sha(dest_p),
              "script": sha(Path(__file__))}, "pages": len(per_page), "per_page": per_page, "sum_S": sS, "sum_G": sG, "I": I,
              "threshold": THRESHOLD, "threshold_crossed": crossed}
    if crossed:
        report["decision"] = "ARRÊT : seuil franchi, note B ; le critère n'est pas appliqué, rien ne part"
    else:
        report["criterion"] = criterion(dec, dest)
        report["decision"] = (f"désigné : {report['criterion']['designated']}" if report["criterion"]["designated"]
                              else "contrôles en désaccord : note B, pas de choix")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sum_S", "sum_G", "I", "threshold_crossed", "decision")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
