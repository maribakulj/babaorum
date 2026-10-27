"""SENSIBILITE-P04-manquees-dissoutes.md (3a1c15b3), class A, descriptive (p1): does the v3.4 ranking change if the true lines of DISSOLVED
groups that became missed of class dans_zone are counted elsewhere (V1 hors_zone -> P-03, V2 ecartees -> P-02 élargi) or removed from every
lot (V3 sans_centre)? V0 = no class changed, must equal the v10.1 decomposition to 1e-9, else STOP.
pB's functions unchanged (p08_decompose caf50c9e: analyse, missed_classes, pool_existence, truth_polys); criterion of p08v9_classA_E27.py
unchanged. Frozen scope: the 21 pages within_10pct of v9, chain c1+ma. Writes one small JSON (--out, refused if it exists); exit 1 on STOP.
    AXEL_RUNS=<abs>/rappel_pages/runs_26/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v101_sensibilite.py \
        --v9 runs/p08v9_runs26/decomposition.json --v10 runs/p08v101_runs26/decomposition.json --out runs/p08v101_runs26/sensibilite_p04.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
dm = importlib.import_module("p08_decompose")  # puts axel and babaorum on sys.path
from axel.layout.gt import load_page  # noqa: E402
from babaorum.p08_oracles import bootstrap_by_groups  # noqa: E402

TOL = 1e-9
EXPECTED_SCOPE = 21
VARIANTS = {"V0": None, "V1": "hors_zone", "V2": "ecartees", "V3": "sans_centre"}
DEST = {"P-04": ("decoupe", "manquees_dans_zone"), "P-03": ("ordre", "manquees_hors_zone"), "P-02e": ("manquees_ecartees",)}
CLS = {"P-04": 2, "P-03": 4, "P-02e": 2}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def destinations(pages: dict, within: list, groups: dict) -> dict:
    """Same computation as runs/p08v9_diag/destinations_unrounded.json (note B v3.4) and NOTE-B-P08v101: c1+ma, 21 pages, bootstrap by groups."""
    rows = []
    for n in within:
        f = pages[n]["forward"]
        p = dict(f["parts"])
        p["ordre"] = f["cer"]["4"]
        rows.append({"page": n, "group": tuple(groups[n]), "chars": f["chars"], **{k: sum(p.get(x, 0.0) for x in xs) for k, xs in DEST.items()}})
    w = sum(r["chars"] for r in rows)
    b = bootstrap_by_groups(rows, list(DEST))
    out = {"groups": b["groups"], "seed": b["seed"], "draws": b["draws"], "dest": {}}
    for k in DEST:
        est = sum(r["chars"] * r[k] for r in rows) / w
        lo, hi = b["ci95"][k]
        out["dest"][k] = {"estimate": est, "ci95": [lo, hi], "class": CLS[k], "score_upper": hi / CLS[k], "score_estimate": est / CLS[k]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v9", required=True)
    ap.add_argument("--v10", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    v9_p, v10_p = Path(a.v9), Path(a.v10)
    v9, v10 = json.loads(v9_p.read_text()), json.loads(v10_p.read_text())
    within = sorted(v9["emprise_rule"]["within_10pct"])
    if len(within) != EXPECTED_SCOPE or sorted(v10["emprise_rule"]["within_10pct"]) != within:
        raise SystemExit("frozen scope differs from the expected 21 pages")
    spec = importlib.util.spec_from_file_location("p08v9_classA_E27", ROOT / "scripts/p08v9_classA_E27.py")
    ca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ca)
    e32 = importlib.import_module("e32")
    bench = e32.BENCHES["dev"]
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    R = dm.runs_root_from_env()
    meta = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    per_variant = {v: {} for v in VARIANTS}
    targeted = {}
    for n in within:
        gt = load_page(Path(meta[n]["xml"]))
        glines = gt.lines()
        arms, zones = e32.build_arms(bench, "c1", n)
        polys = dm.truth_polys(gt)
        pool = [json.loads(l) for l in (R / "pool" / f"{n}.jsonl").open(encoding="utf-8")]
        classes = dm.missed_classes(glines, dm.pool_existence(glines, pool), zones)
        units = arms["ma"].units
        chars = dm.score_page([u.text for u in units], [g.text for g in glines]).chars
        dec0, _, info0 = dm.analyse(units, glines, gt.text_regions, zones, polys, missed_class=classes)
        tgt = sorted(g for g in info0["rlg"]["to_missed_gt"] if classes[g] == "dans_zone")
        targeted[n] = {"lines": len(tgt), "chars": sum(len(g.text) for g in glines if g.id in tgt)}
        for v, new in VARIANTS.items():
            cls = dict(classes) if new is None else {**classes, **{g: new for g in tgt}}
            fw = dec0 if new is None else dm.analyse(units, glines, gt.text_regions, zones, polys, missed_class=cls)[0]
            rv = dm.analyse(units, glines, gt.text_regions, zones, polys, order=dm.REVERSE, missed_class=cls)[0]
            per_variant[v][n] = {"forward": {"cer": fw.cer, "parts": fw.parts, "chars": chars}, "reverse_order": {"cer": rv.cer, "parts": rv.parts}}
    # V0 control against the published v10.1 decomposition
    bad = []
    for n in within:
        pub = v10["pages"][n]["c1+ma"]
        mine = per_variant["V0"][n]
        if pub["forward"]["chars"] != mine["forward"]["chars"]:
            bad.append(f"{n} chars: {pub['forward']['chars']} != {mine['forward']['chars']}")
        for o in ("forward", "reverse_order"):
            for kind in ("cer", "parts"):
                for k in sorted(set(mine[o][kind]) - set(pub[o][kind])):
                    bad.append(f"{n} {o} {kind} {k}: absent from the published decomposition")
                for k, x in pub[o][kind].items():
                    y = mine[o][kind].get(k)
                    if y is None or abs(x - y) > TOL:
                        bad.append(f"{n} {o} {kind} {k}: {x} != {y}")
    report = {"note": "SENSIBILITE-P04-manquees-dissoutes.md 3a1c15b3, classe A, descriptif ; aucune conséquence tirée",
              "inputs": {"v9": sha(v9_p), "v10": sha(v10_p), "script": sha(Path(__file__)), "p08_decompose.py": sha(ROOT / "scripts/p08_decompose.py"),
                         "p08v9_classA_E27.py": sha(ROOT / "scripts/p08v9_classA_E27.py")},
              "targeted_total": {"lines": sum(t["lines"] for t in targeted.values()), "chars": sum(t["chars"] for t in targeted.values())},
              "targeted_per_page": targeted, "v0_control_ok": not bad, "v0_mismatches": bad[:20]}
    if bad:
        report["outcome"] = "ARRÊT : V0 ne reproduit pas la décomposition v10.1 ; rien n'est rapporté"
    else:
        report["variants"] = {}
        for v, new in VARIANTS.items():
            pages = per_variant[v]
            dest = destinations(pages, within, v10["groups"])
            dec_like = {"emprise_rule": {"within_10pct": within}, "groups": v10["groups"], "pages": {n: {"c1+ma": pages[n]} for n in within}}
            crit = ca.criterion(dec_like, dest)
            report["variants"][v] = {"reclassed_to": new, "dest": dest["dest"], "ranking_upper": crit["ranking_upper"],
                                     "tie_within_10pct": crit["tie_within_10pct"], "controls": crit["controls"],
                                     "controls_agree": crit["controls_agree"], "designated_mechanically": crit["designated"]}
        report["outcome"] = "descriptif rapporté"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("targeted_total", "v0_control_ok", "outcome")}, ensure_ascii=False))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
