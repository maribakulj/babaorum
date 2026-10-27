"""NOTE-B-P08-instrument-E27 (v2.8, GO pT/pQ on the method), § 3: the population of suspect groups and the PPS draws of BOTH stages (p1).

Population: the E27 groups (SPLIT, MERGE_VERT, MERGE_COL, TANGLE) of the 21 pages <= 10 % outside emprise, chain c1+ma, holding at least one
predicted line that line_geometry matches 1:1 to a true line (anywhere) with a line CER <= 0.10 (score of P-08) — the definition of
scripts/p08v9_classA_E27.py, unchanged. A group is identified by (page name, sorted true-line ids, sorted predicted-line ids); its weight is
the true characters of its true lines (len, no normalisation). Checks: the weights sum to 80 463 and no true line is in two groups of a page.
Draws (with replacement, p = weight / sum): stage 1, 42 draws with numpy default_rng(20261023); stage 2, 42 draws with default_rng(20261025),
BOTH made now, before any planche or first look (pA, 23/10). Writes one JSON (refused if it exists). Only pB's functions are used, unchanged.
    AXEL_RUNS=<abs>/rappel_pages/runs_26/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v9_instrument_draw.py \
        --decomposition runs/p08v9_runs26/decomposition.json --out runs/p08v9_instrument/draws.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
dm = importlib.import_module("p08_decompose")  # puts axel and babaorum on sys.path; pB's functions as they are
from axel.eval.seg_errors import make_line  # noqa: E402
from axel.layout.gt import load_page  # noqa: E402
from babaorum.line_geometry import evaluate_line_geometry  # noqa: E402

LINE_CER = 0.10
EXPECTED_SUM = 80463
N_STAGE = 42
SEEDS = {"stage1": 20261023, "stage2": 20261025}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def suspect_groups(name: str, page: dict, bench, e32) -> list[dict]:
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
    out, seen = [], set()
    for c in d.components:
        if c.kind not in dm.GROUP_KINDS:
            continue
        gids = sorted(glines[i].id for i in c.gt)
        pids = sorted(units[j].id for j in c.hyp)
        if not gids:
            raise SystemExit(f"{name}: an E27 group without true line")
        if seen & set(gids):
            raise SystemExit(f"{name}: a true line in two E27 groups: {sorted(seen & set(gids))}")
        seen |= set(gids)
        if any(p in good_pred for p in pids):
            out.append({"page": name, "gt_ids": gids, "pred_ids": pids, "kind": c.kind, "chars": sum(len(gt_text[i]) for i in gids)})
    return out


def draw(population: list[dict], seed: int, n: int = N_STAGE) -> list[int]:
    w = np.array([g["chars"] for g in population], dtype=float)
    return [int(i) for i in np.random.default_rng(seed).choice(len(population), size=n, replace=True, p=w / w.sum())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decomposition", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    dec_p = Path(a.decomposition)
    dec = json.loads(dec_p.read_text())
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    e32 = importlib.import_module("e32")
    bench = e32.BENCHES["dev"]
    population = []
    for name in sorted(dec["emprise_rule"]["within_10pct"]):                     # page names sorted (basename = name)
        if Path(pages[name]["xml"]).stem != name:
            raise SystemExit(f"page name {name} is not the basename of its file {pages[name]['xml']}")
        population += suspect_groups(name, pages[name], bench, e32)
    population.sort(key=lambda g: (g["page"], g["gt_ids"], g["pred_ids"]))
    total = sum(g["chars"] for g in population)
    if total != EXPECTED_SUM or len(population) != 855:
        raise SystemExit(f"population differs from the class-A count: {len(population)} groups, {total} chars (expected 855, {EXPECTED_SUM})")
    report = {"note": "NOTE-B-P08-instrument-E27.md v2.8 32342fc7", "inputs": {"decomposition": sha(dec_p), "script": sha(Path(__file__))},
              "population": population, "sum_chars": total,
              "stage1": draw(population, SEEDS["stage1"]), "stage2": draw(population, SEEDS["stage2"]), "seeds": SEEDS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"groups": len(population), "sum_chars": total, "stage1_unique": len(set(report["stage1"])),
                      "stage2_unique": len(set(report["stage2"])), "sha": sha(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
