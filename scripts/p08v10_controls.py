"""NOTE-B-P08-oracle-E27-correction v2.3 § 4, steps 2 and 3 on the 26 pages: two read-only controls of P-08 v10 (p1).

nonreg       : avenant v2.4 § 4.2, parts 1 and 2 (GO pQ 5e34be6c), on the v9 referential enumerated: 24 pages x c1/c1+ma x
               forward/reverse/emprise = 144 calls (true-order control passes iff the block is present), and x full/emprise = 96 link
               blocks with the identities partial, missed + |to_missed_gt|, extra_pred + lines_pred_to_extra, groups - groups_dissolved,
               matched_1to1 + pairs_added, and partial_also_matched (full). Absence = failure. Part 3 (synthetic tests) is pB's.
construction : "contrôle de construction" agreed with pA (25/10), NOT a gate (the gate of v3.4 stays the one of the note). On the pages
               <= 10 % outside emprise, chain c1+ma, the suspect groups are counted on the RAW E27 groups by count_page of
               p08v9_classA_E27.py (imported unchanged), and compared with what v10 dissolved (links.rlg of the full page). Remaining
               suspect groups among the kept ones = suspect_groups - groups_dissolved, and the true lines likewise; expected 0 on every
               page. Scope FROZEN (pQ): the 21 pages within_10pct of v9; empty, truncated or different scope in v10.1 = "non
               évalué", failure. SETS are compared too (Sonnet review, condition a): the true lines of raw suspect groups (suspect_set, which
               mirrors count_page and asserts the same counts) must EQUAL links.rlg.dissolved_gt. Coherence of the run with AXEL_RUNS
               (condition b): raw groups == v10 kept groups (links.linked.groups) + groups_dissolved. Any failure: STOP, note B.
Bootstrap (§ 4.2 "décomposition, bootstrap, parts"; condition c): not compared. The aggregates pool pages where R-LG acted; on the
non-acted blocks the per-page inputs of the bootstrap are equal to 1e-9, so a bootstrap on that subset with the same seeds would be too.
Writes one small JSON (--out, refused if it exists).
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v10_controls.py nonreg --v9 runs/p08v9_runs26/decomposition.json \
        --v10 runs/p08v10_runs26/decomposition.json --out runs/p08v10_runs26/nonreg_v9.json
    AXEL_RUNS=<abs>/rappel_pages/runs_26/runs PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v10_controls.py construction \
        --v9 runs/p08v9_runs26/decomposition.json --v10 runs/p08v10_runs26/decomposition.json \
        --out runs/p08v10_runs26/controle_construction.json
Exit code: 0 only if the control passes (nonreg_ok / construction_ok true); 1 otherwise, and the next step must not start.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


CHAINS = ("c1", "c1+ma")
SCOPES = ("full", "emprise")
CALLS = ("forward", "reverse_order", "emprise")


def links_of(block: dict, scope: str) -> dict:
    return block["links"] if scope == "full" else block["emprise"]["links"]


def nonreg(v9: dict, v10: dict) -> dict:
    """Avenant v2.4 § 4.2, parts 1 and 2, on the v9 referential ENUMERATED (cartesian product; absence = failure; pQ M1).
    Part 1: every call (page x chain x forward / reverse_order / emprise) that passed the true-order control in v9 passes in v10.1. A call
    that fails sets the whole page apart (p08_decompose.dev2), so a call passes iff its block is present.
    Part 2: per link block (page x chain x full / emprise), identities between v9 linked counts and v10.1 linked counts + rlg."""
    calls, call_fail, blocks, block_fail = 0, [], 0, {}
    for page in sorted(v9["pages"]):
        for chain in CHAINS:
            b9 = v9["pages"][page][chain]
            b10 = v10["pages"].get(page, {}).get(chain)
            for call in CALLS:
                calls += 1
                if b10 is None or call not in b10:
                    call_fail.append(f"{page}|{chain}|{call}")
            for scope in SCOPES:
                blocks += 1
                key = f"{page}|{chain}|{scope}"
                if b10 is None:
                    block_fail[key] = ["absent from v10.1"]
                    continue
                l9, l10 = links_of(b9, scope)["linked"], links_of(b10, scope)["linked"]
                r = links_of(b10, scope)["rlg"]
                expect = {"partial": l9["partial"], "missed": l9["missed"] + len(r["to_missed_gt"]),
                          "extra_pred": l9["extra_pred"] + r["lines_pred_to_extra"], "groups": l9["groups"] - r["groups_dissolved"],
                          "matched_1to1": l9["matched_1to1"] + r["pairs_added"]}
                bad = [f"{k}: v10.1 {l10[k]} != expected {v}" for k, v in expect.items() if l10[k] != v]
                if scope == "full":
                    pa9, pa10 = b9["forward"]["counts"]["partial_also_matched"], b10["forward"]["counts"]["partial_also_matched"]
                    if pa9 != pa10:
                        bad.append(f"partial_also_matched: v10.1 {pa10} != v9 {pa9}")
                if bad:
                    block_fail[key] = bad
    if calls != 144 or blocks != 96:                                              # referential of the avenant (pQ M1): 24 v9 pages
        raise SystemExit(f"v9 referential is not the expected one: {calls} calls, {blocks} blocks (expected 144, 96)")
    lost = sorted({c.split("|")[0] for c in call_fail})
    ok = not call_fail and not block_fail and all(p in v9["pages"] for p in RECEPTION)
    return {"calls_enumerated": calls, "calls_failed": call_fail, "pages_lost": lost, "blocks_enumerated": blocks,
            "blocks_failed": block_fail, "reception_4_pages": {p: p in v9["pages"] and p not in lost for p in RECEPTION},
            "nonreg_ok": ok, "outcome": "non-régression OK (parties 1 et 2)" if ok else "ÉCHEC de la non-régression : ARRÊT, note B"}


RECEPTION = ("La_Presse_0514199_T0000004", "La_Presse_0545076_T0000002", "Le_Matin_0574578_T0000002", "Le_Matin_0576768_T0000002")


def suspect_set(ca, name: str, page: dict, bench, e32) -> tuple[set, int]:
    """The true-line ids of the raw suspect groups, by the same steps as count_page (which returns counts only)."""
    dm = ca.dm
    gt = ca.load_page(Path(page["xml"]))
    glines = gt.lines()
    arms, _ = e32.build_arms(bench, "c1", name)
    units = arms["ma"].units
    d = dm.decompose([ca.make_line(g.id, g.polygon, g.baseline, g.text) for g in glines],
                     [ca.make_line(u.id, u.polygon, u.baseline, u.text) for u in units], [r.polygon for r in gt.text_regions])
    lg = ca.evaluate_line_geometry(dm.gt_geo_of(glines), dm.pred_geo_of(units))
    gt_text = {g.id: g.text for g in glines}
    pred_text = {u.id: u.text for u in units}
    good_pred = {m.pred_id for m in lg.matches if dm.score([pred_text[m.pred_id]], [gt_text[m.gt_id]]) <= ca.LINE_CER}
    in_suspect, n = set(), 0
    for c in d.components:
        if c.kind in dm.GROUP_KINDS and any(units[j].id in good_pred for j in c.hyp):
            n += 1
            in_suspect |= {glines[i].id for i in c.gt}
    return in_suspect, n


EXPECTED_SCOPE = 21                                                            # v3.4: the 21 pages <= 10 % outside emprise (v9)


def construction(v9: dict, v10: dict) -> dict:
    within = sorted(v9["emprise_rule"]["within_10pct"])                          # FROZEN scope: v9's, not the run's (pQ, NO-GO 1)
    if len(within) != EXPECTED_SCOPE:
        raise SystemExit(f"frozen scope is not the expected one: {len(within)} pages (expected {EXPECTED_SCOPE})")
    v10_within = sorted(v10["emprise_rule"]["within_10pct"])
    missing = [n for n in within if n not in v10["pages"] or "c1+ma" not in v10["pages"][n]]
    if missing or v10_within != within:
        return {"pages": 0, "expected_scope": within, "v10_within_10pct": v10_within, "pages_missing_in_v10": missing,
                "construction_ok": False, "outcome": "NON ÉVALUÉ : portée vide, tronquée ou différente de la portée figée ; ARRÊT, note B"}
    spec = importlib.util.spec_from_file_location("p08v9_classA_E27", ROOT / "scripts/p08v9_classA_E27.py")
    ca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ca)                                                   # count_page as it is (also puts axel on sys.path)
    import importlib as il
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    e32 = il.import_module("e32")
    bench = e32.BENCHES["dev"]
    per_page = []
    for n in within:
        raw = ca.count_page(n, pages[n], bench, e32)
        links = v10["pages"][n]["c1+ma"]["links"]
        rlg = links["rlg"]
        sset, n_s = suspect_set(ca, n, pages[n], bench, e32)
        if n_s != raw["suspect_groups"] or len(sset) != raw["true_lines_in_suspect_groups"]:
            raise SystemExit(f"{n}: suspect_set disagrees with count_page")
        dset = set(rlg["dissolved_gt"])
        per_page.append({"page": n, "suspect_groups_raw": raw["suspect_groups"], "groups_dissolved_v10": rlg["groups_dissolved"],
                         "true_lines_in_suspect_groups_raw": raw["true_lines_in_suspect_groups"],
                         "true_lines_dissolved_v10": len(rlg["dissolved_gt"]),
                         "remaining_suspect_groups": raw["suspect_groups"] - rlg["groups_dissolved"],
                         "remaining_suspect_true_lines": raw["true_lines_in_suspect_groups"] - len(rlg["dissolved_gt"]),
                         "suspect_not_dissolved": sorted(sset - dset), "dissolved_not_suspect": sorted(dset - sset),
                         "groups_raw": raw["groups"], "groups_kept_v10": links["linked"]["groups"],
                         "coherent": raw["groups"] == links["linked"]["groups"] + rlg["groups_dissolved"]})
    bad = [r["page"] for r in per_page if r["remaining_suspect_groups"] or r["remaining_suspect_true_lines"]
           or r["suspect_not_dissolved"] or r["dissolved_not_suspect"] or not r["coherent"]]
    if len(per_page) != EXPECTED_SCOPE:
        raise SystemExit(f"evaluated {len(per_page)} pages, expected {EXPECTED_SCOPE}")
    return {"pages": len(per_page), "expected_scope": within, "chain": "c1+ma", "per_page": per_page,
            "remaining_suspect_groups_total": sum(r["remaining_suspect_groups"] for r in per_page),
            "remaining_suspect_true_lines_total": sum(r["remaining_suspect_true_lines"] for r in per_page),
            "pages_not_zero": bad, "construction_ok": not bad,
            "outcome": "contrôle de construction = 0 : OK" if not bad else "contrôle de construction NON NUL : ARRÊT, note B"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("nonreg")
    p.add_argument("--v9", required=True)
    p.add_argument("--v10", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("construction")
    p.add_argument("--v9", required=True)
    p.add_argument("--v10", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    v10_p = Path(a.v10)
    v10 = json.loads(v10_p.read_text())
    inputs = {"v10": sha(v10_p), "script": sha(Path(__file__))}
    if a.cmd == "nonreg":
        v9_p = Path(a.v9)
        inputs["v9"] = sha(v9_p)
        report = {"note": "NOTE-B-P08-oracle-E27-correction avenant v2.4 § 4.2 (5e34be6c)", "inputs": inputs, **nonreg(json.loads(v9_p.read_text()), v10)}
    else:
        v9_p = Path(a.v9)
        inputs["v9"] = sha(v9_p)
        inputs["p08v9_classA_E27.py"] = sha(ROOT / "scripts/p08v9_classA_E27.py")
        report = {"note": "NOTE-B-P08-oracle-E27-correction v2.3 § 4.3 ; contrôle de construction (pA, 25/10), pas une porte",
                  "inputs": inputs, **construction(json.loads(v9_p.read_text()), v10)}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_page", "blocks_failed", "calls_failed", "inputs")}, ensure_ascii=False))
    return 0 if report.get("nonreg_ok", report.get("construction_ok")) is True else 1   # non-zero on failure (pQ, NO-GO 2)


if __name__ == "__main__":
    raise SystemExit(main())
