"""LOT-P02 measure on the dev n° 2 (6 pages of P-07), covered vs p02, c1 and c1+ma. No inference, no tuning. Code of axel = the worktree of p02-select.

    # covered (DEDUP = None, caches of P-07 as they are), read only:
    AXEL_RUNS=<rappel_pages>/runs python scripts/p02_measure.py --mode covered
    # p02 (a COPY of the caches in etude/p02/runs_p02: D-563 and the reads of the added lines are redone on the p02 base):
    AXEL_RUNS=<rappel_pages>/etude/p02/runs_p02 python scripts/p02_measure.py --mode p02
    # control with DEDUP = None on the E12 dev (5 exact CER), ~/axel/runs read only (the worktree has no runs/ of its own):
    AXEL_RUNS=$HOME/axel/runs python scripts/p02_measure.py --mode control
    AXEL_RUNS=<rappel_pages>/runs python scripts/p02_measure.py --mode permutations
    python scripts/p02_measure.py --mode report
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

WT = Path("/private/tmp/claude-501/-Users-marcel/f3fa213e-6b10-4268-affa-db2b929ae165/scratchpad/axel-p02")  # git archive of p02-select 7b753037 (no worktree in ~/axel)
ROOT = Path(__file__).resolve().parents[1]
R = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
OUT_BEFORE = R / "etude/p02"
sys.path[:0] = [str(WT / "src"), str(WT / "scripts"), str(ROOT / "src")]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=("covered", "p02", "control", "permutations", "report"))
ap.add_argument("--out", default=str(R / "etude/p02_v11"), help="output folder (v11: etude/p02_v11 ; 26 pages: etude/p02_26)")
args = ap.parse_args()
OUT = Path(args.out)

if args.mode != "report":
    from axel.layout import chains as C  # noqa: E402
    from axel.layout.gt import load_page  # noqa: E402
    from axel.layout.pagetext import score_page  # noqa: E402
    from axel.layout.fill import select_extra  # noqa: E402
    assert Path(C.__file__).resolve().is_relative_to(WT), f"wrong axel code: {C.__file__}"
    e28 = importlib.import_module("e28_fill")
    e32 = importlib.import_module("e32")
    from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402
    from babaorum.experimental import joint_reconcile_selection as S  # noqa: E402
    from babaorum.experimental.joint_reconcile import _nonzero_segments, _page_baseline, _resample  # noqa: E402
    from babaorum.experimental.joint_reconcile_selection import _overlap_direction, _overlaps  # noqa: E402
    from babaorum.experimental.joint_reconcile import ReconcileConfig  # noqa: E402
    from babaorum import p02_select  # noqa: E402
    from babaorum.p02_adapter import baseline_length, rows_to_proposals  # noqa: E402
    from babaorum.p02_adapter import line_height as p02_line_height  # noqa: E402
    from babaorum.p02_fast import build_actions_fast, build_graph_fast  # noqa: E402
    from babaorum.p02_select import MAX_ACTIONS, WEIGHTS  # noqa: E402

LISTING_CMD = "cd ~/axel && find runs -type f -exec stat -f '%m %z %N' {} + | sort -k3"


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- DEDUP hook (always reset to None)
EXPLAIN: list[dict] = []


@contextlib.contextmanager
def dedup_hook(fn):
    assert C.DEDUP is None, "DEDUP must be None before the hook"
    C.DEDUP = fn
    try:
        yield
    finally:
        C.DEDUP = None


def explain(page, cands, kept):
    """Recompute graph/actions/solution for the same candidates (cheap) to report abstention and the shared-support exclusions."""
    ctx = p02_select.CONTEXT[page]
    ranks = ctx.get("source_rank", {})
    rows = [{"origin": "zone", "zone_selected_by_select_blocks": True, "zone_id": c["_block"], "line_id": c["id"], "baseline": c["baseline"],
             "polygon": c["boundary"], "zone_crop": ctx["zone_crops"][str(c["_block"])], "text": "",
             "_src": ranks.get(c["id"])} for c in cands]
    props, row_of, h, skipped = rows_to_proposals(rows, {"page_key": page, "document_key": ctx.get("document_key"), "width": ctx["width"],
                                                         "height": ctx["height"], "image_sha256": ctx["image_sha256"]})
    if not props:
        return {"candidates": len(cands), "kept": len(kept)}
    graph = build_graph_fast(props, ReconcileConfig(normalization_height_px=h))
    built = build_actions_fast(graph)
    sol = S.solve(built, graph, WEIGHTS, max_actions=MAX_ACTIONS)
    emitted_solver = {a.split(":", 1)[1] for a in sol["chosen_actions"]}
    kept_ids = {k["id"] for k in kept}
    pid_of_line = {r["line_id"]: p for p, r in row_of.items()}
    kept_pids = {pid_of_line[i] for i in kept_ids if i in pid_of_line}
    supports_of = {p: set() for p in row_of}
    for sid, members in built["supports"].items():
        for p in members:
            supports_of[p].add(sid)
    labelled = {frozenset(e["source_ids"]): set(e["labels"]) for e in graph["edges"] if e["labels"]}
    excl = []
    node_of = {n["proposal"]["proposal_id"]: n for n in graph["nodes"]}

    def projection_overlap(a, b):
        sa, sb = _nonzero_segments(tuple(node_of[a]["baseline_page_xy"])), _nonzero_segments(tuple(node_of[b]["baseline_page_xy"]))
        d = _overlap_direction(sa, sb)
        if d is None:
            return float("-inf")
        proj = lambda pts: [x * d[0] + y * d[1] for x, y in pts]
        pa, pb = proj(sa), proj(sb)
        return min(max(pa), max(pb)) - max(min(pa), min(pb))

    for p in sorted(set(row_of) - kept_pids):
        # removed with no labelled edge to any kept line, but sharing a support with a kept line: excluded by the shared support alone
        if any(frozenset((p, k)) in labelled for k in kept_pids):
            continue
        sharers = sorted(k for k in kept_pids if supports_of[p] & supports_of[k])
        if sharers:
            ov = {k: projection_overlap(p, k) for k in sharers}
            # v11 (pJ): END-TO-END exclusion = there EXISTS at least one kept line sharing a support whose projection overlap is zero or negative
            end_to_end = any(o <= 0 for o in ov.values())
            excl.append({"removed": row_of[p]["line_id"], "by": row_of[sharers[0]]["line_id"], "sharers": len(sharers), "end_to_end": end_to_end,
                         "min_projection_overlap_px": min(ov.values()), "length_h": baseline_length(row_of[p]["baseline"]) / h})
    return {"candidates": len(cands), "judged": len(props), "unjudged_kept": len(skipped), "kept": len(kept), "removed": len(cands) - len(kept),
            "emitted_by_solver": len(emitted_solver), "abstained_components": len(sol["abstained_components"]),
            "abstained_actions": sol["abstained_action_count"], "abstention_rate_lines": sol["abstained_action_count"] / max(len(props), 1),
            "max_component": max(sol["component_sizes"]), "exclusion_support_partage": excl,
            "exclusion_bout_a_bout": [e for e in excl if e["end_to_end"]], "median_h": h}


def recording(page, cands):
    kept = p02_select.dedup(page, cands)
    EXPLAIN.append({"page": page, **explain(page, cands, kept)})
    return kept


# ---------------------------------------------------------------- measures of one chain
def gt_geo(glines):
    height = lambda poly: max(q[1] for q in poly) - min(q[1] for q in poly)
    return [GroundTruthLine(g.id, g.baseline, height(g.polygon), "ground_truth") if g.baseline and len(g.baseline) >= 2 and g.polygon
            else GroundTruthLine(g.id, None, None, None, "missing") for g in glines]


def pred_geo(units):
    return [PredictedLine(u.id, [list(q) for q in u.baseline], "predicted") if u.baseline and len(u.baseline) >= 2
            else PredictedLine(u.id, None, None, "missing") for u in units]


def centre(poly):
    xs, ys = [q[0] for q in poly], [q[1] for q in poly]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def in_poly(pt, poly):
    x, y = pt
    inside, n = False, len(poly)
    for i in range(n):
        (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
        if (ya > y) != (yb > y) and x < (xb - xa) * (y - ya) / (yb - ya) + xa:
            inside = not inside
    return inside


def duplicates(units, h):
    """v11: pairs of EMITTED lines for which the solver's own overlap relation holds in one direction or the other (o > 0.5 min(L), t <= 0.5 h)."""
    segs, samp = {}, {}
    for u in units:
        seg = _nonzero_segments(tuple(tuple(q) for q in u.baseline)) if u.baseline and len(u.baseline) >= 2 else None
        if seg is not None:
            segs[u.id], samp[u.id] = seg, _resample(seg, 0.125 * h)
    reach = 0.5 * h + 1e-6
    boxes = {i: (min(x for x, _ in sg) - reach, min(y for _, y in sg) - reach, max(x for x, _ in sg) + reach, max(y for _, y in sg) + reach) for i, sg in segs.items()}
    cell = max(4 * reach, 1.0)
    grid = {}
    for i, b in boxes.items():
        for gx in range(int(b[0] // cell), int(b[2] // cell) + 1):
            for gy in range(int(b[1] // cell), int(b[3] // cell) + 1):
                grid.setdefault((gx, gy), []).append(i)
    seen, n = set(), 0
    for members in grid.values():
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                i, j = sorted((members[a], members[c]))
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                bi, bj = boxes[i], boxes[j]
                if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                    continue
                if _overlaps(segs[i], samp[i], segs[j], h) or _overlaps(segs[j], samp[j], segs[i], h):
                    n += 1
    return n


def measure_chain(units, gt, glines, h=None):
    lg = evaluate_line_geometry(gt_geo(glines), pred_geo(units))
    ps = score_page([u.text for u in units], [g.text for g in glines])
    polys = [r.polygon for r in gt.text_regions if r.lines]
    outside = sum(1 for u in units if not any(in_poly(centre(u.polygon), q) for q in polys))
    return {"units": len(units), "n_gt": lg.n_gt, "matched": lg.matched_count, "recall": lg.matched_count / lg.n_gt,
            "precision": lg.matched_count / max(len(units), 1), "cer_reordered": ps.cer_reordered, "chars": ps.chars,
            "outside_emprise": outside, "share_outside_emprise": outside / max(len(units), 1),
            "matched_gt_ids": sorted(m.gt_id for m in lg.matches), "unit_ids": [u.id for u in units],
            "duplicates_emitted_pairs": duplicates(units, h) if h else None}


def page_contexts(RUNS):
    pages = json.loads((RUNS / "e12/pages.json").read_text())
    for p in pages:
        rows = [json.loads(l) for l in (R / "pool" / f"{p['page']}.jsonl").open(encoding="utf-8")]
        ctx = p02_select.context_from_pool(rows, {"width": p["width"], "height": p["height"], "document_key": p["page"].split("_")[-2]})
        from babaorum.p02_adapter import source_rank_from_lines  # v16: persistent source rank, read before any permutation
        ctx["source_rank"] = source_rank_from_lines(RUNS / "e12/lines/ppdl3_z" / f"{p['page']}.json")
        p02_select.register_page(p["page"], ctx)
    return pages


def main_measure(mode):
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    bench = e32.BENCHES["dev"]
    before = subprocess.run(LISTING_CMD, shell=True, capture_output=True, text=True).stdout
    pages = page_contexts(RUNS)
    out = {"mode": mode, "axel_code": str(WT), "axel_runs_env": str(RUNS), "pages": {}}
    for p in pages:
        name = p["page"]
        gt = load_page(Path(p["xml"]))
        glines = gt.lines()
        entry = {}
        h_page = statistics.median(p02_line_height(r["polygon"], r["baseline"]) for r in
                                   (json.loads(l) for l in (R / "pool" / f"{name}.jsonl").open(encoding="utf-8"))
                                   if r["origin"] == "zone" and r.get("zone_selected_by_select_blocks") and r.get("baseline") and len(r["baseline"]) >= 2)
        entry["h_page"] = h_page
        if mode == "p02":
            with dedup_hook(recording):
                EXPLAIN.clear()
                g_base, g_m = e32.geometry(bench, name, False), e32.geometry(bench, name, True)
                zones = [blk.box for blk in g_base.blocks]
                cands_a = e28.candidates("rest_z", name)
                got_a = select_extra(cands_a, zones, [(u.polygon, u.baseline) for u in g_base.units])
                a_path = e32.add_path(bench, "a_z", name)
                a_path.parent.mkdir(parents=True, exist_ok=True)
                e32._write_add(a_path, len(cands_a), got_a)
                cands_m = e32.candidates(bench, name, True)
                got_m = select_extra(cands_m, zones, [(u.polygon, u.baseline) for u in g_m.units])
                e32._write_add(e32.add_path(bench, "a_m", name), len(cands_m), got_m)
                # reads of the added lines: the chain's own, then the extra, then read_rest_all (every rest candidate was read there)
                texts, ctc = {}, {}
                for src in (R / "read_rest_all" / f"{name}.json", RUNS / "e32/dev/read/mccatmus_press/extra" / f"{name}.json",
                            RUNS / "e28/read/mccatmus_press/a_z" / f"{name}.json"):
                    if src.exists():
                        d = json.loads(src.read_text())
                        texts.update(d["texts"]); ctc.update(d.get("ctc") or {})
                dest = RUNS / "e32/dev/read/mccatmus_press/extra" / f"{name}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps({"reader": "mccatmus_press", "texts": texts, "ctc": ctc}, ensure_ascii=False))
                arms, _zones = e32.build_arms(bench, "c1", name)
                dedup_calls = list(EXPLAIN)
            assert C.DEDUP is None
            entry["dedup_calls"] = dedup_calls
        else:
            arms, _zones = e32.build_arms(bench, "c1", name)
            assert C.DEDUP is None
        entry["c1"] = measure_chain(arms["base"].units, gt, glines, h_page)
        entry["c1+ma"] = measure_chain(arms["ma"].units, gt, glines, h_page)
        entry["c1+m"] = measure_chain(arms["m"].units, gt, glines, h_page)
        out["pages"][name] = entry
        print(mode, name, {k: (entry[k]["units"], entry[k]["matched"], round(entry[k]["cer_reordered"], 5)) for k in ("c1", "c1+ma")}, flush=True)
    after = subprocess.run(LISTING_CMD, shell=True, capture_output=True, text=True).stdout
    out["axel_runs_listing_identical"] = before == after
    out["dedup_after"] = None if C.DEDUP is None else "NOT RESET"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"measure_{mode}.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")


def main_control():
    assert Path(os.environ["AXEL_RUNS"]).resolve() == (Path.home() / "axel/runs").resolve(), "the E12 control reads ~/axel/runs (read only)"
    from axel.eval.doc_bootstrap import DocScore, micro
    bench = e32.BENCHES["dev"]
    before = subprocess.run(LISTING_CMD, shell=True, capture_output=True, text=True).stdout
    docs = []
    for p in e28.dev_pages():
        gt = load_page(Path(p["xml"]))
        arms, _ = e32.build_arms(bench, "c1", p["page"])
        ps = score_page([u.text for u in arms["ma"].units], [g.text for g in gt.lines()])
        docs.append(DocScore(p["page"], ps.cer_reordered * ps.chars, ps.chars))
        assert C.DEDUP is None
    value = micro(docs)
    value = value["value"] if isinstance(value, dict) else value
    after = subprocess.run(LISTING_CMD, shell=True, capture_output=True, text=True).stdout
    res = {"dedup": None, "cer_0_micro_c1ma": value, "expected": 0.05188611919580012, "exact": abs(value - 0.05188611919580012) < 1e-12,
           "axel_runs_listing_identical": before == after, "pages": len(docs), "code": str(WT), "chains_file": C.__file__}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "control_dedup_none.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["exact"] and res["axel_runs_listing_identical"] else 1


def main_permutations():
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = page_contexts(RUNS)
    rng = random.Random(20261017)
    out = {}
    for name in ("La_Presse_0545076_T0000002", "Le_Matin_0551841_T0000003", "Marie-Claire_4621234_T0000008"):
        data = json.loads((RUNS / "e12/lines/ppdl3_z" / f"{name}.json").read_text())
        blocks = C.select_blocks(data["blocks"], C.FROZEN["ppdl3_z"])
        blocks.sort(key=lambda b: b.get("rank", 0))
        tup = C._tup
        def cands_of(blks):
            res = []
            for b in blks:
                for ln in b["lines"]:
                    if len(ln["baseline"]) < 2 or len(ln["boundary"]) < 3:
                        continue
                    res.append({**ln, "_box": C.bbox(tup(ln["boundary"])), "_block": b["id"]})
            return res
        base = cands_of(blocks)
        ref = sorted(c["id"] for c in p02_select.dedup(name, base))
        stats_ref = dict(p02_select.STATS[name])
        identical = 0
        for k in range(20):
            shuffled_blocks = [dict(b, lines=rng.sample(b["lines"], len(b["lines"]))) for b in rng.sample(blocks, len(blocks))]
            kept = sorted(c["id"] for c in p02_select.dedup(name, cands_of(shuffled_blocks)))
            identical += kept == ref
        out[name] = {"candidates": len(base), "kept_reference": len(ref), "permutations": 20, "identical": identical, "stats": stats_ref}
        print(name, out[name], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "permutations.json").write_text(json.dumps(out, indent=1) + "\n")
    assert all(v["identical"] == 20 for v in out.values()), "a permutation changed the emitted set"


def main_report():
    """v14: the 10 % set is fixed by COVERED c1+ma; bootstrap by groups (number AND title); balance by newspaper title; three controls per chain."""
    import re
    from babaorum.g0_keys import title_of_prefix

    cov = json.loads((OUT / "measure_covered.json").read_text())
    p02 = json.loads((OUT / "measure_p02.json").read_text())
    names = sorted(cov["pages"])
    num = lambda n: str(int(re.search(r"(?<!\d)(\d{6,8})(?!\d)", n).group(1)))
    group_of = {n: (num(n), title_of_prefix(n) or n) for n in names}       # P-07b groups: same number AND same title
    title_of = {n: (title_of_prefix(n) or n) for n in names}

    def totals(src, chain, subset):
        rows = [src["pages"][n][chain] for n in subset]
        gt, w = sum(r["n_gt"] for r in rows), sum(r["chars"] for r in rows)
        return {"R_final": sum(r["matched"] for r in rows) / gt, "precision": sum(r["matched"] for r in rows) / sum(r["units"] for r in rows),
                "cer_reordered": sum(r["cer_reordered"] * r["chars"] for r in rows) / w, "lines": sum(r["units"] for r in rows)}

    def boot_groups(chain, subset, draws=10000):
        """Draw as many groups as the set contains, uniformly with replacement; a group contributes ONLY its pages of the set.
        dR = micro recall (matched true lines over true lines), p02 minus covered; dCER weighted by true characters. A draw without any true line
        or any character STOPS the lot."""
        groups = sorted({group_of[n] for n in subset})
        if len(groups) < 2:
            return {"groups": len(groups), "switch_possible": False, "reason": "fewer than two groups: no switch"}
        pages_of = {g: [n for n in subset if group_of[n] == g] for g in groups}
        rng = np.random.default_rng(20261017)                    # a NEW generator for this computation
        dr, dc = [], []
        for _ in range(draws):
            picked = rng.integers(0, len(groups), len(groups))
            ps = [n for gi in picked for n in pages_of[groups[gi]]]
            gtl = sum(cov["pages"][n][chain]["n_gt"] for n in ps)
            chars = sum(cov["pages"][n][chain]["chars"] for n in ps)
            if gtl == 0 or chars == 0:
                raise SystemExit("a bootstrap draw has no true line or no character: stop")
            dr.append(sum(p02["pages"][n][chain]["matched"] for n in ps) / gtl - sum(cov["pages"][n][chain]["matched"] for n in ps) / gtl)
            dc.append(sum(p02["pages"][n][chain]["cer_reordered"] * p02["pages"][n][chain]["chars"] for n in ps) / chars
                      - sum(cov["pages"][n][chain]["cer_reordered"] * cov["pages"][n][chain]["chars"] for n in ps) / chars)
        return {"groups": len(groups), "switch_possible": True, "seed": 20261017, "draws": draws,
                "dR_micro_ci95": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
                "dCER_char_weighted_ci95": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))]}

    clean = [n for n in names if cov["pages"][n]["c1+ma"]["share_outside_emprise"] <= 0.10]       # FIXED BY covered c1+ma (v14)
    report = {"pages": names, "pages_within_10pct_fixed_by_covered": clean,
              "pages_within_10pct_by_p02_reported_only": [n for n in names if p02["pages"][n]["c1+ma"]["share_outside_emprise"] <= 0.10],
              "share_outside_emprise": {n: {"covered": cov["pages"][n]["c1+ma"]["share_outside_emprise"], "p02": p02["pages"][n]["c1+ma"]["share_outside_emprise"]} for n in names},
              "groups": {n: list(group_of[n]) for n in names}, "aggregates": {}, "bootstrap_by_groups": {}, "per_page": {}, "recovered_lost_per_page": {}, "by_title": {}}
    for chain in ("c1", "c1+ma"):
        for label, subset in (("all_pages", names), ("pages_within_10pct", clean)):
            c, q = totals(cov, chain, subset), totals(p02, chain, subset)
            report["aggregates"][f"{chain}|{label}"] = {"covered": c, "p02": q, "diff": {k: q[k] - c[k] for k in c}}
        report["bootstrap_by_groups"][chain] = boot_groups(chain, clean)
        for n in names:
            a, b = cov["pages"][n][chain], p02["pages"][n][chain]
            ga, gb = set(a["matched_gt_ids"]), set(b["matched_gt_ids"])
            report["per_page"].setdefault(n, {})[chain] = {"covered": {k: a[k] for k in ("units", "matched", "recall", "precision", "cer_reordered")},
                                                          "p02": {k: b[k] for k in ("units", "matched", "recall", "precision", "cer_reordered")}}
            report["recovered_lost_per_page"].setdefault(n, {})[chain] = {"recovered": len(gb - ga), "lost": len(ga - gb), "emitted_covered": a["units"], "emitted_p02": b["units"]}
        # balance by NEWSPAPER TITLE, every page (no 10 % filter): lost and recovered = differences of the matched true ids, summed over the title
        for t in sorted(set(title_of.values())):
            ts = [n for n in names if title_of[n] == t]
            rec = sum(report["recovered_lost_per_page"][n][chain]["recovered"] for n in ts)
            lost = sum(report["recovered_lost_per_page"][n][chain]["lost"] for n in ts)
            report["by_title"].setdefault(t, {})[chain] = {"pages": len(ts), "recovered": rec, "lost": lost, "lost_le_recovered": lost <= rec}
    calls = {n: p02["pages"][n]["dedup_calls"] for n in names}
    uniq = {n: {"c1": calls[n][0], "c1+ma": calls[n][1]} for n in names}          # base-chain call for c1, M-chain call for c1+ma (the next two repeat)
    ctrl = {}
    for chain in ("c1", "c1+ma"):
        cs = [uniq[n][chain] for n in names]
        e2e = [dict(page=n, **e) for n in names for e in uniq[n][chain].get("exclusion_bout_a_bout", [])]
        judged = sum(c.get("judged", 0) for c in cs)
        ctrl[chain] = {"end_to_end_exclusions": len(e2e), "end_to_end_list": e2e,
                       "exclusion_support_partage": sum(len(c.get("exclusion_support_partage", [])) for c in cs),
                       "duplicates": {"covered": sum(cov["pages"][n][chain]["duplicates_emitted_pairs"] for n in names), "p02": sum(p02["pages"][n][chain]["duplicates_emitted_pairs"] for n in names)},
                       "abstained_proposals_before_fallback": sum(c.get("abstained_actions", 0) for c in cs), "judged_proposals": judged,
                       "abstention_rate": sum(c.get("abstained_actions", 0) for c in cs) / max(judged, 1),
                       "unjudged_kept": sum(c.get("unjudged_kept", 0) for c in cs), "removed": sum(c.get("removed", 0) for c in cs), "candidates": sum(c.get("candidates", 0) for c in cs)}
        ctrl[chain].update({"zero_end_to_end_ok": ctrl[chain]["end_to_end_exclusions"] == 0, "duplicates_ok": ctrl[chain]["duplicates"]["p02"] <= ctrl[chain]["duplicates"]["covered"],
                            "abstention_ok": ctrl[chain]["abstention_rate"] <= 0.10})
    report["controls_per_chain"] = ctrl
    bs = report["bootstrap_by_groups"]["c1+ma"]
    report["switch_criterion_v13_v14"] = {
        "scope": "c1+ma, pages within 10 % outside the transcribed extent FIXED BY covered, bootstrap by groups (number and title), default_rng(20261017), 10 000 draws",
        "1_recall_lower_bound_ge_minus_0.002": bool(bs.get("switch_possible") and bs["dR_micro_ci95"][0] >= -0.002),
        "2_cer_upper_bound_le_plus_0.002": bool(bs.get("switch_possible") and bs["dCER_char_weighted_ci95"][1] <= 0.002),
        "3_by_title_lost_le_recovered_c1ma": all(v["c1+ma"]["lost_le_recovered"] for v in report["by_title"].values()),
        "3_by_title_c1_reported": {t: v["c1"] for t, v in report["by_title"].items()},
        "4_zero_end_to_end_both_chains": all(c["zero_end_to_end_ok"] for c in ctrl.values()),
        "4_duplicates_p02_le_covered_both_chains": all(c["duplicates_ok"] for c in ctrl.values()),
        "4_abstention_le_10pct_both_chains": all(c["abstention_ok"] for c in ctrl.values())}
    k = report["switch_criterion_v13_v14"]
    k["proposal_to_pA"] = all(v for key, v in k.items() if key[0] in "1234" and isinstance(v, bool))
    (OUT / "p02_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"criterion": k, "bootstrap": bs, "controls": {c: {x: y for x, y in v.items() if x != "end_to_end_list"} for c, v in ctrl.items()}, "pages_within_10pct": clean}, indent=1)[:5000])


if args.mode in ("covered", "p02"):
    main_measure(args.mode)
elif args.mode == "control":
    raise SystemExit(main_control())
elif args.mode == "permutations":
    main_permutations()
else:
    main_report()
