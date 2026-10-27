"""LOT-P02: identity of the accelerated functions against the S1-fix originals on REAL pool lines (no truth read).

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p02_identity_check.py --page La_Presse_0545076_T0000002 --lines 150
    ... --whole-page      # original (not accelerated) on the whole page: ~5 min, run in the background, ONLY after pJ's go
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
R = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"

from babaorum.experimental import joint_reconcile_selection as S  # noqa: E402
from babaorum.experimental.joint_reconcile import ReconcileConfig, build_reconciliation_graph  # noqa: E402
from babaorum.p02_adapter import rows_to_proposals  # noqa: E402
from babaorum.p02_fast import build_actions_fast, build_graph_fast  # noqa: E402

W1 = {"w_cov": 1.0, "kappa": 10.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--lines", type=int, default=150)
    ap.add_argument("--whole-page", action="store_true")
    ap.add_argument("--around-centre", action="store_true", help="the N lines nearest to the page centre (a spatial window) instead of the first N by proposal_id")
    a = ap.parse_args()
    pages = json.loads((R / "runs/e12/pages.json").read_text())
    pg = next(p for p in pages if p["page"] == a.page)
    rows = [json.loads(l) for l in (R / "pool" / f"{a.page}.jsonl").open(encoding="utf-8")]
    page = {"page_key": a.page, "document_key": a.page.split("_")[-2], "width": pg["width"], "height": pg["height"], "image_sha256": rows[0]["image_sha256"]}
    props, row_of, h, skipped = rows_to_proposals(rows, page)
    props = sorted(props, key=lambda p: p.proposal_id)
    if a.around_centre and not a.whole_page:
        cx, cy = pg["width"] / 2, pg["height"] / 2
        props = sorted(props, key=lambda p: (lambda q: abs(q[0][0] - cx) * 0.2 + abs(q[0][1] - cy))(p.crop_transform.local_to_page(p.baseline_xy[0]) and [p.crop_transform.local_to_page(p.baseline_xy[0])]))[: a.lines]
    elif not a.whole_page:
        props = props[: a.lines]
    cfg = ReconcileConfig(normalization_height_px=h)
    out = {"page": a.page, "lines": len(props), "h": h, "skipped": len(skipped), "mode": "whole page" if a.whole_page else (f"{a.lines} lines nearest to the centre" if a.around_centre else f"first {a.lines} by proposal_id")}
    t0 = time.time(); g1 = build_graph_fast(props, cfg); out["t_graph_fast"] = round(time.time() - t0, 2)
    t0 = time.time(); b1 = build_actions_fast(g1); out["t_actions_fast"] = round(time.time() - t0, 2)
    t0 = time.time(); s1 = S.solve(b1, g1, W1); out["t_solve"] = round(time.time() - t0, 2)
    t0 = time.time(); g0 = build_reconciliation_graph(props, cfg); out["t_graph_original"] = round(time.time() - t0, 2)
    t0 = time.time(); b0 = S.build_actions(g0); out["t_actions_original"] = round(time.time() - t0, 2)
    s0 = S.solve(b0, g0, W1)
    labelled = lambda es: [e for e in es if e["labels"]]
    checks = {
        "nodes_identical": g0["nodes"] == g1["nodes"],
        "labelled_edges_identical": labelled(g0["edges"]) == labelled(g1["edges"]),   # edges with at least one label; no_candidate edges are not compared (pJ)
        "supports_identical": b0["supports"] == b1["supports"],
        "actions_identical": b0["actions"] == b1["actions"],
        "emitted_set_identical": s0["chosen_actions"] == s1["chosen_actions"],
        "solution_identical": s0 == s1,
    }
    out.update(checks)
    out.update({"edges_original": len(g0["edges"]), "edges_fast": len(g1["edges"]), "labelled_edges": len(labelled(g0["edges"])),
                "supports": len(b0["supports"]), "emitted": len(s0["chosen_actions"]), "max_component": max(s0["component_sizes"]),
                "abstention_rate": s0["abstention_rate"]})
    print(json.dumps(out, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
