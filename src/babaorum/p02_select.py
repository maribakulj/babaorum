"""LOT-P02: the order-independent line selection that replaces ``_covered`` (axel branch ``p02-select``, hook ``chains.DEDUP``).

``dedup(page, cands)`` has the signature axel expects: ``cands`` are the candidate line dicts of one page in the walking order (keys ``id``,
``baseline``, ``boundary``, ``_block`` ...); it returns the kept ones, as a subset in the SAME order. The page context (size, image sha,
document key, real zone crops) is registered beforehand with ``register_page`` (built from ``pool/<page>.jsonl``). No text, no confidence and
no truth is read.

Pipeline: adapter (``p02_adapter``) -> accelerated graph and actions (``p02_fast``) -> exact S1-fix ``solve`` (weights W1: w_cov = 1, kappa = 10,
all others 0; ``max_actions`` = 18; the median line height of the candidates is the only per-page scale) -> for an ABSTAINED component, the
fixed fallback: keep the line covering the most supports, then the longest baseline, then the smallest proposal_id, remove the lines sharing a
support with it, repeat (statuses ``emitted_fallback``) -> final check: no two kept lines in hard conflict. A line the solver cannot judge
(no usable geometry) is KEPT and counted. The result does not depend on the order of ``cands``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from babaorum.experimental import joint_reconcile_selection as S
from babaorum.experimental.joint_reconcile import ReconcileConfig
from babaorum.p02_adapter import baseline_length, rows_to_proposals
from babaorum.p02_fast import build_actions_fast, build_graph_fast

WEIGHTS = {"w_cov": 1.0, "kappa": 10.0}  # fixed before any measure; no grid
MAX_ACTIONS = 18
CONTEXT: dict[str, Mapping[str, Any]] = {}
STATS: dict[str, dict[str, Any]] = {}


def register_page(page: str, context: Mapping[str, Any]) -> None:
    """``context``: width, height, image_sha256, document_key, zone_crops {zone_id: zone_crop of the pool}."""
    CONTEXT[page] = context


def context_from_pool(rows: Sequence[Mapping[str, Any]], page_info: Mapping[str, Any]) -> dict[str, Any]:
    crops = {str(r["zone_id"]): r["zone_crop"] for r in rows if r.get("origin") == "zone" and r.get("zone_crop")}
    return {"width": page_info["width"], "height": page_info["height"], "image_sha256": rows[0]["image_sha256"],
            "document_key": page_info.get("document_key"), "zone_crops": crops}


def _fallback(component_pids: list[str], supports_of: Mapping[str, set[str]], length_of: Mapping[str, float]) -> list[str]:
    """Greedy by a FIXED score (supports covered, then longest baseline, then smallest id): independent of the input order."""
    remaining = sorted(component_pids)
    kept: list[str] = []
    while remaining:
        best = min(remaining, key=lambda p: (-len(supports_of[p]), -length_of[p], p))
        kept.append(best)
        taken = supports_of[best]
        remaining = [p for p in remaining if p != best and not (supports_of[p] & taken)]
    return kept


def dedup(page: str, cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctx = CONTEXT[page]
    ranks = ctx.get("source_rank", {})
    rows = [{"origin": "zone", "zone_selected_by_select_blocks": True, "zone_id": c["_block"], "line_id": c["id"], "baseline": c["baseline"],
             "polygon": c["boundary"], "zone_crop": ctx["zone_crops"][str(c["_block"])], "text": "",
             "_src": ranks.get(c["id"])} for c in cands]   # v16: the source rank travels with the candidate (None -> line id)
    props, row_of, h, skipped = rows_to_proposals(rows, {"page_key": page, "document_key": ctx.get("document_key"), "width": ctx["width"],
                                                         "height": ctx["height"], "image_sha256": ctx["image_sha256"]})
    stats = {"candidates": len(cands), "judged": len(props), "unjudged_kept": len(skipped), "abstained_components": 0,
             "fallback_emitted": 0, "emitted_by_solver": 0, "removed": 0}
    STATS[page] = stats
    if not props:
        return list(cands)
    cfg = ReconcileConfig(normalization_height_px=h)
    graph = build_graph_fast(props, cfg)
    built = build_actions_fast(graph)
    solution = S.solve(built, graph, WEIGHTS, max_actions=MAX_ACTIONS)
    emitted = {a.split(":", 1)[1] for a in solution["chosen_actions"]}
    stats["emitted_by_solver"] = len(emitted)
    supports_of = {pid: set() for pid in row_of}
    for sid, members in built["supports"].items():
        for pid in members:
            supports_of[pid].add(sid)
    length_of = {p.proposal_id: baseline_length(p.baseline_xy) for p in props}
    for ab in solution["abstained_components"]:
        stats["abstained_components"] += 1
        fb = _fallback(list(ab["proposals"]), supports_of, length_of)
        stats["fallback_emitted"] += len(fb)
        emitted |= set(fb)
    # final control (that of S1-fix): no two kept lines in hard conflict
    labels = {frozenset(e["source_ids"]): set(e["labels"]) for e in graph["edges"] if e["labels"]}
    pids = sorted(emitted)
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            if labels.get(frozenset((a, b)), set()) & {"reversed_geometry", "direction_conflict"}:
                raise RuntimeError(f"two kept lines are in hard conflict: {a} / {b}")
    kept_ids = {row_of[p]["line_id"] for p in emitted}
    skipped_ids = {s["line_id"] for s in skipped}
    out = [c for c in cands if c["id"] in kept_ids or c["id"] in skipped_ids]
    stats["removed"] = len(cands) - len(out)
    return out
