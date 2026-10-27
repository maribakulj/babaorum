"""LOT-P08 (v1-v7): decomposition of the page CER by successive oracles. Pure functions, no I/O, no axel import.

The caller supplies the page score (``score(pred_texts, gt_texts) -> float``, in practice axel's ``cer_reordered``) and the links
between predicted and true lines (``line_geometry.v1`` matching, E27 components). Fixed by LOT-P08 and never tuned:

    CER_0   the chain as it is
    1       reading   : a predicted line matched 1 to 1 (outside every group) receives the true text of its true line
    1b      partielles: a predicted line of an E27 *partielle* component receives the true text of its true line, at its position
    2       cutting   : each group (E27 SPLIT / MERGE_VERT / MERGE_COL / TANGLE) is replaced, at the position of its first predicted
                        line, by the true lines of the group in true order; its other predicted lines disappear
    3       missed    : each true line of a MISSED component is inserted right after the counterpart (matched, partial, or member
                        of a group) of the nearest PRECEDING true line that has one; at the head if none
    4       insertions: only the predicted lines of EXTRA components (linked to no true line) are removed; ``ordre`` is what remains
Every predicted line is rewritten ONCE, by the first oracle that concerns it. The parts are differences of consecutive CERs, so
CER_0 - CER_4 is exactly their sum. ``part_positions`` (a part of ``ordre``, outside the sum) puts every reinserted line right in
front of its true successor. The control ``true_order_cer`` sorts the final entries by true index: it must be 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

Score = Callable[[Sequence[str], Sequence[str]], float]
FORWARD = ("lecture", "partielles", "decoupe", "manquees")
REVERSE = ("manquees", "decoupe", "partielles", "lecture")
#: LOT-P08 v9 (NOTE-DECISION v4 a v7): the four passes of oracle 3, disjoint, in this order (reversed for the reverse order)
MISSED_CLASSES = ("ecartees", "dans_zone", "hors_zone", "sans_centre")


@dataclass(frozen=True)
class Group:
    kind: str
    pred_ids: tuple[str, ...]
    gt_ids: tuple[str, ...]


@dataclass(frozen=True)
class Links:
    matched: Mapping[str, str]  # pred_id -> gt_id, 1 to 1 (line_geometry.v1), outside every group
    groups: Sequence[Group]
    missed: Sequence[str]  # gt ids of MISSED components
    extra: Sequence[str] = ()  # pred ids of EXTRA components
    partial: Mapping[str, str] = field(default_factory=dict)  # pred_id -> gt_id, E27 partielle components


@dataclass
class _Entry:
    kind: str  # pred | block | missed
    text: str
    pid: str | None = None
    gid: int | None = None
    gt_id: str | None = None
    done: bool = False  # already rewritten with its true text


@dataclass
class Decomposition:
    cer: dict[str, float] = field(default_factory=dict)
    parts: dict[str, float] = field(default_factory=dict)
    order: tuple[str, ...] = FORWARD
    counts: dict[str, int] = field(default_factory=dict)
    true_order_cer: float | None = None


def _texts(entries: Sequence[_Entry]) -> list[str]:
    return [e.text for e in entries]


def _rewrite(entries: list[_Entry], mapping: Mapping[str, str], gt_text: Mapping[str, str]) -> None:
    for e in entries:
        if e.kind == "pred" and e.pid in mapping and not e.done:
            e.text, e.gt_id, e.done = gt_text[mapping[e.pid]], mapping[e.pid], True


def _apply_decoupe(entries: list[_Entry], links: Links, gt_text: Mapping[str, str], gt_order: Mapping[str, int]) -> None:
    out: list[_Entry] = []
    seen: set[int] = set()
    group_of = {p: i for i, g in enumerate(links.groups) for p in g.pred_ids}
    for e in entries:
        gi = group_of.get(e.pid) if e.kind == "pred" and e.pid is not None else None
        if gi is None:
            out.append(e)
            continue
        if gi in seen:
            continue
        seen.add(gi)
        for gt_id in sorted(links.groups[gi].gt_ids, key=gt_order.__getitem__):
            out.append(_Entry("block", gt_text[gt_id], gid=gi, gt_id=gt_id, done=True))
    entries[:] = out


def _anchor_index(entries: Sequence[_Entry], links: Links, gt_id: str, group_of_gt: Mapping[str, int],
                  pred_of_gt: Mapping[str, str]) -> int | None:
    gi = group_of_gt.get(gt_id)
    if gi is not None:
        blocks = [i for i, e in enumerate(entries) if e.kind == "block" and e.gid == gi]
        if blocks:
            return blocks[-1]
        members = [i for i, e in enumerate(entries) if e.kind == "pred" and e.pid in links.groups[gi].pred_ids]
        return members[0] if members else None
    pid = pred_of_gt.get(gt_id)
    if pid is None:
        return None
    for i, e in enumerate(entries):
        if e.kind == "pred" and e.pid == pid:
            return i
    return None


def _apply_manquees(entries: list[_Entry], links: Links, gt_text: Mapping[str, str], gt_order: Mapping[str, int],
                    only: Callable[[str], bool] | None = None) -> None:
    group_of_gt = {g_id: i for i, g in enumerate(links.groups) for g_id in g.gt_ids}
    pred_of_gt = {g: p for p, g in {**links.partial, **links.matched}.items()}
    ordered = sorted(gt_order, key=gt_order.__getitem__)
    for m in sorted((x for x in links.missed if only is None or only(x)), key=gt_order.__getitem__):
        anchor = None
        for prev in reversed(ordered[: gt_order[m]]):
            anchor = _anchor_index(entries, links, prev, group_of_gt, pred_of_gt)
            if anchor is not None:
                break
        pos = 0 if anchor is None else anchor + 1
        while pos < len(entries) and entries[pos].kind == "missed" and gt_order[entries[pos].gt_id] < gt_order[m]:
            pos += 1
        entries.insert(pos, _Entry("missed", gt_text[m], gt_id=m, done=True))


def _entry_gt(e: _Entry, links: Links) -> str | None:
    if e.gt_id is not None:
        return e.gt_id
    if e.kind == "pred" and e.pid is not None:
        return links.matched.get(e.pid) or links.partial.get(e.pid)
    return None


def _true_order_cer(entries: Sequence[_Entry], links: Links, gt_order: Mapping[str, int], score: Score, gtt: Sequence[str]) -> float:
    """Control: impose the true order on the final entries. Entries with no true line go last (they make the CER > 0)."""
    key = lambda e: gt_order.get(_entry_gt(e, links) or "", len(gt_order))
    return score(_texts(sorted(entries, key=key)), gtt)


def _positions_cer(entries: Sequence[_Entry], links: Links, gt_order: Mapping[str, int], score: Score, gtt: Sequence[str]) -> float:
    """part_positions helper: every reinserted line goes right in front of its true successor (smallest strictly larger true
    index among the non-reinserted entries), or at the end; the other entries keep the chain order."""
    fixed = [e for e in entries if e.kind == "pred"]
    reins = sorted((e for e in entries if e.kind != "pred"), key=lambda e: gt_order[e.gt_id])
    out = list(fixed)
    for r in reins:
        succ = [(gt_order[g], i) for i, e in enumerate(out) if e.kind == "pred" and (g := _entry_gt(e, links)) is not None and gt_order[g] > gt_order[r.gt_id]]
        # insertion before the successor, after the reinserted lines already placed there that precede it in true order
        pos = min(succ)[1] if succ else len(out)
        while pos > 0 and out[pos - 1].kind != "pred" and gt_order[out[pos - 1].gt_id] > gt_order[r.gt_id]:
            pos -= 1
        out.insert(pos, r)
    return score(_texts(out), gtt)


def decompose_page(pred: Sequence[tuple[str, str]], gt: Sequence[tuple[str, str]], links: Links, score: Score,
                   order: Sequence[str] = FORWARD, zone_of_missed: Callable[[str], bool] | None = None,
                   check_true_order: bool = True, missed_class: Mapping[str, str] | None = None) -> Decomposition:
    gt_text = {i: t for i, t in gt}
    gt_order = {i: k for k, (i, _) in enumerate(gt)}
    entries = [_Entry("pred", t, pid=i) for i, t in pred]
    gtt = [t for _, t in gt]
    res = Decomposition(order=tuple(order))
    res.cer["0"] = score(_texts(entries), gtt)
    previous = res.cer["0"]
    for name in order:
        if name == "manquees" and missed_class is not None:
            missing = [x for x in links.missed if x not in missed_class]
            if missing:
                raise ValueError(f"every missed line needs a class: {missing[:3]}")
            if any(c not in MISSED_CLASSES for c in missed_class.values()):
                raise ValueError("unknown missed class")
            sub = MISSED_CLASSES[::-1] if tuple(order) == REVERSE else MISSED_CLASSES
            total = 0.0
            for cls in sub:
                _apply_manquees(entries, links, gt_text, gt_order, only=lambda x, c=cls: missed_class.get(x) == c)
                now = score(_texts(entries), gtt)
                res.parts[f"manquees_{cls}"] = previous - now
                total += previous - now
                previous = now
            res.parts["manquees"] = total
            res.cer["after_manquees"] = previous
            continue
        if name == "manquees" and zone_of_missed is not None:
            _apply_manquees(entries, links, gt_text, gt_order, only=zone_of_missed)
            mid = score(_texts(entries), gtt)
            res.parts["manquees_dans_zone"] = previous - mid
            _apply_manquees(entries, links, gt_text, gt_order, only=lambda x: not zone_of_missed(x))
            now = score(_texts(entries), gtt)
            res.parts["manquees_hors_zone"] = mid - now
        else:
            if name == "lecture":
                _rewrite(entries, links.matched, gt_text)
            elif name == "partielles":
                _rewrite(entries, links.partial, gt_text)
            elif name == "decoupe":
                _apply_decoupe(entries, links, gt_text, gt_order)
            else:
                _apply_manquees(entries, links, gt_text, gt_order)
            now = score(_texts(entries), gtt)
        res.cer["after_" + name] = now
        res.parts[name] = previous - now
        previous = now
    extra = set(links.extra)
    entries[:] = [e for e in entries if not (e.kind == "pred" and e.pid in extra)]
    res.cer["4"] = score(_texts(entries), gtt)
    res.parts["insertions"] = previous - res.cer["4"]
    res.parts["ordre"] = res.cer["4"]
    res.true_order_cer = _true_order_cer(entries, links, gt_order, score, gtt)
    if check_true_order and abs(res.true_order_cer) > 1e-9:
        raise ValueError(f"true-order control failed: CER {res.true_order_cer} != 0 (a true line is covered by no oracle)")
    res.parts["part_positions"] = res.cer["4"] - _positions_cer(entries, links, gt_order, score, gtt)
    res.counts = {"pred": len(pred), "gt": len(gt), "matched": len(links.matched), "partial": len(links.partial),
                  "partial_also_matched": sum(1 for p in links.partial if p in links.matched), "groups": len(links.groups),
                  "missed": len(links.missed), "extra": len(links.extra), "pred_in_groups": sum(len(g.pred_ids) for g in links.groups)}
    return res


def parts_sum_error(d: Decomposition) -> float:
    """(CER_0 - CER_4) minus the sum of the oracle parts and insertions: must be 0 up to rounding."""
    return (d.cer["0"] - d.cer["4"]) - (sum(d.parts[n] for n in d.order) + d.parts["insertions"])


# --- LOT-P08 v9 helpers: groups of pages, bootstrap by groups, disagreement pages ---------------------

def group_of_page(name: str, number: str, title: str) -> tuple[str, str]:
    """Group = the same NUMBER and the same TITLE (LOT-P07b v4). Pages that share neither are alone."""
    return number, title


def bootstrap_by_groups(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], draws: int = 10000, seed: int = 20261017) -> dict[str, Any]:
    """Draw as many groups as there are, with replacement; a group contributes ALL its pages. Statistic: the mean of each key weighted by
    the true characters (rows carry ``group``, ``chars`` and the keys). A draw without any character stops the computation."""
    import numpy as np

    groups = sorted({r["group"] for r in rows})
    by_group = {g: [r for r in rows if r["group"] == g] for g in groups}
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(draws):
        picked = rng.integers(0, len(groups), len(groups))
        drawn = [r for gi in picked for r in by_group[groups[gi]]]
        w = sum(r["chars"] for r in drawn)
        if w == 0:
            raise ValueError("a bootstrap draw has no true character: stop")
        for k in keys:
            out[k].append(sum(r["chars"] * r[k] for r in drawn) / w)
    return {"groups": len(groups), "draws": draws, "seed": seed,
            "ci95": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}}


def split_disagreements(results: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """LOT-P08 v9 point 5: a page whose true-order control fails because E27 and line_geometry disagree is reported apart and
    leaves the aggregates; every other page stays in them. Returns (kept, reported_apart)."""
    kept, apart = {}, {}
    for page, value in results.items():
        if isinstance(value, dict) and value.get("disagreement"):
            apart[page] = value["disagreement"]
        else:
            kept[page] = value
    return kept, apart


#: LOT-P08 v9 (demande p1 ac5dee3a, decision pA 21/10): the check pages named by the request, from random.Random(2) on the
#: 20-page list of tirage_p07b.json (LOT-P07b v4): (decomposition, R_pool). Not recomputed on the 26 pages.
CHECK_PAGES = ("L_OEuvre_4616168_T0000002", "L_OEuvre_4617562_T0000007")


def disagreement_table(apart: Mapping[str, str]) -> str:
    """Markdown table of the pages reported apart (LOT-P08 v9 point 5). They are in no aggregate."""
    lines = ["| page | motif |", "|---|---|"]
    lines += [f"| {page} | {motif.replace('|', '/')} |" for page, motif in sorted(apart.items())]
    return "\n".join(lines) + "\n"


def check_pages_for(names) -> tuple[str, str]:
    """(decomposition, R_pool) check pages. On the 26 pages: CHECK_PAGES (the request). On a set that lacks them (the 6 pages of
    non-regression): random.Random(1) on the sorted names, as the request keeps seed 1 for the 6 pages."""
    import random
    names = sorted(names)
    if set(CHECK_PAGES) <= set(names):
        return CHECK_PAGES
    picked = random.Random(1).sample(names, 2)
    return picked[0], picked[1]

