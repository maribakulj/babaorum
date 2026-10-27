"""Pure helpers for the pipeline bench, part b (dev exposé, D-600). Stdlib only.

Descriptive columns, never criteria: reading-order tau with its support, the
"ligne réussie" column (Kodym et Hradiš, PERO 2021), and ALTO checks that go
beyond the XSD. D-217 is OUR reordering; it is never called flex character accuracy.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

#: pages whose ground-truth order is only a sort by height (D-212): order metrics do not apply
ORDER_NOT_APPLICABLE_PAGES = frozenset({"Ce-soir_7633477_D0000007", "Regards_7635840_D0000005"})
LINE_CER_THRESHOLD = 0.15


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def line_cer(gt_text: str, pred_text: str) -> float:
    """Edit distance over the ground-truth length; an empty truth is 0 only if the prediction is empty too."""
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    return levenshtein(gt_text, pred_text) / len(gt_text)


def kendall_tau(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Kendall tau between two orders of the same items (no ties): (concordant - discordant) / pairs.

    ``pairs`` holds one (rank in the truth, rank in the prediction) per matched line.
    Fewer than two items give None.
    """
    n = len(pairs)
    if n < 2:
        return None
    ordered = sorted(pairs)
    concordant = discordant = 0
    for i in range(n):
        yi = ordered[i][1]
        for j in range(i + 1, n):
            if ordered[j][1] > yi:
                concordant += 1
            elif ordered[j][1] < yi:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def order_report(page: str, matches: Sequence[tuple[str, str]], gt_order: Mapping[str, int],
                 pred_order: Mapping[str, int], n_gt: int) -> dict:
    """Tau and its support (matched lines over truth lines) for one page, or 'non applicable'."""
    support = {"matched": len(matches), "gt_lines": n_gt,
               "ratio": (len(matches) / n_gt) if n_gt else None}
    if page in ORDER_NOT_APPLICABLE_PAGES:
        return {"status": "non applicable", "tau": None, "support": support,
                "reason": "l'ordre de la vérité 1937 n'est qu'un tri par hauteur (D-212)"}
    pairs = [(gt_order[g], pred_order[p]) for g, p in matches]
    tau = kendall_tau(pairs)
    return {"status": "computed" if tau is not None else "insufficient_support", "tau": tau, "support": support}


def line_success(page: str, matches: Sequence[tuple[str, str]], gt_text: Mapping[str, str],
                 pred_text: Mapping[str, str], gt_order: Mapping[str, int], pred_order: Mapping[str, int],
                 n_gt: int, threshold: float = LINE_CER_THRESHOLD) -> dict:
    """Share of truth lines that are matched, read under the threshold, and (where applicable) at the same rank.

    "Same rank" means the same position among the matched lines, in truth order and in prediction order.
    Unmatched truth lines count as failures. On pages where order does not apply, only the first two
    conditions are evaluated and the order condition is reported as 'non applicable'.
    """
    rank_gt = {g: r for r, g in enumerate(sorted((g for g, _ in matches), key=lambda g: gt_order[g]))}
    rank_pred = {p: r for r, p in enumerate(sorted((p for _, p in matches), key=lambda p: pred_order[p]))}
    cer_ok = rank_ok = both_ok = 0
    for g, p in matches:
        ok = line_cer(gt_text[g], pred_text[p]) < threshold
        same = rank_gt[g] == rank_pred[p]
        cer_ok += ok
        rank_ok += same
        both_ok += ok and same
    # variant "same predecessor" (post hoc, descriptive, in the spirit of Clausner 2013 direct-successor relations):
    # the matched predecessor of the true line must also be the predecessor of the predicted line
    gt_sorted = sorted((g for g, _ in matches), key=lambda g: gt_order[g])
    pred_sorted = sorted((p for _, p in matches), key=lambda p: pred_order[p])
    prev_g = {g: (gt_sorted[i - 1] if i else None) for i, g in enumerate(gt_sorted)}
    prev_p = {p: (pred_sorted[i - 1] if i else None) for i, p in enumerate(pred_sorted)}
    partner = dict(matches)
    pred_ok = 0
    for g, p in matches:
        same_pred = (prev_g[g] is None and prev_p[p] is None) or (
            prev_g[g] is not None and prev_p[p] is not None and partner[prev_g[g]] == prev_p[p])
        pred_ok += bool(same_pred and line_cer(gt_text[g], pred_text[p]) < threshold)
    applicable = page not in ORDER_NOT_APPLICABLE_PAGES
    return {"gt_lines": n_gt, "matched": len(matches), "matched_and_cer_under_threshold": cer_ok,
            "matched_cer_and_same_predecessor": pred_ok if applicable else None,
            "share_success_with_predecessor": (pred_ok / n_gt) if (n_gt and applicable) else None,
            "matched_cer_and_same_rank": both_ok if applicable else None,
            "order_condition": "applicable" if applicable else "non applicable",
            "share_matched_and_cer": (cer_ok / n_gt) if n_gt else None,
            "share_success_with_order": (both_ok / n_gt) if (n_gt and applicable) else None,
            "threshold": threshold}


def alto_checks(xml: str | None, graph_line_count: int) -> dict:
    """Checks beyond the XSD: BASELINE on every line, order consistency, lines lost on export."""
    if xml is None:
        return {"status": "not_exported", "lines_lost": graph_line_count}
    root = ET.fromstring(xml)
    lines = [e for e in root.iter() if e.tag.endswith("}TextLine") or e.tag == "TextLine"]
    with_baseline = sum(1 for e in lines if e.get("BASELINE"))
    ids = {e.get("ID") for e in root.iter() if e.get("ID")}
    nexts = [(e.get("ID"), e.get("IDNEXT")) for e in root.iter() if e.get("IDNEXT")]
    has_reading_order = any(e.tag.endswith("}ReadingOrder") or e.tag == "ReadingOrder" for e in root.iter())
    dangling = [t for _, t in nexts if t not in ids]
    successor = dict(nexts)
    cyclic = False
    for start in successor:
        seen, cursor = set(), start
        while cursor in successor:
            if cursor in seen:
                cyclic = True
                break
            seen.add(cursor)
            cursor = successor[cursor]
        if cyclic:
            break
    if not nexts and not has_reading_order:
        order = "not_present"
    else:
        order = "consistent" if not dangling and not cyclic else "inconsistent"
    return {"status": "exported", "text_lines": len(lines), "baseline_present": with_baseline,
            "baseline_missing": len(lines) - with_baseline, "order": order,
            "idnext_links": len(nexts), "idnext_dangling": len(dangling), "idnext_cyclic": cyclic,
            "reading_order_element": has_reading_order, "lines_lost": max(graph_line_count - len(lines), 0)}


def page_checks(xml: str | None, graph_line_count: int) -> dict:
    """PAGE XML counterpart: TextLine count, Baseline on every line, reading-order presence, lines lost."""
    if xml is None:
        return {"status": "not_exported", "lines_lost": graph_line_count}
    root = ET.fromstring(xml)
    lines = [e for e in root.iter() if e.tag.endswith("}TextLine") or e.tag == "TextLine"]
    with_baseline = sum(1 for e in lines if any((c.tag.endswith("}Baseline") or c.tag == "Baseline") and c.get("points") for c in e))
    has_order = any(e.tag.endswith("}ReadingOrder") or e.tag == "ReadingOrder" for e in root.iter())
    return {"status": "exported", "text_lines": len(lines), "baseline_present": with_baseline,
            "baseline_missing": len(lines) - with_baseline, "reading_order_element": has_order,
            "lines_lost": max(graph_line_count - len(lines), 0)}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def texts_from_alto(xml: str) -> list[str]:
    """Line texts read back from an ALTO file: the String CONTENT of each TextLine, joined by one space, in file order."""
    root = ET.fromstring(xml)
    return [" ".join(c.get("CONTENT", "") for c in line if _local(c.tag) == "String")
            for line in root.iter() if _local(line.tag) == "TextLine"]


def texts_from_page(xml: str) -> list[str]:
    """Line texts read back from a PAGE file: the TextEquiv/Unicode of each TextLine, in file order."""
    root = ET.fromstring(xml)
    out = []
    for line in root.iter():
        if _local(line.tag) != "TextLine":
            continue
        text = ""
        for child in line:
            if _local(child.tag) == "TextEquiv":
                for u in child:
                    if _local(u.tag) == "Unicode":
                        text = u.text or ""
        out.append(text)
    return out
