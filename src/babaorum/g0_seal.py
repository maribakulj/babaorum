"""Decision logic for sealing a test split (presse_e7 test v1). Pure functions, no I/O.

A page is refused when it has no truth lines, or when it touches another collection at the
PAGE or ISSUE level (every match is kept). A sealed page whose source is not a single credited
document, or whose title is already seen elsewhere, is sealed with headline=false: the measure
that counts (headline) is computed on headline=true pages only. Nothing here releases anything.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

LEVELS = ("page", "issue", "title")
REFUSING_LEVELS = ("page", "issue")


class SealStop(ValueError):
    """An exact count expected by the lot is not reproduced: the seal must not be written."""


def expect_counts(actual: Mapping[str, int], expected: Mapping[str, int]) -> None:
    wrong = {k: (actual.get(k), v) for k, v in expected.items() if actual.get(k) != v}
    if wrong:
        raise SealStop("expected counts not reproduced (found, expected): " +
                       "; ".join(f"{k}={a}!={e}" for k, (a, e) in wrong.items()))


def find_overlaps(page_keys: Mapping[str, Sequence[str]],
                  collections: Mapping[str, Mapping[str, set[str] | None]]) -> list[dict[str, Any]]:
    """Every match of this page's keys in any collection, at any level. A level with no keys never matches."""
    found = []
    for name, levels in sorted(collections.items()):
        for level in LEVELS:
            values = levels.get(level)
            if not values:
                continue
            for key in page_keys.get(level, ()):
                if key in values:
                    found.append({"collection": name, "level": level, "key": key})
    return found


def decide(*, truth_lines: int, issue_keys: Sequence[str], overlaps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    refusals: list[str] = []
    if truth_lines == 0:
        refusals.append("no_truth_lines")
    for level in REFUSING_LEVELS:
        if any(o["level"] == level for o in overlaps):
            refusals.append(f"overlap_{level}")
    title_seen = any(o["level"] == "title" for o in overlaps)
    if refusals:
        return {"decision": "refused", "refusal_reasons": refusals, "headline": False,
                "headline_false_reasons": ["refused"], "title_seen": title_seen,
                "credited": bool(issue_keys), "montage": len(issue_keys) > 1, "overlaps": list(overlaps)}
    false_reasons = []
    if not issue_keys:
        false_reasons.append("uncredited")
    if len(issue_keys) > 1:
        false_reasons.append("montage")
    if title_seen:
        false_reasons.append("title_seen")
    return {"decision": "sealed", "refusal_reasons": [], "headline": not false_reasons,
            "headline_false_reasons": false_reasons, "title_seen": title_seen,
            "credited": bool(issue_keys), "montage": len(issue_keys) > 1, "overlaps": list(overlaps)}


def headline_pages(sealed_pages: Sequence[Mapping[str, Any]]) -> list[str]:
    """The ids of the pages on which the measure that counts may be computed."""
    return [p["page_id"] for p in sealed_pages if p["headline"] is True]


def check_headline_measure(declared_pages: Sequence[str], sealed_pages: Sequence[Mapping[str, Any]]) -> None:
    """A measure presented as the one that counts must contain only headline=true pages of the seal."""
    allowed = set(headline_pages(sealed_pages))
    bad = [p for p in declared_pages if p not in allowed]
    if bad:
        raise SealStop(f"headline measure contains pages that are not headline=true: {bad[:5]}")
