"""Recount of the 410 sealed pages under the LOT-G0-02 rules (read-only; writes reports/lot02_recount.json).

Two counts: with the REAL registry v2 (the one that counts), and, for information only, with the number keys of the
gallica_scope=possible collections pretended complete in memory (what G0-03 would have to deliver; nothing is admitted from it).
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from babaorum.g0_collections import NO_KEY, eligibility  # noqa: E402

REG = ROOT / "config/g0_collections_v2.json"
SEAL = ROOT / "config/sealed/presse_e7_test_v2.json"


def classify(registry, seal):
    head = set(registry["collections"]["presse_e7_test"]["headline_page_keys"]["values"])
    out = {"test": 0, "test_descriptive": 0, "refused": 0, "headline_test": 0, "sibling_exception_admitted": 0}
    reasons = collections.Counter()
    for p in seal["pages"]:
        issue = p["issue_keys"][0] if p["issue_keys"] else None
        q = dict(page_key=p["scan_page_key"], issue_key=issue, title_key=p["title_key"])
        role = "test" if p["scan_page_key"] in head or issue else "test_descriptive"
        r = eligibility(registry, role, **q)
        if r["allowed"]:
            out[role] += 1
            out["headline_test"] += role == "test" and p["scan_page_key"] in head
            out["sibling_exception_admitted"] += bool(r["sibling_exceptions"])
        else:
            out["refused"] += 1
            for text in set(x.split(" (")[0] if "complete number key" in x or "no usable key" in x else x for x in r["refusals"]):
                reasons[text] += 1
    return out, dict(reasons.most_common())


def main() -> int:
    registry = json.loads(REG.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    real, real_reasons = classify(registry, seal)
    pretend = copy.deepcopy(registry)
    for entry in pretend["collections"].values():
        if entry["gallica_scope"]["value"] == "possible":
            spec = entry["match_keys"]["issue"]
            if spec == NO_KEY:
                spec = entry["match_keys"]["issue"] = {"name": "issue", "values": []}
            spec["coverage"] = {"keyed": 1, "total": 1}
    hypo, hypo_reasons = classify(pretend, seal)
    report = {
        "label": "recomptage des 410 pages scellées sous les règles de LOT-G0-02",
        "registry": {"path": str(REG), "sha256": hashlib.sha256(REG.read_bytes()).hexdigest()},
        "seal": {"path": str(SEAL), "sha256": hashlib.sha256(SEAL.read_bytes()).hexdigest()},
        "pages": len(seal["pages"]), "headline_before": sum(p["headline"] for p in seal["pages"]),
        "real_registry": {"counts": real, "refusal_motifs": real_reasons},
        "if_number_keys_were_complete_information_only": {"counts": hypo, "refusal_motifs": hypo_reasons,
            "note": "rôle test = page headline ou à document source ; rôle test_descriptive = sans document source et non headline ; rien n'est admis sur cette base"},
        "sibling_exception_pages_labelled": sum(p["sibling_exception"] for p in seal["pages"]),
    }
    (ROOT / "reports/lot02_recount.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"real": real, "hypothetical": hypo, "motifs_reels": real_reasons, "motifs_hypo": hypo_reasons}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
