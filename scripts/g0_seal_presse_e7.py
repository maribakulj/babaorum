"""Seal presse_e7 test v1 (G0 v2, step B): by copy and fingerprints, never in place.

Writes only config/sealed/presse_e7_test_v1.json and reports/test_ledger.jsonl in this repository.
Axel and corpus-vt are read-only. config/g0_release.json is never created.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/g0_seal_presse_e7.py

Exact stops (nothing is written if one differs): 454 test pages / 410 with Wikisource truth lines /
44 without ; 322 credited test pages ; 8 montage pages. Every collection's matching key at each level
is recorded in the seal (or "aucune clé à ce niveau" with its reason): headline cannot freeze without it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from babaorum.g0_collections import NO_KEY  # noqa: E402
from babaorum.g0_seal import SealStop, decide, expect_counts, find_overlaps  # noqa: E402
import g0_keys_build as K  # noqa: E402

AX, CV = K.AX, K.CV
E7 = K.E7
OUT_SEAL = ROOT / "config/sealed/presse_e7_test_v1.json"
OUT_LEDGER = ROOT / "reports/test_ledger.jsonl"
EXPECTED = {"test_pages": 454, "test_pages_with_ws_lines": 410, "test_pages_without_lines": 44,
            "credited_test_pages": 322, "montage_pages": 8}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def truth_by_page() -> tuple[dict, Counter]:
    """Per Wikisource page: line count and sha256 of the ordered truth lines. Other corpora are only counted."""
    pages: dict[str, dict] = {}
    other: Counter = Counter()
    for line in (E7 / "test.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["corpus"] != "wikisource":
            other[(r["corpus"], r["page"])] += 1
            continue
        p = pages.setdefault(r["page"], {"lines": 0, "h": hashlib.sha256()})
        p["lines"] += 1
        p["h"].update(f"{r['line_id']}\t{r['truth']}\n".encode("utf-8"))
    return {k: {"lines": v["lines"], "truth_sha256": v["h"].hexdigest()} for k, v in pages.items()}, other


def collections_view(keys: dict) -> dict:
    return {name: {lv: (set(spec[lv]["values"]) if spec[lv] else None) for lv in K.LEVELS} for name, (spec, _) in keys.items()}


def build_ledger() -> list[dict]:
    rows = []
    runs = AX / "runs"
    tokens = [("presse_e7", "presse_e7_test"), ("bnl", "bnl_pack"), ("gallicorpora", "gallicorpora"), ("ocr17", "gallicorpora"),
              ("icdar", "icdar2017_2019"), ("newseye", "newseye"), ("impresso", "impresso"), ("hipe", "hipe_ocrepair")]
    for dirpath, _, files in os.walk(runs):
        for name in sorted(files):
            if ".test.jsonl" not in name:
                continue
            p = Path(dirpath, name)
            coll = sorted({c for t, c in tokens if t in name.lower()}) or ["non_determinee"]
            rows.append({"kind": "read_output_hashed", "path": str(p.relative_to(AX)), "sha256": sha(p), "bytes": p.stat().st_size,
                         "experiment": p.relative_to(runs).parts[0], "collections": coll, "role": "test"})
    seen = {r["path"] for r in rows}
    for dirpath, _, files in os.walk(runs):
        for name in sorted(files):
            p = Path(dirpath, name)
            if "presse_e7" in name and str(p.relative_to(AX)) not in seen:
                rows.append({"kind": "presse_e7_artifact_hashed", "path": str(p.relative_to(AX)), "sha256": sha(p), "bytes": p.stat().st_size,
                             "experiment": p.relative_to(runs).parts[0], "collections": ["presse_e7_test"],
                             "role": "test (échantillon ou journal de lecture)"})
    rows.sort(key=lambda r: r["path"])
    for doc in sorted((AX / "docs").glob("*.md")):
        hits = [i + 1 for i, line in enumerate(doc.read_text(encoding="utf-8", errors="replace").splitlines())
                if "presse_e7" in line and re.search(r"\btest\b", line, re.I)]
        if hits:
            rows.append({"kind": "doc_mentions_presse_e7_test", "path": str(doc.relative_to(AX)), "sha256": sha(doc),
                         "lines": hits[:12], "mention_count": len(hits)})
    return rows


def main() -> int:
    pages = K.e7_pages()
    test_pages = {pid: p for pid, p in pages.items() if p["split"] == "test"}
    truth, other = truth_by_page()
    credited_test = [pid for pid, p in test_pages.items() if p["issue_keys"]]
    montage_all = [pid for pid, p in pages.items() if len(p["issue_keys"]) > 1]
    with_lines = [pid for pid in test_pages if pid in truth]
    counts = {"test_pages": len(test_pages), "test_pages_with_ws_lines": len(with_lines),
              "test_pages_without_lines": len(test_pages) - len(with_lines),
              "credited_test_pages": len(credited_test), "montage_pages": len(montage_all)}
    expect_counts(counts, EXPECTED)
    if {k for k in truth} - set(test_pages):
        raise SealStop("truth lines exist for Wikisource pages outside the test split")
    non_ws = Counter(c for c, _ in other)
    if non_ws != {"impact_kb": 7, "bnf_ocr_corrige": 4}:
        raise SealStop(f"non-Wikisource test pages differ from 7 impact_kb + 4 bnf_ocr_corrige: {dict(non_ws)}")

    keys = K.compute_keys()
    others = collections_view(keys)
    sealed, refused = [], []
    for pid, p in sorted(test_pages.items()):
        overlaps = find_overlaps({"page": [p["scan_page_key"]], "issue": p["issue_keys"], "title": [p["title_key"]]}, others)
        info = truth.get(pid, {"lines": 0, "truth_sha256": None})
        decision = decide(truth_lines=info["lines"], issue_keys=p["issue_keys"], overlaps=overlaps)
        base = {"page_id": pid, "index": p["index"], "title": p["title"], "title_key": p["title_key"], "date": p["date"],
                "scan_page_key": p["scan_page_key"], "issue_keys": p["issue_keys"], "truth_lines": info["lines"],
                "montage": len(p["issue_keys"]) > 1}
        if decision["decision"] == "refused":
            refused.append({**base, "refusal_reasons": decision["refusal_reasons"], "overlaps": decision["overlaps"]})
            continue
        image = Path(p["image"])
        if not image.is_file():
            raise SealStop(f"image missing for a sealed page: {image}")
        sealed.append({**base, "image": pin(image), "truth_sha256": info["truth_sha256"], "headline": decision["headline"],
                       "headline_false_reasons": decision["headline_false_reasons"], "title_seen": decision["title_seen"],
                       "credited": decision["credited"], "title_overlaps": decision["overlaps"]})
    if len(sealed) + len(refused) != EXPECTED["test_pages"]:
        raise SealStop("sealed + refused != 454")

    reasons = Counter(r for p in refused for r in p["refusal_reasons"])
    mercure = [p for p in [*sealed, *refused] if p["title_key"].startswith("mercure")]
    mercure_titles = Counter(p["index"].split(",")[0].replace("Livre:", "") for p in mercure)
    declared = {}
    for name, (spec, why) in keys.items():
        declared[name] = {lv: (NO_KEY if spec[lv] is None else {"name": spec[lv]["name"], "keyed": spec[lv]["coverage"]["keyed"],
                                                           "total": spec[lv]["coverage"]["total"], "distinct_keys": len(spec[lv]["values"])})
                          for lv in K.LEVELS}
        declared[name]["reasons_when_no_key"] = {lv: why[lv] for lv in K.LEVELS if spec[lv] is None}
    unverifiable = {name: [lv for lv in ("page", "issue") if d[lv] == NO_KEY or d[lv]["keyed"] < d[lv]["total"]]
                    for name, d in declared.items()}
    unverifiable = {n: lv for n, lv in unverifiable.items() if lv}
    seal = {
        "schema_version": "babaorum.sealed_test.v1",
        "name": "presse_e7_test_v1",
        "label": "test réutilisé (déjà mesuré par E1, E2, E3, E8, E11 ; voir reports/test_ledger.jsonl), texte Wikisource sans géométrie",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pending_independent_audit",
        "release_file": "config/g0_release.json n'existe pas et n'est pas créé ; ce scellé n'est pas une libération",
        "method": "par copie et empreintes : les sources ne sont ni modifiées ni déplacées",
        "sources": {"pages_jsonl": pin(E7 / "pages.jsonl"), "test_jsonl": pin(E7 / "test.jsonl"),
                    "wikisource_crosswalk": pin(ROOT / "reports/g0_wikisource_crosswalk_investigation.json")},
        "exact_stops": {"expected": EXPECTED, "found": counts, "all_equal": True},
        "non_wikisource_test_pages_not_added": {"impact_kb": 7, "bnf_ocr_corrige": 4,
                                                "why": "elles appartiennent à des collections exposées (kb_impact, bnf_ocr_corrige_presse)"},
        "totals": {"sealed": len(sealed), "refused": len(refused), "headline_true": sum(p["headline"] for p in sealed),
                   "headline_false": sum(not p["headline"] for p in sealed),
                   "headline_false_reasons": dict(Counter(r for p in sealed for r in p["headline_false_reasons"])),
                   "refusal_reasons": dict(reasons)},
        "headline_rule": "headline=true: document Gallica unique crédité, issue_key hors de toute autre collection, titre non vu ailleurs, aucune correspondance de page",
        "headline_limits": {"collections_with_no_or_partial_page_or_issue_keys": unverifiable,
                            "meaning": "pour ces collections le recouvrement ne peut pas être exclu au niveau indiqué (clé absente ou partielle)"},
        "match_keys_declared": declared,
        "montages": [p["page_id"] for p in [*sealed, *refused] if p["montage"]],
        "mercure": {"pages": [{"page_id": p["page_id"], "decision": "sealed" if "image" in p else "refused", "issue_keys": p["issue_keys"],
                               "title_key": p["title_key"], "index": p["index"]} for p in mercure],
                    "documents": dict(mercure_titles),
                    "issue_keys": sorted({k for p in mercure for k in p["issue_keys"]}),
                    "note": "Mercure galant et Mercure de France : titre ambigu, pages scellées avec title_seen=true et headline=false"},
        "pages": sealed,
        "refused": refused,
    }
    OUT_SEAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SEAL.write_text(json.dumps(seal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ledger = build_ledger()
    header = {"kind": "header", "generated_utc": seal["generated_utc"], "scope": "usages antérieurs de presse_e7 test et de tous les *.test.jsonl* de axel/runs",
              "find_equivalent": "os.walk(~/axel/runs) pour les fichiers dont le nom contient .test.jsonl", "files_hashed": sum(r["kind"] == "read_output_hashed" for r in ledger),
              "presse_e7_test_read_outputs": sum(r["kind"] in ("read_output_hashed", "presse_e7_artifact_hashed") and "presse_e7_test" in r["collections"] for r in ledger),
              "experiments_reading_presse_e7_test": sorted({r["experiment"] for r in ledger if r["kind"] in ("read_output_hashed", "presse_e7_artifact_hashed") and "presse_e7_test" in r["collections"]}),
              "docs_mentioning_presse_e7_test": sorted(r["path"] for r in ledger if r["kind"] == "doc_mentions_presse_e7_test"),
              "seal_sha256": sha(OUT_SEAL)}
    OUT_LEDGER.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [header, *ledger]) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, **seal["totals"], "ledger_files": header["files_hashed"],
                      "experiments": header["experiments_reading_presse_e7_test"], "montages": len(seal["montages"]),
                      "mercure_pages": len(mercure), "seal_sha256": header["seal_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
