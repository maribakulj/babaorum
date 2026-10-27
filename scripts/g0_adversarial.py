"""G0-03 step A: the ten adversarial cases of audit_codex/adversarial.py, on the anchored registry v2.1 and on the complete fixture.

    cd /tmp && /usr/bin/python3 -I ~/babaorum/scripts/g0_adversarial.py <dossier_date_neuf>
Writes ONLY into the folder given as argument (created, must not exist). Nothing is written under config/, audit_codex/ or the repository.
Each case states the codes it must produce (the G0-03 v2 table, columns v2.1 real and complete registry); any difference is reported, not fixed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/fixtures")]
from babaorum.g0_collections import CollectionRegistryError, eligibility, verify_collections  # noqa: E402
from g0_registry_complete import complete_registry  # noqa: E402

I9 = ["INCOMPLETE_NUMBER_KEY:as_anno_183", "INCOMPLETE_NUMBER_KEY:newseye", "INCOMPLETE_NUMBER_KEY:gallicorpora",
      "INCOMPLETE_NUMBER_KEY:hipe_ocrepair", "INCOMPLETE_NUMBER_KEY:presse_e7_train_dev", "INCOMPLETE_NUMBER_KEY:icdar2017_2019",
      "INCOMPLETE_NUMBER_KEY:e12_test", "INCOMPLETE_NUMBER_KEY:bnf_ocr_corrige_presse", "INCOMPLETE_NUMBER_KEY:gallica_pseudo"]
I10 = I9[:4] + ["INCOMPLETE_NUMBER_KEY:pleias_french_pd_322"] + I9[4:]   # pleias inserted between hipe_ocrepair and presse_e7_train_dev


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: g0_adversarial.py <dossier_date_neuf>")
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists: give a new dated folder")
    out.mkdir(parents=True)
    seal = json.loads((ROOT / "config/sealed/presse_e7_test_v2_1.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "config/g0_collections_v2_2.json").read_text(encoding="utf-8"))
    complete = complete_registry()
    p = next(x for x in seal["pages"] if x["headline"])
    args = {"page_key": p["scan_page_key"], "issue_key": p["issue_keys"][0], "title_key": p["title_key"]}
    uncredited = next(x for x in seal["pages"] if not x["issue_keys"])

    def case_registry(base, mutate):
        r = copy.deepcopy(base)
        mutate(r)
        return r

    def m_strictest(r):
        r["collections"]["pleias_french_pd_322"]["match_keys"]["issue"] = {"name": "issue", "values": [args["issue_key"]]}

    def m_bnl(r):
        r["collections"]["presse_e7_test"]["match_keys"]["page"] = {"name": "page", "values": ["audit:bnl_page"]}

    def m_partial(r):
        r["collections"]["presse_e7_test"]["match_keys"]["page"] = {"name": "page", "values": ["audit:exposed_partial"]}
        r["collections"]["pleias_french_pd_322"]["match_keys"]["issue"] = {"name": "issue", "values": ["gallica:12345"], "coverage": {"keyed": 1, "total": 2}}

    def m_sibling(r):
        r["collections"]["presse_e7_test_refused"]["match_keys"]["page"] = {"name": "page", "values": [p["scan_page_key"]]}

    def m_nzz(r):
        r["collections"]["presse_e7_test"]["match_keys"]["page"] = {"name": "page", "values": ["audit:nzz"]}

    def m_fabricated(r):
        spec = r["collections"]["presse_e7_test"]["match_keys"]["page"]
        real = json.loads(Path(spec["values_file"]["path"]).read_text(encoding="utf-8"))
        spec.pop("values_file")
        spec["values"] = real + ["audit:not_in_pinned_manifest"]
        spec["coverage"] = {"keyed": len(real) + 1, "total": len(real) + 1}

    cases = [
        # name, registry mutation, role args, expected codes on v2.1 (real), expected codes on the complete fixture, expected allowed (complete)
        ("real_headline_baseline_pending_audit", None, dict(args), I9, [], True),
        ("no_identity", None, {}, "ERROR:NO_IDENTITY", "ERROR:NO_IDENTITY", False),
        ("unsealed_page", None, dict(page_key="audit:outside", issue_key="gallica:987654321", title_key="audit titre neuf"),
         ["NOT_SEALED"] + I9, ["NOT_SEALED"], False),
        ("strictest_known_exposed_issue_wins", m_strictest, dict(args),
         ["EXPOSED_MATCH:pleias_french_pd_322:issue"] + I10, ["EXPOSED_MATCH:pleias_french_pd_322:issue", "INCOMPLETE_NUMBER_KEY:pleias_french_pd_322"], False),
        ("sealed_but_exposed_collection_has_no_keys", m_bnl, dict(page_key="audit:bnl_page", issue_key="gallica:987654321", title_key="audit titre neuf"),
         ["SEAL_NOT_PINNED:presse_e7_test"] + I9, ["SEAL_NOT_PINNED:presse_e7_test"], False),
        ("sealed_but_exposure_coverage_partial", m_partial, dict(page_key="audit:exposed_partial", issue_key="gallica:987654321", title_key="audit titre neuf"),
         ["SEAL_NOT_PINNED:presse_e7_test"] + I10, ["SEAL_NOT_PINNED:presse_e7_test", "INCOMPLETE_NUMBER_KEY:pleias_french_pd_322"], False),
        ("sealed_and_reserved_sibling_same_page", m_sibling, dict(args),
         ["SIBLING_PAGE_CONFLICT:presse_e7_test_refused"] + I9, ["SIBLING_PAGE_CONFLICT:presse_e7_test_refused"], False),
        ("known_title_only_locked_collection_nzz", m_nzz, dict(page_key="audit:nzz", issue_key="gallica:987654321", title_key="neue zurcher zeitung"),
         ["SEAL_NOT_PINNED:presse_e7_test", "COMPLETE_TITLE_MATCH:nzz_impresso"] + I9, ["SEAL_NOT_PINNED:presse_e7_test", "COMPLETE_TITLE_MATCH:nzz_impresso"], False),
        ("real_page_without_source_document", None, dict(page_key=uncredited["scan_page_key"], title_key=uncredited["title_key"]),
         ["NO_SOURCE_DOCUMENT"] + I9, ["NO_SOURCE_DOCUMENT"], False),
        ("validated_registry_page_absent_from_pinned_manifest", m_fabricated,
         dict(page_key="audit:not_in_pinned_manifest", issue_key="gallica:987654321", title_key="audit titre neuf"),
         ["SEAL_NOT_PINNED:presse_e7_test"] + I9, ["SEAL_NOT_PINNED:presse_e7_test"], False),
    ]
    rows, mismatches = [], []
    for name, mutate, kw, exp_real, exp_complete, exp_allowed in cases:
        for column, base, expected in (("v2.1", registry, exp_real), ("complete", complete, exp_complete)):
            reg = case_registry(base, mutate) if mutate else copy.deepcopy(base)
            if isinstance(expected, str) and expected.startswith("ERROR:"):
                try:
                    eligibility(reg, "test", **kw)
                    got = "NO_ERROR"
                except CollectionRegistryError as exc:
                    got = f"ERROR:{exc.code}"
                ok = got == expected
                row = {"case": name, "column": column, "codes": got, "expected": expected, "match": ok, "allowed": None}
            else:
                res = eligibility(reg, "test", **kw)
                got = res["refusal_codes"]
                ok = got == expected
                row = {"case": name, "column": column, "codes": got, "expected": expected, "match": ok, "allowed": res["allowed"],
                       "refusals": res["refusals"]}
            if column == "complete":
                row["allowed_expected"] = exp_allowed
                row["allowed_match"] = (row["allowed"] == exp_allowed) if row["allowed"] is not None else None
            rows.append(row)
            if not ok:
                mismatches.append(row)
    # real headline on the v2.1 registry: admitted in the complete column only (the table)
    # verify_collections on the fabricated registry (the code KEYS_DIFFER_FROM_PINNED)
    fab = case_registry(registry, m_fabricated)
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "adversarial_registry.json"
        fp.write_text(json.dumps(fab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            verify_collections(fp)
            vcode = "NO_ERROR"
        except CollectionRegistryError as exc:
            vcode = exc.code
    rows.append({"case": "validated_registry_page_absent_from_pinned_manifest", "column": "verify_collections", "codes": vcode,
                 "expected": "KEYS_DIFFER_FROM_PINNED", "match": vcode == "KEYS_DIFFER_FROM_PINNED"})
    if vcode != "KEYS_DIFFER_FROM_PINNED":
        mismatches.append(rows[-1])
    summary = {"folder": str(out), "rows": rows, "mismatches": mismatches, "mismatch_count": len(mismatches)}
    (out / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"{r['case']:52s} {r['column']:19s} {'OK ' if r['match'] else 'ECART'} codes={r['codes'] if isinstance(r['codes'], str) else len(r['codes'])}"
             for r in rows]
    (out / "table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for label, path in (("registry v2.2", ROOT / "config/g0_collections_v2_2.json"),):
        (out / f"sha256_{label.replace(' ', '_').replace('.', '_')}.txt").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    print("\n".join(lines))
    print(f"ECARTS: {len(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
