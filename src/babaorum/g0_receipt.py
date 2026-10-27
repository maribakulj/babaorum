"""Verify a G0 metadata inventory receipt against its actual input sources.

This is a consistency gate, not an independent split-release mechanism. A real
release still requires a separate independent audit and pinned authorization.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


class ReceiptError(ValueError):
    pass


_SHA = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_EMPTY_LISTS = (
    "protected_page_or_issue", "protected_title", "contacts", "identity_gaps",
    "press_same_issue", "exposure",
)
_EXPOSURE_CATALOGUE = {
    "C7", "E12", "E22", "E7", "NewsEye_validation", "AS", "E6_E8_press",
    "E14", "E15", "E24", "C2", "E1_BnL", "R2_LM_PleIAs_French_PD",
}
_ROLE_PATH_COUNTS = {
    "C7": 2, "E12": 1, "E22": 1, "E7": 6, "NewsEye_validation": 1,
    "AS": 1, "E6_E8_press": 2, "R2_LM_PleIAs_French_PD": 1,
}
_EXPECTED_VALIDATION_DIR = Path("/Users/marcel/corpus-vt/newseye/val/ATR_ValidationSet_BnF_Newseye_M2+")
_EXPECTED_VALIDATION_NAMES = [
    "0253902-001.xml", "0401692-003.xml", "752234-003.xml",
    "Ce-soir_7635415_D0000002.xml", "Marianne_7644869_D0000002.xml",
    "Paris-Soir_7636525_D0000002.xml", "Regards_7635838_D0000005.xml",
    "Regards_7639182_D0000004.xml",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_unique(raw: bytes) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ReceiptError(f"duplicate JSON key in G0 source: {key}")
            value[key] = item
        return value

    def constant(value):
        raise ReceiptError(f"non-finite JSON constant in G0 source: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _read_pinned_json(path: str | Path, inputs: dict[str, str]) -> Any:
    name = str(path)
    _require(name in inputs, f"G0 metadata JSON is not pinned in inputs: {name}")
    raw = Path(name).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == inputs[name], f"G0 metadata JSON changed before parsing: {name}")
    try:
        return _json_unique(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"G0 metadata JSON is invalid: {name}") from exc


def _strict_file_map(receipt: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any], float]:
    inputs = receipt.get("inputs")
    _require(isinstance(inputs, dict) and bool(inputs), "G0 input hash map missing or empty")
    latest_source_mtime = 0.0
    for name, expected in inputs.items():
        _require(isinstance(name, str) and Path(name).is_absolute(), "G0 input path must be absolute")
        _require(isinstance(expected, str) and _SHA.fullmatch(expected) is not None,
                 f"invalid input digest: {name}")
        path = Path(name)
        _require(path.is_file(), f"G0 input file missing: {name}")
        _require(_sha(path) == expected, f"G0 input hash changed: {name}")
        latest_source_mtime = max(latest_source_mtime, path.stat().st_mtime)

    dirs = receipt.get("directory_inputs")
    _require(isinstance(dirs, dict) and bool(dirs), "G0 directory input map missing or empty")
    normalized_dirs: dict[str, Any] = {}
    for name, desc in dirs.items():
        _require(isinstance(name, str) and Path(name).is_absolute(), "G0 directory path must be absolute")
        _require(isinstance(desc, dict) and set(desc) == {"pattern", "names"},
                 f"invalid directory receipt: {name}")
        pattern, names = desc["pattern"], desc["names"]
        _require(isinstance(pattern, str) and bool(pattern.strip()), f"empty directory glob: {name}")
        _require(isinstance(names, list) and bool(names) and
                 all(isinstance(x, str) and bool(x) and Path(x).name == x for x in names),
                 f"invalid/empty directory listing: {name}")
        _require(names == sorted(set(names)), f"directory listing must be sorted and unique: {name}")
        directory = Path(name)
        _require(directory.is_dir(), f"G0 input directory missing: {name}")
        current = sorted(p.name for p in directory.glob(pattern))
        _require(current == names, f"G0 directory listing changed: {name}")
        for filename in names:
            item = directory / filename
            _require(item.is_file(), f"G0 directory input is not a file: {item}")
            latest_source_mtime = max(latest_source_mtime, item.stat().st_mtime)
        normalized_dirs[name] = desc
    return inputs, normalized_dirs, latest_source_mtime


def _catalogue_fields(receipt: dict[str, Any]) -> None:
    missing = receipt.get("missing_exposure_sources")
    _require(isinstance(missing, list) and all(isinstance(x, str) for x in missing)
             and missing == sorted(set(missing)), "G0 missing exposure-source list invalid")
    sources = receipt.get("exposure_sources")
    _require(isinstance(sources, dict) and set(sources) == _EXPOSURE_CATALOGUE,
             "G0 exposure-source catalogue missing or incomplete")
    absent = []
    for key, desc in sources.items():
        _require(isinstance(desc, dict) and set(desc) == {"status"},
                 f"invalid G0 exposure source descriptor: {key}")
        _require(desc["status"] in {"metadata_loaded", "missing_metadata_manifest"},
                 f"invalid G0 exposure-source status: {key}")
        if desc["status"] == "missing_metadata_manifest":
            absent.append(key)
    _require(missing == sorted(absent), "G0 missing exposure-source list disagrees with source catalogue")
    source_inputs = receipt.get("source_inputs")
    _require(isinstance(source_inputs, dict) and set(source_inputs) == _EXPOSURE_CATALOGUE,
             "G0 source-role map missing or incomplete")
    for key, paths in source_inputs.items():
        _require(isinstance(paths, list), f"G0 source-role file list invalid: {key}")
        _require(all(isinstance(p, str) and Path(p).is_absolute() for p in paths),
                 f"G0 source-role paths must be absolute: {key}")
        if sources[key]["status"] == "missing_metadata_manifest":
            _require(not paths, f"missing exposure source unexpectedly has files: {key}")
        else:
            _require(bool(paths), f"loaded exposure source has no files: {key}")
            expected_count = _ROLE_PATH_COUNTS.get(key)
            if expected_count is not None:
                _require(len(paths) == expected_count, f"G0 source-role file list incomplete: {key}")


def _release_fields(receipt: dict[str, Any]) -> None:
    _require(receipt.get("metadata_current") is True, "G0 metadata receipt is not current")
    _require(receipt.get("split_release_status") == "released_after_independent_audit",
             "G0 split release is not explicitly authorized after independent audit")
    _require(receipt.get("exposure_identity_complete") is True, "G0 unresolved exposure identities block release")
    unresolved = receipt.get("unresolved_exposures")
    _require(isinstance(unresolved, list) and unresolved == [], "G0 unresolved exposures block release")
    _require(receipt.get("missing_exposure_sources") == [], "G0 exposure-source catalogue is incomplete")


def _coherent_inventory_state(receipt: dict[str, Any]) -> None:
    unresolved = receipt.get("unresolved_exposures")
    _require(isinstance(unresolved, list), "G0 unresolved exposure list missing")
    missing = receipt["missing_exposure_sources"]
    complete = not unresolved and not missing
    _require(type(receipt.get("exposure_identity_complete")) is bool
             and receipt["exposure_identity_complete"] is complete,
             "G0 exposure completeness flag disagrees with evidence")
    status = receipt.get("split_release_status")
    if unresolved:
        expected = "blocked_unresolved_exposures"
        _require(status == expected, "G0 split status disagrees with unresolved exposures")
    elif missing:
        _require(status == "blocked_missing_exposure_sources",
                 "G0 split status disagrees with missing exposure sources")
    else:
        _require(status in {"pending_independent_audit", "released_after_independent_audit"},
                 "G0 split status is inconsistent with complete inventory")


def _source_inventory(receipt: dict[str, Any], inputs: dict[str, str]) -> list[dict[str, Any]]:
    desc = receipt.get("inventory_source")
    _require(isinstance(desc, dict) and set(desc) == {"path", "sha256"},
             "G0 inventory source descriptor missing")
    path_text, expected = desc["path"], desc["sha256"]
    _require(isinstance(path_text, str) and Path(path_text).is_absolute(),
             "G0 inventory source path must be absolute")
    _require(isinstance(expected, str) and _SHA.fullmatch(expected) is not None,
             "G0 inventory source hash invalid")
    _require(inputs.get(path_text) == expected, "G0 inventory source must be pinned in inputs")
    raw = Path(path_text).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected, "G0 inventory source hash mismatch")
    source = _read_pinned_json(path_text, inputs)
    _require(isinstance(source, dict) and isinstance(source.get("pages"), list),
             "G0 source inventory pages missing")
    rows = source["pages"]
    for row in rows:
        _require(isinstance(row, dict), "invalid G0 source page")
        _require(isinstance(row.get("source"), str) and bool(row["source"].strip()),
                 "G0 source page source missing")
        _require(isinstance(row.get("page"), str) and bool(row["page"].strip()),
                 "G0 source page name missing")
        _require(row.get("split") in {"train", "dev", "test"}, "invalid source split")
        _require(row.get("excluded") is None or row.get("excluded") is False or isinstance(row.get("excluded"), str),
                 "invalid source exclusion")
    return rows


def _validate_source_roles(receipt: dict[str, Any], inputs: dict[str, str], dirs: dict[str, Any],
                           *, expected_axel_root: str | Path | None = None) -> None:
    seg_path = Path(receipt["inventory_source"]["path"])
    _require(seg_path.parts[-4:] == ("data", "prepared", "seg", "manifest.json"),
             "inventory source is not the canonical segmentation manifest")
    axel_root = seg_path.parents[3]
    if expected_axel_root is not None:
        _require(axel_root == Path(expected_axel_root), "G0 source tree is outside expected Axel root")
    inventory_script = (str(Path(__file__).resolve().parents[2] / "scripts/g0_inventory.py")
                        if expected_axel_root is not None else
                        next((path for path in inputs if path.endswith("/scripts/g0_inventory.py")), ""))
    babaorum_root = Path(inventory_script).parent.parent
    if expected_axel_root is not None:
        _require(inventory_script in inputs, "G0 receipt does not pin this repository's inventory implementation")
        for relative in ("scripts/g0_inventory.py", "src/babaorum/identity.py",
                         "src/babaorum/legacy/data/doc_key.py", "src/babaorum/contracts.py"):
            path = babaorum_root / relative
            _require(str(path) in inputs and inputs[str(path)] == _sha(path),
                     f"G0 receipt does not pin the current implementation: {relative}")
    source_inputs = receipt["source_inputs"]

    expected = {
        "C7": [str(seg_path), str(axel_root / "runs/c7/audit.json")],
        "E12": [str(axel_root / "runs/e12/pages.json")],
        "E22": [str(axel_root / "runs/e22/bench/runs/e12/pages.json")],
        "E7": [str(axel_root / "data/prepared/presse_test_manifest.json"),
               str(axel_root / "data/prepared/presse_e7/pages_bnf.jsonl"),
               str(axel_root / "data/prepared/presse_e7/pages_impact.jsonl"),
               str(axel_root / "data/prepared/presse_e7/pages.jsonl"),
               str(babaorum_root / "config/g0_metadata_supplements.json")],
        "AS": [str(axel_root / "data/prepared/olr/newseye_as.json")],
        "E6_E8_press": [str(axel_root / "data/prepared/press/train.jsonl"),
                        str(axel_root / "data/prepared/press/dev.jsonl")],
    }
    supplement_path = Path(expected["E7"][-1])
    _require(str(supplement_path) in inputs, "G0 metadata supplement configuration not pinned")
    config = _read_pinned_json(supplement_path, inputs)
    _require(isinstance(config, dict) and config.get("schema_version") == "babaorum.g0_metadata_supplements.v2",
             "unsupported G0 metadata supplement config")
    descriptor = config.get("wikisource_crosswalk")
    _require(isinstance(descriptor, dict) and set(descriptor) ==
             {"path", "pages_metadata_path", "sha256", "pages_metadata_sha256"},
             "Wikisource crosswalk descriptor invalid")
    pages_path = str(axel_root / "data/prepared/presse_e7/pages.jsonl")
    _require(descriptor["pages_metadata_path"] == pages_path,
             "Wikisource crosswalk does not bind the pinned E7 pages metadata")
    crosswalk_path = descriptor["path"]
    _require(isinstance(crosswalk_path, str) and Path(crosswalk_path).is_absolute(),
             "Wikisource crosswalk path must be absolute")
    expected["E7"].append(crosswalk_path)
    _require(descriptor["sha256"] == inputs.get(crosswalk_path),
             "Wikisource crosswalk hash does not match pinned input")
    _require(descriptor["pages_metadata_sha256"] == inputs.get(pages_path),
             "Wikisource crosswalk page-table hash does not match pinned input")
    crosswalk = _read_pinned_json(crosswalk_path, inputs)
    _require(isinstance(crosswalk, dict) and isinstance(crosswalk.get("local_inputs"), dict)
             and crosswalk["local_inputs"].get(pages_path) == inputs[pages_path],
             "Wikisource crosswalk metadata does not bind the pinned E7 page table")
    lm_descriptor = config.get("french_pd_lm_manifest")
    _require(isinstance(lm_descriptor, dict) and set(lm_descriptor) == {"path", "sha256"},
             "R2 French-PD LM manifest descriptor invalid")
    lm_path = lm_descriptor["path"]
    _require(isinstance(lm_path, str) and Path(lm_path).is_absolute(),
             "R2 French-PD LM manifest path must be absolute")
    if expected_axel_root is not None:
        _require(lm_path == "/Users/marcel/corpus-vt/french-pd-sample/manifest.json",
                 "R2 French-PD LM manifest path is not the pinned local manifest")
    _require(isinstance(lm_descriptor["sha256"], str) and
             _SHA.fullmatch(lm_descriptor["sha256"]) is not None and
             lm_descriptor["sha256"] == inputs.get(lm_path),
             "R2 French-PD LM manifest SHA-256 does not match pinned input")
    expected["R2_LM_PleIAs_French_PD"] = [lm_path]
    _require(len(dirs) == 1, "G0 validation directory source missing or ambiguous")
    validation_dir = next(iter(dirs))
    if expected_axel_root is not None:
        _require(Path(validation_dir) == _EXPECTED_VALIDATION_DIR,
                 "G0 validation directory is not the configured NewsEye metadata directory")
        _require(dirs[validation_dir]["pattern"] == "*.xml"
                 and dirs[validation_dir]["names"] == _EXPECTED_VALIDATION_NAMES,
                 "G0 validation directory listing/pattern differs from the pinned NewsEye inventory")
    else:
        _require(Path(validation_dir).name == _EXPECTED_VALIDATION_DIR.name,
                 "G0 validation directory is not the configured NewsEye metadata directory")
    expected["NewsEye_validation"] = [validation_dir]
    for name in ("E14", "E15", "E24", "C2", "E1_BnL"):
        expected[name] = []
    _require(source_inputs == expected, "G0 source-role map does not match canonical source paths")
    for role, paths in expected.items():
        status = receipt["exposure_sources"][role]["status"]
        _require((status == "metadata_loaded") == bool(paths),
                 f"G0 source role/status mismatch: {role}")
        for path in paths:
            _require(path in inputs or path in dirs,
                     f"G0 role source is not covered by the actual file hashes/listing: {role}")


def _check_complete_pages(receipt: dict[str, Any], source_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pages = receipt.get("pages")
    _require(isinstance(pages, list) and bool(pages), "G0 inventory pages missing or empty")
    # A multiset comparison detects truncation, duplication, substitution, and
    # split/exclusion edits even when an attacker also changes the reported count.
    source_multiset = Counter((r["source"], r["page"], r["split"], r.get("excluded"))
                              for r in source_rows)
    inventory_multiset = Counter()
    for row in pages:
        _require(isinstance(row, dict), "invalid G0 inventory page record")
        _require(isinstance(row.get("source"), str) and isinstance(row.get("page"), str),
                 "G0 inventory source/page missing")
        _require("source_excluded" in row, "G0 inventory source_excluded evidence missing")
        _require(row.get("source_split") in {"train", "dev", "test"},
                 "G0 inventory source split invalid")
        _require(row.get("source_excluded") is None or isinstance(row.get("source_excluded"), str),
                 "G0 inventory exclusion invalid")
        inventory_multiset[(row["source"], row["page"], row["source_split"], row["source_excluded"])] += 1
    _require(inventory_multiset == source_multiset, "G0 inventory pages do not match complete source/page multiset")
    as_metadata = _verify_source_identities(receipt, source_rows, pages)

    counts = receipt.get("counts")
    _require(isinstance(counts, dict), "G0 counts missing")
    count_keys = {
        "unresolved_exposure_records", "unresolved_exposures_by_label", "seg_manifest_pages",
        "missing_exposure_sources",
        "E12_pages", "E22_independent_pages", "nonexcluded_source_train_pages",
        "nonexcluded_train_pages_without_known_overlap", "known_overlaps", "source_excluded_pages",
        "text_contact_pages", "missing_page_identity", "missing_issue_identity",
        "nonexcluded_train_missing_title", "nonexcluded_train_title_seen_in_dev_test",
        "nonexcluded_train_in_press_train", "blind_test_eligible", "title_ood_test_eligible",
        "r2_lm_pretraining_issue_count", "r2_lm_pretraining_candidate_overlap_pages",
    }
    _require(count_keys <= set(counts), "G0 counts are incomplete")
    for key in count_keys - {"unresolved_exposures_by_label"}:
        _require(type(counts.get(key)) is int and counts[key] >= 0, f"G0 count missing/invalid: {key}")
    _require(isinstance(counts.get("unresolved_exposures_by_label"), dict),
             "G0 unresolved-exposure label counts missing")
    _require(all(isinstance(k, str) and type(v) is int and v >= 0
                 for k, v in counts["unresolved_exposures_by_label"].items()),
             "G0 unresolved-exposure label counts invalid")
    nonexcluded_train = [r for r in pages if r["source_split"] == "train"
                         and r["source_excluded"] in (None, False, "")]
    protected_titles = receipt.get("protected_titles", {})
    unresolved = receipt["unresolved_exposures"]
    by_label = Counter(r.get("label") for r in unresolved if isinstance(r, dict) and isinstance(r.get("label"), str))
    source_inputs = receipt["source_inputs"]

    def page_count(role: str) -> int:
        path = Path(source_inputs[role][0])
        payload = _read_pinned_json(path, receipt["inputs"])
        value = payload.get("pages") if isinstance(payload, dict) else payload
        _require(isinstance(value, list), f"G0 {role} source page list missing")
        return len(value)

    lm_manifest = _read_pinned_json(source_inputs["R2_LM_PleIAs_French_PD"][0], receipt["inputs"])
    _require(isinstance(lm_manifest, dict) and
             lm_manifest.get("source") == "PleIAs/French-PD-Newspapers (datasets-server)" and
             lm_manifest.get("min_ocr") == 95 and lm_manifest.get("seed") == 2 and
             lm_manifest.get("chars") == 58032354 and lm_manifest.get("protected_ids") == 370 and
             lm_manifest.get("skipped") == {"ocr": 378, "protected": 0},
             "R2 French-PD LM manifest metadata differs from reviewed receipt")
    lm_ids = lm_manifest.get("kept_file_ids")
    _require(isinstance(lm_ids, list) and len(lm_ids) == 322 and
             all(isinstance(value, str) for value in lm_ids) and len(set(lm_ids)) == len(lm_ids),
             "R2 French-PD LM manifest must contain 322 unique issue IDs")
    from .identity import canonical_issue
    lm_issues = set()
    for file_id in lm_ids:
        try:
            issue = canonical_issue(file_id)
        except Exception as exc:
            raise ReceiptError(f"invalid R2 French-PD issue identity: {file_id}") from exc
        _require(isinstance(issue, str) and issue.startswith("gallica:"),
                 f"invalid R2 French-PD issue identity: {file_id}")
        lm_issues.add(issue)
    _require(len(lm_issues) == len(lm_ids), "R2 French-PD manifest issue IDs are not unique canonically")

    exact = {
        "seg_manifest_pages": len(source_rows),
        "missing_exposure_sources": len(receipt["missing_exposure_sources"]),
        "E12_pages": page_count("E12"),
        "E22_independent_pages": page_count("E22"),
        "nonexcluded_source_train_pages": len(nonexcluded_train),
        "nonexcluded_train_pages_without_known_overlap": sum(
            not r.get("protected_page_or_issue") and not r.get("contacts")
            and "G0_review_B1_KB_quarantine" not in r.get("heldout_blockers", [])
            for r in nonexcluded_train),
        "known_overlaps": sum(bool(r.get("protected_page_or_issue")) for r in pages),
        "source_excluded_pages": sum(r.get("source_excluded") not in (None, False, "") for r in pages),
        "text_contact_pages": sum(bool(r.get("contacts")) for r in pages),
        "missing_page_identity": sum(r.get("page_key") is None for r in pages),
        "missing_issue_identity": sum(r.get("issue_key") is None for r in pages),
        "nonexcluded_train_missing_title": sum(not r.get("title_key") for r in nonexcluded_train),
        "nonexcluded_train_title_seen_in_dev_test": sum(bool(r.get("protected_title")) for r in nonexcluded_train),
        "nonexcluded_train_in_press_train": sum("E6_E8_press:train" in r.get("exposure", [])
                                                   for r in nonexcluded_train),
        "r2_lm_pretraining_issue_count": len(lm_issues),
        "r2_lm_pretraining_candidate_overlap_pages": sum(r.get("issue_key") in lm_issues for r in pages),
        "unresolved_exposure_records": len(unresolved),
        "unresolved_exposures_by_label": dict(sorted(by_label.items())),
        "blind_test_eligible": sum(r.get("blind_test_eligible") is True for r in pages),
        "title_ood_test_eligible": sum(r.get("title_ood_test_eligible") is True for r in pages),
    }
    _require(all(counts[k] == value for k, value in exact.items()), "G0 reported counts do not match page facts")
    return as_metadata


def _recompute_and_compare(receipt: dict[str, Any], inputs: dict[str, str], dirs: dict[str, Any],
                           expected_axel_root: str | Path) -> None:
    """Rerun the pinned inventory builder and compare all derived evidence."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts/g0_inventory.py"
    _require(str(script) in inputs and inputs[str(script)] == _sha(script),
             "cannot recompute G0 with an unpinned inventory implementation")
    spec = importlib.util.spec_from_file_location("babaorum_pinned_g0_inventory", script)
    _require(spec is not None and spec.loader is not None, "pinned G0 inventory implementation cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        recomputed = module.compute_inventory(expected_axel_root)
    except Exception as exc:
        raise ReceiptError(f"pinned G0 source recomputation failed: {type(exc).__name__}: {exc}") from exc
    _require(recomputed.get("metadata_current") is True,
             "pinned G0 sources changed during independent recomputation")
    _require(recomputed.get("inputs") == inputs,
             "G0 pinned input set or hashes differ from a fresh source recomputation")
    _require(recomputed.get("directory_inputs") == dirs,
             "G0 pinned directory inventory differs from a fresh source recomputation")
    _compare_derived_fields(receipt, recomputed)


def _compare_derived_fields(receipt: dict[str, Any], recomputed: dict[str, Any]) -> None:
    volatile = {"created_utc", "inputs", "directory_inputs", "axel_head",
                "axel_worktree_before", "axel_worktree_after", "metadata_current"}
    fields = (set(receipt) | set(recomputed)) - volatile
    mismatched = sorted(key for key in fields if receipt.get(key) != recomputed.get(key))
    _require(not mismatched,
             "G0 derived evidence differs from pinned source recomputation: " + ", ".join(mismatched[:12]))


def _verify_source_identities(receipt: dict[str, Any], source_rows: list[dict[str, Any]],
                              page_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from .identity import canonical_page, canonical_issue, IdentityError

    as_path = receipt["source_inputs"]["AS"][0]
    as_manifest = _read_pinned_json(as_path, receipt["inputs"])
    _require(isinstance(as_manifest, dict) and isinstance(as_manifest.get("pages"), list),
             "AS identity crosswalk pages missing")
    as_map = {}
    for row in as_manifest["pages"]:
        _require(isinstance(row, dict) and isinstance(row.get("page"), str),
                 "AS identity crosswalk row invalid")
        _require(row["page"] not in as_map, "duplicate AS identity crosswalk page")
        as_map[row["page"]] = row

    by_source_page: dict[tuple[str, str, str, Any], list[dict[str, Any]]] = {}
    for record in page_records:
        key = (record["source"], record["page"], record["source_split"], record["source_excluded"])
        by_source_page.setdefault(key, []).append(record)
    _require(all(len(rows) == 1 for rows in by_source_page.values()),
             "G0 source page must have exactly one inventory record")
    for source_row in source_rows:
        key = (source_row["source"], source_row["page"], source_row["split"], source_row.get("excluded"))
        candidates = by_source_page[key]
        source = source_row["source"]
        name = source_row["page"]
        if source == "as":
            as_entry = as_map.get(name)
            linked = as_entry.get("read_page") if as_entry else None
            source_page = None
            if linked:
                try:
                    source_page = canonical_page(linked)
                except IdentityError as exc:
                    raise ReceiptError(f"invalid AS crosswalk page identity: {name}") from exc
        else:
            source_page = None
            if source in {"read", "bnf", "kb"}:
                try:
                    source_page = canonical_page(name)
                except IdentityError:
                    source_page = None
                ark = source_row.get("ark")
                if source_page is not None and ark:
                    try:
                        if canonical_issue(ark) != source_page.split("/f", 1)[0]:
                            source_page = None
                    except IdentityError:
                        source_page = None
        source_issue = source_page.split("/f", 1)[0] if source_page and source_page.startswith("gallica:") else None
        if source_page is None and source != "kb":
            ark = source_row.get("ark")
            try:
                as_identity = as_map.get(name, {}).get("read_page") if source == "as" else None
                source_issue = canonical_issue(ark or (as_identity if source == "as" else name))
            except IdentityError:
                source_issue = None
        for record in candidates:
            if source == "as" and source_row.get("doc"):
                try:
                    _require(canonical_page(source_row["doc"]) == source_page,
                             f"AS source doc conflicts with crosswalk page identity: {name}")
                except IdentityError as exc:
                    raise ReceiptError(f"invalid AS source doc identity: {name}") from exc
            _require(record.get("page_key") == source_page,
                     f"G0 page key does not match canonical source identity: {source}:{name}")
            if source_issue is not None:
                _require(isinstance(record.get("issue_key"), str) and bool(record["issue_key"].strip()),
                         f"G0 historical issue identity unresolved: {source}:{name}")
            _require(record.get("issue_key") == source_issue,
                     f"G0 issue does not match canonical source identity: {source}:{name}")
    return as_map


def _check_clean_page(record: dict[str, Any], manifest_exposure: str) -> None:
    for field in _REQUIRED_EMPTY_LISTS:
        value = record.get(field)
        _require(isinstance(value, list) and value == [], f"G0 page evidence must be explicitly empty: {field}")
    excluded = record.get("source_excluded", object())
    _require(excluded is None or excluded is False, "source-excluded page cannot be released")
    _require(isinstance(record.get("heldout_blockers"), list) and not record["heldout_blockers"],
             "G0 page has held-out blockers")
    _require(record.get("eligible_for_split") is True, "G0 page not released")
    _require(isinstance(record.get("title"), str) and bool(record["title"].strip()),
             "protected title identity unresolved")
    _require(isinstance(record.get("title_key"), str) and bool(record["title_key"].strip()),
             "protected title identity unresolved")
    _require(len(record["title_key"].strip()) >= 4, "protected title identity too generic")
    if manifest_exposure == "sealed_test":
        _require(record.get("blind_test_eligible") is True, "G0 page is not explicitly released for blind test")


def _title_key(value: str) -> str | None:
    value = value.casefold().replace("œ", "oe").replace("æ", "ae")
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    key = " ".join(re.findall(r"[a-z0-9]+", value)) or None
    if key == "jdpl":
        return "journal des debats"
    if key in {"presse luxembourgeoise bnl titres non nommes", "titres non nommes"}:
        return None
    return key


def verify_metadata_receipt(receipt: dict[str, Any], *,
                            expected_axel_root: str | Path | None = None) -> None:
    """Verify freshness and metadata provenance without releasing a split.

    Pending or blocked inventories can be current. This helper never treats
    their status as authorization; ``verify_receipt`` adds the admission gate.
    """
    _require(receipt.get("schema_version") in {"0.3.0", "0.4.0"}, "unsupported G0 inventory schema")
    _require(receipt.get("role") == "metadata_inventory", "invalid G0 audit role")
    _require(type(receipt.get("metadata_current")) is bool, "G0 metadata_current flag missing/invalid")
    _catalogue_fields(receipt)
    _coherent_inventory_state(receipt)
    inputs, dirs, source_mtime = _strict_file_map(receipt)
    for suffix in ("/scripts/g0_inventory.py", "/src/babaorum/identity.py",
                   "/src/babaorum/legacy/data/doc_key.py", "/src/babaorum/contracts.py"):
        _require(any(path.endswith(suffix) for path in inputs),
                 f"G0 receipt is missing the hashed inventory implementation: {suffix}")
    before = receipt.get("axel_worktree_before")
    after = receipt.get("axel_worktree_after")
    _require(isinstance(before, dict) and set(before) == {"head", "status_porcelain", "tracked_diff_sha256"},
             "G0 pre-inventory source state missing")
    _require(isinstance(after, dict) and set(after) == {"head", "status_porcelain", "tracked_diff_sha256"},
             "G0 post-inventory source state missing")
    _require(before == after and receipt.get("axel_head") == before["head"],
             "Axel sources changed during G0 inventory generation")
    _require(isinstance(before["head"], str) and bool(before["head"])
             and isinstance(before["status_porcelain"], str)
             and isinstance(before["tracked_diff_sha256"], str)
             and _SHA.fullmatch(before["tracked_diff_sha256"]) is not None,
             "invalid G0 source state receipt")
    source_rows = _source_inventory(receipt, inputs)
    _validate_source_roles(receipt, inputs, dirs, expected_axel_root=expected_axel_root)
    source_inputs = receipt["source_inputs"]
    known_files = set(inputs)
    known_directories = set(dirs)
    role_paths = {p for paths in source_inputs.values() for p in paths}
    _require(role_paths <= known_files | known_directories,
             "G0 source-role path is not pinned in input hashes or directory listings")
    _require(set(source_inputs["C7"]) >= {receipt["inventory_source"]["path"]},
             "G0 C7 source role does not include the inventory source")
    created = receipt.get("created_utc")
    _require(isinstance(created, str), "G0 creation timestamp missing")
    try:
        instant = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReceiptError("G0 creation timestamp invalid") from exc
    _require(instant.tzinfo is not None and instant.utcoffset() is not None,
             "G0 creation timestamp must include timezone")
    instant_utc = instant.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    _require(instant_utc <= now, "G0 creation timestamp is in the future")
    _require(instant_utc.timestamp() + 1.0 >= source_mtime,
             "G0 creation timestamp predates one or more source metadata files")
    as_metadata = _check_complete_pages(receipt, source_rows)
    if expected_axel_root is not None:
        _recompute_and_compare(receipt, inputs, dirs, expected_axel_root)

    source_metadata: dict[tuple[str, str, str, Any], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (row["source"], row["page"], row["split"], row.get("excluded"))
        source_metadata.setdefault(key, []).append(row)

    titles = receipt.get("protected_titles")
    _require(isinstance(titles, dict), "G0 protected title map missing")
    for record in receipt["pages"]:
        _require(isinstance(record.get("protected_page_or_issue"), list), "protected_page_or_issue missing")
        _require(isinstance(record.get("protected_title"), list), "protected_title missing")
        _require(isinstance(record.get("contacts"), list), "contacts missing")
        _require(isinstance(record.get("identity_gaps"), list), "identity_gaps missing")
        _require(isinstance(record.get("press_same_issue"), list), "press_same_issue missing")
        _require(isinstance(record.get("exposure"), list), "exposure missing")
        title_key = record.get("title_key")
        _require(title_key is None or isinstance(title_key, str), "invalid title key")
        if record.get("title") is not None:
            _require(isinstance(record["title"], str), "invalid title metadata")
            normalized = _title_key(record["title"])
            _require(normalized == title_key,
                     "G0 title key does not match normalized title metadata")
        source_key = (record["source"], record["page"], record["source_split"], record["source_excluded"])
        source_matches = source_metadata[source_key]
        source_titles = {r.get("title") for r in source_matches if isinstance(r.get("title"), str) and r["title"].strip()}
        if record["source"] == "as":
            crosswalk_title = as_metadata.get(record["page"], {}).get("title")
            if isinstance(crosswalk_title, str) and crosswalk_title.strip():
                source_titles.add(crosswalk_title)
        if source_titles:
            _require(record["title"] in source_titles,
                     "G0 title is not supported by the pinned source metadata")
        source_dates = {r.get("date") for r in source_matches
                        if isinstance(r.get("date"), str) and r["date"].strip()}
        if source_dates:
            _require(record.get("date") in source_dates,
                     "G0 date is not supported by the pinned source metadata")
        _require(type(record.get("eligible_for_split")) is bool,
                 "G0 eligible_for_split must be an explicit boolean")
        if record.get("eligible_for_split") is True:
            _check_clean_page(record, "development")
            _require(isinstance(title_key, str) and bool(title_key.strip()),
                     "protected title identity unresolved")
            _require(len(title_key.strip()) >= 4, "protected title identity too generic")
            _require(title_key not in titles, "page title appears in protected title registry")
            _require(bool(source_titles) and record.get("title") in source_titles,
                     "release title must be corroborated by source or crosswalk metadata")
    if receipt["split_release_status"] != "released_after_independent_audit":
        _require(not any(r["eligible_for_split"] for r in receipt["pages"]),
                 "blocked/pending G0 receipt cannot mark pages eligible")


def verify_receipt(receipt: dict[str, Any], *, expected_axel_root: str | Path | None = None,
                   release_artifact: dict[str, Any] | None = None,
                   receipt_sha256: str | None = None) -> None:
    """Verify a receipt plus a separately pinned release decision.

    The metadata receipt cannot authorize itself. The release artifact must be
    carried by the frozen registry and bind the exact receipt and Axel root.
    """
    verify_metadata_receipt(receipt, expected_axel_root=expected_axel_root)
    _release_fields(receipt)
    _require(expected_axel_root is not None, "expected Axel root is required for G0 release admission")
    _require(isinstance(release_artifact, dict), "separate G0 release artifact is required")
    _require(release_artifact.get("schema_version") == "babaorum.g0_release.v1"
             and release_artifact.get("decision") == "released_after_independent_audit",
             "invalid independent G0 release artifact")
    _require(receipt_sha256 is not None and release_artifact.get("g0_receipt_sha256") == receipt_sha256,
             "G0 release artifact does not bind this exact receipt")
    _require(release_artifact.get("axel_root") == str(Path(expected_axel_root)),
             "G0 release artifact does not bind the expected Axel root")
    _require(release_artifact.get("inputs") == receipt.get("inputs"),
             "G0 release artifact source hashes differ from the metadata receipt")


def verify_page_release(receipt: dict[str, Any], page_key: str, issue_key: str, exposure: str) -> None:
    """Require one unique, clean, explicitly released page and issue record."""
    from .identity import canonical_page, canonical_issue, IdentityError

    try:
        wanted_page, wanted_issue = canonical_page(page_key), canonical_issue(issue_key)
    except IdentityError as exc:
        raise ReceiptError(f"invalid G0 release identity: {exc}") from exc
    matches = []
    for record in receipt["pages"]:
        try:
            page = canonical_page(record.get("page_key")) if record.get("page_key") else None
            issue = canonical_issue(record.get("issue_key")) if record.get("issue_key") else None
        except IdentityError as exc:
            raise ReceiptError("invalid canonical identity in G0 page inventory") from exc
        if page == wanted_page:
            matches.append((record, issue))
    _require(len(matches) == 1, f"G0 page must have exactly one canonical audit record: {wanted_page}")
    record, issue = matches[0]
    _require(isinstance(record.get("issue_key"), str) and bool(record["issue_key"].strip()),
             f"G0 historical issue identity unresolved: {wanted_page}")
    _require(issue == wanted_issue, f"G0 issue does not match manifest: {wanted_page}")
    _check_clean_page(record, exposure)
