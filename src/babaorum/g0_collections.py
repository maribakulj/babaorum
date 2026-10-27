"""G0 v2 registry of collections, statuses and the eligibility function.

Rule (decided by Marcel, 8 October 2026): a source document is eligible for the
TEST only if it belongs to a sealed test manifest and to no collection that was
used for training, tuning, development or an earlier test. This module checks
the registry itself (schema, required collections, manifest hashes, file-tree
fingerprints) and answers ``eligibility`` queries. It releases nothing and never
creates config/g0_release.json.

Two meanings of "exposed" are kept apart:
- exposed to TRAINING/TUNING/DEV: the document may be reused for train/dev, never for the test;
- already MEASURED (``test_used``): allowed nowhere, because it was read as a test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA_VERSION = "babaorum.g0_collections.v3"  # the frozen v1 registry (audited)
SCHEMA_VERSION_V4 = "babaorum.g0_collections.v4"  # adds gallica_scope and headline_page_keys (LOT-G0-02)
SCHEMAS = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_V4})
NO_KEY = "aucune clé à ce niveau"
LEVELS = ("page", "issue", "title")
#: test_descriptive: a page of the sealed test with headline=false and no usable source document; never in the measure that counts
ROLES = ("train", "dev", "test", "test_descriptive")
#: gallica_scope (v4): can the collection contain a Gallica-credited document? "not_auditable" is for models (declared limits)
GALLICA_SCOPES = frozenset({"possible", "excluded", "not_auditable", "not_applicable"})

#: statuses that forbid the TEST role (training/tuning/dev exposure, or not sealed)
TRAINING_EXPOSED = frozenset({"exposed", "exposed_not_auditable"})
#: statuses that forbid train AND dev (measured as a test, or reserved for a test)
TEST_LOCKED = frozenset({"test_used", "reserved_test", "sealed_test"})
STATUSES = TRAINING_EXPOSED | TEST_LOCKED | {"disjoint_by_construction", "not_exposed"}

REQUIRED_COLLECTIONS = frozenset({
    "bnl_pack", "as_anno_183", "kb_impact", "newseye", "mccatmus", "gallicorpora", "hipe_ocrepair",
    "pleias_french_pd_322", "presse_e7_train_dev", "mille_feuilles_sources",
    "icdar2017_2019", "bnf_ocr_corrige_presse", "gallica_pseudo", "gallica_e13", "e22_bench",
    "gallica_e31_reserve", "reader_lm_corpus", "kraken_blla_2026", "dots_ocr", "presse_e7_test",
    "impresso", "nzz_impresso", "e12_dev", "e12_test", "presse_e7_test_refused",
})
_SHA = re.compile(r"^[0-9a-f]{64}$")


class CollectionRegistryError(ValueError):
    """A refusal with a stable code (G0-03 v2): the code is what the tests assert, the message is for people."""

    def __init__(self, message: str, code: str = "UNCLASSIFIED") -> None:
        super().__init__(message)
        self.code = code


def _require(ok: bool, message: str, code: str) -> None:
    if not ok:
        raise CollectionRegistryError(message, code)


# --- v2.2: paths are {"root": <declared name>, "path": <relative>}; roots come from the environment or a local, uncommitted file ---

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_NAMES = ("repo", "corpus_root", "axel_root", "outils_root", "heritage_root", "reviews_root")  # reviews_root: avenant AV-2026-10-24-REVIEWS-ROOT
LOCAL_ROOTS_FILE = REPO_ROOT / "config" / "g0_roots.local.json"   # ignored by git, never committed


def declared_root(name: str) -> Path:
    """The absolute directory of a declared root. Never a default: an unknown or unset root refuses (ROOT_UNDECLARED:<name>)."""
    if name not in ROOT_NAMES:
        raise CollectionRegistryError(f"unknown root: {name}", f"ROOT_UNDECLARED:{name}")
    if name == "repo":
        return REPO_ROOT
    value = os.environ.get(f"G0_{name.upper()}")
    if not value and LOCAL_ROOTS_FILE.exists():
        value = json.loads(LOCAL_ROOTS_FILE.read_text(encoding="utf-8")).get(name)
    if not value or not Path(value).is_absolute():
        raise CollectionRegistryError(f"root not declared on this machine: {name}", f"ROOT_UNDECLARED:{name}")
    return Path(value)


def resolve_path(ref: Any) -> Path:
    """A {root, path} reference to an absolute path. A bare string, an absolute path or one with '..' refuses (PATH_NOT_RELATIVE)."""
    if isinstance(ref, str):
        raise CollectionRegistryError(f"path is not a {{root, path}} reference: {ref}", "PATH_NOT_RELATIVE")
    if not (isinstance(ref, dict) and set(ref) == {"root", "path"} and isinstance(ref["root"], str) and isinstance(ref["path"], str)):
        raise CollectionRegistryError(f"malformed path reference: {ref}", "PATH_FORM")
    rel = Path(ref["path"])
    if not ref["path"] or rel.is_absolute() or ".." in rel.parts:
        raise CollectionRegistryError(f"path must be relative and without '..': {ref['path']}", "PATH_NOT_RELATIVE")
    return declared_root(ref["root"]) / rel


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_fingerprint(root: Path) -> tuple[int, str]:
    items = sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for name, size in items:
        h.update(f"{name}\0{size}\n".encode())
    return len(items), h.hexdigest()


def _check_pin(m: Mapping[str, Any], name: str) -> None:
    p = resolve_path(m.get("path"))
    _require(p.is_file(), f"manifest missing: {name}: {p}", "MANIFEST_MISSING")
    _require(bool(_SHA.match(str(m.get("sha256", "")))), f"manifest hash malformed: {name}: {p}", "MANIFEST_HASH_MALFORMED")
    _require(_file_sha256(p) == m["sha256"], f"manifest hash changed: {name}: {p}", "MANIFEST_HASH_CHANGED")
    _require(p.stat().st_size == m.get("bytes"), f"manifest size changed: {name}: {p}", "MANIFEST_SIZE_CHANGED")


def _check_match_keys(entry: Mapping[str, Any], name: str) -> None:
    keys = entry.get("match_keys")
    _require(isinstance(keys, dict) and set(keys) == set(LEVELS), f"match_keys must name the three levels: {name}", "MATCH_KEYS_LEVELS")
    reasons = entry.get("match_keys_reasons") or {}
    for level in LEVELS:
        spec = keys[level]
        if spec == NO_KEY:
            _require(isinstance(reasons.get(level), str) and reasons[level].strip(),
                     f"{NO_KEY!r} at level {level} needs a written reason: {name}", "NO_KEY_REASON_MISSING")
            continue
        _require(isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"],
                 f"match key at level {level} must be a key name or {NO_KEY!r}: {name}", "MATCH_KEY_NAME")
        has_values = isinstance(spec.get("values"), list)
        has_file = isinstance(spec.get("values_file"), dict)
        _require(has_values != has_file, f"match key {level} needs exactly one of values/values_file: {name}", "MATCH_KEY_VALUES_XOR_FILE")
        if has_file:
            _check_pin(spec["values_file"], f"{name}.{level}")
        if "complete" in spec:
            _require(spec["complete"] is True and level == "title", f"'complete' is only valid as true on the title level: {name}", "COMPLETE_ONLY_TITLE")
            cov = spec.get("coverage")
            _require(isinstance(cov, dict) and cov.get("keyed") == cov.get("total"),
                     f"a complete title set must key every unit: {name}", "COMPLETE_NEEDS_FULL_COVERAGE")
        coverage = spec.get("coverage")
        if coverage is not None:
            _require(isinstance(coverage, dict) and isinstance(coverage.get("keyed"), int)
                     and isinstance(coverage.get("total"), int) and 0 <= coverage["keyed"] <= coverage["total"],
                     f"match key coverage malformed at level {level}: {name}", "COVERAGE_MALFORMED")


def declared_keys(registry: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """For every collection and level: the key name used, or the NO_KEY phrase. Headline cannot freeze without it."""
    table: dict[str, dict[str, str]] = {}
    for name, entry in registry["collections"].items():
        table[name] = {level: (entry["match_keys"][level] if entry["match_keys"][level] == NO_KEY
                               else entry["match_keys"][level]["name"]) for level in LEVELS}
    return table


def verify_collections(path: str | Path, *, recompute_keys: bool = True) -> dict[str, Any]:
    """Return a summary, or raise CollectionRegistryError on any inconsistency.

    ``recompute_keys`` (default True): the declared keys of EVERY collection are recomputed from the pinned
    sources (babaorum.g0_keys, and the pinned seal for the sealed test and its sibling) and must be equal,
    in both directions. Only unit tests on synthetic registries switch it off.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
        registry = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise CollectionRegistryError(f"collections registry unreadable: {exc}", "REGISTRY_UNREADABLE") from exc
    _require(isinstance(registry, dict), "collections registry must be an object", "REGISTRY_NOT_OBJECT")
    _require(registry.get("schema_version") in SCHEMAS, "collections registry schema_version mismatch", "SCHEMA_MISMATCH")
    collections = registry.get("collections")
    _require(isinstance(collections, dict), "collections registry has no collections map", "NO_COLLECTIONS_MAP")
    _require(set(collections) == REQUIRED_COLLECTIONS,
             "collections registry does not list exactly the required collections", "COLLECTION_SET_MISMATCH")
    limits = registry.get("limits")
    _require(isinstance(limits, list) and bool(limits), "collections registry must state its limits", "NO_LIMITS")
    manifests = trees = 0
    for name, entry in collections.items():
        _require(isinstance(entry, dict), f"collection entry invalid: {name}", "ENTRY_INVALID")
        status = entry.get("status")
        _require(status in STATUSES, f"collection status invalid: {name}", "STATUS_INVALID")
        _require(isinstance(entry.get("manifests"), list) and isinstance(entry.get("trees"), list),
                 f"collection manifests/trees invalid: {name}", "PINS_MALFORMED")
        _require(bool(entry["manifests"]) or bool(entry["trees"]), f"collection pins nothing: {name}", "PINS_NOTHING")
        _check_match_keys(entry, name)
        if registry["schema_version"] == SCHEMA_VERSION_V4:
            scope = entry.get("gallica_scope")
            _require(isinstance(scope, dict) and scope.get("value") in GALLICA_SCOPES and isinstance(scope.get("reason"), str)
                     and scope["reason"].strip(), f"gallica_scope must be possible, excluded or not_auditable with a written reason: {name}", "GALLICA_SCOPE_INVALID")
            _require((status == "exposed_not_auditable") == (scope["value"] == "not_auditable"),
                     f"not_auditable is for models only (exposed_not_auditable): {name}", "NOT_AUDITABLE_MISMATCH")
            _require((status == "sealed_test" or "sibling_of" in entry) == (scope["value"] == "not_applicable"),
                     f"not_applicable is for the seal and its sibling only: {name}", "NOT_APPLICABLE_MISMATCH")
        if status == "sealed_test":
            _require(isinstance(entry.get("sealed_manifest"), dict), f"sealed_test needs a pinned sealed_manifest: {name}", "SEALED_MANIFEST_MISSING")
            _check_pin(entry["sealed_manifest"], f"{name}.sealed_manifest")
            _require(entry["match_keys"]["page"] != NO_KEY, f"sealed_test needs page-level keys: {name}", "SEALED_NO_PAGE_KEYS")
        if "sibling_of" in entry:
            target = collections.get(entry["sibling_of"])
            _require(isinstance(target, dict) and target.get("status") == "sealed_test" and status == "reserved_test",
                     f"sibling_of must point to a sealed_test and belong to a reserved_test: {name}", "SIBLING_TARGET_INVALID")
        if status == "not_exposed":
            evidence = entry.get("evidence")
            _require(isinstance(evidence, dict) and isinstance(evidence.get("command"), str)
                     and evidence["command"].strip() and evidence.get("result_empty") is True,
                     f"not_exposed requires evidence with a command and an empty result: {name}", "NOT_EXPOSED_EVIDENCE_MISSING")
        for m in entry["manifests"]:
            _check_pin(m, name)
            manifests += 1
        for t in entry["trees"]:
            root = resolve_path(t.get("root"))
            _require(root.is_dir(), f"tree root missing: {name}: {root}", "TREE_ROOT_MISSING")
            count, fingerprint = _tree_fingerprint(root)
            _require(count == t.get("file_count") and fingerprint == t.get("tree_sha256_names_and_sizes"),
                     f"tree changed: {name}: {root}", "TREE_CHANGED")
            trees += 1
    for name, entry in collections.items():
        if "sibling_of" in entry:
            mine = _key_values(entry["match_keys"]["page"]) or set()
            theirs = _key_values(collections[entry["sibling_of"]]["match_keys"]["page"]) or set()
            _require(not (mine & theirs), f"page sets of a sealed test and its sibling must be disjoint "
                                          f"({len(mine & theirs)} shared page keys): {name}", "SIBLING_PAGES_OVERLAP")
    if recompute_keys:
        _compare_with_sources(registry)
    return {
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "collections": len(collections),
        "by_status": {s: sum(c["status"] == s for c in collections.values()) for s in sorted(STATUSES)
                      if any(c["status"] == s for c in collections.values())},
        "exposed": sum(c["status"].startswith("exposed") for c in collections.values()),
        "manifests": manifests,
        "trees": trees,
    }


def _compare_with_sources(registry: Mapping[str, Any]) -> None:
    """Declared keys == keys recomputed from the pinned sources, both ways, for every collection."""
    from babaorum import g0_keys  # lazy: reads the sources, which unit tests on synthetic registries do not have

    collections = registry["collections"]
    expected = dict(g0_keys.compute_keys())
    headline: dict[str, set[str]] = {}
    for name, entry in collections.items():
        if entry["status"] == "sealed_test":
            seal = json.loads(resolve_path(entry["sealed_manifest"]["path"]).read_text(encoding="utf-8"))
            expected[name] = g0_keys.keys_from_seal(seal, "pages")
            headline[name] = set(g0_keys.headline_keys_from_seal(seal))
    for name, entry in collections.items():
        if "sibling_of" in entry:
            target = collections[entry["sibling_of"]]
            seal = json.loads(resolve_path(target["sealed_manifest"]["path"]).read_text(encoding="utf-8"))
            expected[name] = g0_keys.keys_from_seal(seal, "refused")
    for name, entry in collections.items():
        _require(name in expected, f"no source from which to recompute the keys of {name}", "NO_SOURCE_FOR_KEYS")
        keys, _ = expected[name]
        for level in LEVELS:
            declared, exp = entry["match_keys"][level], keys[level]
            if declared == NO_KEY:
                _require(exp is None, f"the sources hold keys that the registry declares absent: {name}.{level}", "KEYS_DECLARED_ABSENT")
                continue
            _require(exp is not None, f"keys are declared but the sources hold none: {name}.{level}", "KEYS_WITHOUT_SOURCE")
            got, want = _key_values(declared) or set(), set(exp["values"])
            _require(got == want, f"declared keys differ from the keys recomputed from the pinned sources: {name}.{level}: "
                                  f"{len(got - want)} only declared, {len(want - got)} only in the sources", "KEYS_DIFFER_FROM_PINNED")
            if declared.get("coverage") is not None:
                _require(declared["coverage"] == exp["coverage"], f"declared coverage differs from the sources: {name}.{level}", "COVERAGE_DIFFERS")
            _require(bool(declared.get("complete")) == bool(exp.get("complete")), f"'complete' differs from the sources: {name}.{level}", "COMPLETE_DIFFERS")
        if name in headline and entry.get("headline_page_keys") is not None:
            _require((_key_values(entry["headline_page_keys"]) or set()) == headline[name],
                     f"headline_page_keys differ from the headline pages of the pinned seal: {name}", "HEADLINE_KEYS_DIFFER")


# --- coverage: every data entry on disk is classified ---------------------------------

COVERAGE_SCHEMA = "babaorum.g0_coverage.v1"


def _live_entries(coverage: Mapping[str, Any]) -> set[str]:
    roots = coverage["roots"]
    live = {f"corpus-vt/{p.name}" for p in resolve_path(roots["corpus_vt"]).iterdir()}
    live |= {f"axel/runs/{p.name}" for p in resolve_path(roots["axel_runs"]).iterdir()}
    prepared = resolve_path(roots["axel_prepared"])
    live |= {f"axel/data/prepared/{p.relative_to(prepared)}" for p in prepared.rglob("*") if p.is_file()}
    return live


def verify_coverage(coverage_path: str | Path, registry_path: str | Path) -> dict[str, Any]:
    """Every entry of corpus-vt, axel/runs and axel/data/prepared must be classified.

    An entry names the collections it touches (all of them known to the registry) and
    cites its source, or is ``no_page_data`` with a recorded proof. A new entry on disk
    that is not in the file, an unclassified entry, or an unknown collection fails.
    """
    try:
        coverage = json.loads(Path(coverage_path).read_bytes())
        registry = json.loads(Path(registry_path).read_bytes())
    except (OSError, ValueError) as exc:
        raise CollectionRegistryError(f"coverage unreadable: {exc}", "COVERAGE_UNREADABLE") from exc
    _require(coverage.get("schema_version") == COVERAGE_SCHEMA, "coverage schema_version mismatch", "COVERAGE_SCHEMA_MISMATCH")
    entries = coverage.get("entries")
    _require(isinstance(entries, dict) and bool(entries), "coverage has no entries", "COVERAGE_EMPTY")
    known = set(registry["collections"])
    counts = {"classified": 0, "no_page_data": 0}
    for key, entry in entries.items():
        _require(not entry.get("unclassified"), f"unclassified entry: {key}", "COVERAGE_UNCLASSIFIED_ENTRY")
        names = entry.get("collections")
        _require(isinstance(names, list), f"coverage entry malformed: {key}", "COVERAGE_ENTRY_MALFORMED")
        _require(set(names) <= known, f"coverage entry names an unknown collection: {key}", "COVERAGE_UNKNOWN_COLLECTION")
        if names:
            _require(bool(entry.get("source_files")), f"classified entry cites no source: {key}", "COVERAGE_NO_SOURCE")
            for cited in entry["source_files"]:
                if re.fullmatch(r"(axel|corpus-vt)/\S+", cited):
                    # the citation goes through the same reference rule as every path: absolute or '..' is refused
                    root_name = "axel_root" if cited.startswith("axel/") else "corpus_root"
                    target = resolve_path({"root": root_name, "path": cited.split("/", 1)[1]})
                    _require(target.exists(), f"cited source missing: {key}: {cited}", "COVERAGE_SOURCE_MISSING")
            counts["classified"] += 1
        else:
            proof = entry.get("no_page_data")
            _require(isinstance(proof, dict) and isinstance(proof.get("command"), str) and proof["command"].strip()
                     and proof.get("result_empty") is True, f"no_page_data needs a recorded proof: {key}", "COVERAGE_NO_PROOF")
            counts["no_page_data"] += 1
    missing = sorted(_live_entries(coverage) - set(entries))
    _require(not missing, f"entries on disk not classified: {missing[:5]}{' ...' if len(missing) > 5 else ''}", "COVERAGE_ENTRIES_UNCLASSIFIED")
    return {"entries": len(entries), **counts, "unclassified": 0}


# --- eligibility ---------------------------------------------------------------------

def _key_values(spec: Any) -> set[str] | None:
    """The key set of one level of one collection, or None when it has no key there."""
    if spec == NO_KEY:
        return None
    if "values" in spec:
        return {str(v) for v in spec["values"]}
    values = json.loads(resolve_path(spec["values_file"]["path"]).read_text(encoding="utf-8"))
    return {str(v) for v in values}


def _scope(entry: Mapping[str, Any]) -> str | None:
    scope = entry.get("gallica_scope")
    return scope["value"] if isinstance(scope, dict) else None  # a v3 registry has none


def _issue_keys_complete(entry: Mapping[str, Any]) -> bool:
    """Is the NUMBER-level (issue) key set of the collection complete (every unit keyed)?"""
    spec = entry["match_keys"]["issue"]
    if spec == NO_KEY:
        return False
    cov = spec.get("coverage")
    return isinstance(cov, dict) and cov.get("keyed") == cov.get("total")


# --- G0-03 v3: the anchor. The registry is read from the disk, at a path resolved from the repository root, and its hash is pinned. ---

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "g0_collections_v2_2.json"
REGISTRY_SHA256 = "c03e1f60e33ff96928590d1b63816d51f894d60999c8ae0c22eca667c24a379f"


def _disk_registry() -> dict[str, Any]:
    """The anchored registry, read from REGISTRY_PATH; its SHA-256 must be REGISTRY_SHA256 (code REGISTRY_ANCHOR_CHANGED otherwise)."""
    try:
        raw = REGISTRY_PATH.read_bytes()
    except OSError as exc:
        raise CollectionRegistryError(f"anchor unreadable: {exc}", "REGISTRY_ANCHOR_CHANGED") from exc
    if hashlib.sha256(raw).hexdigest() != REGISTRY_SHA256:
        raise CollectionRegistryError(f"anchor changed: {REGISTRY_PATH}", "REGISTRY_ANCHOR_CHANGED")
    return json.loads(raw)


def _pinned_as_on_disk(spec: Any, disk_spec: Any) -> bool:
    """True when ``spec`` is a file whose triple (path, sha256, bytes) equals the one of the anchored registry, and whose file still matches it."""
    if not isinstance(spec, dict) or not isinstance(disk_spec, dict) or "values_file" not in spec or "values_file" not in disk_spec:
        return False
    if spec["values_file"] != disk_spec["values_file"]:
        return False
    f = spec["values_file"]
    try:
        raw = resolve_path(f["path"]).read_bytes()
    except OSError:
        return False
    return hashlib.sha256(raw).hexdigest() == f["sha256"] and len(raw) == f["bytes"]


def _pin_codes(registry: Mapping[str, Any], name: str, entry: Mapping[str, Any], level: str, role: str) -> list[str]:
    """SEAL_NOT_PINNED / HEADLINE_LIST_NOT_PINNED for one sealed collection, from the in-memory entry against the anchored registry."""
    disk_entry = _disk_registry()["collections"].get(name, {})
    if level == "page":
        return [] if _pinned_as_on_disk(entry["match_keys"]["page"], disk_entry.get("match_keys", {}).get("page")) else [f"SEAL_NOT_PINNED:{name}"]
    if role == "test_descriptive":
        ok = _pinned_as_on_disk(entry.get("headline_page_keys"), disk_entry.get("headline_page_keys"))
        return [] if ok else [f"HEADLINE_LIST_NOT_PINNED:{name}"]
    return []


def eligibility(registry: Mapping[str, Any], role: str, *, page_key: str | None = None,
                issue_key: str | None = None, title_key: str | None = None) -> dict[str, Any]:
    """May a document with these keys be used for ``role``? Lists EVERY collection it touches.

    Every refusal carries a code (``refusal_codes``, parallel to ``refusals``, same order). Emission order: 1 NOT_SEALED, 2 SEAL_NOT_PINNED,
    3 HEADLINE_LIST_NOT_PINNED, 4 SIBLING_PAGE_CONFLICT / EXPOSED_MATCH / LOCKED_MATCH (registry order), 5 COMPLETE_TITLE_MATCH,
    6 NO_SOURCE_DOCUMENT, 7 NO_HEADLINE_LIST / HEADLINE_WRONG_ROLE, 8 NO_USABLE_KEY, 9 INCOMPLETE_NUMBER_KEY (registry order).
    Matching is exact on the keys the caller supplies. Title-level matches are reported (``title_seen``) but do not refuse alone, unless the
    title set of the collection is declared complete.
    """
    _require(role in ROLES, f"unknown role: {role}", "UNKNOWN_ROLE")
    supplied = {"page": page_key, "issue": issue_key, "title": title_key}
    _require(any(v is not None for v in supplied.values()), "eligibility needs at least one key", "NO_IDENTITY")
    matches: list[dict[str, Any]] = []
    cannot_clear: list[str] = []
    for name, entry in registry["collections"].items():
        status = entry["status"]
        usable_level = False
        for level in LEVELS:
            spec = entry["match_keys"][level]
            wanted = supplied[level]
            values = _key_values(spec)
            if values is None or wanted is None:
                continue
            complete_title = level == "title" and spec.get("complete") is True
            usable_level = usable_level or level != "title" or complete_title
            if str(wanted) in values:
                matches.append({"collection": name, "level": level, "key": str(wanted), "status": status,
                                "complete_title": complete_title})
        if status in TEST_LOCKED and not usable_level:
            cannot_clear.append(name)  # no page/issue key can be compared: it cannot be shown that the document is outside it
    codes: list[str] = []
    refusals: list[str] = []
    exceptions: list[dict[str, Any]] = []

    def refuse(code: str, text: str) -> None:
        codes.append(code)
        refusals.append(text)

    page_issue = [m for m in matches if m["level"] in ("page", "issue")]
    is_test = role in ("test", "test_descriptive")
    if is_test:
        sealed_page = [m for m in matches if m["status"] == "sealed_test" and m["level"] == "page"]
        sealed_names = sorted({m["collection"] for m in sealed_page})
        if not sealed_page:
            refuse("NOT_SEALED", "not in any sealed_test manifest (only a sealed test can make a page eligible)")
        for name in sealed_names:
            for code in _pin_codes(registry, name, registry["collections"][name], "page", role):
                refuse(code, f"{code.split(':')[0].lower()}: {name} (the in-memory key list is not the pinned file of the anchored registry)")
        if role == "test_descriptive" and sealed_names:
            for name in sealed_names:
                for code in _pin_codes(registry, name, registry["collections"][name], "headline", role):
                    refuse(code, f"{code.split(':')[0].lower()}: {name} (the headline list is not the pinned file)")
        siblings = {n for n, e in registry["collections"].items() if e.get("sibling_of") in sealed_names}
        for m in page_issue:
            if m["collection"] in sealed_names:
                continue
            if m["collection"] in siblings:
                if m["level"] == "issue":
                    exceptions.append({**m, "label": "sibling_exception"})
                else:
                    refuse(f"SIBLING_PAGE_CONFLICT:{m['collection']}",
                           f"page key also in {m['collection']}, the unsealed sibling of the sealed test: conflict")
                continue
            refuse(f"EXPOSED_MATCH:{m['collection']}:{m['level']}", f"also in {m['collection']} ({m['status']}) at {m['level']} level")
        for m in matches:
            if (m["level"] == "title" and m["complete_title"] and m["collection"] not in sealed_names
                    and m["status"] in TRAINING_EXPOSED | {"test_used", "reserved_test"}):
                refuse(f"COMPLETE_TITLE_MATCH:{m['collection']}",
                       f"title equals the title of {m['collection']} ({m['status']}), whose title set is complete")
        if role == "test" and issue_key is None:
            refuse("NO_SOURCE_DOCUMENT", "no source document (issue key): refused for test; test_descriptive may admit a headline=false page")
        if role == "test_descriptive" and sealed_names:
            spec = registry["collections"][sealed_names[0]].get("headline_page_keys")
            if spec is None:
                refuse(f"NO_HEADLINE_LIST:{sealed_names[0]}",
                       "the sealed collection does not say which pages are headline: test_descriptive cannot be granted")
            elif str(page_key) in (_key_values(spec) or set()):
                refuse("HEADLINE_WRONG_ROLE", "a headline page is judged with role test, not test_descriptive")
        for name in cannot_clear:
            if name not in sealed_names:
                refuse(f"NO_USABLE_KEY:{name}",
                       f"{name} ({registry['collections'][name]['status']}) has no usable key: cautious refusal")
    else:
        for m in page_issue:
            if m["status"] in TEST_LOCKED:
                refuse(f"LOCKED_MATCH:{m['collection']}:{m['level']}", f"in {m['collection']} ({m['status']}) at {m['level']} level: forbidden for {role}")
        for name in cannot_clear:
            refuse(f"NO_USABLE_KEY:{name}", f"{name} ({registry['collections'][name]['status']}) has no usable key: cautious refusal")
    if is_test:  # rules 1 and 2: a collection that may hold a Gallica document, with an incomplete number key, clears no page
        for name, entry in registry["collections"].items():
            if _scope(entry) == "possible" and not _issue_keys_complete(entry):
                refuse(f"INCOMPLETE_NUMBER_KEY:{name}",
                       f"{name} ({entry['status']}, gallica_scope possible) has no complete number key: it cannot show that the page is outside it")
    return {
        "role": role,
        "allowed": not refusals,
        "refusals": refusals,
        "refusal_codes": codes,
        "matches": matches,
        "matched_collections": sorted({m["collection"] for m in matches}),
        "title_seen": any(m["level"] == "title" for m in matches),
        "sibling_exceptions": exceptions,
        "supplied": {k: v for k, v in supplied.items() if v is not None},
    }
