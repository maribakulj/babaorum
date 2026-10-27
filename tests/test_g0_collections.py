"""G0 v2 registry: schema gate and eligibility, with synthetic files only."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from babaorum.g0_collections import (COVERAGE_SCHEMA, NO_KEY, REQUIRED_COLLECTIONS, SCHEMA_VERSION, CollectionRegistryError,
                                     eligibility, verify_coverage)
from babaorum.g0_collections import verify_collections as _verify_collections
from babaorum.g0_collections import _key_values

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from g0_registry_complete import complete_registry  # noqa: E402


def verify_collections(path):
    """Synthetic registries have no sources to recompute the keys from: only the schema gate is exercised here.
    The recomputation (default) is exercised on the real registry by test_g0_audit_counterexamples.py."""
    return _verify_collections(path, recompute_keys=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_keys():
    return {level: NO_KEY for level in ("page", "issue", "title")}


def tmp_ref(path, tmp_path):
    """A v2.2 path reference inside the test's temporary corpus root (declared by the autouse fixture below)."""
    return {"root": "corpus_root", "path": Path(path).relative_to(tmp_path).as_posix()}


@pytest.fixture(autouse=True)
def _corpus_root_is_the_test_directory(request, monkeypatch):
    """Synthetic registries live in tmp_path: that directory is the declared corpus_root, for these tests only."""
    if "tmp_path" in request.fixturenames:
        tmp = request.getfixturevalue("tmp_path")
        monkeypatch.setenv("G0_CORPUS_ROOT", str(tmp))
        monkeypatch.setenv("G0_AXEL_ROOT", str(tmp / "axel"))   # cited 'axel/…' sources resolve here, never on this machine


def build(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"doc_id": "d1"}\n', encoding="utf-8")
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"xx")
    h = hashlib.sha256()
    h.update(b"a.bin\x002\n")
    entry = {
        "status": "exposed",
        "manifests": [{"path": tmp_ref(manifest, tmp_path), "sha256": sha(manifest), "bytes": manifest.stat().st_size}],
        "trees": [{"root": tmp_ref(root, tmp_path), "file_count": 1, "tree_sha256_names_and_sizes": h.hexdigest()}],
        "match_keys": no_keys(),
        "match_keys_reasons": {level: "test fixture: no identity at this level" for level in ("page", "issue", "title")},
    }
    registry = {"schema_version": SCHEMA_VERSION, "limits": ["pretraining not auditable"],
                "collections": {name: copy.deepcopy(entry) for name in REQUIRED_COLLECTIONS}}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path, registry, manifest, root


def write(path, registry):
    path.write_text(json.dumps(registry), encoding="utf-8")


# --- schema gate ------------------------------------------------------------------------

def test_valid_registry(tmp_path):
    path, _, _, _ = build(tmp_path)
    summary = verify_collections(path)
    assert summary["collections"] == len(REQUIRED_COLLECTIONS)
    assert summary["exposed"] == len(REQUIRED_COLLECTIONS)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["collections"].pop("bnl_pack"), "required collections"),
    (lambda r: r["collections"].update(surprise=copy.deepcopy(r["collections"]["newseye"])), "required collections"),
    (lambda r: r["collections"]["kb_impact"].update(status="clean"), "status invalid"),
    (lambda r: r["collections"]["hipe_ocrepair"].update(manifests=[], trees=[]), "pins nothing"),
    (lambda r: r.update(limits=[]), "limits"),
    (lambda r: r.update(schema_version="babaorum.g0_collections.v2"), "schema_version"),
    (lambda r: r["collections"]["newseye"].pop("match_keys"), "three levels"),
    (lambda r: r["collections"]["newseye"]["match_keys"].update(page="doc_key"), "key name or"),
    (lambda r: r["collections"]["newseye"]["match_keys"].update(page={"name": "doc_key"}), "exactly one of"),
])
def test_schema_violations_are_refused(tmp_path, mutate, match):
    path, registry, _, _ = build(tmp_path)
    mutate(registry)
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match=match):
        verify_collections(path)


def test_changed_manifest_is_refused(tmp_path):
    path, _, manifest, _ = build(tmp_path)
    manifest.write_text('{"doc_id": "d2"}\n', encoding="utf-8")
    with pytest.raises(CollectionRegistryError, match="hash changed|size changed"):
        verify_collections(path)


def test_changed_tree_is_refused(tmp_path):
    path, _, _, root = build(tmp_path)
    (root / "b.bin").write_bytes(b"y")
    with pytest.raises(CollectionRegistryError, match="tree changed"):
        verify_collections(path)


def test_missing_manifest_is_refused(tmp_path):
    path, _, manifest, _ = build(tmp_path)
    manifest.unlink()
    with pytest.raises(CollectionRegistryError, match="manifest missing"):
        verify_collections(path)


def test_not_exposed_requires_a_proof(tmp_path):
    path, registry, _, _ = build(tmp_path)
    registry["collections"]["mille_feuilles_sources"]["status"] = "not_exposed"
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="not_exposed requires evidence"):
        verify_collections(path)
    registry["collections"]["mille_feuilles_sources"]["evidence"] = {"command": "   ", "result_empty": True}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="not_exposed requires evidence"):
        verify_collections(path)
    registry["collections"]["mille_feuilles_sources"]["evidence"] = {"command": "grep -rn X ~/axel/docs", "result_empty": False}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="not_exposed requires evidence"):
        verify_collections(path)
    registry["collections"]["mille_feuilles_sources"]["evidence"] = {"command": "grep -rn X ~/axel/docs", "result_empty": True}
    write(path, registry)
    assert verify_collections(path)["by_status"]["not_exposed"] == 1


def test_key_values_file_is_pinned(tmp_path):
    path, registry, _, _ = build(tmp_path)
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(["p1"]), encoding="utf-8")
    pin = {"path": tmp_ref(keys, tmp_path), "sha256": sha(keys), "bytes": keys.stat().st_size}
    registry["collections"]["newseye"]["match_keys"]["page"] = {"name": "doc_key", "values_file": pin}
    write(path, registry)
    verify_collections(path)
    keys.write_text(json.dumps(["p1", "p2"]), encoding="utf-8")
    with pytest.raises(CollectionRegistryError, match="hash changed|size changed"):
        verify_collections(path)


# --- eligibility ------------------------------------------------------------------------

def reg(**overrides):
    """A registry where locked collections have an (empty) page key set so they can clear a document."""
    collections = {name: {"status": "exposed", "match_keys": no_keys()} for name in REQUIRED_COLLECTIONS}
    for name in ("gallica_e13", "e22_bench"):
        collections[name]["status"] = "test_used"
    for name in ("gallica_e31_reserve", "presse_e7_test"):
        collections[name]["status"] = "reserved_test"
    for name in ("gallica_e13", "e22_bench", "gallica_e31_reserve", "presse_e7_test"):
        collections[name]["match_keys"] = {**no_keys(), "page": {"name": "doc_key", "values": []}}
    for name, spec in overrides.items():
        collections[name].update(spec)
    return {"collections": collections}


def keyed(status, page=(), issue=(), title=()):
    def level(values):
        return {"name": "k", "values": list(values)} if values else NO_KEY
    return {"status": status, "match_keys": {"page": level(page), "issue": level(issue), "title": level(title)}}


# --- pinned registry: the sealed list of presse_e7_test is read from its pinned file (SEAL_NOT_PINNED otherwise) ----------

_SEALED = complete_registry()["collections"]["presse_e7_test"]
PIN_PAGES = sorted(_key_values(_SEALED["match_keys"]["page"]))
HEAD_PAGES = set(_key_values(_SEALED["headline_page_keys"]))
S1 = next(k for k in PIN_PAGES if k not in HEAD_PAGES)             # a sealed, non-headline page
D1 = next(k for k in PIN_PAGES if k not in HEAD_PAGES and k != S1)
H1 = next(k for k in PIN_PAGES if k in HEAD_PAGES)                 # a sealed headline page
I1 = sorted(_key_values(_SEALED["match_keys"]["issue"]))[0]         # a real issue key of the sealed collection


def pinned(**overrides):
    """Neutral baseline (reg) for every collection, so that one defence is tested alone; the sealed entry presse_e7_test is the
    one of the complete pinned registry (fixture), with its pinned page and headline files. The real nzz_impresso entry has no
    usable key and would refuse every page: it is not used here."""
    assert "presse_e7_test" not in overrides, "the sealed lists are read from the pinned files, not written in memory"
    registry = reg()
    registry["collections"]["presse_e7_test"] = copy.deepcopy(_SEALED)
    for name, spec in overrides.items():
        registry["collections"][name].update(spec)
    return registry


def test_test_used_is_refused_for_train_and_dev():
    registry = reg(gallica_e13=keyed("test_used", page=["p1"]))
    for role in ("train", "dev"):
        result = eligibility(registry, role, page_key="p1")
        assert not result["allowed"] and result["matched_collections"] == ["gallica_e13"]
        assert "forbidden for" in result["refusals"][0]


def test_exposed_is_refused_for_test_but_allowed_for_train_and_dev():
    registry = reg(newseye=keyed("exposed_not_auditable", page=["p1"]))
    assert not eligibility(registry, "test", page_key="p1")["allowed"]
    for role in ("train", "dev"):
        assert eligibility(registry, role, page_key="p1")["allowed"]


def test_a_document_in_two_collections_returns_both_and_the_strictest_applies():
    registry = reg(newseye=keyed("exposed", page=["p1"]), gallica_e13=keyed("test_used", page=["p1"]))
    result = eligibility(registry, "train", page_key="p1")
    assert result["matched_collections"] == ["gallica_e13", "newseye"]
    assert not result["allowed"]  # test_used wins over exposed
    # a document matching only an exposed collection stays allowed for train
    assert eligibility(reg(newseye=keyed("exposed", page=["p1"])), "train", page_key="p1")["allowed"]


def test_only_a_sealed_test_makes_a_page_eligible_for_the_test():
    registry = pinned()
    assert eligibility(registry, "test", page_key=S1, issue_key="i-new")["allowed"]  # a source document is required (rule 5)
    assert not eligibility(registry, "test", page_key="other", issue_key="i-new")["allowed"]
    for role in ("train", "dev"):
        assert not eligibility(registry, role, page_key=S1)["allowed"]


def test_a_sealed_page_that_is_also_in_an_exposed_collection_is_refused_for_the_test():
    registry = reg(presse_e7_test=keyed("sealed_test", page=["s1"]), newseye=keyed("exposed", page=["s1"]))
    result = eligibility(registry, "test", page_key="s1")
    assert not result["allowed"] and result["matched_collections"] == ["newseye", "presse_e7_test"]


def test_reserved_test_refuses_train_dev_and_test():
    registry = reg(gallica_e31_reserve=keyed("reserved_test", page=["e31-1"]))
    for role in ("train", "dev", "test"):
        assert not eligibility(registry, role, page_key="e31-1")["allowed"]


def test_title_level_match_is_reported_but_does_not_refuse():
    registry = pinned(newseye=keyed("exposed", title=["mercure"]))
    result = eligibility(registry, "test", page_key=S1, issue_key="i-new", title_key="mercure")
    assert result["allowed"] and result["title_seen"]


def test_issue_level_match_refuses_the_test():
    registry = reg(presse_e7_test=keyed("sealed_test", page=["s1"]), newseye=keyed("exposed", issue=["i7"]))
    assert not eligibility(registry, "test", page_key="s1", issue_key="i7")["allowed"]


def test_locked_collection_without_a_usable_key_gives_a_cautious_refusal():
    registry = reg(gallica_e13={"status": "test_used", "match_keys": no_keys()})
    result = eligibility(registry, "train", page_key="p1")
    assert not result["allowed"] and "cautious refusal" in result["refusals"][0]
    # a key at another level cannot clear a document that was queried by page only
    registry = reg(gallica_e13=keyed("test_used", issue=["i1"]))
    assert not eligibility(registry, "train", page_key="p1")["allowed"]
    assert eligibility(registry, "train", page_key="p1", issue_key="i2")["allowed"]


def test_collection_without_keys_does_not_refuse_train_when_it_is_only_exposed():
    registry = reg(newseye={"status": "exposed", "match_keys": no_keys()})
    assert eligibility(registry, "train", page_key="p1")["allowed"]


def test_disjoint_and_not_exposed_collections_never_make_a_page_test_eligible():
    registry = reg(mille_feuilles_sources=keyed("disjoint_by_construction", page=["m1"]))
    assert eligibility(registry, "train", page_key="m1")["allowed"]
    assert not eligibility(registry, "test", page_key="m1")["allowed"]
    registry = reg(mille_feuilles_sources=keyed("not_exposed", page=["m1"]))
    assert not eligibility(registry, "test", page_key="m1")["allowed"]


def test_eligibility_needs_a_known_role_and_a_key():
    registry = reg()
    with pytest.raises(CollectionRegistryError, match="unknown role"):
        eligibility(registry, "prod", page_key="p1")
    with pytest.raises(CollectionRegistryError, match="at least one key"):
        eligibility(registry, "train")


# --- coverage ---------------------------------------------------------------------------

def coverage_fixture(tmp_path):
    corpus, runs, prepared = tmp_path / "corpus-vt", tmp_path / "runs", tmp_path / "prepared"
    for d in (corpus / "a", runs / "r1", prepared):
        d.mkdir(parents=True)
    (prepared / "x.jsonl").write_text("{}\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"collections": {"newseye": {}, "bnl_pack": {}}}), encoding="utf-8")
    entries = {
        "corpus-vt/a": {"collections": ["newseye"], "method": "explicit_table", "source_files": ["nom du dossier a"]},
        "axel/runs/r1": {"collections": [], "method": "no_page_data_proof",
                         "no_page_data": {"command": "os.walk(r1)", "result_empty": True}},
        "axel/data/prepared/x.jsonl": {"collections": ["bnl_pack"], "method": "file_names", "source_files": ["fichier x.jsonl"]},
    }
    coverage = {"schema_version": COVERAGE_SCHEMA,
                "roots": {"corpus_vt": tmp_ref(corpus, tmp_path), "axel_runs": tmp_ref(runs, tmp_path),
                          "axel_prepared": tmp_ref(prepared, tmp_path)},
                "entries": entries}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage), encoding="utf-8")
    return path, registry, coverage, corpus, runs


def test_complete_coverage_is_accepted(tmp_path):
    path, registry, _, _, _ = coverage_fixture(tmp_path)
    assert verify_coverage(path, registry) == {"entries": 3, "classified": 2, "no_page_data": 1, "unclassified": 0}


def test_a_new_entry_on_disk_that_is_not_classified_fails(tmp_path):
    path, registry, _, corpus, runs = coverage_fixture(tmp_path)
    (runs / "r2_new").mkdir()
    with pytest.raises(CollectionRegistryError, match="entries on disk not classified.*r2_new"):
        verify_coverage(path, registry)
    (runs / "r2_new").rmdir()
    (corpus / "b_new").mkdir()
    with pytest.raises(CollectionRegistryError, match="not classified.*b_new"):
        verify_coverage(path, registry)


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e["corpus-vt/a"].update(unclassified=True), "unclassified entry"),
    (lambda e: e["corpus-vt/a"].update(collections=["mystery"]), "unknown collection"),
    (lambda e: e["corpus-vt/a"].update(source_files=[]), "cites no source"),
    (lambda e: e["axel/runs/r1"].update(no_page_data={"command": "x", "result_empty": False}), "needs a recorded proof"),
    (lambda e: e["axel/runs/r1"].update(no_page_data=None), "needs a recorded proof"),
    (lambda e: e["corpus-vt/a"].update(source_files=["axel/docs/does_not_exist_zzz.md"]), "cited source missing"),
])
def test_coverage_violations_are_refused(tmp_path, mutate, match):
    path, registry, coverage, _, _ = coverage_fixture(tmp_path)
    mutate(coverage["entries"])
    path.write_text(json.dumps(coverage), encoding="utf-8")
    with pytest.raises(CollectionRegistryError, match=match):
        verify_coverage(path, registry)


# --- step B additions: reasons, sealed manifest, complete title sets ---------------------

def test_a_missing_key_level_needs_a_written_reason(tmp_path):
    path, registry, _, _ = build(tmp_path)
    registry["collections"]["newseye"]["match_keys_reasons"].pop("issue")
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="needs a written reason"):
        verify_collections(path)
    registry["collections"]["newseye"]["match_keys_reasons"]["issue"] = "   "
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="needs a written reason"):
        verify_collections(path)


def test_sealed_test_needs_a_pinned_manifest_and_page_keys(tmp_path):
    path, registry, manifest, _ = build(tmp_path)
    entry = registry["collections"]["presse_e7_test"]
    entry["status"] = "sealed_test"
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="sealed_test needs a pinned sealed_manifest"):
        verify_collections(path)
    entry["sealed_manifest"] = {"path": tmp_ref(manifest, tmp_path), "sha256": sha(manifest), "bytes": manifest.stat().st_size}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="needs page-level keys"):
        verify_collections(path)
    entry["match_keys"]["page"] = {"name": "scan_page_key", "values": ["s1"]}
    write(path, registry)
    assert verify_collections(path)["by_status"]["sealed_test"] == 1


def test_complete_is_only_valid_on_a_fully_keyed_title_level(tmp_path):
    path, registry, _, _ = build(tmp_path)
    good = {"name": "title", "values": ["neue zurcher zeitung"], "coverage": {"keyed": 167, "total": 167}, "complete": True}
    registry["collections"]["nzz_impresso"]["match_keys"]["title"] = good
    write(path, registry)
    verify_collections(path)
    registry["collections"]["nzz_impresso"]["match_keys"]["title"] = {**good, "coverage": {"keyed": 100, "total": 167}}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="complete title set must key every unit"):
        verify_collections(path)
    registry["collections"]["nzz_impresso"]["match_keys"]["title"] = good
    registry["collections"]["nzz_impresso"]["match_keys"]["issue"] = {**good, "name": "issue"}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="only valid as true on the title level"):
        verify_collections(path)


def test_a_complete_title_set_lets_a_locked_collection_clear_a_document():
    nzz = keyed("test_used", title=["neue zurcher zeitung"])
    nzz["match_keys"]["title"]["complete"] = True
    registry = reg(gallica_e13=nzz)
    assert eligibility(registry, "train", page_key="p1", title_key="le temps")["allowed"]
    hit = eligibility(registry, "train", page_key="p1", title_key="neue zurcher zeitung")
    assert hit["allowed"] and hit["title_seen"]  # a title match alone never refuses
    # without the declaration the same collection cannot clear anything: cautious refusal
    incomplete = reg(gallica_e13=keyed("test_used", title=["neue zurcher zeitung"]))
    assert not eligibility(incomplete, "train", page_key="p1", title_key="le temps")["allowed"]
    # a complete title set cannot clear a query that supplies no title
    assert not eligibility(registry, "train", page_key="p1")["allowed"]


def test_declared_keys_lists_the_key_name_or_the_phrase_for_every_level(tmp_path):
    path, registry, _, _ = build(tmp_path)
    registry["collections"]["kb_impact"]["match_keys"]["page"] = {"name": "doc_key", "values": ["kb:1"]}
    write(path, registry)
    from babaorum.g0_collections import declared_keys
    table = declared_keys(registry)
    assert table["kb_impact"] == {"page": "doc_key", "issue": NO_KEY, "title": NO_KEY}
    assert set(table) == set(REQUIRED_COLLECTIONS)


def test_a_sibling_of_the_sealed_test_locks_train_and_dev_but_not_the_sealed_pages_themselves():
    refused = {**keyed("reserved_test", page=["u1"], issue=[I1]), "sibling_of": "presse_e7_test"}
    registry = pinned(presse_e7_test_refused=refused)
    assert eligibility(registry, "test", page_key=S1, issue_key=I1)["allowed"]
    for role in ("train", "dev"):
        assert not eligibility(registry, role, page_key="x", issue_key=I1)["allowed"]  # same document as a locked page
    # without the declaration the unsealed pages of the same document would block the sealed page
    plain = pinned(presse_e7_test_refused=keyed("reserved_test", page=["u1"], issue=[I1]))
    assert not eligibility(plain, "test", page_key=S1, issue_key=I1)["allowed"]
    # an unrelated reserved test still blocks
    other = pinned(gallica_e31_reserve=keyed("reserved_test", issue=[I1]))
    assert not eligibility(other, "test", page_key=S1, issue_key=I1)["allowed"]


def test_sibling_of_must_point_to_a_sealed_test(tmp_path):
    path, registry, manifest, _ = build(tmp_path)
    registry["collections"]["presse_e7_test_refused"].update(status="reserved_test", sibling_of="presse_e7_test")
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="sibling_of must point to a sealed_test"):
        verify_collections(path)


# --- LOT-G0-02: sibling exception, complete titles, test_descriptive -----------------------------

def sealed_and_sibling(sibling_pages=("u1",)):
    return pinned(presse_e7_test_refused={**keyed("reserved_test", page=list(sibling_pages), issue=[I1]),
                                          "sibling_of": "presse_e7_test"})


def test_the_sibling_exception_is_limited_to_the_issue_level_and_is_labelled():
    result = eligibility(sealed_and_sibling(), "test", page_key=S1, issue_key=I1)
    assert result["allowed"] and [e["label"] for e in result["sibling_exceptions"]] == ["sibling_exception"]
    conflict = eligibility(sealed_and_sibling(sibling_pages=(S1,)), "test", page_key=S1, issue_key=I1)
    assert not conflict["allowed"] and "conflict" in conflict["refusals"][0]


@pytest.mark.parametrize("status", ["exposed", "exposed_not_auditable", "test_used", "reserved_test"])
def test_a_title_equal_to_a_complete_title_set_is_refused_for_every_status_that_can_hold_it(status):
    holder = keyed(status, title=["neue zurcher zeitung"])
    holder["match_keys"]["title"]["complete"] = True
    registry = reg(presse_e7_test=keyed("sealed_test", page=["s1"]), nzz_impresso=holder)
    result = eligibility(registry, "test", page_key="s1", issue_key="i-new", title_key="neue zurcher zeitung")
    assert not result["allowed"] and any("title set is complete" in r for r in result["refusals"])


def test_test_descriptive_admits_only_a_sealed_page_that_is_not_headline():
    registry = pinned()
    assert eligibility(registry, "test_descriptive", page_key=D1)["allowed"]  # no source document needed
    assert not eligibility(registry, "test_descriptive", page_key=H1)["allowed"]  # headline: role test
    assert not eligibility(registry, "test", page_key=D1)["allowed"]  # no source document: not a test page
    assert not eligibility(registry, "test_descriptive", page_key="outside")["allowed"]
    forged = pinned()
    forged["collections"]["presse_e7_test"]["headline_page_keys"] = {"name": "k", "values": [H1]}  # an in-memory headline list
    result = eligibility(forged, "test_descriptive", page_key=D1)
    assert not result["allowed"] and "HEADLINE_LIST_NOT_PINNED:presse_e7_test" in result["refusal_codes"]
    # a sealed list without any headline list cannot grant test_descriptive
    assert not eligibility(reg(presse_e7_test=keyed("sealed_test", page=["d1"])), "test_descriptive", page_key="d1")["allowed"]


def test_a_pinned_registry_without_headline_list_refuses_test_descriptive_with_both_codes():
    # the pin check (HEADLINE_LIST_NOT_PINNED) runs first, then the rule itself (NO_HEADLINE_LIST), both on the same entry
    registry = pinned()
    del registry["collections"]["presse_e7_test"]["headline_page_keys"]
    result = eligibility(registry, "test_descriptive", page_key=D1)
    assert not result["allowed"]
    assert result["refusal_codes"] == ["HEADLINE_LIST_NOT_PINNED:presse_e7_test", "NO_HEADLINE_LIST:presse_e7_test"]


def test_an_unpinned_sealed_list_is_refused_with_seal_not_pinned():
    registry = reg(presse_e7_test=keyed("sealed_test", page=["s1"]))
    result = eligibility(registry, "test", page_key="s1", issue_key="i-new")
    assert not result["allowed"] and result["refusal_codes"] == ["SEAL_NOT_PINNED:presse_e7_test"]


def test_sibling_page_sets_must_be_disjoint_in_the_registry(tmp_path):
    path, registry, _, _ = build(tmp_path)
    registry["collections"]["presse_e7_test"].update(status="sealed_test", sealed_manifest=registry["collections"]["newseye"]["manifests"][0])
    registry["collections"]["presse_e7_test"]["match_keys"]["page"] = {"name": "k", "values": ["p1", "p2"]}
    registry["collections"]["presse_e7_test_refused"].update(status="reserved_test", sibling_of="presse_e7_test")
    registry["collections"]["presse_e7_test_refused"]["match_keys"]["page"] = {"name": "k", "values": ["p2", "p3"]}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="must be disjoint"):
        verify_collections(path)
    registry["collections"]["presse_e7_test_refused"]["match_keys"]["page"] = {"name": "k", "values": ["p3"]}
    write(path, registry)
    verify_collections(path)


def test_v4_registries_need_a_reasoned_gallica_scope(tmp_path):
    path, registry, _, _ = build(tmp_path)
    registry["schema_version"] = "babaorum.g0_collections.v4"
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="gallica_scope"):
        verify_collections(path)
    for name, entry in registry["collections"].items():
        entry["gallica_scope"] = {"value": "possible", "reason": "test"}
    registry["collections"]["mccatmus"]["status"] = "exposed_not_auditable"
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="not_auditable is for models only"):
        verify_collections(path)
    registry["collections"]["mccatmus"]["gallica_scope"] = {"value": "not_auditable", "reason": "modèle"}
    registry["collections"]["bnl_pack"]["gallica_scope"] = {"value": "excluded", "reason": "   "}
    write(path, registry)
    with pytest.raises(CollectionRegistryError, match="written reason"):
        verify_collections(path)
