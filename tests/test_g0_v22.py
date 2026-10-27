"""G0 v2.2 : chemins relatifs à des racines déclarées (décision p1 / pA, 22/10).

Les racines viennent de G0_<NOM>_ROOT ou de config/g0_roots.local.json (non commité), jamais de valeurs écrites ici.
Les preuves d'équivalence comparent aux anciens fichiers à chemins absolus (v2.1 et scellé v2) : elles ne tournent que là où ces
fichiers et les racines existent, et le résultat est archivé dans la passation. Le reste tourne partout.
"""
import json
import re
from pathlib import Path

import pytest

from babaorum.g0_collections import ROOT_NAMES, CollectionRegistryError, declared_root, resolve_path

ROOT = Path(__file__).resolve().parents[1]
OLD_REG = ROOT / "config" / "g0_collections_v2_1.json"
NEW_REG = ROOT / "config" / "g0_collections_v2_2.json"
OLD_SEAL = ROOT / "config" / "sealed" / "presse_e7_test_v2.json"
NEW_SEAL = ROOT / "config" / "sealed" / "presse_e7_test_v2_1.json"


def roots_declared() -> bool:
    try:
        return all(declared_root(name) for name in ROOT_NAMES)
    except CollectionRegistryError:
        return False


needs_roots = pytest.mark.skipif(not roots_declared(), reason="racines G0_* non déclarées (ni variable d'environnement, ni g0_roots.local.json)")
needs_old_registry = pytest.mark.skipif(not OLD_REG.exists(), reason="g0_collections_v2_1.json absent : équivalence non vérifiable")
needs_old_seal = pytest.mark.skipif(not OLD_SEAL.exists(), reason="scellé v2 absent (non commité) : équivalence non vérifiable ici")


def unref(obj):
    """Replace every {root, path} reference by its absolute string, through the loader and the declared roots."""
    if isinstance(obj, dict):
        if set(obj) == {"root", "path"}:
            return str(resolve_path(obj))
        return {k: unref(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unref(v) for v in obj]
    return obj


@needs_roots
@needs_old_registry
def test_the_registry_differs_from_v21_only_in_the_paths():
    old = json.loads(OLD_REG.read_text(encoding="utf-8"))
    new = unref(json.loads(NEW_REG.read_text(encoding="utf-8")))
    # the sealed manifest points to the sealed file v2.1: its name, sha256 and size change, nothing else
    old_sealed = old["collections"]["presse_e7_test"]["sealed_manifest"]
    new_sealed = new["collections"]["presse_e7_test"]["sealed_manifest"]
    assert new_sealed["path"] == old_sealed["path"].replace("presse_e7_test_v2.json", "presse_e7_test_v2_1.json")
    assert new_sealed.keys() == old_sealed.keys()
    new["collections"]["presse_e7_test"]["sealed_manifest"] = old_sealed
    # the dated avenant AV-2026-10-22-NZZ-GT is the ONLY other difference: it is checked here, then undone
    avenants = new.pop("avenants")
    assert [a["id"] for a in avenants] == ["AV-2026-10-22-NZZ-GT", "AV-2026-10-24-REVIEWS-ROOT"]
    assert avenants[1]["changes"] == [] and avenants[1]["effect"].startswith("Aucun chemin")   # reviews_root changes no path
    avenant = [avenants[0]]
    tree = new["collections"]["nzz_impresso"]["trees"][0]
    assert tree.pop("avenant") == "AV-2026-10-22-NZZ-GT"
    assert {"file_count": tree["file_count"], "tree_sha256_names_and_sizes": tree["tree_sha256_names_and_sizes"]} == avenant[0]["new"]
    tree.update(avenant[0]["previous"])
    for key in old:
        if key != "collections":
            assert new[key] == old[key], key
    assert new["collections"].keys() == old["collections"].keys()
    for name, entry in old["collections"].items():
        assert new["collections"][name] == entry, name


@needs_roots
@needs_old_seal
def test_the_sealed_file_differs_from_v2_only_in_the_paths():
    old = json.loads(OLD_SEAL.read_text(encoding="utf-8"))
    new = json.loads(NEW_SEAL.read_text(encoding="utf-8"))
    assert unref(new) == old
    old_pages = {p["page_id"]: p for p in old["pages"]}
    new_pages = {p["page_id"]: p for p in new["pages"]}
    assert len(old_pages) == len(new_pages) == 410 and old_pages.keys() == new_pages.keys()
    for pid, p in old_pages.items():
        for key in ("page_id", "scan_page_key", "issue_keys", "title_key", "truth_lines", "truth_sha256"):
            assert new_pages[pid][key] == p[key], (pid, key)


def test_an_undeclared_root_is_refused_with_its_name(monkeypatch):
    monkeypatch.setattr("babaorum.g0_collections.LOCAL_ROOTS_FILE", ROOT / "config" / "no_such_local_file.json")
    monkeypatch.delenv("G0_AXEL_ROOT", raising=False)
    with pytest.raises(CollectionRegistryError) as exc:
        resolve_path({"root": "axel_root", "path": "runs/e1"})
    assert exc.value.code == "ROOT_UNDECLARED:axel_root"
    with pytest.raises(CollectionRegistryError) as exc:
        resolve_path({"root": "inconnue", "path": "x"})
    assert exc.value.code == "ROOT_UNDECLARED:inconnue"


@pytest.mark.parametrize("ref", [
    "/tmp/x",                                       # a bare absolute string
    {"root": "repo", "path": "/etc/passwd"},       # an absolute path inside a reference
    {"root": "repo", "path": "../secret.json"},     # '..'
    {"root": "repo", "path": "a/../../b"},
])
def test_a_path_that_is_not_relative_is_refused(ref, monkeypatch):
    monkeypatch.setattr("babaorum.g0_collections.LOCAL_ROOTS_FILE", ROOT / "config" / "no_such_local_file.json")
    with pytest.raises(CollectionRegistryError) as exc:
        resolve_path(ref)
    assert exc.value.code == "PATH_NOT_RELATIVE"


def test_the_new_files_contain_no_absolute_path():
    for path in (NEW_REG, NEW_SEAL, ROOT / "config" / "g0_coverage_v2_2.json"):
        assert not re.search(r'"/(Users|private|etc|tmp|home|var)', path.read_text(encoding="utf-8")), path.name


# --- audit pQ (22/10) : aucun accès hors des racines déclarées ---------------------------------

def _empty_coverage(tmp_path, cited):
    """A coverage whose only entry cites `cited`; the three roots are empty, so no live entry is missing."""
    import sys
    from babaorum.g0_collections import COVERAGE_SCHEMA
    corpus, runs, prepared = tmp_path / "corpus", tmp_path / "runs", tmp_path / "prepared"
    for d in (corpus, runs, prepared):
        d.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"collections": {"newseye": {}}}), encoding="utf-8")
    coverage = {"schema_version": COVERAGE_SCHEMA, "roots": {
        "corpus_vt": {"root": "corpus_root", "path": "corpus"},
        "axel_runs": {"root": "corpus_root", "path": "runs"},
        "axel_prepared": {"root": "corpus_root", "path": "prepared"}},
        "entries": {"corpus-vt/a": {"collections": ["newseye"], "method": "explicit_table", "source_files": [cited]}}}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage), encoding="utf-8")
    return path, registry


@pytest.mark.parametrize("cited", ["axel//etc/passwd", "axel/../secret", "corpus-vt/../../secret"])
def test_a_cited_source_that_leaves_its_root_is_refused(tmp_path, monkeypatch, cited):
    from babaorum.g0_collections import verify_coverage
    monkeypatch.setenv("G0_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("G0_AXEL_ROOT", str(tmp_path / "axel"))
    path, registry = _empty_coverage(tmp_path, cited)
    with pytest.raises(CollectionRegistryError) as exc:
        verify_coverage(path, registry)
    assert exc.value.code == "PATH_NOT_RELATIVE"


def test_a_cited_source_inside_its_root_is_checked_there(tmp_path, monkeypatch):
    from babaorum.g0_collections import verify_coverage
    monkeypatch.setenv("G0_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("G0_AXEL_ROOT", str(tmp_path / "axel"))
    (tmp_path / "axel").mkdir()
    path, registry = _empty_coverage(tmp_path, "axel/present.txt")
    with pytest.raises(CollectionRegistryError) as exc:
        verify_coverage(path, registry)
    assert exc.value.code == "COVERAGE_SOURCE_MISSING"           # the root is used, and the file is not there
    (tmp_path / "axel" / "present.txt").write_text("x", encoding="utf-8")
    assert verify_coverage(path, registry)["classified"] == 1


def test_the_key_recomputation_never_reads_the_home_directory(tmp_path, monkeypatch):
    from pathlib import Path as _Path
    from babaorum import g0_keys

    def no_home(*a, **k):
        raise RuntimeError("home accessed")
    monkeypatch.setattr(_Path, "home", staticmethod(no_home))
    monkeypatch.setenv("G0_AXEL_ROOT", str(tmp_path / "axel"))
    monkeypatch.setenv("G0_CORPUS_ROOT", str(tmp_path))
    with pytest.raises(BaseException) as exc:
        g0_keys.compute_keys()
    assert "home accessed" not in str(exc.value)                    # it reads the declared roots, which are empty here


def test_the_key_recomputation_refuses_an_undeclared_root(monkeypatch):
    from babaorum import g0_keys
    monkeypatch.setattr("babaorum.g0_collections.LOCAL_ROOTS_FILE", ROOT / "config" / "no_such_local_file.json")
    monkeypatch.delenv("G0_AXEL_ROOT", raising=False)
    with pytest.raises(CollectionRegistryError) as exc:
        g0_keys.compute_keys()
    assert exc.value.code == "ROOT_UNDECLARED:axel_root"


# --- avenant AV-2026-10-22-NZZ-GT : la vérification complète passe avec la nouvelle pin et échoue avec l'ancienne -----------

def _nzz_pins():
    reg = json.loads(NEW_REG.read_text(encoding="utf-8"))
    avenant = reg["avenants"][0]
    return avenant["previous"], avenant["new"]


@needs_roots
def test_the_full_verification_passes_with_the_avenant_pin():
    from babaorum.g0_collections import verify_collections
    assert verify_collections(NEW_REG)["collections"] > 0


@needs_roots
def test_the_full_verification_fails_with_the_previous_pin(tmp_path):
    from babaorum.g0_collections import CollectionRegistryError, verify_collections
    previous, _ = _nzz_pins()
    reg = json.loads(NEW_REG.read_text(encoding="utf-8"))
    tree = reg["collections"]["nzz_impresso"]["trees"][0]
    tree.update(previous)
    old = tmp_path / "registry_with_previous_nzz_pin.json"
    old.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CollectionRegistryError) as exc:
        verify_collections(old)
    assert exc.value.code == "TREE_CHANGED" and "nzz_impresso" in str(exc.value)


# --- avenant AV-2026-10-24-REVIEWS-ROOT : racine reviews_root, déclarée hors dépôt, refus sans repli ---------------------------

def test_reviews_root_refuses_when_undeclared(monkeypatch):
    from babaorum import g0_collections
    monkeypatch.setattr("babaorum.g0_collections.LOCAL_ROOTS_FILE", ROOT / "config" / "no_such_local_file.json")
    monkeypatch.delenv("G0_REVIEWS_ROOT", raising=False)
    with pytest.raises(CollectionRegistryError) as exc:
        resolve_path({"root": "reviews_root", "path": "rappel_pages"})
    assert exc.value.code == "ROOT_UNDECLARED:reviews_root"


def test_reviews_root_resolves_from_the_environment_and_refuses_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("G0_REVIEWS_ROOT", str(tmp_path))
    assert resolve_path({"root": "reviews_root", "path": "rappel_pages/tirage.json"}) == tmp_path / "rappel_pages/tirage.json"
    with pytest.raises(CollectionRegistryError) as exc:
        resolve_path({"root": "reviews_root", "path": "/etc/passwd"})
    assert exc.value.code == "PATH_NOT_RELATIVE"
