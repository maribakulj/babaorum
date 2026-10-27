"""loop.py calls the G0 eligibility function and checks coverage (synthetic data only)."""
import importlib.util
import json
import subprocess
from pathlib import Path

from babaorum.evidence import atomic_write
from babaorum.g0_collections import NO_KEY

ROOT = Path(__file__).resolve().parents[1]


def load_loop():
    spec = importlib.util.spec_from_file_location("loop_g0_gate_test", ROOT / "scripts/loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry(sealed_pages=("s1",)):
    keys = {"page": {"name": "doc_key", "values": list(sealed_pages)}, "issue": NO_KEY, "title": NO_KEY}
    empty = {"page": {"name": "doc_key", "values": []}, "issue": NO_KEY, "title": NO_KEY}
    return {"collections": {
        "presse_e7_test": {"status": "sealed_test", "match_keys": keys},
        "gallica_e13": {"status": "test_used", "match_keys": empty},
        "newseye": {"status": "exposed", "match_keys": {"page": NO_KEY, "issue": NO_KEY, "title": NO_KEY}},
    }}


def prepare(tmp_path, monkeypatch, reg=None):
    module = load_loop()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "inventory", lambda axel: {"free_bytes": 0})
    monkeypatch.setattr(module, "verify_port", lambda: {})
    monkeypatch.setattr(module, "g0_receipt_current", lambda *args: True)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""))
    atomic_write(tmp_path / "reports/g0_inventory.json", {"metadata_current": True, "counts": {}})
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config/g0_collections_v2_2.json").write_text(json.dumps(reg or registry()), encoding="utf-8")
    calls = []
    real = module.eligibility

    def spy(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "eligibility", spy)
    return module, calls


def manifest(tmp_path, pages, name="train.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return path


def test_a_sealed_page_inserted_in_a_train_manifest_fails_the_loop_and_proves_eligibility_is_called(tmp_path, monkeypatch):
    module, calls = prepare(tmp_path, monkeypatch)
    train = manifest(tmp_path, [{"page_key": "free-1"}, {"page_key": "s1"}])
    result = module.cycle(Path("/Users/marcel/axel"), False, [f"train:{train}"])
    assert calls == ["train", "train"]  # the loop really calls eligibility for every page
    assert result["checks_passed"] is False
    assert result["gates"]["G0"]["status"] == "manifest_refused"
    check = result["g0_manifests"]
    assert check["ok"] is False and check["checked"] == 2 and check["refused_count"] == 1
    assert check["refused"][0]["page"] == {"page_key": "s1"}
    assert "forbidden for train" in check["refused"][0]["refusals"][0]


def test_a_clean_train_manifest_passes_the_manifest_gate(tmp_path, monkeypatch):
    module, calls = prepare(tmp_path, monkeypatch)
    train = manifest(tmp_path, [{"page_key": "free-1"}, {"page_key": "free-2"}])
    result = module.cycle(Path("/Users/marcel/axel"), False, [f"train:{train}"])
    assert calls == ["train", "train"]
    assert result["g0_manifests"] == {"ok": True, "checked": 2, "refused_count": 0, "refused": []}
    assert result["gates"]["G0"]["status"] != "manifest_refused"


def test_without_manifests_nothing_is_checked(tmp_path, monkeypatch):
    module, calls = prepare(tmp_path, monkeypatch)
    result = module.cycle(Path("/Users/marcel/axel"), False)
    assert result["g0_manifests"] is None and calls == []


def test_unreadable_manifest_fails_closed(tmp_path, monkeypatch):
    module, _ = prepare(tmp_path, monkeypatch)
    result = module.cycle(Path("/Users/marcel/axel"), False, [f"train:{tmp_path / 'absent.json'}"])
    assert result["g0_manifests"]["ok"] is False and "error" in result["g0_manifests"]
    assert result["checks_passed"] is False
    wrong_role = manifest(tmp_path, [{"page_key": "x"}], "wrong.json")
    result = module.cycle(Path("/Users/marcel/axel"), False, [f"prod:{wrong_role}"])
    assert result["g0_manifests"]["ok"] is False


def test_missing_coverage_file_is_reported_and_fails_the_gate(tmp_path, monkeypatch):
    module, _ = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "verify_collections", lambda path: {"collections": 1})
    result = module.cycle(Path("/Users/marcel/axel"), False)
    assert result["g0_coverage"]["verified"] is False
    assert result["gates"]["G0"]["status"] == "coverage_invalid"


def seal_file(tmp_path):
    (tmp_path / "config/sealed").mkdir(parents=True, exist_ok=True)
    pages = [{"page_id": "a", "headline": True}, {"page_id": "b", "headline": False}, {"page_id": "c", "headline": True}]
    (tmp_path / "config/sealed/presse_e7_test_v2_1.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")


def test_a_headline_measure_with_a_headline_false_page_fails_the_loop(tmp_path, monkeypatch):
    module, _ = prepare(tmp_path, monkeypatch)
    seal_file(tmp_path)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"pages": ["a", "c"]}), encoding="utf-8")
    ok = module.cycle(Path("/Users/marcel/axel"), False, (), [str(good)])
    assert ok["g0_headline_measure"] == {"ok": True, "checked": 2}
    assert ok["gates"]["G0"]["status"] != "headline_measure_invalid"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"pages": ["a", "b"]}), encoding="utf-8")
    failed = module.cycle(Path("/Users/marcel/axel"), False, (), [str(bad)])
    assert failed["checks_passed"] is False
    assert failed["gates"]["G0"]["status"] == "headline_measure_invalid"
    assert "not headline=true" in failed["g0_headline_measure"]["error"]


def test_a_headline_measure_without_a_seal_fails_closed(tmp_path, monkeypatch):
    module, _ = prepare(tmp_path, monkeypatch)
    measure = tmp_path / "m.json"
    measure.write_text(json.dumps({"pages": ["a"]}), encoding="utf-8")
    result = module.cycle(Path("/Users/marcel/axel"), False, (), [str(measure)])
    assert result["g0_headline_measure"]["ok"] is False and result["checks_passed"] is False


def test_a_page_admitted_only_as_test_descriptive_cannot_enter_the_measure_that_counts(tmp_path, monkeypatch):
    """LOT-G0-02 rule 5: test_descriptive is for headline=false pages; one of them in a headline measure fails the loop."""
    module, _ = prepare(tmp_path, monkeypatch)
    seal_file(tmp_path)  # page "b" is headline=false, the only kind test_descriptive may admit
    measure = tmp_path / "counting.json"
    measure.write_text(json.dumps({"pages": ["a", "b", "c"]}), encoding="utf-8")
    result = module.cycle(Path("/Users/marcel/axel"), False, (), [str(measure)])
    assert result["checks_passed"] is False and result["gates"]["G0"]["status"] == "headline_measure_invalid"
