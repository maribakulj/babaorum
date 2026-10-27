import hashlib
import json
import copy
from pathlib import Path

import pytest

from babaorum.comparison import paired_intervals, compare
from babaorum.contracts import ContractError


def row(page, document, support, ref, candidate):
    return {
        "page_key": page,
        "document_key": document,
        "support": support,
        "errors_reference": ref,
        "errors_candidate": candidate,
    }


def test_micro_difference_uses_total_support_not_mean_page_cer():
    rows = [row("p1", "d1", 90, 0, 10), row("p2", "d2", 10, 0, 10)]
    result = paired_intervals(rows, "d150", resamples=100, seed=8)
    assert result["page"]["delta_candidate_minus_reference_pp"] == pytest.approx(20.0)
    assert result["page"]["delta_candidate_minus_reference_pp"] != pytest.approx((100 * 10 / 90 + 100) / 2)


def test_document_cluster_keeps_pages_together_and_needs_two_documents():
    result = paired_intervals([
        row("p1", "issue", 50, 2, 1),
        row("p2", "issue", 50, 4, 5),
    ], "raw", resamples=50, seed=3)
    assert result["page"]["clusters"] == 2
    assert result["page"]["interval_pp"] is not None
    assert result["document"]["clusters"] == 1
    assert result["document"]["interval_pp"] is None
    assert result["document"]["status"] == "insufficient_clusters"


def test_swapping_systems_reverses_point_and_interval():
    rows = [row("p1", "d1", 20, 5, 1), row("p2", "d2", 20, 1, 4), row("p3", "d3", 20, 2, 4)]
    a = paired_intervals(rows, "d150", resamples=300, seed=71)
    swapped = [row(r["page_key"], r["document_key"], r["support"], r["errors_candidate"], r["errors_reference"]) for r in rows]
    b = paired_intervals(swapped, "d150", resamples=300, seed=71)
    assert b["page"]["delta_candidate_minus_reference_pp"] == -a["page"]["delta_candidate_minus_reference_pp"]
    assert b["page"]["interval_pp"] == pytest.approx([-a["page"]["interval_pp"][1], -a["page"]["interval_pp"][0]])


def test_bootstrap_is_deterministic_and_reports_quantile_and_draw_counts():
    rows = [row("p1", "d1", 20, 4, 3), row("p2", "d2", 20, 2, 3), row("p3", "d3", 20, 1, 2)]
    a = paired_intervals(rows, "raw", resamples=200, seed=12, confidence=.9)
    b = paired_intervals(rows, "raw", resamples=200, seed=12, confidence=.9)
    assert a == b
    assert a["page"]["resamples_requested"] == 200
    assert a["page"]["resamples_used"] + a["page"]["skipped_zero_support_draws"] == 200
    assert a["page"]["quantile_method"].startswith("linear interpolation")
    assert a["page"]["seed"] == 12


@pytest.mark.parametrize("rows,kwargs", [
    ([row("p", "d", 1, 0, 0), {**row("p", "d", 1, 0, 0)}], {}),
    ([{**row("p", "d", 1, 0, 0), "extra": 1}], {}),
    ([row("p", "d", 0, 1, 0)], {}),
    ([row("p", "d", -1, 0, 0)], {}),
    ([row("p", "d", 1, 0.5, 0)], {}),
    ([row("p", "d", 1, 0, 0)], {"resamples": 0}),
    ([row("p", "d", 1, 0, 0)], {"confidence": 1.0}),
    ([row("p", "d", 1, 0, 0)], {"seed": True}),
])
def test_invalid_statistical_inputs_rejected(rows, kwargs):
    with pytest.raises(ContractError):
        paired_intervals(rows, "d150", **kwargs)


def test_low_level_api_marks_missing_audited_document_as_no_document_ci():
    result = paired_intervals([
        row("gallica:1/f1", None, 10, 1, 2),
        row("gallica:2/f1", "gallica:2", 10, 1, 1),
    ], "d150", resamples=20)
    assert result["page"]["interval_pp"] is not None
    assert result["document"]["status"] == "unknown_document_identity"
    assert result["document"]["interval_pp"] is None


def test_all_zero_support_is_explicit_and_draws_are_skipped():
    result = paired_intervals([row("p1", "d1", 0, 0, 0), row("p2", "d2", 0, 0, 0)], "raw", resamples=20, seed=2)
    assert result["page"]["support_chars"] == 0
    assert result["page"]["delta_candidate_minus_reference_pp"] is None
    assert result["page"]["interval_pp"] is None
    assert result["page"]["resamples_used"] == 0
    assert result["page"]["skipped_zero_support_draws"] == 20
    assert result["page"]["status"] == "no_valid_draws"


def _model(name):
    return {
        "name": name,
        "code_revision": "fixture-revision",
        "checkpoint_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "alphabet_sha256": "3" * 64,
        "license": "fixture",
        "pretraining_audit": "fixture",
    }


def _prediction(manifest, model, rows):
    raw = manifest.read_bytes()
    return {
        "schema_version": "0.1.0",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "model": _model(model),
        "rows": rows,
    }


def _write_predictions(path, manifest_path, model, rows, audit="fixture"):
    prediction = _prediction(manifest_path, model, rows)
    prediction["model"]["pretraining_audit"] = audit
    if audit == "reviewed":
        prediction["model"].update(weights_license="CC-BY-4.0", weights_license_audit_status="reviewed")
    path.write_text(json.dumps(prediction))
    return path


def test_compare_reports_d150_raw_suppression_coverage_and_equal_hash(tmp_path):
    pytest.importorskip("rapidfuzz")
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures" / "line.json"
    reference = _prediction(fixture, "ref", [{"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"}])
    # Same cached output bytes is a valid zero-difference comparison, not a reason to invent a CI.
    prediction_path = tmp_path / "same.json"
    prediction_path.write_text(json.dumps(reference))
    result = compare(fixture, prediction_path, prediction_path, resamples=20, seed=1)
    assert result["scope"] == "READER_ONLY"
    assert result["reference"]["predictions_sha256"] == result["candidate"]["predictions_sha256"]
    assert result["metrics"]["d150"]["delta_candidate_minus_reference_pp"] == 0
    assert result["metrics"]["raw"]["delta_candidate_minus_reference_pp"] == 0
    assert result["metrics"]["d150"]["paired_intervals"]["document"]["interval_pp"] is None

    missing_candidate = _prediction(fixture, "cand", [])
    candidate_path = tmp_path / "missing.json"
    candidate_path.write_text(json.dumps(missing_candidate))
    suppressed = compare(fixture, prediction_path, candidate_path, resamples=20, seed=1)
    metric = suppressed["metrics"]["d150"]
    assert metric["coverage"]["candidate"]["missing"] == 1
    assert metric["coverage"]["candidate"]["coverage"] == 0
    assert metric["errors_candidate"] == len("son")

    failed_candidate = _prediction(fixture, "failed", [{"id": "a", "crop_sha256": "b" * 64, "status": "error", "text": ""}])
    failed_path = tmp_path / "failed.json"
    failed_path.write_text(json.dumps(failed_candidate))
    failed = compare(fixture, prediction_path, failed_path, resamples=20, seed=1)
    failed_metric = failed["metrics"]["d150"]
    assert failed_metric["coverage"]["candidate"]["error"] == 1
    assert failed_metric["coverage"]["candidate"]["coverage"] == 0
    assert failed_metric["errors_candidate"] == len("son")


def test_compare_rejects_prediction_for_different_crop(tmp_path):
    pytest.importorskip("rapidfuzz")
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures" / "line.json"
    good = _prediction(fixture, "good", [{"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"}])
    bad = _prediction(fixture, "bad", [{"id": "a", "crop_sha256": "c" * 64, "status": "ok", "text": "son"}])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(good)); b.write_text(json.dumps(bad))
    with pytest.raises(ContractError, match="different crop"):
        compare(fixture, a, b, resamples=10)


def test_empty_truth_keeps_zero_support_explicit(tmp_path):
    pytest.importorskip("rapidfuzz")
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures" / "empty_truth.json"
    truth = _prediction(fixture, "truth", [{"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": ""}])
    out = tmp_path / "empty.json"
    out.write_text(json.dumps(truth))
    result = compare(fixture, out, out, resamples=10)
    assert result["metrics"]["d150"]["support_chars"] == 0
    assert result["metrics"]["d150"]["delta_candidate_minus_reference_pp"] is None
    assert result["metrics"]["d150"]["paired_intervals"]["page"]["status"] == "insufficient_clusters"


def test_empty_truth_outputs_are_counted_but_do_not_change_legacy_cer(tmp_path):
    pytest.importorskip("rapidfuzz")
    fixture = Path(__file__).parent / "fixtures" / "empty_truth.json"
    reference = _write_predictions(tmp_path / "ref.json", fixture, "ref", [
        {"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "invented"}
    ])
    candidate = _write_predictions(tmp_path / "cand.json", fixture, "cand", [
        {"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": ""}
    ])
    result = compare(fixture, reference, candidate, resamples=20)
    metric = result["metrics"]["d150"]
    assert metric["support_chars"] == 0
    assert metric["errors_reference"] == metric["errors_candidate"] == 0
    assert metric["coverage"]["reference"]["empty_truth_lines"] == 1
    assert metric["coverage"]["reference"]["empty_truth_output_chars"] == len("invented")
    assert metric["coverage"]["candidate"]["empty_truth_lines"] == 1
    assert metric["coverage"]["candidate"]["empty_truth_output_chars"] == 0


def test_forbidden_manifest_is_rejected_before_prediction_paths_are_read(tmp_path):
    manifest = json.loads((Path(__file__).parent / "fixtures" / "line.json").read_text())
    manifest["exposure"] = "sealed_test"
    line = manifest["lines"][0]
    line["split"] = "test"
    line["page_id"] = "gallica:7635415/f2"
    line["document_key"] = "gallica:7635415"
    manifest_path = tmp_path / "forbidden.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="historically exposed"):
        compare(manifest_path, tmp_path / "missing-reference.json", tmp_path / "missing-candidate.json")


def test_real_aliases_cluster_by_c7_issue_and_check_page_issue_consistency(tmp_path, monkeypatch):
    pytest.importorskip("rapidfuzz")
    import babaorum.comparison as comparison

    manifest = json.loads((Path(__file__).parent / "fixtures" / "line.json").read_text())
    manifest["exposure"] = "historical"
    manifest["key_policy"] = "declared_document_v0"
    first = copy.deepcopy(manifest["lines"][0])
    second = copy.deepcopy(first)
    first.update(id="a", page_id="gallica:7635415/f2", document_key="gallica:7635415/f2",
                 image_sha256="a" * 64, crop_sha256="b" * 64)
    second.update(id="b", page_id="bpt6k76354152/f3", document_key="gallica:7635415/f3",
                  image_sha256="c" * 64, crop_sha256="d" * 64, box=[1, 0, 31, 20])
    manifest["lines"] = [first, second]
    manifest_path = tmp_path / "real.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(comparison, "allow_development", lambda _manifest: None)
    reference = _write_predictions(tmp_path / "ref.json", manifest_path, "ref", [
        {"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"},
        {"id": "b", "crop_sha256": "d" * 64, "status": "ok", "text": "son"},
    ], audit="reviewed")
    candidate = _write_predictions(tmp_path / "cand.json", manifest_path, "cand", [
        {"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"},
        {"id": "b", "crop_sha256": "d" * 64, "status": "ok", "text": "son"},
    ], audit="reviewed")
    result = compare(manifest_path, reference, candidate, resamples=20)
    d150 = result["metrics"]["d150"]
    assert {item["page_key"] for item in d150["rows"]} == {"gallica:7635415/f2", "gallica:7635415/f3"}
    assert {item["document_key"] for item in d150["rows"]} == {"gallica:7635415"}
    assert d150["paired_intervals"]["document"]["clusters"] == 1
    assert d150["paired_intervals"]["document"]["interval_pp"] is None

    second["document_key"] = "gallica:7636525/f3"
    manifest["lines"] = [first, second]
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(manifest))
    bad_ref = _write_predictions(tmp_path / "bad-ref.json", mismatch_path, "ref", [
        {"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"},
        {"id": "b", "crop_sha256": "d" * 64, "status": "ok", "text": "son"},
    ], audit="reviewed")
    bad_candidate = _write_predictions(tmp_path / "bad-candidate.json", mismatch_path, "cand", [
        {"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"},
        {"id": "b", "crop_sha256": "d" * 64, "status": "ok", "text": "son"},
    ], audit="reviewed")
    with pytest.raises(ContractError, match="issue mismatch"):
        compare(mismatch_path, bad_ref, bad_candidate, resamples=20)


def test_real_kb_benchmark_is_blocked_when_issue_identity_is_unknown(tmp_path, monkeypatch):
    pytest.importorskip("rapidfuzz")
    import babaorum.comparison as comparison

    manifest = json.loads((Path(__file__).parent / "fixtures" / "line.json").read_text())
    manifest["exposure"] = "historical"
    manifest["key_policy"] = "declared_document_v0"
    manifest["lines"][0].update(page_id="kb:00531072", document_key="kb:00531072")
    manifest_path = tmp_path / "kb.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(comparison, "allow_development", lambda _manifest: None)
    rows = [{"id": "a", "crop_sha256": "b" * 64, "status": "ok", "text": "son"}]
    ref = _write_predictions(tmp_path / "kb-ref.json", manifest_path, "ref", rows, audit="reviewed")
    cand = _write_predictions(tmp_path / "kb-cand.json", manifest_path, "cand", rows, audit="reviewed")
    with pytest.raises(ContractError, match="historical issue"):
        compare(manifest_path, ref, cand, resamples=20)
