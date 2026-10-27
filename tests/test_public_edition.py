"""The public source snapshot must not acquire permissive data-admission defaults."""
from pathlib import Path

import pytest

from babaorum import g0_suivi


ROOT = Path(__file__).resolve().parents[1]


def test_private_usage_inputs_are_omitted_and_never_trusted():
    inputs = g0_suivi.load_inputs()
    assert set(inputs.raw) == set(g0_suivi.PINNED_FILES)
    assert all(value is None for value in inputs.raw.values())


def test_absent_provenance_refuses_a_synthetic_query():
    result = g0_suivi.eligibility(
        {"collections": {}}, "suivi", page_key="gallica:9000000/f1"
    )
    assert result["allowed"] is False
    assert any(code.startswith("SEAL_NOT_PINNED:") for code in result["refusal_codes"])


def test_private_artifact_directories_are_not_bundled():
    assert not any((ROOT / name).exists() for name in ("reports", "runs", "data", "corpus"))


def test_benchmark_does_not_bypass_the_missing_sealing_registry(tmp_path):
    import json
    from babaorum.benchmark import evaluate
    from test_contracts import manifest

    path = tmp_path / "synthetic-manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        evaluate(path, tmp_path / "predictions-not-opened.json")
