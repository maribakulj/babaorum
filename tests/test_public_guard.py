from pathlib import Path
import importlib.util

import pytest


spec = importlib.util.spec_from_file_location(
    "public_guard", Path(__file__).resolve().parents[1] / "scripts/check_public_source.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize("path", [
    "reports/reader_items.jsonl", "runs/result.json", "data/page.png",
    "config/g0_page_usage_v3.json", "config/g0_usage_sources_v3.json",
    "config/sealed_registry.json", "docs/reviews/internal.md",
    "tests/fixtures/data1_shard0/data1_manifest.json", "weights.safetensors",
])
def test_corpus_metadata_and_internal_artifacts_are_rejected(path):
    assert guard.path_allowed(path) is False


def test_source_and_explicit_synthetic_fixtures_are_accepted():
    assert guard.path_allowed("src/babaorum/g0_suivi.py")
    assert guard.path_allowed("config/g0_suivi_inputs_schema_v1.json")
    assert guard.path_allowed("tests/fixtures/pero_red.png")


def test_stored_notebook_results_are_rejected():
    assert guard.notebook_is_source_only({"cells": [{"outputs": [], "execution_count": None}]})
    assert not guard.notebook_is_source_only({"cells": [{"outputs": [{"text": "result"}], "execution_count": 1}]})


def test_credential_pattern_is_detected_without_a_real_credential():
    example = "ghp_" + "A" * 40
    assert any(pattern.search(example) for pattern in guard.CREDENTIALS)
