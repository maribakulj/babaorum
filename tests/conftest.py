"""Public-edition collection policy; original scientific test modules are unchanged."""
from pathlib import Path

import pytest


LOCAL_RESOURCE_TESTS = {
    ("test_ref_gallica_vt_convert.py", "test_aller_retour_sur_vrai_fichier_bbvlm"),
    ("test_g0_lm_ngram8.py", "test_norm_is_the_norm_of_e6_when_axel_is_available"),
    # These benchmark entry-point tests load the real sealing registry before
    # they examine their synthetic prediction rows or historical aliases.
    ("test_contracts.py", "test_missing_prediction_counts_as_deletion"),
    ("test_contracts.py", "test_empty_truth_invention_reported"),
    ("test_contracts.py", "test_error_result_is_counted"),
    ("test_contracts.py", "test_benchmark_rejects_exposed_alias_before_predictions"),
}


def pytest_collection_modifyitems(items):
    for item in items:
        name = getattr(item, "originalname", None) or item.name
        if (Path(str(item.path)).name, name) in LOCAL_RESOURCE_TESTS:
            item.add_marker(pytest.mark.private_data)
