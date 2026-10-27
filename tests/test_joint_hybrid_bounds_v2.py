import json
from pathlib import Path

import pytest

from babaorum.experimental.joint_hybrid_bounds import build_full_grid, line_recall_bounds


def test_half_rate_example_reports_both_single_path_comparisons():
    result = line_recall_bounds(0.5, 0.5, 0.5)
    assert result["cascade_recall"] == pytest.approx(0.25)
    assert result["global_recall"] == pytest.approx(0.5)
    assert result["independent_oracle_union_recall"] == pytest.approx(0.625)
    assert result["independent_gain_over_cascade"] == pytest.approx(0.375)
    assert result["independent_gain_over_global"] == pytest.approx(0.125)
    assert result["independent_gain_over_best_single"] == pytest.approx(0.125)
    assert result["frechet_gain_over_best_single_lower"] == 0.0
    assert result["frechet_gain_over_best_single_upper"] == pytest.approx(0.25)


def test_full_cube_bounds_and_union_domination():
    config = json.loads(Path("config/joint_hybrid_bounds_v2.json").read_text())
    rows = build_full_grid(config)
    tolerance = config["numeric_tolerance"]
    assert len(rows) == 1331
    for row in rows:
        union = row["independent_oracle_union_recall"]
        assert row["frechet_union_lower"] <= union + tolerance
        assert union <= row["frechet_union_upper"] + tolerance
        assert union + tolerance >= row["best_single_recall"]
        assert abs(row["frechet_gain_over_best_single_lower"]) <= tolerance
        assert row["frechet_gain_over_best_single_upper"] >= -tolerance


@pytest.mark.parametrize("p,q,r", [(-0.1, 0.2, 0.3), (0.2, 1.1, 0.3), (0.2, 0.3, float("nan"))])
def test_invalid_rates_rejected(p, q, r):
    with pytest.raises(ValueError):
        line_recall_bounds(p, q, r)


def test_recall_is_monotone_nonincreasing_in_each_miss_probability():
    baseline = line_recall_bounds(0.2, 0.3, 0.4)["independent_oracle_union_recall"]
    assert line_recall_bounds(0.3, 0.3, 0.4)["independent_oracle_union_recall"] <= baseline
    assert line_recall_bounds(0.2, 0.4, 0.4)["independent_oracle_union_recall"] <= baseline
    assert line_recall_bounds(0.2, 0.3, 0.5)["independent_oracle_union_recall"] <= baseline
