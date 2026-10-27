import pytest

from babaorum.experimental.hybrid_bounds import build_grid, recall_bounds


def test_recall_bounds_known_cases():
    assert recall_bounds(0.0, 0.0, 0.0) == {
        "cascade_recall": 1.0,
        "global_recall": 1.0,
        "independent_oracle_union_recall": 1.0,
        "frechet_union_lower": 1.0,
        "frechet_union_upper": 1.0,
        "independent_gain_over_cascade": 0.0,
    }
    result = recall_bounds(1.0, 1.0, 0.0)
    assert result["cascade_recall"] == 0.0
    assert result["global_recall"] == result["independent_oracle_union_recall"] == 1.0


@pytest.mark.parametrize("p,q,r", [(-0.01, 0.2, 0.3), (0.2, 1.01, 0.3), (0.2, 0.3, float("nan"))])
def test_invalid_rates_rejected(p, q, r):
    with pytest.raises(ValueError):
        recall_bounds(p, q, r)


def test_grid_union_dominates_each_path_and_stays_inside_frechet_bounds():
    import json
    from pathlib import Path

    config = json.loads(Path("config/hybrid_miss_ceiling.json").read_text())
    rows = build_grid(config)
    assert len(rows) == 242
    for row in rows:
        union = row["independent_oracle_union_recall"]
        assert row["frechet_union_lower"] <= union <= row["frechet_union_upper"]
        assert union >= max(row["cascade_recall"], row["global_recall"])


def test_independent_union_decreases_with_each_miss_rate():
    base = recall_bounds(0.2, 0.3, 0.4)["independent_oracle_union_recall"]
    assert recall_bounds(0.3, 0.3, 0.4)["independent_oracle_union_recall"] <= base
    assert recall_bounds(0.2, 0.4, 0.4)["independent_oracle_union_recall"] <= base
    assert recall_bounds(0.2, 0.3, 0.5)["independent_oracle_union_recall"] <= base


def test_independent_closed_form_matches_exhaustive_boolean_events():
    p, q, r = 0.37, 0.61, 0.22
    enumerated_union = 0.0
    for block_present in (False, True):
        p_block = (1.0 - p) if block_present else p
        for local_line_detected in (False, True):
            p_local = (1.0 - q) if local_line_detected else q
            for global_line_detected in (False, True):
                p_global = (1.0 - r) if global_line_detected else r
                if (block_present and local_line_detected) or global_line_detected:
                    enumerated_union += p_block * p_local * p_global
    assert enumerated_union == pytest.approx(
        recall_bounds(p, q, r)["independent_oracle_union_recall"]
    )
